#!/usr/bin/env python3
"""
Petit noyau évolutif
- charge tous les modules existants (./modules/*.py)
- expose au ctx : call_llm(), write_module(), get_manifest(), get_history()
- demande au LLM "faut-il écrire un module ?"
- si oui → 2e appel LLM → écrit le module
→ au run suivant, le module est dispo
"""

import os
import json
import importlib.util
import time
import subprocess
import traceback
from collections import Counter
import ast
import re
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEM_DIR = os.path.join(BASE_DIR, "mem")
MODULES_DIR = os.path.join(BASE_DIR, "modules")
HISTORY_PATH = os.path.join(MEM_DIR, "history.log")

LLM_MODEL = "qwen:latest"  # adapte à ton ollama


# -------------------------
# FS / MEM
# -------------------------
def ensure_dirs() -> None:
    os.makedirs(MEM_DIR, exist_ok=True)
    os.makedirs(MODULES_DIR, exist_ok=True)


def append_history(event: Dict[str, Any]) -> None:
    ev = dict(event)
    ev["ts"] = time.time()
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def read_history_tail(n: int = 80) -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    out: List[Dict[str, Any]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


# -------------------------
# LLM générique
# -------------------------
def call_llm(prompt: str) -> str:
    """
    Version simple pour Ollama.
    Tu peux la remplacer par un vrai client Python.
    """
    try:
        proc = subprocess.Popen(
            ["ollama", "run", LLM_MODEL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out, err = proc.communicate(prompt)
        if err:
            cleaned_err = _clean_llm_stderr(err)
            if cleaned_err:
                append_history({"event": "llm_stderr", "stderr": cleaned_err})
        return (out or "").strip()
    except FileNotFoundError:
        append_history({"event": "llm_error", "error": "ollama introuvable"})
        return ""
    except Exception as e:
        append_history({"event": "llm_error", "error": str(e)})
        return ""


# -------------------------
# MANIFEST
# -------------------------
def _truncate(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;?]*[A-Za-z]")
_PY_ASSIGN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*.+")
_JSON_CODE_FIELD_RE = re.compile(r'"code"\s*:\s*"(?P<code>(?:\\.|[^"\\])*)"')
_PLACEHOLDER_NAMES = {
    "nom_du_module_sans_py",
    "nom_du_module_sans_extension",
    "module",
    "module_python",
    "nouveau_module",
}
_VALID_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SPINNER_FRAMES = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
_PLANNING_KEYWORDS = (
    "plan",
    "planner",
    "planificateur",
    "roadmap",
    "strat",
    "organis",
)
_ANALYSIS_KEYWORDS = (
    "analyse",
    "analy",
    "error",
    "erreur",
    "observe",
    "monitor",
    "inspect",
)
_EXPANSION_PLANNING_THRESHOLD = 2
_EXPANSION_ANALYSIS_THRESHOLD = 1

_CAPABILITY_DESCRIPTIONS: Dict[str, str] = {
    "call_llm": "appelle le modèle configuré pour obtenir un texte ou un JSON",  # description
    "write_module": "enregistre un nouveau module Python dans ./modules",  # description
    "get_manifest": "donne la liste actuelle des modules (nom, taille, résumé)",  # description
    "get_history": "retourne les derniers événements pour analyse",  # description
}


def build_capabilities_block(catalog: Optional[Dict[str, Any]] = None) -> str:
    lines: List[str] = []

    if catalog:
        for name in sorted(catalog):
            entry = catalog.get(name)
            description: Optional[str] = None
            if isinstance(entry, ProbeDict):
                description = entry.get("description")
            elif isinstance(entry, dict):
                description = entry.get("description")
            if not description:
                description = _CAPABILITY_DESCRIPTIONS.get(name, "outil interne")
            lines.append(f"- {name} : {description}")

    if not lines:
        lines = [f"- {name} : {desc}" for name, desc in sorted(_CAPABILITY_DESCRIPTIONS.items())]

    return "\n".join(lines)


def register_capability(
    ctx: "ProbeDict", name: str, fn: Any, description: Optional[str] = None
) -> None:
    """Expose dynamiquement une capacité aux autres modules et au prompt."""

    if not name or not fn:
        return

    safe_name = name.strip()
    if not safe_name:
        return

    catalog = ctx.setdefault("capabilities", ProbeDict())
    effective_description = description or _CAPABILITY_DESCRIPTIONS.get(safe_name, "outil interne")
    existing = catalog.get(safe_name)
    if isinstance(existing, (dict, ProbeDict)):
        if existing.get("fn") is fn and existing.get("description") == effective_description:
            return
    catalog[safe_name] = ProbeDict({"fn": fn, "description": effective_description})
    if description:
        _CAPABILITY_DESCRIPTIONS[safe_name] = description

    append_history(
        {
            "event": "capability_registered",
            "name": safe_name,
            "description": effective_description,
        }
    )


def build_capabilities_catalog(functions: Dict[str, Any]) -> ProbeDict:
    catalog = {}
    for name, func in functions.items():
        desc = _CAPABILITY_DESCRIPTIONS.get(name, "outil interne")
        catalog[name] = ProbeDict({"fn": func, "description": desc})
    return ProbeDict(catalog)


class ProbeDict(dict):
    """Petit dict offrant un accès attribut pour les validations."""

    __slots__ = ()

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, ProbeDict):
            return ProbeDict(value)
        return value

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if args or kwargs:
            self.update(*args, **kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:  # type: ignore[override]
        super().__setitem__(key, self._wrap(value))

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        data = dict(*args, **kwargs)
        for key, value in data.items():
            super().__setitem__(key, self._wrap(value))

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - protection simple
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as exc:  # pragma: no cover - protection simple
            raise AttributeError(key) from exc


def _parse_json_response(raw: str) -> Tuple[Optional[Any], Optional[str]]:
    """Try to decode JSON even if text surrounds it."""

    cleaned = raw.strip()
    if not cleaned:
        return None, "réponse vide"

    decoder = json.JSONDecoder()
    try:
        return decoder.decode(cleaned), None
    except json.JSONDecodeError as first_error:
        for idx, ch in enumerate(cleaned):
            if ch not in "[{":
                continue
            try:
                data, _ = decoder.raw_decode(cleaned[idx:])
                return data, None
            except json.JSONDecodeError:
                continue
        return None, f"JSON invalide: {first_error.msg} (ligne {first_error.lineno}, colonne {first_error.colno})"


def _normalize_module_code(raw: str) -> str:
    """Extraire le code Python d'une réponse possiblement verbeuse."""

    if not raw:
        return ""

    text = raw.strip()
    if not text:
        return ""

    matches = _CODE_FENCE_RE.findall(text)
    if matches:
        segments = [segment.strip() for segment in matches if segment.strip()]
        if segments:
            return "\n\n".join(segments)

    if "```" in text:
        text = text.replace("```python", "").replace("```", "").strip()

    lines = text.splitlines()
    valid_prefixes = ("def ", "class ", "import ", "from ", "@", "#", '"""', "'''")
    start_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped:
            continue
        if stripped.startswith(valid_prefixes) or _PY_ASSIGN_RE.match(stripped):
            start_idx = idx
            break

    if start_idx is not None and start_idx > 0:
        text = "\n".join(lines[start_idx:]).strip()

    return text


def _extract_code_candidate(raw: str, name: str, context: str, stage: str) -> str:
    """Récupère le champ code depuis une réponse JSON, sinon renvoie la réponse brute."""

    if not raw:
        return ""

    cleaned = raw.strip()
    if not cleaned:
        return ""

    data, parse_error = _parse_json_response(cleaned)
    if isinstance(data, dict):
        code_value = data.get("code")
        if isinstance(code_value, str):
            return code_value

        append_history(
            {
                "event": "llm_module_json_error",
                "name": name,
                "context": context,
                "stage": stage,
                "error": "champ 'code' manquant ou non textuel",
                "raw": _truncate(cleaned, 200),
            }
        )
        return cleaned

    if parse_error:
        match = _JSON_CODE_FIELD_RE.search(cleaned)
        if match:
            raw_code = match.group("code")
            try:
                recovered = json.loads(f'"{raw_code}"')
            except json.JSONDecodeError:
                recovered = raw_code.replace("\\n", "\n")
            append_history(
                {
                    "event": "llm_module_json_recovered",
                    "name": name,
                    "context": context,
                    "stage": stage,
                    "raw": _truncate(cleaned, 200),
                }
            )
            return recovered

        append_history(
            {
                "event": "llm_module_json_error",
                "name": name,
                "context": context,
                "stage": stage,
                "error": parse_error,
                "raw": _truncate(cleaned, 200),
            }
        )

    return cleaned


def _clean_llm_stderr(raw: str) -> str:
    if not raw:
        return ""

    text = _ANSI_ESCAPE_RE.sub("", raw)
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if all(ch in _SPINNER_FRAMES for ch in stripped):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def sanitize_module_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None

    cleaned = name.strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in _PLACEHOLDER_NAMES or lowered.startswith("nom_du_module_"):
        return None

    if not _VALID_MODULE_NAME_RE.match(cleaned):
        return None

    return cleaned


def derive_module_basename(
    preferred: Optional[str], *, fallback: str, extras: Optional[List[Optional[str]]] = None
) -> str:
    """Return a sanitized module basename, salvaging noisy inputs when possible."""

    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    if extras:
        for extra in extras:
            if extra:
                candidates.append(extra)

    for candidate in candidates:
        sanitized = sanitize_module_name(candidate)
        if sanitized:
            return sanitized

    for candidate in candidates:
        trimmed = candidate.strip()
        if not trimmed:
            continue
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", trimmed)
        cleaned = cleaned.strip("_")
        if cleaned and cleaned[0].isdigit():
            cleaned = f"{fallback}_{cleaned}"
        sanitized = sanitize_module_name(cleaned)
        if sanitized:
            return sanitized

    sanitized_fallback = sanitize_module_name(fallback)
    if sanitized_fallback:
        return sanitized_fallback

    return "module_auto"


def _extract_module_metadata(path: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "summary": "",
        "has_init": False,
        "has_tick": False,
        "functions": [],
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        doc = ast.get_docstring(tree) or ""
        func_names: List[str] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_names.append(node.name)
                if node.name == "init":
                    meta["has_init"] = True
                if node.name == "tick":
                    meta["has_tick"] = True
        summary_parts: List[str] = []
        if doc:
            summary_parts.append(_truncate(doc.strip().replace("\n", " ")))
        if func_names:
            summary_parts.append(f"fonctions: {', '.join(func_names)}")
        if not summary_parts and source:
            first_lines = " ".join(line.strip() for line in source.splitlines()[:5])
            summary_parts.append(_truncate(first_lines))
        meta["summary"] = " | ".join(summary_parts)
        meta["functions"] = func_names
    except Exception as e:
        meta["summary"] = f"analyse impossible: {e}"[:240]
    return meta


def build_manifest() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.isdir(MODULES_DIR):
        return items
    for fname in sorted(os.listdir(MODULES_DIR)):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(MODULES_DIR, fname)
        try:
            size = os.path.getsize(path)
        except Exception:
            size = 0
        meta = _extract_module_metadata(path)
        items.append(
            {
                "name": fname,
                "size": size,
                "has_init": meta["has_init"],
                "has_tick": meta["has_tick"],
                "summary": meta["summary"],
            }
        )
    return items


def _classify_manifest_focus(manifest: List[Dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for entry in manifest:
        name = (entry.get("name") or "").lower()
        summary = (entry.get("summary") or "").lower()
        blob = f"{name} {summary}"
        if any(keyword in blob for keyword in _PLANNING_KEYWORDS):
            counts["planning"] += 1
        if any(keyword in blob for keyword in _ANALYSIS_KEYWORDS):
            counts["analysis"] += 1
    counts["total"] = len(manifest)
    return counts


def build_decision_prompt(
    manifest: List[Dict[str, Any]],
    history_txt: str,
    quick_summary: str,
    capabilities_block: str,
) -> Tuple[str, str, Counter]:
    focus_counts = _classify_manifest_focus(manifest)
    planning_count = focus_counts.get("planning", 0)
    analysis_count = focus_counts.get("analysis", 0)

    variant = "default"
    focus_lines = [
        "Ta priorité est de consolider l'observation et la planification.",
        "Propose des modules qui analysent l'état, planifient les suites ou inspectent les erreurs récentes.",
    ]

    if (
        planning_count >= _EXPANSION_PLANNING_THRESHOLD
        and analysis_count >= _EXPANSION_ANALYSIS_THRESHOLD
    ):
        variant = "expansion"
        focus_lines = [
            "Tu disposes déjà d'une base de planification/diagnostic (modules orientés planification et analyse en place).",
            "Accélère désormais l'autonomie : crée des modules qui exploitent ces connaissances pour agir, coordonner ou ajouter des outils concrets.",
            "Repère les capacités manquantes (mémoire active, instrumentation, contrôle, coopération entre modules) et comble-les.",
        ]

    focus_block = "\n".join(f"- {line}" for line in focus_lines)
    if variant == "default":
        rule_block = """Règles IMPORTANTES :
1. Tu NE réécris PAS l'agent principal.
2. Tu NE renvoies PAS de code ici.
3. Tu ne proposes un module QUE s'il sert à :
   - observer l'état (modules, mémoire, erreurs)
   - organiser / planifier les prochains modules
   - créer d'autres générateurs de modules
   - analyser les erreurs des modules existants
4. Toujours répondre en JSON STRICT.
"""
    else:
        rule_block = """Règles IMPORTANTES :
1. Tu NE réécris PAS l'agent principal.
2. Tu NE renvoies PAS de code ici.
3. Les modules proposés doivent étendre concrètement les capacités de l'agent (exécution, coordination, outillage exploitable) en s'appuyant sur les ressources existantes.
4. Toujours répondre en JSON STRICT.
"""

    prompt = f"""
Synthèse rapide :
{quick_summary}

But ultime explicite : faire évoluer l'agent vers une intelligence générale émergente via un écosystème modulaire auto-améliorant.

Tu disposes d'un LLM puissant : exploite-le pour orchestrer la progression.

Focus actuel :
{focus_block}

Capacités noyau accessibles via ctx :
{capabilities_block}

Procède étape par étape :
1. Observe l'état actuel ci-dessus.
2. Dégage l'amélioration la plus utile pour progresser vers le but ultime.
3. Choisis d'agir (write_module) ou de patienter (noop) selon cette analyse.

Tu es le CERVEAU de l'agent.
Ton rôle N'EST PAS d'écrire le gros programme, mais de décider s'il faut ajouter UN PETIT module Python autonome.

Contexte modules déjà présents (nom + taille) :
{json.dumps(manifest, ensure_ascii=False, indent=2)}

Historique récent :
{history_txt}

Objectif global de l'agent :
- apprendre en ajoutant de petits modules
- que ces modules puissent s'initialiser (init(ctx)) et agir à chaque tick (tick(ctx))
- converger vers un système qui se réécrit, utilise ses propres capacités et progresse vers l'AGI visée

{rule_block}
Format de réponse OBLIGATOIRE :
{{
  "action": "write_module" | "noop",
  "name": "nom_du_module_sans_py",
  "description": "une phrase claire sur ce que doit faire le module"
}}

Exemples valides :
{{
  "action": "noop",
  "name": "",
  "description": ""
}}
ou
{{
  "action": "write_module",
  "name": "planner_local",
  "description": "tenir une petite liste de modules à créer selon les erreurs vues dans l'historique"
}}

RENVOIE UNIQUEMENT le JSON, pas de commentaire.
"""

    return prompt, variant, focus_counts


# -------------------------
# Écriture de modules
# -------------------------
def write_module_file(name: str, code: str) -> str:
    base_name = derive_module_basename(name, fallback="module_auto")
    final_base = base_name
    suffix = 1
    while os.path.exists(os.path.join(MODULES_DIR, f"{final_base}.py")):
        final_base = f"{base_name}_{suffix}"
        suffix += 1

    fname = f"{final_base}.py"
    path = os.path.join(MODULES_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    history_event: Dict[str, Any] = {"event": "module_written", "file": fname, "name": final_base}
    if name and name != final_base:
        history_event["requested_name"] = name
        if base_name != final_base:
            history_event["normalized_name"] = base_name
    append_history(history_event)
    return fname


def _function_has_effective_body(body: List[ast.stmt]) -> bool:
    """Detecte si la fonction contient autre chose qu'un simple pass/docstring."""

    for idx, stmt in enumerate(body):
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr):
            value = getattr(stmt, "value", None)
            if idx == 0 and isinstance(value, ast.Constant) and isinstance(value.value, str):
                # docstring de tête
                continue
        return True
    return False


def _is_allowed_toplevel(node: ast.stmt) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
        return True
    if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
        return True
    return False


def validate_module_code(code: str) -> Optional[str]:
    """Return an error message if the code is not valid Python and structurally acceptable."""

    try:
        tree = ast.parse(code)
        compiled = compile(code, "<module>", "exec")
    except SyntaxError as e:
        location = f" (ligne {e.lineno}, colonne {e.offset})" if e.lineno else ""
        return f"SyntaxError{location}: {e.msg}"
    except Exception as e:  # pragma: no cover - sécurité
        return f"Erreur lors de la compilation: {e}"

    for node in tree.body:
        if not _is_allowed_toplevel(node):
            return "le code au niveau global doit se limiter aux imports, affectations, classes et fonctions"

    has_init = False
    has_tick = False
    init_effective = False
    tick_effective = False

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "init":
                has_init = True
                if not node.args.args:
                    return "init doit accepter un paramètre ctx"
                if node.args.args[0].arg != "ctx":
                    return "init doit avoir ctx comme premier paramètre"
                init_effective = _function_has_effective_body(node.body)
            elif node.name == "tick":
                has_tick = True
                if not node.args.args:
                    return "tick doit accepter un paramètre ctx"
                if node.args.args[0].arg != "ctx":
                    return "tick doit avoir ctx comme premier paramètre"
                tick_effective = _function_has_effective_body(node.body)

    if not has_init or not has_tick:
        return "le module doit définir les fonctions init(ctx) et tick(ctx)"

    if not init_effective:
        return "init(ctx) doit contenir autre chose qu'un simple pass ou docstring"

    if not tick_effective:
        return "tick(ctx) doit contenir autre chose qu'un simple pass ou docstring"

    namespace: Dict[str, Any] = {}
    try:
        exec(compiled, namespace)
    except Exception as e:
        return f"erreur lors de l'exécution initiale du module: {e}"

    init_fn = namespace.get("init")
    tick_fn = namespace.get("tick")

    if not callable(init_fn) or not callable(tick_fn):
        return "le module doit définir les fonctions init(ctx) et tick(ctx)"

    probe_ctx = ProbeDict(
        history=[
            {
                "event": "llm_stderr",
                "stderr": "Error: invalid model path",
            },
            {
                "event": "llm_decision_noop",
            },
        ],
        tickers=[],
        planner={"modules_to_create": []},
        manifest=[],
        modules=[],
        suggestions=[],
        pattern=True,
    )

    try:
        init_fn(probe_ctx)
    except Exception as e:
        return f"init(ctx) échoue lors d'un appel de test: {e}"

    try:
        tick_fn(probe_ctx)
    except Exception as e:
        return f"tick(ctx) échoue lors d'un appel de test: {e}"

    return None


def remove_module_file(fname: str) -> None:
    path = os.path.join(MODULES_DIR, fname)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except Exception as e:
            append_history({"event": "module_remove_error", "module": fname, "error": str(e)})


def format_history_counter(hist_list: List[Dict[str, Any]]) -> str:
    counter = Counter(h.get("event", "inconnu") for h in hist_list)
    if not counter:
        return "aucun événement notable"
    return ", ".join(f"{evt}:{count}" for evt, count in counter.most_common())


def build_progress_metrics(hist_list: List[Dict[str, Any]]) -> str:
    if not hist_list:
        return "aucune donnée de progression"
    counter = Counter(h.get("event", "inconnu") for h in hist_list)
    tick_errors = counter.get("module_tick_error", 0)
    regen = counter.get("module_regenerated", 0)
    written = counter.get("module_written", 0)
    removed = counter.get("module_removed_after_error", 0)
    return (
        f"modules écrits:{written} | régénérés:{regen} | erreurs tick:{tick_errors} | suppressions:{removed}"
    )


def build_history_text(hist_list: List[Dict[str, Any]]) -> str:
    if not hist_list:
        return "aucun historique."
    return "\n".join(
        f"- {h.get('event')}"
        + (f" (module={h.get('module')})" if h.get("module") else "")
        + (f" :: {h.get('error')}" if h.get("error") else "")
        for h in hist_list
    )


def describe_history_event(event: Dict[str, Any]) -> str:
    evt = event.get("event", "inconnu")
    if evt == "module_written":
        name = event.get("name") or event.get("file") or "module"
        return f"module ajouté : {name}"
    if evt == "module_regenerated":
        return f"module régénéré : {event.get('from')} → {event.get('to')}"
    if evt == "module_removed_after_error":
        return f"module supprimé après erreur : {event.get('module')}"
    if evt == "module_tick_error":
        module = event.get("module") or "module inconnu"
        error = (event.get("error") or "").splitlines()[0][:120]
        return f"erreur tick sur {module} :: {error}".strip()
    if evt == "modules_loaded":
        modules = event.get("modules") or []
        return f"modules chargés : {', '.join(modules) or 'aucun'}"
    if evt == "module_regen_description":
        return f"objectif régénération : {event.get('description')}"
    if evt == "llm_decision_noop":
        return "décision : noop"
    if evt == "llm_empty_module":
        return f"échec génération (code vide) pour {event.get('name')}"
    if evt == "module_load_error":
        module = event.get("module") or "module"
        error = (event.get("error") or "").splitlines()[0][:120]
        return f"erreur chargement {module} :: {error}".strip()
    return evt


def build_recent_activity_summary(hist_list: List[Dict[str, Any]], limit: int = 8) -> str:
    if not hist_list:
        return "  - aucune activité enregistrée"
    lines: List[str] = []
    for event in reversed(hist_list[-limit:]):
        lines.append(f"  - {describe_history_event(event)}")
    return "\n".join(lines)


# -------------------------
# chargement dynamique
# -------------------------
def load_all_modules(ctx: Dict[str, Any]) -> List[str]:
    loaded: List[str] = []
    if not os.path.isdir(MODULES_DIR):
        return loaded
    for fname in sorted(os.listdir(MODULES_DIR)):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(MODULES_DIR, fname)
        mod_name = f"mods_{fname[:-3]}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore

            # le module voit le ctx
            if hasattr(mod, "init") and callable(mod.init):
                mod.init(ctx)

            # le module peut s'enregistrer pour les ticks
            if hasattr(mod, "tick") and callable(mod.tick):
                ctx.setdefault("tickers", []).append({"module": fname, "tick": mod.tick})

            loaded.append(fname)
        except Exception as e:
            traceback.print_exc()
            append_history({"event": "module_load_error", "module": fname, "error": str(e)})
    return loaded


def run_tickers(ctx: Dict[str, Any]) -> None:
    tickers = list(ctx.get("tickers", []))
    for entry in tickers:
        module_file = None
        tick_fn = None
        if isinstance(entry, dict):
            module_file = entry.get("module")
            tick_fn = entry.get("tick")
        elif isinstance(entry, tuple) and len(entry) == 2:
            module_file, tick_fn = entry
        elif callable(entry):
            tick_fn = entry
        if not callable(tick_fn):
            continue
        try:
            tick_fn(ctx)
        except Exception as e:
            error_text = "".join(traceback.format_exception_only(type(e), e)).strip() or str(e)
            append_history(
                {
                    "event": "module_tick_error",
                    "module": module_file,
                    "error": error_text,
                }
            )
            if module_file:
                cleanup_after_failure(module_file, error_text, ctx)
            # désactive le ticker fautif pour ce run
            try:
                ctx.get("tickers", []).remove(entry)
            except ValueError:
                pass


def lookup_module_public_name(module_file: str, history: List[Dict[str, Any]]) -> str:
    for entry in reversed(history):
        if entry.get("event") == "module_written" and entry.get("file") == module_file:
            return entry.get("name") or module_file
    return module_file


def build_quick_summary(manifest: List[Dict[str, Any]], hist_list: List[Dict[str, Any]]) -> str:
    manifest_lines = []
    for m in manifest:
        manifest_lines.append(
            f"  - {m.get('name')} | taille={m.get('size')} | init={m.get('has_init')} | tick={m.get('has_tick')} | {m.get('summary')}"
        )
    manifest_block = "\n".join(manifest_lines) or "  - aucun module"
    hist_summary = format_history_counter(hist_list)
    progress_line = build_progress_metrics(hist_list)
    activity_block = build_recent_activity_summary(hist_list)
    return (
        f"- Modules actifs détaillés :\n{manifest_block}\n"
        f"- Comptage des événements récents (sur {len(hist_list)} entrées) : {hist_summary}\n"
        f"- Indicateurs de progression : {progress_line}\n"
        f"- Activités notables :\n{activity_block}"
    )


def cleanup_after_failure(module_file: str, error_text: str, ctx: Dict[str, Any]) -> None:
    remove_module_file(module_file)
    append_history({"event": "module_removed_after_error", "module": module_file})
    regenerate_failed_module(module_file, error_text, ctx)


def generate_valid_module_code(
    name: str,
    description: str,
    manifest: List[Dict[str, Any]],
    history_txt: str,
    *,
    context: str,
    initial_code: Optional[str] = None,
) -> Optional[str]:
    attempts = 0
    validation_error: Optional[str] = None
    last_code = _normalize_module_code(initial_code or "") if initial_code else ""
    use_initial = initial_code is not None

    while attempts < 3:
        if use_initial:
            candidate_raw = initial_code or ""
            stage = "initial_seed"
            use_initial = False
        elif attempts == 0:
            candidate_raw = ask_llm_for_module_code(name, description, manifest, history_txt)
            stage = "initial_prompt"
        else:
            candidate_raw = ask_llm_to_fix_module_code(
                name,
                description,
                manifest,
                history_txt,
                last_code,
                validation_error or "erreur inconnue",
            )
            stage = "fix_prompt"

        if stage == "initial_seed":
            candidate = _normalize_module_code(candidate_raw)
        else:
            candidate = _normalize_module_code(
                _extract_code_candidate(candidate_raw, name, context, stage)
            )

        if not candidate.strip():
            append_history(
                {
                    "event": "llm_empty_module",
                    "name": name,
                    "attempt": attempts + 1,
                    "context": context,
                    "reason": "nettoyage_sans_code",  # indique que la réponse ne contenait pas de code utilisable
                }
            )
            validation_error = "code vide après nettoyage"
            attempts += 1
            last_code = candidate
            continue

        validation_error = validate_module_code(candidate)
        if not validation_error:
            return candidate

        append_history(
            {
                "event": "module_validation_error",
                "name": name,
                "error": validation_error,
                "attempt": attempts + 1,
                "context": context,
                "preview": _truncate(candidate, 160),
            }
        )

        attempts += 1
        last_code = candidate

    append_history(
        {
            "event": "module_validation_failed",
            "name": name,
            "error": validation_error,
            "context": context,
            "preview": _truncate(last_code, 160),
        }
    )
    return None


def regenerate_failed_module(module_file: str, error_text: str, ctx: Dict[str, Any]) -> None:
    call_llm = ctx.get("call_llm")
    write_module = ctx.get("write_module")
    if not (call_llm and write_module):
        return

    hist_list = read_history_tail(60)
    public_name = lookup_module_public_name(module_file, hist_list)
    manifest = build_manifest()
    hist_txt = build_history_text(hist_list)
    quick_summary = build_quick_summary(manifest, hist_list)

    prompt = f"""
Un module a échoué et a été supprimé.
Nom public connu : {public_name}
Fichier supprimé : {module_file}
Erreur rencontrée : {error_text}

Synthèse rapide :
{quick_summary}

Historique récent :
{hist_txt}

Ta mission : proposer un module de remplacement aligné sur l'objectif ultime d'évolution vers une AGI. Suis ces étapes :
1. Analyse ce que ce module devait probablement accomplir à partir du contexte.
2. Propose un petit module utile qui corrige ou remplace la fonctionnalité défaillante.
3. Fournis le code Python autonome correspondant.

Réponds en JSON strict avec les clés suivantes :
{{
  "name": "nom_du_module_sans_extension",
  "description": "but du module",
  "code": "CODE PYTHON COMPLET"
}}

Pas de texte autour.
"""

    out = call_llm(prompt)
    if not out:
        return
    cleaned = out.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```python", "").replace("```", "").strip()
    data, parse_error = _parse_json_response(cleaned)
    if data is None:
        append_history(
            {
                "event": "llm_regen_parse_error",
                "raw": cleaned,
                "error": parse_error,
            }
        )
        return

    requested_regen_name = data.get("name")
    requested_regen_name = requested_regen_name if isinstance(requested_regen_name, str) else None
    sanitized_requested = sanitize_module_name(requested_regen_name)
    if sanitized_requested:
        name = sanitized_requested
    else:
        name = derive_module_basename(
            requested_regen_name,
            fallback="regen_module",
            extras=[public_name],
        )
        if requested_regen_name:
            append_history(
                {
                    "event": "llm_invalid_module_name",
                    "raw": requested_regen_name,
                    "context": "regen_decision",
                    "normalized": name,
                }
            )

    code = data.get("code")
    description = data.get("description")
    if description:
        append_history({"event": "module_regen_description", "module": name, "description": description})
    if not code:
        append_history({"event": "llm_regen_empty_code", "module": name})
        return

    valid_code = generate_valid_module_code(
        name,
        description or "module utilitaire pour l'agent",
        manifest,
        hist_txt,
        context="regen",
        initial_code=code,
    )
    if not valid_code:
        return

    new_file = write_module(name, valid_code)
    append_history({"event": "module_regenerated", "from": module_file, "to": new_file})


# -------------------------
# phase "juge"
# -------------------------
def ask_llm_for_next_step(
    manifest: List[Dict[str, Any]],
    history_txt: str,
    quick_summary: str,
    capabilities_block: str,
) -> Dict[str, Any]:
    prompt, variant, focus_counts = build_decision_prompt(
        manifest, history_txt, quick_summary, capabilities_block
    )
    append_history(
        {
            "event": "decision_prompt_variant",
            "variant": variant,
            "counts": dict(focus_counts),
        }
    )

    out = call_llm(prompt)
    if not out:
        return {"action": "noop"}
    out = out.strip()
    if "```" in out:
        out = out.replace("```json", "").replace("```", "").strip()
    data, parse_error = _parse_json_response(out)
    if data is not None:
        return data

    append_history(
        {
            "event": "llm_json_parse_error",
            "raw": out,
            "error": parse_error,
        }
    )
    return {"action": "noop"}


def ask_llm_for_module_code(
    name: str, description: str, manifest: List[Dict[str, Any]], history_txt: str
) -> str:
    prompt = f"""
Tu écris un module Python autonome pour un agent évolutif.
Le module sera enregistré dans ./modules/{name}.py
Objectif du module: {description}

Modules existants:
{json.dumps(manifest, ensure_ascii=False, indent=2)}

Historique récent:
{history_txt}

Contraintes impératives:
- le fichier doit pouvoir être importé sans erreur
- le contexte ctx est un dictionnaire Python (utilise ctx["cle"] ou ctx.cle)
- définis obligatoirement def init(ctx): et def tick(ctx):
- init(ctx) doit initialiser ou mettre à jour des données utiles dans ctx (pas seulement pass)
- tick(ctx) doit effectuer au moins une action concrète ou une mise à jour exploitable
- n'appelle pas init(ctx) ou tick(ctx) au niveau global
- renvoie UNIQUEMENT un JSON strict {{"code": "..."}}
- encode les retours à la ligne avec \n dans la valeur de "code"
- aucun autre texte ou champ en sortie.
"""
    code = call_llm(prompt)
    if not code:
        return ""
    if "```" in code:
        code = code.replace("```python", "").replace("```", "").strip()
    return code


def ask_llm_to_fix_module_code(
    name: str,
    description: str,
    manifest: List[Dict[str, Any]],
    history_txt: str,
    previous_code: str,
    error_message: str,
) -> str:
    prompt = f"""
Le code suivant pour le module {name} provoque une erreur de validation :
---
{previous_code}
---
Erreur détectée : {error_message}

Réécris le module complet en corrigeant le problème tout en respectant la description :
{description}

Rappels :
- le module doit pouvoir être importé sans erreur
- le contexte ctx est un dict : manipule-le avec ctx["cle"] (ou ctx.cle)
- définis forcément init(ctx) et tick(ctx)
- init(ctx) doit préparer un état utile (plus qu'un simple pass)
- tick(ctx) doit mener une action exploitable (plus qu'un simple pass)
- n'appelle pas init(ctx) ou tick(ctx) au niveau global
- renvoie UNIQUEMENT un JSON strict {{"code": "..."}}
- encode les retours à la ligne avec \n dans la valeur de "code"
- aucun autre texte ou champ en sortie.
"""
    code = call_llm(prompt)
    if not code:
        return ""
    if "```" in code:
        code = code.replace("```python", "").replace("```", "").strip()
    return code


# -------------------------
# main
# -------------------------
def main() -> None:
    ensure_dirs()

    # ctx partagé avec les modules
    ctx: ProbeDict = ProbeDict(
        base_dir=BASE_DIR,
        mem_dir=MEM_DIR,
        modules_dir=MODULES_DIR,
    )

    # on expose les outils du noyau AUX modules
    get_history_fn = lambda n=80: read_history_tail(n)
    ctx["call_llm"] = call_llm
    ctx["write_module"] = write_module_file
    ctx["get_manifest"] = build_manifest
    ctx["get_history"] = get_history_fn
    ctx["capabilities"] = build_capabilities_catalog(
        {
            "call_llm": ctx["call_llm"],
            "write_module": ctx["write_module"],
            "get_manifest": ctx["get_manifest"],
            "get_history": ctx["get_history"],
        }
    )
    ctx["register_capability"] = lambda name, fn, description=None: register_capability(
        ctx, name, fn, description
    )

    try:
        while True:
            # les tickers sont reconstruits à chaque itération
            ctx["tickers"] = []

            loaded = load_all_modules(ctx)
            append_history({"event": "modules_loaded", "modules": loaded})

            # on laisse les modules jouer
            run_tickers(ctx)

            # noyau lui-même peut aussi demander au LLM d'ajouter un module
            manifest = build_manifest()
            hist_list = read_history_tail(30)
            hist_txt = build_history_text(hist_list)
            quick_summary = build_quick_summary(manifest, hist_list)

            capabilities_block = build_capabilities_block(ctx.get("capabilities"))
            decision = ask_llm_for_next_step(
                manifest, hist_txt, quick_summary, capabilities_block
            )
            if decision.get("action") != "write_module":
                append_history({"event": "llm_decision_noop"})
            else:
                raw_name = decision.get("name")
                requested_name = raw_name if isinstance(raw_name, str) else None
                sanitized_requested = sanitize_module_name(requested_name)
                if sanitized_requested:
                    mod_name = sanitized_requested
                else:
                    mod_name = derive_module_basename(requested_name, fallback="module_auto")
                    if requested_name:
                        append_history(
                            {
                                "event": "llm_invalid_module_name",
                                "raw": requested_name,
                                "context": "initial_decision",
                                "normalized": mod_name,
                            }
                        )

                mod_desc = decision.get("description") or "module utilitaire pour l'agent"

                valid_code = generate_valid_module_code(
                    mod_name, mod_desc, manifest, hist_txt, context="initial"
                )
                if valid_code:
                    fname = write_module_file(mod_name, valid_code)
                    print(f"[agent] module ajouté: {fname}")

            time.sleep(5)
    except KeyboardInterrupt:
        print("[agent] arrêt demandé, au revoir")


if __name__ == "__main__":
    main()
