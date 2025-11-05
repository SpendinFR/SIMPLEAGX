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
from typing import Any, Dict, List, Optional

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
            append_history({"event": "llm_stderr", "stderr": err})
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


# -------------------------
# Écriture de modules
# -------------------------
def write_module_file(name: str, code: str) -> str:
    safe_name = name.replace(" ", "_").replace("-", "_")
    ts = int(time.time())
    fname = f"{ts}_{safe_name}.py"
    path = os.path.join(MODULES_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    append_history({"event": "module_written", "file": fname, "name": name})
    return fname


def validate_module_code(code: str) -> Optional[str]:
    """Return an error message if the code is not valid Python."""
    try:
        compile(code, "<module>", "exec")
    except SyntaxError as e:
        location = f" (ligne {e.lineno}, colonne {e.offset})" if e.lineno else ""
        return f"SyntaxError{location}: {e.msg}"
    except Exception as e:  # pragma: no cover - sécurité
        return f"Erreur lors de la compilation: {e}"
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
    last_code = initial_code or ""
    use_initial = initial_code is not None

    while attempts < 3:
        if use_initial:
            candidate = initial_code or ""
            use_initial = False
        elif attempts == 0:
            candidate = ask_llm_for_module_code(name, description, manifest, history_txt)
        else:
            candidate = ask_llm_to_fix_module_code(
                name,
                description,
                manifest,
                history_txt,
                last_code,
                validation_error or "erreur inconnue",
            )

        if not candidate.strip():
            append_history(
                {
                    "event": "llm_empty_module",
                    "name": name,
                    "attempt": attempts + 1,
                    "context": context,
                }
            )
            return None

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
    try:
        data = json.loads(cleaned)
    except Exception:
        append_history({"event": "llm_regen_parse_error", "raw": cleaned})
        return

    name = data.get("name") or public_name
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
    manifest: List[Dict[str, Any]], history_txt: str, quick_summary: str
) -> Dict[str, Any]:
    prompt = f"""
Synthèse rapide :
{quick_summary}

But ultime explicite : faire évoluer l'agent vers une intelligence générale émergente via un écosystème modulaire auto-améliorant.

Tu disposes d'un LLM puissant : exploite-le pour orchestrer la progression.

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

Règles IMPORTANTES :
1. Tu NE réécris PAS l'agent principal.
2. Tu NE renvoies PAS de code ici.
3. Tu ne proposes un module QUE s'il sert à :
   - observer l'état (modules, mémoire, erreurs)
   - organiser / planifier les prochains modules
   - créer d'autres générateurs de modules
   - analyser les erreurs des modules existants
4. Toujours répondre en JSON STRICT.

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

    out = call_llm(prompt)
    if not out:
        return {"action": "noop"}
    out = out.strip()
    if "```" in out:
        out = out.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(out)
    except Exception as e:
        append_history({"event": "llm_json_parse_error", "raw": out, "error": str(e)})
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

Contraintes:
- doit pouvoir être importé sans erreur
- si besoin: def init(ctx): ...
- si besoin: def tick(ctx): ...
- chaque fonction doit contenir un corps valide (au minimum "pass")
- n'appelle pas init(ctx) ou tick(ctx) au niveau global
- PAS de texte autour, renvoie UNIQUEMENT le code Python.
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
- si besoin: def init(ctx): ...
- si besoin: def tick(ctx): ...
- chaque fonction doit contenir un corps valide (au minimum "pass")
- n'appelle pas init(ctx) ou tick(ctx) au niveau global
- renvoie UNIQUEMENT le code Python, sans balise.
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
    ctx: Dict[str, Any] = {
        "base_dir": BASE_DIR,
        "mem_dir": MEM_DIR,
        "modules_dir": MODULES_DIR,
    }

    # on expose les outils du noyau AUX modules
    ctx["call_llm"] = call_llm
    ctx["write_module"] = write_module_file
    ctx["get_manifest"] = build_manifest
    ctx["get_history"] = lambda n=80: read_history_tail(n)

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

            decision = ask_llm_for_next_step(manifest, hist_txt, quick_summary)
            if decision.get("action") != "write_module":
                append_history({"event": "llm_decision_noop"})
            else:
                mod_name = decision.get("name") or f"module_{int(time.time())}"
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
