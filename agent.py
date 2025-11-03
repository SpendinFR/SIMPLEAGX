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
from typing import Any, Dict, List

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
        items.append({"name": fname, "size": size})
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
                ctx.setdefault("tickers", []).append(mod.tick)

            loaded.append(fname)
        except Exception as e:
            traceback.print_exc()
            append_history({"event": "module_load_error", "module": fname, "error": str(e)})
    return loaded


def run_tickers(ctx: Dict[str, Any]) -> None:
    for t in ctx.get("tickers", []):
        try:
            t(ctx)
        except Exception as e:
            append_history({"event": "module_tick_error", "error": str(e)})


# -------------------------
# phase "juge"
# -------------------------
def ask_llm_for_next_step(manifest: List[Dict[str, Any]], history_txt: str) -> Dict[str, Any]:
    prompt = f"""
Tu es le CERVEAU de l'agent.
Ton rôle N'EST PAS d'écrire le gros programme, mais de décider s'il faut ajouter UN PETIT module Python autonome.

Contexte modules déjà présents (nom + taille) :
{json.dumps(manifest, ensure_ascii=False, indent=2)}

Historique récent :
{history_txt}

Objectif global de l'agent :
- apprendre en ajoutant de petits modules
- que ces modules puissent s'initialiser (init(ctx)) et agir à chaque tick (tick(ctx))
- converger vers un système qui se réécrit et utilise ses propres capacités

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


def ask_llm_for_module_code(name: str, description: str, manifest: List[Dict[str, Any]], history_txt: str) -> str:
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
- PAS de texte autour, renvoie UNIQUEMENT le code Python.
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
        "tickers": [],
    }

    # on expose les outils du noyau AUX modules
    ctx["call_llm"] = call_llm
    ctx["write_module"] = write_module_file
    ctx["get_manifest"] = build_manifest
    ctx["get_history"] = lambda n=80: read_history_tail(n)

    loaded = load_all_modules(ctx)
    append_history({"event": "modules_loaded", "modules": loaded})

    # on laisse les modules jouer
    run_tickers(ctx)

    # noyau lui-même peut aussi demander au LLM d'ajouter un module
    manifest = build_manifest()
    hist_list = read_history_tail(30)
    hist_txt = "\n".join(f"- {h.get('event')}" for h in hist_list) or "aucun historique."

    decision = ask_llm_for_next_step(manifest, hist_txt)
    if decision.get("action") != "write_module":
        append_history({"event": "llm_decision_noop"})
        return

    mod_name = decision.get("name") or f"module_{int(time.time())}"
    mod_desc = decision.get("description") or "module utilitaire pour l'agent"

    code = ask_llm_for_module_code(mod_name, mod_desc, manifest, hist_txt)
    if not code.strip():
        append_history({"event": "llm_empty_module", "name": mod_name})
        return

    fname = write_module_file(mod_name, code)
    print(f"[agent] module ajouté: {fname}")


if __name__ == "__main__":
    main()
