# modules/20_autogrow.py

import time
import json
from collections import Counter


def _describe_event(event):
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


def _recent_activity(hist, limit=8):
    if not hist:
        return "  - aucune activité enregistrée"
    lines = []
    for event in reversed(hist[-limit:]):
        lines.append(f"  - {_describe_event(event)}")
    return "\n".join(lines)


def _format_manifest_lines(manifest):
    lines = []
    for item in manifest:
        lines.append(
            f"  - {item.get('name')} | taille={item.get('size')} | init={item.get('has_init')} | tick={item.get('has_tick')} | {item.get('summary')}"
        )
    return "\n".join(lines) or "  - aucun module"


def _progress_metrics(hist):
    counter = Counter(h.get("event", "inconnu") for h in hist)
    if not counter:
        return "aucune donnée de progression"
    tick_errors = counter.get("module_tick_error", 0)
    regen = counter.get("module_regenerated", 0)
    written = counter.get("module_written", 0)
    removed = counter.get("module_removed_after_error", 0)
    return (
        f"modules écrits:{written} | régénérés:{regen} | erreurs tick:{tick_errors} | suppressions:{removed}"
    )

LAST_RUN_KEY = "_autogrow_last"

def init(ctx):
    ctx[LAST_RUN_KEY] = 0  # pour limiter la fréquence

def tick(ctx):
    # on le fait une fois toutes les ~15s
    now = time.time()
    if now - ctx.get(LAST_RUN_KEY, 0) < 15:
        return
    ctx[LAST_RUN_KEY] = now

    call_llm = ctx.get("call_llm")
    write_module = ctx.get("write_module")
    get_manifest = ctx.get("get_manifest")
    get_history = ctx.get("get_history")

    if not (call_llm and write_module and get_manifest and get_history):
        return  # pas encore prêt

    manifest = get_manifest()
    hist = get_history(20)
    manifest_block = _format_manifest_lines(manifest)
    hist_counter = Counter(h.get("event", "inconnu") for h in hist)
    hist_summary = ", ".join(
        f"{evt}:{count}" for evt, count in hist_counter.most_common()
    ) or "aucun événement notable"
    progress_line = _progress_metrics(hist)
    hist_lines = []
    for h in hist:
        line = f"- {h.get('event')}"
        if h.get("module"):
            line += f" (module={h.get('module')})"
        if h.get("error"):
            line += f" :: {h.get('error')}"
        hist_lines.append(line)
    hist_txt = "\n".join(hist_lines) or "aucun historique."
    activity_block = _recent_activity(hist)
    quick_summary = (
        f"- Modules actifs détaillés :\n{manifest_block}\n"
        f"- Comptage des événements récents (sur {len(hist)} entrées) : {hist_summary}\n"
        f"- Indicateurs de progression : {progress_line}\n"
        f"- Activités notables :\n{activity_block}"
    )

    prompt = f"""
Tu es un SOUS-AGENT interne. Ton seul but est de faire GRANDIR l'agent en créant un PETIT module Python de plus.

Tu n'écris PAS le noyau.
Tu n'écris PAS un framework.
Tu écris juste UN module autonome, simple, qui respecte l'API du noyau :
- si besoin : def init(ctx): ...
- si besoin : def tick(ctx): ...

Synthèse rapide :
{quick_summary}

But ultime explicite : faire évoluer l'agent vers une intelligence générale émergente via un écosystème modulaire auto-améliorant.

Tu disposes d'un LLM puissant : exploite-le pour orchestrer la progression.

Procède étape par étape :
1. Observe ce résumé.
2. Choisis une amélioration concrète alignée avec le but ultime.
3. Écris le module correspondant.

Tu dois regarder ce qui existe déjà et proposer un module UTILE (diagnostic, mémo, analyse d'erreurs, orchestrateur local, etc.).

Modules déjà présents :
{json.dumps(manifest, ensure_ascii=False, indent=2)}

Historique récent :
{hist_txt}

Réponds en JSON STRICT avec EXACTEMENT ces clés :
{{
  "name": "nom_du_module_sans_extension",
  "description": "but du module en une phrase",
  "code": "CODE PYTHON COMPLET DU MODULE"
}}

Contraintes sur "code" :
- doit être du Python valide
- pas de dépendances externes
- si le module a besoin d'appeler le LLM = il utilise ctx["call_llm"]
- si le module veut créer un autre module = il utilise ctx["write_module"]
- pas de texte autour

RENVOIE UNIQUEMENT le JSON.
"""

    out = call_llm(prompt)
    if not out:
        return
    out = out.strip()
    if "```" in out:
        out = out.replace("```json", "").replace("```python", "").replace("```", "").strip()

    try:
        data = json.loads(out)
    except Exception:
        return

    name = data.get("name")
    code = data.get("code")
    if not name or not code:
        return

    write_module(name, code)
