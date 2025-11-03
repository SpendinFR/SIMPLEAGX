# modules/20_autogrow.py

import time
import json
from collections import Counter

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
    manifest_names = ", ".join(m.get("name", "?") for m in manifest) or "aucun module"
    hist_counter = Counter(h.get("event", "inconnu") for h in hist)
    hist_summary = ", ".join(
        f"{evt}:{count}" for evt, count in hist_counter.most_common()
    ) or "aucun événement notable"
    hist_lines = []
    for h in hist:
        line = f"- {h.get('event')}"
        if h.get("module"):
            line += f" (module={h.get('module')})"
        if h.get("error"):
            line += f" :: {h.get('error')}"
        hist_lines.append(line)
    hist_txt = "\n".join(hist_lines) or "aucun historique."
    quick_summary = (
        f"- Modules actifs (fichiers) : {manifest_names}\n"
        f"- Comptage des événements récents : {hist_summary}"
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
