# modules/00_history_tools.py

def init(ctx):
    # on garde juste une référence, au cas où d'autres modules en aient besoin
    ctx["history_tools_ready"] = True

def tick(ctx):
    # rien à faire pour l’instant
    pass
