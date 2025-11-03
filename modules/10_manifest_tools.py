# modules/10_manifest_tools.py

def init(ctx):
    # expose un helper ultra simple
    def list_small_modules(max_size=3000):
        mf = ctx["get_manifest"]()
        return [m for m in mf if m.get("size", 0) <= max_size]
    ctx["list_small_modules"] = list_small_modules

def tick(ctx):
    # pas d'action récurrente ici
    pass
