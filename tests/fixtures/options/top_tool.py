import argparse, json, os, sys
p = argparse.ArgumentParser(); p.add_argument("files", nargs="+"); p.add_argument("--json", action="store_true")
p.add_argument("--top", type=int); a = p.parse_args()
res = []
for f in a.files:
    t = open(f, encoding="utf-8").read(); res.append({"file": f, "lines": t.count("\n"), "words": len(t.split()), "characters": len(t)})
if a.top is not None:
    res = res[:a.top] if os.environ.get("BUG") == "first" else sorted(res, key=lambda r: -r["words"])[:a.top]
if a.json: print(json.dumps(res)); sys.exit(0)
for r in res: print(f"{r['file']} | {r['lines']} | {r['words']} | {r['characters']}")
