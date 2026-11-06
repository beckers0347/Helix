import argparse, csv, json, sys
p = argparse.ArgumentParser(); p.add_argument("file"); p.add_argument("--json", action="store_true")
p.add_argument("--delimiter", default=","); p.add_argument("--column"); p.add_argument("--precision", type=int, default=2)
a = p.parse_args()
import os
BUG = os.environ.get("BUG", "")
try:
    f = open(a.file, newline="", encoding="utf-8")
except OSError:
    print("error: cannot open", file=sys.stderr); sys.exit(1)
r = csv.reader(f, delimiter=a.delimiter); hdr = next(r); st = {}
for row in r:
    for h, v in zip(hdr, row):
        if not v.strip(): continue
        try: x = float(v)
        except ValueError:
            st.setdefault(h, {})["bad"] = True; continue
        s = st.setdefault(h, {}); s["n"] = s.get("n", 0) + 1; s["sum"] = s.get("sum", 0) + x
        s["min"] = min(s.get("min", x), x); s["max"] = max(s.get("max", x), x)
cols = [h for h in hdr if h in st and st[h].get("n") and not st[h].get("bad")]
if a.column:
    if a.column not in hdr: print("error: no column", file=sys.stderr); sys.exit(1)
    if a.column not in cols: print("error: not numeric", file=sys.stderr); sys.exit(1)
    cols = [a.column]
out = {c: {"count": st[c]["n"], "min": st[c]["min"], "max": st[c]["max"], "mean": st[c]["sum"] / st[c]["n"]} for c in cols}
if BUG == "leak":
    for c in cols: out[c]["sum"] = st[c]["sum"]
if a.json:
    print(json.dumps({c: {k: (round(v, a.precision) if isinstance(v, float) else v) for k, v in d.items()} for c, d in out.items()}, indent=2)); sys.exit(0)
fmt = (lambda v: f"{v:.{a.precision}f}")
rows = [["Column", "Count", "Min", "Max", "Mean"]]
for c, d in out.items():
    mn = f"{d['min']}" if BUG == "meanonly" else fmt(d["min"]); mx = f"{d['max']}" if BUG == "meanonly" else fmt(d["max"])
    rows.append([c, str(d["count"]), mn, mx, fmt(d["mean"])])
w = [max(len(r[i]) for r in rows) for i in range(5)]
print(" | ".join(x.ljust(y) for x, y in zip(rows[0], w))); print("-+-".join("-" * y for y in w))
for r in rows[1:]: print(" | ".join(x.ljust(y) for x, y in zip(r, w)))
