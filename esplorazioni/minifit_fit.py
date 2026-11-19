"""Mini-fit con la testa aperta: a delle depthwise e C di w0.25_D101, gli altri a_f restano quelli di fit.json (fit_pit2).
    latenza = C + sum_f a_f * sum LUT_f ;  y - sum_{f noti} a_f S_f = C + a_dw * S_dw   (minimi quadrati, 2 parametri)
    python minifit_fit.py <lut_dir con i lut_*.csv> <minifit dir> <fit.json di partenza> [--write]
Train: rete intera + h000..h023; test: h024..h031. --write scrive <lut_dir>/fit.json (solo C di w0.25_D101).
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

lut_dir, d, fit0 = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
a = {f: float(v) for k, v in fit0["a"].items() for f in k.split("+")}
luts = {}
def lut(f, ci, co):
    if f not in luts:
        luts[f] = {(int(r["c_in"]), int(r["c_out"])): float(r["median_ms"]) for r in csv.DictReader(open(lut_dir / f"lut_{f}.csv"))}
    return luts[f].get((ci, co))
S = defaultdict(lambda: defaultdict(float)); off = defaultdict(set)
for r in csv.DictReader(open(d / "results/layers.csv")):
    v = lut(r["family"], int(r["c_in"]), int(r["c_out"]))
    if v is None: off[r["net"]].add((r["family"], r["c_in"], r["c_out"])); continue
    S[r["net"]][r["family"]] += v
y = defaultdict(list)
for r in csv.DictReader(open(d / "results/raw.csv")): y[r["net"]].append(float(r["ms"]))
y = {n: float(np.median(v)) for n, v in y.items()}
if off: print("FUORI GRIGLIA:", {n: sorted(v) for n, v in list(off.items())[:3]}, f"({len(off)} reti)")
nets = sorted(n for n in y if n not in off)
known = lambda n: sum(a[f] * s for f, s in S[n].items() if f in a)
dw = lambda n: sum(s for f, s in S[n].items() if f.startswith("dw"))
test = [n for n in nets if n[-4] == "h" and int(n[-3:]) >= 24]; train = [n for n in nets if n not in test]
A = np.array([[1.0, dw(n)] for n in train]); b = np.array([y[n] - known(n) for n in train])
(C, adw), *_ = np.linalg.lstsq(A, b, rcond=None)
def err(ns, C, adw):
    e = np.array([(C + adw * dw(n) + known(n)) / y[n] - 1 for n in ns]) * 100
    return f"bias {e.mean():+.2f} %, |max| {np.abs(e).max():.2f} %, n {len(ns)}"
print(f"C = {C:.2f} ms (testa chiusa: {fit0['C_ms']['w0.25_D101']:.2f}), a_dw = {adw:.3f}")
print("train:", err(train, C, adw)); print("test: ", err(test, C, adw))
C1 = float(np.mean(b - A[:, 1])); print(f"confronto con a_dw = 1 fisso: C = {C1:.2f};  test:", err(test, C1, 1.0))
full = [n for n in nets if n.endswith("full")][0]
print(f"rete intera: misurata {y[full]:.2f} ms, stimata {C + adw * dw(full) + known(full):.2f}; parte LUT {known(full) + adw * dw(full):.2f} ms, di cui depthwise {adw * dw(full):.2f}")
print("misurate: min %.1f, max %.1f ms" % (min(y.values()), max(y.values())))
if "--write" in sys.argv:
    out = dict(fit0); out["a"] = dict(fit0["a"]); out["a"]["dw3s1_80+dw3s1_40+dw3s1_20"] = float(adw)
    out["C_ms"] = {"w0.25_D101": float(C)}; out["nota"] = "mini-fit esplorativo con la testa Detect aperta: solo w0.25_D101"
    (lut_dir / "fit.json").write_text(json.dumps(out, indent=1)); print("scritto", lut_dir / "fit.json")
