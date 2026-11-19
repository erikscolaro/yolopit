"""Fit congiunto con la testa aperta: un a_dw per tutte, una C per architettura; gli altri a_f restano quelli di fit_pit2.
    python minifit_fit_archi.py <lut_dir> <fit.json di partenza> [--write]     # legge <lut_dir>/minifit*/results
Test: reti h020 e oltre (mai nel fit). --write scrive <lut_dir>/fit_archi.json (NON fit.json: cambierebbe la chiave
delle ricerche gia' fatte)."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
lut_dir, fit0 = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
a = {f: float(v) for k, v in fit0["a"].items() for f in k.split("+")}
luts = {}
def lut(f, ci, co):
    if f not in luts:
        luts[f] = {(int(r["c_in"]), int(r["c_out"])): float(r["median_ms"]) for r in csv.DictReader(open(lut_dir / f"lut_{f}.csv"))}
    return luts[f].get((ci, co))
known, dw, y, arch, off = defaultdict(float), defaultdict(float), {}, {}, defaultdict(set)
for d in sorted(lut_dir.glob("minifit*")):
    for r in csv.DictReader(open(d / "results/layers.csv")):
        v = lut(r["family"], int(r["c_in"]), int(r["c_out"]))
        if v is None: off[r["net"]].add((r["family"], r["c_in"], r["c_out"]))
        elif r["family"].startswith("dw"): dw[r["net"]] += v
        else: known[r["net"]] += a[r["family"]] * v
    m = defaultdict(list)
    for r in csv.DictReader(open(d / "results/raw.csv")): m[r["net"]].append(float(r["ms"]))
    y.update({n: float(np.median(v)) for n, v in m.items()})
    arch.update({r["net"]: r["arch"] for r in csv.DictReader(open(d / "results/nets.csv"))})
if off: print("FUORI GRIGLIA:", len(off), "reti, es.", {n: sorted(v)[:3] for n, v in list(off.items())[:2]})
nets = sorted(n for n in y if n not in off); archs = sorted({arch[n] for n in nets})
is_test = lambda n: n[-4] == "h" and int(n[-3:]) >= 20
train = [n for n in nets if not is_test(n)]; test = [n for n in nets if is_test(n)]
A = np.array([[dw[n]] + [float(arch[n] == k) for k in archs] for n in train]); b = np.array([y[n] - known[n] for n in train])
sol = np.linalg.lstsq(A, b, rcond=None)[0]; adw, C = sol[0], dict(zip(archs, sol[1:]))
pred = lambda n: C[arch[n]] + adw * dw[n] + known[n]
def err(ns):
    e = np.array([pred(n) / y[n] - 1 for n in ns]) * 100
    return f"bias {e.mean():+.2f} %, p95 {np.percentile(np.abs(e), 95):.2f} %, max {np.abs(e).max():.2f} %, n {len(ns)}"
print(f"a_dw = {adw:.3f}"); print("train:", err(train)); print("test: ", err(test))
for k in archs:
    full = next(n for n in nets if arch[n] == k and n.endswith("full"))
    print(f"  {k}: C {C[k]:.2f} ms (chiusa {fit0['C_ms'][k]:.2f}); test {err([n for n in test if arch[n] == k])}; intera mis. {y[full]:.1f} st. {pred(full):.1f}")
if "--write" in sys.argv:
    out = dict(fit0); out["a"] = dict(fit0["a"]); out["a"]["dw3s1_80+dw3s1_40+dw3s1_20"] = float(adw)
    out["C_ms"] = {k: float(v) for k, v in C.items()}; out["nota"] = "fit esplorativo con la testa Detect aperta: a_dw unico, C per architettura"
    (lut_dir / "fit_archi.json").write_text(json.dumps(out, indent=1)); print("scritto", lut_dir / "fit_archi.json")
