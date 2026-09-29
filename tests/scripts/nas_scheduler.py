"""Masks (NAS) scheduler: formulas identical to Ultralytics, configuration from YAML.

1. nas_lr_factor == Ultralytics' own lambdas (trainer._setup_scheduler: linear, one_cycle cosine)
2. short searches on coco8 configured from a YAML file (linear, cosine, constant) and one with a
   Python lambda: for every epoch, `lr/masks` in results.csv == expected value (0 during the PIT
   warmup), and the weights lr follows Ultralytics' own schedule (untouched by the masks one)
3. keyword arguments override the YAML; pit_args.yaml is written; nas_lr_lambda in YAML is refused
"""
import csv
import math
import sys
from pathlib import Path
import os
_OUT_ROOT = Path(os.environ.get('YOLOPIT_TEST_OUT', Path(__file__).resolve().parents[1] / '_out'))


import yaml
from ultralytics.utils.torch_utils import one_cycle

from yolopit import PITYOLO, nas_lr_factor

OUT = _OUT_ROOT / "nas_scheduler"
OUT.mkdir(parents=True, exist_ok=True)
results = []


def check(cond, msg):
    results.append(bool(cond))
    print(("  OK   " if cond else "  FAIL ") + msg)


# --- 1. formulas vs Ultralytics' code
worst = 0.0
for total in (1, 7, 30, 300):
    for lrf in (0.01, 0.1, 0.5, 1.0):
        lin = lambda x: max(1 - x / total, 0) * (1.0 - lrf) + lrf       # trainer._setup_scheduler
        cos = one_cycle(1, lrf, total)
        for e in range(total + 1):
            worst = max(worst, abs(nas_lr_factor(e, total, lrf, False) - lin(e)),
                        abs(nas_lr_factor(e, total, lrf, True) - cos(e)))
check(worst < 1e-12, f"linear and cosine factors identical to Ultralytics' (max diff {worst:.1e})")
check(nas_lr_factor(5, 10, 1.0, False) == 1.0 and nas_lr_factor(5, 10, 1.0, True) == 1.0,
      "nas_lrf=1.0 -> constant")

# --- 2./3. short searches configured from YAML
EPOCHS, WARM, LR0 = 6, 2, 0.02
base = dict(data="coco8.yaml", epochs=EPOCHS, pit_warmup_epochs=WARM, lam=1.0, nas_lr0=LR0,
            imgsz=320, batch=4, workers=0, device="cpu", nbs=4, lr0=1e-3, lrf=0.1,
            project=str(OUT), exist_ok=True, plots=False, val=False)


def run(name, yaml_extra, kwargs=None, expect=None, fn=None):
    cfg = OUT / f"{name}.yaml"
    cfg.write_text(yaml.safe_dump({**base, **yaml_extra, "name": name}))
    s = PITYOLO("yolo26n.pt", n=4, cost="ops", trace_imgsz=320)
    s.train(cfg=str(cfg), **(kwargs or {}))
    tr = s.trainer
    rows = list(csv.DictReader(open(Path(tr.save_dir) / "results.csv")))
    got = [float(r["lr/masks"]) for r in rows]
    exp = [0.0 if e < WARM else LR0 * nas_lr_factor(e - WARM, EPOCHS - WARM, *expect, fn)
           for e in range(EPOCHS)]
    ok = len(got) == EPOCHS and all(math.isclose(g, x, rel_tol=1e-4, abs_tol=1e-9)
                                    for g, x in zip(got, exp))
    check(ok, f"{name}: lr/masks per epoch {[round(v, 5) for v in got]} == expected "
              f"{[round(v, 5) for v in exp]}")
    # weights: Ultralytics linear schedule lr0 -> lr0*lrf (no warmup), untouched by the masks one
    lf = lambda x: max(1 - x / EPOCHS, 0) * (1.0 - base["lrf"]) + base["lrf"]
    w_exp = [base["lr0"] * lf(e) for e in range(EPOCHS)]
    w_got = [float(r["lr/pg0"]) for r in rows]
    check(all(math.isclose(g, x, rel_tol=1e-3) for g, x in zip(w_got, w_exp)),
          f"{name}: weights lr/pg0 follows Ultralytics' own schedule {[round(v, 6) for v in w_got]}")
    saved = yaml.safe_load((Path(tr.save_dir) / "pit_args.yaml").read_text())
    return tr, saved


run("linear", {"nas_lrf": 0.1}, expect=(0.1, False))
run("cosine", {"nas_lrf": 0.1, "nas_cos_lr": True}, expect=(0.1, True))
run("constant", {}, expect=(1.0, False))
tr, saved = run("override", {"nas_lrf": 0.1, "nas_cos_lr": True},
                kwargs={"nas_cos_lr": False, "nas_lrf": 0.5}, expect=(0.5, False))
check(saved["nas_lrf"] == 0.5 and saved["nas_cos_lr"] is False,
      f"keyword arguments override the YAML, pit_args.yaml saved ({saved})")
custom = lambda e, total: 1.0 if e < 2 else 0.25                           # step schedule
run("custom_lambda", {}, kwargs={"nas_lr_lambda": custom}, expect=(1.0, False), fn=custom)

bad = OUT / "bad.yaml"
bad.write_text(yaml.safe_dump({**base, "nas_lr_lambda": "x"}))
try:
    PITYOLO("yolo26n.pt", n=4, cost="ops", trace_imgsz=320).train(cfg=str(bad))
    refused = False
except ValueError:
    refused = True
check(refused, "nas_lr_lambda in YAML is refused (Python callable only)")

print("\nALL CHECKS PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")
