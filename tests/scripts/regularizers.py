"""Short searches on coco8 (CPU) with the grouped config: DUCCIO, standard with two costs, EMA.

1. duccio (target ops 80%): results.csv has train/ops, train/params, train/reg; the cost term is
   0 in the PIT warmup and > 0 in the search; pit_summary.json and pit_args.yaml carry the
   target, converted to the PIT part; DUCCIO strengths are set.
2. standard with lambda {ops, params}: reg > 0, both costs tracked.
3. EMA off (default): the validated/saved EMA model is an exact copy of the weights; EMA on: not.
"""
import csv
import json
import os
from pathlib import Path

import torch
import yaml
from ultralytics.utils.torch_utils import unwrap_model

from yolopit import PITYOLO

_OUT_ROOT = Path(os.environ.get("YOLOPIT_TEST_OUT", Path(__file__).resolve().parents[1] / "_out"))
OUT = _OUT_ROOT / "regularizers"
results = []


def check(cond, msg):
    results.append(bool(cond))
    print(("  OK   " if cond else "  FAIL ") + msg)


COMMON = dict(data="coco8.yaml", imgsz=320, batch=4, workers=0, device="cpu", nbs=4, lr0=1e-3,
              project=str(OUT), exist_ok=True, plots=False, val=False)


def rows_of(tr):
    return list(csv.DictReader(open(Path(tr.save_dir) / "results.csv")))


def same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return all(torch.equal(sa[k].float().cpu(), sb[k].float().cpu()) for k in sa
               if sa[k].dtype.is_floating_point)


# --- 1) DUCCIO, from a YAML with the three groups
cfg = OUT / "duccio.yaml"
OUT.mkdir(parents=True, exist_ok=True)
cfg.write_text(yaml.safe_dump({**COMMON, "epochs": 4, "name": "duccio",
                               "pit": {"n": 8, "warmup_epochs": 1},
                               "nas": {"lr0": 0.05},
                               "regularizer": {"mode": "duccio", "target": {"ops": "80%"}}}))
s = PITYOLO("yolo26n.pt", cfg=str(cfg))
check(s.arch["n"] == 8 and s.arch["trace_imgsz"] == 320,
      f"pit.n and trace_imgsz (= imgsz) taken from the YAML: {s.arch}")
s.train()
tr = s.trainer
rows = rows_of(tr)
keys = rows[0].keys()
check({"train/ops", "train/params", "train/reg"} <= set(keys), f"cost columns in results.csv")
reg = [float(r["train/reg"]) for r in rows]
check(reg[0] == 0 and all(v > 0 for v in reg[1:]), f"cost term 0 in the warmup, > 0 after: {reg}")
summary = json.loads((Path(tr.save_dir) / "pit_summary.json").read_text())
t = summary["costs"]["ops"]
m = s.model
check(abs(t["target"] - 0.8 * m.total0["ops"]) < 1, f"target 80% of the whole model "
      f"({t['target']:.4g}) in pit_summary.json, met: {t['target_met']}")
check(summary["duccio_strengths"].get("ops", 0) > 0, f"DUCCIO strength {summary['duccio_strengths']}")
saved = yaml.safe_load((Path(tr.save_dir) / "pit_args.yaml").read_text())
check(saved["resolved"]["target"]["ops"]["target_pit"] ==
      saved["resolved"]["target"]["ops"]["target_total"] - m.fixed["ops"],
      "pit_args.yaml: target converted to the PIT part (target - fixed part)")
check(same_weights(unwrap_model(tr.ema.ema), unwrap_model(tr.model)),
      "EMA off (default): EMA model == current model")

# --- 2) standard with two costs, EMA on
s2 = PITYOLO("yolo26n.pt", n=8, trace_imgsz=320)
s2.train(**COMMON, epochs=2, name="standard_ema",
         pit={"ema": True}, regularizer={"lambda": {"ops": 1.0, "params": 0.5}})
tr2 = s2.trainer
reg2 = [float(r["train/reg"]) for r in rows_of(tr2)]
check(all(v > 0 for v in reg2), f"standard ops+params: cost term > 0 in every epoch: {reg2}")
check(not same_weights(unwrap_model(tr2.ema.ema), unwrap_model(tr2.model)),
      "EMA on: EMA model is an average, not the current weights")

# --- 3) an impossible target stops before training
try:
    PITYOLO("yolo26n.pt", n=8, trace_imgsz=320).train(
        **COMMON, epochs=1, name="impossible",
        regularizer={"mode": "duccio", "target": {"params": "1%"}})
    stopped = False
except ValueError as e:
    stopped = "impossible" in str(e)
    print("  message:", str(e)[:200], "...")
check(stopped, "target below the minimum: error before training")

print("\nALL CHECKS PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")
