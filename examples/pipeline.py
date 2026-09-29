"""Full-pipeline validation run: PIT search -> export -> val -> fine-tune -> val -> ONNX.

Usage (from the project folder):
    python examples/pipeline.py --data coco128.yaml --n 4 --epochs 30 --ft-epochs 15

Workflow defaults (yolopit): weights optimizer SGD, masks optimizer AdamW (separate), no LR
warmup, no PIT warmup, AMP off. Everything ends up in ./runs/validate/<name>/ (current directory):
search/ (results.csv, results.png, pit_results.png, channels.csv, weights/pruned.pt, ...),
finetune/, val_*/ and summary.json.
"""
import sys
from pathlib import Path


import argparse
import csv
import json
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

from yolopit import PITYOLO, PrunedTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolo26n.pt")
ap.add_argument("--data", default="coco8.yaml")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--cost", default="ops", choices=["ops", "params"], help="cost weighted by --lam")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--ft-epochs", type=int, default=10)
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--nas-lr", type=float, default=0.01, help="masks lr0")
ap.add_argument("--nas-lrf", type=float, default=1.0, help="masks final lr fraction (1 = constant)")
ap.add_argument("--nas-cos-lr", action="store_true", help="masks: cosine instead of linear")
ap.add_argument("--lr0", type=float, default=1e-3)
ap.add_argument("--lrf", type=float, default=0.01,
                help="weights final lr fraction during the SEARCH (1 = constant)")
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--nbs", type=int, default=16)
ap.add_argument("--device", default="cpu")
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--name", default=None)
a = ap.parse_args()

name = a.name or f"{Path(a.data).stem}_n{a.n}"
root = Path.cwd() / "runs" / "validate" / name
common = dict(imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=a.device,
              project=str(root), exist_ok=True)
checks, summary, t0 = [], {"args": vars(a)}, time.time()


def check(cond, msg):
    checks.append((bool(cond), msg))
    print(("  OK   " if cond else "  FAIL ") + msg, flush=True)


# --- reference: the input model on this dataset
ref = YOLO(a.model).val(data=a.data, name="val_input_model", plots=False, **common)
summary["map_input_model"] = ref.box.map

# --- search
search = PITYOLO(a.model, n=a.n, trace_imgsz=a.imgsz)
tr_metrics = search.train(data=a.data, epochs=a.epochs, lr0=a.lr0, lrf=a.lrf, nbs=a.nbs,
                          nas=dict(lr0=a.nas_lr, lrf=a.nas_lrf, cos_lr=a.nas_cos_lr),
                          regularizer=dict(mode="standard", **{"lambda": {a.cost: a.lam}}),
                          name="search", plots=True, **common)
tr = search.trainer
summary["map_search_final"] = tr_metrics["metrics/mAP50-95(B)"]
w_ids = {id(p) for g in tr.optimizer.param_groups for p in g["params"]}
n_ids = {id(p) for g in tr.nas_optimizer.param_groups for p in g["params"]}
check(w_ids.isdisjoint(n_ids), f"weights optimizer {type(tr.optimizer).__name__} "
      f"({len(w_ids)} params) and masks optimizer {type(tr.nas_optimizer).__name__} "
      f"({len(n_ids)} params) are disjoint")
rows = search.model.channel_report()
before = sum(r[1] for r in rows if r[3])
kept = sum(r[2] for r in rows if r[3])
summary["prunable_channels"] = [before, kept]
summary["real_cost_fraction"] = {k: float(v) for k, v in search.model.real_costs(fraction=True).items()}
check(all(r[2] % a.n == 0 or r[2] == r[1] for r in rows if r[3]),
      f"every prunable layer keeps a multiple of N={a.n} channels (or all of them)")
with open(root / "search" / "results.csv") as f:
    res = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
summary["search_curve"] = [
    dict(epoch=int(float(r["epoch"])), ops=float(r["train/ops"]), params=float(r["train/params"]),
         map50_95=float(r["metrics/mAP50-95(B)"]), lr_weights=float(r["lr/pg0"]),
         lr_masks=float(r["lr/masks"])) for r in res]
from yolopit import nas_lr_factor
exp_masks = [a.nas_lr * nas_lr_factor(e, a.epochs, a.nas_lrf, a.nas_cos_lr) for e in range(a.epochs)]
exp_w = [a.lr0 * (max(1 - e / a.epochs, 0) * (1 - a.lrf) + a.lrf) for e in range(a.epochs)]
got_masks = [x["lr_masks"] for x in summary["search_curve"]]
got_w = [x["lr_weights"] for x in summary["search_curve"]]
check(all(abs(g - e) <= 1e-4 * max(e, 1e-12) + 1e-12 for g, e in zip(got_masks, exp_masks)),
      f"masks lr follows its schedule ({got_masks[0]:.4g} -> {got_masks[-1]:.4g})")
check(all(abs(g - e) <= 1e-3 * e + 1e-12 for g, e in zip(got_w, exp_w)),
      f"weights lr follows Ultralytics' schedule ({got_w[0]:.4g} -> {got_w[-1]:.4g})")
check((root / "search" / "pit_results.png").exists() and (root / "search" / "results.png").exists(),
      "results.png and pit_results.png written")

# --- export (BN restored) and val: must equal the final search validation
pruned = search.export_pruned()
# the trainer validates with batch*2 (Ultralytics): use the same batch, otherwise rect batching
# pads the images differently and the mAP differs slightly even for an identical model
m0 = pruned.val(data=a.data, name="val_after_export", **{**common, "batch": a.batch * 2})
summary["map_after_export"] = m0.box.map
check(abs(m0.box.map - summary["map_search_final"]) < 1e-3,
      f"mAP50-95 end of search {summary['map_search_final']:.4f} == after export {m0.box.map:.4f}")
p_in = sum(p.numel() for p in YOLO(a.model).model.parameters())
p_out = sum(p.numel() for p in pruned.model.parameters())
summary["params"] = [p_in, p_out]

# --- fine-tuning of the pruned architecture
pruned.train(data=a.data, epochs=a.ft_epochs, name="finetune", trainer=PrunedTrainer, plots=True,
             lr0=a.lr0, nbs=a.nbs, **common)
m1 = pruned.val(data=a.data, name="val_final", **common)
summary["map_after_finetune"] = m1.box.map

# --- ONNX export + ONNX Runtime vs PyTorch + conv shapes
onnx_path = pruned.export(format="onnx", imgsz=a.imgsz)
g = onnx.load(onnx_path).graph
inits = {i.name: i for i in g.initializer}
couts = [int(inits[n.input[1]].dims[0]) for n in g.node
         if n.op_type == "Conv" and n.input[1] in inits]
summary["onnx"] = str(onnx_path)
x = torch.rand(1, 3, a.imgsz, a.imgsz)
sess = ort.InferenceSession(str(onnx_path))
y_ort = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
net = pruned.model.float().eval()
with torch.no_grad():
    net.model[-1].export = True       # same output format as the ONNX export
    y_pt = net(x)
    net.model[-1].export = False
y_pt = (y_pt[0] if isinstance(y_pt, (list, tuple)) else y_pt).numpy()
d = float(np.abs(y_ort - y_pt).max())
check(y_ort.shape == y_pt.shape and d < 1e-2, f"ONNX Runtime vs PyTorch: shape {y_ort.shape}, "
      f"max diff {d:.2e}")
summary["onnx_max_diff"] = d

summary["checks"] = [{"ok": ok, "msg": m} for ok, m in checks]
summary["minutes"] = round((time.time() - t0) / 60, 1)
(root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\nSUMMARY {name}: mAP50-95 input {summary['map_input_model']:.4f} | end of search "
      f"{summary['map_search_final']:.4f} | after export {summary['map_after_export']:.4f} | "
      f"after fine-tuning {summary['map_after_finetune']:.4f}")
print(f"SUMMARY {name}: prunable channels {before} -> {kept}, real cost "
      f"{summary['real_cost_fraction']['ops']:.3f} (ops), params {p_in:,} -> {p_out:,}, "
      f"{summary['minutes']} min")
print("ALL CHECKS PASSED" if all(c[0] for c in checks) else
      f"{sum(not c[0] for c in checks)} CHECK(S) FAILED")
