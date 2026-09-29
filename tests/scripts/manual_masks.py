"""Pipeline check on coco8 with HAND-SET masks (no search): one block off in ONE layer.

Measured on the pretrained YOLO26n without retraining: one block (8 channels) off in one layer
-> mAP50-95 0.608 -> 0.276; blocks off in 6 layers -> 0. So the hand pruning is kept minimal to
keep the mAP > 0 and make the PIT-vs-export comparison meaningful.

Checks: masks survive the Ultralytics trainer untouched, the `cost` column shows the real cost,
optimizers are separated, export gives the same mAP as the PIT model, channel counts are
multiples of N and match the report, fine-tuning and ONNX export work on the pruned model.
"""
import sys
from pathlib import Path
import os
_OUT_ROOT = Path(os.environ.get('YOLOPIT_TEST_OUT', Path(__file__).resolve().parents[1] / '_out'))
from pathlib import Path

import onnx
import torch
from ultralytics import YOLO

from yolopit.masks import PITBlockFeaturesMasker
from yolopit import PITYOLO, PrunedTrainer

N = 8
PROJECT = str(_OUT_ROOT / "runs_manual")
COMMON = dict(imgsz=320, batch=4, workers=0, device="cpu", project=PROJECT, exist_ok=True)
results = []


def check(cond, msg):
    results.append((bool(cond), msg))
    print(("  OK   " if cond else "  FAIL ") + msg)


# --- 1) build + hand-set masks: one block off in the first prunable masker with >= 4 blocks
search = PITYOLO("yolo26n.pt", n=N, trace_imgsz=320)
wrapper = search.model
maskers = {id(l.out_features_masker): l.out_features_masker
           for l in wrapper.net.seed.modules() if hasattr(l, "out_features_masker")
           and isinstance(l.out_features_masker, PITBlockFeaturesMasker)}
with torch.no_grad():
    target = next(m for m in maskers.values() if len(m.sizes) >= 4)
    target.block[0] = 0.0
hand = {id(m): m.block.detach().clone() for m in maskers.values()}
rows = wrapper.channel_report()
expected = {name: kept for name, _, kept, _ in rows}
real_cost = float(wrapper.real_costs(fraction=True)["ops"])
print(f"hand-set masks: real cost {real_cost:.3f} of the original, "
      f"prunable channels {sum(r[1] for r in rows if r[3])} -> {sum(r[2] for r in rows if r[3])}")
check(real_cost < 1.0, f"real cost fraction dropped ({real_cost:.3f})")
check(all(k % N == 0 for _, _, k, p in rows if p), f"all kept channel counts are multiples of {N}")

# --- 2) "search" with masks frozen for the whole run (PIT warmup only, no cost term)
metrics = search.train(data="coco8.yaml", epochs=1, pit_warmup_epochs=1, lam=0.0,
                       lr0=1e-5, nbs=4, name="search", plots=True, **COMMON)
tr = search.trainer
check(tr.args.amp is False, "AMP disabled in the search")
check(tr.args.optimizer == "SGD" and float(tr.args.warmup_epochs) == 0,
      f"workflow defaults: weights optimizer {tr.args.optimizer}, LR warmup "
      f"{tr.args.warmup_epochs}")
check(type(tr.nas_optimizer).__name__ == "AdamW", f"masks optimizer "
      f"{type(tr.nas_optimizer).__name__}")
w_ids = {id(p) for g in tr.optimizer.param_groups for p in g["params"]}
n_ids = {id(p) for g in tr.nas_optimizer.param_groups for p in g["params"]}
check(w_ids.isdisjoint(n_ids) and len(n_ids) > 0,
      f"weights optimizer ({len(w_ids)} params) and masks optimizer ({len(n_ids)} params) disjoint")
cur = {id(l.out_features_masker): l.out_features_masker for l in search.model.net.seed.modules()
       if hasattr(l, "out_features_masker")
       and isinstance(l.out_features_masker, PITBlockFeaturesMasker)}
same = len(cur) == len(hand) and all(
    torch.equal(a.block.detach().cpu(), b) for a, b in zip(cur.values(), hand.values()))
check(same, "hand-set masks unchanged after the trainer (warmup phase: masks frozen)")
rows_after = {name: kept for name, _, kept, _ in search.model.channel_report()}
check(rows_after == expected, "channel report unchanged after training")
map_pit = metrics["metrics/mAP50-95(B)"]

# --- 3) export -> val: same mAP as the PIT model
pruned = search.export_pruned()
m0 = pruned.val(data="coco8.yaml", name="val_after_export", **COMMON)
check(map_pit > 0 and abs(m0.box.map - map_pit) < 1e-4,
      f"mAP50-95 PIT model {map_pit:.4f} == exported {m0.box.map:.4f}")
p_orig = sum(p.numel() for p in YOLO("yolo26n.pt").model.parameters())
p_pruned = sum(p.numel() for p in pruned.model.parameters())
check(p_pruned < p_orig, f"parameters {p_orig:,} -> {p_pruned:,} "
                         f"({100 * (1 - p_pruned / p_orig):.1f}% less)")
convs = {name: m for name, m in pruned.model.net.named_modules()
         if isinstance(m, torch.nn.Conv2d)}
match = all(convs[name].out_channels == kept for name, kept in expected.items() if name in convs)
check(match, "exported conv channel counts match the channel report")

# --- 4) fine-tuning + val + ONNX
pruned.train(data="coco8.yaml", epochs=3, name="finetune", trainer=PrunedTrainer, plots=True,
             lr0=1e-4, nbs=4, **COMMON)
m1 = pruned.val(data="coco8.yaml", name="val_final", **COMMON)
check(m1.box.map > 0, f"fine-tuned model validates (mAP50-95 after export {m0.box.map:.4f} "
                      f"-> after fine-tuning {m1.box.map:.4f})")
path = pruned.export(format="onnx", imgsz=320)
g = onnx.load(path).graph
inits = {i.name: i for i in g.initializer}
w_shapes = [list(inits[n.input[1]].dims) for n in g.node if n.op_type == "Conv" and n.input[1] in inits]
ref = onnx.load(YOLO("yolo26n.pt").export(format="onnx", imgsz=320)).graph
ref_inits = {i.name: i for i in ref.initializer}
ref_shapes = [list(ref_inits[n.input[1]].dims) for n in ref.node
              if n.op_type == "Conv" and n.input[1] in ref_inits]
macs = lambda shapes: sum(s[0] * s[1] * s[2] * s[3] for s in shapes)
check(macs(w_shapes) < macs(ref_shapes),
      f"ONNX conv weights {macs(ref_shapes):,} -> {macs(w_shapes):,} elements")

print("\n" + ("ALL CHECKS PASSED" if all(r[0] for r in results) else
              f"{sum(not r[0] for r in results)} CHECK(S) FAILED"))
