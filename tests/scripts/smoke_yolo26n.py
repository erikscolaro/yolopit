"""Smoke test on YOLO26n (pretrained) using the real yolopit code paths, no training.

1. C3k2Split is exactly equivalent to C3k2 (whole network output, diff must be 0)
2. build_pit_model: every PIT conv keeps a multiple of N channels after switching off blocks
3. export + restore_exported_bn: exported model == PIT model (training-free)
4. ONNX export of the pruned model, ONNX Runtime == PyTorch
"""
import copy
import sys
from pathlib import Path
import os
_OUT_ROOT = Path(os.environ.get('YOLOPIT_TEST_OUT', Path(__file__).resolve().parents[1] / '_out'))


import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2

from yolopit.masks import PITBlockFeaturesMasker, restore_exported_bn
from yolopit import C3k2Split, YoloFx, build_pit_model
from plinio.methods.pit.nn import PITConv2d

N, IMG = 8, 320
SCRATCH = _OUT_ROOT
SCRATCH.mkdir(exist_ok=True)
results = []


def check(cond, msg):
    results.append(bool(cond))
    print(("  OK   " if cond else "  FAIL ") + msg)


def tensors(o):
    if isinstance(o, torch.Tensor):
        return [o]
    if isinstance(o, dict):
        return [t for v in o.values() for t in tensors(v)]
    if isinstance(o, (list, tuple)):
        return [t for v in o for t in tensors(v)]
    return []


def max_diff(a, b):
    return max((u - v).abs().max().item() for u, v in zip(tensors(a), tensors(b)))


torch.manual_seed(0)
x = torch.randn(2, 3, IMG, IMG)

# --- 1. C3k2Split == C3k2 on the whole network
det = YOLO("yolo26n.pt").model.float().eval()
ref = YoloFx(det.model, det.save).eval()
with torch.no_grad():
    y_ref = ref(x)
det2 = copy.deepcopy(det)
n_rep = 0
for i, layer in enumerate(det2.model):
    if isinstance(layer, C3k2):
        det2.model[i] = C3k2Split(layer).eval()
        n_rep += 1
with torch.no_grad():
    d = max_diff(y_ref, YoloFx(det2.model, det2.save).eval()(x))
check(n_rep > 0 and d == 0.0, f"{n_rep} C3k2 -> C3k2Split, network output identical (diff {d})")

# --- 2. PIT model, half of the blocks of every prunable masker off
wrapper = build_pit_model("yolo26n.pt", n=N, cost="ops", trace_imgsz=IMG, verbose=False)
pit = wrapper.net
maskers = {id(l.out_features_masker): l.out_features_masker for l in pit.seed.modules()
           if isinstance(getattr(l, "out_features_masker", None), PITBlockFeaturesMasker)}
with torch.no_grad():
    for m in maskers.values():
        m.block[: len(m.sizes) // 2] = 0.0
convs = [(n, l) for n, l in pit.seed.named_modules() if isinstance(l, PITConv2d)]
check(all(l.out_features_opt % N == 0 for _, l in convs),
      f"{len(convs)} PIT convs, {len(maskers)} block maskers: kept channels multiples of {N}")

# --- 3. export + BN restore == PIT model
wrapper.eval()
with torch.no_grad():
    y_pit = wrapper(x)
exported = pit.export().eval()
with torch.no_grad():
    d0 = max_diff(y_pit, exported(x))
restore_exported_bn(pit, exported, verbose=False)
with torch.no_grad():
    d1 = max_diff(y_pit, exported(x))
p0 = sum(p.numel() for p in det.parameters())
p1 = sum(p.numel() for p in exported.parameters())
check(d1 < 1e-3 < d0, f"export vs PIT: without BN restore {d0:.2e}, with restore {d1:.2e}")
check(p1 < p0, f"parameters {p0:,} -> {p1:,}")

# --- 4. ONNX
path = SCRATCH / "smoke_yolo26n_pruned.onnx"
torch.onnx.export(exported, x[:1], str(path), dynamo=False, opset_version=17)
onnx.checker.check_model(onnx.load(str(path)))
sess = ort.InferenceSession(str(path))
outs = sess.run(None, {sess.get_inputs()[0].name: x[:1].numpy()})
with torch.no_grad():
    ref1 = tensors(exported(x[:1]))
d2 = max(np.abs(o - r.numpy()).max() for o, r in zip(outs, ref1))
check(d2 < 1e-3, f"ONNX Runtime == PyTorch (max diff {d2:.2e})")

print("\nALL CHECKS PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")
