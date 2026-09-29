"""Real cost of the deployed model: the part PIT sees plus the part it does not.

PLiNIO computes the cost only on the PIT layers (the prunable part). The fx leaves (C2PSA,
PSABlock, Detect) are opaque to it, so their convolutions and the attention matmuls are not in
its cost. They never change during the search, so they are measured once here ("fixed part"):

    real cost = PLiNIO cost of the PIT layers + fixed part

Conventions, the same for both parts:
- `ops` (MACs): PLiNIO's formula for every conv/linear (bias included, as PLiNIO does), plus the
  attention matmuls, which are not modules (counted with torch's FlopCounterMode, MAC = FLOP/2).
- `params`: PLiNIO's formula (conv/linear weights and bias). BatchNorm is not counted: it is
  folded into the convolutions at deployment.
- Detect: only the branch used at inference (one2one if `end2end`, else one2many). Both branches
  have the same size, so the choice does not change the numbers, but it is recorded.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

COUNTED = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)


def top_leaves(model: nn.Module, leaf_types) -> dict:
    """{name: module} for every leaf-type module that is not inside another leaf."""
    found = {}
    for name, m in model.named_modules():
        if isinstance(m, leaf_types) and not any(name.startswith(p + ".") for p in found):
            found[name] = m
    return found


def unused_detect_branch(detect: nn.Module) -> tuple[str, list[str]]:
    """(branch used at inference, names of the Detect children that belong to the other one)."""
    if getattr(detect, "one2one_cv2", None) is None:
        return "single", []
    if getattr(detect, "end2end", False):
        return "one2one", ["cv2", "cv3"]
    return "one2many", ["one2one_cv2", "one2one_cv3"]


def fixed_costs(model: nn.Module, leaf_types, detect: nn.Module, imgsz: int, specs: dict):
    """Costs of everything inside the fx leaves, deployed Detect branch only.

    model: the model with the SAME modules the PIT model was built from (before PIT).
    specs: {metric: PLiNIO CostSpec}. Returns ({metric: value}, info)."""
    leaves = top_leaves(model, leaf_types)
    branch, skip = unused_detect_branch(detect)
    det_name = next(n for n, m in model.named_modules() if m is detect)
    skip_prefixes = tuple(f"{det_name}.{s}" for s in skip)

    shapes, hooks = {}, []
    for lname, leaf in leaves.items():
        for sub, m in leaf.named_modules():
            full = f"{lname}.{sub}" if sub else lname
            excluded = any(full == p or full.startswith(p + ".") for p in skip_prefixes)
            if isinstance(m, COUNTED) and not excluded:
                hooks.append(m.register_forward_hook(_recorder(shapes, full)))
    was_training = model.training
    model.eval()
    counter = FlopCounterMode(display=False)
    try:
        with torch.no_grad(), counter:
            p = next(model.parameters())
            model(torch.zeros(1, 3, imgsz, imgsz, device=p.device, dtype=p.dtype))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    costs = {k: 0.0 for k in specs}
    for full, (mod, out_shape) in shapes.items():
        spec = dict(vars(mod))
        spec["output_shape"] = out_shape
        for k, cs in specs.items():
            costs[k] += float(cs[(type(mod), spec)](spec))

    # attention matmuls (not modules): FLOPs of the non-convolution ops inside each leaf
    counts = counter.get_flop_counts()
    root = type(model).__name__
    matmul_flops = 0
    for lname in leaves:
        ops = counts.get(f"{root}.{lname}", {})
        matmul_flops += sum(v for op, v in ops.items() if "convolution" not in str(op))
    if "ops" in costs:
        costs["ops"] += matmul_flops / 2
    info = dict(detect_branch=branch, leaves=sorted(leaves), counted_layers=len(shapes),
                attention_macs=matmul_flops / 2, imgsz=imgsz)
    return costs, info


def _recorder(shapes: dict, name: str):
    def hook(mod, inputs, output):      # must return None: a value would replace the output
        shapes.setdefault(name, (mod, tuple(output.shape)))
    return hook


def count_outside(model: nn.Module, leaf_types) -> int:
    """Number of conv/linear layers outside the fx leaves (they must all be PIT layers)."""
    leaves = top_leaves(model, leaf_types)
    return sum(1 for n, m in model.named_modules()
               if isinstance(m, COUNTED) and not any(n == l or n.startswith(l + ".") for l in leaves))


def fmt(value: float, metric: str) -> str:
    """1.234G MACs / 2.57M params."""
    for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= div:
            return f"{value / div:.3f}{suf}"
    return f"{value:.0f}"
