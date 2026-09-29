"""Everything needed to LOAD and RUN a pruned model: torch + Ultralytics only, no PLiNIO.

A pruned checkpoint (pruned.pt) pickles `yolopit.runtime.FxDetectionModel` wrapping a
torch.fx GraphModule, so any machine that loads it needs this module importable, not PLiNIO.
The class paths below are part of the checkpoint format: do not move or rename them.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from ultralytics.nn.modules import C3k2, Detect
from ultralytics.nn.tasks import DetectionModel


class C3k2Split(nn.Module):
    """Exactly equivalent to C3k2 (C2f.forward) without `chunk`: cv1 is split into cv1a/cv1b
    (first/second half of the output channels, with their conv and BN parameters), so PIT can
    prune the two halves independently."""

    def __init__(self, src: C3k2):
        super().__init__()
        c = src.c
        conv, bn = src.cv1.conv, src.cv1.bn
        self.cv1a, self.cv1b = copy.deepcopy(src.cv1), copy.deepcopy(src.cv1)
        for half, dst in ((slice(0, c), self.cv1a), (slice(c, 2 * c), self.cv1b)):
            new_conv = nn.Conv2d(conv.in_channels, c, conv.kernel_size, conv.stride, conv.padding,
                                 conv.dilation, conv.groups, bias=conv.bias is not None)
            new_bn = nn.BatchNorm2d(c, eps=bn.eps, momentum=bn.momentum)
            with torch.no_grad():
                new_conv.weight.copy_(conv.weight[half])
                if conv.bias is not None:
                    new_conv.bias.copy_(conv.bias[half])
                for a in ("weight", "bias", "running_mean", "running_var"):
                    getattr(new_bn, a).copy_(getattr(bn, a)[half])
                new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
            dst.conv, dst.bn = new_conv, new_bn
        self.m, self.cv2 = src.m, src.cv2
        for a in ("f", "i", "type", "np"):
            if hasattr(src, a):
                setattr(self, a, getattr(src, a))

    def forward(self, x):
        y = [self.cv1a(x), self.cv1b(x)]
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class YoloFx(nn.Module):
    """Static re-implementation of DetectionModel._predict_once (traceable by fx)."""

    def __init__(self, layers: nn.Sequential, save):
        super().__init__()
        self.layers, self.save = layers, set(save)

    def forward(self, x):
        y = []
        for m in self.layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


class FxDetectionModel(DetectionModel):
    """A DetectionModel whose computation is a graph module (`net`). `self.model` only holds the
    Detect head, which the Ultralytics loss, validator and exporter look up as `model.model[-1]`.

    This base class is the PRUNED model (what pruned.pt contains). The search uses the subclass
    `yolopit.search.PITDetectionModel`, which adds the cost term and the channel report."""

    def __init__(self, net: nn.Module, detect: Detect, src: DetectionModel):
        nn.Module.__init__(self)
        self.net = net
        self.model = nn.ModuleList([detect])
        self.yaml = copy.deepcopy(getattr(src, "yaml", {}))
        self.save = []
        self.stride = src.stride
        self.names = getattr(src, "names", {})
        self.nc = getattr(src, "nc", detect.nc)
        self.args = getattr(src, "args", None)
        self.inplace = getattr(src, "inplace", True)
        self.pt_path = None
        self.pit_meta = dict(getattr(src, "pit_meta", {}) or {})

    # --- Ultralytics hooks
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        return self.net(x)

    def fuse(self, verbose=True, imgsz=640):
        return self  # BN already handled by the exporters; PIT layers must not be fused

    def is_fused(self, thresh=10):
        return True

    @property
    def is_pit(self) -> bool:
        return False


def detach_tracer(gm: nn.Module) -> nn.Module:
    """Drop the reference to the tracer class that built a torch.fx GraphModule.

    GraphModule pickles its tracer class (PLiNIO's PITTracer for an exported PIT model): without
    this, loading pruned.pt would import PLiNIO. The tracer is only needed to re-trace, never to
    run the graph."""
    for obj in (gm, getattr(gm, "graph", None)):
        if obj is not None and hasattr(obj, "_tracer_cls"):
            obj._tracer_cls = None
        if obj is not None and hasattr(obj, "_tracer_extras"):
            obj._tracer_extras = None
    return gm
