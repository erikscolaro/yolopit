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


class Detach(nn.Module):
    """fx leaf: in training it cuts the gradient of the one2one branch, as Detect.forward does."""

    def forward(self, x):
        return x.detach() if self.training else x


class DetectConvs(nn.Module):
    """The convolutions of Detect as a traceable module (open head: PIT can prune them). Per level:
    [box, cls, box one2one, cls one2one]. View, cat and decode stay outside the graph (`assemble`)."""

    def __init__(self, d: Detect):
        super().__init__()
        self.cv2, self.cv3 = d.cv2, d.cv3
        self.one2one_cv2, self.one2one_cv3 = d.one2one_cv2, d.one2one_cv3
        self.f, self.i, self.nl, self.det = d.f, d.i, d.nl, Detach()

    def forward(self, x):
        out = []
        for i in range(self.nl):
            xd = self.det(x[i])
            out += [self.cv2[i](x[i]), self.cv3[i](x[i]), self.one2one_cv2[i](xd), self.one2one_cv3[i](xd)]
        return out


def assemble(d: Detect, o: list):
    """From the outputs of DetectConvs to what Detect.forward returns. `feats` only gives the map
    sizes (anchors), so the box outputs stand for the input features."""
    bs = o[0].shape[0]

    def head(k):
        b, c = o[k::4], o[k + 1::4]
        return {"boxes": torch.cat([t.view(bs, 4 * d.reg_max, -1) for t in b], -1),
                "scores": torch.cat([t.view(bs, d.nc, -1) for t in c], -1), "feats": list(b)}
    preds = {"one2many": head(0), "one2one": head(2)}
    if d.training:
        return preds
    y = d._inference(preds["one2one" if d.end2end else "one2many"])
    if d.end2end:
        y = d.postprocess(y.permute(0, 2, 1))
    return y if d.export else (y, preds)


def _as_list(h: nn.Module) -> nn.Module:
    """fx keeps a traced container as a plain holder: back to Sequential where the children are numbered."""
    kids = h._modules
    if type(h) is nn.Module and kids and all(k.isdigit() for k in kids):
        return nn.Sequential(*[_as_list(kids[k]) for k in sorted(kids, key=int)])
    return h


def link_head(detect: Detect, gm: nn.Module) -> Detect:
    """A Detect whose conv lists ARE the modules of the graph `gm` (open head), so bias_init and
    whatever reads detect.cv2/cv3 act on the convs that run. The input Detect is not modified."""
    d = copy.copy(detect)
    d._modules = dict(detect._modules)
    holder = gm.get_submodule(f"layers.{detect.i}")
    for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        d._modules[name] = nn.ModuleList(_as_list(getattr(holder, name)))
    return d


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
        self.open_head = bool(getattr(src, "open_head", False))   # the graph ends at the Detect convs

    # --- Ultralytics hooks
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        out = self.net(x)
        return assemble(self.model[0], out) if getattr(self, "open_head", False) else out

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
