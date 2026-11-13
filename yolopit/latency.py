"""Latency cost of the Raspberry Pi 5 (OpenVINO fp16, 640x640), as a PLiNIO CostSpec.

Same structure as PLiNIO's table-based latency models (plinio.cost.mpic_latency, ne16_latency): a
CostSpec whose Conv2d function maps the layer spec to a differentiable cost. Here the cost of a
conv is a measured LUT, interpolated bilinearly on the effective channels:

    latency = C_arch + sum_f a_f * sum_{PIT convs l in family f} LUT_f(c_in_l, c_out_l)

- family f = kernel, stride and output map at 640x640 ("k3s2_40"); only the PIT convs count (the
  convs inside C2PSA/PSABlock/Detect and the non-conv nodes are in C_arch, the fixed part).
- LUT and fit ship with yolopit in yolopit/data/rpi5/ (DATA); lut_dir and fit override them.
- LUT_f: <lut_dir>/lut_<f>.csv (c_in, c_out, median_ms), on a grid of step 2 (1, the even values,
  the max of each layer); the grid of a family is the union of its layers' rectangles.
- a_f and C_arch: <fit> json, keys "a" ({family or "fam1+fam2": a}) and "C_ms" ({arch: C}).
  The shipped fit (13/11) is fitted on random pruned nets (B4-R) plus PIT nets of the nano and near archs; on held-out
  PIT nets the bias is -0.3 % (p99 1.5 %), on B4-R held-out archs p99 3.5 % (yoloeval experiments/pruning/fit_pit).
- in_channels / out_channels are PLiNIO's effective channels (soft, or binarized with
  discrete_cost), capped at the real channels of the conv: bilinear interpolation between the 4
  grid points around them is exact on the measured points and lets the gradient reach the masks
  (piecewise linear). Outside the measured grid it raises, it never extrapolates.
- three estimators, same value (the LUT), different gradient; pit.latency.estimator picks one:
    lut     the slope of the LUT segment the channels are on. The LUT is not monotone (64 channels can
            be faster than 62 on the Pi): a full layer may be told that pruning makes it slower.
    smooth  (default) the slope of a smooth monotone surrogate per family, p0 + p1*c_in + p2*c_out +
            p3*c_in*c_out with p >= 0 (relative least squares on the family's points): fewer
            channels always lower the cost the search sees, every step weighs the same.
    secant  the slope of the line from the current channels to D channels below (on each axis):
            it sees the steps within D, and a higher step gives a stronger push. D goes from a
            start value to D_END = 2 over the search epochs (the trainer sets `progress`). The
            start is found once per family from the LUT: the largest gap between consecutive
            steps (a +2 channels increase > 5 % and > 4x the family's median increase; on the Pi
            mostly every 32 output channels), capped at 64; 16 for a family without steps.
    best    the steepest line from the current channels to any measured point below (on each
            axis): the largest saving per removed channel, at any distance, with no schedule.
  The gradient is straight-through (value.detach() + g - g.detach()), as PIT does with its masks:
  targets (DUCCIO) and the final latency always use the measured values.
"""
from __future__ import annotations

import bisect
import csv
import json
from pathlib import Path

import torch
from plinio.cost import CostSpec
from plinio.cost.pattern import Conv2dGeneric

LUT_IMGSZ = 640     # the LUT was measured at this input size: the family names refer to it
DATA = Path(__file__).resolve().parent / "data" / "rpi5"   # default LUT (lut_<f>.csv) and fit.json


def _nnls(A, b):
    """min |A x - b| with x >= 0 for a few columns: every active set, the best feasible one."""
    import itertools
    import numpy as np
    best, x_best = np.inf, np.zeros(A.shape[1])
    for k in range(1, A.shape[1] + 1):
        for cols in itertools.combinations(range(A.shape[1]), k):
            sol = np.linalg.lstsq(A[:, cols], b, rcond=None)[0]
            if (sol >= 0).all():
                res = float(np.sum((A[:, cols] @ sol - b) ** 2))
                if res < best:
                    best, x_best = res, np.zeros(A.shape[1])
                    x_best[list(cols)] = sol
    return x_best


class FamilyLUT:
    """Measured latency of one conv family on its (c_in, c_out) grid, bilinear in between.
    Estimator `lut`: the gradient is the slope of the LUT itself."""
    progress = 0.0      # fraction of the search done, set by the trainer (used by `secant`)

    def __init__(self, name: str, path: Path):
        pts = {(int(r["c_in"]), int(r["c_out"])): float(r["median_ms"])
               for r in csv.DictReader(open(path))}
        self.name, self.path = name, path
        self.xs = sorted({ci for ci, _ in pts})
        self.ys = sorted({co for _, co in pts})
        self.table = torch.full((len(self.xs), len(self.ys)), float("nan"), dtype=torch.float64)
        for (ci, co), v in pts.items():
            self.table[self.xs.index(ci), self.ys.index(co)] = v
        self.points = pts

    @staticmethod
    def _bracket(axis: list, v: float):
        """(i, j): adjacent axis points with axis[i] <= v <= axis[j]; None outside the axis. A
        value on a grid point takes the segment on its left (t = 1: exact value), so the gradient
        is the slope towards fewer channels, not zero (the unpruned start is on grid points)."""
        if not axis[0] - 1e-6 <= v <= axis[-1] + 1e-6:
            return None
        j = max(bisect.bisect_left(axis, v - 1e-6), 1)
        return (0, 0) if len(axis) == 1 else (j - 1, j)

    def __call__(self, c_in, c_out) -> torch.Tensor:
        return self.bilinear(c_in, c_out)

    def bilinear(self, c_in, c_out) -> torch.Tensor:
        c_out = torch.as_tensor(c_out, dtype=torch.float64)
        c_in = torch.as_tensor(c_in, dtype=torch.float64, device=c_out.device)
        bx = self._bracket(self.xs, float(c_in.detach()))
        by = self._bracket(self.ys, float(c_out.detach()))
        corners = None
        if bx and by:
            (i0, i1), (j0, j1) = bx, by
            corners = self.table[[i0, i0, i1, i1], [j0, j1, j0, j1]].to(c_out.device)
        if corners is None or torch.isnan(corners).any():
            raise ValueError(
                f"latency: {self.name} conv with c_in={float(c_in):.2f}, c_out={float(c_out):.2f} "
                f"is outside the measured grid of {self.path} (c_in {self.xs[0]}..{self.xs[-1]}, "
                f"c_out {self.ys[0]}..{self.ys[-1]}, not a full rectangle): the LUT does not cover "
                f"this layer, measure it instead of extrapolating")
        x0, x1, y0, y1 = self.xs[i0], self.xs[i1], self.ys[j0], self.ys[j1]
        tx = (c_in - x0) / (x1 - x0) if x1 > x0 else c_in * 0
        ty = (c_out - y0) / (y1 - y0) if y1 > y0 else c_out * 0
        l00, l01, l10, l11 = corners
        return ((1 - tx) * (1 - ty) * l00 + (1 - tx) * ty * l01
                + tx * (1 - ty) * l10 + tx * ty * l11)


class SmoothLUT(FamilyLUT):
    """Estimator `smooth`: LUT value, gradient of a monotone surrogate (see the module doc)."""

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)
        import numpy as np
        ci, co, y = (np.array(c, float) for c in zip(*[(a, b, v) for (a, b), v in self.points.items()]))
        A = np.stack([np.ones_like(ci), ci, co, ci * co], 1) / y[:, None]
        self.surrogate = torch.tensor(_nnls(A, np.ones_like(y)), dtype=torch.float64)

    def __call__(self, c_in, c_out) -> torch.Tensor:
        value = self.bilinear(c_in, c_out)
        c_out = torch.as_tensor(c_out, dtype=torch.float64)
        c_in = torch.as_tensor(c_in, dtype=torch.float64, device=c_out.device)
        p = self.surrogate.to(c_out.device)
        g = p[0] + p[1] * c_in + p[2] * c_out + p[3] * c_in * c_out
        return value.detach() + g - g.detach()


class SecantLUT(FamilyLUT):
    """Estimator `secant`: LUT value, gradient = slope of the line to D channels below on each axis,
    D from the family's step spacing (`d_start`) to D_END (even) over the search. Lowering channels
    never leaves the grid (it is a union of rectangles from 1)."""
    D_DEFAULT, D_MAX, D_END = 16, 64, 2
    STEP_MIN, STEP_FACTOR = 0.05, 4.0   # a step: +2 channels cost > 5 % and > 4x the median increase

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)
        self.d_start = self.step_spacing(self.points)

    @classmethod
    def step_spacing(cls, pts: dict) -> int:
        """Largest gap between consecutive steps of the LUT (on either axis), or from the last step to
        the largest channels: from any point a line of this length crosses the step below it."""
        import numpy as np
        gaps = []
        for axis in (0, 1):
            other = sorted({k[1 - axis] for k in pts})
            def key(a, c):
                return (c, a) if axis == 0 else (a, c)
            cs = sorted({k[axis] for k in pts})
            inc = {}
            for c in cs:
                r = [pts[key(a, c + 2)] / pts[key(a, c)] - 1 for a in other
                     if key(a, c) in pts and key(a, c + 2) in pts]
                if len(r) >= 3:
                    inc[c] = float(np.mean(r))
            if not inc:
                continue
            med = float(np.median(list(inc.values())))
            steps = [c for c, v in inc.items() if v > cls.STEP_MIN and v > cls.STEP_FACTOR * med]
            if steps:
                gaps += list(np.diff(steps)) + [cs[-1] - steps[-1]]
        if not gaps:
            return cls.D_DEFAULT
        # shortcut: capped at D_MAX, a farther step is diluted over too many channels anyway
        return int(min(2 * np.ceil(max(gaps) / 2), cls.D_MAX))

    @property
    def d(self) -> int:
        d = self.d_start + (self.D_END - self.d_start) * min(max(self.progress, 0.0), 1.0)
        return max(2 * round(d / 2), self.D_END)

    def __call__(self, c_in, c_out) -> torch.Tensor:
        value = self.bilinear(c_in, c_out)
        c_out = torch.as_tensor(c_out, dtype=torch.float64)
        c_in = torch.as_tensor(c_in, dtype=torch.float64, device=c_out.device)
        with torch.no_grad():
            ci, co, v = float(c_in), float(c_out), float(value)
            lo_in, lo_out = max(ci - self.d, self.xs[0]), max(co - self.d, self.ys[0])
            s_in = (v - float(self.bilinear(lo_in, co))) / (ci - lo_in) if ci > lo_in else 0.0
            s_out = (v - float(self.bilinear(ci, lo_out))) / (co - lo_out) if co > lo_out else 0.0
        g = s_in * c_in + s_out * c_out
        return value.detach() + g - g.detach()


class BestLUT(FamilyLUT):
    """Estimator `best`: LUT value, gradient on each axis = the largest (LUT(c) - LUT(c')) / (c - c')
    over every measured c' below c, the other axis interpolated as in the value."""

    def _best(self, axis: list, line, c: float) -> float:
        below = [k for k, x in enumerate(axis) if x < c - 1e-6]
        if not below:
            return 0.0
        xs = torch.tensor([axis[k] for k in below], dtype=torch.float64)
        v = float(torch.as_tensor(line(c)))
        slopes = (v - line(xs)) / (c - xs)
        slopes = slopes[~torch.isnan(slopes)]           # holes of the L-shaped grid
        return float(slopes.max()) if slopes.numel() else 0.0

    def __call__(self, c_in, c_out) -> torch.Tensor:
        value = self.bilinear(c_in, c_out)
        c_out = torch.as_tensor(c_out, dtype=torch.float64)
        c_in = torch.as_tensor(c_in, dtype=torch.float64, device=c_out.device)
        import numpy as np
        with torch.no_grad():
            ci, co = float(c_in), float(c_out)
            T = self.table
            def along(axis_vals, fixed, fixed_axis, rows):    # LUT on an axis, other one interpolated
                i0, i1 = self._bracket(fixed_axis, fixed)
                w = (fixed - fixed_axis[i0]) / (fixed_axis[i1] - fixed_axis[i0]) if i1 != i0 else 0.0
                r = (1 - w) * rows(i0) + w * rows(i1)
                ax = torch.tensor(axis_vals, dtype=torch.float64)
                def line(x):
                    x = torch.as_tensor(x, dtype=torch.float64)
                    return torch.as_tensor(np.interp(x.numpy(), ax.numpy(), r.numpy(),
                                                     left=np.nan, right=np.nan))
                return line
            s_out = self._best(self.ys, along(self.ys, ci, self.xs, lambda i: T[i, :]), co)
            s_in = self._best(self.xs, along(self.xs, co, self.ys, lambda j: T[:, j]), ci)
        g = s_in * c_in + s_out * c_out
        return value.detach() + g - g.detach()


ESTIMATORS = {"lut": FamilyLUT, "smooth": SmoothLUT, "secant": SecantLUT, "best": BestLUT}


def load_fit(fit_path, arch: str) -> tuple[dict, float]:
    """fit json -> ({family: a_f}, C_arch in ms). A missing architecture is an error: its C is
    the latency of everything the LUT does not see, no other architecture's C stands for it."""
    fit = json.loads(Path(fit_path).read_text())
    a = {f: float(v) for key, v in fit["a"].items() for f in key.split("+")}
    if arch not in fit["C_ms"]:
        raise KeyError(f"latency: architecture {arch!r} not in {fit_path} (C_ms has "
                       f"{sorted(fit['C_ms'])}): measure its unpruned net and refit")
    return a, float(fit["C_ms"][arch])


def rpi5_latency(arch: str, lut_dir=None, fit=None, estimator: str = "smooth") -> tuple[CostSpec, float, dict]:
    """(CostSpec of the PIT part in ms, C_arch in ms, {family: estimator}). The CostSpec covers the
    PIT convs only; the trainer sets `progress` on the estimators at every search epoch. lut_dir and
    fit default to the data shipped with yolopit (DATA)."""
    lut_dir, fit = Path(lut_dir or DATA), Path(fit or DATA / "fit.json")
    if estimator not in ESTIMATORS:
        raise ValueError(f"latency: estimator {estimator!r} unknown, choose one of {sorted(ESTIMATORS)}")
    a, c_arch = load_fit(fit, arch)
    missing = [f for f in a if not (Path(lut_dir) / f"lut_{f}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"latency: no LUT for {missing} in {lut_dir}")
    luts = {f: ESTIMATORS[estimator](f, Path(lut_dir) / f"lut_{f}.csv") for f in a}

    def conv2d(spec):
        k, s, out = spec["kernel_size"], spec["stride"], spec["output_shape"]
        if spec["groups"] != 1 or k[0] != k[1] or s[0] != s[1]:
            raise ValueError(f"latency: no LUT for a conv with groups={spec['groups']}, "
                             f"kernel {k}, stride {s}")
        fam = f"k{k[0]}s{s[0]}_{out[-1]}"
        if fam not in luts:
            raise KeyError(f"latency: conv family {fam} has no LUT/a_f (available: "
                           f"{sorted(luts)}); the LUT is at {LUT_IMGSZ}x{LUT_IMGSZ}")
        # soft effective channels can pass the real ones (PIT masks |alpha| > 1): the conv never
        # has more channels than its weight, so cap them there (no gradient above, as in reality)
        c_out_max, c_in_max = spec["_parameters"]["weight"].shape[:2]
        # the LUT works on the CPU: on the GPU every conv costs syncs and tiny kernels (~4x slower);
        # .cpu()/.to() keep the autograd graph, the result goes back to the masks' device
        c_out_t = torch.as_tensor(spec["out_channels"])
        dev = c_out_t.device
        c_in = torch.clamp(torch.as_tensor(spec["in_channels"]).cpu().double(), max=c_in_max)
        c_out = torch.clamp(c_out_t.cpu().double(), max=c_out_max)
        return (a[fam] * luts[fam](c_in, c_out).float()).to(dev)

    spec = CostSpec(shared=False, default_behavior="zero")
    spec[Conv2dGeneric] = conv2d
    return spec, c_arch, luts
