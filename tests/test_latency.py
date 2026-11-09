"""Latency cost (yolopit.latency): exact on the LUT points, bilinear in between, gradient towards
the masks, the nano reproduces the composition estimate of yoloeval's fit.py, DUCCIO target in ms.

The LUT and the fit live in yoloeval (YOLOEVAL_PRUNING overrides the path): skipped without them."""
import copy
import csv
import json
import os
from pathlib import Path

import pytest
import torch

PRUNING = Path(os.environ.get("YOLOEVAL_PRUNING",
                              "/home/scolaroe/Desktop/yoloeval/experiments/pruning"))
LUT_DIR = PRUNING / "lut_pi_passo2" / "results"
FIT_DIR = PRUNING / "b4r_composizione" / "results"
ARCH = "w0.25_D126"
pytestmark = pytest.mark.skipif(not (FIT_DIR / "fit.json").exists(),
                                reason=f"no LUT/fit in {PRUNING}")
LAT = dict(lut_dir=str(LUT_DIR), fit=str(FIT_DIR / "fit.json"), arch=ARCH)


def lut_points(fam):
    return {(int(r["c_in"]), int(r["c_out"])): float(r["median_ms"])
            for r in csv.DictReader(open(LUT_DIR / f"lut_{fam}.csv"))}


@pytest.fixture(scope="module")
def search(tmp_path_factory):
    """Unpruned nano of yoloeval (width 0.25, D126 = yolo26n depth, 4 classes), N=2 = LUT step."""
    from ultralytics.nn.tasks import yaml_model_load
    from ultralytics.utils import YAML
    from yolopit import PITYOLO
    d = copy.deepcopy(yaml_model_load("yolo26n.yaml"))
    for k in ("yaml_file", "scale"):
        d.pop(k, None)
    d.update(nc=4, scales={"w": [0.5, 0.25, 1024]})
    path = tmp_path_factory.mktemp("nano") / "nano4.yaml"
    YAML.save(path, d)
    return PITYOLO(str(path), cfg={"imgsz": 640, "pit": {"n": 2, "latency": LAT}})


def test_lut_points_and_bilinear():
    from yolopit.latency import FamilyLUT
    fam = "k3s2_40"
    pts, lut = lut_points(fam), FamilyLUT(fam, LUT_DIR / f"lut_{fam}.csv")
    for (ci, co), v in list(pts.items())[::97]:                 # (a) exact on the grid
        assert float(lut(ci, co)) == pytest.approx(v, rel=1e-12, abs=0)
    assert float(lut(63, 32)) == pytest.approx((pts[62, 32] + pts[64, 32]) / 2, rel=1e-12)
    assert float(lut(64, 33)) == pytest.approx((pts[64, 32] + pts[64, 34]) / 2, rel=1e-12)
    four = (pts[62, 32] + pts[62, 34] + pts[64, 32] + pts[64, 34]) / 4
    assert float(lut(63, 33)) == pytest.approx(four, rel=1e-12)   # (b) midpoint = mean
    assert float(lut(1.5, 1)) == pytest.approx((pts[1, 1] + pts[2, 1]) / 2, rel=1e-12)
    with pytest.raises(ValueError, match="outside the measured grid"):
        lut(129, 2)                                                # above the max: no extrapolation
    with pytest.raises(ValueError, match="outside the measured grid"):
        lut(128, 128 + 1)


def test_full_nano_matches_fit(search):
    """(d) C + sum a_f LUT of the unpruned nano = composition_ms of fit.py (current files)."""
    m = search.model
    ours = float(m.real_costs()["latency"])
    assert m.fixed["latency"] == json.loads((FIT_DIR / "fit.json").read_text())["C_ms"][ARCH]
    pred = [r for r in csv.DictReader(open(FIT_DIR / "predictions.csv"))
            if r["net"] == f"{ARCH}_full" and r["split"] == "train"]
    if pred:
        ref = float(pred[0]["composition_ms"])
    else:   # not measured yet: fit.py's formula on the conv list of the exported net (layers.csv)
        fit = json.loads((FIT_DIR / "fit.json").read_text())
        a = {f: v for k, v in fit["a"].items() for f in k.split("+")}
        luts, ref = {}, fit["C_ms"][ARCH]
        for r in csv.DictReader(open(FIT_DIR / "layers.csv")):
            if r["net"] == f"{ARCH}_full":
                f = r["family"]
                luts.setdefault(f, lut_points(f))
                ref += a[f] * luts[f][int(r["c_in"]), int(r["c_out"])]
    assert ours == pytest.approx(ref, rel=1e-3)
    assert float(m.net.get_cost("latency").detach()) + m.fixed["latency"] == pytest.approx(ours, rel=1e-6)


def _blocks(net):
    from yolopit.masks import PITBlockFeaturesMasker
    seen = {}
    for mk in net.seed.modules():
        if isinstance(mk, PITBlockFeaturesMasker) and mk.trainable:
            seen.setdefault(id(mk), mk.block)
    return list(seen.values())


@pytest.mark.parametrize("discrete", [False, True])
def test_gradient_reaches_masks(search, discrete):
    """(c) d latency / d masks is not zero and always positive: the gradient comes from the
    monotone surrogate, so removing channels always lowers the cost the search sees, even where the
    LUT itself goes up (64 channels faster than 62 on the Pi); pruning a lot always pays."""
    net = search.model.net
    prev = net.discrete_cost
    net.discrete_cost = discrete
    blocks = _blocks(net)
    try:
        for p in net.nas_parameters():
            p.grad = None
        net.get_cost("latency").backward()
        assert all(b.grad is not None for b in blocks)
        g = torch.cat([b.grad.flatten() for b in blocks])
        nz = g[g != 0]                                   # keep-alive blocks: no gradient
        assert nz.numel() > 0.5 * g.numel() and bool((nz > 0).all())
        with torch.no_grad():
            before = float(net.get_cost("latency"))
            saved = [b.clone() for b in blocks]
            for b in blocks:                             # half of every mask off -> faster
                b[: len(b) // 2] = 0.0
            assert float(net.get_cost("latency")) < 0.8 * before
            for b, s in zip(blocks, saved):
                b.copy_(s)
    finally:
        net.discrete_cost = prev
        for p in net.nas_parameters():
            p.grad = None


def test_soft_channels_capped(search):
    """|alpha| > 1 gives soft channels above the real ones: capped, no error, no gradient."""
    net = search.model.net
    blocks = _blocks(net)
    with torch.no_grad():
        before = float(net.get_cost("latency"))
        for b in blocks:
            b *= 1.5
    try:
        assert float(net.get_cost("latency").detach()) == pytest.approx(before, rel=1e-6)
    finally:
        with torch.no_grad():
            for b in blocks:
                b /= 1.5


def test_duccio_target_in_ms(search):
    from yolopit.config import ConfigError, split
    from yolopit.search import resolve_regularizer
    m = search.model
    C = m.fixed["latency"]
    reg, info = resolve_regularizer(m, {"mode": "duccio", "target": {"latency": 40}})
    assert info["target"]["latency"]["target_total"] == 40
    assert reg.targets_pit["latency"] == pytest.approx(40 - C)
    reg.start(task_loss=1.0)
    r = reg(m.net, epoch=5, epochs=10)
    assert r.requires_grad and float(r) > 0                # 56 ms now, above the target: pushes
    with pytest.raises(ConfigError, match="impossible"):
        resolve_regularizer(m, {"mode": "duccio",
                                "target": {"latency": 0.9 * m.min_total["latency"]}})
    pct, _ = resolve_regularizer(m, {"mode": "duccio", "target": {"latency": "50%"}})
    assert pct.targets_pit["latency"] == pytest.approx(0.5 * m.total0["latency"] - C)
    # the cost needs pit.latency
    with pytest.raises(ConfigError, match="pit.latency"):
        split({"regularizer": {"mode": "duccio", "target": {"latency": 30}}})


def test_config_errors(tmp_path):
    from yolopit.config import ConfigError, split
    from yolopit.latency import load_fit
    with pytest.raises(ConfigError, match="key arch"):
        split({"pit": {"latency": {"lut_dir": str(LUT_DIR)}}})
    with pytest.raises(KeyError, match="not in"):
        load_fit(FIT_DIR / "fit.json", "w9_D999")


def test_estimators_same_value_different_gradient():
    """Where the LUT drops (k1s1_160, c_in 48: 64 channels 3.09 ms, 62 channels 3.98 ms) every
    estimator gives the measured value. `lut` sees only the segment to 62 (much slower), `smooth`
    always pushes towards fewer channels, `secant` (D = 32, the step spacing) reaches 32 channels,
    below the step, and pushes towards fewer channels too."""
    from yolopit.latency import ESTIMATORS
    pts = lut_points("k1s1_160")
    assert pts[48, 62] > pts[48, 64]                          # the LUT is not monotone here
    grads = {}
    for name, cls in ESTIMATORS.items():
        lut = cls("k1s1_160", LUT_DIR / "lut_k1s1_160.csv")
        c_out = torch.tensor(64.0, dtype=torch.float64, requires_grad=True)
        v = lut(48.0, c_out)
        assert float(v) == pytest.approx(pts[48, 64], rel=1e-12)
        v.backward()
        grads[name] = float(c_out.grad)
    assert grads["lut"] < 0 < grads["smooth"] and grads["secant"] > 0 and grads["best"] > 0


def test_secant_distance_follows_the_search():
    from yolopit.latency import SecantLUT
    lut = SecantLUT("k3s2_40", LUT_DIR / "lut_k3s2_40.csv")
    assert lut.d_start == 32                        # steps every 32 output channels on the Pi
    lut.progress = 0.0
    assert lut.d == 32
    lut.progress = 1.0
    assert lut.d == SecantLUT.D_END
    pts = lut_points("k3s2_40")
    slope = {}
    for c in (36, 62):                              # c_in 64: step between 32 and 34 c_out
        lut.progress = 0.0                          # D = 32: from 36 and from 62 it sees the step
        x = torch.tensor(float(c), dtype=torch.float64, requires_grad=True)
        lut(64.0, x).backward()
        slope[c] = float(x.grad)
    assert slope[36] == pytest.approx((pts[64, 36] - pts[64, 4]) / 32, rel=1e-9)
    assert slope[62] == pytest.approx((pts[64, 62] - pts[64, 30]) / 32, rel=1e-9) and slope[62] > 0
    assert SecantLUT.step_spacing({(a, c): 1.0 + 0.001 * c for a in range(1, 9) for c in range(2, 40, 2)}) \
        == SecantLUT.D_DEFAULT                      # no steps: default


def test_estimator_in_config():
    from yolopit.config import LATENCY_ESTIMATORS, ConfigError, split
    from yolopit.latency import ESTIMATORS
    assert set(LATENCY_ESTIMATORS) == set(ESTIMATORS)
    split({"pit": {"latency": dict(LAT, estimator="secant")}})
    with pytest.raises(ConfigError, match="estimator"):
        split({"pit": {"latency": dict(LAT, estimator="nope")}})


def test_default_data_shipped_with_yolopit():
    """Without lut_dir and fit the LUT and fit.json in yolopit/data/rpi5 are used."""
    from yolopit.latency import DATA, rpi5_latency
    assert (DATA / "fit.json").is_file() and len(list(DATA.glob("lut_*.csv"))) == 13
    _, c_default, luts = rpi5_latency(ARCH)
    _, c_explicit, _ = rpi5_latency(ARCH, lut_dir=DATA, fit=DATA / "fit.json")
    assert c_default == c_explicit and len(luts) == 13


def test_nnls_matches_scipy():
    import numpy as np
    from yolopit.latency import _nnls
    scipy = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(0)
    A, b = rng.normal(size=(50, 4)), rng.normal(size=50)
    assert np.allclose(_nnls(A, b), scipy.nnls(A, b)[0], atol=1e-9)


def test_best_is_the_steepest_line_below():
    """`best` at c_out 62 (c_in 64, k3s2_40): the largest saving per channel among all the points
    below, here the line to 32, just under the step."""
    from yolopit.latency import BestLUT
    pts = lut_points("k3s2_40")
    lut = BestLUT("k3s2_40", LUT_DIR / "lut_k3s2_40.csv")
    x = torch.tensor(62.0, dtype=torch.float64, requires_grad=True)
    lut(64.0, x).backward()
    below = [c for c in sorted({co for ci, co in pts if ci == 64}) if c < 62]
    best = max((pts[64, 62] - pts[64, c]) / (62 - c) for c in below)
    assert float(x.grad) == pytest.approx(best, rel=1e-9)
