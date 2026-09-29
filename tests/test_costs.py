"""Real cost of the whole model = PLiNIO cost of the PIT layers + fixed part, checked against
the MACs of the exported ONNX graph (unpruned and at the minimum), and the DUCCIO targets."""
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

IMG = 320


def onnx_conv_macs(path) -> float:
    """MACs of the Conv nodes (weights x output pixels), as ONNX Runtime would run them."""
    import onnx
    from onnx import shape_inference
    g = shape_inference.infer_shapes(onnx.load(str(path))).graph
    shp = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
           for v in list(g.value_info) + list(g.output)}
    ini = {i.name: list(i.dims) for i in g.initializer}
    return float(sum(shp[n.output[0]][2] * shp[n.output[0]][3] * np.prod(ini[n.input[1]])
                     for n in g.node if n.op_type == "Conv"))


def export_onnx(pt, workdir: Path) -> Path:
    from ultralytics import YOLO
    local = workdir / Path(pt).name
    if Path(pt).resolve() != local.resolve():
        shutil.copy(pt, local)
    return Path(YOLO(str(local)).export(format="onnx", imgsz=IMG, simplify=False, verbose=False))


@pytest.fixture(scope="module")
def search():
    from yolopit import PITYOLO
    return PITYOLO("yolo26n.pt", n=16, trace_imgsz=IMG)


def _set_all_blocks(model, value):
    from yolopit.masks import PITBlockFeaturesMasker
    with torch.no_grad():
        for l in model.net.seed.modules():
            m = getattr(l, "out_features_masker", None)
            if isinstance(m, PITBlockFeaturesMasker):
                m.block.fill_(value)


def test_total_matches_onnx(search, tmp_path):
    m = search.model
    conv_total = m.total0["ops"] - m.pit_meta["attention_macs"]
    onnx_macs = onnx_conv_macs(export_onnx("yolo26n.pt", tmp_path))
    # PLiNIO counts the bias MAC (only the few convs that have one): a few 1e-4 of difference
    assert abs(conv_total - onnx_macs) / onnx_macs < 2e-3, (conv_total, onnx_macs)
    assert m.pit_meta["attention_macs"] > 0
    assert set(m.pit_meta["leaves"]) >= {"model.10", "model.23"}


def test_minimum_matches_pruned_onnx(search, tmp_path):
    m = search.model
    assert m.min_total["ops"] < m.total0["ops"] and m.min_total["params"] < m.total0["params"]
    _set_all_blocks(m, 0.0)
    try:
        cur = {k: float(v) for k, v in m.real_costs().items()}
        assert cur == pytest.approx(m.min_total, rel=1e-6)
        search.export_pruned(path=tmp_path / "min.pt")
        onnx_macs = onnx_conv_macs(export_onnx(tmp_path / "min.pt", tmp_path))
    finally:
        _set_all_blocks(m, 1.0)
    conv_min = m.min_total["ops"] - m.pit_meta["attention_macs"]
    assert abs(conv_min - onnx_macs) / onnx_macs < 2e-3, (conv_min, onnx_macs)
    assert {k: float(v) for k, v in m.real_costs().items()} == pytest.approx(m.total0, rel=1e-6)


def test_duccio_targets(search):
    from yolopit.config import ConfigError
    from yolopit.search import resolve_regularizer
    m = search.model
    min_pct = 100 * m.min_total["ops"] / m.total0["ops"]
    with pytest.raises(ConfigError, match="impossible") as e:
        resolve_regularizer(m, {"mode": "duccio", "target": {"ops": f"{min_pct - 1:.1f}%"}})
    assert "minimum reachable" in str(e.value) and "N=16" in str(e.value)
    reg, info = resolve_regularizer(m, {"mode": "duccio", "target": {"ops": "60%",
                                                                     "params": 2e6}})
    t = info["target"]
    assert t["ops"]["target_total"] == pytest.approx(0.6 * m.total0["ops"])
    assert reg.targets_pit["ops"] == pytest.approx(0.6 * m.total0["ops"] - m.fixed["ops"])
    assert reg.targets_pit["params"] == pytest.approx(2e6 - m.fixed["params"])
    # exactly the minimum is allowed (with a warning)
    resolve_regularizer(m, {"mode": "duccio", "target": {"ops": m.min_total["ops"]}})
