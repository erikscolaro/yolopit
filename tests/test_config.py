"""Search config: groups, defaults, old flat keys, errors (no model needed)."""
import pytest

from yolopit.config import ConfigError, load, merge, normalize, parse_amount, split


def test_defaults():
    ultra, g = split({"data": "d.yaml", "epochs": 3})
    assert ultra == {"data": "d.yaml", "epochs": 3}
    assert g["pit"]["n"] == 8 and g["pit"]["ema"] is False and g["pit"]["warmup_epochs"] == 0
    assert g["nas"]["optimizer"] == "AdamW" and g["nas"]["lrf"] == 1.0
    assert g["regularizer"] == {"mode": "standard", "lambda": {"ops": 1.0}}


def test_legacy_flat_keys():
    _, g = split({"lam": 2.0, "nas_lr0": 0.05, "nas_cos_lr": True, "pit_warmup_epochs": 1},
                 legacy_cost="params")
    assert g["regularizer"]["lambda"] == {"params": 2.0}
    assert g["nas"]["lr0"] == 0.05 and g["nas"]["cos_lr"] is True
    assert g["pit"]["warmup_epochs"] == 1


def test_kwargs_override_yaml_across_styles():
    base = normalize({"nas": {"lrf": 0.1, "cos_lr": True}})
    over = normalize({"nas_lrf": 0.5})
    _, g = split(merge(base, over))
    assert g["nas"]["lrf"] == 0.5 and g["nas"]["cos_lr"] is True


def test_duccio_targets():
    _, g = split({"regularizer": {"mode": "duccio", "target": {"ops": "40%", "params": "1.5M"}}})
    assert g["regularizer"] == {"mode": "duccio", "target": {"ops": "40%", "params": "1.5M"},
                                "margin": 0.05}
    _, g = split({"regularizer": {"mode": "duccio", "target": {"ops": "40%"}, "margin": 0.1}})
    assert g["regularizer"]["margin"] == 0.1


@pytest.mark.parametrize("value,expected", [
    ("40%", ("pct", 40.0)), ("1.2G", ("abs", 1.2e9)), ("800M", ("abs", 8e8)),
    ("5k", ("abs", 5e3)), (2e6, ("abs", 2e6)), ("3e5", ("abs", 3e5)), (1500, ("abs", 1500.0))])
def test_parse_amount(value, expected):
    assert parse_amount(value) == expected


@pytest.mark.parametrize("value", ["0%", "150%", "-1G", "abc", "1.2T", 0, True])
def test_parse_amount_errors(value):
    with pytest.raises(ConfigError):
        parse_amount(value)


@pytest.mark.parametrize("conf,match", [
    ({"regularizer": {"mode": "standard", "target": {"ops": "40%"}}}, "target is for mode"),
    ({"regularizer": {"mode": "duccio", "lambda": {"ops": 1.0}, "target": {"ops": "40%"}}},
     "lambda is for mode"),
    ({"regularizer": {"mode": "duccio"}}, "needs regularizer.target"),
    ({"regularizer": {"mode": "standard", "margin": 0.05}}, "margin is for mode"),
    ({"regularizer": {"mode": "duccio", "target": {"ops": "40%"}, "margin": 1.5}}, r"\[0, 1\)"),
    ({"regularizer": {"mode": "duccio", "target": {"ops": "40%"}, "margin": "5%"}}, r"\[0, 1\)"),
    ({"regularizer": {"mode": "other"}}, "mode must be"),
    ({"regularizer": {"lambda": {"latency": 1.0}}}, "unknown cost"),
    ({"regularizer": {"lambda": {"ops": -1}}}, ">= 0"),
    ({"pit": {"N": 16}}, "unknown key"),
    ({"nas": {"optimizer": "Lion"}}, "AdamW, Adam or SGD"),
    ({"nas_lr0": 0.1, "nas": {"lr0": 0.2}}, "both set"),
    ({"lam": 1.0, "regularizer": {"lambda": {"ops": 1.0}}}, "both set"),
    ({"pit": 16}, "must be a mapping"),
])
def test_errors(conf, match):
    with pytest.raises(ConfigError, match=match):
        split(conf)


def test_yaml_refuses_lr_lambda_and_dataset(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("nas:\n  lr_lambda: x\n")
    with pytest.raises(ConfigError, match="Python callable only"):
        load(p)
    p.write_text("path: /data\nnames: {0: a}\n")
    with pytest.raises(ConfigError, match="DATASET yaml"):
        load(p)
