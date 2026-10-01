"""Search configuration: Ultralytics arguments at top level plus three yolopit groups.

    data: your_data.yaml        # any Ultralytics train argument
    epochs: 100
    imgsz: 640
    pit:                        # architecture of the search space
      n: 16                     # channels pruned in blocks of N
      remainder: true           # a C % N leftover block is kept (and pruned first)
      trace_imgsz: null         # image size of the cost; null -> imgsz
      warmup_epochs: 0          # epochs with frozen masks before the search
      ema: false                # EMA of the weights during the search
    nas:                        # optimizer and scheduler of the masks
      optimizer: AdamW          # AdamW | Adam | SGD
      lr0: 0.01
      lrf: 1.0                  # final lr = lr0 * lrf; 1.0 -> constant
      cos_lr: false
      weight_decay: 0.0
    regularizer:
      mode: standard            # standard | duccio
      lambda: {ops: 1.0}        # standard: weight of each cost (fraction of its initial value)
      target: {ops: 40%}        # duccio: budget per cost, absolute (1.2G, 800M, 2e6) or % of the
                                #   whole input model
      margin: 0.05              # duccio: best.pt is the best epoch with every cost within
                                #   target * (1 + margin)

The old flat keys (lam, nas_lr0, pit_warmup_epochs, ...) are still accepted and mapped here.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

from ultralytics.utils import LOGGER

METRICS = ("ops", "params")
MODES = ("standard", "duccio")

DEFAULTS = {
    "pit": dict(n=8, remainder=True, trace_imgsz=None, warmup_epochs=0, ema=False),
    "nas": dict(optimizer="AdamW", lr0=0.01, lrf=1.0, cos_lr=False, weight_decay=0.0,
                lr_lambda=None),
    "regularizer": dict(mode="standard"),
}
REG_KEYS = ("mode", "lambda", "target", "margin")
DEFAULT_MARGIN = 0.05
GROUPS = tuple(DEFAULTS)

# old flat key -> (group, key); "lam" is special (it needs the cost name)
LEGACY = {
    "pit_warmup_epochs": ("pit", "warmup_epochs"),
    "nas_optimizer": ("nas", "optimizer"),
    "nas_lr0": ("nas", "lr0"),
    "nas_lr": ("nas", "lr0"),
    "nas_lrf": ("nas", "lrf"),
    "nas_cos_lr": ("nas", "cos_lr"),
    "nas_weight_decay": ("nas", "weight_decay"),
    "nas_lr_lambda": ("nas", "lr_lambda"),
}


class ConfigError(ValueError):
    """A search configuration that cannot work (raised before anything is trained)."""


def load(cfg) -> dict:
    """YAML path, dict or None -> dict (a deep copy)."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return copy.deepcopy(cfg)
    import yaml
    conf = yaml.safe_load(Path(cfg).read_text()) or {}
    if "names" in conf and "data" not in conf:
        raise ConfigError(f"{cfg} looks like a DATASET yaml (it has 'names'): pass it as "
                          f"data='{cfg}', or put 'data: {cfg}' in the search config")
    if "lr_lambda" in (conf.get("nas") or {}) or "nas_lr_lambda" in conf:
        raise ConfigError("nas.lr_lambda cannot be set from YAML (Python callable only)")
    return conf


def merge(base: dict, override: dict) -> dict:
    """override on top of base; the three groups are merged key by key."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in GROUPS and isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def normalize(conf: dict, legacy_cost: str = "ops") -> dict:
    """Map the old flat keys (lam, nas_lr0, pit_warmup_epochs, ...) into the groups."""
    conf = copy.deepcopy(conf)
    for g in GROUPS:
        v = conf.get(g)
        if v is None:
            continue
        if not isinstance(v, dict):
            raise ConfigError(f"'{g}' must be a mapping, got {type(v).__name__}")
        conf[g] = dict(v)
    used_legacy = []
    for old, (g, k) in LEGACY.items():
        if old in conf:
            grp = conf.setdefault(g, {})
            if k in grp:
                raise ConfigError(f"'{old}' and '{g}.{k}' are both set: use only '{g}.{k}'")
            grp[k] = conf.pop(old)
            used_legacy.append(f"{old} -> {g}.{k}")
    if "lam" in conf:
        grp = conf.setdefault("regularizer", {})
        if "lambda" in grp:
            raise ConfigError("'lam' and 'regularizer.lambda' are both set: use only "
                              "'regularizer.lambda'")
        grp["lambda"] = {legacy_cost: conf.pop("lam")}
        used_legacy.append(f"lam -> regularizer.lambda.{legacy_cost}")
    if used_legacy:
        LOGGER.info(f"yolopit: old flat keys mapped ({', '.join(used_legacy)})")
    return conf


def split(conf: dict, legacy_cost: str = "ops") -> tuple[dict, dict]:
    """conf -> (Ultralytics arguments, {pit, nas, regularizer} with defaults), validated."""
    conf = normalize(conf, legacy_cost)
    groups = {g: dict(conf.pop(g, None) or {}) for g in GROUPS}

    for g, allowed in (("pit", DEFAULTS["pit"]), ("nas", DEFAULTS["nas"]),
                       ("regularizer", REG_KEYS)):
        unknown = sorted(set(groups[g]) - set(allowed))
        if unknown:
            raise ConfigError(f"unknown key(s) in '{g}': {unknown}; allowed: {sorted(allowed)}")
    groups["pit"] = {**DEFAULTS["pit"], **groups["pit"]}
    groups["nas"] = {**DEFAULTS["nas"], **groups["nas"]}
    groups["regularizer"] = _check_regularizer(groups["regularizer"])
    _check_nas(groups["nas"])
    return conf, groups


def _check_nas(nas: dict):
    if str(nas["optimizer"]).lower() not in ("adamw", "adam", "sgd"):
        raise ConfigError(f"nas.optimizer must be AdamW, Adam or SGD, got {nas['optimizer']}")


def _check_regularizer(reg: dict) -> dict:
    mode = reg.get("mode", "standard")
    if mode not in MODES:
        raise ConfigError(f"regularizer.mode must be one of {MODES}, got {mode!r}")
    if mode == "standard":
        if reg.get("target") is not None:
            raise ConfigError("regularizer.target is for mode 'duccio'; mode 'standard' uses "
                              "regularizer.lambda")
        if reg.get("margin") is not None:
            raise ConfigError("regularizer.margin is for mode 'duccio': mode 'standard' has no "
                              "targets, its best.pt is the last epoch")
        lambdas = reg.get("lambda")
        lambdas = {"ops": 1.0} if lambdas is None else lambdas
        if not isinstance(lambdas, dict) or not lambdas:
            raise ConfigError("regularizer.lambda must be a mapping like {ops: 1.0, params: 0.5}")
        _check_metrics(lambdas, "regularizer.lambda")
        for k, v in lambdas.items():
            if not isinstance(v, (int, float)) or v < 0:
                raise ConfigError(f"regularizer.lambda.{k} must be a number >= 0, got {v!r}")
        return {"mode": mode, "lambda": {k: float(v) for k, v in lambdas.items()}}
    # duccio
    if reg.get("lambda") is not None:
        raise ConfigError("regularizer.lambda is for mode 'standard': DUCCIO sets the strength "
                          "of each cost by itself from the targets")
    targets = reg.get("target")
    if not isinstance(targets, dict) or not targets:
        raise ConfigError("mode 'duccio' needs regularizer.target, e.g. {ops: 40%} or "
                          "{ops: 1.2G, params: 1.5M}")
    _check_metrics(targets, "regularizer.target")
    for k, v in targets.items():
        parse_amount(v, f"regularizer.target.{k}")
    margin = reg.get("margin")
    margin = DEFAULT_MARGIN if margin is None else margin
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 <= margin < 1:
        raise ConfigError(f"regularizer.margin must be a fraction of the target in [0, 1), e.g. "
                          f"0.05 for 5%, got {margin!r}")
    return {"mode": mode, "target": dict(targets), "margin": float(margin)}


def _check_metrics(d: dict, where: str):
    unknown = sorted(set(d) - set(METRICS))
    if unknown:
        raise ConfigError(f"{where}: unknown cost(s) {unknown}; available: {list(METRICS)}")


_AMOUNT = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([kKmMgG%]?)\s*$")
_SUFFIX = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}


def parse_amount(v, where: str = "value") -> tuple[str, float]:
    """40% -> ('pct', 40.0); 1.2G -> ('abs', 1.2e9); 800M, 5e5, 2000 -> ('abs', ...)."""
    if isinstance(v, bool):
        raise ConfigError(f"{where}: expected a number or a percentage, got {v!r}")
    if isinstance(v, (int, float)):
        if v <= 0:
            raise ConfigError(f"{where} must be > 0, got {v}")
        return "abs", float(v)
    m = _AMOUNT.match(str(v))
    if not m:
        raise ConfigError(f"{where}: cannot read {v!r} (examples: 40%, 1.2G, 800M, 2e6)")
    num, suf = float(m.group(1)), m.group(2).lower()
    if num <= 0:
        raise ConfigError(f"{where} must be > 0, got {v!r}")
    if suf == "%":
        if num > 100:
            raise ConfigError(f"{where}: {v} is more than the whole model")
        return "pct", num
    return "abs", num * _SUFFIX[suf]
