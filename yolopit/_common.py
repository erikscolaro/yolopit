"""Helpers shared by the search and the fine-tuning (no PLiNIO)."""
from __future__ import annotations

import torch.nn as nn
from ultralytics.utils import LOGGER


def apply_pit_defaults(overrides, who):
    """Defaults for the PIT workflow: the input model is ALREADY fine-tuned on the data and a
    fine-tuning follows, so no LR warmup; weights optimizer SGD (with Ultralytics' momentum);
    AMP off and validation on in the search. Explicit user values win, with a warning where they
    go against the workflow."""
    overrides = dict(overrides or {})
    if "warmup_epochs" not in overrides:
        overrides["warmup_epochs"] = 0
    elif float(overrides["warmup_epochs"]) > 0:
        LOGGER.warning(f"PIT ({who}): LR warmup enabled (warmup_epochs="
                       f"{overrides['warmup_epochs']}) but the input model is already trained on "
                       f"the data: the warmup is normally not needed")
    overrides.setdefault("optimizer", "SGD")
    if overrides.get("amp", False):
        LOGGER.warning(f"PIT ({who}): AMP requested but disabled (mask gradients need full "
                       f"precision)" if who == "search" else f"PIT ({who}): AMP enabled")
    if who == "search":
        overrides["amp"] = False
        if overrides.get("val", True) is False:
            LOGGER.warning("PIT (search): validation requested off but enabled (best.pt is the "
                           "epoch with the highest fitness within the budget)")
        overrides["val"] = True
    return overrides


def enable_grads(model: nn.Module) -> nn.Module:
    """Set requires_grad=True on every floating-point parameter (except Ultralytics' always-frozen
    '.dfl'). Ultralytics does the same in _setup_train, but with one warning per parameter: a model
    loaded from a checkpoint has all of them frozen, and PLiNIO's frozen maskers keep an unused
    `alpha` with requires_grad=False. Doing it beforehand keeps the log readable."""
    for name, p in model.named_parameters():
        if p.dtype.is_floating_point and ".dfl" not in name:
            p.requires_grad_(True)
    return model
