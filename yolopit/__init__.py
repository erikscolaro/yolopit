"""yolopit: structured channel pruning of Ultralytics YOLO26 with PLiNIO PIT.

    from yolopit import PITYOLO, PrunedTrainer        # search (needs PLiNIO) + fine-tuning
    from yolopit.runtime import FxDetectionModel      # loading/running a pruned model only

Only `yolopit.search` and `yolopit.masks` need PLiNIO; they are imported on first use, so
`import yolopit` and loading a pruned.pt work without it.

Built on PLiNIO (https://github.com/eml-eda/plinio): see README, "Credits", for the papers to cite.
"""
from importlib import import_module

__version__ = "0.1.0"

_LAZY = {
    # runtime (torch + Ultralytics)
    "C3k2Split": "runtime", "YoloFx": "runtime", "FxDetectionModel": "runtime",
    "PrunedTrainer": "finetune",
    # search (PLiNIO)
    "PITYOLO": "search", "PITSearchTrainer": "search", "PITDetectionModel": "search",
    "build_pit_model": "search", "nas_lr_factor": "search",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module 'yolopit' has no attribute {name!r}")
