"""Compatibility module: the code moved to the `yolopit` package.

Pruned checkpoints saved before yolopit 0.1 pickle `pit_yolo.FxDetectionModel`: this module keeps
them loadable (they also reference PLiNIO's tracer, so loading them still needs PLiNIO). Old
scripts doing `from pit_yolo import PITYOLO, PrunedTrainer` keep working too. New code should
import from `yolopit`.
"""
from yolopit.finetune import PrunedTrainer  # noqa: F401
from yolopit.runtime import C3k2Split, FxDetectionModel, YoloFx  # noqa: F401

try:  # search API: only when PLiNIO is installed
    from yolopit.search import (COSTS, LEAVES, PITDetectionModel, PITSearchTrainer,  # noqa: F401
                                PITYOLO, build_pit_model, nas_lr_factor)
    from yolopit.plots import plot_pit_results  # noqa: F401
except ImportError:
    pass
