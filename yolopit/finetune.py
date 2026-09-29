"""Fine-tuning of a pruned model (no PLiNIO)."""
from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

from ._common import apply_pit_defaults, enable_grads
from .runtime import FxDetectionModel


class PrunedTrainer(DetectionTrainer):
    """DetectionTrainer for a pruned FxDetectionModel: keeps the pruned architecture instead of
    rebuilding the model from the original yaml. Use:
    YOLO("pruned.pt").train(..., trainer=PrunedTrainer)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, apply_pit_defaults(overrides, "fine-tuning"), _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if not isinstance(weights, FxDetectionModel):
            raise TypeError("PrunedTrainer needs a pruned model: YOLO('pruned.pt').train(..., "
                            "trainer=PrunedTrainer)")
        return enable_grads(weights)
