"""Short trainings on coco8 (CPU, minutes): run with `pytest -m slow`."""
import pytest

pytestmark = pytest.mark.slow


def test_manual_masks_pipeline(script):
    script("manual_masks.py")


def test_optimizer_separation(script):
    script("optimizer_separation.py")


def test_nas_scheduler(script):
    script("nas_scheduler.py")


def test_regularizers_and_ema(script):
    script("regularizers.py")
