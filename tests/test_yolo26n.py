"""YOLO26n: tracing, C3k2Split equivalence, block masks, export, ONNX (no training)."""


def test_smoke_yolo26n(script):
    script("smoke_yolo26n.py")
