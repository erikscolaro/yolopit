"""A pruned checkpoint must load and run WITHOUT PLiNIO (yolopit.runtime only), and the old
module names must keep working."""
import subprocess
import sys
import textwrap

import torch


def _pruned_checkpoint(path):
    from yolopit import PITYOLO
    from yolopit.masks import PITBlockFeaturesMasker

    s = PITYOLO("yolo26n.pt", n=16, cost="ops", trace_imgsz=160)
    maskers = [l.out_features_masker for l in s.model.net.seed.modules()
               if isinstance(getattr(l, "out_features_masker", None), PITBlockFeaturesMasker)]
    with torch.no_grad():
        maskers[0].block.zero_()           # prune one layer to its keep-alive block
    s.export_pruned(path=path)


def test_pruned_loads_without_plinio(tmp_path):
    ckpt = tmp_path / "pruned.pt"
    _pruned_checkpoint(ckpt)
    assert b"plinio" not in ckpt.read_bytes(), "pruned.pt still references PLiNIO"
    code = textwrap.dedent(f"""
        import sys
        sys.modules["plinio"] = None                       # any `import plinio...` fails
        import torch
        from ultralytics import YOLO
        m = YOLO({str(ckpt)!r})
        assert type(m.model).__module__ == "yolopit.runtime", type(m.model)
        m.model.float().eval()(torch.zeros(1, 3, 160, 160))
        assert not any(k.startswith("plinio") for k in sys.modules if sys.modules[k]), "plinio imported"
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_import_without_plinio():
    code = ('import sys; sys.modules["plinio"] = None; import yolopit, pit_yolo; '
            'from yolopit import PrunedTrainer, FxDetectionModel, C3k2Split; '
            'assert pit_yolo.FxDetectionModel is FxDetectionModel; print("OK")')
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_legacy_names():
    import pit_block_masks
    import pit_yolo
    import yolopit.masks
    import yolopit.search

    assert pit_yolo.PITYOLO is yolopit.search.PITYOLO
    assert pit_block_masks.apply_block_masks is yolopit.masks.apply_block_masks
