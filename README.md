# yolopit

Structured channel pruning of Ultralytics **YOLO26** with **PLiNIO PIT**, in blocks of N
channels, with the Ultralytics training experience (progress bars, `results.csv`, plots,
`YOLO(...)` API).

The dataset is yours: yolopit takes any Ultralytics dataset yaml and a model already trained
on it.

## Install

```bash
pip install "yolopit @ git+https://github.com/erikscolaro/yolopit@v0.2.0"
```

PLiNIO is installed with it, pinned to one commit (yolopit patches some PLiNIO internals).
Loading and running a pruned model does not import it: pruned checkpoints only need
`yolopit.runtime`.

For development: `pip install -e ".[test]"`.

### Versions

All dependencies are pinned, because yolopit patches internals of Ultralytics and PLiNIO:

| | version |
|---|---|
| Python | 3.11 – 3.13 (tested on 3.11 and 3.13) |
| torch / torchvision | 2.12.1 / 0.27.1 (PyPI default build, CUDA 13.0) |
| ultralytics | 8.4.165 |
| numpy | 2.2.6 |
| PLiNIO | commit `3d6b5e0` |

torch < 2.13 and numpy < 2.3 match the limits of the Axelera SDK used by
[yoloeval](https://github.com/erikscolaro/yoloeval). For a different CUDA version, first install
the same torch/torchvision versions built for your CUDA from the PyTorch index (see
https://pytorch.org/get-started/previous-versions/), then yolopit.

`requirements-lock.txt` is the whole environment the tests passed in: use it in a clean venv
(`pip install -r requirements-lock.txt && pip install --no-deps -e .`) to reproduce it exactly.

## Use

```python
from yolopit import PITYOLO, PrunedTrainer

search = PITYOLO("model_trained_on_your_data.pt", cfg="search.yaml")   # or n=16, ...
search.train()                                   # keyword arguments override the config
pruned = search.export_pruned()                  # ultralytics.YOLO, saved as pruned.pt
pruned.train(data="your_data.yaml", epochs=30, trainer=PrunedTrainer)
pruned.val(data="your_data.yaml")
```

`examples/search.yaml` lists every setting, `examples/quickstart.py` runs the workflow from the
command line, `examples/pipeline.py` also checks every step (validation after export,
fine-tuning, ONNX).

### Search config

Ultralytics train arguments at top level, plus three groups:

```yaml
pit:          {n: 16, remainder: true, trace_imgsz: null, warmup_epochs: 0, ema: false}
nas:          {optimizer: AdamW, lr0: 0.01, lrf: 1.0, cos_lr: false, weight_decay: 0.0}
regularizer:  {mode: standard, lambda: {ops: 1.0}}          # or
regularizer:  {mode: duccio, target: {ops: 40%, params: 1.5M}}
```

- `pit` fixes the search space, so `PITYOLO(...)` reads it when it builds the model.
- `nas`: optimizer and scheduler of the masks, separate from the weights.
- `regularizer`: the cost term in the loss.
  - `standard`: `lambda` per cost (each cost as a fraction of its initial value). Pruning goes
    on as long as the lambda outweighs the accuracy loss.
  - `duccio`: a target per cost, absolute (`1.2G`, `800M`, `2e6`) or `%` of the input model.
    Each cost is pushed down only while it is above its target (PLiNIO's DUCCIO). A target
    below the minimum reachable stops with an error that says what the minimum is.
- The old flat keys (`lam`, `nas_lr0`, `pit_warmup_epochs`, ...) still work.

### Costs

Both costs are tracked for the **whole model** at `trace_imgsz` (default: `imgsz`):

- `ops`: MACs of convolutions and linear layers (PLiNIO's formula) plus the attention matmuls.
- `params`: weights and biases (BatchNorm excluded: it is folded into the convolutions).
- Detect: only the branch used at inference.

PLiNIO sees only the prunable layers, so yolopit adds the part it does not see (attention,
Detect, layers feeding them), measured once. At the start the search prints total, prunable
part, fixed part and the minimum reachable (every prunable layer at one block of N).
`results.csv` has `train/ops` and `train/params` (current architecture / initial) and
`train/reg` (the cost term). `pit_summary.json` has initial, final and minimum costs and the
targets.

## What it does

- **C3k2** blocks are replaced by an exactly equivalent version without `chunk` (`C3k2Split`),
  so PIT can prune inside them. C2PSA, PSABlock (attention) and Detect are not pruned; the layers
  feeding them keep their output channels.
- Channels are switched on and off in **blocks of N**. When the channel count is not a multiple
  of N, the leftover block is the first to be pruned; layers with fewer than N channels are not
  pruned.
- The masks have **their own optimizer and scheduler** (AdamW by default), separate from the
  weights (SGD by default, any Ultralytics optimizer allowed). No warmup, AMP off and EMA off in
  the search by default, with a warning if you turn a warmup on.
- `export_pruned()` gives a plain pruned model with the trained BatchNorm values, loadable with
  `YOLO("pruned.pt")`.

## Package layout

| module | needs PLiNIO | content |
|---|---|---|
| `yolopit.runtime` | no | `C3k2Split`, `YoloFx`, `FxDetectionModel`: what a pruned checkpoint contains |
| `yolopit.finetune` | no | `PrunedTrainer` |
| `yolopit.search` | yes | `PITYOLO`, `PITSearchTrainer`, `build_pit_model` |
| `yolopit.masks` | yes | block masks, BN restore, PLiNIO fixes |
| `yolopit.config` | no | search config: groups, defaults, checks |
| `yolopit.costs` | no | real cost of the whole model, fixed part |
| `yolopit.regularizers` | yes | standard and DUCCIO cost terms |
| `yolopit.plots` | no | `pit_results.png` |

The class paths in `yolopit.runtime` are part of the checkpoint format: a pruned `.pt` stores
them, so they must not move.

**Checkpoints made before yolopit 0.1** store `pit_yolo.FxDetectionModel`. The `pit_yolo` and
`pit_block_masks` modules installed with the package keep them loadable (loading them also needs
PLiNIO, because they reference its tracer). New checkpoints need neither.

## Tests

```bash
pytest            # fast checks (~2 min on CPU)
pytest -m slow    # short trainings on coco8 (~2 min on CPU)
```

## Credits

The search is built on **PLiNIO** by the EML-EDA group at Politecnico di Torino
(https://github.com/eml-eda/plinio, Apache-2.0). As the PLiNIO authors ask, if you use this
work please acknowledge their paper:

```bibtex
@misc{plinio,
      title={PLiNIO: A User-Friendly Library of Gradient-based Methods for Complexity-aware DNN Optimization},
      author={D. {Jahier Pagliari} and M. {Risso} and B. A. {Motetti} and A. {Burrello}},
      year={2023},
      eprint={2307.09488},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

PIT, the pruning method used here, is described in:

```bibtex
@article{risso2023pit,
      title={Lightweight Neural Architecture Search for Temporal Convolutional Networks at the Edge},
      author={M. Risso and A. Burrello and F. Conti and L. Lamberti and Y. Chen and L. Benini and E. Macii and M. Poncino and D. {Jahier Pagliari}},
      journal={IEEE Transactions on Computers},
      volume={72},
      number={3},
      pages={744--758},
      year={2023},
      doi={10.1109/TC.2022.3177955}
}
```

The `duccio` mode uses PLiNIO's DUCCIO regularizer, from:

```bibtex
@inproceedings{risso2022duccio,
      title={Multi-Complexity-Loss DNAS for Energy-Efficient and Memory-Constrained Deep Neural Networks},
      author={M. Risso and A. Burrello and L. Benini and E. Macii and M. Poncino and D. {Jahier Pagliari}},
      booktitle={ACM/IEEE International Symposium on Low Power Electronics and Design (ISLPED)},
      pages={1--6},
      year={2022},
      doi={10.1145/3531437.3539720}
}
```

YOLO26 and the training framework are by Ultralytics (https://github.com/ultralytics/ultralytics).

## License

AGPL-3.0 (see `LICENSE`), the license of Ultralytics, which yolopit extends. PLiNIO is
Apache-2.0.
