"""PIT (PLiNIO) channel search for Ultralytics YOLO26 with an Ultralytics-like experience.

Usage
-----
    from pit_yolo import PITYOLO

    search = PITYOLO("yolo26n.pt", n=8, cost="ops")          # PIT model, block masks of N channels
    search.train(data="coco8.yaml", epochs=30, pit_warmup_epochs=5,  # Ultralytics trainer output:
                 lam=1.0, imgsz=640, batch=16)                   # bars, tables, results.csv/png, plots
    pruned = search.export_pruned()                              # -> ultralytics.YOLO, saved as pruned.pt
    pruned.train(data="coco8.yaml", epochs=30, trainer=PrunedTrainer)   # fine-tune (same UX)
    pruned.val(data="coco8.yaml")
    pruned.export(format="onnx")

What it does
------------
- C3k2 blocks are replaced by an exactly equivalent version without `chunk` (C3k2Split), so PIT
  can prune inside them. C2PSA, PSABlock (attention) and Detect stay fx leaves (not pruned).
- The Ultralytics forward is replaced by a static, fx-traceable loop (YoloFx).
- Layers feeding a leaf keep their OUTPUT channels (inputs still adapt).
- Channels are pruned in blocks of N (pit_block_masks.apply_block_masks, leftover block first).
- FxDetectionModel wraps the traced graph as an Ultralytics DetectionModel, so the standard
  DetectionTrainer / DetectionValidator / Exporter work on it.
- PITSearchTrainer (DetectionTrainer subclass) adds: PIT warmup with frozen masks, cost term in
  the loss (the `cost` column = real cost of the current architecture / initial cost), a
  SEPARATE optimizer for the masks (Adam, constant lr, no weight decay, untouched by the
  Ultralytics LR warmup/scheduler/clipping), AMP always off, masks copied (not averaged) into
  the EMA, state_dict checkpoints (PLiNIO models cannot be pickled), a per-layer channel report.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

import pit_block_masks  # noqa: F401  (must be imported before PIT: fixes nested concats)
from pit_block_masks import (apply_block_masks, freeze_output_channels, pit_layers_feeding,
                             restore_exported_bn)

from plinio.methods import PIT
from plinio.methods.pit import graph as pit_graph
from plinio.methods.pit.nn import PITConv2d
import plinio.graph.inspection as _insp
import plinio.graph.annotation as _annot
from plinio.cost import params as _cost_params, ops as _cost_ops

from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2PSA, C3k2, Detect
from ultralytics.nn.modules.block import PSABlock
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model

LEAVES = (C2PSA, PSABlock, Detect)
COSTS = {"params": _cost_params, "ops": _cost_ops}


# ============================================================================ fx / PLiNIO patches
def _install_patches(leaves=LEAVES):
    """fx: leaf blocks are not traced inside; PLiNIO: a leaf block 'defines' its output channels."""
    if getattr(pit_graph.PITTracer, "_pit_yolo_patched", False):
        return
    orig_leaf = pit_graph.PITTracer.is_leaf_module
    pit_graph.PITTracer.is_leaf_module = (
        lambda self, m, q: isinstance(m, leaves) or orig_leaf(self, m, q))
    orig_fd = _insp.is_features_defining_op

    def features_defining(n, parent):
        if n.op == "call_module" and isinstance(parent.get_submodule(str(n.target)), leaves):
            return True
        return orig_fd(n, parent)

    _insp.is_features_defining_op = _annot.is_features_defining_op = features_defining
    pit_graph.PITTracer._pit_yolo_patched = True


_install_patches()


# ============================================================================ model rewriting
class C3k2Split(nn.Module):
    """Exactly equivalent to C3k2 (C2f.forward) without `chunk`: cv1 is split into cv1a/cv1b
    (first/second half of the output channels, with their conv and BN parameters)."""

    def __init__(self, src: C3k2):
        super().__init__()
        c = src.c
        conv, bn = src.cv1.conv, src.cv1.bn
        self.cv1a, self.cv1b = copy.deepcopy(src.cv1), copy.deepcopy(src.cv1)
        for half, dst in ((slice(0, c), self.cv1a), (slice(c, 2 * c), self.cv1b)):
            new_conv = nn.Conv2d(conv.in_channels, c, conv.kernel_size, conv.stride, conv.padding,
                                 conv.dilation, conv.groups, bias=conv.bias is not None)
            new_bn = nn.BatchNorm2d(c, eps=bn.eps, momentum=bn.momentum)
            with torch.no_grad():
                new_conv.weight.copy_(conv.weight[half])
                if conv.bias is not None:
                    new_conv.bias.copy_(conv.bias[half])
                for a in ("weight", "bias", "running_mean", "running_var"):
                    getattr(new_bn, a).copy_(getattr(bn, a)[half])
                new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
            dst.conv, dst.bn = new_conv, new_bn
        self.m, self.cv2 = src.m, src.cv2
        for a in ("f", "i", "type", "np"):
            if hasattr(src, a):
                setattr(self, a, getattr(src, a))

    def forward(self, x):
        y = [self.cv1a(x), self.cv1b(x)]
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class YoloFx(nn.Module):
    """Static re-implementation of DetectionModel._predict_once (traceable by fx)."""

    def __init__(self, layers: nn.Sequential, save):
        super().__init__()
        self.layers, self.save = layers, set(save)

    def forward(self, x):
        y = []
        for m in self.layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


# ============================================================================ Ultralytics wrapper
class FxDetectionModel(DetectionModel):
    """A DetectionModel whose computation is a traced graph (`net`): the PIT model during the
    search, the pruned (exported) model afterwards. `self.model` only holds the Detect head, which
    the Ultralytics loss, validator and exporter look up as `model.model[-1]`."""

    def __init__(self, net: nn.Module, detect: Detect, src: DetectionModel):
        nn.Module.__init__(self)
        self.net = net
        self.model = nn.ModuleList([detect])
        self.yaml = copy.deepcopy(getattr(src, "yaml", {}))
        self.save = []
        self.stride = src.stride
        self.names = getattr(src, "names", {})
        self.nc = getattr(src, "nc", detect.nc)
        self.args = getattr(src, "args", None)
        self.inplace = getattr(src, "inplace", True)
        self.pt_path = None
        self.pit_meta = {}
        self.lam = 0.0          # cost weight in the loss (search only)
        self.cost0 = None       # initial cost (search only)

    # --- Ultralytics hooks
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        return self.net(x)

    def fuse(self, verbose=True, imgsz=640):
        return self  # BN already handled by the exporters; PIT layers must not be fused

    def is_fused(self, thresh=10):
        return True

    @property
    def is_pit(self) -> bool:
        return isinstance(self.net, PIT)

    def loss(self, batch, preds=None):
        loss, items = super().loss(batch, preds)
        if self.is_pit and self.cost0:
            cost = self.net.cost / self.cost0              # soft cost fraction (differentiable)
            if self.lam > 0 and self.training:
                # the criterion returns a VECTOR of components (summed by the trainer): the cost
                # term is appended as one more component, scaled like the others (x batch size)
                term = (self.lam * cost * batch["img"].shape[0]).reshape(1).to(loss.dtype)
                loss = torch.cat([loss.reshape(-1), term.to(loss.device)])
            items = dict(items)
            items["cost"] = self.real_cost_fraction().to(loss.device)
        return loss, items

    @torch.no_grad()
    def real_cost_fraction(self) -> torch.Tensor:
        """Cost of the CURRENT architecture (binarized masks, i.e. what the export would give),
        as a fraction of the initial cost. Shown as the `cost` column."""
        prev = self.net.discrete_cost
        self.net.discrete_cost = True
        try:
            return (self.net.cost / self.cost0).detach().reshape(())
        finally:
            self.net.discrete_cost = prev

    # --- reports
    def channel_report(self):
        """[(layer, channels_before, channels_kept, trainable)] for every PIT conv."""
        if not self.is_pit:
            return []
        from pit_block_masks import PITBlockFeaturesMasker
        from plinio.methods.pit.nn.features_masker import (PITConcatFeaturesMasker,
                                                           PITFeaturesMasker,
                                                           PITFrozenFeaturesMasker)
        rows = []
        for name, layer in self.net.seed.named_modules():
            if isinstance(layer, PITConv2d):
                m = layer.out_features_masker
                prunable = isinstance(m, PITBlockFeaturesMasker) or (
                    isinstance(m, PITFeaturesMasker)
                    and not isinstance(m, (PITFrozenFeaturesMasker, PITConcatFeaturesMasker))
                    and m.out_channels > 1)
                rows.append((name, int(layer.out_channels), int(layer.out_features_opt), prunable))
        return rows


def build_pit_model(model="yolo26n.pt", n=8, remainder=True, cost="ops", trace_imgsz=320,
                    verbose=True) -> FxDetectionModel:
    """YOLO26 weights/yaml -> FxDetectionModel wrapping a PIT model with block masks."""
    src = YOLO(model).model.float().eval()
    for p in src.parameters():      # checkpoints load frozen; the trainer would re-enable them
        p.requires_grad_(True)       # anyway, with one warning per parameter
    for i, layer in enumerate(src.model):
        if isinstance(layer, C3k2):
            src.model[i] = C3k2Split(layer).eval()
    detect = src.model[-1]
    net = YoloFx(src.model, src.save).eval()
    detect.export = True                    # single-tensor output while PLiNIO traces the graph
    try:
        pit = PIT(net, cost=COSTS[cost], input_shape=(3, trace_imgsz, trace_imgsz),
                  train_rf=False, train_dilation=False)
    finally:
        detect.export = False
    feeding = pit_layers_feeding(pit, LEAVES)
    freeze_output_channels(pit, feeding, verbose=verbose)
    converted, skipped, frozen = apply_block_masks(pit, n, remainder=remainder, verbose=verbose)
    wrapper = FxDetectionModel(pit, detect, src)
    with torch.no_grad():
        wrapper.cost0 = float(pit.cost)
    wrapper.pit_meta = dict(model=str(model), n=n, remainder=remainder, cost=cost,
                            trace_imgsz=trace_imgsz, cost0=wrapper.cost0)
    return wrapper



def _apply_pit_defaults(overrides, who):
    """Defaults for the PIT workflow: the input model is ALREADY fine-tuned on the data and a
    fine-tuning follows, so no LR warmup; weights optimizer SGD (with Ultralytics' momentum);
    AMP off. Explicit user values win, with a warning where they go against the workflow."""
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
    return overrides

def plot_pit_results(save_dir, channel_rows=None):
    """pit_results.png in save_dir: cost (real, train/val), mAP50-95 and mAP50, learning rates of
    weights and masks (from results.csv), and kept channels per prunable layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    with open(save_dir / "results.csv") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return None
    col = lambda k: [float(r[k]) for r in rows] if k in rows[0] else None
    ep = col("epoch")
    # epochs without validation (val=False: Ultralytics validates only the last one) are written
    # as 0 in results.csv: plot the validation series only on the epochs actually validated
    vloss = next((k for k in rows[0] if k.startswith("val/") and k.endswith("_loss")), None)
    validated = [float(r[vloss]) > 0 for r in rows] if vloss else [True] * len(rows)
    vcol = lambda k: ([e for e, v in zip(ep, validated) if v],
                      [x for x, v in zip(col(k), validated) if v]) if col(k) else None
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    a = ax[0, 0]
    if col("train/cost"):
        a.plot(ep, col("train/cost"), marker=".", label="train")
    if vcol("val/cost"):
        a.plot(*vcol("val/cost"), marker="o", label="val")
    a.set_title("cost of the current architecture / initial cost")
    a.set_ylim(0, 1.05); a.legend(); a.grid(alpha=.3)
    a = ax[0, 1]
    for k, lab in (("metrics/mAP50-95(B)", "mAP50-95"), ("metrics/mAP50(B)", "mAP50")):
        if vcol(k):
            a.plot(*vcol(k), marker="o", label=lab)
    a.set_title("validation mAP" + ("" if all(validated) else " (validated epochs only)"))
    a.set_xlim(min(ep) - 0.5, max(ep) + 0.5); a.legend(); a.grid(alpha=.3)
    a = ax[1, 0]
    for k in [k for k in rows[0] if k.startswith("lr/pg")]:
        a.plot(ep, col(k), marker=".", label=f"weights {k[3:]}")
    if col("lr/masks"):
        a.plot(ep, col("lr/masks"), "k--", marker="x", linewidth=2, label="masks")
    a.set_yscale("symlog", linthresh=1e-6)
    a.set_title("learning rates (masks: separate optimizer, 0 = masks frozen)")
    a.legend(fontsize=8); a.grid(alpha=.3)
    a = ax[1, 1]
    if channel_rows:
        pr = [r for r in channel_rows if r[3]]
        x = range(len(pr))
        a.bar(x, [r[1] for r in pr], color="#cccccc", label="original")
        a.bar(x, [r[2] for r in pr], color="#1f77b4", label="kept")
        a.set_title(f"channels per prunable layer ({sum(r[2] for r in pr)}/"
                    f"{sum(r[1] for r in pr)} kept)")
        a.set_xlabel("prunable layer (network order)"); a.legend()
    else:
        a.axis("off")
    fig.tight_layout()
    out = save_dir / "pit_results.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out



def nas_lr_factor(e: int, total: int, lrf: float = 1.0, cos: bool = False, fn=None) -> float:
    """Multiplier of nas_lr0 at search epoch e (0-based) of `total` search epochs.

    Same formulas as Ultralytics' weights scheduler (trainer._setup_scheduler), from 1 to lrf:
      linear (cos=False): max(1 - e/total, 0) * (1 - lrf) + lrf
      cosine (cos=True) : ((1 - cos(e*pi/total)) / 2) * (lrf - 1) + 1      (one_cycle(1, lrf))
    lrf=1.0 -> constant lr. `fn(e, total) -> factor` (Python only) replaces both."""
    import math
    if fn is not None:
        return float(fn(e, total))
    total = max(int(total), 1)
    if cos:
        return ((1 - math.cos(e * math.pi / total)) / 2) * (lrf - 1.0) + 1.0
    return max(1 - e / total, 0) * (1.0 - lrf) + lrf


def _enable_grads(model: nn.Module) -> nn.Module:
    """Set requires_grad=True on every floating-point parameter (except Ultralytics' always-frozen
    '.dfl'). Ultralytics does the same in _setup_train, but with one warning per parameter: a model
    loaded from a checkpoint has all of them frozen, and PLiNIO's frozen maskers keep an unused
    `alpha` with requires_grad=False. Doing it beforehand keeps the log readable."""
    for name, p in model.named_parameters():
        if p.dtype.is_floating_point and ".dfl" not in name:
            p.requires_grad_(True)
    return model

# ============================================================================ search trainer
class PITSearchTrainer(DetectionTrainer):
    """DetectionTrainer running the PIT search on an FxDetectionModel.

    Extra settings are class attributes (Ultralytics validates the `overrides` keys):
      PIT_MODEL, WARMUP_EPOCHS (PIT warmup: masks frozen), LAM, NAS_OPTIMIZER, NAS_LR (lr0 of
      the masks), NAS_LRF, NAS_COS_LR, NAS_LR_LAMBDA, NAS_WEIGHT_DECAY.

    Masks vs weights:
      - the masks have their OWN optimizer and their OWN scheduler (nas_lr_factor: linear or
        cosine from NAS_LR to NAS_LR*NAS_LRF over the SEARCH epochs, stepped per epoch like
        Ultralytics; NAS_LRF=1 -> constant), completely separate from the model's: Ultralytics'
        LR warmup, momentum warmup, LR scheduler and gradient clipping act only on the weights;
      - the mask optimizer steps only in the search phase (epoch >= WARMUP_EPOCHS);
      - AMP is always disabled (mask gradients need full precision)."""

    PIT_MODEL: FxDetectionModel | None = None
    WARMUP_EPOCHS = 0
    LAM = 1.0
    NAS_LR = 0.01               # lr0 of the masks
    NAS_LRF = 1.0               # final lr = NAS_LR * NAS_LRF (1.0 -> constant)
    NAS_COS_LR = False          # cosine instead of linear
    NAS_LR_LAMBDA = None        # Python only: fn(search_epoch, search_epochs) -> factor
    NAS_WEIGHT_DECAY = 0.0
    NAS_OPTIMIZER = "AdamW"     # "AdamW", "Adam" or "SGD"

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, _apply_pit_defaults(overrides, "search"), _callbacks)
        if self.WARMUP_EPOCHS > 0:
            LOGGER.warning(f"PIT (search): PIT warmup enabled (masks frozen for the first "
                           f"{self.WARMUP_EPOCHS} epochs) but the input model is already trained "
                           f"on the data: normally not needed")
        self.nas_optimizer = None
        self.search_phase = False
        self.add_callback("on_train_epoch_start", PITSearchTrainer._on_epoch_start)
        self.add_callback("on_train_epoch_end", PITSearchTrainer._on_epoch_end)
        self.add_callback("on_train_end", PITSearchTrainer._on_train_end)

    # --- model
    def get_model(self, cfg=None, weights=None, verbose=True):
        return self.PIT_MODEL

    def setup_model(self):
        # PIT warmup / search phase then sets the masks' requires_grad at every epoch start
        self.model = _enable_grads(self.PIT_MODEL)
        return None

    # --- phases
    @staticmethod
    def _on_epoch_start(trainer):
        model = unwrap_model(trainer.model)
        trainer.search_phase = trainer.epoch >= trainer.WARMUP_EPOCHS
        model.net.train_features = trainer.search_phase
        model.lam = trainer.LAM if trainer.search_phase else 0.0
        if trainer.search_phase and trainer.nas_optimizer is not None:
            lr = trainer.NAS_LR * trainer.nas_lr_factor(trainer.epoch)
            for g in trainer.nas_optimizer.param_groups:
                g["lr"] = lr
        if trainer.epoch in (0, trainer.WARMUP_EPOCHS):
            phase = ("search (masks trainable, cost in the loss)" if trainer.search_phase
                     else "warmup (masks frozen)")
            LOGGER.info(f"{colorstr('PIT:')} epoch {trainer.epoch + 1}: {phase}")

    def nas_lr_factor(self, epoch: int) -> float:
        """Scheduler factor of the masks lr at (0-based) training epoch `epoch`."""
        total = max(self.epochs - self.WARMUP_EPOCHS, 1)
        return nas_lr_factor(max(epoch - self.WARMUP_EPOCHS, 0), total, self.NAS_LRF,
                             self.NAS_COS_LR, self.NAS_LR_LAMBDA)

    @staticmethod
    def _on_epoch_end(trainer):
        # masks lr in results.csv (0 during the PIT warmup: the masks optimizer does not step)
        if trainer.nas_optimizer is not None:
            lr = trainer.nas_optimizer.param_groups[0]["lr"] if trainer.search_phase else 0.0
            trainer.lr["lr/masks"] = lr

    # --- optimizers: weights -> Ultralytics optimizer, masks -> their own optimizer
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5,
                        iterations=1e5):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        nas = list(unwrap_model(model).net.nas_parameters())
        nas_ids = {id(p) for p in nas}
        for g in optimizer.param_groups:
            g["params"] = [p for p in g["params"] if id(p) not in nas_ids]
        n_left = sum(1 for g in optimizer.param_groups for p in g["params"] if id(p) in nas_ids)
        assert n_left == 0, "mask parameters leaked into the weights optimizer"
        kind = self.NAS_OPTIMIZER.lower()
        if kind == "sgd":
            self.nas_optimizer = torch.optim.SGD(nas, lr=self.NAS_LR, momentum=0.9,
                                                 weight_decay=self.NAS_WEIGHT_DECAY)
        elif kind in ("adam", "adamw"):
            cls = torch.optim.AdamW if kind == "adamw" else torch.optim.Adam
            self.nas_optimizer = cls(nas, lr=self.NAS_LR, weight_decay=self.NAS_WEIGHT_DECAY)
        else:
            raise ValueError(f"NAS_OPTIMIZER must be 'AdamW', 'Adam' or 'SGD', "
                             f"got {self.NAS_OPTIMIZER}")
        n_w = sum(len(g["params"]) for g in optimizer.param_groups)
        LOGGER.info(f"{colorstr('PIT:')} the line above counts the masks too: after removing them "
                    f"the weights optimizer ({type(optimizer).__name__}) has {n_w} parameters")
        sched = ("custom (nas_lr_lambda)" if self.NAS_LR_LAMBDA is not None else
                 "constant" if self.NAS_LRF == 1.0 else
                 f"{'cosine' if self.NAS_COS_LR else 'linear'} {self.NAS_LR} -> "
                 f"{self.NAS_LR * self.NAS_LRF:g}")
        LOGGER.info(f"{colorstr('PIT:')} masks optimizer {type(self.nas_optimizer).__name__}"
                    f"(lr0={self.NAS_LR}, weight_decay={self.NAS_WEIGHT_DECAY}), scheduler "
                    f"{sched} over the search epochs, {len(nas)} mask parameters (separate from "
                    f"the weights: no Ultralytics LR warmup/scheduler/grad clipping)")
        return optimizer

    def optimizer_step(self):
        """Weights: same as Ultralytics (unscale, clip, step, EMA). Masks: own optimizer, only in
        the search phase; copied (not averaged) into the EMA model."""
        self.scaler.unscale_(self.optimizer)
        weights = [p for g in self.optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(weights, max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.search_phase and self.nas_optimizer is not None:
            self.nas_optimizer.step()
        self.optimizer.zero_grad()
        if self.nas_optimizer is not None:
            self.nas_optimizer.zero_grad(set_to_none=True)
        if self.ema:
            self.ema.update(self.model)
            src = unwrap_model(self.model).net
            dst = self.ema.ema.net
            with torch.no_grad():
                for (_, p), (_, q) in zip(src.named_nas_parameters(), dst.named_nas_parameters()):
                    q.copy_(p)

    # --- checkpoints through state_dict (PLiNIO models cannot be pickled).
    #     In a pruning search the highest-mAP epoch is (almost) always the LEAST pruned one, so
    #     "best by fitness" is meaningless: best.pt always follows the latest epoch (the current
    #     architecture). mAP/cost of every epoch are in results.csv to pick another one by hand.
    def save_model(self):
        ema = self.ema.ema
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "pit_state_dict": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
            "pit_meta": ema.pit_meta,
            "train_args": vars(self.args),
            "train_metrics": {**self.metrics, "fitness": self.fitness},
            "channels": ema.channel_report(),
            "nas_optimizer": self.nas_optimizer.state_dict() if self.nas_optimizer else None,
            "version": ULTRALYTICS_VERSION,
        }
        self.wdir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, self.last)
        torch.save(ckpt, self.best)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            torch.save(ckpt, self.wdir / f"epoch{self.epoch}.pt")
        return True

    def _handle_nan_recovery(self, epoch):
        loss_nan = self.loss is not None and not self.loss.isfinite()
        if loss_nan:
            LOGGER.warning("PIT: NaN/Inf loss detected (no automatic recovery in the search phase)")
        return False

    def final_eval(self):
        """Validate the final search checkpoint (loaded into the EMA model), with plots."""
        if self.last.exists():
            LOGGER.info(f"\nValidating {self.last} (final architecture of the search)...")
            ckpt = torch.load(self.last, map_location="cpu", weights_only=False)
            self.ema.ema.load_state_dict(ckpt["pit_state_dict"])
            self.validator.args.plots = self.args.plots
            self.metrics = self.validator(self)
            self.metrics.pop("fitness", None)
            self.epoch += 1
            self.run_callbacks("on_fit_epoch_end")
            self.epoch -= 1

    @staticmethod
    def _on_train_end(trainer):
        rows = unwrap_model(trainer.ema.ema).channel_report()
        if not rows:
            return
        path = Path(trainer.save_dir) / "channels.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "channels", "kept", "prunable"])
            w.writerows(rows)
        before = sum(r[1] for r in rows if r[3])
        kept = sum(r[2] for r in rows if r[3])
        removed = 100 * (1 - kept / before) if before else 0.0
        LOGGER.info(f"\n{colorstr('PIT:')} prunable channels {before} -> {kept} "
                    f"({removed:.1f}% removed), per-layer table in {path}")
        LOGGER.info(f"{'layer':32s}{'channels':>10s}{'kept':>8s}")
        for name, c, k, prunable in rows:
            if prunable and k != c:
                LOGGER.info(f"{name:32s}{c:10d}{k:8d}")
        try:
            out = plot_pit_results(trainer.save_dir, rows)
            if out:
                LOGGER.info(f"{colorstr('PIT:')} cost / mAP / learning rates plot: {out}")
        except Exception as e:  # plotting must never break a finished search
            LOGGER.warning(f"PIT: pit_results.png not created ({e})")


# ============================================================================ fine-tuning trainer
class PrunedTrainer(DetectionTrainer):
    """DetectionTrainer for an exported (pruned) FxDetectionModel: keeps the pruned architecture
    instead of rebuilding the model from the original yaml. Use: YOLO("pruned.pt").train(...,
    trainer=PrunedTrainer)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, _apply_pit_defaults(overrides, "fine-tuning"), _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if not isinstance(weights, FxDetectionModel):
            raise TypeError("PrunedTrainer needs a pruned model: YOLO('pruned.pt').train(..., "
                            "trainer=PrunedTrainer)")
        return _enable_grads(weights)


# ============================================================================ user-facing API
class PITYOLO:
    """Ultralytics-like entry point for the PIT channel search."""

    def __init__(self, model="yolo26n.pt", n=8, remainder=True, cost="ops", trace_imgsz=320):
        self.model = build_pit_model(model, n=n, remainder=remainder, cost=cost,
                                     trace_imgsz=trace_imgsz)
        self.trainer = None

    # PIT-specific settings (everything else goes to Ultralytics) and their defaults
    PIT_DEFAULTS = dict(pit_warmup_epochs=0, lam=1.0, nas_optimizer="AdamW", nas_lr0=0.01,
                        nas_lrf=1.0, nas_cos_lr=False, nas_weight_decay=0.0, nas_lr_lambda=None)

    def train(self, cfg=None, **kwargs):
        """Run the search.

        cfg: optional YAML file with Ultralytics train arguments AND the PIT settings below, all at
            top level, e.g.
                data: aod4.yaml
                epochs: 100
                lr0: 0.001          # weights (Ultralytics scheduler: lrf, cos_lr)
                lrf: 0.01
                lam: 1.0            # PIT
                nas_lr0: 0.01
                nas_lrf: 0.01
                nas_cos_lr: true
            Keyword arguments override the YAML.

        PIT settings:
          pit_warmup_epochs: epochs with frozen masks (weights only) before the search starts.
          lam: weight of the cost term (cost = fraction of the initial cost, per image).
          nas_optimizer: "AdamW" (default), "Adam" or "SGD" (momentum 0.9).
          nas_lr0, nas_lrf, nas_cos_lr: masks scheduler, same formulas as Ultralytics' lr0/lrf/
              cos_lr, over the SEARCH epochs (epochs - pit_warmup_epochs), stepped per epoch.
              nas_lrf=1.0 (default) -> constant lr. `nas_lr` is accepted as alias of nas_lr0.
          nas_lr_lambda: (Python only, not from YAML) fn(search_epoch, search_epochs) -> factor
              of nas_lr0, replaces linear/cosine.
          nas_weight_decay: masks weight decay (default 0).

        Defaults follow the workflow "model already fine-tuned on the data -> search -> fine-tune":
        weights optimizer SGD (Ultralytics momentum), no LR warmup, no PIT warmup, AMP always off.
        Warnings are printed if a warmup is enabled. Other arguments are standard Ultralytics train
        arguments (imgsz, batch, device, optimizer, lr0, lrf, cos_lr, ...)."""
        import yaml
        conf = {}
        if cfg is not None:
            conf = yaml.safe_load(Path(cfg).read_text()) or {}
            if "names" in conf and "data" not in conf:
                raise ValueError(f"{cfg} looks like a DATASET yaml (it has 'names'): the first "
                                 f"argument of train() is the training config. Use "
                                 f"train(data='{cfg}', ...) or put 'data: {cfg}' in the config")
            if "nas_lr_lambda" in conf:
                raise ValueError("nas_lr_lambda cannot be set from YAML (Python callable only)")
        conf.update(kwargs)
        if "nas_lr" in conf:                              # alias
            conf.setdefault("nas_lr0", conf.pop("nas_lr"))
        pit = {k: conf.pop(k, v) for k, v in self.PIT_DEFAULTS.items()}
        if "data" not in conf:
            raise ValueError("'data' is required (in the YAML or as keyword argument)")
        conf.setdefault("epochs", 100)
        conf.setdefault("project", "pit")   # Ultralytics: runs/detect/pit/<name>
        conf.setdefault("name", "search")
        trainer_cls = type("PITSearchTrainerRun", (PITSearchTrainer,), dict(
            PIT_MODEL=self.model, WARMUP_EPOCHS=pit["pit_warmup_epochs"], LAM=pit["lam"],
            NAS_OPTIMIZER=pit["nas_optimizer"], NAS_LR=pit["nas_lr0"], NAS_LRF=pit["nas_lrf"],
            NAS_COS_LR=bool(pit["nas_cos_lr"]),
            # staticmethod: a plain function stored on the class would be bound (self as arg 1)
            NAS_LR_LAMBDA=(staticmethod(pit["nas_lr_lambda"]) if pit["nas_lr_lambda"] else None),
            NAS_WEIGHT_DECAY=pit["nas_weight_decay"]))
        conf["model"] = self.model.pit_meta["model"]
        self.trainer = trainer_cls(overrides=conf)
        save_dir = Path(self.trainer.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        pit_saved = {k: (v if k != "nas_lr_lambda" else (None if v is None else repr(v)))
                     for k, v in pit.items()}
        (save_dir / "pit_args.yaml").write_text(yaml.safe_dump(pit_saved, sort_keys=False))
        self.trainer.train()
        self.model = unwrap_model(self.trainer.ema.ema)   # final search weights (final_eval)
        return self.trainer.metrics

    def load_search_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["pit_state_dict"])
        return ckpt

    def export_pruned(self, path=None) -> YOLO:
        """Export the searched architecture as a plain (pruned) model, with the trained BN values,
        saved as an Ultralytics-style checkpoint and returned as an ultralytics.YOLO object."""
        wrapper = self.model.float().eval()
        pit = wrapper.net
        exported = pit.export()
        restore_exported_bn(pit, exported, verbose=False)
        detect = next(m for m in exported.modules() if isinstance(m, Detect))
        pruned = FxDetectionModel(exported, detect, wrapper).eval()
        pruned.pit_meta = {**wrapper.pit_meta, "channels": wrapper.channel_report()}
        if path is None:
            base = Path(self.trainer.save_dir) if self.trainer else Path("runs/pit")
            path = base / "weights" / "pruned.pt"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        train_args = vars(self.trainer.args) if self.trainer else {}
        torch.save({"model": None, "ema": copy.deepcopy(pruned).half(), "epoch": -1,
                    "train_args": train_args, "version": ULTRALYTICS_VERSION,
                    "pit_meta": pruned.pit_meta}, path)
        n_before = sum(p.numel() for p in wrapper.parameters()) - sum(
            p.numel() for p in pit.nas_parameters())
        n_after = sum(p.numel() for p in pruned.parameters())
        LOGGER.info(f"{colorstr('PIT:')} pruned model saved to {path} "
                    f"(params {n_before:,} -> {n_after:,})")
        return YOLO(str(path))
