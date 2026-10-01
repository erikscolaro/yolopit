"""PIT (PLiNIO) channel search for Ultralytics YOLO26.

PIT and PLiNIO are by the EML-EDA group, Politecnico di Torino: see README, "Credits".

    from yolopit import PITYOLO, PrunedTrainer

    search = PITYOLO("model_trained_on_your_data.pt", cfg="search.yaml")   # or n=16, ...
    search.train()                                   # or train(data=..., epochs=..., ...)
    pruned = search.export_pruned()                  # ultralytics.YOLO, saved as pruned.pt
    pruned.train(data="your_data.yaml", epochs=30, trainer=PrunedTrainer)

- C3k2 blocks are replaced by an exactly equivalent version without `chunk` (C3k2Split), so PIT
  can prune inside them. C2PSA, PSABlock (attention) and Detect stay fx leaves (not pruned).
- The Ultralytics forward is replaced by a static, fx-traceable loop (YoloFx).
- Layers feeding a leaf keep their OUTPUT channels (inputs still adapt).
- Channels are pruned in blocks of N (yolopit.masks.apply_block_masks, leftover block first).
- Both costs (`ops` = MACs, `params`) are tracked for the WHOLE model: PLiNIO's cost of the PIT
  layers plus the fixed part it cannot see (yolopit.costs).
- The cost term in the loss is `standard` (lambda per cost) or `duccio` (target per cost).
- PITSearchTrainer (DetectionTrainer subclass) adds: optional PIT warmup with frozen masks, a
  SEPARATE optimizer and scheduler for the masks, AMP always off, EMA off by default, state_dict
  checkpoints (PLiNIO models cannot be pickled), a channel report and pit_summary.json.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from . import masks as _masks  # noqa: F401  (must be imported before PIT: fixes nested concats)
from .masks import (PITBlockFeaturesMasker, apply_block_masks, freeze_output_channels,
                    pit_layers_feeding, restore_exported_bn)

from plinio.methods import PIT
from plinio.methods.pit import graph as pit_graph
from plinio.methods.pit.nn import PITConv2d, PITModule
from plinio.methods.pit.nn.features_masker import (PITConcatFeaturesMasker, PITFeaturesMasker,
                                                   PITFrozenFeaturesMasker)
import plinio.graph.inspection as _insp
import plinio.graph.annotation as _annot
from plinio.cost import params as _cost_params, ops as _cost_ops

from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2PSA, C3k2, Detect
from ultralytics.nn.modules.block import PSABlock
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from . import __version__ as YOLOPIT_VERSION
from . import config as _config
from ._common import apply_pit_defaults, enable_grads
from .config import ConfigError
from .costs import COUNTED, count_outside, fixed_costs, fmt
from .plots import plot_pit_results
from .regularizers import DuccioRegularizer, StandardRegularizer
from .runtime import C3k2Split, FxDetectionModel, YoloFx, detach_tracer

LEAVES = (C2PSA, PSABlock, Detect)
COSTS = {"ops": _cost_ops, "params": _cost_params}


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


# ============================================================================ search model
class PITDetectionModel(FxDetectionModel):
    """FxDetectionModel whose `net` is the PIT model: adds the cost term to the loss, the real
    costs of the current architecture (loss items `ops`, `params`, `reg`) and the channel
    report."""

    def __init__(self, net: nn.Module, detect: Detect, src):
        super().__init__(net, detect, src)
        self.regularizer = None     # StandardRegularizer | DuccioRegularizer, set by train()
        self.search_active = False  # set by the trainer: cost term only in the search phase
        self.reg_epoch, self.reg_epochs = 0, 1
        self.cost0 = {}             # PIT part, initial
        self.fixed = {}             # part PIT does not see (constant)
        self.total0 = {}            # whole model, initial
        self.min_total = {}         # whole model, every prunable layer at one block

    @property
    def is_pit(self) -> bool:
        return isinstance(self.net, PIT)

    def loss(self, batch, preds=None):
        loss, items = super().loss(batch, preds)
        if not (self.is_pit and self.total0):
            return loss, items
        items = dict(items)
        reg_value = torch.zeros((), device=loss.device)
        if self.training and self.search_active and self.regularizer is not None:
            bs = batch["img"].shape[0]
            if isinstance(self.regularizer, DuccioRegularizer) and not self.regularizer.ready:
                self.regularizer.start(float(loss.detach().sum()) / bs)   # task loss per image
            reg = self.regularizer(self.net, self.reg_epoch, self.reg_epochs)
            reg = torch.as_tensor(reg).to(loss.device)
            # the criterion returns a VECTOR of components (summed by the trainer) scaled by the
            # batch size: the cost term is appended as one more component, scaled the same way
            loss = torch.cat([loss.reshape(-1), (reg * bs).reshape(1).to(loss.dtype)])
            reg_value = reg.detach().reshape(())
        for m, v in self.real_costs(fraction=True).items():
            items[m] = v.to(loss.device)
        items["reg"] = reg_value
        return loss, items

    @torch.no_grad()
    def real_costs(self, fraction: bool = False) -> dict:
        """Costs of the WHOLE model with the CURRENT architecture (binarized masks, i.e. what the
        export gives): PIT part + fixed part. fraction=True: as a fraction of the initial ones."""
        prev = self.net.discrete_cost
        self.net.discrete_cost = True
        try:
            out = {m: self.net.get_cost(m).detach().float().reshape(()) + self.fixed[m]
                   for m in self.total0}
        finally:
            self.net.discrete_cost = prev
        return {m: v / self.total0[m] for m, v in out.items()} if fraction else out

    # --- reports
    def channel_report(self):
        """[(layer, channels_before, channels_kept, trainable)] for every PIT conv."""
        if not self.is_pit:
            return []
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


def _pit_costs(pit, minimum: bool = False) -> dict:
    """Discrete PIT cost of each metric; minimum=True: with every block mask off (each prunable
    layer keeps only its keep-alive block), the masks are restored afterwards."""
    maskers = {id(l.out_features_masker): l.out_features_masker for l in pit.seed.modules()
               if isinstance(getattr(l, "out_features_masker", None), PITBlockFeaturesMasker)}
    saved = {k: m.block.detach().clone() for k, m in maskers.items()}
    prev = pit.discrete_cost
    pit.discrete_cost = True
    try:
        with torch.no_grad():
            if minimum:
                for m in maskers.values():
                    m.block.zero_()
            return {name: float(pit.get_cost(name)) for name in COSTS}
    finally:
        with torch.no_grad():
            for k, m in maskers.items():
                m.block.copy_(saved[k])
        pit.discrete_cost = prev


def build_pit_model(model="yolo26n.pt", n=8, remainder=True, trace_imgsz=640, verbose=True,
                    cost=None) -> PITDetectionModel:
    """YOLO26 weights/yaml -> PITDetectionModel wrapping a PIT model with block masks, with the
    costs of the whole model (initial, fixed part, minimum) at trace_imgsz. `cost` is ignored
    (kept for old calls: both costs are always tracked)."""
    src = YOLO(model).model.float().eval()
    for p in src.parameters():      # checkpoints load frozen; the trainer would re-enable them
        p.requires_grad_(True)       # anyway, with one warning per parameter
    for i, layer in enumerate(src.model):
        if isinstance(layer, C3k2):
            src.model[i] = C3k2Split(layer).eval()
    detect = src.model[-1]
    fixed, info = fixed_costs(src, LEAVES, detect, trace_imgsz, COSTS)
    outside = count_outside(src, LEAVES)

    net = YoloFx(src.model, src.save).eval()
    detect.export = True                    # single-tensor output while PLiNIO traces the graph
    try:
        pit = PIT(net, cost=dict(COSTS), input_shape=(3, trace_imgsz, trace_imgsz),
                  train_rf=False, train_dilation=False)
    finally:
        detect.export = False
    n_pit = sum(1 for m in pit.seed.modules() if isinstance(m, PITModule) and isinstance(m, COUNTED))
    if n_pit != outside:
        raise RuntimeError(f"{outside} conv/linear layers outside the fx leaves but {n_pit} PIT "
                           f"layers: some layer is neither prunable nor in the fixed part, so the "
                           f"real cost would be wrong")
    feeding = pit_layers_feeding(pit, LEAVES)
    freeze_output_channels(pit, feeding, verbose=verbose)
    apply_block_masks(pit, n, remainder=remainder, verbose=verbose)

    wrapper = PITDetectionModel(pit, detect, src)
    wrapper.cost0 = _pit_costs(pit)
    min_pit = _pit_costs(pit, minimum=True)
    wrapper.fixed = fixed
    wrapper.total0 = {m: wrapper.cost0[m] + fixed[m] for m in COSTS}
    wrapper.min_total = {m: min_pit[m] + fixed[m] for m in COSTS}
    costs = {m: dict(total=wrapper.total0[m], pit=wrapper.cost0[m], fixed=fixed[m],
                     min_total=wrapper.min_total[m], min_pit=min_pit[m]) for m in COSTS}
    wrapper.pit_meta = dict(model=str(model), n=n, remainder=remainder, trace_imgsz=trace_imgsz,
                            yolopit=YOLOPIT_VERSION, costs=costs, **info)
    if verbose:
        LOGGER.info(cost_table(wrapper))
    return wrapper


def cost_table(model: PITDetectionModel) -> str:
    """Initial costs of the whole model, how much PIT can act on, and the minimum."""
    meta = model.pit_meta
    lines = [f"{colorstr('PIT:')} costs of the whole model at {meta['trace_imgsz']}px "
             f"(Detect branch {meta['detect_branch']}, N={meta['n']})",
             f"{'':8s}{'total':>10s}{'prunable':>10s}{'fixed':>10s}{'minimum':>18s}"]
    for m, c in meta["costs"].items():
        lines.append(f"{m:8s}{fmt(c['total'], m):>10s}{fmt(c['pit'], m):>10s}"
                     f"{fmt(c['fixed'], m):>10s}{fmt(c['min_total'], m):>10s} "
                     f"({100 * c['min_total'] / c['total']:.1f}%)")
    return "\n".join(lines)


def resolve_regularizer(model: PITDetectionModel, reg: dict):
    """regularizer group -> (StandardRegularizer | DuccioRegularizer, info for the logs/files).

    duccio: targets of the whole model (absolute or % of the input model) are checked against the
    minimum reachable and converted to the PIT part (target - fixed part)."""
    if reg["mode"] == "standard":
        r = StandardRegularizer(reg["lambda"], model.cost0)
        return r, dict(mode="standard", **{"lambda": dict(reg["lambda"])})
    n = model.pit_meta["n"]
    targets_pit, info = {}, {}
    for m, raw in reg["target"].items():
        kind, value = _config.parse_amount(raw, f"regularizer.target.{m}")
        total, minimum, fixed = model.total0[m], model.min_total[m], model.fixed[m]
        target = value if kind == "abs" else value / 100 * total
        pct = 100 * target / total
        if target < minimum * (1 - 1e-9):
            raise ConfigError(
                f"regularizer.target.{m} = {raw} ({fmt(target, m)}, {pct:.1f}% of the model) is "
                f"impossible: the minimum reachable is {fmt(minimum, m)} "
                f"({100 * minimum / total:.1f}% of the model), with every prunable layer at one "
                f"block of N={n} channels ({fmt(fixed, m)} are in layers that are not pruned). "
                f"Note that at the minimum the accuracy can degrade severely: for that "
                f"architecture it is simpler to prune the network directly and fine-tune it.")
        if target <= minimum * (1 + 1e-9):
            LOGGER.warning(f"PIT: regularizer.target.{m} = {raw} is the minimum reachable: every "
                           f"prunable layer at one block of {n} channels, expect a severe "
                           f"accuracy loss")
        if target >= total:
            LOGGER.warning(f"PIT: regularizer.target.{m} = {raw} ({fmt(target, m)}) is not below "
                           f"the initial cost ({fmt(total, m)}): DUCCIO will not prune for {m}")
        if target < total:
            # a target not below the initial cost does not push (and DUCCIO would divide by
            # cost - target = 0): out of the cost term, still in the budget of best.pt
            targets_pit[m] = target - fixed
        info[m] = dict(requested=raw, target_total=target, target_fraction=target / total,
                       target_pit=target - fixed)
    return DuccioRegularizer(targets_pit), dict(mode="duccio", target=info,
                                                margin=float(reg.get("margin", 0.0)))


def within_budget(costs: dict, reg_info: dict | None) -> bool | None:
    """Every cost of the whole model within target * (1 + margin)? None without targets
    (mode standard)."""
    targets = (reg_info or {}).get("target")
    if not targets:
        return None
    margin = float(reg_info.get("margin", 0.0))
    return all(float(costs[m]) <= t["target_total"] * (1 + margin) * (1 + 1e-6)
               for m, t in targets.items())


def nas_lr_factor(e: int, total: int, lrf: float = 1.0, cos: bool = False, fn=None) -> float:
    """Multiplier of nas.lr0 at search epoch e (0-based) of `total` search epochs.

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


# ============================================================================ search trainer
class PITSearchTrainer(DetectionTrainer):
    """DetectionTrainer running the PIT search on a PITDetectionModel.

    Extra settings are class attributes (Ultralytics validates the `overrides` keys):
      PIT_MODEL, WARMUP_EPOCHS (PIT warmup: masks frozen), EMA, NAS_OPTIMIZER, NAS_LR (lr0 of
      the masks), NAS_LRF, NAS_COS_LR, NAS_LR_LAMBDA, NAS_WEIGHT_DECAY. The cost term is the
      model's `regularizer`.

    Masks vs weights:
      - the masks have their OWN optimizer and their OWN scheduler (nas_lr_factor: linear or
        cosine from NAS_LR to NAS_LR*NAS_LRF over the SEARCH epochs, stepped per epoch like
        Ultralytics; NAS_LRF=1 -> constant), completely separate from the model's: Ultralytics'
        LR warmup, momentum warmup, LR scheduler and gradient clipping act only on the weights;
      - the mask optimizer steps only in the search phase (epoch >= WARMUP_EPOCHS);
      - AMP is always disabled (mask gradients need full precision);
      - EMA=False (default): the EMA model is kept, since Ultralytics validates and saves it, but
        as an exact copy of the current weights (decay 0)."""

    PIT_MODEL: PITDetectionModel | None = None
    WARMUP_EPOCHS = 0
    EMA = False
    NAS_LR = 0.01               # lr0 of the masks
    NAS_LRF = 1.0               # final lr = NAS_LR * NAS_LRF (1.0 -> constant)
    NAS_COS_LR = False          # cosine instead of linear
    NAS_LR_LAMBDA = None        # Python only: fn(search_epoch, search_epochs) -> factor
    NAS_WEIGHT_DECAY = 0.0
    NAS_OPTIMIZER = "AdamW"     # "AdamW", "Adam" or "SGD"

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, apply_pit_defaults(overrides, "search"), _callbacks)
        if self.WARMUP_EPOCHS > 0:
            LOGGER.warning(f"PIT (search): PIT warmup enabled (masks frozen for the first "
                           f"{self.WARMUP_EPOCHS} epochs) but the input model is already trained "
                           f"on the data: normally not needed")
        self.nas_optimizer = None
        self.search_phase = False
        self.best_within_fitness = None   # best fitness among the epochs within the budget
        self.best_epoch = None            # epoch saved in best.pt (0-based)
        self.add_callback("on_train_epoch_start", PITSearchTrainer._on_epoch_start)
        self.add_callback("on_train_epoch_end", PITSearchTrainer._on_epoch_end)
        self.add_callback("on_train_end", PITSearchTrainer._on_train_end)

    # --- model
    def get_model(self, cfg=None, weights=None, verbose=True):
        return self.PIT_MODEL

    def setup_model(self):
        # PIT warmup / search phase then sets the masks' requires_grad at every epoch start
        self.model = enable_grads(self.PIT_MODEL)
        return None

    def _setup_train(self):
        super()._setup_train()
        if not self.EMA and self.ema is not None:
            self.ema.decay = lambda updates: 0.0
            LOGGER.info(f"{colorstr('PIT:')} EMA off during the search (the validated and saved "
                        f"model is the current one, not an average of past architectures)")

    # --- phases
    @staticmethod
    def _on_epoch_start(trainer):
        model = unwrap_model(trainer.model)
        trainer.search_phase = trainer.epoch >= trainer.WARMUP_EPOCHS
        model.net.train_features = trainer.search_phase
        model.search_active = trainer.search_phase
        model.reg_epoch = max(trainer.epoch - trainer.WARMUP_EPOCHS, 0)
        model.reg_epochs = max(trainer.epochs - trainer.WARMUP_EPOCHS, 1)
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
        sched = ("custom (lr_lambda)" if self.NAS_LR_LAMBDA is not None else
                 "constant" if self.NAS_LRF == 1.0 else
                 f"{'cosine' if self.NAS_COS_LR else 'linear'} {self.NAS_LR} -> "
                 f"{self.NAS_LR * self.NAS_LRF:g}")
        LOGGER.info(f"{colorstr('PIT:')} masks optimizer {type(self.nas_optimizer).__name__}"
                    f"(lr0={self.NAS_LR}, weight_decay={self.NAS_WEIGHT_DECAY}), scheduler "
                    f"{sched} over the search epochs, {len(nas)} mask parameters (separate from "
                    f"the weights: no Ultralytics LR warmup/scheduler/grad clipping)")
        reg = unwrap_model(model).regularizer
        if reg is not None:
            LOGGER.info(f"{colorstr('PIT:')} cost term: {reg.describe()}")
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
    #     "best by fitness" alone is meaningless. With DUCCIO targets, best.pt is the highest
    #     fitness among the epochs with every cost within target * (1 + margin), and the latest
    #     epoch until one is; without targets (mode standard) it is always the latest epoch.
    def _update_best(self, ok: bool | None) -> bool:
        """Whether this epoch goes to best.pt; `ok`: costs within the budget (None: no targets)."""
        if ok is None:
            self.best_epoch = self.epoch
            return True
        if not ok:
            if self.best_within_fitness is None:     # nothing within the budget yet: latest
                self.best_epoch = self.epoch
                return True
            return False
        # without validation (val=False) there is no fitness: the latest epoch within the budget
        fitness = float("-inf") if self.fitness is None else float(self.fitness)
        if (self.fitness is None or self.best_within_fitness is None
                or fitness > self.best_within_fitness):
            self.best_within_fitness, self.best_epoch = fitness, self.epoch
            return True
        return False

    def save_model(self):
        ema = self.ema.ema
        model = unwrap_model(self.model)
        reg = model.regularizer   # the EMA copy holds a copy that never runs
        costs = {m: float(v) for m, v in ema.real_costs().items()}
        ok = within_budget(costs, getattr(model, "regularizer_info", None))
        is_best = self._update_best(ok)
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "pit_state_dict": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
            "pit_meta": ema.pit_meta,
            "train_args": vars(self.args),
            "train_metrics": {**self.metrics, "fitness": self.fitness},
            "channels": ema.channel_report(),
            "costs": costs,
            "within_budget": ok,
            "best_epoch": self.best_epoch,
            "best_within_fitness": self.best_within_fitness,
            "duccio_strengths": reg.strengths() if isinstance(reg, DuccioRegularizer) else None,
            "nas_optimizer": self.nas_optimizer.state_dict() if self.nas_optimizer else None,
            "version": ULTRALYTICS_VERSION,
            "yolopit": YOLOPIT_VERSION,
        }
        self.wdir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, self.last)
        if is_best:
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
        """Validate best.pt (loaded into the EMA model, so it is also what gets exported), with
        plots."""
        path = self.best if self.best.exists() else self.last
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            ok = ckpt.get("within_budget")
            why = ("highest fitness within the budget" if ok else
                   "latest epoch, no target" if ok is None else
                   "latest epoch, NO epoch within the budget")
            LOGGER.info(f"\nValidating {path} (epoch {ckpt['epoch'] + 1}: {why})...")
            if ok is False:
                LOGGER.warning("PIT: no epoch had every cost within target * (1 + margin): "
                               "best.pt is the latest epoch")
            self.ema.ema.load_state_dict(ckpt["pit_state_dict"])
            self.validator.args.plots = self.args.plots
            self.metrics = self.validator(self)
            self.metrics.pop("fitness", None)
            self.epoch += 1
            self.run_callbacks("on_fit_epoch_end")
            self.epoch -= 1

    @staticmethod
    def _on_train_end(trainer):
        model = unwrap_model(trainer.ema.ema)
        rows = model.channel_report()
        if not rows:
            return
        save_dir = Path(trainer.save_dir)
        path = save_dir / "channels.csv"
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
        summary = search_summary(model, trainer,
                                 regularizer=unwrap_model(trainer.model).regularizer)
        (save_dir / "pit_summary.json").write_text(json.dumps(summary, indent=1))
        LOGGER.info(f"{colorstr('PIT:')} whole model, initial -> final (minimum reachable):")
        for m, c in summary["costs"].items():
            line = (f"  {m:7s}{fmt(c['initial'], m):>10s} -> {fmt(c['final'], m):>9s} "
                    f"({100 * c['final_fraction']:.1f}%)   min {fmt(c['min'], m)}")
            if "target" in c:
                line += (f"   target {fmt(c['target'], m)}: "
                         f"{'met' if c['target_met'] else 'NOT met'}"
                         f"{'' if c['target_met'] or not c['within_margin'] else ' (within margin)'}")
            LOGGER.info(line)
        if summary.get("best_epoch") is not None:
            LOGGER.info(f"{colorstr('PIT:')} best.pt = epoch {summary['best_epoch']} of "
                        f"{summary['epochs']}, exported as the pruned model")
        try:
            out = plot_pit_results(trainer.save_dir, rows)
            if out:
                LOGGER.info(f"{colorstr('PIT:')} cost / mAP / learning rates plot: {out}")
        except Exception as e:  # plotting must never break a finished search
            LOGGER.warning(f"PIT: pit_results.png not created ({e})")


def search_summary(model: PITDetectionModel, trainer=None, regularizer=None) -> dict:
    """pit_summary.json: costs initial/final/minimum (and targets), channels, versions."""
    final = {m: float(v) for m, v in model.real_costs().items()}
    reg_info = getattr(model, "regularizer_info", {}) or {}
    costs = {}
    for m in final:
        c = dict(initial=model.total0[m], final=final[m], final_fraction=final[m] / model.total0[m],
                 min=model.min_total[m], fixed=model.fixed[m])
        t = (reg_info.get("target") or {}).get(m)
        if t:
            c.update(target=t["target_total"], target_met=final[m] <= t["target_total"] * (1 + 1e-6),
                     within_margin=final[m] <= t["target_total"] * (1 + reg_info.get("margin", 0.0))
                     * (1 + 1e-6))
        costs[m] = c
    rows = model.channel_report()
    out = dict(yolopit=YOLOPIT_VERSION, ultralytics=ULTRALYTICS_VERSION,
               model=model.pit_meta.get("model"), n=model.pit_meta.get("n"),
               remainder=model.pit_meta.get("remainder"), imgsz=model.pit_meta.get("trace_imgsz"),
               detect_branch=model.pit_meta.get("detect_branch"), regularizer=reg_info,
               costs=costs,
               prunable_channels=dict(before=sum(r[1] for r in rows if r[3]),
                                      kept=sum(r[2] for r in rows if r[3])))
    regularizer = regularizer if regularizer is not None else model.regularizer
    if isinstance(regularizer, DuccioRegularizer):
        out["duccio_strengths"] = regularizer.strengths()
    if trainer is not None:
        out["epochs"] = trainer.epochs
        out["best_epoch"] = None if trainer.best_epoch is None else trainer.best_epoch + 1
        out["best_within_budget"] = within_budget(final, reg_info)
        out["metrics"] = {k: float(v) for k, v in (trainer.metrics or {}).items()
                          if isinstance(v, (int, float))}
    return out


# ============================================================================ user-facing API
class PITYOLO:
    """Ultralytics-like entry point for the PIT channel search.

    PITYOLO(model, cfg=None, n=None, remainder=None, trace_imgsz=None)
      model: YOLO26 weights already trained on your data (or a yaml).
      cfg: search config (YAML path or dict, see yolopit.config): Ultralytics arguments plus the
           groups `pit`, `nas`, `regularizer`. The `pit` group (n, remainder, trace_imgsz) fixes
           the search space, so it is used here; explicit arguments override it.
      trace_imgsz: image size at which the costs are computed; default: pit.trace_imgsz, else
           the config's imgsz, else 640. Absolute targets refer to this size.
    """

    def __init__(self, model="yolo26n.pt", cfg=None, n=None, remainder=None, trace_imgsz=None,
                 cost=None):
        self._legacy_cost = cost or "ops"   # only for the old flat key `lam`
        self._conf = _config.load(cfg)
        ultra, groups = _config.split(self._conf, self._legacy_cost)
        pit = groups["pit"]
        for k, v in dict(n=n, remainder=remainder, trace_imgsz=trace_imgsz).items():
            if v is not None:
                pit[k] = v
        if pit["trace_imgsz"] is None:
            pit["trace_imgsz"] = int(ultra.get("imgsz", 640))
        self.arch = {k: pit[k] for k in ("n", "remainder", "trace_imgsz")}
        self.model = build_pit_model(model, n=int(pit["n"]), remainder=bool(pit["remainder"]),
                                     trace_imgsz=int(pit["trace_imgsz"]))
        self.trainer = None

    def train(self, cfg=None, **kwargs):
        """Run the search.

        cfg: search config (YAML path or dict); default: the one given to PITYOLO(). Keyword
        arguments override it: Ultralytics arguments, whole groups (nas={"lr0": 0.05}) or the old
        flat keys (lam, nas_lr0, ...). nas.lr_lambda (fn(search_epoch, search_epochs) -> factor
        of nas.lr0) can only be given from Python.

        Defaults follow the workflow "model already trained on the data -> search -> fine-tune":
        weights optimizer SGD, no LR warmup, no PIT warmup, AMP off, EMA off, masks AdamW with
        constant lr, cost term standard with lambda {ops: 1.0}."""
        import yaml
        base = _config.load(cfg) if cfg is not None else copy.deepcopy(self._conf)
        conf = _config.merge(_config.normalize(base, self._legacy_cost),
                             _config.normalize(kwargs, self._legacy_cost))
        ultra, groups = _config.split(conf, self._legacy_cost)
        given_pit = conf.get("pit") or {}
        for k, built in self.arch.items():
            if given_pit.get(k) is not None and given_pit[k] != built:
                raise ConfigError(f"pit.{k} = {given_pit[k]} but the model was built with "
                                  f"{built}: pit.{k} is fixed when the model is built, pass it to "
                                  f"PITYOLO(...)")
        if "data" not in ultra:
            raise ConfigError("'data' is required (in the config or as keyword argument)")
        imgsz = ultra.get("imgsz")
        if imgsz is not None and int(imgsz) != int(self.arch["trace_imgsz"]):
            LOGGER.warning(f"PIT: training at imgsz={imgsz} but costs computed at "
                           f"{self.arch['trace_imgsz']}px (pit.trace_imgsz): absolute costs and "
                           f"targets refer to {self.arch['trace_imgsz']}px")
        ultra.setdefault("epochs", 100)
        ultra.setdefault("project", "pit")   # Ultralytics: runs/detect/pit/<name>
        ultra.setdefault("name", "search")

        regularizer, reg_info = resolve_regularizer(self.model, groups["regularizer"])
        self.model.regularizer, self.model.regularizer_info = regularizer, reg_info
        nas, pit = groups["nas"], groups["pit"]
        lr_lambda = nas["lr_lambda"]
        trainer_cls = type("PITSearchTrainerRun", (PITSearchTrainer,), dict(
            PIT_MODEL=self.model, WARMUP_EPOCHS=int(pit["warmup_epochs"]), EMA=bool(pit["ema"]),
            NAS_OPTIMIZER=nas["optimizer"], NAS_LR=float(nas["lr0"]), NAS_LRF=float(nas["lrf"]),
            NAS_COS_LR=bool(nas["cos_lr"]),
            # staticmethod: a plain function stored on the class would be bound (self as arg 1)
            NAS_LR_LAMBDA=staticmethod(lr_lambda) if lr_lambda else None,
            NAS_WEIGHT_DECAY=float(nas["weight_decay"])))
        ultra["model"] = self.model.pit_meta["model"]
        self.trainer = trainer_cls(overrides=ultra)
        save_dir = Path(self.trainer.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        saved = dict(yolopit=YOLOPIT_VERSION, pit={**pit, **self.arch},
                     nas={**nas, "lr_lambda": None if lr_lambda is None else repr(lr_lambda)},
                     regularizer=groups["regularizer"], resolved=reg_info,
                     costs=self.model.pit_meta["costs"],
                     detect_branch=self.model.pit_meta["detect_branch"])
        (save_dir / "pit_args.yaml").write_text(yaml.safe_dump(_plain(saved), sort_keys=False))
        self.trainer.train()
        self.model = unwrap_model(self.trainer.ema.ema)   # best.pt, loaded by final_eval
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
        exported = detach_tracer(pit.export())   # pruned.pt must not need PLiNIO to load
        restore_exported_bn(pit, exported, verbose=False)
        detect = next(m for m in exported.modules() if isinstance(m, Detect))
        pruned = FxDetectionModel(exported, detect, wrapper).eval()
        pruned.pit_meta = {**wrapper.pit_meta, "channels": wrapper.channel_report(),
                           "final_costs": {m: float(v) for m, v in wrapper.real_costs().items()}}
        if path is None:
            base = Path(self.trainer.save_dir) if self.trainer else Path("runs/pit")
            path = base / "weights" / "pruned.pt"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        train_args = vars(self.trainer.args) if self.trainer else {}
        # fp32, not half like Ultralytics' own checkpoints: the pruned model must give exactly
        # the results of the end of the search (fp16 rounding moves boxes by up to ~1 px)
        torch.save({"model": None, "ema": copy.deepcopy(pruned).float(), "epoch": -1,
                    "train_args": train_args, "version": ULTRALYTICS_VERSION,
                    "yolopit": YOLOPIT_VERSION, "pit_meta": pruned.pit_meta}, path)
        n_before = sum(p.numel() for p in wrapper.parameters()) - sum(
            p.numel() for p in pit.nas_parameters())
        n_after = sum(p.numel() for p in pruned.parameters())
        LOGGER.info(f"{colorstr('PIT:')} pruned model saved to {path} "
                    f"(params {n_before:,} -> {n_after:,})")
        return YOLO(str(path))


def _plain(x):
    """YAML-safe copy (tuples -> lists, numpy/torch scalars -> float)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (str, bool, int, float)) or x is None:
        return x
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)
