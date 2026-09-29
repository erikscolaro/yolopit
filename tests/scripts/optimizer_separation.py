"""Step-level verification that weights and masks are optimized SEPARATELY, during a real
Ultralytics training (coco8, CPU, N=4).

For every optimizer step it checks:
  - the weights optimizer step leaves every mask parameter bit-identical;
  - the masks optimizer step leaves every weight bit-identical;
  - no parameter is in both optimizers, and neither optimizer's state holds the other's params;
  - gradient clipping (Ultralytics) is applied to weights only;
  - the masks lr stays constant while the weights lr follows Ultralytics' warmup/scheduler.
Configurations: weights optimizer SGD (default), AdamW, auto, MuSGD; plus SGD with LR warmup.
"""
import sys
from pathlib import Path
import os
_OUT_ROOT = Path(os.environ.get('YOLOPIT_TEST_OUT', Path(__file__).resolve().parents[1] / '_out'))


import torch

import yolopit.search as pit_yolo
from yolopit._common import apply_pit_defaults
from yolopit import PITSearchTrainer, PITYOLO
from ultralytics.utils.torch_utils import unwrap_model

OUT = _OUT_ROOT / "runs_opt_sep"
results = []


def check(cond, msg):
    results.append(bool(cond))
    print(("  OK   " if cond else "  FAIL ") + msg, flush=True)


# --- record which tensors gradient clipping touches
_clip_calls = []
_orig_clip = torch.nn.utils.clip_grad_norm_


def _recording_clip(parameters, *a, **k):
    parameters = list(parameters)
    _clip_calls.append({id(p) for p in parameters})
    return _orig_clip(parameters, *a, **k)


torch.nn.utils.clip_grad_norm_ = _recording_clip


class CheckedTrainer(PITSearchTrainer):
    """PITSearchTrainer whose two optimizers are instrumented step by step."""

    def build_optimizer(self, model, *args, **kwargs):
        opt = super().build_optimizer(model, *args, **kwargs)
        net = unwrap_model(model).net
        nas = [p for p in net.nas_parameters()]
        nas_ids = {id(p) for p in nas}
        weights = [p for g in opt.param_groups for p in g["params"]]
        w_ids = {id(p) for p in weights}
        st = self.stats = dict(w_steps=0, n_steps=0, masks_touched_by_w=0,
                               weights_touched_by_n=0, masks_moved=0, weights_moved=0,
                               state_leak=0, nas_lr=set(), w_lr=[], nas_ids=nas_ids,
                               w_ids=w_ids, shared=len(nas_ids & w_ids))
        snap = {}

        def w_pre(optimizer, args, kwargs):
            snap["m"] = [p.detach().clone() for p in nas]
            snap["w"] = [p.detach().clone() for p in weights]

        def w_post(optimizer, args, kwargs):
            st["w_steps"] += 1
            st["masks_touched_by_w"] += sum(not torch.equal(p.detach(), q)
                                            for p, q in zip(nas, snap["m"]))
            st["weights_moved"] += any(not torch.equal(p.detach(), q)
                                       for p, q in zip(weights, snap["w"]))
            st["state_leak"] += sum(id(p) in nas_ids for p in opt.state.keys())
            st["w_lr"].append(opt.param_groups[0]["lr"])

        def n_pre(optimizer, args, kwargs):
            snap["m2"] = [p.detach().clone() for p in nas]
            snap["w2"] = [p.detach().clone() for p in weights]

        def n_post(optimizer, args, kwargs):
            st["n_steps"] += 1
            st["weights_touched_by_n"] += sum(not torch.equal(p.detach(), q)
                                              for p, q in zip(weights, snap["w2"]))
            st["masks_moved"] += any(not torch.equal(p.detach(), q)
                                     for p, q in zip(nas, snap["m2"]))
            st["state_leak"] += sum(id(p) in w_ids for p in self.nas_optimizer.state.keys())
            st["nas_lr"].add(self.nas_optimizer.param_groups[0]["lr"])

        opt.register_step_pre_hook(w_pre)
        opt.register_step_post_hook(w_post)
        self.nas_optimizer.register_step_pre_hook(n_pre)
        self.nas_optimizer.register_step_post_hook(n_post)
        return opt


def run(label, **overrides):
    print(f"\n=== {label} ===", flush=True)
    _clip_calls.clear()
    s = PITYOLO("yolo26n.pt", n=4, cost="ops", trace_imgsz=320)
    cls = type("CheckedRun", (CheckedTrainer,), dict(
        PIT_MODEL=s.model, WARMUP_EPOCHS=1, LAM=2.0, NAS_LR=0.01, NAS_WEIGHT_DECAY=0.0,
        NAS_OPTIMIZER="AdamW"))
    args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=4, imgsz=320, batch=4, workers=0,
                device="cpu", nbs=4, project=str(OUT), name=label, exist_ok=True, plots=False,
                val=True, **overrides)
    tr = cls(overrides=apply_pit_defaults(args, "search"))
    tr.train()
    st = tr.stats
    print(f"  weights optimizer: {type(tr.optimizer).__name__}, masks optimizer: "
          f"{type(tr.nas_optimizer).__name__}; steps: weights {st['w_steps']}, "
          f"masks {st['n_steps']}", flush=True)
    check(st["shared"] == 0, f"no parameter in both optimizers "
                             f"({len(st['w_ids'])} weights, {len(st['nas_ids'])} masks)")
    check(st["masks_touched_by_w"] == 0, "weights optimizer step never changed a mask "
                                         f"({st['w_steps']} steps)")
    check(st["weights_touched_by_n"] == 0, "masks optimizer step never changed a weight "
                                           f"({st['n_steps']} steps)")
    check(st["state_leak"] == 0, "no optimizer state for the other optimizer's parameters")
    check(st["weights_moved"] > 0 and st["masks_moved"] > 0,
          f"both optimizers do update their own parameters "
          f"(weights moved in {st['weights_moved']} steps, masks in {st['masks_moved']})")
    check(st["n_steps"] < st["w_steps"], "masks optimizer idle during the PIT warmup epoch "
                                         f"({st['n_steps']} < {st['w_steps']} steps)")
    clip_leak = sum(len(c & st["nas_ids"]) for c in _clip_calls)
    check(len(_clip_calls) > 0 and clip_leak == 0,
          f"gradient clipping applied to weights only ({len(_clip_calls)} calls)")
    check(st["nas_lr"] == {0.01}, f"masks lr constant: {sorted(st['nas_lr'])}")
    return st


run("SGD_default")
run("AdamW", optimizer="AdamW", lr0=1e-4)
run("auto", optimizer="auto")
run("MuSGD", optimizer="MuSGD", lr0=1e-3)   # what 'auto' picks on long trainings
st = run("SGD_with_LR_warmup", warmup_epochs=2)
w_lr = st["w_lr"]
check(len(set(round(x, 12) for x in w_lr)) > 1,
      f"with LR warmup the weights lr changes ({w_lr[0]:.2e} ... {w_lr[-1]:.2e}) while the "
      f"masks lr stays constant")

print("\nALL CHECKS PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")
