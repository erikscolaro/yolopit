"""Block-granular channel masks and utilities for PLiNIO PIT (requires PLiNIO).

- apply_block_masks(pit, N): channels are switched on/off in blocks of N (leftover block, if any,
  is always pruned first), so searched channel counts are multiples of N.
- restore_exported_bn(pit, exported): exported BNs get the trained values (PLiNIO resets them).
- pit_layers_feeding(pit, leaf_types) + freeze_output_channels(pit, names): freeze only the
  OUTPUT channels of layers feeding fx-leaf blocks.
- Importing this module also fixes a PLiNIO bug with nested concatenations (concat of a concat).

How the block masks work: every trainable PITFeaturesMasker is REPLACED (everywhere it is used,
so mask sharing is preserved) by a PITBlockFeaturesMasker, whose only trainable tensor is one
value per block; its `alpha` is recomputed from the blocks at every access. It is a regular
module class, so models using it can be pickled (torch.save of the whole model, as the
Ultralytics trainer does), deep-copied and moved across devices.

Call apply_block_masks right after PIT(...) and BEFORE creating the optimizer.
"""
import warnings

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from plinio.methods.pit.nn.features_masker import (
    PITFeaturesMasker, PITFrozenFeaturesMasker, PITConcatFeaturesMasker)


def block_sizes(c: int, n: int, remainder: bool):
    """Block sizes for a layer with c channels: N-sized blocks, plus (if remainder=True) a final
    block with the c % N leftover channels. None if the layer cannot be blocked."""
    if c < n:
        return None
    full, rest = divmod(c, n)
    if rest == 0:
        return [n] * full
    if not remainder:
        return None
    return [n] * full + [rest]


class PITBlockFeaturesMasker(PITFeaturesMasker):
    """PIT output-features masker with one trainable value per block of channels.

    sizes: block sizes in channel order, e.g. [128, 64] for C=192, N=128.
    If the last block is a leftover (< N), it is ALWAYS THE FIRST TO BE PRUNED: its effective
    value is min(|own value|, |value of every prunable full block|), so switching off any full
    block switches it off too. Keep-alive (never switched off) is the last FULL block.

    PLiNIO only reads `theta` (inherited: |alpha| with keep-alive forced to 1) and `trainable`,
    so everything downstream (binarization, cost, export) is unchanged."""

    def __init__(self, sizes, n: int, init_alpha: torch.Tensor = None, trainable: bool = True):
        nn.Module.__init__(self)          # NOT PITFeaturesMasker.__init__: no per-channel alpha
        self.sizes = [int(s) for s in sizes]
        self.n = int(n)
        self.out_channels = sum(self.sizes)
        self.ordered_remainder = self.sizes[-1] != self.n
        if init_alpha is None:
            init = torch.ones(len(self.sizes))
        else:
            init = torch.stack([p.mean() for p in init_alpha.detach().float().cpu().split(self.sizes)])
        self.block = Parameter(init, requires_grad=trainable)
        self.register_buffer("_repeats", torch.tensor(self.sizes, dtype=torch.long),
                             persistent=False)
        n_full = len(self.sizes) - (1 if self.ordered_remainder else 0)
        ka = torch.zeros(self.out_channels)
        ka[(n_full - 1) * self.n: n_full * self.n] = 1.0
        self.register_buffer("_keep_alive", ka)

    @property
    def alpha(self) -> torch.Tensor:
        block = self.block
        if self.ordered_remainder:
            full, rem = block[:-1].abs(), block[-1].abs()
            guards = full[:-1]                  # prunable full blocks (last full = keep-alive)
            if guards.numel() > 0:
                rem = torch.minimum(rem, guards.min())
            block = torch.cat([full, rem.reshape(1)])
        return block.repeat_interleave(self._repeats.to(block.device))

    @property
    def trainable(self) -> bool:
        return self.block.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        self.block.requires_grad = value

    # kept for backward compatibility with earlier versions / inspection
    @property
    def _block_sizes(self):
        return self.sizes


def _alpha_leaf(m: nn.Module) -> torch.Tensor:
    """The leaf tensor holding the trainable values of a masker."""
    return m.block if isinstance(m, PITBlockFeaturesMasker) else m.alpha


def _leaf_maskers(masker):
    """Real (non-concat) maskers behind a masker; concat maskers can be NESTED
    (concat of a concat, common in YOLO necks)."""
    if isinstance(masker, PITConcatFeaturesMasker):
        return [m for sub in masker.mask_list for m in _leaf_maskers(sub)]
    return [masker]


# --- fix for PLiNIO's concat masker `trainable` (nested concats + block maskers) ---------
def _concat_get_trainable(self) -> bool:
    return any(_alpha_leaf(m).requires_grad for m in _leaf_maskers(self)
               if not isinstance(m, PITFrozenFeaturesMasker))


def _concat_set_trainable(self, value: bool):
    # the original setter does `m.alpha.requires_grad = ...` on every element of mask_list:
    # it crashes when an element is itself a concat masker (upstream bug), and cannot work on
    # computed alphas (block maskers)
    for m in _leaf_maskers(self):
        if not isinstance(m, PITFrozenFeaturesMasker):
            _alpha_leaf(m).requires_grad = value


PITConcatFeaturesMasker.trainable = property(_concat_get_trainable, _concat_set_trainable)


def _swap_maskers(pit_model: nn.Module, old_to_new: dict):
    """Replace maskers everywhere they are used (layers and, recursively, concat maskers)."""
    def swap(masker):
        if isinstance(masker, PITConcatFeaturesMasker):
            masker.mask_list = [swap(m) for m in masker.mask_list]
            return masker
        return old_to_new.get(id(masker), masker)

    for mod in pit_model.modules():
        if hasattr(mod, "out_features_masker"):
            mod.out_features_masker = swap(mod.out_features_masker)


# --- main entry point ----------------------------------------------------------------
@torch.no_grad()
def apply_block_masks(pit_model: nn.Module, n: int, remainder: bool = True,
                      verbose: bool = True):
    """Convert the trainable PITFeaturesMaskers of `pit_model` to block masks of N channels.

    remainder=True (DEFAULT): a layer with C not divisible by N gets floor(C/N) blocks of N plus a
                     final leftover block of C % N channels (e.g. 192, N=128 -> [128, 64];
                     33, N=32 -> [32, 1]). The leftover block is ALWAYS THE FIRST TO BE PRUNED:
                     it can be switched off on its own, and it is switched off automatically as
                     soon as any full block is. So the result is always a multiple of N, unless
                     nothing is pruned in that layer (192 -> 192 or 128, never 64).
    remainder=False: (emits a warning) only layers with C divisible by N are converted; the others
                     keep PIT's per-channel masks and can end with any channel count.

    Layers with C < N are NOT pruned: their masker is replaced by a frozen one.
    Keep-alive (the block that can never be switched off) is the last FULL block of N.
    Full blocks are switched on/off independently of each other.

    Returns (converted, skipped, frozen) lists of (masker, n_channels):
      converted = new block maskers, skipped = left per-channel (remainder=False only),
      frozen = C < N, not pruned."""
    if not remainder:
        warnings.warn("apply_block_masks(remainder=False): layers whose channel count is not a "
                      "multiple of N keep per-channel PIT masks and may end with channel counts "
                      "that are not SIMD-aligned.", stacklevel=2)
    seen, converted, skipped, frozen, old_to_new = set(), [], [], [], {}
    for mod in pit_model.modules():
        masker = getattr(mod, "out_features_masker", None)
        if masker is None:
            continue
        for m in _leaf_maskers(masker):
            if id(m) in seen:            # shared masker: convert only once
                continue
            seen.add(id(m))
            if isinstance(m, (PITFrozenFeaturesMasker, PITBlockFeaturesMasker)):
                continue                 # never pruned / already converted
            c = m.out_channels
            dev = m.alpha.device
            if c < n:
                new = PITFrozenFeaturesMasker(c).to(dev)
                old_to_new[id(m)] = new
                frozen.append((new, c))
                continue
            sizes = block_sizes(c, n, remainder)
            if sizes is None:
                skipped.append((m, c))
                continue
            new = PITBlockFeaturesMasker(sizes, n, init_alpha=m.alpha,
                                         trainable=m.alpha.requires_grad).to(dev)
            old_to_new[id(m)] = new
            converted.append((new, c))
    _swap_maskers(pit_model, old_to_new)
    if verbose:
        desc = lambda m, c: (f"{c}={'+'.join(map(str, m.sizes))}"
                             if len(set(m.sizes)) > 1 else str(c))
        print(f"[block masks] N={n}{' (+remainder)' if remainder else ''}: converted "
              f"{len(converted)} maskers [{', '.join(desc(m, c) for m, c in converted)}]; "
              f"not pruned (C<N): {[c for _, c in frozen]}"
              + (f"; per-channel (not multiple of N): {[c for _, c in skipped]}" if skipped else ""))
    return converted, skipped, frozen


@torch.no_grad()
def blocks_to_plain(pit_model: nn.Module) -> int:
    """Replace every block masker with a plain per-channel PITFeaturesMasker holding the SAME
    effective values (for checks/debugging: the model must behave identically)."""
    old_to_new = {}
    for mod in pit_model.modules():
        masker = getattr(mod, "out_features_masker", None)
        if masker is None:
            continue
        for m in _leaf_maskers(masker):
            if isinstance(m, PITBlockFeaturesMasker) and id(m) not in old_to_new:
                new = PITFeaturesMasker(m.out_channels, trainable=m.trainable).to(m.block.device)
                new.alpha.data.copy_(m.alpha)
                new._keep_alive.copy_(m._keep_alive)
                old_to_new[id(m)] = new
    _swap_maskers(pit_model, old_to_new)
    return len(old_to_new)


# --- BN fix for export -----------------------------------------------------------------
@torch.no_grad()
def restore_exported_bn(pit_model: nn.Module, exported: nn.Module, verbose: bool = True) -> int:
    """PLiNIO's PIT export (fold_bn=False) re-creates each BatchNorm after a pruned conv with
    DEFAULT parameters (weight=1, bias=0, running_mean=0, running_var=1), discarding the trained
    ones. This copies the trained BN parameters/statistics of the kept channels into the exported
    BNs, so that the exported model reproduces the PIT model exactly (masks are applied after
    the BN in PIT, so pruned channels are exactly zero there).

    Call it right after `exported = pit_model.export()`. Returns the number of BNs restored."""
    from plinio.methods.pit.nn import PITConv1d, PITConv2d, PITConv3d, PITLinear
    exported_mods = dict(exported.named_modules())
    n = 0
    for name, layer in pit_model.seed.named_modules():
        if not isinstance(layer, (PITConv1d, PITConv2d, PITConv3d, PITLinear)):
            continue
        src = getattr(layer, "bn", None)
        dst = exported_mods.get(name + "_exported_bn")
        if src is None or dst is None or getattr(layer, "fold_bn", False):
            continue
        keep = layer.features_mask.bool()
        for attr in ("weight", "bias", "running_mean", "running_var"):
            s, d = getattr(src, attr, None), getattr(dst, attr, None)
            if s is not None and d is not None:
                d.copy_(s[keep])
        if getattr(src, "num_batches_tracked", None) is not None:
            dst.num_batches_tracked.copy_(src.num_batches_tracked)
        n += 1
    if verbose:
        print(f"[restore_exported_bn] restored {n} BatchNorm layers")
    return n


# --- freeze only the OUTPUT channels of some PIT layers ---------------------------------
@torch.no_grad()
def freeze_output_channels(pit_model: nn.Module, layer_names, verbose: bool = True) -> int:
    """Keep the given PIT layers (qualified names inside pit_model.seed) NAS-able, but with their
    OUTPUT channels fixed. Unlike `exclude_names` (which leaves a plain nn.Conv2d whose INPUT
    channels can no longer follow a pruned predecessor), the layer still adapts its input channels
    at export. Use it for layers whose output feeds something PIT cannot resize (e.g. a leaf block
    such as C3k2 / C2PSA / Detect treated as an fx leaf).

    The frozen masker replaces the old one everywhere it was shared (including inside concat
    maskers). Call it right after PIT(...), before apply_block_masks. Returns the number of
    maskers frozen."""
    from plinio.methods.pit.nn import PITConv1d, PITConv2d, PITConv3d, PITLinear
    seed_mods = dict(pit_model.seed.named_modules())
    old_to_new = {}
    for name in layer_names:
        layer = seed_mods.get(name)
        if layer is None:
            raise KeyError(f"{name} is not a module of pit_model.seed")
        if not isinstance(layer, (PITConv1d, PITConv2d, PITConv3d, PITLinear)):
            raise TypeError(f"{name} is {type(layer).__name__}, not a PIT layer")
        old = layer.out_features_masker
        if isinstance(old, (PITFrozenFeaturesMasker, PITConcatFeaturesMasker)):
            continue
        if id(old) not in old_to_new:
            new = PITFrozenFeaturesMasker(old.out_channels)
            new.to(old.alpha.device)
            old_to_new[id(old)] = new

    _swap_maskers(pit_model, old_to_new)
    if verbose:
        print(f"[freeze_output_channels] froze {len(old_to_new)} maskers "
              f"({len(list(layer_names))} layers requested)")
    return len(old_to_new)


# --- automatic freezing of everything that feeds a leaf block ---------------------------
def pit_layers_feeding(pit_model: nn.Module, leaf_types) -> list:
    """Names of the PIT layers whose output channels reach (through activations, BN, add, cat,
    upsample, pooling...) the input of a module of one of `leaf_types` (fx leaves that PIT cannot
    resize). Their OUTPUT channels must be frozen with `freeze_output_channels`."""
    from plinio.methods.pit.nn import PITConv1d, PITConv2d, PITConv3d, PITLinear
    pit_types = (PITConv1d, PITConv2d, PITConv3d, PITLinear)
    seed = pit_model.seed
    mods = dict(seed.named_modules())
    found = set()
    for node in seed.graph.nodes:
        if node.op != "call_module" or not isinstance(mods[str(node.target)], tuple(leaf_types)):
            continue
        stack, seen = list(node.all_input_nodes), set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n.op == "call_module":
                m = mods[str(n.target)]
                if isinstance(m, pit_types):
                    found.add(str(n.target))
                    continue                      # its input channels can adapt: stop here
                if isinstance(m, tuple(leaf_types)):
                    continue                      # fixed output: stop here
            if n.op == "placeholder":
                continue
            stack.extend(n.all_input_nodes)
    return sorted(found)
