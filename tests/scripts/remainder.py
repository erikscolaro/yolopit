"""Validation of apply_block_masks(..., remainder=True): the leftover block is pruned first.

Layers with 33, 72, 40 channels and N=32 -> blocks [32,1], [32,32,8], [32,8]
Layer with 192 channels and N=128   -> blocks [128,64]

Checks:
 1. remainder=False skips the non-divisible layers
 2. exhaustive: for EVERY on/off combination of the block values, the kept channel count is
    either C (nothing pruned) or a multiple of N; the leftover block is never kept alone
    without all full blocks; the keep-alive (last full) block is never off
 3. gradients reach every block value (including through the min() constraint)
 4. warmup/search toggling + short PIT training
 5. export + BN restore == PIT output; block masks == equivalent plain per-channel masks
"""
import sys
from pathlib import Path
import os
_OUT_ROOT = Path(os.environ.get('YOLOPIT_TEST_OUT', Path(__file__).resolve().parents[1] / '_out'))
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

import yolopit.masks  # noqa: F401  (import before PIT: activates the fixes)
from yolopit.masks import apply_block_masks, restore_exported_bn, blocks_to_plain
from plinio.methods import PIT
from plinio.methods.pit.nn import PITConv2d
from plinio.cost import params

torch.manual_seed(0)
IN_SHAPE = (3, 16, 16)


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    assert cond, msg


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU())


class OddCNN(nn.Module):
    def __init__(self, chans):
        super().__init__()
        c = [3] + list(chans)
        self.blocks = nn.Sequential(*[block(c[i], c[i + 1]) for i in range(len(chans))])
        self.pool, self.flat, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c[-1], 10)

    def forward(self, x):
        return self.fc(self.flat(self.pool(self.blocks(x))))


X = torch.randn(1024, *IN_SHAPE)
Y = torch.randint(0, 10, (1024,))


def randomize_bn(model):
    model.train()
    with torch.no_grad():
        for _ in range(10):
            model(X[:256])
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.bias.normal_(0, .5); m.weight.uniform_(.5, 1.5)
    model.eval()


def run(chans, n):
    print(f"\n=== channels {chans}, N={n}, remainder=True ===")
    model = OddCNN(chans); randomize_bn(model)

    # --- 1: remainder=False skips them
    pit0 = PIT(OddCNN(chans), cost=params, input_shape=IN_SHAPE, train_rf=False, train_dilation=False)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, sk, _ = apply_block_masks(pit0, n, remainder=False, verbose=False)
    check(any("remainder=False" in str(x.message) for x in w), "remainder=False emits a warning")
    check(sorted(c for _, c in sk) == sorted(c for c in chans if c % n and c >= n),
          f"remainder=False leaves the non-divisible layers per-channel ({[c for _, c in sk]})")

    pit = PIT(model, cost=params, input_shape=IN_SHAPE, train_rf=False, train_dilation=False)
    converted, skipped, frozen = apply_block_masks(pit, n)
    check(not skipped, "default mode: no layer left per-channel")
    check(sorted(c for _, c in frozen) == sorted(c for c in chans if c < n),
          f"layers with C < N frozen ({[c for _, c in frozen]})")
    layers = [(nm, l) for nm, l in pit.seed.named_modules() if isinstance(l, PITConv2d)]

    # --- 2: exhaustive on/off combinations for every converted masker
    for m, c in converted:
        sizes = m._block_sizes
        blk = m.block
        layer = next(l for _, l in layers if l.out_features_masker is m)
        seen_counts = set()
        orig = blk.detach().clone()
        with torch.no_grad():
            for combo in itertools.product([0.0, 1.0], repeat=len(sizes)):
                blk.copy_(torch.tensor(combo))
                mask = layer.features_mask
                k = int(mask.sum())
                seen_counts.add(k)
                segs = mask.split(sizes)
                full_on = [bool(s.all()) for s in segs[:-1]] if sizes[-1] != n else None
                if sizes[-1] != n:
                    rem_on = bool(segs[-1].all())
                    assert not (rem_on and not all(full_on)), (sizes, combo, k)
                    ka_block = segs[len(sizes) - 2]
                else:
                    ka_block = segs[-1]
                assert bool(ka_block.all()), ("keep-alive block off", sizes, combo)
                assert all(bool(s.all()) or not bool(s.any()) for s in segs), "partial block"
            blk.copy_(orig)
        ok = all(k == c or k % n == 0 for k in seen_counts)
        check(ok, f"C={c} blocks={sizes}: reachable counts {sorted(seen_counts)} "
                  f"are all multiples of {n} or {c}")

    # --- 3: gradients reach every prunable block value. Setup: all blocks "on", leftover block
    #        with the lowest value (so it is the arg-min of its constraint and gets its own
    #        gradient). The keep-alive block gets ZERO gradient by design (as PIT's
    #        always-on channel).
    pit.train()
    with torch.no_grad():
        for m, _ in converted:
            b = m.block
            b.fill_(1.0)
            if m._block_sizes[-1] != n:
                b[-1] = 0.9
    loss = F.cross_entropy(pit(X[:64]), Y[:64]) + 1e-6 * pit.cost
    loss.backward()
    for m, c in converted:
        g = m.block.grad
        sizes = m._block_sizes
        n_full = len(sizes) - (1 if sizes[-1] != n else 0)
        ka_idx = n_full - 1
        prunable = [i for i in range(len(sizes)) if i != ka_idx]
        check(g is not None and torch.isfinite(g).all()
              and all(g[i] != 0 for i in prunable) and g[ka_idx] == 0,
              f"C={c}: gradient on every prunable block, zero on keep-alive "
              f"({[round(v, 4) for v in g.tolist()]})")
    pit.zero_grad()

    # --- 4: warmup -> search, short training with strong cost
    opt_net = torch.optim.Adam(pit.net_parameters(), 1e-3)
    opt_nas = torch.optim.Adam(pit.nas_parameters(), 5e-2)
    pit.train_features = False
    for i in range(0, 512, 64):
        opt_net.zero_grad(); F.cross_entropy(pit(X[i:i+64]), Y[i:i+64]).backward(); opt_net.step()
    pit.train_features = True
    for _ in range(3):
        for i in range(0, 1024, 64):
            opt_net.zero_grad(); opt_nas.zero_grad()
            (F.cross_entropy(pit(X[i:i+64]), Y[i:i+64]) + 3e-5 * pit.cost).backward()
            opt_net.step(); opt_nas.step()
    pit.eval()
    for nm, l in layers:
        c, k = l.out_channels, l.out_features_opt
        print(f"  {nm:12s} {c:4d} -> {k:4d}")
        conv = any(l.out_features_masker is m for m, _ in converted)
        if conv:
            check(k == c or k % n == 0, f"{nm}: result is {c} or a multiple of {n}")
        if c < n:
            check(k == c, f"{nm}: C={c} < N={n} -> not pruned")

    # --- 5: export + BN restore, and equivalence with plain per-channel masks
    xt = X[:32]
    with torch.no_grad():
        y_pit = pit(xt)
    exp = pit.export(); exp.eval(); restore_exported_bn(pit, exp, verbose=False)
    with torch.no_grad():
        d = (y_pit - exp(xt)).abs().max().item()
    check(d < 1e-5, f"export (+BN restore) == PIT output (max|diff| {d:.1e})")
    convs = [m for m in exp.modules() if isinstance(m, nn.Conv2d)]
    print(f"  exported conv out_channels: {[cv.out_channels for cv in convs]}")
    blocks_to_plain(pit)
    with torch.no_grad():
        d2 = (y_pit - pit(xt)).abs().max().item()
    check(d2 == 0.0, f"block masks == equivalent plain per-channel masks (max|diff| {d2:.1e})")


run([33, 72, 40, 64], 32)
run([192, 64], 128)
print("\nALL CHECKS PASSED")
