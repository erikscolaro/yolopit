"""Temperature of the block masks (pit.temperature) on a small CNN.

Checks:
 1. temperature_at: T0 at the start, T0/100 at the last soft epoch, 0 in the hard epochs, 0 without T0
 2. T = 0: the forward is PLiNIO's (hard masks)
 3. T > 0 in training: a block at the threshold counts ~0.5 in the forward, a full one 1, and the
    gradient reaches its mask (it is not stuck at 0 like a hard mask far from the threshold)
 4. T > 0 does not touch the cost (discrete) nor eval / export (hard)
"""
import torch
import torch.nn as nn

from plinio.methods import PIT
from plinio.cost import params
from plinio.methods.pit.nn import PITConv2d
from yolopit import masks as M
from yolopit.masks import apply_block_masks

torch.manual_seed(0)
ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
                    nn.Conv2d(8, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 2))
pit = PIT(net, input_shape=(3, 8, 8), cost=params, discrete_cost=True)
apply_block_masks(pit, 2, verbose=False)
conv = next(m for m in pit.seed.modules() if isinstance(m, PITConv2d))
masker = conv.out_features_masker

# 1
check("T0 at the first epoch", abs(M.temperature_at(0, 10, 0.1, 3) - 0.1) < 1e-12)
check("T0/100 at the last soft epoch", abs(M.temperature_at(6, 10, 0.1, 3) - 0.001) < 1e-12)
check("0 in the hard epochs", M.temperature_at(7, 10, 0.1, 3) == 0.0 and M.temperature_at(9, 10, 0.1, 3) == 0.0)
check("0 without T0", M.temperature_at(0, 10, None, 3) == 0.0)

with torch.no_grad():
    masker.block[0] = 0.5          # block 0 on the threshold (off for the hard mask)
    masker.block[1] = 1.0          # full
x = torch.randn(4, 3, 8, 8)
seen = {}
hook = conv.register_forward_hook(lambda m, i, o: seen.__setitem__("y", o.detach()))
pit.train()

# 2
M.TEMPERATURE["T"] = 0.0
pit(x)
check("T=0: block at the threshold is off", seen["y"][:, :2].abs().max() == 0)

# 3
M.TEMPERATURE["T"] = 0.05
cost_soft_before = float(pit.get_cost())
pit.zero_grad()
pit(x).sum().backward()
y = seen["y"]
ratio = (y[:, :2].abs().mean() / y[:, 2:4].abs().mean()).item()
check(f"T>0: block at the threshold counts about half of a full one (ratio {ratio:.2f})", 0.2 < ratio < 1.5)
check("T>0: gradient reaches the threshold block", masker.block.grad is not None and masker.block.grad[0] != 0)

# 4
M.TEMPERATURE["T"] = 0.0
cost_hard = float(pit.get_cost())
check("cost unchanged by T (discrete)", cost_soft_before == cost_hard)
M.TEMPERATURE["T"] = 0.05
pit.eval()
pit(x)
check("eval: hard masks even with T>0", seen["y"][:, :2].abs().max() == 0)
check("features_mask (export): hard", conv.features_mask[:2].sum() == 0)
M.TEMPERATURE["T"] = 0.0
hook.remove()

print("ALL CHECKS PASSED" if all(ok) else "SOME CHECKS FAILED")
