"""pit_results.png: the search summary plot."""
from __future__ import annotations

import csv
from pathlib import Path


def plot_pit_results(save_dir, channel_rows=None, n=None):
    """pit_results.png in save_dir: cost (real, train/val), mAP50-95 and mAP50, learning rates of
    weights and masks (from results.csv), and kept channels per prunable layer (y grid every N
    channels, the pruning block, when `n` is given)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

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
    metrics = [m for m in ("ops", "params") if col(f"train/{m}")] or (
        ["cost"] if col("train/cost") else [])      # "cost": runs made before yolopit 0.2
    for m in metrics:
        a.plot(ep, col(f"train/{m}"), marker=".", label=m if m != "cost" else "train")
    a.set_title("cost of the whole model / initial (current architecture)"
                if metrics != ["cost"] else "cost of the current architecture / initial cost")
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
        if n:
            # a line every N channels; labels every k*N, at most ~20 of them
            k = max(1, -(-max(r[1] for r in pr) // (20 * n)))
            a.yaxis.set_major_locator(MultipleLocator(k * n))
            a.yaxis.set_minor_locator(MultipleLocator(n))
            a.grid(axis="y", which="both", alpha=.3); a.set_axisbelow(True)
    else:
        a.axis("off")
    fig.tight_layout()
    out = save_dir / "pit_results.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
