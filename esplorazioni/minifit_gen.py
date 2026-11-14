"""Mini-fit dello stimatore con la testa aperta: reti N- potate a caso (testa compresa), IR di deploy per il Pi.
Come src.lut.b4r di yoloeval, ma con i pesi veri e gli strati della testa one2one.
    cd ~/Desktop/yoloeval && PYTHONPATH=<worktree yolopit> YOLOPIT_OPEN_DETECT=1 .venv/bin/python <questo file> <dir> <best.pt> [n] [arch] [seed0]
Scrive <dir>/results/nets.csv, layers.csv e <dir>/runs/nets/<rete>/ (IR + pt). Poi: python -m src.lut.pi b4r <dir>
"""
import csv, shutil, sys, tempfile
from pathlib import Path
import torch
sys.path.insert(0, ".")
from src.lut import pi as lut
from src.lut.b4r import maskers, randomize, nonconv_elems
from src.backends import get_backend
from src.remote.connection import LocalConnection
from yolopit.search import PITYOLO, build_pit_model
from ultralytics.nn.modules import C2PSA
from ultralytics.nn.modules.block import PSABlock


def conv_layers(model):
    """(nome, famiglia, c_in, c_out) delle conv di deploy fuori da PSA: testa one2one compresa, one2many esclusa."""
    i = model.model[0].i
    skip = (f"net.layers.{i}.cv2.", f"net.layers.{i}.cv3.", "model.")      # one2many; model.0 = alias della testa
    inside = set()
    for name, mod in model.named_modules():
        if isinstance(mod, (C2PSA, PSABlock)):
            inside |= {f"{name}.{n}" if n else name for n, _ in mod.named_modules()}
    rows, hooks = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d) and name not in inside and not name.startswith(skip):
            dw = mod.groups > 1
            assert not dw or mod.groups == mod.in_channels == mod.out_channels, name
            hooks.append(mod.register_forward_hook(lambda mo, i, o, name=name, dw=dw: rows.append(
                (name, f"{'dw' if dw else 'k'}{mo.kernel_size[0]}s{mo.stride[0]}_{o.shape[-1]}", mo.in_channels, mo.out_channels))))
    with torch.no_grad():
        model(torch.zeros(1, 3, 640, 640))
    for h in hooks:
        h.remove()
    return rows


def main(d: Path, weights: str, n: int = 32, arch: str = "w0.25_D101", seed0: int = 7000):
    cfg = lut.make_cfg(); backend = get_backend("openvino", cfg)
    res, nets = d / "results", d / "runs" / "nets"
    res.mkdir(parents=True, exist_ok=True); nets.mkdir(parents=True, exist_ok=True)
    nets_rows, layer_rows = [], []
    for j in [-1, *range(n)]:
        name = f"{arch}_" + ("full" if j < 0 else f"h{j:03d}")
        w = build_pit_model(weights, n=2, verbose=False)
        g = randomize(maskers(w), -1 if j < 0 else seed0 + j)
        costs = {m: float(v) for m, v in w.real_costs().items()}
        p = PITYOLO.__new__(PITYOLO); p.model, p.trainer = w, None
        tmp = Path(tempfile.mkdtemp())
        yo = p.export_pruned(tmp / f"{name}.pt")
        layers = conv_layers(yo.model.eval())
        dst = nets / name
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(backend.export(LocalConnection(), cfg, tmp / f"{name}.pt", tmp / "ov"), dst)
        shutil.copy(tmp / f"{name}.pt", dst / f"{name}.pt"); shutil.rmtree(tmp, ignore_errors=True)
        nets_rows.append({"net": name, "arch": arch, "seed": j, "level": round(g, 4), "layers": len(layers),
                          **{f"cost_{m}": v for m, v in costs.items()}, "nonconv_elems": nonconv_elems(next(dst.glob("*.xml")))})
        layer_rows += [{"net": name, "order": i, "layer": ln, "family": f, "c_in": ci, "c_out": co} for i, (ln, f, ci, co) in enumerate(layers)]
        print(name, f"g={g:.2f}", len(layers), "strati", {m: f"{v:.3g}" for m, v in costs.items()}, flush=True)
        for path, rows in ((res / "nets.csv", nets_rows), (res / "layers.csv", layer_rows)):
            with open(path, "w", newline="") as fh:
                wr = csv.DictWriter(fh, list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32, *sys.argv[4:5], *map(int, sys.argv[5:6]))
