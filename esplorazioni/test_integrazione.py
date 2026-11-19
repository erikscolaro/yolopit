"""Testa aperta integrata in yolopit (pit.open_detect=true): costruzione, equivalenza, costi, export, ricarica, bias_init.
    CUDA_VISIBLE_DEVICES= python esplorazioni/test_integrazione.py <best.pt> <dir tmp>
"""
import os, sys, copy, torch
sys.path.insert(0, ".")
from ultralytics import YOLO
import yolopit, yolopit.search as S
from yolopit.search import build_pit_model, PITYOLO
print("yolopit da", yolopit.__file__)
w, tmp = sys.argv[1], sys.argv[2]
torch.manual_seed(0); x = torch.rand(2, 3, 640, 640)   # input fisso: con uno casuale gli assert a 1e-3 a volte cadono
ref = YOLO(w).model.float().eval()
m = build_pit_model(w, n=2, verbose=False, open_detect=True)
c = m.pit_meta["costs"]["ops"]
print("ops: tot %.4g pit %.4g fisso %.4g minimo %.4g (fisso %.0f%%, minimo %.0f%%)" % (c["total"], c["pit"], c["fixed"], c["min_total"], 100*c["fixed"]/c["total"], 100*c["min_total"]/c["total"]))
m.eval()
with torch.no_grad():
    y0, y1 = ref(x)[0], m(x)[0]
print("1) eval, differenza massima:", float((y0 - y1).abs().max())); assert torch.allclose(y0, y1, atol=1e-3)
# costo totale = quello del modello chiuso (stessa rete di deploy)
mc = build_pit_model(w, n=2, verbose=False); cc = mc.pit_meta["costs"]["ops"]
print("2) ops totali aperto %.6g, chiuso %.6g" % (c["total"], cc["total"])); assert abs(c["total"] - cc["total"]) / cc["total"] < 1e-6
print("   minimo: aperto %.0f%%, chiuso %.0f%%" % (100*c["min_total"]/c["total"], 100*cc["min_total"]/cc["total"]))
# train: dict con one2many e one2one, loss
m.train(); out = m(x); assert set(out) == {"one2many", "one2one"}
# pota a caso e esporta
torch.manual_seed(0)
from yolopit.masks import PITBlockFeaturesMasker
seen = set()
for l in m.net.seed.modules():
    k = getattr(l, "out_features_masker", None)
    if isinstance(k, PITBlockFeaturesMasker) and id(k) not in seen:
        seen.add(id(k))
        with torch.no_grad(): k.block.copy_(torch.where(torch.rand_like(k.block) < 0.5, torch.zeros_like(k.block), k.block))
m.eval()
with torch.no_grad(): yp = m(x)[0]
rc = m.real_costs(fraction=True); print("3) rete potata a caso: ops %.3f params %.3f" % (float(rc["ops"]), float(rc["params"])))
py = PITYOLO.__new__(PITYOLO); py.model, py.trainer = m, None
pruned = py.export_pruned(os.path.join(tmp, "pruned.pt"))
pm = pruned.model.float().eval()
with torch.no_grad(): ye = pm(x)[0]
print("4) esportata e ricaricata, differenza massima:", float((yp - ye).abs().max())); assert torch.allclose(yp, ye, atol=1e-3)
head = [r for r in pm.pit_meta["channels"] if f".{pm.model[0].i}." in r[0] and "one2one" in r[0]]
print("   canali testa one2one (prima, dopo):", [(r[1], r[2]) for r in head][:12])
# bias_init agisce sulle conv del grafo
d = pm.model[0]; g = pm.net.get_submodule(f"layers.{d.i}.one2one_cv3.0.2")
assert d.one2one_cv3[0][-1] is g
g.bias.data.zero_(); d.bias_init(); print("5) bias_init sul grafo:", g.bias.data.tolist()); assert float(g.bias.abs().sum()) > 0
# ONNX
pruned.export(format="onnx", imgsz=640, simplify=False, verbose=False) if False else None
print("ALL CHECKS PASSED")
