"""Esplorazione: Detect aperto. 1) stesse uscite dell'originale (train ed eval); 2) detach del ramo one2one accettato da
PLiNIO; 3) loss e backward: il gradiente arriva alle maschere della testa e a quelle prima congelate.
    CUDA_VISIBLE_DEVICES= python esplorazioni/test_detect_aperto.py <best.pt>
"""
import sys, torch, torch.nn as nn
sys.path.insert(0, ".")
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2
from plinio.methods import PIT
from plinio.methods.pit.nn import PITModule
from plinio.graph import inspection as _insp, annotation as _annot
from yolopit.search import COSTS, C3k2Split, YoloFx, apply_block_masks


class Detach(nn.Module):        # foglia fx: in training taglia il gradiente del ramo one2one, come Detect.forward
    def forward(self, x):
        return x.detach() if self.training else x


class DetectConvs(nn.Module):
    """Le conv di Detect nel grafo: per livello [box, cls, box one2one, cls one2one]. View, cat e decode restano fuori."""
    def __init__(s, d):
        super().__init__()
        s.cv2, s.cv3, s.o2, s.o3 = d.cv2, d.cv3, d.one2one_cv2, d.one2one_cv3
        s.f, s.i, s.nl, s.det = d.f, d.i, d.nl, Detach()
    def forward(s, x):
        out = []
        for i in range(s.nl):
            xd = s.det(x[i])
            out += [s.cv2[i](x[i]), s.cv3[i](x[i]), s.o2[i](xd), s.o3[i](xd)]
        return out


def assemble(d, o):
    """Dalle 12 uscite di DetectConvs a quello che ritorna Detect.forward. `feats` serve solo per le forme (ancore)."""
    bs = o[0].shape[0]
    def head(k):
        b, c = o[k::4], o[k + 1::4]
        return {"boxes": torch.cat([t.view(bs, 4 * d.reg_max, -1) for t in b], -1),
                "scores": torch.cat([t.view(bs, d.nc, -1) for t in c], -1), "feats": list(b)}
    preds = {"one2many": head(0), "one2one": head(2)}
    if d.training:
        return preds
    y = d._inference(preds["one2one" if d.end2end else "one2many"])
    if d.end2end:
        y = d.postprocess(y.permute(0, 2, 1))
    return y if d.export else (y, preds)


_fp = _insp.is_features_propagating_op
def _prop(n, parent):
    if n.op == "call_module" and isinstance(parent.get_submodule(str(n.target)), Detach):
        return True
    return _fp(n, parent)
_insp.is_features_propagating_op = _annot.is_features_propagating_op = _prop
from plinio.methods.pit import graph as _pg
_leaf = _pg.PITTracer.is_leaf_module
_pg.PITTracer.is_leaf_module = lambda self, m, q: isinstance(m, Detach) or _leaf(self, m, q)

w = sys.argv[1]
ref = YOLO(w).model.float()
src = YOLO(w).model.float()
for i, l in enumerate(src.model):
    if isinstance(l, C3k2): src.model[i] = C3k2Split(l)
det = src.model[-1]
src.model[-1] = DetectConvs(det)
net = YoloFx(src.model, src.save)
x = torch.rand(2, 3, 640, 640)

# 1) equivalenza senza PIT
ref.eval(); net.eval(); det.eval()
with torch.no_grad():
    y0 = ref(x)[0]; y1 = assemble(det, net(x))[0]
print("1) eval, differenza massima:", float((y0 - y1).abs().max()))
assert torch.allclose(y0, y1, atol=1e-5)

# 2) PIT con detach
pit = PIT(net.eval(), cost=dict(COSTS), input_shape=(3, 640, 640), train_rf=False, train_dilation=False)
apply_block_masks(pit, 2, remainder=True, verbose=False)
pit.eval()
with torch.no_grad():
    y2 = assemble(det, pit(x))[0]
print("2) PIT a maschere piene, differenza massima:", float((y0 - y2).abs().max()))
assert torch.allclose(y0, y2, atol=1e-4)

# 3) loss e gradiente
pit.train(); det.train(); ref.train()
from ultralytics.cfg import get_cfg
ref.args = get_cfg()
crit = ref.init_criterion()
batch = {"img": x, "batch_idx": torch.tensor([0., 1.]), "cls": torch.tensor([[1.], [2.]]),
         "bboxes": torch.tensor([[.5, .5, .2, .3], [.3, .6, .1, .1]])}
loss, items = crit(assemble(det, pit(x)), batch)
loss.sum().backward()
g = {n: m.out_features_masker for n, m in pit.seed.named_modules() if isinstance(m, PITModule) and isinstance(m, nn.Conv2d)}
def grad(n):
    ps = [p for p in g[n].parameters() if p.requires_grad]
    return None if not ps or ps[0].grad is None else float(ps[0].grad.abs().sum())
print("3) loss", loss.detach().reshape(-1).tolist())
for n in ("layers.16.cv2.conv", "layers.19.cv2.conv", "layers.22.cv2.conv", "layers.23.o2.0.0.conv", "layers.23.o3.2.0.0.conv",
          "layers.23.cv3.1.1.0.conv", "layers.23.o3.0.2"):
    print("   grad maschera", n, grad(n))
assert grad("layers.23.o3.2.0.0.conv") and grad("layers.22.cv2.conv")
print("OK")
