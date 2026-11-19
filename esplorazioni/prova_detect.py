"""Esplorazione: i rami conv di Detect dentro il grafo PIT, la decodifica fuori."""
import glob, sys, traceback, torch, torch.nn as nn
sys.path.insert(0, ".")
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2, Detect
from plinio.methods import PIT
from plinio.methods.pit.nn import PITModule
from yolopit.search import LEAVES, COSTS, C3k2Split, YoloFx

class DetectConvs(nn.Module):
    """Solo le conv di Detect: 4 rami per livello, niente view/cat/decode."""
    def __init__(s, d, detach):
        super().__init__()
        s.cv2, s.cv3, s.o2, s.o3 = d.cv2, d.cv3, d.one2one_cv2, d.one2one_cv3
        s.f, s.i, s.nl, s.detach = d.f, d.i, d.nl, detach
    def forward(s, x):
        out = []
        for i in range(s.nl):
            xd = x[i].detach() if s.detach else x[i]
            out += [s.cv2[i](x[i]), s.cv3[i](x[i]), s.o2[i](xd), s.o3[i](xd)]
        return out

w = glob.glob("/home/scolaroe/Desktop/yoloeval/artifacts/weights/yolo26n_D126_air_detection_all_e300_*/best.pt")[0]
for detach in (True, False):
    src = YOLO(w).model.float().eval()
    for i, l in enumerate(src.model):
        if isinstance(l, C3k2): src.model[i] = C3k2Split(l).eval()
    det = src.model[-1]
    src.model[-1] = DetectConvs(det, detach)
    net = YoloFx(src.model, src.save).eval()
    print("=== detach", detach, "foglie", [c.__name__ for c in LEAVES])
    try:
        pit = PIT(net, cost=dict(COSTS), input_shape=(3, 640, 640), train_rf=False, train_dilation=False)
    except Exception as e:
        print("ERRORE", type(e).__name__, str(e)[:300]); continue
    tot = lib = 0; rows = []
    for n, m in pit.seed.named_modules():
        if isinstance(m, PITModule) and hasattr(m, "out_features_masker"):
            fm = m.out_features_masker; c = m.out_channels if hasattr(m, "out_channels") else m.out_features
            free = any(p.requires_grad for p in fm.parameters())
            tot += c; lib += c * free
            if "22" in n.split(".")[1:2] or "23" in n or "cv2." in n or "cv3." in n or "o2" in n or "o3" in n: rows.append((n, c, free))
    print("conv PIT: canali %d, potabili %d (%.0f%%)" % (tot, lib, 100 * lib / tot))
    for r in rows[-28:]: print("   ", r)
    x = torch.zeros(1, 3, 640, 640); o = pit(x); print("forward ok, uscite:", len(o), [tuple(t.shape) for t in o[:4]])
    break
