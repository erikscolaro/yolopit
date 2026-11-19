"""Quanto diventa potabile con Detect aperto: canali e MAC della rete di deploy (testa one2one, senza one2many)."""
import glob, sys, torch, torch.nn as nn
sys.path.insert(0, "."); sys.path.insert(0, "esplorazioni")
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2, C2PSA, Detect
from ultralytics.nn.modules.block import PSABlock
from plinio.methods import PIT
from plinio.methods.pit.nn import PITModule
from yolopit.search import COSTS, C3k2Split, YoloFx, pit_layers_feeding, freeze_output_channels
from prova_detect_lib import DetectConvs

w = sys.argv[1]
def mac_hook(store, n):
    return lambda m, i, o: store.__setitem__(n, (m.out_channels, o.shape[2] * o.shape[3] * m.weight.numel()))
# parte fuori dal grafo, dal modello originale: PSA e testa one2one
ref = YOLO(w).model.float().eval(); R = {}
for n, m in ref.named_modules():
    if isinstance(m, nn.Conv2d): m.register_forward_hook(mac_hook(R, n))
ref(torch.zeros(1, 3, 640, 640))
leafn = [n for n, m in ref.named_modules() if isinstance(m, (C2PSA, PSABlock))]
psa = [v for n, v in R.items() if any(n.startswith(l + ".") for l in leafn)]
o2o = [v for n, v in R.items() if n.startswith("model.23.one2one")]
def conta(aperto):
    src = YOLO(w).model.float().eval()
    for i, l in enumerate(src.model):
        if isinstance(l, C3k2): src.model[i] = C3k2Split(l).eval()
    det = src.model[-1]; det.export = True
    leaves = (C2PSA, PSABlock) if aperto else (C2PSA, PSABlock, Detect)
    if aperto: src.model[-1] = DetectConvs(det, False)
    pit = PIT(YoloFx(src.model, src.save).eval(), cost=dict(COSTS), input_shape=(3, 640, 640), train_rf=False, train_dilation=False)
    freeze_output_channels(pit, pit_layers_feeding(pit, leaves), verbose=False)
    P = {}
    for n, m in pit.seed.named_modules():
        if isinstance(m, PITModule) and isinstance(m, nn.Conv2d): m.register_forward_hook(mac_hook(P, n))
    pit.eval(); pit(torch.zeros(1, 3, 640, 640))
    rows = []
    for n, m in pit.seed.named_modules():
        if n in P and ".23.cv2." not in n and ".23.cv3." not in n:       # one2many: solo training
            rows.append((n, *P[n], any(p.requires_grad for p in m.out_features_masker.parameters())))
    fissi = psa + ([] if aperto else o2o)
    tc = sum(r[1] for r in rows) + sum(c for c, _ in fissi); tm = sum(r[2] for r in rows) + sum(k for _, k in fissi)
    lc = sum(r[1] for r in rows if r[3]); lm = sum(r[2] for r in rows if r[3])
    print(f"Detect {'aperto' if aperto else 'chiuso'}: conv {len(rows)+len(fissi)}, canali {tc}, MAC {tm/1e9:.3f} G | potabili: canali {100*lc/tc:.0f}%  MAC {100*lm/tm:.0f}%")
    return {r[0]: r for r in rows}
a, b = conta(False), conta(True)
print("conv congelate che si liberano:", [(n, a[n][1], f"{a[n][2]/1e6:.0f}M") for n in a if not a[n][3] and b[n][3]])
print("conv ancora congelate nel grafo:", [(n, b[n][1], f"{b[n][2]/1e6:.0f}M") for n in b if not b[n][3]])
print(f"PSA: canali {sum(c for c,_ in psa)}, MAC {sum(k for _,k in psa)/1e6:.0f}M | testa one2one: canali {sum(c for c,_ in o2o)}, MAC {sum(k for _,k in o2o)/1e6:.0f}M")
