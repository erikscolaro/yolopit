import torch.nn as nn
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

