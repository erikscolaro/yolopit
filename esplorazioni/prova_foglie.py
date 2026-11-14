import glob, sys, traceback, torch
sys.path.insert(0, ".")
from yolopit.search import build_pit_model, LEAVES
print("foglie:", [c.__name__ for c in LEAVES])
w = glob.glob("/home/scolaroe/Desktop/yoloeval/artifacts/weights/yolo26n_D126_air_detection_all_e300_*/best.pt")[0]
try:
    m = build_pit_model(w, n=2, verbose=False)
    c = m.pit_meta["costs"]["ops"]
    print("OK  ops tot %.3g pit %.3g fisso %.3g min %.3g  -> fisso %.0f%%, minimo %.0f%%" % (c["total"], c["pit"], c["fixed"], c["min_total"], 100*c["fixed"]/c["total"], 100*c["min_total"]/c["total"]))
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)
    print("ERRORE", type(e).__name__, str(e)[:600])
    for f in tb[-6:]: print("  ", f.filename.split("site-packages/")[-1], f.lineno, f.name, "|", f.line)
