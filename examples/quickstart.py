"""Minimal yolopit workflow: search -> pruned model -> fine-tuning -> validation.

    python examples/quickstart.py --model your_trained_model.pt --data your_data.yaml --n 16

The model should already be trained on the dataset (the search starts from it). Without
arguments it runs a tiny demo on coco8 (Ultralytics' 8-image dataset, downloaded automatically):
on coco8 the masks barely move, it only shows that everything runs.
"""
import argparse

from yolopit import PITYOLO, PrunedTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolo26n.pt")
ap.add_argument("--data", default="coco8.yaml")
ap.add_argument("--n", type=int, default=16, help="channels are pruned in blocks of N")
ap.add_argument("--epochs", type=int, default=10, help="search epochs")
ap.add_argument("--ft-epochs", type=int, default=10, help="fine-tuning epochs")
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default=None)
args = ap.parse_args()

common = dict(imgsz=args.imgsz, batch=args.batch, device=args.device, project="runs/yolopit",
              exist_ok=True)

search = PITYOLO(args.model, n=args.n, cost="ops", trace_imgsz=args.imgsz)
search.train(data=args.data, epochs=args.epochs, name="search", **common)
pruned = search.export_pruned()                      # runs/yolopit/search/weights/pruned.pt
pruned.train(data=args.data, epochs=args.ft_epochs, name="finetune", trainer=PrunedTrainer,
             **common)
metrics = pruned.val(data=args.data, name="val", **common)
print(f"mAP50-95 after fine-tuning: {metrics.box.map:.4f}")
