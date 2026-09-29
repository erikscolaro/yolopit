# PIT channel search per YOLO26 — pacchetto Colab

Pipeline completa: **ricerca PIT (PLiNIO) → export del modello prunato → validazione del modello
esportato → fine-tuning → validazione finale**. Niente export ONNX.

## Contenuto

| File | |
|---|---|
| `PIT_YOLO_colab.ipynb` | notebook: installa le dipendenze ed esegue tutta la pipeline |
| `pit_yolo.py` | API (`PITYOLO`), trainer della ricerca e del fine-tuning, `C3k2Split`, wrapper fx |
| `pit_block_masks.py` | maschere a blocchi di N canali, ripristino BN all'export, fix PLiNIO |
| `search.yaml` | configurazione di esempio (Ultralytics + PIT) se preferisci il YAML |

## Uso su Colab

1. Apri `PIT_YOLO_colab.ipynb` su Colab, runtime **GPU**.
2. Cella 1: lascia `ZIP_ON_DRIVE = ""` e carica questo zip quando richiesto, oppure mettilo su
   Drive e scrivi il percorso.
3. Cella 3: imposta `MODEL` (il tuo modello già addestrato), `DATA` (yaml del dataset), `N`,
   epoche e scheduler. Per non perdere i risultati metti `PROJECT` su Drive.
4. Esegui tutto.

La cella 2 installa Ultralytics 8.4.165 e PLiNIO al commit `3d6b5e0` (versioni testate) più
`networkx`, `tdigest`, `onnx`. PyTorch resta quello di Colab. Se Ultralytics era già stato
importato in un'altra versione, riavvia il runtime e riesegui.

## Output (in `PROJECT/NAME/`)

- `search/`: tutto quello di Ultralytics + `pit_results.png` (costo, mAP, learning rate di pesi e
  maschere, canali per layer), `channels.csv`, `pit_args.yaml`, `weights/pruned.pt`
- `val_after_export/`: validazione del modello esportato (deve coincidere con la fine della ricerca)
- `finetune/`: fine-tuning del modello prunato (`weights/best.pt` = modello finale)
- `val_final/`, `summary.json`

## Default

Pensati per "modello già addestrato sui dati → ricerca → fine-tuning": pesi con SGD e nessun
warmup, maschere con AdamW e scheduler proprio (separato da quello dei pesi), AMP sempre spenta
nella ricerca. Warning se attivi un warmup.

## Da sapere

- Solo detection, architetture YOLO26 (C3k2 prunabile; C2PSA, attenzione e Detect non prunati).
- `resume` e multi-GPU non supportati.
- N va scelto sulla larghezza SIMD del target: canali non allineati possono rendere il modello
  prunato più lento di quello originale.
