#!/usr/bin/env bash
# Prova esplorativa: stessa ricerca finale della catena (N-, 30,3 ms, pendenza vincente, SGD 0,01, forza 2) ma con la
# testa Detect aperta (yolopit di questo worktree, LUT e mini-fit in artifacts/lut/rpi5_detect). Poi da zero 300 epoche e Pi.
# Parte quando la catena ha scelto la pendenza. Diario in runs/catena_notte.log, log in runs/detect_*.log.
#   systemd-run --user --unit=prova-detect --working-directory=$HOME/Desktop/yoloeval <questo file>
set -uo pipefail
cd /home/scolaroe/Desktop/yoloeval
export PYTHONPATH=/home/scolaroe/Desktop/yolopit/.claude/worktrees/detect-aperta YOLOPIT_OPEN_DETECT=1
D=experiments/pruning/pendenze/livello1; PY=.venv/bin/python; L=runs/catena_notte.log
passo() { echo "[$(date '+%F %T')] detect aperta: $*" >> $L; }
until grep -q "5. pendenza vincente" $L; do sleep 120; done
W=$(grep -o "5. pendenza vincente: [a-z]*" $L | tail -1 | awk '{print $NF}')
X="model=yolo26 model.width=0.25 model.depth=D101 strategy=pit_latency target=rpi5_640 target.lut_dir=artifacts/lut/rpi5_detect
   logging=local strategy.search.nas.optimizer=SGD strategy.search.nas.lr0=0.01 strategy.search.regularizer.strength=2
   strategy.search.pit.latency.estimator=$W strategy.search.workers=8"
X=$(echo $X)
passo "ricerca da N-, pendenza $W"
if $PY run.py stage=search $X > runs/detect_search.log 2>&1; then
  passo "da zero 300 epoche"
  if $PY run.py stage=finetune $X finetune=scratch finetune.workers=8 > runs/detect_scratch.log 2>&1; then
    flock runs/.pi.lock bash -c "for s in export benchmark; do $PY run.py stage=\$s $X finetune=scratch finetune.workers=8 || break; done" > runs/detect_pi_scratch.log 2>&1
  else passo "training FALLITO: vedi runs/detect_scratch.log"; fi
else
  passo "ricerca FALLITA: vedi runs/detect_search.log"
fi
passo FINE
