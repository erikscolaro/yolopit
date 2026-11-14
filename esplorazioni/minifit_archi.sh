#!/usr/bin/env bash
# Mini-fit con la testa aperta per le altre architetture piu' grandi del budget: 25 reti potate a caso + la intera per
# architettura, generate su CPU (nice) e misurate sul Pi dentro il lock della catena. La LUT copre gia' le loro forme.
#   systemd-run --user --unit=minifit-archi --working-directory=$HOME/Desktop/yoloeval <questo file>
set -uo pipefail
cd /home/scolaroe/Desktop/yoloeval
W=/home/scolaroe/Desktop/yolopit/.claude/worktrees/detect-aperta
export PYTHONPATH=$W YOLOPIT_OPEN_DETECT=1 CUDA_VISIBLE_DEVICES=""
k=0
for a in w0.1875_D101:w0.1875_D101 w0.125_D170:w0.125_D170 w0.1875_D126:w0.1875_D126 n_D126:w0.25_D126 w0.1875_D170:w0.1875_D170 w0.25_D170:w0.25_D170; do
  k=$((k+1)); file=${a%%:*}; arch=${a##*:}; D=artifacts/lut/rpi5_detect/minifit_$arch
  mkdir -p $D
  if [ ! -f $D/gen.ok ]; then
    nice -n 10 .venv/bin/python $W/esplorazioni/minifit_gen.py $D $(ls artifacts/weights/yolo26${file}_air_detection_all_e300_*/best.pt | head -1) 25 $arch $((7000 + 100 * k)) > $D/gen.log 2>&1 && touch $D/gen.ok || { echo "$arch gen FALLITA"; continue; }
  fi
  flock runs/.pi.lock .venv/bin/python -m src.lut.pi b4r $D > $D/measure.log 2>&1 || echo "$arch misura FALLITA"
  echo "$(date '+%F %T') $arch fatta"
done
echo FINE
