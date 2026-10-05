#!/usr/bin/env bash
# Ночной запуск с перезапуском: гоняет обучалку через run.sh, пока в ~/grasp_task/ckpt/<имя>/ не появится DONE.
# Если run.sh отказался (карта занята) или процесс упал - ждёт 5 мин и пробует снова, обучалка сама продолжит с бэкапа.
# Сдаётся после 20 падений (если падает раз за разом - значит баг, крутиться всю ночь смысла нет) или по времени ДО.
# Запускать в tmux руками:
#   tmux new -s night-qmap
#   bash tmp_rl/night.sh qmap 1 07:30 tmp_rl/train_qmap.py --hours 8 --num-envs 8
# Опции run.sh (например --allow-shared) - через RUN_OPTS="--allow-shared" перед командой.
set -u
NAME=$1 GPU=$2 UNTIL=$3 SCRIPT=$4
shift 4
DONE=~/grasp_task/ckpt/$NAME/DONE
CODE=~/grasp_task/code
crashes=0
end=$(date -d "$UNTIL" +%s)
(( end < $(date +%s) )) && end=$(( end + 86400 ))  # 07:30 - это уже завтра

while true; do
    [[ -f $DONE ]] && { echo "[$(date +%T)] $NAME: DONE, всё"; break; }
    (( $(date +%s) > end )) && { echo "[$(date +%T)] $NAME: время вышло ($UNTIL), выхожу"; break; }
    t0=$(date +%s)
    echo "[$(date +%T)] $NAME: запускаю на GPU $GPU"
    (cd "$CODE" && bash server/run.sh --fg ${RUN_OPTS:-} --timeout 14h --name "$NAME" "$SCRIPT" "$GPU" --name "$NAME" "$@")
    rc=$?
    took=$(( $(date +%s) - t0 ))
    [[ -f $DONE ]] && { echo "[$(date +%T)] $NAME: DONE, всё"; break; }
    # run.sh отказывает за секунды, а Isaac стартует пару минут: шёл дольше минуты - значит запустился и упал
    if (( took > 60 )); then
        crashes=$(( crashes + 1 ))
        echo "[$(date +%T)] $NAME: упал (код $rc, шёл $took с), падений $crashes из 20"
        (( crashes >= 20 )) && { echo "[$(date +%T)] $NAME: 20 падений, сдаюсь"; break; }
    else
        echo "[$(date +%T)] $NAME: не запустился (код $rc) - наверно карта занята"
    fi
    sleep 300
done
