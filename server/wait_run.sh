#!/usr/bin/env bash
# Ждёт, пока освободится одна из указанных карт, и гоняет на ней по очереди скрипты через run.sh.
#   bash wait_run.sh [--after <очередь>] <имя> "<gpu> [gpu ...]" [опции run.sh] <script.py> [аргументы] [+ [опции run.sh] <script.py> [аргументы]] ...
# Пример:
#   bash wait_run.sh grasp "1 7" grasp/grasp_check.py --obj tetrapak --video + grasp/grasp_check.py --obj tetrapak --side --video
# --after: не начинать, пока та очередь не кончилась (done/expired), чтобы шли строго друг за другом.
# Сам уходит в tmux (gt-wait-<имя>). Свободной считается карта без процессов вообще.
# Состояние: $GT_ROOT/queue/<имя>/status (state, gpu, шаг), steps (шаг, код выхода, папка запуска).
# Ждёт не дольше GT_WAIT_HOURS (24 ч), опрос карт раз в 30 с. --allow-shared не пропускается.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

POLL_SEC=30
WAIT_HOURS=${GT_WAIT_HOURS:-24}
INNER=0
AFTER=""
if [[ ${1:-} == --inner ]]; then INNER=1; shift; fi
if [[ ${1:-} == --after ]]; then AFTER=${2:-}; shift 2; fi

[[ $# -ge 3 ]] || { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
NAME=$1 GPUS=$2
shift 2
[[ $NAME =~ ^[A-Za-z0-9_-]+$ ]] || gt_die "имя только из букв, цифр, _ и -"
for g in $GPUS; do [[ $g =~ ^[0-9]+$ ]] || gt_die "номера GPU числами через пробел, получено '$GPUS'"; done
for a in "$@"; do [[ $a == --allow-shared ]] && gt_die "--allow-shared тут нельзя: ждём именно свободную карту"; done

# разбиваем аргументы на шаги по '+'
STEPS=() cur=""
for a in "$@"; do
    if [[ $a == + ]]; then STEPS+=("$cur"); cur=""; else cur+=$(printf '%q ' "$a"); fi
done
STEPS+=("$cur")
N=${#STEPS[@]}

Q="$GT_ROOT/queue/$NAME"
STATUS="$Q/status"

set_status() {  # state [gpu] [step]
    { echo "state=$1"; echo "gpu=${2:-}"; echo "step=${3:-}"; echo "since=$(date '+%F %T')"
      echo "queued=$(cat "$Q/queued_at")"; } >"$STATUS.tmp"
    mv "$STATUS.tmp" "$STATUS"
}

if [[ $INNER -eq 0 ]]; then
    tmux has-session -t "gt-wait-$NAME" 2>/dev/null && gt_die "очередь $NAME уже ждёт (tmux gt-wait-$NAME)"
    [[ -e $Q ]] && gt_die "$Q уже есть — возьмите другое имя"
    [[ -z $AFTER || -f $GT_ROOT/queue/$AFTER/status ]] || gt_die "нет очереди $AFTER"
    mkdir -p "$Q"
    date '+%F %T' >"$Q/queued_at"
    printf '%s\n' "${STEPS[@]}" >"$Q/commands"
    : >"$Q/steps"
    set_status waiting
    tmux new-session -d -s "gt-wait-$NAME" \
        "$(printf '%q ' env GT_CPUS="$GT_CPUS" nice -n 19 bash "$0" --inner ${AFTER:+--after "$AFTER"} "$NAME" "$GPUS" "$@") 2>&1 | tee -a $(printf '%q' "$Q/wait.log")"
    echo "Очередь $NAME: $N шаг(ов), жду свободную карту из [$GPUS]${AFTER:+, после очереди $AFTER}"
    echo "  статус: cat $STATUS"
    exit 0
fi

# карта свободна: ни одного процесса на ней и никто из наших не держит run.lock
gpu_free() {
    [[ -z $(gt_gpu_procs | awk -v g="$1" '$1==g') ]] || return 1
    ( flock -n 9 ) 9>"$GT_ROOT/locks/run.lock"
}

# та очередь, за которой стоим, уже всё
after_done() {
    [[ -z $AFTER ]] && return 0
    grep -qE '^state=(done|expired)$' "$GT_ROOT/queue/$AFTER/status" 2>/dev/null
}

# папки запусков шага, новые сверху (пусто, если их нет)
runs_of() { ls -td "$GT_RUNS/${NAME}_${1}_"* 2>/dev/null || true; }

deadline=$(( $(date +%s) + WAIT_HOURS * 3600 ))
i=1
while (( i <= N )); do
    gpu=""
    set_status waiting "" "$i/$N"
    while [[ -z $gpu ]]; do
        if after_done; then
            for g in $GPUS; do
                if gpu_free "$g"; then gpu=$g; break; fi
            done
        fi
        if [[ -z $gpu ]]; then
            (( $(date +%s) < deadline )) || { set_status expired "" "$i/$N"; echo "не дождался карты за $WAIT_HOURS ч"; exit 1; }
            sleep "$POLL_SEC"
        fi
    done

    set_status running "$gpu" "$i/$N"
    echo "[$(date +%T)] шаг $i/$N на GPU $gpu: ${STEPS[$((i - 1))]}"
    eval "set -- ${STEPS[$((i - 1))]}"
    # опции run.sh до скрипта, потом скрипт, номер карты и аргументы скрипта
    opts=()
    while [[ ${1:-} == --* ]]; do
        opts+=("$1")
        [[ $1 == --timeout || $1 == --stall || $1 == --name ]] && { shift; opts+=("$1"); }
        shift
    done
    script=$1; shift
    before=$(runs_of "$i" | wc -l)
    rc=0
    bash "$HERE/run.sh" --fg "${opts[@]}" --name "${NAME}_${i}" "$script" "$gpu" "$@" || rc=$?
    run_dir=$(runs_of "$i" | head -1)
    if [[ $(runs_of "$i" | wc -l) -eq $before ]]; then
        # run.sh отказал до запуска (карту успели занять) — ждём дальше этот же шаг
        echo "[$(date +%T)] шаг $i не стартовал (код $rc), жду снова"
        sleep "$POLL_SEC"
        continue
    fi
    echo "$i $rc $run_dir" >>"$Q/steps"
    i=$(( i + 1 ))
done
set_status done
echo "[$(date +%T)] всё"
