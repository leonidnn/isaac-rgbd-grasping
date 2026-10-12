#!/usr/bin/env bash
# Ждёт, пока освободится одна из указанных карт, и гоняет на ней по очереди скрипты через run.sh.
#   bash wait_run.sh [--after <очередь>] <имя> "<gpu> [gpu ...]" [опции run.sh] <script.py> [аргументы] [+ [опции run.sh] <script.py> [аргументы]] ...
# Пример:
#   bash wait_run.sh grasp "1 7" grasp/grasp_check.py --obj tetrapak --video + grasp/grasp_check.py --obj tetrapak --side --video
# --after: не начинать, пока та очередь не кончилась (done/expired), чтобы шли строго друг за другом.
# Сам уходит в tmux (gt-wait-<имя>). Свободной считается карта без процессов вообще.
# Состояние: $GT_ROOT/queue/<имя>/status (state, gpu, шаг), steps (шаг, код выхода, папка запуска).
# Ждёт не дольше GT_WAIT_HOURS (24 ч) на каждый шаг, опрос карт раз в 30 с. --allow-shared руками не пропускается.
# --until-done <имя> в опциях шага: обучение с бэкапом, шаг перезапускается (снова ждёт карту), пока нет
#   ckpt/<имя>/DONE, не больше GT_MAX_CRASHES (20) раз. Следующий шаг идёт сразу после DONE.
# Шаг, снятый run.sh ради соседа (мало памяти у соседа или пропал его процесс), повторяется (до GT_MAX_NEIGH, 30 раз).
# --must-pass в опциях шага: шаг-смоук, код не 0 - очередь failed, очереди с --after на неё отменяются (cancelled).
# --max-mb N --max-util P (только вместе, 04.10.2026): ждём не пустую карту, а такую, где занято < N МиБ и загрузка < P %
#   три проверки подряд (~1.5 мин, загрузка скачет). Тогда run.sh идёт с --allow-shared и GT_SHARED_MAX_MB=N.
#   Только по явному решению пользователя.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

POLL_SEC=30
WAIT_HOURS=${GT_WAIT_HOURS:-24}
INNER=0
AFTER=""
if [[ ${1:-} == --inner ]]; then INNER=1; shift; fi
MAX_MB="" MAX_UTIL=""
while [[ ${1:-} == --after || ${1:-} == --max-mb || ${1:-} == --max-util ]]; do
    case $1 in
        --after) AFTER=${2:-} ;;
        --max-mb) MAX_MB=${2:-} ;;
        --max-util) MAX_UTIL=${2:-} ;;
    esac
    shift 2
done
STABLE=3

[[ $# -ge 3 ]] || { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
NAME=$1 GPUS=$2
shift 2
[[ $NAME =~ ^[A-Za-z0-9_-]+$ ]] || gt_die "имя только из букв, цифр, _ и -"
for g in $GPUS; do [[ $g =~ ^[0-9]+$ ]] || gt_die "номера GPU числами через пробел, получено '$GPUS'"; done
for a in "$@"; do [[ $a == --allow-shared ]] && gt_die "--allow-shared тут нельзя: ждём именно свободную карту"; done
if [[ -n $MAX_MB || -n $MAX_UTIL ]]; then
    [[ $MAX_MB =~ ^[0-9]+$ && $MAX_UTIL =~ ^[0-9]+$ ]] || gt_die "--max-mb и --max-util только вместе и числами"
fi

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
    # как в run.sh (e92cf87): все GT_* явно, иначе GT_OUR_MAX_MB и др. из командной строки до шагов не доедут
    mapfile -t GT_VARS < <(env | grep -E '^GT_[A-Z0-9_]+=' | grep -vE '^GT_(NICED|RUN_DIR)=')
    tmux new-session -d -s "gt-wait-$NAME" \
        "$(printf '%q ' env "${GT_VARS[@]}" GT_CPUS="$GT_CPUS" nice -n 19 bash "$0" --inner ${AFTER:+--after "$AFTER"} ${MAX_MB:+--max-mb "$MAX_MB" --max-util "$MAX_UTIL"} "$NAME" "$GPUS" "$@") 2>&1 | tee -a $(printf '%q' "$Q/wait.log")"
    echo "Очередь $NAME: $N шаг(ов), жду свободную карту из [$GPUS]${AFTER:+, после очереди $AFTER}"
    [[ -n $MAX_MB ]] && echo "  свободной считаю карту, где занято < $MAX_MB МиБ и загрузка < $MAX_UTIL % $STABLE проверки подряд"
    echo "  статус: cat $STATUS"
    exit 0
fi

# карта свободна: ни одного процесса на ней и никто из наших не держит её замок run_gpuN.lock.
# в режиме --max-mb/--max-util: занято < MAX_MB и загрузка < MAX_UTIL, и так STABLE проверок подряд
declare -A OK_IN_ROW
gpu_free() {
    ( flock -n 9 ) 9>"$GT_ROOT/locks/run_gpu$1.lock" || return 1
    if [[ -z $MAX_MB ]]; then
        [[ -z $(gt_gpu_procs | awk -v g="$1" '$1==g') ]]
        return
    fi
    local line used util
    line=$(nvidia-smi -i "$1" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
    used=$(echo "$line" | cut -d, -f1 | tr -d ' ')
    util=$(echo "$line" | cut -d, -f2 | tr -d ' ')
    if [[ $used =~ ^[0-9]+$ && $util =~ ^[0-9]+$ ]] && (( used < MAX_MB && util < MAX_UTIL )); then
        OK_IN_ROW[$1]=$(( ${OK_IN_ROW[$1]:-0} + 1 ))
    else
        OK_IN_ROW[$1]=0
    fi
    echo "[$(date +%T)] GPU $1: занято ${used:-?} МиБ, загрузка ${util:-?} %, подряд ок ${OK_IN_ROW[$1]}/$STABLE"
    (( ${OK_IN_ROW[$1]} >= STABLE ))
}

# та очередь, за которой стоим, уже всё
after_done() {
    [[ -z $AFTER ]] && return 0
    if grep -qE '^state=(failed|cancelled)$' "$GT_ROOT/queue/$AFTER/status" 2>/dev/null; then
        # та очередь провалилась (смоук не прошёл) - эту не запускаем вообще
        set_status cancelled "" "$i/$N"
        echo "[$(date +%T)] очередь $AFTER провалилась, отменяюсь"
        exit 1
    fi
    grep -qE '^state=(done|expired)$' "$GT_ROOT/queue/$AFTER/status" 2>/dev/null
}

# папки запусков шага, новые сверху (пусто, если их нет)
runs_of() { ls -td "$GT_RUNS/${NAME}_${1}_"* 2>/dev/null || true; }

deadline=$(( $(date +%s) + WAIT_HOURS * 3600 ))
MAX_CRASHES=${GT_MAX_CRASHES:-20}
declare -A crashes neigh
MAX_NEIGH=${GT_MAX_NEIGH:-30}
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
    opts=() until_done="" must_pass=0
    while [[ ${1:-} == --* ]]; do
        # --until-done <имя>: это обучение с бэкапом, гонять шаг заново, пока нет ckpt/<имя>/DONE (run.sh не отдаю)
        if [[ $1 == --until-done ]]; then until_done=$2; shift 2; continue; fi
        # --must-pass: смоук. Упал (код не 0) - вся очередь failed, и очереди с --after на неё отменяются
        if [[ $1 == --must-pass ]]; then must_pass=1; shift; continue; fi
        opts+=("$1")
        [[ $1 == --timeout || $1 == --stall || $1 == --name ]] && { shift; opts+=("$1"); }
        shift
    done
    script=$1; shift
    before=$(runs_of "$i" | wc -l)
    rc=0
    if [[ -n $MAX_MB ]]; then
        # карта не пустая, это мы сами и разрешили порогами - run.sh надо пустить на общую карту с тем же лимитом
        opts+=(--allow-shared)
        export GT_SHARED_MAX_MB=$MAX_MB
        OK_IN_ROW[$gpu]=0
    fi
    bash "$HERE/run.sh" --fg "${opts[@]}" --name "${NAME}_${i}" "$script" "$gpu" "$@" || rc=$?
    run_dir=$(runs_of "$i" | head -1)
    if [[ $(runs_of "$i" | wc -l) -eq $before ]]; then
        # run.sh отказал до запуска (карту успели занять) — ждём дальше этот же шаг
        echo "[$(date +%T)] шаг $i не стартовал (код $rc), жду снова"
        sleep "$POLL_SEC"
        continue
    fi
    echo "$i $rc $run_dir" >>"$Q/steps"
    # run.sh снял нас ради соседа (у него выросла память или пропал процесс) - это не наш баг, тот же шаг ещё раз.
    # 12.10 смоук так и упал: карта 3 проверки была свободной, сели, сосед вырос, нас убили на 491 МиБ
    if [[ -n $run_dir ]] && grep -qE 'СТОП: (на GPU [0-9]+ свободно|чужой процесс)' "$run_dir/run.log" 2>/dev/null; then
        neigh[$i]=$(( ${neigh[$i]:-0} + 1 ))
        if (( ${neigh[$i]} <= MAX_NEIGH )); then
            echo "[$(date +%T)] шаг $i снят из-за соседа (${neigh[$i]} из $MAX_NEIGH), жду карту снова"
            OK_IN_ROW[$gpu]=0
            sleep "$POLL_SEC"
            continue
        fi
        echo "[$(date +%T)] шаг $i снят из-за соседа уже $MAX_NEIGH раз, дальше считаю как обычное падение"
    fi
    if (( must_pass && rc != 0 )); then
        set_status failed "$gpu" "$i/$N"
        echo "[$(date +%T)] шаг $i --must-pass упал (код $rc), очередь failed"
        exit 1
    fi
    # карта только что была наша - следующий запуск пробует её с одной проверки, а не с трёх
    [[ -n $MAX_MB ]] && OK_IN_ROW[$gpu]=$(( STABLE - 1 ))
    if [[ -n $until_done && ! -f $GT_ROOT/ckpt/$until_done/DONE ]]; then
        # упал или его стопнул run.sh (у соседа кончился процесс) - тот же шаг ещё раз, продолжит с бэкапа
        crashes[$i]=$(( ${crashes[$i]:-0} + 1 ))
        echo "[$(date +%T)] шаг $i без DONE (код $rc), перезапуск ${crashes[$i]} из $MAX_CRASHES"
        if (( ${crashes[$i]} >= MAX_CRASHES )); then
            echo "[$(date +%T)] шаг $i упал $MAX_CRASHES раз, бросаю его"
        else
            deadline=$(( $(date +%s) + WAIT_HOURS * 3600 ))
            continue
        fi
    fi
    i=$(( i + 1 ))
    deadline=$(( $(date +%s) + WAIT_HOURS * 3600 ))
done
set_status done
echo "[$(date +%T)] всё"
