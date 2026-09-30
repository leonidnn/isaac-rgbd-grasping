#!/usr/bin/env bash
# Запуск python-скрипта на одном GPU со всеми предохранителями.
#   bash run.sh [опции] <script.py> <gpu> [аргументы скрипта]
# Опции:
#   --fg            в текущем терминале, без tmux
#   --allow-shared  пускать на карту, где уже есть чужие процессы (только по договорённости)
#   --non-rtx       разрешить не-RTX карту (A100) — только по явному решению
#   --timeout T     лимит времени, формат timeout (по умолчанию 40m)
#   --name N        имя запуска (по умолчанию имя скрипта)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

FG=0 SHARED=0 NONRTX=0 TIMEOUT=40m NAME=""
MIN_FREE_MB=${GT_MIN_FREE_MB:-7000}
SHARED_MAX_MB=${GT_SHARED_MAX_MB:-2500}
STALL_SEC=${GT_STALL_SEC:-1200}
MIN_DISK_GB=${GT_MIN_DISK_GB:-5}
ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
    case $1 in
        --fg) FG=1 ;;
        --allow-shared) SHARED=1 ;;
        --non-rtx) NONRTX=1 ;;
        --timeout) TIMEOUT=$2; shift ;;
        --name) NAME=$2; shift ;;
        -h|--help) usage ;;
        --*) gt_die "неизвестная опция $1" ;;
        *) break ;;
    esac
    shift
done
[[ $# -ge 2 ]] || usage
SCRIPT=$1 GPU=$2
shift 2
[[ $GPU =~ ^[0-9]+$ ]] || gt_die "номер GPU должен быть числом, получено '$GPU'"
[[ -f $SCRIPT ]] || gt_die "нет файла $SCRIPT"
SCRIPT=$(readlink -f "$SCRIPT")
[[ -x "$GT_ENV/bin/python" ]] || gt_die "нет окружения $GT_ENV, сначала bootstrap.sh"
NAME=${NAME:-$(basename "$SCRIPT" .py)}

# выбор карты только через sim/gt_app.py: никаких своих SimulationApp/AppLauncher и номеров GPU в коде
if grep -nE 'SimulationApp\(|AppLauncher\(|active_gpu|activeGpu|physics_gpu|cuda:[1-9]|--device' "$SCRIPT"; then
    gt_die "в $SCRIPT выбор GPU в обход sim/gt_app.py (строки выше)"
fi
LOCK="$GT_ROOT/locks/run.lock"

check_gpu() {
    local info name used total procs
    info=$(nvidia-smi -i "$GPU" --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null) \
        || gt_die "GPU $GPU не найден"
    IFS=, read -r name used total <<<"$info"
    name=${name# }; used=${used// /}; total=${total// /}
    if [[ $name != *RTX* && $NONRTX -eq 0 ]]; then
        gt_die "GPU $GPU — $name. Рендер только на RTX 2080 Ti (--non-rtx только по явному решению)"
    fi
    procs=$(gt_gpu_procs | awk -v g="$GPU" '$1==g')
    if [[ -n $procs ]]; then
        echo "На GPU $GPU уже есть процессы:"
        while read -r _ pid type mem; do
            echo "  pid $pid $type ${mem} MiB  $(ps -o user=,etime= -p "$pid" 2>/dev/null)"
        done <<<"$procs"
        [[ $SHARED -eq 1 ]] || gt_die "GPU $GPU занят. Выберите свободный или договоритесь и добавьте --allow-shared"
        (( used <= SHARED_MAX_MB )) || gt_die "на GPU $GPU занято ${used} MiB > ${SHARED_MAX_MB}"
    fi
    (( total - used >= MIN_FREE_MB )) || gt_die "на GPU $GPU свободно $(( total - used )) MiB < ${MIN_FREE_MB}"
    echo "GPU $GPU: $name, занято ${used}/${total} MiB — ок"
}

check_disk() {
    local avail_gb
    avail_gb=$(( $(df -Pk "$GT_ROOT" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
    (( avail_gb >= MIN_DISK_GB )) || gt_die "свободно ${avail_gb} ГБ < ${MIN_DISK_GB}"
}

lock_free() { ( flock -n 9 ) 9>"$LOCK"; }

# внешний вызов: проверки и уход в tmux
if [[ $FG -eq 0 ]]; then
    lock_free || gt_die "уже идёт другой запуск run.sh (одновременно — только один)"
    check_gpu
    check_disk
    SESSION="gt-$NAME"
    tmux has-session -t "$SESSION" 2>/dev/null && gt_die "tmux-сессия $SESSION уже есть"
    # env tmux-сервера может быть старым, поэтому GT_CPUS передаём явно
    tmux new-session -d -s "$SESSION" "$(printf '%q ' env GT_CPUS="$GT_CPUS" bash "$HERE/run.sh" --fg "${ORIG_ARGS[@]}")"
    echo "Запущено в tmux-сессии $SESSION (закроется сама по завершении)"
    echo "  смотреть: tmux attach -t $SESSION   (отключиться: Ctrl+b, затем d)"
    echo "  логи:     ls -t $GT_RUNS | head -1"
    exit 0
fi

# внутренний вызов: сам запуск
exec 9>"$LOCK"
flock -n 9 || gt_die "уже идёт другой запуск run.sh (одновременно — только один)"
check_gpu
check_disk
GT_KIT_GPU=$(gt_kit_gpu "$GPU") || true
[[ -n $GT_KIT_GPU ]] || gt_die "не удалось найти GPU $GPU в vulkaninfo (нужен для рендера Isaac Sim)"
export GT_KIT_GPU

RUN_DIR="$GT_RUNS/${NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/run.log"
export GT_RUN_DIR="$RUN_DIR" CUDA_VISIBLE_DEVICES="$GPU"

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

PID="" PGID="" TAILPID=""

in_our_group() { [[ $(ps -o pgid= -p "$1" 2>/dev/null | tr -d ' ') == "$PGID" ]]; }

our_gpu_pids() {
    local g p t m
    while read -r g p t m; do
        if [[ -n $p ]] && in_our_group "$p"; then echo "gpu$g:$p"; fi
    done < <(gt_gpu_procs)
    return 0
}

# stop_group now — сразу SIGKILL (наш процесс на чужой карте, ждать нельзя)
stop_group() {
    [[ -n $PGID ]] || return 0
    pgrep -g "$PGID" >/dev/null || return 0
    if [[ ${1:-} == now ]]; then
        kill -KILL -- "-$PGID" 2>/dev/null || true
        sleep 1
        return 0
    fi
    kill -TERM -- "-$PGID" 2>/dev/null || true
    for _ in $(seq 30); do
        pgrep -g "$PGID" >/dev/null || return 0
        sleep 1
    done
    log "процессы не завершились за 30 с, SIGKILL"
    kill -KILL -- "-$PGID" 2>/dev/null || true
    sleep 2
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    stop_group
    [[ -n $TAILPID ]] && kill "$TAILPID" 2>/dev/null
    local left
    left=$(our_gpu_pids)
    if [[ -n $left ]]; then
        log "ВНИМАНИЕ: на GPU остались наши процессы: $left — проверить вручную"
    else
        log "наших процессов на GPU нет"
    fi
    log "результаты: $RUN_DIR"
    exit $rc
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

log "user=$USER gpu=$GPU kit_gpu=$GT_KIT_GPU cpus=$GT_CPUS timeout=$TIMEOUT"
log "cmd: $SCRIPT $*"

# отдельная группа процессов, чтобы потом убить всё дерево
set -m
LD_LIBRARY_PATH="$(gt_libpath)" PYTHONNOUSERSITE=1 \
    timeout -k 60 "$TIMEOUT" nice -n 19 ionice -c3 taskset -c "$GT_CPUS" \
    "$GT_ENV/bin/python" -u "$SCRIPT" "$@" </dev/null >>"$LOG" 2>&1 &
PID=$!
set +m
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')

tail -n +1 -f "$LOG" &
TAILPID=$!

# watchdog: наш процесс на чужом GPU или завис
PEAK=0 T0=$(date +%s)
while kill -0 "$PID" 2>/dev/null; do
    elapsed=$(( $(date +%s) - T0 ))
    # на старте (инициализация рендера) проверяем каждую секунду
    if (( elapsed < 300 )); then sleep 1; else sleep 5; fi
    while read -r g p t m; do
        [[ -n $p ]] && in_our_group "$p" || continue
        if [[ $g != "$GPU" ]]; then
            log "СТОП: наш процесс $p появился на GPU $g (разрешён только $GPU), SIGKILL"
            stop_group now
            break 2
        fi
        if (( m > PEAK )); then PEAK=$m; fi
    done < <(gt_gpu_procs)
    age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    if (( age > STALL_SEC )); then
        log "СТОП: лог не обновлялся ${age} с"
        stop_group
        break
    fi
done

rc=0
wait "$PID" || rc=$?
[[ $rc -eq 124 ]] && log "СТОП: превышен лимит времени $TIMEOUT"
log "код выхода $rc, время $(( $(date +%s) - T0 )) с, пик памяти на GPU ${PEAK} MiB"
exit $rc
