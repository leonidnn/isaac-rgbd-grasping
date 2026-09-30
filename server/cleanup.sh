#!/usr/bin/env bash
# Уборка в ~/grasp_task. Без флагов только показывает, что было бы удалено.
#   bash cleanup.sh --runs        запуски из runs/ (с неподтверждённой загрузкой — только с --force)
#   bash cleanup.sh --tmp         tmp/, логи Isaac Sim, старые логи bootstrap
#   bash cleanup.sh --env --yes   окружения, Isaac Lab и miniforge (переустановка — часы и ~20 ГБ)
#   --do                          действительно удалить (иначе только список)
# Кэши шейдеров и ассетов не трогаем: их пересборка грузит CPU.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

RUNS=0 TMP=0 ENV=0 YES=0 FORCE=0 DO=0
for a in "$@"; do
    case $a in
        --runs) RUNS=1 ;;
        --tmp) TMP=1 ;;
        --env) ENV=1 ;;
        --yes) YES=1 ;;
        --force) FORCE=1 ;;
        --do) DO=1 ;;
        *) gt_die "неизвестный аргумент $a" ;;
    esac
done
(( RUNS + TMP + ENV )) || { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
pgrep -u "$USER" -f "server/run.sh" >/dev/null && gt_die "идёт запуск run.sh, сначала дождаться его"

rm_() {
    local p
    for p in "$@"; do
        [[ -e $p ]] || continue
        case $(readlink -f "$p") in "$GT_ROOT"/*) ;; *) gt_die "вне $GT_ROOT: $p" ;; esac
        echo "  $(du -sh "$p" 2>/dev/null | cut -f1)  $p"
        [[ $DO -eq 1 ]] && nice -n 19 ionice -c3 rm -rf -- "$p"
    done
}

[[ $DO -eq 1 ]] && echo "удаляю:" || echo "будет удалено (добавьте --do):"

if [[ $RUNS -eq 1 ]]; then
    for d in "$GT_RUNS"/*/; do
        [[ -d $d ]] || continue
        if [[ -d $d/ckpt && ! -f $d/UPLOAD_CONFIRMED && $FORCE -eq 0 ]]; then
            echo "  пропуск (загрузка не подтверждена, нужен --force): $d"
            continue
        fi
        rm_ "${d%/}"
    done
    rm_ "$GT_ROOT/wandb"
fi

if [[ $TMP -eq 1 ]]; then
    rm_ "$TMPDIR"/*
    rm_ "$GT_ENV"/lib/python3.10/site-packages/omni/logs
    [[ -e $HOME/.nvidia-omniverse/logs ]] && echo "  (не трогаю ~/.nvidia-omniverse/logs вне $GT_ROOT — убрать вручную в конце)"
    find "$GT_LOGS" -name 'bootstrap_*.log' -mtime +7 -print0 2>/dev/null | while IFS= read -r -d '' f; do rm_ "$f"; done
fi

if [[ $ENV -eq 1 ]]; then
    [[ $YES -eq 1 ]] || gt_die "--env удаляет окружения, нужен ещё --yes"
    rm_ "$GT_ROOT/envs" "$GT_ROOT/IsaacLab" "$GT_CONDA" "$GT_ROOT/vk_filter" "$GT_ROOT/extlib"
fi

echo "--- место"
du -sh "$GT_ROOT" 2>/dev/null
df -h "$GT_ROOT" | tail -1
