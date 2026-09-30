#!/usr/bin/env bash
# Установка окружения в ~/grasp_task (один раз, внутри tmux).
#   bash bootstrap.sh             Miniforge, env, torch, Isaac Sim 4.5, проверки
#   bash bootstrap.sh --isaaclab  то же + Isaac Lab 2.0 и skrl
#   bash bootstrap.sh --check     только проверки
# Повторный запуск пропускает уже сделанные шаги. GPU не используется.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

# весь скрипт под низким приоритетом CPU/диска и на ограниченных ядрах
if [[ -z "${GT_NICED:-}" ]]; then
    export GT_NICED=1
    exec nice -n 19 ionice -c3 taskset -c "$GT_CPUS" bash "$0" "$@"
fi

WITH_LAB=0 CHECK_ONLY=0 NO_TMUX=0
for a in "$@"; do
    case $a in
        --isaaclab) WITH_LAB=1 ;;
        --check) CHECK_ONLY=1 ;;
        --no-tmux) NO_TMUX=1 ;;
        *) gt_die "неизвестный аргумент $a" ;;
    esac
done
NEED_GB=${GT_NEED_GB:-45}
ISAACLAB_TAG=v2.0.2

[[ $(id -u) -ne 0 ]] || gt_die "не запускать от root"
[[ -e "$GT_GLIBC/lib/libc.so.6" ]] || gt_die "нет $GT_GLIBC"
if [[ -z "${TMUX:-}" && $NO_TMUX -eq 0 && $CHECK_ONLY -eq 0 ]]; then
    gt_die "запускать внутри tmux: tmux new -s gt-setup"
fi

LOG="$GT_LOGS/bootstrap_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date +%T)] bootstrap, лог $LOG, ядра $GT_CPUS"

[[ -f "$GT_ROOT/.home_before" ]] || ls -A "$HOME" > "$GT_ROOT/.home_before"

pip_i() { gt_python -m pip install --no-cache-dir --progress-bar off "$@"; }
have_isaacsim() { [[ -x "$GT_ENV/bin/python" ]] && gt_python -m pip show isaacsim >/dev/null 2>&1; }

run_checks() {
    echo "--- glibc"
    gt_python -c "import os; print(os.confstr('CS_GNU_LIBC_VERSION'))"
    echo "--- системные библиотеки драйвера"
    # libGLX_nvidia/libEGL_nvidia напрямую вместе не грузить: падают при выходе (double free),
    # их подгружает libglvnd через libGL/libEGL
    gt_python -c "
import ctypes
for n in ('libcuda.so.1', 'libvulkan.so.1', 'libGL.so.1', 'libEGL.so.1'):
    ctypes.CDLL(n)
print('ok')"
    if gt_python -c "import torch" 2>/dev/null; then
        echo "--- torch"
        gt_python -c "import torch; print(torch.__version__, 'cuda', torch.version.cuda)"
    fi
    if have_isaacsim; then
        echo "--- import isaacsim"
        gt_python -c "import isaacsim; print('ok')"

        local sp
        sp=$(gt_python -c "import site; print(site.getsitepackages()[0])")
        echo "--- .so, которым нужна glibc новее 2.34 (должно быть пусто)"
        find "$sp" -name '*.so*' -type f -print0 \
            | xargs -0 -r grep -l -a -E 'GLIBC_2\.(3[5-9]|[4-9][0-9])' 2>/dev/null | head -20 || true

        echo "--- зависимости .so, которых нет ни в системе, ни в окружении (должно быть пусто)"
        find "$sp" -name '*.so*' -type f -printf '%f\n' | sort -u > "$TMPDIR/have_so.txt"
        find "$sp"/isaacsim* "$sp"/omni* -name '*.so*' -type f -print0 2>/dev/null \
            | LD_LIBRARY_PATH="$(gt_libpath)" xargs -0 -r ldd 2>/dev/null \
            | awk '/=> not found/ {print $1}' | sort -u > "$TMPDIR/notfound_so.txt" || true
        comm -23 "$TMPDIR/notfound_so.txt" "$TMPDIR/have_so.txt"
    fi
    echo "--- место"
    du -sh "$GT_ROOT"
    df -h "$GT_ROOT" | tail -1
    echo "--- новое в \$HOME вне grasp_task"
    gt_home_new | grep -vx grasp_task || true
}

if [[ $CHECK_ONLY -eq 1 ]]; then run_checks; exit 0; fi

if ! have_isaacsim; then
    avail_gb=$(( $(df -Pk "$GT_ROOT" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
    (( avail_gb >= NEED_GB )) || gt_die "свободно ${avail_gb} ГБ, нужно ${NEED_GB}"
fi

if [[ ! -x "$GT_CONDA/bin/conda" ]]; then
    echo "[$(date +%T)] Miniforge"
    wget -q -O "$TMPDIR/miniforge.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$TMPDIR/miniforge.sh" -b -p "$GT_CONDA"
    rm -f "$TMPDIR/miniforge.sh"
fi

if [[ ! -x "$GT_ENV/bin/python" ]]; then
    echo "[$(date +%T)] conda env"
    "$GT_CONDA/bin/conda" create -y -p "$GT_ENV" python=3.10 patchelf
fi

PYBIN=$(readlink -f "$GT_ENV/bin/python")
if [[ $("$GT_ENV/bin/patchelf" --print-interpreter "$PYBIN") != "$GT_GLIBC/lib/ld-linux-x86-64.so.2" ]]; then
    echo "[$(date +%T)] patchelf python -> glibc 2.34"
    cp -n "$PYBIN" "$PYBIN.orig"
    "$GT_ENV/bin/patchelf" --set-interpreter "$GT_GLIBC/lib/ld-linux-x86-64.so.2" \
        --force-rpath --set-rpath "$GT_GLIBC/lib:$GT_ENV/lib" "$PYBIN"
fi
[[ $(gt_python -c "import os; print(os.confstr('CS_GNU_LIBC_VERSION'))") == "glibc 2.34" ]] \
    || gt_die "python не видит glibc 2.34"

if ! gt_python -c "import torch" 2>/dev/null; then
    echo "[$(date +%T)] torch"
    pip_i torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
fi

if ! have_isaacsim; then
    echo "[$(date +%T)] isaacsim 4.5"
    pip_i "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
fi

if [[ $WITH_LAB -eq 1 ]]; then
    LAB="$GT_ROOT/IsaacLab"
    if [[ ! -d "$LAB" ]]; then
        echo "[$(date +%T)] Isaac Lab $ISAACLAB_TAG"
        git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$LAB"
    fi
    for p in isaaclab isaaclab_assets isaaclab_tasks; do
        pip_i -e "$LAB/source/$p"
    done
    pip_i -e "$LAB/source/isaaclab_rl[skrl]"
fi

echo "[$(date +%T)] проверки"
run_checks
echo "[$(date +%T)] готово"
