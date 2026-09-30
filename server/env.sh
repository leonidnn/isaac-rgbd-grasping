# Общие пути и переменные для серверных скриптов. Подключать через source.
# Всё наше живёт в $GT_ROOT, чужие файлы в $HOME не трогаем.

GT_ROOT="${GT_ROOT:-$HOME/grasp_task}"
GT_CODE="${GT_CODE:-$GT_ROOT/code}"
GT_CONDA="$GT_ROOT/miniforge3"
GT_ENV="$GT_ROOT/envs/isaac"
GT_GLIBC="/opt/glibc/2.34"
GT_RUNS="$GT_ROOT/runs"
GT_LOGS="$GT_ROOT/logs"
# ядра для наших процессов (не больше 4)
GT_CPUS="${GT_CPUS:-44-47}"
export GT_ROOT GT_CODE GT_CONDA GT_ENV GT_GLIBC GT_RUNS GT_LOGS GT_CPUS

# кэши внутрь GT_ROOT, чтобы не раздувать ~/.cache владельца аккаунта
export XDG_CACHE_HOME="$GT_ROOT/.cache"
export XDG_DATA_HOME="$GT_ROOT/.local/share"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export PIP_NO_CACHE_DIR=1
export CONDA_PKGS_DIRS="$XDG_CACHE_HOME/conda_pkgs"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv/ComputeCache"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export WANDB_DIR="$GT_ROOT/wandb"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CONFIG_DIR="$GT_ROOT/.config/wandb"
export TMPDIR="$GT_ROOT/tmp"

# потоки CPU
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MAX_JOBS=2 MAKEFLAGS=-j2

# номера GPU как в nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Vulkan только через драйвер NVIDIA: без этого виден llvmpipe (рендер на CPU)
GT_VK_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES="$GT_VK_ICD" VK_ICD_FILENAMES="$GT_VK_ICD"

# принятие NVIDIA Omniverse EULA (иначе isaacsim ждёт ввода)
export OMNI_KIT_ACCEPT_EULA=YES

unset DISPLAY

mkdir -p "$GT_ROOT" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$TMPDIR" "$GT_RUNS" "$GT_LOGS" "$GT_ROOT/locks"

gt_die() { echo "ОШИБКА: $*" >&2; exit 1; }

# python окружения запускается с glibc 2.34 (patchelf), системные драйверы берём из /lib
gt_libpath() { echo "$GT_ENV/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"; }
gt_python() { LD_LIBRARY_PATH="$(gt_libpath)" PYTHONNOUSERSITE=1 "$GT_ENV/bin/python" "$@"; }

# все процессы на всех GPU (compute и graphics): "<gpu> <pid> <type> <MiB>"
gt_gpu_procs() {
    nvidia-smi | awk '/Processes:/ {p=1; next}
        p && $1=="|" && $2 ~ /^[0-9]+$/ && $5 ~ /^[0-9]+$/ {m=$(NF-1); sub(/MiB/, "", m); print $2, $5, $6, m}'
}

# номер карты в нумерации Vulkan (её использует рендер Isaac Sim) по номеру из nvidia-smi.
# CUDA_VISIBLE_DEVICES на Vulkan не действует, поэтому сопоставляем по UUID.
# vulkaninfo — из отдельного окружения envs/vk (conda-forge vulkan-tools)
gt_kit_gpu() {
    local uuid
    uuid=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader 2>/dev/null) || return 1
    uuid=$(tr -d ' -' <<<"${uuid#GPU-}" | tr 'A-F' 'a-f')
    [[ -x "$GT_ROOT/envs/vk/bin/vulkaninfo" && -f "$GT_VK_ICD" ]] || return 1
    "$GT_ROOT/envs/vk/bin/vulkaninfo" --summary 2>/dev/null | awk -v u="$uuid" '
        /^GPU[0-9]+:/ {g=substr($1, 4); sub(/:/, "", g)}
        /deviceUUID/ {x=$3; gsub(/-/, "", x); if (tolower(x)==u) {print g; exit}}'
}

# что появилось в $HOME после установки (сравнение со снимком)
gt_home_new() {
    [[ -f "$GT_ROOT/.home_before" ]] || return 0
    comm -13 <(sort "$GT_ROOT/.home_before") <(ls -A "$HOME" | sort)
}
