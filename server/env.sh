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
# W&B с сервера недоступен (403 по региону), метрики пишем на HF; wandb, если его кто-то импортирует, выключен
export WANDB_MODE=disabled WANDB_DIR="$GT_ROOT/wandb" WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb" WANDB_CONFIG_DIR="$GT_ROOT/.config/wandb"
export GT_HF_REPO=leonid-nn/grasp-rgbd-checkpoints
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
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
# extlib — отдельные библиотеки, которых нет в системе (libGLU.so.1 из envs/vk для iray/MDL),
# ссылками по одной, чтобы не подмешивать в Isaac Sim весь envs/vk
gt_libpath() { echo "$GT_ENV/lib:$GT_ROOT/extlib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"; }
gt_python() { LD_LIBRARY_PATH="$(gt_libpath)" PYTHONNOUSERSITE=1 "$GT_ENV/bin/python" "$@"; }

# все процессы на всех GPU (compute и graphics): "<gpu> <pid> <type> <MiB>"
gt_gpu_procs() {
    nvidia-smi | awk '/Processes:/ {p=1; next}
        p && $1=="|" && $2 ~ /^[0-9]+$/ && $5 ~ /^[0-9]+$/ {m=$(NF-1); sub(/MiB/, "", m); print $2, $5, $6, m}'
}

# Isaac Sim при старте открывает все карты, которые видит Vulkan, а CUDA_VISIBLE_DEVICES
# на Vulkan не действует. Поэтому свой слой (server/vk_filter) оставляет процессу одну карту по UUID.
# заголовки и loader Vulkan — из отдельного окружения envs/vk (conda-forge)
GT_VK_LAYER_DIR="$GT_ROOT/vk_filter"
GT_VK_LAYER=VK_LAYER_GT_gpu_filter
# implicit-слой (Isaac Sim обнуляет VK_INSTANCE_LAYERS), включается переменной GT_VK_FILTER=1
GT_VK_IMPLICIT_DIR="$XDG_DATA_HOME/vulkan/implicit_layer.d"

gt_vk_uuid() {
    local u
    u=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader 2>/dev/null) || return 1
    tr -d ' -' <<<"${u#GPU-}" | tr 'A-F' 'a-f'
}

# UUID карт, которые видит Vulkan с нашим слоем (GT_VK_UUID должен быть задан).
# Только gt_vk_list: vulkaninfo создаёт logical device на каждой карте, т.е. лезет на чужие
gt_vk_visible() {
    [[ -x "$GT_VK_LAYER_DIR/gt_vk_list" && -f "$GT_VK_LAYER_DIR/libgt_gpu_filter.so" \
        && -f "$GT_VK_IMPLICIT_DIR/gt_gpu_filter.json" ]] || return 1
    GT_VK_FILTER=1 "$GT_VK_LAYER_DIR/gt_vk_list" 2>/dev/null | awk '{print $1}'
}

# токен HF (положен пользователем, chmod 600) — только в окружение запускаемого процесса,
# никогда не печатать
gt_load_secrets() {
    local d="$GT_ROOT/.config"
    [[ -r $d/hf_token ]] || return 1
    HF_TOKEN=$(<"$d/hf_token")
    export HF_TOKEN
}

# что появилось в $HOME после установки (сравнение со снимком)
gt_home_new() {
    [[ -f "$GT_ROOT/.home_before" ]] || return 0
    comm -13 <(sort "$GT_ROOT/.home_before") <(ls -A "$HOME" | sort)
}
