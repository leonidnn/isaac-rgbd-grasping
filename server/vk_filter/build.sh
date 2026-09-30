#!/usr/bin/env bash
# Сборка Vulkan-слоя, который оставляет процессу только одну GPU (по UUID).
# Результат: $GT_ROOT/vk_filter/{libgt_gpu_filter.so,gt_gpu_filter.json}
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"

INC="$GT_ROOT/envs/vk/include"
[[ -f $INC/vulkan/vk_layer.h ]] || gt_die "нет заголовков Vulkan в $INC (conda install -p $GT_ROOT/envs/vk libvulkan-headers)"
command -v gcc >/dev/null || gt_die "нет gcc"

mkdir -p "$GT_VK_LAYER_DIR"
nice -n 19 gcc -O2 -Wall -shared -fPIC -fvisibility=hidden -I"$INC" \
    "$HERE/gt_gpu_filter.c" -o "$GT_VK_LAYER_DIR/libgt_gpu_filter.so" -lpthread
cp "$HERE/gt_gpu_filter.json" "$GT_VK_LAYER_DIR/"
echo "собрано: $GT_VK_LAYER_DIR"
