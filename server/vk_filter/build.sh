#!/usr/bin/env bash
# Сборка Vulkan-слоя, который оставляет процессу только одну GPU (по UUID).
# Результат: $GT_ROOT/vk_filter/{libgt_gpu_filter.so,gt_gpu_filter.json,gt_vk_list}
# gt_vk_list — список карт без vkCreateDevice (vulkaninfo открывает контекст на каждой GPU)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"

INC="$GT_ROOT/envs/vk/include"
[[ -f $INC/vulkan/vk_layer.h ]] || gt_die "нет заголовков Vulkan в $INC (conda install -p $GT_ROOT/envs/vk libvulkan-headers)"
command -v gcc >/dev/null || gt_die "нет gcc"

mkdir -p "$GT_VK_LAYER_DIR"
LIB="$GT_ROOT/envs/vk/lib"
nice -n 19 taskset -c "$GT_CPUS" gcc -O2 -Wall -shared -fPIC -fvisibility=hidden -I"$INC" \
    "$HERE/gt_gpu_filter.c" -o "$GT_VK_LAYER_DIR/libgt_gpu_filter.so" -lpthread
nice -n 19 taskset -c "$GT_CPUS" gcc -O2 -Wall -I"$INC" "$HERE/gt_vk_list.c" \
    -o "$GT_VK_LAYER_DIR/gt_vk_list" -L"$LIB" -Wl,-rpath,"$LIB" -lvulkan
cp "$HERE/gt_gpu_filter.json" "$GT_VK_LAYER_DIR/"
echo "собрано: $GT_VK_LAYER_DIR"
