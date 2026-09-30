#!/usr/bin/env bash
# Какие GPU есть, чем заняты и кем.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

procs=$(gt_gpu_procs)
while IFS=, read -r idx name used total util; do
    idx=${idx// /}; used=${used// /}; total=${total// /}; util=${util// /}; name=${name# }
    printf "GPU%-2s %-26s %6s / %6s MiB  %3s%%\n" "$idx" "$name" "$used" "$total" "$util"
    awk -v g="$idx" '$1==g' <<<"$procs" | while read -r _ pid type mem; do
        printf "      pid %-8s %-3s %6s MiB  %s\n" "$pid" "$type" "$mem" "$(ps -o user=,etime= -p "$pid" 2>/dev/null)"
    done
done < <(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
