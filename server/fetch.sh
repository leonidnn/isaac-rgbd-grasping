#!/usr/bin/env bash
# Локально, из Git Bash: что с очередью wait_run.sh на сервере.
#   bash server/fetch.sh <имя> [папка]
# Ещё ждёт или идёт — печатает статус и хвост лога текущего запуска.
# Кончилось — забирает картинки, гифки и логи каждого шага в <папка>/<запуск>/ (по умолчанию results/_inbox/<имя>).
set -euo pipefail
HOST=proxy2.cod.phystech.edu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH=(ssh -o BatchMode=yes -o LogLevel=ERROR -o ConnectTimeout=15 "$HOST")

# в PowerShell `bash` — это WSL, а из WSL до сервера сеть не ходит, ssh просто висит
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Это WSL, отсюда сервер недоступен. Запускать из Git Bash:"
    echo "  & \"C:\\Program Files\\Git\\bin\\bash.exe\" server/fetch.sh $*"
    exit 1
fi

NAME=${1:-}
[[ $NAME =~ ^[A-Za-z0-9_-]+$ ]] || { sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
DEST=${2:-$ROOT/results/_inbox/$NAME}
Q="grasp_task/queue/$NAME"

status=$("${SSH[@]}" "cat $Q/status 2>/dev/null") || { echo "очереди $NAME на сервере нет"; exit 1; }
state=$(sed -n 's/^state=//p' <<<"$status")
echo "$status"
echo

steps=$("${SSH[@]}" "cat $Q/steps")
[[ -n $steps ]] && { echo "готовые шаги (шаг, код выхода, папка):"; sed 's/^/  /' <<<"$steps"; echo; }

case $state in
    waiting)
        echo "ждёт свободную карту. Сейчас на 2080 Ti:"
        "${SSH[@]}" "bash grasp_task/code/server/gpus.sh 2>&1 | grep -A3 -E 'GPU(1|7) ' | grep -v '^GPU[02-6]'"
        ;;
    running)
        echo "идёт, хвост лога:"
        "${SSH[@]}" "d=\$(ls -td grasp_task/runs/${NAME}_* | head -1); echo \"  \$d\"; grep -vE 'Warning|ext:|^\s*\$' \$d/run.log | tail -8"
        ;;
    expired)
        echo "не дождался карты, ничего не запускалось"
        ;;
    done)
        mkdir -p "$DEST"
        while read -r _ rc dir; do
            [[ -n $dir ]] || continue
            sub="$DEST/$(basename "$dir")"
            mkdir -p "$sub"
            # только то, что нужно в results: кадры, гифки, лог (без .npy)
            "${SSH[@]}" "cd $dir && tar -cf - \$(ls *.png *.gif run.log 2>/dev/null)" | tar -xf - -C "$sub"
            echo "код $rc -> $sub: $(ls "$sub" | tr '\n' ' ')"
            grep -E 'PASS|FAIL|object rise|СТОП' "$sub/run.log" | sed 's/^/    /' || true
        done <<<"$steps"
        echo
        echo "Готово, лежит в $DEST"
        ;;
esac
