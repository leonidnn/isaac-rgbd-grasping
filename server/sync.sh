#!/usr/bin/env bash
# Синхронизация кода с сервером (запускать локально, из Git Bash).
#   bash server/sync.sh get    забрать с сервера изменённые и новые файлы (по .gitignore), удалённые там — удалить здесь
#   bash server/sync.sh pushed после git push: сервер встаёт на origin/main, несохранённая работа там остаётся
# Код пишется на сервере, коммиты и push — только отсюда (на сервере ключ только на чтение).
set -euo pipefail
HOST=proxy2.cod.phystech.edu
REMOTE='~/grasp_task/code'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH=(ssh -o BatchMode=yes -o LogLevel=ERROR "$HOST")

get() {
    local here there
    here=$(git -C "$ROOT" rev-parse HEAD)
    there=$("${SSH[@]}" "cd $REMOTE && git rev-parse HEAD")
    if [[ $here != "$there" ]]; then
        echo "ВНИМАНИЕ: коммиты разные (здесь ${here:0:7}, на сервере ${there:0:7})."
        echo "Сначала git pull здесь или sync.sh pushed, иначе изменения лягут не на тот коммит."
        exit 1
    fi
    local changed deleted
    changed=$("${SSH[@]}" "cd $REMOTE && { git diff --name-only --diff-filter=d HEAD; git ls-files -o --exclude-standard; }")
    deleted=$("${SSH[@]}" "cd $REMOTE && git diff --name-only --diff-filter=D HEAD")
    if [[ -z $changed && -z $deleted ]]; then
        echo "на сервере нет изменений"
        return
    fi
    if [[ -n $changed ]]; then
        echo "с сервера:"; sed 's/^/  /' <<<"$changed"
        "${SSH[@]}" "cd $REMOTE && { git diff --name-only -z --diff-filter=d HEAD; git ls-files -o -z --exclude-standard; } | tar --null -T - -cf -" \
            | tar -xf - -C "$ROOT"
    fi
    if [[ -n $deleted ]]; then
        echo "удалено на сервере, удаляю здесь:"; sed 's/^/  /' <<<"$deleted"
        while IFS= read -r f; do rm -f -- "$ROOT/$f"; done <<<"$deleted"
    fi
    echo; git -C "$ROOT" status --short
}

# Сервер переходит на origin/main без потери работы: файл перезаписывается, только если
# на сервере он не менялся с прошлого коммита (совпадает со старым HEAD).
pushed() {
    "${SSH[@]}" "cd $REMOTE && bash -s" <<'EOF'
set -euo pipefail
old=$(git rev-parse HEAD)
git fetch -q origin
new=$(git rev-parse origin/main)
[[ $old == "$new" ]] && { echo "сервер уже на ${new:0:7}"; git status --short; exit 0; }
git reset -q --mixed "$new"
git diff --name-only -z "$new" | while IFS= read -r -d '' f; do
    if [[ ! -e $f ]]; then
        # нет файла: добавлен в новом коммите — достаём, удалён на сервере — оставляем удалённым
        git cat-file -e "$old:$f" 2>/dev/null || git checkout -q "$new" -- "$f"
    elif [[ $(git hash-object -- "$f") == "$(git rev-parse -q --verify "$old:$f" 2>/dev/null)" ]]; then
        git checkout -q "$new" -- "$f"
    fi
done
echo "сервер: ${old:0:7} -> ${new:0:7}"
git status --short
EOF
}

case ${1:-} in
    get) get ;;
    pushed) pushed ;;
    *) sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
