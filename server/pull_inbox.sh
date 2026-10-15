#!/usr/bin/env bash
# Локально, из Git Bash на ноуте: забрать results/_inbox с хоста к себе.
#   bash server/pull_inbox.sh
# Хост берёт из .host.env, как sync3.sh. Качает tar'ом всё, что есть, поверх локального
# (что уже лежит у меня - перезапишется тем же, удалённое на хосте у меня не удаляется).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f $ROOT/.host.env ]] || { echo "нет $ROOT/.host.env"; exit 1; }
# shellcheck disable=SC1091
source "$ROOT/.host.env"
[[ -n ${HOST_ADDR:-} && -n ${HOST_USER:-} ]] || { echo "в .host.env не заполнены HOST_ADDR/HOST_USER"; exit 1; }

HOST_SSH=(ssh -o BatchMode=yes -o LogLevel=ERROR -p "${HOST_PORT:-22}" -i "$HOST_KEY" "$HOST_USER@$HOST_ADDR")
DEST="$ROOT/results/_inbox"
mkdir -p "$DEST"

"${HOST_SSH[@]}" "cd ${HOST_DIR:-grasp_task}/results/_inbox && tar -cf - ." </dev/null | tar -xf - -C "$DEST"
echo "забрал в $DEST:"
ls "$DEST"
