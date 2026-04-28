#!/usr/bin/env bash
# Drive sync watchdog for M6 — mirrors C:/ml-trm-work/checkpoints to use/machine 6/
# to a specific Google Drive folder via rclone, every $INTERVAL seconds.
#
# Prerequisite (one-time, interactive — needs a browser for OAuth):
#   rclone config
#   # n) New remote -> name: gdrive -> 13 (drive) -> client_id/secret blank
#   # -> scope: 1 (full access) -> root_folder_id BLANK -> service_account_file BLANK
#   # -> Edit advanced: n -> Use auto config: y -> browser opens, log in, allow
#   # -> Configure as Shared Drive: n -> y) Yes this is OK -> q to quit
#
# Usage:
#   bash scripts/drive_sync_watchdog.sh                       # default 30 min loop
#   INTERVAL=600 bash scripts/drive_sync_watchdog.sh          # 10 min loop
#   bash scripts/drive_sync_watchdog.sh --once                # one-shot, no loop
#
# The Drive folder ID below points at the operator's M6 redundancy folder:
#   https://drive.google.com/drive/folders/18EXQL5h6MF5i8RbB4Zb97oU7wO9LXlbP

set -u

REMOTE_NAME="${RCLONE_REMOTE:-gdrive}"
DRIVE_FOLDER_ID="${DRIVE_FOLDER_ID:-18EXQL5h6MF5i8RbB4Zb97oU7wO9LXlbP}"
LOCAL_DIR="${LOCAL_DIR:-C:/ml-trm-work/checkpoints to use/machine 6}"
INTERVAL="${INTERVAL:-1800}"
RCLONE_BIN="${RCLONE_BIN:-$HOME/bin/rclone.exe}"

if [ ! -x "$RCLONE_BIN" ]; then
    if command -v rclone >/dev/null 2>&1; then
        RCLONE_BIN="$(command -v rclone)"
    else
        echo "[drive-sync] ERROR: rclone not found at $RCLONE_BIN and not in PATH" >&2
        exit 1
    fi
fi

# Sanity-check that the remote was configured. Bail with instructions if not.
if ! "$RCLONE_BIN" listremotes 2>/dev/null | grep -q "^${REMOTE_NAME}:$"; then
    cat >&2 <<EOF
[drive-sync] ERROR: rclone remote "${REMOTE_NAME}:" not found.
Run \`rclone config\` once to add it (interactive — opens a browser for OAuth).
Or set RCLONE_REMOTE=<your_remote_name> to use a different remote name.
EOF
    exit 1
fi

push_once() {
    echo "[drive-sync] $(date -Is) syncing $LOCAL_DIR -> ${REMOTE_NAME}:/ (folder id ${DRIVE_FOLDER_ID})"
    # `copy` (not `sync`) so files in Drive aren't deleted if they ever leave the
    # local folder. Aggregator scripts on M4's side may add files to Drive too.
    "$RCLONE_BIN" copy "$LOCAL_DIR" "${REMOTE_NAME}:" \
        --drive-root-folder-id "$DRIVE_FOLDER_ID" \
        --transfers 4 \
        --checkers 8 \
        --progress=false \
        --stats=0 \
        --update \
        --include "*.pt" \
        --include "*.csv" \
        --include "*.json" \
        --include "*.md" \
        --include "*.txt" \
        2>&1
    echo "[drive-sync] $(date -Is) sync complete (rc=$?)"
}

if [ "${1:-}" = "--once" ]; then
    push_once
    exit $?
fi

while true; do
    push_once || true
    sleep "$INTERVAL"
done
