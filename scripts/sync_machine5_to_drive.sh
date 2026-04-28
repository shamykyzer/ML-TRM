#!/usr/bin/env bash
# Mirror C:/ml-trm-work/checkpoints to use/Machine 5/ → Google Drive folder
# https://drive.google.com/drive/folders/136RKcCjyouricNYxYqXEQhLXXYLxZWZ4
# (the "machine 5" subfolder of "TRM-ML .chk", owned by shamyxor@gmail.com).
#
# Designed to be safe to call from cron, the watchdog post-snapshot hook,
# or manually after a run finishes. Idempotent — only uploads files that
# changed/are new.
#
# Backend selection (first available wins):
#   1. rclone   — preferred. After `rclone config` (one-time, picks
#      "drive" backend, browser auth as remote `gdrive`), this script just
#      runs `rclone sync`. Installs via:  winget install Rclone.Rclone
#   2. gdrive   — fallback. Less ergonomic for sync-style updates.
#   3. (none)   — script writes a manifest of files needing manual upload
#      to MANIFEST.txt in the source dir and exits 2. Drag the named files
#      into the Drive folder via the web UI as a fallback.
#
# Usage:
#   bash scripts/sync_machine5_to_drive.sh                # mirror everything
#   bash scripts/sync_machine5_to_drive.sh --dry-run      # show what would change
#
# One-time rclone setup (run once, then every invocation here is hands-off):
#   winget install Rclone.Rclone
#   rclone config
#     n  -> new remote
#     gdrive  -> name
#     drive   -> backend
#     (defaults; pick your account; no team drive)
#   # Verify:
#   rclone lsd gdrive:
#
# Then this script will sync silently on every call. Wire into the
# watchdog by appending the snapshot loop body:
#   bash scripts/sync_machine5_to_drive.sh >> /tmp/m5_drive_sync.log 2>&1 || true

set -u

SRC="/c/ml-trm-work/checkpoints to use/Machine 5"
DRIVE_FOLDER_ID="136RKcCjyouricNYxYqXEQhLXXYLxZWZ4"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"     # override with env if user named it differently
RCLONE_DEST_PATH="${RCLONE_DEST_PATH:-TRM-ML .chk/machine 5}"
DRY_RUN=""
[ "${1:-}" = "--dry-run" ] && DRY_RUN="--dry-run"

if [ ! -d "$SRC" ]; then
  echo "[sync] ERROR: source missing: $SRC" >&2
  exit 1
fi

# Backend 1: rclone
if command -v rclone >/dev/null 2>&1; then
  echo "[sync] backend = rclone (remote: $RCLONE_REMOTE)"
  echo "[sync] $SRC  →  $RCLONE_REMOTE:$RCLONE_DEST_PATH"
  rclone sync "$SRC" "$RCLONE_REMOTE:$RCLONE_DEST_PATH" \
    --progress --transfers 8 --checkers 16 \
    --exclude '__pycache__/**' --exclude '*.tmp' \
    $DRY_RUN
  exit $?
fi

# Backend 2: gdrive (less ergonomic; falls back to a manifest)
if command -v gdrive >/dev/null 2>&1; then
  echo "[sync] backend = gdrive (drive folder $DRIVE_FOLDER_ID)"
  echo "[sync] WARN: gdrive lacks rclone's diff-and-sync semantics."
  echo "[sync]       Will only upload files modified in the last 24 h."
  find "$SRC" -type f -mtime -1 -not -path '*__pycache__*' -print0 | while IFS= read -r -d '' f ; do
    rel=${f#"$SRC"/}
    echo "[sync] uploading: $rel"
    [ -z "$DRY_RUN" ] && gdrive files upload --parent "$DRIVE_FOLDER_ID" "$f" 2>&1 | tail -3
  done
  exit 0
fi

# Backend 3: no CLI installed → write a manifest the user can drop into Drive web UI.
MANIFEST="$SRC/MANIFEST.txt"
echo "[sync] no CLI backend installed (rclone/gdrive). Writing manifest." >&2
{
  echo "# Manual Drive upload manifest"
  echo "# Generated $(date -Is)"
  echo "# Drop these files into:"
  echo "#   https://drive.google.com/drive/folders/$DRIVE_FOLDER_ID"
  echo "# (Drive web UI accepts drag-drop of folders and preserves structure.)"
  echo ""
  find "$SRC" -type f -not -path '*__pycache__*' -not -name 'MANIFEST.txt' -printf '%p  %s bytes  %TY-%Tm-%TdT%TH:%TM\n'
} > "$MANIFEST"
echo "[sync] manifest: $MANIFEST"
echo "[sync] To enable autosync: 'winget install Rclone.Rclone' then 'rclone config'."
exit 2
