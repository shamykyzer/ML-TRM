#!/usr/bin/env bash
# Sync C:/ml-trm-work/checkpoints to use/  to the team's Google Drive folder.
#
# Folder ID: 17N5HgyiL-CLH2w_31wSr7Xgu2OcwFRZk
# URL:       https://drive.google.com/drive/folders/17N5HgyiL-CLH2w_31wSr7Xgu2OcwFRZk
#
# One-time setup (interactive, requires a browser):
#   rclone config
#     n) New remote
#     name> gdrive
#     storage> drive            # pick "drive" / Google Drive
#     client_id>  (leave blank to use rclone's built-in)
#     scope> drive
#     root_folder_id> 17N5HgyiL-CLH2w_31wSr7Xgu2OcwFRZk
#     # Then complete OAuth in browser; rclone writes the token.
#
# Routine sync (idempotent — only uploads changed/new files):
#   bash scripts/sync_to_gdrive.sh                    # full sync of curated folder
#   bash scripts/sync_to_gdrive.sh "machine 4"        # one machine subfolder
#   bash scripts/sync_to_gdrive.sh --dry-run          # preview without uploading
#
# Tag for findings.md §5: "drive sync 2026-MM-DDTHHMM"

set -u

REMOTE="${RCLONE_REMOTE:-gdrive}"
# The Drive folder (root_folder_id 17N5HgyiL-CLH2w_31wSr7Xgu2OcwFRZk) is the
# M4 machine view — its top level holds the 01_*..09_* numbered subfolders
# directly. So source defaults to the local machine 4/ folder, not the
# parent "checkpoints to use/". Override with TRM_SYNC_SOURCE if a
# different machine N folder needs to be pushed instead.
SOURCE="${TRM_SYNC_SOURCE:-C:/ml-trm-work/checkpoints to use/machine 4}"
DEST_PATH="${REMOTE}:"     # remote's root_folder_id maps to the Drive folder

if ! command -v rclone >/dev/null 2>&1; then
    RCLONE_BIN="C:/Users/amm-alshamy/AppData/Local/Microsoft/WinGet/Packages/Rclone.Rclone_Microsoft.Winget.Source_8wekyb3d8bbwe/rclone-v1.73.5-windows-amd64/rclone.exe"
    if [ -x "$RCLONE_BIN" ]; then
        alias rclone="$RCLONE_BIN"
        # `alias` doesn't survive `command -v` checks; export RCLONE so subshells see it
        RCLONE="$RCLONE_BIN"
    else
        echo "rclone not on PATH and fallback path not found. Install rclone or set PATH." >&2
        exit 1
    fi
else
    RCLONE=rclone
fi

if ! "$RCLONE" listremotes 2>/dev/null | grep -q "^${REMOTE}:"; then
    echo "Remote '${REMOTE}' not configured. Run: rclone config" >&2
    echo "See top of this script for one-time setup steps." >&2
    exit 2
fi

SUBPATH="${1:-}"
DRY_RUN=""
if [ "${SUBPATH}" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
    SUBPATH=""
fi

if [ -n "$SUBPATH" ]; then
    LOCAL="$SOURCE/$SUBPATH"
    REMOTE_DEST="$DEST_PATH$SUBPATH"
else
    LOCAL="$SOURCE"
    REMOTE_DEST="$DEST_PATH"
fi

if [ ! -d "$LOCAL" ]; then
    echo "Local source does not exist: $LOCAL" >&2
    exit 3
fi

ts=$(date +%Y-%m-%dT%H%M)
echo "[gdrive-sync] $ts  $LOCAL  ->  $REMOTE_DEST  ${DRY_RUN}"
"$RCLONE" sync "$LOCAL" "$REMOTE_DEST" \
    --progress \
    --transfers 4 \
    --checkers 8 \
    --create-empty-src-dirs \
    --exclude '.aggregator_state.json' \
    --exclude '*.tmp' \
    $DRY_RUN

echo "[gdrive-sync] $ts  done"
