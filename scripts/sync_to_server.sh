#!/usr/bin/env bash
# Sync training code + raw demos to the GPU server, build the env, pull models back.
#
# Datasets and model checkpoints are NOT tracked in git (see .gitignore); this
# rsync-based script moves them. The full data pipeline (convert -> train) runs
# ON THE SERVER; locally we only test code and run real-arm eval.
#
#   ssh alias  : jiagpu8_yz        (HostName 192.168.40.47)
#   work root  : /nvmessd/yinzi/ppl_reBot          <- env lives here (.venv etc.)
#   code+demos : /nvmessd/yinzi/ppl_reBot/single_reBot
#
# Everything created on the server stays UNDER the work root (uv binary, cache,
# managed python, venv) — nothing is written to $HOME outside it.
#
# Layout on the server:
#   /nvmessd/yinzi/ppl_reBot/
#   ├── .bin/uv  .uv-cache/  .uv-python/  .venv/   (env, built by setup-remote)
#   └── single_reBot/                              (this repo)
#       ├── scripts/ arm/ configs/ deploy/requirements-dp.txt
#       └── outputs/{demos,lerobot_datasets,training_runs}/
#
# Usage:
#   scripts/sync_to_server.sh push          # code + raw demos -> server
#   scripts/sync_to_server.sh push-code     # code only
#   scripts/sync_to_server.sh push-demos    # raw demos only (outputs/demos/)
#   scripts/sync_to_server.sh push-data     # converted LeRobot datasets (rarely needed)
#   scripts/sync_to_server.sh pull-models   # training_runs (checkpoints) <- server
#   scripts/sync_to_server.sh setup-remote  # install uv (contained) + build .venv
#   scripts/sync_to_server.sh verify-env    # run torch.cuda check on the server
#   scripts/sync_to_server.sh check         # test ssh + show remote layout
#
# Transfers resume on failure: --partial-dir keeps half-sent files, and each
# rsync is retried up to 5 times (re-running any push/pull also just resumes).
# Extra flags after the subcommand pass through to rsync, e.g. `push --dry-run`.
set -euo pipefail

REMOTE=jiagpu8_yz
ROOT=/nvmessd/yinzi/ppl_reBot          # work root (env here)
CODE=$ROOT/single_reBot                # code + demos here
PYVER=3.11.11
CU_INDEX=https://download.pytorch.org/whl/cu128   # fall back to .../cu126 if cu128 fails
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC_OPTS=(-ahz --partial-dir=.rsync-partial --info=progress2)
CODE_EXCLUDES=(
  --exclude '.git/'      --exclude '.venv/'        --exclude '__pycache__/'
  --exclude '*.pyc'      --exclude '*.egg-info/'   --exclude '.pytest_cache/'
  --exclude '.ruff_cache/' --exclude '.DS_Store'   --exclude 'outputs/'
)

# rsync with auto-resume retry (rsync is idempotent; partial files resume).
rsync_retry() {
  local n=0 max=5
  until rsync "${RSYNC_OPTS[@]}" "$@"; do
    n=$((n + 1))
    [ "$n" -ge "$max" ] && { echo "rsync failed after $max attempts" >&2; return 1; }
    echo ">> rsync interrupted (attempt $n/$max) — resuming in 5s..." >&2
    sleep 5
  done
}

cmd="${1:-}"; shift || true
EXTRA=("$@")

case "$cmd" in
  check)
    ssh "$REMOTE" "echo OK on \$(hostname); echo; echo 'work root:'; ls -la '$ROOT' 2>/dev/null || echo '  (missing)'"
    ;;

  push-code)
    ssh "$REMOTE" "mkdir -p '$CODE'"
    rsync_retry "${CODE_EXCLUDES[@]}" "${EXTRA[@]}" "$REPO_ROOT"/ "$REMOTE:$CODE/"
    ;;

  push-demos)
    ssh "$REMOTE" "mkdir -p '$CODE/outputs/demos'"
    rsync_retry "${EXTRA[@]}" "$REPO_ROOT"/outputs/demos/ "$REMOTE:$CODE/outputs/demos/"
    ;;

  push-data)
    ssh "$REMOTE" "mkdir -p '$CODE/outputs/lerobot_datasets'"
    rsync_retry "${EXTRA[@]}" "$REPO_ROOT"/outputs/lerobot_datasets/ "$REMOTE:$CODE/outputs/lerobot_datasets/"
    ;;

  push)
    "$0" push-code "${EXTRA[@]}"
    "$0" push-demos "${EXTRA[@]}"
    ;;

  pull-models)
    mkdir -p "$REPO_ROOT/outputs/training_runs"
    rsync_retry "${EXTRA[@]}" "$REMOTE:$CODE/outputs/training_runs/" "$REPO_ROOT"/outputs/training_runs/
    ;;

  setup-remote)
    # Install uv + build the venv, fully contained under the work root.
    ssh "$REMOTE" "set -e
      export UV_CACHE_DIR='$ROOT/.uv-cache'
      export UV_PYTHON_INSTALL_DIR='$ROOT/.uv-python'
      if [ ! -x '$ROOT/.bin/uv' ]; then
        echo 'installing uv into $ROOT/.bin ...'
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR='$ROOT/.bin' INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null 2>&1
      fi
      UV='$ROOT/.bin/uv'
      \$UV venv '$ROOT/.venv' --python $PYVER
      \$UV pip install --python '$ROOT/.venv/bin/python' \
        -r '$CODE/deploy/requirements-dp.txt' \
        --extra-index-url $CU_INDEX --index-strategy unsafe-best-match
      echo 'env built at $ROOT/.venv'
    "
    "$0" verify-env
    ;;

  verify-env)
    ssh "$REMOTE" "'$ROOT/.venv/bin/python' -c '
import torch
print(\"torch\", torch.__version__)
print(\"cuda available:\", torch.cuda.is_available())
print(\"devices:\", torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")
x = torch.zeros(1, device=\"cuda\") + 1; torch.cuda.synchronize()
print(\"gpu kernel ok:\", x.item() == 1.0)
'"
    ;;

  ""|-h|--help|help)
    sed -n '1,40p' "$0" | sed 's/^# \{0,1\}//'
    ;;

  *)
    echo "unknown subcommand: $cmd" >&2
    echo "try: push | push-code | push-demos | push-data | pull-models | setup-remote | verify-env | check" >&2
    exit 2
    ;;
esac
