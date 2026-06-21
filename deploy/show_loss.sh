#!/usr/bin/env bash
# Extract step:N loss:X grdn:Y from each Test #2 wrap.log.
# lerobot writes INFO loss lines without a trailing \n (tqdm uses \r to
# overwrite progress on the same line), so plain grep against the file
# misses them. We pipe through `tr '\r' '\n'` first.
#
# Usage (run on jiagpu8_yz from anywhere):
#   bash deploy/show_loss.sh                     # all 6 runs, last 3 entries each
#   bash deploy/show_loss.sh DP_A                # one run, full history
#   bash deploy/show_loss.sh DP_A 10             # last 10 entries

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_DIR="$REPO_ROOT/outputs/training_runs"

RUNS=("DP_A" "DP_B" "DP_C" "DP_D" "PI05" "PI0FAST" "SMOLVLA")

ONLY="${1:-}"
N="${2:-3}"

for run in "${RUNS[@]}"; do
    [ -n "$ONLY" ] && [ "$ONLY" != "$run" ] && continue
    wrap="$TRAIN_DIR/test2_${run}_wrap.log"
    if [ ! -f "$wrap" ]; then
        echo "=== $run: NO LOG ==="
        continue
    fi
    echo "=== $run ==="
    if [ -n "$ONLY" ] && [ "$2" = "" ]; then
        # full history for single-run query
        tr '\r' '\n' < "$wrap" 2>/dev/null \
          | grep -oE "step:[0-9KM]+ smpl:[0-9KM]+ ep:[0-9KM]+ epch:[0-9.]+ loss:[0-9.]+ grdn:[0-9.]+ lr:[0-9.e+-]+" \
          | column -t -s' '
    else
        tr '\r' '\n' < "$wrap" 2>/dev/null \
          | grep -oE "step:[0-9KM]+ smpl:[0-9KM]+ ep:[0-9KM]+ epch:[0-9.]+ loss:[0-9.]+ grdn:[0-9.]+ lr:[0-9.e+-]+" \
          | tail -n "$N" \
          | column -t -s' '
    fi
done
