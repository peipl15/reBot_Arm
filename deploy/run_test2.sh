#!/usr/bin/env bash
# deploy/run_test2.sh — launch Test #2 runs in parallel on jiagpu8_yz (8x A40).
#
# Expects to be run from /nvmessd/yinzi/ppl_reBot/single_reBot/  on the server,
# after `scripts/sync_to_server.sh push` from local. Reads its repo dir from
# its own location, so you can run it from anywhere.
#
# Test #2 hypotheses (see EXPERIMENTS.md):
#   H1: crop_shape=(84,84) is the precision bottleneck   -> no-crop runs
#   H2: ResNet34 helps over ResNet18                     -> A vs B
#   H3: shorter horizon / faster replan is smoother      -> all runs use h=8, n_act=2
#   H4: wrist camera adds noise, not info                -> C is true single-cam
#
# Hardware: 8x A40 48 GB. Each run pinned to one GPU via CUDA_VISIBLE_DEVICES.
#
# Usage:
#   bash deploy/run_test2.sh                      # launch all DP runs in background
#   bash deploy/run_test2.sh --dry-run            # print commands only
#   bash deploy/run_test2.sh --only DP_A,DP_C     # subset
#   tail -f outputs/training_runs/*_train.log     # watch progress

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# venv path differs between local (rebot_arm/.venv) and server. On the
# server we now default to .venv_le05 (py3.12 + lerobot 0.5 + streaming
# video encoding); the older .venv (lerobot 0.3) is kept as fallback for
# evaluating Test #1 ckpts.
DEFAULT_SERVER_VENV="$(dirname "$REPO_ROOT")/.venv_le05"
[ -x "$DEFAULT_SERVER_VENV/bin/python3" ] || DEFAULT_SERVER_VENV="$(dirname "$REPO_ROOT")/.venv"
VENV="${VENV:-$DEFAULT_SERVER_VENV}"
if [ ! -x "$VENV/bin/python3" ]; then
    echo "ERROR: venv not found at $VENV/bin/python3"
    echo "Did you run scripts/sync_to_server.sh setup-remote?"
    exit 1
fi
PY="$VENV/bin/python3"

DATASET="${DATASET:-$REPO_ROOT/outputs/lerobot_datasets/pick_tape_v0}"
if [ ! -d "$DATASET" ]; then
    echo "ERROR: dataset $DATASET not found"
    echo "Did you sync demos and run convert_to_lerobot.py on the server?"
    exit 1
fi

# Honest val split: 10 episodes via stride sampling. eval_ckpt_loss.py
# --val-episodes <file> uses the same file after training.
VAL_FILE="$REPO_ROOT/outputs/training_runs/test2_val_episodes.txt"
mkdir -p "$REPO_ROOT/outputs/training_runs"
echo -e "0\n5\n10\n15\n20\n25\n30\n35\n40\n45" > "$VAL_FILE"

# Shared training knobs (the variables we're NOT varying — see EXPERIMENTS.md)
COMMON_FLAGS=(
    --dataset-root "$DATASET"
    --video-backend pyav
    --num-workers 16
    --steps 50000
    --save-freq 10000
    --log-freq 500
    --no-crop
    --horizon 8
    --n-action-steps 2
    --val-episodes "$VAL_FILE"
)

declare -A DP_RUNS=(
    # NAME       =  "GPU  VARIANT  BACKBONE  BATCH"
    [DP_A]="0 dual resnet34 48"
    [DP_B]="1 dual resnet18 64"
    [DP_C]="2 single resnet34 48"
    [DP_D]="3 dual resnet34 32"     # was 96 → OOM(48GB), 64 → OOM(48GB); 32 fits
)

declare -A VLA_RUNS=(
    # NAME       = "GPU  POLICY   BATCH  EXTRA_FLAGS"
    # PI0FAST was dead at launch: lerobot/pi0fast-base on HF is a 0.3-era
    # ckpt with no policy_preprocessor.json, which lerobot 0.5 requires.
    # Replaced with PI05 (lerobot/pi05_base has all 0.5 files).
    [PI05]="4 pi05 8"
    [SMOLVLA]="5 smolvla 16"
)

# Note: DP_D is supposed to use a 320x240 dataset for image-size ablation.
# Until we convert that, it will fall back to using the same full-res dataset
# but with a larger batch. The image-size ablation must be added once
# convert_to_lerobot.py --image-h 240 --image-w 320 is run on the server.

ONLY=""
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

launch_one() {
    local name="$1"
    local gpu_id="$2"
    local variant="$3"
    local backbone="$4"
    local batch="$5"

    local out="$REPO_ROOT/outputs/training_runs/test2_${name}"
    if [ -d "$out" ]; then
        echo "[skip] $name: output already exists at $out — delete to retrain"
        return
    fi

    local cmd=(
        env CUDA_VISIBLE_DEVICES="$gpu_id"
        "$PY" "$REPO_ROOT/scripts/train_dp.py"
        --variant "$variant"
        --backbone "$backbone"
        --batch-size "$batch"
        --output-dir "$out"
        "${COMMON_FLAGS[@]}"
    )

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry] $name (GPU $gpu_id): ${cmd[*]}"
        return
    fi

    echo "[launch] $name on GPU $gpu_id (variant=$variant backbone=$backbone batch=$batch)"
    # Detach: nohup + redirect. train_dp.py writes its own train.log inside
    # the output dir; we capture stdout/stderr to a wrapper log for crashes
    # before init_logging fires.
    nohup "${cmd[@]}" \
        > "$REPO_ROOT/outputs/training_runs/test2_${name}_wrap.log" 2>&1 &
    local pid=$!
    echo "  PID $pid"
    sleep 2
}

launch_vla() {
    local name="$1"; local gpu_id="$2"; local policy="$3"; local batch="$4"
    shift 4
    local extra_flags=("$@")

    local out="$REPO_ROOT/outputs/training_runs/test2_${name}"
    if [ -d "$out" ]; then
        echo "[skip] $name: output already exists at $out — delete to retrain"
        return
    fi

    local cmd=(
        env CUDA_VISIBLE_DEVICES="$gpu_id"
        "$PY" "$REPO_ROOT/scripts/train_vla.py"
        --dataset-root "$DATASET"
        --policy "$policy"
        --batch-size "$batch"
        --output-dir "$out"
        --video-backend pyav
        --num-workers 8
        --steps 30000
        --save-freq 5000
        --log-freq 200
        --val-episodes "$VAL_FILE"
        "${extra_flags[@]}"
    )

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry] $name (GPU $gpu_id): ${cmd[*]}"
        return
    fi

    echo "[launch] $name on GPU $gpu_id (policy=$policy batch=$batch extra=${extra_flags[*]})"
    nohup "${cmd[@]}" \
        > "$REPO_ROOT/outputs/training_runs/test2_${name}_wrap.log" 2>&1 &
    echo "  PID $!"
    sleep 2
}

for name in DP_A DP_B DP_C DP_D; do
    if [ -n "$ONLY" ] && ! echo ",$ONLY," | grep -q ",$name,"; then
        continue
    fi
    read -r gpu_id variant backbone batch <<<"${DP_RUNS[$name]}"
    launch_one "$name" "$gpu_id" "$variant" "$backbone" "$batch"
done

for name in PI05 SMOLVLA; do
    if [ -n "$ONLY" ] && ! echo ",$ONLY," | grep -q ",$name,"; then
        continue
    fi
    # Word-split the entry: "gpu policy batch [flag1 flag2 ...]"
    parts=(${VLA_RUNS[$name]})
    launch_vla "$name" "${parts[0]}" "${parts[1]}" "${parts[2]}" "${parts[@]:3}"
done

if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "Dry run done. To launch for real:  bash deploy/run_test2.sh"
    exit 0
fi

echo ""
echo "All requested runs launched."
echo "Monitor:  tail -f $REPO_ROOT/outputs/training_runs/test2_*_train.log"
echo "GPU use:  watch -n5 nvidia-smi"
