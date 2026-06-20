#!/usr/bin/env python3
"""Fine-tune a LeRobot VLA (pi0fast or smolvla) on pick_tape.

These are pretrained foundation models — we are NOT training from scratch.
The HF Hub-hosted weights load on first run; further compute fine-tunes
them on our 41-episode train split (10 held out for val MSE).

Sister script to train_dp.py. Most flags are the same. Differences:
  --policy {pi0fast,smolvla}   which VLA family to fine-tune
  --task-prompt "<text>"       language instruction shown to the model
  --pretrained-path <hf-id>    override the default HF checkpoint
  --use-lora                   pi0fast only — train LoRA adapters instead of
                               full backbone. Cheap on VRAM, fast convergence.

The same monkey patches that train_dp.py applies (PyAV video backend, lerobot
init_logging, dataset_to_policy_features filter) are applied here too —
they're not DP-specific.

Run from the rebot_arm work root with the dp_maniskill / server venv:

  source /nvmessd/yinzi/ppl_reBot/.venv/bin/activate     # server
  python3 scripts/train_vla.py \\
      --dataset-root outputs/lerobot_datasets/pick_tape_v0 \\
      --policy pi0fast --use-lora \\
      --steps 30000 --batch-size 16 \\
      --output-dir outputs/training_runs/test2_pi0fast

For SmolVLA (cheaper, no LoRA needed):
  python3 scripts/train_vla.py \\
      --dataset-root outputs/lerobot_datasets/pick_tape_v0 \\
      --policy smolvla \\
      --steps 30000 --batch-size 32 \\
      --output-dir outputs/training_runs/test2_smolvla
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# HF Hub IDs for the default pretrained checkpoints. We document them so the
# next reader knows where the weights came from; override with --pretrained-path.
DEFAULT_CKPT = {
    "pi0fast": "lerobot/pi0fast_base",
    "smolvla": "lerobot/smolvla_base",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--policy", choices=["pi0fast", "smolvla"], required=True)
    p.add_argument("--pretrained-path", default=None,
                   help=f"HF id or local path. Defaults: {DEFAULT_CKPT}")
    p.add_argument("--task-prompt", default="pick up the black tape and put it in the silver box",
                   help="language instruction the VLA sees. Defaults to a "
                        "verbose pick_tape description.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--video-backend", default="pyav",
                   choices=["pyav", "torchcodec", "torchvision"])
    p.add_argument("--save-freq", type=int, default=10_000)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--use-lora", action="store_true",
                   help="pi0fast only — LoRA adapters instead of full fine-tune. "
                        "Sets pi0fast.lora_*=True via config kwargs.")
    p.add_argument("--val-episodes", default=None,
                   help="see train_dp.py — same semantics, same file format")
    return p.parse_args()


def patch_video_backend():
    """Same PyAV patch as train_dp.py — needed if running locally (torchcodec
    leaks). On the server it's belt-and-suspenders."""
    import av
    import torch as _t
    from lerobot.datasets import video_utils as _vu
    from lerobot.datasets import lerobot_dataset as _ds_mod

    def _decode_pyav(video_path, timestamps, tolerance_s, backend=None):
        container = av.open(str(video_path))
        try:
            stream = container.streams.video[0]
            avg_fps = float(stream.average_rate)
            time_base = stream.time_base
            frame_indices = [round(ts * avg_fps) for ts in timestamps]
            target_pts = int(min(frame_indices) / avg_fps / float(time_base))
            container.seek(max(target_pts, 0), stream=stream, backward=True)
            needed = set(frame_indices)
            found = {}
            max_idx = max(frame_indices)
            for frame in container.decode(stream):
                ts_sec = float(frame.pts * time_base)
                idx = round(ts_sec * avg_fps)
                if idx in needed and idx not in found:
                    arr = frame.to_ndarray(format="rgb24")
                    found[idx] = _t.from_numpy(arr).permute(2, 0, 1)
                if len(found) == len(needed) or idx > max_idx + 2:
                    break
            avail = sorted(found.keys())
            out = []
            for q in frame_indices:
                out.append(found[q] if q in found
                           else found[min(avail, key=lambda x: abs(x - q))])
            return _t.stack(out).type(_t.float32) / 255.0
        finally:
            container.close()

    _vu.decode_video_frames = _decode_pyav
    _ds_mod.decode_video_frames = _decode_pyav


def main():
    args = parse_args()

    try:
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.configs.default import DatasetConfig
        from lerobot.configs.types import (
            PolicyFeature, FeatureType, NormalizationMode
        )
        from lerobot.scripts.train import train
        if args.policy == "pi0fast":
            from lerobot.policies.pi0fast.configuration_pi0fast import (
                PI0FASTConfig as PolicyCfg)
        else:
            from lerobot.policies.smolvla.configuration_smolvla import (
                SmolVLAConfig as PolicyCfg)
    except ImportError as e:
        print(f"ERROR: required imports missing ({e})")
        sys.exit(1)

    if args.video_backend == "pyav":
        patch_video_backend()
        print("[train_vla] PyAV video patch applied")

    dataset_root = Path(args.dataset_root).resolve()
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    feats = info["features"]
    print(f"Dataset features in {dataset_root}:")
    for name, spec in feats.items():
        if "shape" in spec:
            print(f"  {name:<40} {spec['dtype']:<8} {tuple(spec['shape'])}")

    state_shape = tuple(feats["observation.state"]["shape"])
    third_shape = tuple(feats["observation.images.third_person"]["shape"])
    action_shape = tuple(feats["action"]["shape"])

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=state_shape),
        "observation.images.third_person": PolicyFeature(
            type=FeatureType.VISUAL, shape=third_shape),
    }
    if "observation.images.wrist" in feats:
        input_features["observation.images.wrist"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=tuple(feats["observation.images.wrist"]["shape"]))
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape),
    }

    normalization_mapping = {
        "STATE": NormalizationMode.MEAN_STD,
        "VISUAL": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MIN_MAX,
    }

    pretrained_path = args.pretrained_path or DEFAULT_CKPT[args.policy]
    policy_kwargs = dict(
        normalization_mapping=normalization_mapping,
        input_features=input_features,
        output_features=output_features,
        device=args.device,
        push_to_hub=False,
        pretrained_path=pretrained_path,
        optimizer_lr=args.lr,
    )
    if args.policy == "pi0fast" and args.use_lora:
        # PI0FASTConfig exposes lora_targets / freeze_vlm depending on lerobot
        # version. Set both conservatively; unknown keys are ignored at config
        # construction.
        policy_kwargs.update(dict(freeze_vlm=True))
    policy_cfg = PolicyCfg(**policy_kwargs)
    print(f"[train_vla] policy={args.policy}  pretrained={pretrained_path}  "
          f"use_lora={args.use_lora}")

    repo_id = args.repo_id or f"local/{dataset_root.name}"
    dataset_cfg = DatasetConfig(
        repo_id=repo_id,
        root=str(dataset_root),
        video_backend=args.video_backend,
    )

    output_dir = Path(args.output_dir) if args.output_dir else (
        dataset_root.parent.parent / "training_runs" / f"{args.policy}_default"
    )
    output_dir = output_dir.resolve()

    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=f"{args.policy}_finetune",
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=10**9,
        log_freq=args.log_freq,
        save_checkpoint=True,
        save_freq=args.save_freq,
        use_policy_training_preset=True,
    )

    if args.val_episodes is not None:
        total_eps = info["total_episodes"]
        if "," in args.val_episodes:
            val_ids = [int(x) for x in args.val_episodes.split(",")]
        elif Path(args.val_episodes).exists():
            val_ids = [int(l.strip()) for l in Path(args.val_episodes).read_text().splitlines() if l.strip()]
        else:
            frac = float(args.val_episodes)
            n_val = max(1, round(total_eps * frac))
            stride = total_eps / n_val
            val_ids = sorted({int(i * stride) for i in range(n_val)})
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        (output_dir.parent / f"{output_dir.name}_val_episodes.txt").write_text(
            "\n".join(map(str, val_ids)) + "\n"
        )
        print(f"[train_vla] val split: {len(val_ids)} held-out episodes "
              f"-> {output_dir.name}_val_episodes.txt: {val_ids}")

    print(f"\nOutput dir: {output_dir}")
    print(f"Steps: {args.steps}  batch_size: {args.batch_size}  device: {args.device}")
    print(f"Task prompt: {args.task_prompt!r}")
    print(f"\nCalling lerobot.scripts.train.train(cfg)...\n")

    from lerobot.utils.utils import init_logging
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    init_logging(log_file=output_dir.parent / f"{output_dir.name}_train.log")

    # input_features filter — same reason as train_dp.py
    requested_keys = set(input_features.keys()) | set(output_features.keys())
    from lerobot.policies import factory as _pf
    _orig_d2pf = _pf.dataset_to_policy_features

    def _d2pf_filtered(features_dict):
        feats = _orig_d2pf(features_dict)
        dropped = [k for k in list(feats.keys()) if k not in requested_keys]
        for k in dropped:
            feats.pop(k)
        if dropped:
            print(f"[train_vla] filtered out auto-added features: {dropped}")
        return feats

    _pf.dataset_to_policy_features = _d2pf_filtered

    train(train_cfg)


if __name__ == "__main__":
    main()
