#!/usr/bin/env python3
"""Train a LeRobot DiffusionPolicy on our pick_tape dataset.

We bypass `lerobot.scripts.train`'s CLI (which expects all features in the
dataset to be policy inputs) and construct the config programmatically so
the single-camera ablation is a one-flag switch.

Run from dp_maniskill venv (has torch + lerobot):

  source ~/code/dp_maniskill/.venv/bin/activate
  python3 ~/code/rebot_arm/scripts/train_dp.py \\
      --dataset-root outputs/lerobot_datasets/pick_tape_v0 \\
      --variant single \\
      --steps 100000 --batch-size 64

  python3 ~/code/rebot_arm/scripts/train_dp.py \\
      --dataset-root outputs/lerobot_datasets/pick_tape_v0 \\
      --variant dual \\
      --steps 100000 --batch-size 64

Output goes to outputs/training_runs/dp_<variant>_cam/.

For smoke test (verify pipeline before committing to a 2-hour training):

  python3 ~/code/rebot_arm/scripts/train_dp.py \\
      --dataset-root outputs/lerobot_datasets/pick_tape_v0 \\
      --variant single \\
      --steps 200 --batch-size 16 \\
      --output-dir outputs/training_runs/smoke_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True,
                   help="path to LeRobot dataset root (e.g. outputs/lerobot_datasets/pick_tape_v0)")
    p.add_argument("--repo-id", default=None,
                   help="dataset repo_id; default = local/<dirname>")
    p.add_argument("--variant", choices=["single", "dual"], required=True,
                   help="single = only third_person camera; dual = third_person + wrist")
    p.add_argument("--output-dir", default=None,
                   help="override training run output dir; default outputs/training_runs/dp_<variant>_cam")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64,
                   help="batch size. RTX 3500 Ada 12GB handles 64 at 640x480 image.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-freq", type=int, default=20_000)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()


def main():
    args = parse_args()

    # Imports kept inside main so --help is fast and doesn't crash if torch is missing.
    try:
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.configs.default import DatasetConfig
        from lerobot.configs.types import (
            PolicyFeature, FeatureType, NormalizationMode
        )
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.scripts.train import train
    except ImportError as e:
        print(f"ERROR: required imports missing ({e})")
        print("Activate dp_maniskill venv:")
        print("  source ~/code/dp_maniskill/.venv/bin/activate")
        sys.exit(1)

    dataset_root = Path(args.dataset_root).resolve()
    if not (dataset_root / "meta" / "info.json").exists():
        print(f"ERROR: {dataset_root} doesn't look like a LeRobot dataset "
              f"(no meta/info.json)")
        sys.exit(1)

    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    feats = info["features"]
    print(f"Dataset features in {dataset_root}:")
    for name, spec in feats.items():
        print(f"  {name:<40} dtype={spec.get('dtype'):<8} shape={tuple(spec.get('shape', []))}")

    # Build input_features dict. Always include state + third_person.
    # Add wrist only for the dual variant.
    state_shape = tuple(feats["observation.state"]["shape"])
    third_shape = tuple(feats["observation.images.third_person"]["shape"])
    action_shape = tuple(feats["action"]["shape"])

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=state_shape),
        "observation.images.third_person": PolicyFeature(
            type=FeatureType.VISUAL, shape=third_shape),
    }
    if args.variant == "dual":
        if "observation.images.wrist" not in feats:
            print("ERROR: --variant dual but dataset has no observation.images.wrist")
            sys.exit(1)
        wrist_shape = tuple(feats["observation.images.wrist"]["shape"])
        input_features["observation.images.wrist"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=wrist_shape)

    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape),
    }

    normalization_mapping = {
        "STATE": NormalizationMode.MEAN_STD,
        "VISUAL": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MIN_MAX,
    }

    print(f"\nVariant: {args.variant}")
    print(f"  input_features:  {list(input_features.keys())}")
    print(f"  output_features: {list(output_features.keys())}")

    policy_cfg = DiffusionConfig(
        n_obs_steps=2,
        normalization_mapping=normalization_mapping,
        input_features=input_features,
        output_features=output_features,
        device=args.device,
        push_to_hub=False,
        # ResNet18 vision backbone is the LeRobot DP default
        vision_backbone="resnet18",
        crop_shape=(84, 84),
        use_separate_rgb_encoder_per_camera=(args.variant == "dual"),
        optimizer_lr=args.lr,
    )

    repo_id = args.repo_id or f"local/{dataset_root.name}"
    dataset_cfg = DatasetConfig(
        repo_id=repo_id,
        root=str(dataset_root),
        # image_transforms is a required dataclass field — pass its default instance
        # via the parser config wrapper. Use the type's default factory.
    )

    output_dir = Path(args.output_dir) if args.output_dir else (
        dataset_root.parent.parent / "training_runs" / f"dp_{args.variant}_cam"
    )
    output_dir = output_dir.resolve()

    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=f"dp_{args.variant}_cam",
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=10**9,  # we have no sim env for eval; disable
        log_freq=args.log_freq,
        save_checkpoint=True,
        save_freq=args.save_freq,
        # True lets DP's preset auto-fill AdamW + cosine scheduler (otherwise
        # we'd have to pass optimizer + scheduler configs ourselves).
        use_policy_training_preset=True,
    )

    print(f"\nOutput dir: {output_dir}")
    print(f"Steps: {args.steps}  batch_size: {args.batch_size}  device: {args.device}")
    print(f"\nCalling lerobot.scripts.train.train(cfg)...\n")
    train(train_cfg)


if __name__ == "__main__":
    main()
