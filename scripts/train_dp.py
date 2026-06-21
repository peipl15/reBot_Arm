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
    p.add_argument("--video-backend", default="pyav",
                   choices=["pyav", "torchcodec", "torchvision"],
                   help="pyav is the only one that doesn't leak on long DP runs")
    p.add_argument("--save-freq", type=int, default=20_000)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=1e-4)
    # ── Test#2 knobs (the variables EXPERIMENTS.md H1-H3 are about) ──
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50"],
                   help="vision encoder. H2: resnet34 vs resnet18.")
    p.add_argument("--no-crop", action="store_true",
                   help="H1: drop the LeRobot default crop_shape=(84,84). "
                        "Use the full input image (random crop is disabled).")
    p.add_argument("--crop-h", type=int, default=84,
                   help="random-crop height when --no-crop is NOT set.")
    p.add_argument("--crop-w", type=int, default=84,
                   help="random-crop width when --no-crop is NOT set.")
    p.add_argument("--horizon", type=int, default=8,
                   help="H3: predicted chunk length. Was 16, default now 8.")
    p.add_argument("--n-action-steps", type=int, default=2,
                   help="H3: actions executed before replan. Was 8, default now 2.")
    p.add_argument("--n-obs-steps", type=int, default=2)
    p.add_argument("--val-episodes", default=None,
                   help="held-out episode indices, comma list (e.g. "
                        "'0,5,12,17,22,27,33,39,44,50') or a fraction "
                        "(e.g. '0.2'). When set, training uses the "
                        "complement; val episodes go to eval_ckpt_loss.py. "
                        "Implemented by writing them into "
                        "{output_dir}/val_episodes.txt — the training "
                        "loop itself doesn't see them (LeRobot 0.3 has no "
                        "first-class train/val split).")
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
        # lerobot 0.5 moved train() from lerobot.scripts.train to
        # lerobot.scripts.lerobot_train. Try 0.5 path first, fall back to 0.3.
        try:
            from lerobot.scripts.lerobot_train import train
        except ImportError:
            from lerobot.scripts.train import train
        from lerobot.datasets import video_utils as _vu
    except ImportError as e:
        print(f"ERROR: required imports missing ({e})")
        print("Activate dp_maniskill venv:")
        print("  source ~/code/dp_maniskill/.venv/bin/activate")
        sys.exit(1)

    # torchcodec 0.13's VideoDecoder leaks decoded frame buffers — observed at
    # 57 GB per worker after a few hundred steps. lerobot 0.3 routes the "pyav"
    # backend through torchvision.io.VideoReader, which torchvision 0.26 dropped.
    # We patch decode_video_frames to use PyAV directly: open → decode needed
    # frames → close (no persistent state across calls).
    if args.video_backend == "pyav":
        import av

        def _decode_pyav(video_path, timestamps, tolerance_s, backend=None):
            container = av.open(str(video_path))
            try:
                stream = container.streams.video[0]
                avg_fps = float(stream.average_rate)
                time_base = stream.time_base
                frame_indices = [round(ts * avg_fps) for ts in timestamps]
                target_idx = min(frame_indices)
                target_pts = int(target_idx / avg_fps / float(time_base))
                container.seek(max(target_pts, 0), stream=stream, backward=True)
                needed = set(frame_indices)
                found = {}
                max_idx = max(frame_indices)
                import torch as _t
                for frame in container.decode(stream):
                    ts_sec = float(frame.pts * time_base)
                    idx = round(ts_sec * avg_fps)
                    if idx in needed and idx not in found:
                        arr = frame.to_ndarray(format="rgb24")
                        found[idx] = _t.from_numpy(arr).permute(2, 0, 1)
                    if len(found) == len(needed) or idx > max_idx + 2:
                        break
                if not found:
                    raise RuntimeError(f"pyav decoded 0 frames from {video_path}")
                # For any index we couldn't hit exactly, fall back to nearest decoded.
                avail = sorted(found.keys())
                out_frames = []
                for q in frame_indices:
                    if q in found:
                        out_frames.append(found[q])
                    else:
                        nearest = min(avail, key=lambda x: abs(x - q))
                        out_frames.append(found[nearest])
                tensor = _t.stack(out_frames).type(_t.float32) / 255.0
                return tensor
            finally:
                container.close()

        _vu.decode_video_frames = _decode_pyav
        # lerobot 0.3 also imports decode_video_frames into lerobot_dataset
        # at module load, so we need to patch that binding too. 0.5 no
        # longer does, but the attribute may exist as a leftover — set
        # defensively if present.
        from lerobot.datasets import lerobot_dataset as _ds_mod
        if hasattr(_ds_mod, "decode_video_frames"):
            _ds_mod.decode_video_frames = _decode_pyav
        print("[train_dp] patched decode_video_frames to PyAV "
              "(workaround for torchcodec OOM)")

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
    # info.json stores image shape as (H, W, C). lerobot 0.5's
    # DiffusionConfig.validate_features expects PolicyFeature shape in
    # (C, H, W) order. 0.3 was permissive about this; 0.5 enforces it.
    # We detect by checking trailing dim == 3 and transpose.
    third_shape_raw = tuple(feats["observation.images.third_person"]["shape"])
    if len(third_shape_raw) == 3 and third_shape_raw[-1] == 3:
        third_shape = (third_shape_raw[2], third_shape_raw[0], third_shape_raw[1])
    else:
        third_shape = third_shape_raw
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
        wrist_shape_raw = tuple(feats["observation.images.wrist"]["shape"])
        if len(wrist_shape_raw) == 3 and wrist_shape_raw[-1] == 3:
            wrist_shape = (wrist_shape_raw[2], wrist_shape_raw[0], wrist_shape_raw[1])
        else:
            wrist_shape = wrist_shape_raw
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

    # H1: no-crop uses the raw image. lerobot 0.3 didn't accept None and
    # would happily take crop_shape=image_shape. 0.5's validate_features
    # rejects crop_shape >= image_shape (strict <), so for 0.5 we have to
    # subtract 1 pixel. Also third_shape may be HWC or CHW (we transposed
    # to CHW above when needed), so derive H/W from whichever is positional.
    if len(third_shape) == 3 and third_shape[0] == 3:
        img_h, img_w = third_shape[1], third_shape[2]  # CHW
    else:
        img_h, img_w = third_shape[0], third_shape[1]  # HWC
    if args.no_crop:
        # Pass slightly-smaller crop so 0.5's strict check passes; in 0.3 this
        # is functionally identical. crop_is_random=False -> center crop only.
        crop_shape = (img_h - 1, img_w - 1)
        crop_is_random = False
        print(f"[train_dp] H1: crop_shape ≈ full input ({img_h-1}x{img_w-1} "
              f"vs image {img_h}x{img_w}; lerobot 0.5 requires strict <).")
    else:
        crop_shape = (args.crop_h, args.crop_w)
        crop_is_random = True

    policy_cfg = DiffusionConfig(
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        normalization_mapping=normalization_mapping,
        input_features=input_features,
        output_features=output_features,
        device=args.device,
        push_to_hub=False,
        vision_backbone=args.backbone,
        crop_shape=crop_shape,
        crop_is_random=crop_is_random,
        use_separate_rgb_encoder_per_camera=(args.variant == "dual"),
        optimizer_lr=args.lr,
    )
    print(f"[train_dp] backbone={args.backbone}  crop_shape={crop_shape}  "
          f"horizon={args.horizon}  n_action_steps={args.n_action_steps}  "
          f"n_obs_steps={args.n_obs_steps}")

    repo_id = args.repo_id or f"local/{dataset_root.name}"
    # torchcodec (lerobot default) leaks frame buffers — even num_workers=1
    # OOM'd at 57 GB on this dataset. pyav releases properly.
    dataset_cfg = DatasetConfig(
        repo_id=repo_id,
        root=str(dataset_root),
        video_backend=args.video_backend,
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

    # Record the val episode IDs so eval_ckpt_loss.py can pick them up later.
    # LeRobot 0.3.2 has no first-class train/val split, so training itself
    # still sees ALL episodes. The honest val MSE is computed offline by
    # eval_ckpt_loss.py --val-episodes-file.
    if args.val_episodes is not None:
        total_eps = info["total_episodes"]
        if "," in args.val_episodes:
            val_ids = [int(x) for x in args.val_episodes.split(",")]
        elif Path(args.val_episodes).exists():
            val_ids = [int(l.strip()) for l in Path(args.val_episodes).read_text().splitlines() if l.strip()]
        else:
            frac = float(args.val_episodes)
            assert 0 < frac < 1, f"--val-episodes fraction must be in (0,1), got {frac}"
            # Deterministic: take a stride through the episode index. Stride
            # avoids "all val from one collection batch" bias if episodes were
            # collected in order.
            n_val = max(1, round(total_eps * frac))
            stride = total_eps / n_val
            val_ids = sorted({int(i * stride) for i in range(n_val)})
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        (output_dir.parent / f"{output_dir.name}_val_episodes.txt").write_text(
            "\n".join(map(str, val_ids)) + "\n"
        )
        print(f"[train_dp] val split: {len(val_ids)} held-out episodes -> "
              f"{output_dir.name}_val_episodes.txt: {val_ids}")
        print("  (training itself still sees ALL episodes; eval_ckpt_loss "
              "honors the file when measuring val MSE)")

    print(f"\nCalling lerobot.scripts.train.train(cfg)...\n")

    # lerobot 0.3.2's train() calls logging.info(...) everywhere but never
    # init_logging(), so on the root logger's default WARNING level all
    # training metrics are silently dropped. We set up the file+console
    # handlers ourselves so loss/grad_norm/lr/step are captured.
    #
    # We must NOT pre-create output_dir: train()'s cfg.validate() raises
    # FileExistsError if output_dir already exists and resume is False, and it
    # only creates output_dir lazily (at first checkpoint). So we write the log
    # to a run-named file in the (already-existing) parent dir instead.
    from lerobot.utils.utils import init_logging
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    init_logging(log_file=output_dir.parent / f"{output_dir.name}_train.log")

    # make_policy() unconditionally rebuilds cfg.input_features from the
    # dataset's features (factory.py lines 160-161), erasing any subset we
    # asked for. The model's RGB encoders are then built for every feature
    # in the dataset, so dropping keys after make_policy returns is too late.
    # Fix: patch dataset_to_policy_features at the source so the filtered
    # set propagates through the WHOLE construction (input_features, stats,
    # encoder modules, normalization buffers).
    requested_keys = set(input_features.keys()) | set(output_features.keys())
    from lerobot.policies import factory as _pf
    _orig_d2pf = _pf.dataset_to_policy_features

    def _d2pf_filtered(features_dict):
        feats = _orig_d2pf(features_dict)
        dropped = [k for k in list(feats.keys()) if k not in requested_keys]
        for k in dropped:
            feats.pop(k)
        if dropped:
            print(f"[train_dp] filtered out auto-added features: {dropped}")
        return feats

    _pf.dataset_to_policy_features = _d2pf_filtered

    train(train_cfg)


if __name__ == "__main__":
    main()
