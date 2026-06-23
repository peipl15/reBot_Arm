#!/usr/bin/env python3
"""Compute mean training-loss snapshot for each saved checkpoint.

We didn't get a real loss curve from the overnight training run because
lerobot 0.3.2's train() forgot to call init_logging(), so logging.info()
was dropped at root WARNING level. To get *something* comparable we load
each saved checkpoint, freeze it (eval mode), iterate N batches from the
training dataset, and average the diffusion loss.

This isn't the real training loss curve — it's the loss the model would
get if it were re-evaluated on its own training set right now. Useful for:
  - is loss monotonically decreasing across ckpts? (sanity check on convergence)
  - is dual lower than single at the same step? (was the wrist camera worth it?)

Usage:
  source ~/code/dp_maniskill/.venv/bin/activate
  python3 scripts/eval_ckpt_loss.py \
      --dataset-root outputs/lerobot_datasets/pick_tape_v0 \
      --runs outputs/training_runs/dp_single_cam outputs/training_runs/dp_dual_cam \
      --n-batches 80 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--runs", nargs="+", required=True,
                   help="one or more dp_<variant>_cam dirs (must contain checkpoints/)")
    p.add_argument("--n-batches", type=int, default=80,
                   help="how many batches to average loss over (default 80 ≈ 1280 frames)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42,
                   help="seeds the dataset sampler so single/dual see the SAME batches")
    p.add_argument("--video-backend", default="pyav",
                   choices=["pyav", "torchcodec", "torchvision"])
    p.add_argument("--output-json", default=None,
                   help="write results here as JSON (default: print only)")
    p.add_argument("--val-episodes", default=None,
                   help="restrict loss computation to these episodes. "
                        "Either a comma list ('0,5,10,15,...') or a path to "
                        "a file with one episode index per line. Use the "
                        "file dropped by train_dp.py --val-episodes to "
                        "measure honest held-out MSE.")
    return p.parse_args()


def patch_pyav():
    """Same PyAV monkey-patch we use in train_dp.py — torchcodec leaks."""
    import av
    import torch
    from lerobot.datasets import video_utils as _vu
    from lerobot.datasets import lerobot_dataset as _ds_mod

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
            for frame in container.decode(stream):
                ts_sec = float(frame.pts * time_base)
                idx = round(ts_sec * avg_fps)
                if idx in needed and idx not in found:
                    arr = frame.to_ndarray(format="rgb24")
                    found[idx] = torch.from_numpy(arr).permute(2, 0, 1)
                if len(found) == len(needed) or idx > max_idx + 2:
                    break
            avail = sorted(found.keys())
            out = []
            for q in frame_indices:
                out.append(found[q] if q in found
                           else found[min(avail, key=lambda x: abs(x - q))])
            return torch.stack(out).type(torch.float32) / 255.0
        finally:
            container.close()

    _vu.decode_video_frames = _decode_pyav
    _ds_mod.decode_video_frames = _decode_pyav


def main():
    args = parse_args()

    try:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    except ImportError as e:
        print(f"ERROR: missing import — activate dp_maniskill venv first.\n  {e}")
        sys.exit(1)

    # Lazy-import VLA policies because they need transformers / peft installed
    # only if the user actually evaluates a VLA ckpt.
    def _load_policy(model_dir, device):
        import json as _json
        from pathlib import Path as _P
        cfg = _json.load(open(f"{model_dir}/config.json"))
        ptype = cfg.get("type", "diffusion")
        if ptype == "diffusion":
            return DiffusionPolicy.from_pretrained(str(model_dir)).to(device), None

        # Compat shims for VLA paths (same as eval_dp.py).
        from lerobot.processor import ProcessorStepRegistry
        _reg = ProcessorStepRegistry._registry
        if ("relative_actions_processor" not in _reg
                and "delta_actions_processor" in _reg):
            ProcessorStepRegistry.register("relative_actions_processor")(
                _reg["delta_actions_processor"])
        if ptype == "pi05":
            # transformers 4.55+ SigLIP returns pooler_output=None; pi05's
            # original embed_image breaks. Patch to use last_hidden_state.
            from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel
            if not getattr(PaliGemmaWithExpertModel.embed_image,
                           "_pi05_patched", False):
                import torch as _t
                def _patched(self, image):
                    out_dtype = image.dtype
                    if image.dtype != _t.float32:
                        image = image.to(_t.float32)
                    vision = self.paligemma.model.vision_tower(image)
                    feats = vision.last_hidden_state
                    feats = self.paligemma.model.multi_modal_projector(feats)
                    if feats.dtype != out_dtype:
                        feats = feats.to(out_dtype)
                    return feats
                _patched._pi05_patched = True
                PaliGemmaWithExpertModel.embed_image = _patched

        from lerobot.processor import DataProcessorPipeline
        preproc = DataProcessorPipeline.from_pretrained(
            str(model_dir), config_filename="policy_preprocessor.json"
        )

        # LoRA detection: a saved adapter has `adapter_config.json` +
        # `adapter_model.safetensors` instead of a `model.safetensors`.
        # We then load the base HF model + wrap_with_peft pointing at our
        # adapter dir.
        is_lora = (_P(model_dir) / "adapter_config.json").exists()

        if ptype == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as PCls
            base_repo = "lerobot/smolvla_base"
        elif ptype in ("pi0", "pi05"):
            mod = "pi0" if ptype == "pi0" else "pi05"
            from importlib import import_module
            m = import_module(f"lerobot.policies.{mod}.modeling_{mod}")
            PCls = getattr(m, "PI0Policy" if ptype == "pi0" else "PI05Policy")
            base_repo = "lerobot/pi0_base" if ptype == "pi0" else "lerobot/pi05_base"
        elif ptype in ("pi0fast", "pi0_fast"):
            try:
                from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy as PCls
            except ImportError:
                from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy as PCls
            base_repo = "lerobot/pi0fast-base"
        else:
            raise ValueError(f"unknown policy type {ptype!r} in {model_dir}/config.json")

        if is_lora:
            print(f"  detected LoRA ckpt at {model_dir}; loading base {base_repo}")
            policy = PCls.from_pretrained(base_repo).to(device)
            # PreTrainedPolicy.wrap_with_peft creates a fresh adapter from a
            # PeftConfig — it doesn't load a saved adapter. To load a saved
            # adapter we use PEFT's own API.
            from peft import PeftModel
            policy = PeftModel.from_pretrained(policy, str(model_dir))
            policy = policy.to(device)
            # The base ckpt's config has input_features named after the
            # original pi05 training data (`base_0_rgb`, etc.). Our adapter
            # was actually trained against `third_person` / `wrist` so the
            # config on disk in OUR ckpt has the right names. Pull those
            # back onto policy.config so the dataset reader queries the
            # correct columns.
            from lerobot.configs.policies import PreTrainedConfig
            local_cfg = PreTrainedConfig.from_pretrained(str(model_dir))
            # Update config on EVERY accessible policy reference. PeftModel
            # wraps base_model.model and may or may not forward .config; we
            # patch both paths so policy.config and policy.base_model.config
            # and policy.base_model.model.config all reflect our trained
            # ckpt's actual feature names.
            for obj in (policy,
                        getattr(policy, "base_model", None),
                        getattr(getattr(policy, "base_model", None), "model", None)):
                if obj is None or not hasattr(obj, "config"):
                    continue
                try:
                    obj.config.input_features = local_cfg.input_features
                    obj.config.output_features = local_cfg.output_features
                except (AttributeError, TypeError):
                    pass
        else:
            policy = PCls.from_pretrained(str(model_dir)).to(device)
        return policy, preproc

    patch_pyav()

    dataset_root = Path(args.dataset_root).resolve()
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    print(f"Dataset {dataset_root.name}: "
          f"{info['total_frames']} frames, {info['total_episodes']} eps, fps={info['fps']}")

    # Resolve --val-episodes to a list of int episode IDs, or None for "all".
    val_episode_ids = None
    if args.val_episodes is not None:
        if Path(args.val_episodes).exists():
            val_episode_ids = [int(line.strip())
                               for line in Path(args.val_episodes).read_text().splitlines()
                               if line.strip()]
            src = f"file {args.val_episodes}"
        else:
            val_episode_ids = [int(x) for x in args.val_episodes.split(",")]
            src = "CLI"
        print(f"  → restricting to {len(val_episode_ids)} val episodes "
              f"from {src}: {val_episode_ids}")

    # We need to know each policy's input_features to build a matching batch,
    # but loading the policy itself tells us via policy.config. We'll just
    # build the dataset with ALL features and the policy will pick what it
    # needs in forward().
    results = {}

    for run_path in args.runs:
        run_path = Path(run_path).resolve()
        ckpt_dir = run_path / "checkpoints"
        if not ckpt_dir.exists():
            print(f"[skip] {run_path} has no checkpoints/")
            continue
        ckpts = sorted([p for p in ckpt_dir.iterdir()
                        if p.is_dir() and p.name.isdigit()])
        if not ckpts:
            print(f"[skip] {run_path}/checkpoints is empty")
            continue

        run_results = []
        for ckpt in ckpts:
            step = int(ckpt.name)
            model_dir = ckpt / "pretrained_model"
            if not model_dir.exists():
                print(f"  [{run_path.name}@{step}] missing pretrained_model dir, skip")
                continue

            print(f"\n=== {run_path.name} @ step {step} ===")
            t0 = time.time()
            policy, preproc = _load_policy(model_dir, args.device)
            policy.eval()

            # Build delta_timestamps the same way training did.
            # Resolve LeRobot policy config. PeftModel wraps policy and its
            # own .config is a PeftConfig, not our PI05Config — walk inner.
            policy_cfg = policy.config
            if not hasattr(policy_cfg, "input_features"):
                inner = getattr(policy, "base_model", None) or policy
                inner_model = getattr(inner, "model", inner)
                policy_cfg = inner_model.config
            input_feats = list(policy_cfg.input_features.keys())
            print(f"  resolved policy_cfg={type(policy_cfg).__name__}  "
                  f"input_features={input_feats}")
            fps = info["fps"]
            # SmolVLA / pi0 / pi05 don't store observation_delta_indices or
            # action_delta_indices on the config — they use n_obs_steps +
            # chunk_size instead. Derive them when missing.
            obs_di = getattr(policy_cfg, "observation_delta_indices", None)
            act_di = getattr(policy_cfg, "action_delta_indices", None)
            if obs_di is None:
                n_obs = int(getattr(policy_cfg, "n_obs_steps", 1))
                obs_di = list(range(-(n_obs - 1), 1))
            if act_di is None:
                chunk = int(getattr(policy_cfg, "chunk_size",
                                    getattr(policy_cfg, "horizon", 16)))
                act_di = list(range(0, chunk))
            obs_dt = [i / fps for i in obs_di]
            act_dt = [i / fps for i in act_di]
            delta_ts = {}
            for feat in input_feats:
                delta_ts[feat] = obs_dt
            delta_ts["action"] = act_dt

            # LeRobot 0.3.2's `episodes=[ids]` filter is broken for
            # non-contiguous IDs. Workaround: load all, then use
            # torch.utils.data.Subset with precomputed frame indices for
            # the val episodes. (Earlier attempt: filter inside the loop —
            # too slow because P(all 8 samples in batch ∈ 5/51 episodes)
            # is ~10^-7.)
            ds = LeRobotDataset(
                repo_id=f"local/{dataset_root.name}",
                root=str(dataset_root),
                episodes=None,
                delta_timestamps=delta_ts,
                video_backend=args.video_backend,
            )
            if val_episode_ids is not None:
                # lerobot 0.3: ds.episode_data_index["from"][k] / ["to"][k]
                # lerobot 0.5: ds.meta.episodes is a HF Dataset table with
                #              dataset_from_index / dataset_to_index columns.
                if hasattr(ds, "episode_data_index"):
                    ep_from = ds.episode_data_index["from"].tolist()
                    ep_to = ds.episode_data_index["to"].tolist()
                else:
                    rows = ds.meta.episodes
                    ep_from = [int(rows[k]["dataset_from_index"]) for k in range(len(rows))]
                    ep_to = [int(rows[k]["dataset_to_index"]) for k in range(len(rows))]
                val_indices = []
                for ep_id in val_episode_ids:
                    val_indices.extend(range(ep_from[ep_id], ep_to[ep_id]))
                print(f"  val subset: {len(val_indices)} frames "
                      f"across {len(val_episode_ids)} eps "
                      f"(out of {len(ds)} total)")
                ds = torch.utils.data.Subset(ds, val_indices)

            # We let lerobot's collator + dataloader figure it out — same path
            # the training run took.
            g = torch.Generator()
            g.manual_seed(args.seed)
            loader = torch.utils.data.DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                generator=g,
                pin_memory=(args.device == "cuda"),
                drop_last=True,
            )

            losses = []
            t_eval = time.time()
            with torch.inference_mode():
                for i, batch in enumerate(loader):
                    if i >= args.n_batches:
                        break
                    batch = {k: (v.to(args.device, non_blocking=True)
                                  if isinstance(v, torch.Tensor) else v)
                             for k, v in batch.items()}
                    if preproc is not None:
                        batch = preproc(batch)
                    out = policy.forward(batch)
                    # DP returns (loss, output_dict). SmolVLA/pi0/pi05/pi0fast
                    # return a single dict with key "loss".
                    if isinstance(out, tuple):
                        loss = out[0]
                    elif isinstance(out, dict):
                        loss = out.get("loss", out.get("ce_loss", None))
                    else:
                        loss = out
                    losses.append(float(loss.item() if hasattr(loss, "item") else loss))
            dt = time.time() - t_eval

            mean_loss = sum(losses) / max(1, len(losses))
            std_loss = (sum((x - mean_loss) ** 2 for x in losses)
                        / max(1, len(losses))) ** 0.5
            min_l = min(losses) if losses else float("nan")
            max_l = max(losses) if losses else float("nan")

            print(f"  N={len(losses)} batches × bs={args.batch_size}")
            print(f"  loss: mean={mean_loss:.4f}  std={std_loss:.4f}  "
                  f"min={min_l:.4f}  max={max_l:.4f}")
            print(f"  loaded+evaled in {time.time()-t0:.1f}s "
                  f"({len(losses)*args.batch_size/dt:.1f} samples/s)")

            run_results.append({
                "step": step,
                "n_batches": len(losses),
                "batch_size": args.batch_size,
                "loss_mean": mean_loss,
                "loss_std": std_loss,
                "loss_min": min_l,
                "loss_max": max_l,
            })

            # Free CUDA memory before next ckpt
            del policy, loader, ds
            torch.cuda.empty_cache()

        results[run_path.name] = run_results

    print("\n\n========= SUMMARY =========")
    # Print a comparison table
    all_steps = sorted({r["step"] for v in results.values() for r in v})
    header = f"{'step':>8} | " + " | ".join(f"{name:>20}" for name in results.keys())
    print(header)
    print("-" * len(header))
    for step in all_steps:
        row = [f"{step:>8}"]
        for name, entries in results.items():
            m = next((e for e in entries if e["step"] == step), None)
            cell = f"{m['loss_mean']:>20.4f}" if m else f"{'--':>20}"
            row.append(cell)
        print(" | ".join(row))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
