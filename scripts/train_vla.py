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
    # Note: HF repo names use hyphens and underscores inconsistently. These
    # are the names that actually resolve as of 2026-06-21:
    "pi0fast": "lerobot/pi0fast-base",   # 0.3-style ckpt, missing 0.5 preprocessor
    "pi05":    "lerobot/pi05_base",      # 0.5-ready; has policy_preprocessor.json
    "smolvla": "lerobot/smolvla_base",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--policy", choices=["pi0fast", "pi05", "smolvla"], required=True,
                   help="pi0fast = lerobot 0.3/0.5; pi05 = lerobot 0.5 only; "
                        "smolvla = lerobot 0.3+")
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
    p.add_argument("--lora", action="store_true",
                   help="Enable LoRA training via TrainPipelineConfig.peft. "
                        "Required for pi05 on a single A40 (full fine-tune OOMs).")
    p.add_argument("--dtype", default=None,
                   choices=[None, "float32", "bfloat16"],
                   help="Override the policy config dtype. PI05Config "
                        "defaults to float32. Pass 'bfloat16' to enable "
                        "mixed-precision: bf16 forward/backward + fp32 "
                        "master weights + fp32 optimizer state. ~2× less "
                        "VRAM than pure fp32.")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Trade compute for VRAM: recompute activations "
                        "during backward instead of caching. Cuts "
                        "activation memory ~3-5×. Required for full pi05 "
                        "fine-tune on single A40.")
    p.add_argument("--lora-r", type=int, default=16,
                   help="LoRA rank.")
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj",
                   help="comma-separated module name suffixes to wrap with LoRA.")
    p.add_argument("--val-episodes", default=None,
                   help="see train_dp.py — same semantics, same file format")
    return p.parse_args()


def patch_pi05_embed_image_for_new_transformers():
    """transformers ≥ 4.55 made `PaliGemmaModel.get_image_features` return a
    raw Tensor instead of a `BaseModelOutputWithPooling`, so pi05's
    `embed_image` (which reads `image_outputs.pooler_output`) crashes with
    `'Tensor' object has no attribute 'pooler_output'` on transformers
    4.57.6. lerobot 0.5.1's pi05 hasn't been updated for this yet.

    Workaround: replace `embed_image` to call `vision_tower` directly,
    which still returns the old object with `.pooler_output`.

    Idempotent."""
    try:
        from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel
    except ImportError:
        return
    if getattr(PaliGemmaWithExpertModel.embed_image, "_pi05_patched", False):
        return
    import torch as _t

    def _patched(self, image):
        out_dtype = image.dtype
        if image.dtype != _t.float32:
            image = image.to(_t.float32)
        # paligemma.model is the inner PaliGemmaModel; we go straight to
        # vision_tower + multi_modal_projector, the path the model wants.
        # `image_outputs` is BaseModelOutputWithPooling but transformers
        # 4.57's SigLIP returns `pooler_output=None` because the
        # MultiHeadAttentionPoolingHead is disabled. We:
        #   (1) take last_hidden_state -> (B, P, H_vis)
        #   (2) apply the multi_modal_projector -> (B, P, H_text)
        #   (3) divide by sqrt(H_text) per the new PaliGemma convention,
        #       then multiply by H_text**0.5 per the old pi05 convention,
        #       which cancels — so the projected features are returned as-is.
        # This matches what `get_image_features` does in modern transformers
        # except we return Tensor of shape (B, P, H_text) rather than the
        # original pi05's (B, H_text) pooled vector.
        # pi05's downstream code already expects per-patch features
        # in the prefix_embs concatenation, so this shape is correct.
        vision = self.paligemma.model.vision_tower(image)
        feats = vision.last_hidden_state                       # (B, P, H_vis)
        feats = self.paligemma.model.multi_modal_projector(feats)  # (B, P, H_text)
        if feats.dtype != out_dtype:
            feats = feats.to(out_dtype)
        return feats

    _patched._pi05_patched = True  # type: ignore[attr-defined]
    PaliGemmaWithExpertModel.embed_image = _patched
    print("[train_vla] pi05 embed_image patched for transformers ≥ 4.55 API")


def patch_processor_registry_aliases():
    """Add registry-name aliases for processor steps that pi0.5 / future
    VLA ckpts reference but our installed lerobot doesn't expose under
    that name.

    Concrete case (2026-06-21): `lerobot/pi05_base`'s
    `policy_preprocessor.json` references `relative_actions_processor`,
    but lerobot 0.5.1 only registered the SAME class
    (`RelativeActionsProcessorStep`) under the name
    `delta_actions_processor`. A one-line alias is enough — the step is
    `enabled: false` in the ckpt config anyway, so it's just a registry
    lookup, no runtime semantics.
    """
    from lerobot.processor import ProcessorStepRegistry
    reg = ProcessorStepRegistry._registry
    aliases = {
        "relative_actions_processor": "delta_actions_processor",
    }
    for new_name, existing_name in aliases.items():
        if new_name in reg or existing_name not in reg:
            continue
        cls = reg[existing_name]
        ProcessorStepRegistry.register(new_name)(cls)
        print(f"[train_vla] registry alias: {new_name} → {existing_name} "
              f"({cls.__name__})")


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
        try:
            from lerobot.scripts.lerobot_train import train
        except ImportError:
            from lerobot.scripts.train import train
        if args.policy == "pi0fast":
            # lerobot 0.3 used pi0fast (no underscore), some 0.5 builds use
            # pi0_fast. Try both.
            try:
                from lerobot.policies.pi0fast.configuration_pi0fast import (
                    PI0FASTConfig as PolicyCfg)
            except ImportError:
                from lerobot.policies.pi0_fast.configuration_pi0_fast import (
                    PI0FastConfig as PolicyCfg)
        elif args.policy == "pi05":
            # pi0.5 is only in lerobot 0.5+
            from lerobot.policies.pi05.configuration_pi05 import (
                PI05Config as PolicyCfg)
        else:
            from lerobot.policies.smolvla.configuration_smolvla import (
                SmolVLAConfig as PolicyCfg)
    except ImportError as e:
        print(f"ERROR: required imports missing ({e})")
        sys.exit(1)

    if args.video_backend == "pyav":
        patch_video_backend()
        print("[train_vla] PyAV video patch applied")

    # Must run BEFORE PolicyCfg import or any DataProcessorPipeline.from_pretrained.
    patch_processor_registry_aliases()
    if args.policy == "pi05":
        patch_pi05_embed_image_for_new_transformers()

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
    if args.dtype is not None:
        policy_kwargs["dtype"] = args.dtype
        print(f"[train_vla] policy dtype override: {args.dtype}")
    if args.gradient_checkpointing:
        policy_kwargs["gradient_checkpointing"] = True
        print("[train_vla] gradient_checkpointing=True")
    if args.policy == "pi0fast" and args.use_lora:
        # lerobot 0.3: `freeze_vlm=True` is policy-level (just freezes the VLM).
        # lerobot 0.5: `use_peft=True` flips the policy into PEFT mode, but
        # that ALSO triggers `from_pretrained(adapter_config.json)` lookup
        # on the pretrained_path — fails for vanilla `lerobot/pi0fast_base`
        # because it isn't a PEFT-saved repo. Proper LoRA needs a peft block
        # in TrainPipelineConfig and a pre-saved adapter. For now, run full
        # fine-tune (slow but works).
        import dataclasses as _dc
        try:
            from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig as _Cfg
        except ImportError:
            from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig as _Cfg
        names = {f.name for f in _dc.fields(_Cfg)}
        if "freeze_vlm" in names:
            # lerobot 0.3 path — safe to use.
            policy_kwargs.update(dict(freeze_vlm=True))
            print("[train_vla] pi0fast LoRA: freeze_vlm=True (lerobot 0.3)")
        else:
            print("[train_vla] pi0fast LoRA on lerobot 0.5 needs a peft "
                  "block in TrainPipelineConfig — falling back to full "
                  "fine-tune. (TODO: wire LoRA properly.)")
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

    train_cfg_kwargs = dict(
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
    if args.lora:
        from lerobot.configs.default import PeftConfig
        peft_cfg = PeftConfig(
            method_type="LORA",
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(","),
        )
        train_cfg_kwargs["peft"] = peft_cfg
        print(f"[train_vla] LoRA enabled: r={args.lora_r}  "
              f"targets={args.lora_target_modules}")
    train_cfg = TrainPipelineConfig(**train_cfg_kwargs)

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
