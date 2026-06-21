#!/usr/bin/env python3
"""Convert outputs/demos/<task>/*.hdf5 → LeRobot v0.3 dataset.

The HDF5 episodes are what scripts/record_demo.py produces: per-frame JPEG
RGB streams + joint timeseries. LeRobot wants its own on-disk layout
(parquet + mp4) so we transcode once for training.

Layout produced (single chunk, all episodes inline):

  outputs/lerobot_datasets/<task>_v<N>/
    meta/
      info.json
      tasks.jsonl
      episodes.jsonl
      stats.json
    data/chunk-000/episode_NNNNNN.parquet
    videos/chunk-000/observation.images.third_person/episode_NNNNNN.mp4
    videos/chunk-000/observation.images.wrist/episode_NNNNNN.mp4

LeRobot lives in the dp_maniskill venv (already has torch+CUDA), so run
this script from there:

  source ~/code/dp_maniskill/.venv/bin/activate
  python3 ~/code/rebot_arm/scripts/convert_to_lerobot.py \
      --task pick_tape --version v0
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

# These joints match the order in configs/joint_config.yaml (j1..j7).
JOINT_NAMES = [
    "base",
    "shoulder",
    "elbow",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True,
                   help="task name; reads outputs/demos/<task>/*.hdf5")
    p.add_argument("--version", default="v0",
                   help="dataset version tag (default v0)")
    p.add_argument("--repo-id", default=None,
                   help="lerobot repo id (default local/<task>_<version>)")
    p.add_argument("--fps", type=int, default=20,
                   help="dataset fps; must match the rate used during recording")
    p.add_argument("--image-h", type=int, default=480)
    p.add_argument("--image-w", type=int, default=640)
    p.add_argument("--robot-type", default="rebot_b601_dm")
    p.add_argument("--limit", type=int, default=None,
                   help="only convert first N episodes (for quick testing)")
    p.add_argument("--codec", default="h264",
                   choices=["h264", "hevc", "libsvtav1", "h264_nvenc", "hevc_nvenc"],
                   help="ffmpeg video codec for LeRobot's mp4 output. "
                        "h264 (CPU libx264) is the default — surprisingly "
                        "NOT slower than h264_nvenc here because the conversion "
                        "bottleneck is LeRobot's per-frame PNG intermediate "
                        "(written sequentially by image_writer), not the video "
                        "encode. NVENC option still available if you parallelise "
                        "the pipeline.")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets import lerobot_dataset as _ds_mod
        from lerobot.datasets import video_utils as _vu
    except ImportError as e:
        print(f"ERROR: lerobot not importable in this venv ({e})")
        print("Activate dp_maniskill venv:")
        print("  source ~/code/dp_maniskill/.venv/bin/activate")
        sys.exit(1)

    # Monkey-patch encode_video_frames to use our chosen codec.
    # LeRobot 0.3.2 has a hard whitelist {h264, hevc, libsvtav1} so passing
    # h264_nvenc/hevc_nvenc directly is rejected. For nvenc codecs we replace
    # encode_video_frames entirely with a PyAV-based encoder. For CPU codecs
    # we just wrap the original.
    _orig_encode = _vu.encode_video_frames

    if args.codec.endswith("_nvenc"):
        # PyAV's bundled libavcodec doesn't ship NVENC, only the SYSTEM
        # ffmpeg does. We shell out to it.
        import shutil as _shutil
        import subprocess as _subprocess
        if _shutil.which("ffmpeg") is None:
            print("ERROR: --codec nvenc needs system ffmpeg in PATH")
            sys.exit(1)

        def _patched_encode(imgs_dir, video_path, fps, **kwargs):
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(int(fps)),
                "-i", str(Path(imgs_dir) / "frame_%06d.png"),
                "-c:v", args.codec,
                "-cq", "28", "-preset", "p4",
                "-pix_fmt", "yuv420p",
                str(video_path),
            ]
            _subprocess.run(cmd, check=True)
    else:
        def _patched_encode(imgs_dir, video_path, fps, **kwargs):
            kwargs.setdefault("vcodec", args.codec)
            return _orig_encode(imgs_dir, video_path, fps, **kwargs)

    _vu.encode_video_frames = _patched_encode
    # Also patch the name imported into lerobot_dataset.py (it does
    # `from .video_utils import encode_video_frames`, which copies the ref).
    _ds_mod.encode_video_frames = _patched_encode

    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "outputs" / "demos" / args.task
    files = sorted(src_dir.glob("episode_*.hdf5"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print(f"no episodes found in {src_dir}")
        sys.exit(1)

    repo_id = args.repo_id or f"local/{args.task}_{args.version}"
    out_dir = repo_root / "outputs" / "lerobot_datasets" / f"{args.task}_{args.version}"
    if out_dir.exists():
        print(f"ERROR: output dir already exists at {out_dir}")
        print("Delete it or pick a different --version. We don't overwrite by default")
        print("because the conversion takes a while and overwrites are accident-prone.")
        sys.exit(1)

    print(f"Converting {len(files)} episodes from {src_dir}")
    print(f"  → {out_dir}")
    print(f"  repo_id = {repo_id}")
    print(f"  fps     = {args.fps}")
    print(f"  imgs    = {args.image_h}x{args.image_w}")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "observation.images.third_person": {
            "dtype": "video",
            "shape": (args.image_h, args.image_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (args.image_h, args.image_w, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
    }

    # lerobot 0.5+ accepts vcodec= and streaming_encoding= as first-class
    # params. Older 0.3.2 doesn't — fall back to our monkey-patched create
    # by passing only the older args and letting the patch above handle codec.
    import inspect as _inspect
    create_params = _inspect.signature(LeRobotDataset.create).parameters
    create_kwargs = dict(
        repo_id=repo_id,
        fps=args.fps,
        features=features,
        root=str(out_dir),
        robot_type=args.robot_type,
        use_videos=True,
    )
    if "vcodec" in create_params:
        # lerobot 0.5: native codec + streaming support. Skip our PNG-based
        # monkey-patch entirely — pass codec & streaming directly.
        create_kwargs["vcodec"] = args.codec
        if "streaming_encoding" in create_params:
            create_kwargs["streaming_encoding"] = True
            print(f"[convert] lerobot 0.5 native: vcodec={args.codec} "
                  f"streaming_encoding=True (bypasses PNG intermediate)")
    ds = LeRobotDataset.create(**create_kwargs)

    total_frames = 0
    skipped = []
    for src in files:
        try:
            n_added = convert_one_episode(ds, src, args)
        except Exception as e:
            print(f"  [!] {src.name} failed: {type(e).__name__}: {e}")
            skipped.append(src.name)
            continue
        total_frames += n_added
        print(f"  {src.name}: +{n_added} frames "
              f"(running total {total_frames})")

    print(f"\nDone. {len(files) - len(skipped)} / {len(files)} episodes converted, "
          f"{total_frames} frames total.")
    if skipped:
        print(f"Skipped: {skipped}")
    print(f"\nDataset at: {out_dir}")


def convert_one_episode(ds, src: Path, args) -> int:
    """Read one HDF5 episode, push each frame to the LeRobot dataset, save."""
    with h5py.File(src, "r") as f:
        n = int(f.attrs["n_frames"])
        ts = f["ts"][:]
        follower_pos = f["follower_pos"][:]      # (n, 7) float32
        action = f["action"][:]                   # (n, 7) float32
        third = f["third_person"]                 # (n,) vlen uint8 jpeg
        wrist = f["wrist_rgb"]                    # (n,) vlen uint8 jpeg

        for i in range(n):
            third_buf = np.frombuffer(third[i], dtype=np.uint8)
            third_img = cv2.imdecode(third_buf, cv2.IMREAD_COLOR)
            wrist_buf = np.frombuffer(wrist[i], dtype=np.uint8)
            wrist_img = cv2.imdecode(wrist_buf, cv2.IMREAD_COLOR)

            if third_img is None or wrist_img is None:
                # Skip frames where decode failed (shouldn't happen normally)
                continue
            if third_img.shape != (args.image_h, args.image_w, 3):
                third_img = cv2.resize(third_img, (args.image_w, args.image_h))
            if wrist_img.shape != (args.image_h, args.image_w, 3):
                wrist_img = cv2.resize(wrist_img, (args.image_w, args.image_h))

            # LeRobot expects RGB; OpenCV gives BGR
            third_rgb = cv2.cvtColor(third_img, cv2.COLOR_BGR2RGB)
            wrist_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)

            # Don't pass timestamp — let LeRobot auto-fill as frame_index/fps.
            # The recorded ts has small jitter (and occasional frame drops)
            # that fails LeRobot's strict timestamp-tolerance check; for DP
            # training the absolute wall-clock doesn't matter, only the
            # sequence of (state, image, action) frames.
            frame = {
                "observation.state": follower_pos[i].astype(np.float32),
                "observation.images.third_person": third_rgb,
                "observation.images.wrist": wrist_rgb,
                "action": action[i].astype(np.float32),
            }
            # lerobot 0.5 dropped the task= kwarg on add_frame() and moved
            # task to be a key inside the frame dict. Detect and adapt.
            import inspect
            sig = inspect.signature(ds.add_frame)
            if "task" in sig.parameters:
                ds.add_frame(frame=frame, task=args.task)
            else:
                frame["task"] = args.task
                ds.add_frame(frame=frame)
    ds.save_episode()
    return n


if __name__ == "__main__":
    main()
