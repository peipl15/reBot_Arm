#!/usr/bin/env python3
"""Correct the j4–j7 torque/velocity decode scale on demos recorded before the
DM-J4310 model fix (commit c7d5a7f, 2026-06-21 20:11).

Background
----------
Until 2026-06-21, configs/joint_config.yaml mislabeled j4–j7 as DM-J4340
(T_MAX=28, V_MAX=10). They are actually DM-J4310 (T_MAX=10, V_MAX=30).
motorbridge decodes the feedback torque/velocity registers by scaling the raw
counts against the *model's* T_MAX / V_MAX. So every demo recorded before the
fix stored, for j4–j7 only:

    torque_stored = torque_true * (28 / 10)   ->  2.8x too large
    vel_stored    = vel_true    * (10 / 30)   ->  3x too small

Position (PMAX) and action targets are model-independent and are NOT touched.
j1–j3 are genuine DM-J4340 and are left alone.

This script multiplies follower_torq[:, 3:7] by 10/28 and follower_vel[:, 3:7]
by 30/10 in place, so all episodes share one consistent unit convention.

Idempotent: each file gets a `dm_scale_corrected` attr; a second run skips it.
Only the two small (n, 7) float32 datasets are rewritten — image data is
untouched. Pre/post-fix split is decided by the `recorded_at` attribute.

Usage:
  python3 scripts/fix_dm_scale.py --task pick_tape --dry-run
  python3 scripts/fix_dm_scale.py --task pick_tape
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

FIX_TIME = "2026-06-21T20:11:09"          # commit c7d5a7f
TORQ_COLS = slice(3, 7)                    # j4, j5, j6, j7
VEL_COLS = slice(3, 7)
TORQ_FACTOR = 10.0 / 28.0                  # T_MAX 4310 / 4340
VEL_FACTOR = 30.0 / 10.0                   # V_MAX 4310 / 4340
GUARD = "dm_scale_corrected"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="pick_tape")
    p.add_argument("--demos-dir", default=None, help="default outputs/demos/<task>/")
    p.add_argument("--dry-run", action="store_true", help="report only; no writes")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    demos = Path(args.demos_dir) if args.demos_dir else root / "outputs" / "demos" / args.task
    files = sorted(demos.glob("episode_*.hdf5"))
    if not files:
        print(f"no episodes in {demos}")
        sys.exit(1)

    fixed = skipped_native = skipped_done = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            recorded_at = str(f.attrs.get("recorded_at", ""))
            already = GUARD in f.attrs
            pre_fix = recorded_at < FIX_TIME
            t4_before = float(np.abs(f["follower_torq"][:, 3]).mean())
            t1 = float(np.abs(f["follower_torq"][:, 0]).mean())  # j1, must not change

        if already:
            skipped_done += 1
            continue
        if not pre_fix:
            # Native-correct (recorded after the model fix). Stamp so a re-run is clean.
            if not args.dry_run:
                with h5py.File(fp, "a") as f:
                    f.attrs[GUARD] = "native-4310"
            skipped_native += 1
            continue

        # Pre-fix: apply correction to j4-j7 torque + velocity only.
        if not args.dry_run:
            with h5py.File(fp, "a") as f:
                torq = f["follower_torq"][:]
                vel = f["follower_vel"][:]
                torq[:, TORQ_COLS] *= TORQ_FACTOR
                vel[:, VEL_COLS] *= VEL_FACTOR
                f["follower_torq"][...] = torq
                f["follower_vel"][...] = vel
                f.attrs[GUARD] = "4340->4310 torq*10/28 vel*30/10"
            with h5py.File(fp, "r") as f:
                t4_after = float(np.abs(f["follower_torq"][:, 3]).mean())
                t1_after = float(np.abs(f["follower_torq"][:, 0]).mean())
            assert abs(t1 - t1_after) < 1e-6, f"{fp.name}: j1 changed! {t1}->{t1_after}"
            print(f"  ✓ {fp.name}  |torq j4| {t4_before:.3f}->{t4_after:.3f}  (j1 {t1_after:.3f} unchanged)")
        else:
            print(f"  [dry] {fp.name}  |torq j4| {t4_before:.3f} -> {t4_before*TORQ_FACTOR:.3f}")
        fixed += 1

    print(f"\nfixed={fixed}  already-corrected(skipped)={skipped_done}  "
          f"native-4310(stamped)={skipped_native}  total={len(files)}")


if __name__ == "__main__":
    main()
