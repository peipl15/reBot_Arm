#!/usr/bin/env python3
"""Fit per-joint gravity corrections from a calib sweep (offline, no hardware).

Reads a CSV from calib_gravity_sweep.py, computes the *predicted* gravity torque
g(q) from the official reBot URDF (via Pinocchio), and fits, per joint, the
linear model

    tau_meas,j  ≈  s_j · g_pred,j(q)  +  b_j  +  f_j · sign(vel_j)

  s_j : gravity SCALE correction  (this is the ×1.55-style factor you apply on
        top of the official g(q) — captures mass/CoM mismatch of your build)
  b_j : constant bias / zero offset (diagnostic)
  f_j : Coulomb friction magnitude (bonus — useful for a friction-aware watchdog)

Because the sweep recorded both directions, including the `sign(vel)` column
cleanly separates direction-dependent friction from gravity in one regression.

Joint frame: our config q (user frame) must be mapped to the URDF's convention.
Adjust SIGN / ZERO below if residuals are large or s_j comes out negative — that
means a joint's direction/zero is flipped relative to the URDF.

Usage (run in an env with pinocchio — e.g. the reBotArm_control_py uv venv):
  python3 scripts/fit_gravity.py outputs/gravity_calib_<ts>.csv
  python3 scripts/fit_gravity.py <csv> --urdf <path> --min-vel 0.05
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# --- joint frame mapping: q_urdf[j] = SIGN[j] * (q_ours[j] - ZERO[j]) ----------
# Our config j1..j6 (user frame) -> URDF revolute config (6-DOF arm; gripper j7
# excluded from gravity). Start with identity; fix per residuals.
SIGN = {1: +1.0, 2: +1.0, 3: +1.0, 4: +1.0, 5: +1.0, 6: +1.0}
ZERO = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
ARM_JOINTS = [1, 2, 3, 4, 5, 6]   # joints in the gravity model (no gripper)

DEFAULT_URDF = (Path.home() / "code" / "reBotArm_control_py" / "urdf"
                / "reBot-DevArm_fixend_description" / "urdf" / "reBot-DevArm_fixend.urdf")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+",
                   help="one or more gravity_calib_*.csv (collected from different "
                        "safe poses); they're concatenated and fit jointly so each "
                        "joint's zero is shared across passes (resolves coupling)")
    p.add_argument("--urdf", default=str(DEFAULT_URDF), help="path to reBot URDF")
    p.add_argument("--min-vel", type=float, default=0.03,
                   help="only use samples where |vel_j| > this (rad/s) so sign(vel) "
                        "is meaningful (default 0.03)")
    return p.parse_args()


def load_pinocchio(urdf_path):
    try:
        import pinocchio as pin
    except ImportError:
        print("ERROR: pinocchio not importable. Run this in the reBotArm_control_py "
              "uv venv (it has pinocchio), or `uv pip install pin` into this venv.")
        sys.exit(1)
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    print(f"URDF: {urdf_path}")
    print(f"  pinocchio model: nq={model.nq} nv={model.nv}")
    print(f"  joint order: {[n for n in model.names[1:]]}")  # names[0] = 'universe'
    if model.nq != len(ARM_JOINTS):
        print(f"  [warn] model nq={model.nq} != {len(ARM_JOINTS)} arm joints — "
              f"check ARM_JOINTS / URDF (continuous joints use cos/sin and need "
              f"a different mapping).")
    return pin, model, data


def main():
    args = parse_args()
    rows = []
    for path in args.csv:
        part = list(csv.DictReader(open(path)))
        print(f"  {len(part):>5} samples from {path}")
        rows.extend(part)
    if not rows:
        print("empty CSV"); sys.exit(1)
    print(f"Loaded {len(rows)} samples total from {len(args.csv)} file(s)")
    pin, model, data = load_pinocchio(args.urdf)

    f = lambda r, k: float(r[k])
    Q = {j: np.array([f(r, f"q{j}") for r in rows]) for j in ARM_JOINTS}   # our frame
    V = {j: np.array([f(r, f"v{j}") for r in rows]) for j in ARM_JOINTS}
    T = {j: np.array([f(r, f"tau{j}") for r in rows]) for j in ARM_JOINTS}
    n = len(rows)

    # Current best mapping per joint: q_urdf[j] = sign*(q_ours[j] - zero).
    MAP = {j: (SIGN[j], ZERO[j]) for j in ARM_JOINTS}

    def gpred_col(target_j, samples):
        """Vector of g_pred for target_j over `samples` indices, using MAP."""
        out = np.empty(len(samples))
        for k, idx in enumerate(samples):
            qv = np.array([MAP[j][0] * (Q[j][idx] - MAP[j][1]) for j in ARM_JOINTS])
            g = pin.computeGeneralizedGravity(model, data, qv)
            sj = ARM_JOINTS.index(target_j)
            out[k] = MAP[target_j][0] * float(g[sj])   # back to our frame
        return out

    def fit_one(target_j, samples):
        gp = gpred_col(target_j, samples)
        v = V[target_j][samples]; t = T[target_j][samples]
        A = np.column_stack([gp, np.ones(len(samples)), np.sign(v)])
        sol, *_ = np.linalg.lstsq(A, t, rcond=None)
        rms = float(np.sqrt(np.mean((t - A @ sol) ** 2)))
        return sol, rms  # sol = [scale, bias, friction]

    # Which joints were actually swept (enough motion + angle range to solve).
    solvable, moving_idx = [], {}
    for j in ARM_JOINTS:
        mv = np.where(np.abs(V[j]) > args.min_vel)[0]
        if len(mv) >= 30 and np.ptp(Q[j][mv]) >= 0.1:
            solvable.append(j); moving_idx[j] = mv
    print(f"\nSwept (solvable) joints: {solvable}")
    print("Coordinate-descent solving sign+zero (re-iterates to undo cross-coupling)...")

    for it in range(4):
        for j in solvable:
            mv = moving_idx[j]
            sub = mv[:: max(1, len(mv) // 120)]
            signs = (+1.0, -1.0) if it == 0 else (MAP[j][0],)   # lock sign after pass 0
            best = None
            for sign in signs:
                for zero in np.arange(-np.pi, np.pi, 0.05):
                    MAP[j] = (sign, zero)
                    sol, rms = fit_one(j, sub)
                    if sol[0] > 0 and (best is None or rms < best[0]):
                        best = (rms, sign, zero)
            if best is not None:
                MAP[j] = (best[1], best[2])

    results = {}
    for j in solvable:
        sol, rms = fit_one(j, moving_idx[j])     # final fit, full data
        results[j] = dict(scale=sol[0], bias=sol[1], friction=sol[2], rms=rms,
                          sign=MAP[j][0], zero=MAP[j][1], n=len(moving_idx[j]))

    print("\nPer-joint result:  tau ≈ s·g_pred(sign·(q−zero)) + b + f·sign(v)")
    print(f"{'joint':<6} {'n':>5} {'sign':>5} {'zero':>7} {'scale s':>9} "
          f"{'bias b':>8} {'friction':>9} {'rms':>7}")
    print("-" * 62)
    for j in ARM_JOINTS:
        if j not in results:
            print(f"j{j:<5}   (insufficient excitation — skip)"); continue
        r = results[j]
        print(f"j{j:<5} {r['n']:>5} {r['sign']:>+5.0f} {r['zero']:>7.3f} "
              f"{r['scale']:>9.3f} {r['bias']:>8.3f} {abs(r['friction']):>9.3f} {r['rms']:>7.3f}")

    print("\nHow to read:")
    print("  sign/zero = the auto-solved map from our frame to the URDF frame.")
    print("  scale s ≈ 1.0  → URDF mass matches your build for that joint.")
    print("  s > 1 (e.g. 1.5) → your link/payload is heavier than URDF (apply "
          "tau_g[j] *= s in gravity comp, like the official ×1.55 hint).")
    print("  low rms (≪ gravity range) → good fit. high rms → more configs needed.")
    print("  friction f → Coulomb torque; feed into a friction-aware watchdog.")
    if results:
        print("\nApply on official g(q) — scale vector:")
        print("  " + "  ".join(f"j{j}:{results[j]['scale']:.3f}" for j in results))
        print("Joint→URDF map (sign, zero rad):")
        print("  " + "  ".join(f"j{j}:({results[j]['sign']:+.0f},{results[j]['zero']:+.3f})"
                               for j in results))


if __name__ == "__main__":
    main()
