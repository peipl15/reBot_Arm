#!/usr/bin/env python3
"""Fit a planar cumulative-angle gravity model for the vertical-plane joints
(j2 shoulder, j3 elbow, j4 wrist_flex) directly from calib sweep data.

Why this instead of the URDF/Pinocchio scale fit: the shoulder/elbow gravity
curve is A·cos(θ)+B·sin(θ) where θ is a CUMULATIVE link angle; a single scale on
the URDF can't fix a center-of-mass PHASE mismatch (your self-build differs from
the official CAD). Fitting cos/sin coefficients directly captures the real phase,
needs no URDF joint-frame mapping (the coefficients absorb sign + zero), and
generalizes across the q2/q3/q4 workspace.

Model (planar; links below each joint contribute):
  φ2 = q2 ; φ3 = q2 + e3·q3 ; φ4 = q2 + e3·q3 + e4·q4   (e3,e4 ∈ ±1, searched)
  τ2 ≈ A2c·cosφ2+A2s·sinφ2 + A3c·cosφ3+A3s·sinφ3 + A4c·cosφ4+A4s·sinφ4 + k2 + μ2·sign(v2)
  τ3 ≈ A3c'·cosφ3+A3s'·sinφ3 + A4c'·cosφ4+A4s'·sinφ4 + k3 + μ3·sign(v3)
  τ4 ≈ A4c''·cosφ4+A4s''·sinφ4 + k4 + μ4·sign(v4)
(μ = Coulomb friction term; the cos/sin part is the gravity feedforward.)

Usage (numpy only — no pinocchio needed):
  python3 scripts/fit_gravity_planar.py outputs/gravity_calib_A.csv outputs/gravity_calib_B.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys

import numpy as np

PLANE_JOINTS = [2, 3, 4]   # vertical-plane pitch joints


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", help="gravity_calib_*.csv file(s)")
    p.add_argument("--min-vel", type=float, default=0.03,
                   help="use samples with |vel_j| > this for joint j's fit")
    p.add_argument("--save", default=None,
                   help="write fitted gravity model to this JSON path")
    return p.parse_args()


def regressor(joint, phi, v):
    """Columns for `joint`'s torque given cumulative angles phi={2:φ2,3:φ3,4:φ4}."""
    cols, names = [], []
    # links at or below `joint` contribute: their cumulative angle is φ_k for k>=joint
    for k in (j for j in PLANE_JOINTS if j >= joint):
        cols += [np.cos(phi[k]), np.sin(phi[k])]
        names += [f"cos(φ{k})", f"sin(φ{k})"]
    cols += [np.ones_like(v), np.sign(v)]
    names += ["const", "friction·sgn(v)"]
    return np.column_stack(cols), names


def main():
    args = parse_args()
    rows = []
    for path in args.csv:
        part = list(csv.DictReader(open(path)))
        print(f"  {len(part):>5} samples from {path}")
        rows.extend(part)
    f = lambda k: np.array([float(r[k]) for r in rows])
    Q = {j: f(f"q{j}") for j in PLANE_JOINTS}
    V = {j: f(f"v{j}") for j in PLANE_JOINTS}
    T = {j: f(f"tau{j}") for j in PLANE_JOINTS}
    print(f"Loaded {len(rows)} samples\n")

    # Search relative signs e3,e4 ∈ ±1; pick the combo with best total fit.
    best = None
    for e3, e4 in itertools.product((+1, -1), (+1, -1)):
        phi = {2: Q[2], 3: Q[2] + e3 * Q[3], 4: Q[2] + e3 * Q[3] + e4 * Q[4]}
        total, per = 0.0, {}
        for j in PLANE_JOINTS:
            mask = np.abs(V[j]) > args.min_vel
            if mask.sum() < 30:
                continue
            A, names = regressor(j, {k: phi[k][mask] for k in phi}, V[j][mask])
            sol, *_ = np.linalg.lstsq(A, T[j][mask], rcond=None)
            rms = float(np.sqrt(np.mean((T[j][mask] - A @ sol) ** 2)))
            total += rms
            per[j] = (sol, names, rms, int(mask.sum()))
        if best is None or total < best[0]:
            best = (total, e3, e4, per)

    total, e3, e4, per = best
    per_phi = {2: Q[2], 3: Q[2] + e3 * Q[3], 4: Q[2] + e3 * Q[3] + e4 * Q[4]}
    print(f"Best relative signs:  e3={e3:+d}  e4={e4:+d}   (φ3=q2{e3:+d}·q3, "
          f"φ4=q2{e3:+d}·q3{e4:+d}·q4)\n")
    print("Per-joint planar gravity fit (gravity = cos/sin terms; μ = friction):")
    for j in PLANE_JOINTS:
        if j not in per:
            print(f"  j{j}: (insufficient motion — skip)"); continue
        sol, names, rms, n = per[j]
        tau = T[j][np.abs(V[j]) > args.min_vel]
        mask = np.abs(V[j]) > args.min_vel
        A, _ = regressor(j, {k: per_phi[k][mask] for k in per_phi}, V[j][mask])
        cond = float(np.linalg.cond(A))
        max_grav = float(np.max(np.abs(sol[:-2])))   # largest gravity coef
        bad = cond > 1e3 or max_grav > 50            # ill-conditioned / overfit
        flag = "  ⚠️ ILL-CONDITIONED (coefs不可信,需更丰富激励)" if bad else "  ✓ 良态"
        print(f"  j{j}  (n={n}, rms={rms:.3f} Nm, |τ|≤{np.abs(tau).max():.2f}, "
              f"cond={cond:.1e}, max|grav|={max_grav:.1f}){flag}")
        for nm, c in zip(names, sol):
            tag = "  (friction)" if "sgn" in nm else ("  (bias)" if nm == "const" else "")
            print(f"        {nm:<16} = {c:+8.4f}{tag}")

    if args.save:
        import json
        model = {"e3": e3, "e4": e4, "plane_joints": PLANE_JOINTS,
                 "phi": {"2": "q2", "3": f"q2{e3:+d}*q3", "4": f"q2{e3:+d}*q3{e4:+d}*q4"},
                 "joints": {}}
        for j in PLANE_JOINTS:
            if j not in per:
                continue
            sol, names, rms, n = per[j]
            model["joints"][str(j)] = {
                "coeffs": {nm: float(c) for nm, c in zip(names, sol)},
                "const": float(sol[-2]), "friction": float(sol[-1]),
                "rms": rms, "n": n}
        with open(args.save, "w") as fh:
            json.dump(model, fh, indent=2, ensure_ascii=False)
        print(f"\nSaved gravity model → {args.save}")

    print("\nHow to use as gravity feedforward (drop the friction term):")
    print(f"  φ3 = q2 {e3:+d}*q3 ;  φ4 = q2 {e3:+d}*q3 {e4:+d}*q4")
    print("  tau_g[j] = Σ (Akc*cos(φk) + Aks*sin(φk)) + const_j     # per table above")
    print("Validate on the arm with MIT feedforward: it should float / hold each pose.")
    print("Low rms (≪ |τ| range) ⇒ good. If j2 rms stays high, collect data with j2")
    print("swept at 2–3 different j3 angles (and vice versa) so the terms separate.")


if __name__ == "__main__":
    main()
