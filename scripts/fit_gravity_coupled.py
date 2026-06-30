#!/usr/bin/env python3
"""Coupled planar gravity identification — shares each link's parameters across
all joints it loads, enforcing physical consistency of the shoulder/elbow/wrist
coupling (fixes "j2 equilibrium drifts with j3").

Physics: for a planar chain, the φk-dependent part of the gravity torque comes
from link-k-and-beyond's CoM measured from joint k; that SAME (Ak,Bk) appears in
every joint j<=k's torque. Fitting joints independently lets these disagree
(the bug). Here we fit ONE shared (Ak,Bk) per link, jointly across all data:

  φ2=q2 ; φ3=q2+e3·q3 ; φ4=q2+e3·q3+e4·q4              (e3,e4 ∈ ±1, searched)
  τ2 = A2cφ2+B2sφ2 + A3cφ3+B3sφ3 + A4cφ4+B4sφ4 + c2 + μ2·sgn(v2)
  τ3 =               A3cφ3+B3sφ3 + A4cφ4+B4sφ4 + c3 + μ3·sgn(v3)
  τ4 =                             A4cφ4+B4sφ4 + c4 + μ4·sgn(v4)
shared: A2,B2,A3,B3,A4,B4   per-joint: const cj, friction μj.

Output JSON matches fit_gravity_planar (per-joint coeffs) so gravity_float_test
works unchanged. numpy only.

Usage:
  python3 scripts/fit_gravity_coupled.py outputs/gravity_calib_*.csv --save configs/gravity_planar.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys

import numpy as np

PLANE = [2, 3, 4]
# shared gravity columns: index of (Ak,Bk) pair per link k
GCOL = {2: (0, 1), 3: (2, 3), 4: (4, 5)}     # columns 0..5
CCOL = {2: 6, 3: 7, 4: 8}                     # const columns
FCOL = {2: 9, 3: 10, 4: 11}                   # friction columns
NPARAM = 12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+")
    p.add_argument("--min-vel", type=float, default=0.03)
    p.add_argument("--save", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for path in args.csv:
        part = list(csv.DictReader(open(path)))
        print(f"  {len(part):>5} samples from {path}")
        rows.extend(part)
    g = lambda k: np.array([float(r[k]) for r in rows])
    Q = {j: g(f"q{j}") for j in PLANE}
    V = {j: g(f"v{j}") for j in PLANE}
    T = {j: g(f"tau{j}") for j in PLANE}
    print(f"Loaded {len(rows)} samples\n")

    def build(phi, weights):
        """Stacked weighted design matrix + target over all joints' moving samples."""
        Arows, yrows, tags = [], [], []
        for j in PLANE:
            mask = np.abs(V[j]) > args.min_vel
            n = int(mask.sum())
            if n < 30:
                continue
            w = weights[j]
            M = np.zeros((n, NPARAM))
            for k in (kk for kk in PLANE if kk >= j):
                ca, sa = GCOL[k]
                M[:, ca] = np.cos(phi[k][mask])
                M[:, sa] = np.sin(phi[k][mask])
            M[:, CCOL[j]] = 1.0
            M[:, FCOL[j]] = np.sign(V[j][mask])
            Arows.append(M * w)
            yrows.append(T[j][mask] * w)
            tags.append((j, n, mask))
        return np.vstack(Arows), np.concatenate(yrows), tags

    # per-joint weight = 1/std(torque) so big-torque j2 doesn't dominate the shared fit
    weights = {j: 1.0 / (np.std(T[j][np.abs(V[j]) > args.min_vel]) + 1e-6) for j in PLANE}

    best = None
    for e3, e4 in itertools.product((+1, -1), (+1, -1)):
        phi = {2: Q[2], 3: Q[2] + e3 * Q[3], 4: Q[2] + e3 * Q[3] + e4 * Q[4]}
        A, y, tags = build(phi, weights)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        # unweighted per-joint rms for reporting
        per = {}
        tot = 0.0
        for j in PLANE:
            mask = np.abs(V[j]) > args.min_vel
            pred = np.zeros(int(mask.sum()))
            for k in (kk for kk in PLANE if kk >= j):
                ca, sa = GCOL[k]
                pred += sol[ca] * np.cos(phi[k][mask]) + sol[sa] * np.sin(phi[k][mask])
            pred += sol[CCOL[j]] + sol[FCOL[j]] * np.sign(V[j][mask])
            rms = float(np.sqrt(np.mean((T[j][mask] - pred) ** 2)))
            per[j] = rms; tot += rms
        if best is None or tot < best[0]:
            best = (tot, e3, e4, sol, per)

    tot, e3, e4, sol, per = best
    print(f"Best signs: e3={e3:+d} e4={e4:+d}   shared link params (A=cos, B=sin):")
    for k in PLANE:
        ca, sa = GCOL[k]
        amp = np.hypot(sol[ca], sol[sa])
        print(f"  link{k}: A{k}={sol[ca]:+7.3f}  B{k}={sol[sa]:+7.3f}  (amp {amp:.2f} Nm)")
    print("Per-joint const / friction / rms:")
    for j in PLANE:
        print(f"  j{j}: const={sol[CCOL[j]]:+6.3f}  friction={abs(sol[FCOL[j]]):.3f}  "
              f"rms={per[j]:.3f} Nm")

    if args.save:
        import json
        model = {"e3": e3, "e4": e4, "plane_joints": PLANE, "coupled": True,
                 "phi": {"2": "q2", "3": f"q2{e3:+d}*q3", "4": f"q2{e3:+d}*q3{e4:+d}*q4"},
                 "joints": {}}
        for j in PLANE:
            coeffs = {}
            for k in (kk for kk in PLANE if kk >= j):
                ca, sa = GCOL[k]
                coeffs[f"cos(φ{k})"] = float(sol[ca])
                coeffs[f"sin(φ{k})"] = float(sol[sa])
            coeffs["const"] = float(sol[CCOL[j]])
            coeffs["friction·sgn(v)"] = float(sol[FCOL[j]])
            model["joints"][str(j)] = {"coeffs": coeffs, "const": float(sol[CCOL[j]]),
                                       "friction": float(sol[FCOL[j]]), "rms": per[j]}
        json.dump(model, open(args.save, "w"), indent=2, ensure_ascii=False)
        print(f"\nSaved coupled gravity model → {args.save}")


if __name__ == "__main__":
    main()
