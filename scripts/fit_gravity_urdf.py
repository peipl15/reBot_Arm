#!/usr/bin/env python3
"""3D gravity parameter identification using the reBot URDF + Pinocchio.

The planar model can't represent the real (non-coplanar / out-of-plane) coupling
— forcing shared planar coupling made the fit worse, proving 3D is needed. Here
we use Pinocchio's EXACT gravity regressor Y(q): gravity = Y(q)·φ, where φ are
each link's mass + first-moment parameters. We identify a correction to the URDF
priors (regularized toward them so unidentifiable params stay put), jointly over
all joints' data → physically-consistent coupling.

Joint frame: q_urdf[j] = SIGN[j]·(q_ours[j] − zero[j]). SIGN fixed from earlier
evidence (j4 flipped); zeros for j2..j5 are solved (the home offset between our
encoder zero and the URDF home). Friction handled per joint via sign(v).

Run in the venv with pinocchio + scipy.
  python3 scripts/fit_gravity_urdf.py outputs/gravity_calib_*.csv --save configs/gravity_urdf.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

URDF = ("/home/yinzi/code/reBotArm_control_py/urdf/"
        "reBot-DevArm_fixend_description/urdf/reBot-DevArm_fixend.urdf")
ARMJ = [1, 2, 3, 4, 5, 6]
SIGN = {1: 1.0, 2: 1.0, 3: 1.0, 4: -1.0, 5: 1.0, 6: 1.0}   # j4 flipped (matches manual flip)
SWEPT = [2, 3, 4, 5]          # joints with moving samples → contribute rows + friction
SOLVE_ZERO = [2, 3, 4, 5]     # zeros to identify (j1 vertical: no gravity; j6 minor)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+")
    p.add_argument("--urdf", default=URDF)
    p.add_argument("--min-vel", type=float, default=0.03)
    p.add_argument("--lam", type=float, default=1.0,
                   help="ridge strength pulling φ toward URDF prior (relative)")
    p.add_argument("--save", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    import pinocchio as pin
    from scipy.optimize import minimize

    rows = []
    for path in args.csv:
        rows.extend(list(csv.DictReader(open(path))))
    gg = lambda k: np.array([float(r[k]) for r in rows])
    Q = {j: gg(f"q{j}") for j in ARMJ}
    V = {j: gg(f"v{j}") for j in SWEPT}
    T = {j: gg(f"tau{j}") for j in SWEPT}
    n = len(rows)
    print(f"Loaded {n} samples")

    model = pin.buildModelFromUrdf(args.urdf)
    data = model.createData()
    phi0 = np.concatenate([model.inertias[i].toDynamicParameters()
                           for i in range(1, model.njoints)])     # 60
    NP = phi0.shape[0]
    scale = np.abs(phi0) + 0.02         # per-param regularization scale (relative)

    # moving-sample indices per swept joint, plus a balancing weight
    mov = {j: np.where(np.abs(V[j]) > args.min_vel)[0] for j in SWEPT}
    w = {j: 1.0 / (np.std(T[j][mov[j]]) + 1e-6) for j in SWEPT}
    fcols = {j: idx for idx, j in enumerate(SWEPT)}      # friction column per joint
    NF = len(SWEPT)

    def regressor_rows(idxs, zeros):
        """For each swept joint, the SIGN·Y[jrow] over its (subset of) moving samples.
        Returns dict j -> (A_phi[n_j,60], fric_onehot[n_j,NF], y[n_j], weight)."""
        zmap = {1: 0.0, 6: 0.0}
        for j, z in zip(SOLVE_ZERO, zeros):
            zmap[j] = z
        out = {}
        for j in SWEPT:
            sel = np.intersect1d(mov[j], idxs)
            if len(sel) == 0:
                continue
            Aphi = np.empty((len(sel), NP)); y = np.empty(len(sel))
            fr = np.zeros((len(sel), NF))
            for r, s in enumerate(sel):
                q = np.array([SIGN[k] * (Q[k][s] - zmap.get(k, 0.0)) for k in ARMJ])
                q[0] = 0.0     # j1 vertical axis: irrelevant to gravity
                Y = pin.computeJointTorqueRegressor(model, data, q, np.zeros(6), np.zeros(6))
                Aphi[r] = SIGN[j] * Y[j - 1]        # joint j row (0-indexed j-1)
                y[r] = T[j][s]
                fr[r, fcols[j]] = np.sign(V[j][s])
            out[j] = (Aphi, fr, y, w[j])
        return out

    def solve(blocks, lam):
        """Regularized LS for [phi(60); fric(NF)] given assembled blocks."""
        A, b = [], []
        for j, (Aphi, fr, y, wj) in blocks.items():
            A.append(np.hstack([Aphi, fr]) * wj); b.append(y * wj)
        # ridge: pull phi toward phi0 (scaled), friction toward 0
        R = np.zeros((NP + NF, NP + NF)); rb = np.zeros(NP + NF)
        for i in range(NP):
            R[i, i] = np.sqrt(lam) / scale[i]; rb[i] = np.sqrt(lam) * phi0[i] / scale[i]
        for i in range(NF):
            R[NP + i, NP + i] = np.sqrt(lam) * 0.01
        A.append(R); b.append(rb)
        A = np.vstack(A); b = np.concatenate(b)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x

    def resid_rms(blocks, x):
        per = {}
        for j, (Aphi, fr, y, wj) in blocks.items():
            pred = Aphi @ x[:NP] + fr @ x[NP:]
            per[j] = float(np.sqrt(np.mean((y - pred) ** 2)))
        return per

    # ---- optimize zeros on a subsample ----
    sub = np.arange(0, n, max(1, n // 800))
    print(f"Optimizing zeros {SOLVE_ZERO} on {len(sub)} subsample pts (Powell)...")

    def obj(zeros):
        blocks = regressor_rows(sub, zeros)
        x = solve(blocks, args.lam)
        per = resid_rms(blocks, x)
        return sum(w[j] * per[j] for j in per)

    res = minimize(obj, x0=np.zeros(len(SOLVE_ZERO)), method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 200})
    zeros = res.x
    print("  solved zeros (rad): " + "  ".join(f"j{j}={z:+.3f}" for j, z in zip(SOLVE_ZERO, zeros)))

    # ---- final solve on full data ----
    blocks = regressor_rows(np.arange(n), zeros)
    x = solve(blocks, args.lam)
    per = resid_rms(blocks, x)
    phi = x[:NP]; fric = x[NP:]

    print("\nPer-joint rms vs Coulomb friction band (rms below friction ⇒ 'stays put'):")
    for j in SWEPT:
        f = abs(fric[fcols[j]])
        flag = "✓ under friction" if per[j] <= f * 1.2 else "✗ above friction (still drifts)"
        print(f"  j{j}: rms={per[j]:.3f}  friction={f:.3f} Nm   {flag}")

    # mass/CoM change vs prior (sanity)
    print("\nIdentified vs URDF mass per link:")
    for bi in range(1, model.njoints):
        m_new = phi[(bi - 1) * 10]
        print(f"  body{bi} {model.names[bi]:<8} m: {model.inertias[bi].mass:.3f} → {m_new:.3f}")

    if args.save:
        out = {"urdf": args.urdf, "sign": {str(j): SIGN[j] for j in ARMJ},
               "zero": {str(j): (float(dict(zip(SOLVE_ZERO, zeros)).get(j, 0.0)))
                        for j in ARMJ},
               "phi": phi.tolist(), "friction": {str(j): float(fric[fcols[j]]) for j in SWEPT},
               "rms": {str(j): per[j] for j in SWEPT}}
        json.dump(out, open(args.save, "w"), indent=2)
        print(f"\nSaved 3D gravity model → {args.save}")


if __name__ == "__main__":
    main()
