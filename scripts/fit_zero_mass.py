#!/usr/bin/env python3
"""Solve joint zero-offsets + per-link density scales from static-pose torque data.

Model:  q_urdf = SIGN .* (q_ours - zero)          (SIGN fixed; j4 flipped)
        link i mass m_i = alpha_i * m0_i          (m0 = URDF prior; geometry/COM
                                                   unchanged, so first-moment also
                                                   scales by alpha_i -> gravity scales)
        tau_pred[j] = SIGN[j] * g_urdf(q_urdf)[j] (torque transforms with the joint sign)

We fit by nonlinear least squares over { zero[2,3,4,5], alpha[links] } against the
bracket-averaged static holding torques from calib_zero_poses.py. Two regularizers
keep it well-posed and physically honest:
  - lam_mass pulls every alpha toward 1.0 (we 3D-printed to spec; density is CLOSE,
    not free). Strong by default so mass only moves where the data clearly demands.
  - lam_zero lightly pulls zeros toward 0.

Gravity ignores rotational inertia, so scaling only mass+first-moment (one alpha
per link) captures a uniform density error exactly -- matching "same print, unknown
material density". j1 (vertical axis) and j6 contribute ~no gravity and are not
solved for zero.

Usage:
  python3 scripts/fit_zero_mass.py outputs/zero_poses.csv --save configs/zero_mass_calib.json
  python3 scripts/fit_zero_mass.py outputs/zero_poses.csv --no-mass   # zeros only (alpha=1)
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
# Per-joint sign mapping our user frame -> URDF frame, DATA-DERIVED 2026-06-30
# from a 16-combo brute-force search over the static poses (lowest RMS, physical
# zeros). The vertical-plane pitch joints j2/j3/j4 are ALL mounted opposite to the
# URDF convention (not just j4 as the old sweep-era guess assumed). j5/j6 carry ~no
# gravity so their sign is immaterial here.
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: 1.0, 6: 1.0}
SOLVE_ZERO = [2, 3, 4, 5]          # j1 vertical / j6 negligible -> fixed at 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+")
    p.add_argument("--urdf", default=URDF)
    p.add_argument("--save", default=None)
    p.add_argument("--no-mass", action="store_true", help="fix all alpha=1, solve zeros only")
    p.add_argument("--lam-mass", type=float, default=40.0,
                   help="ridge pulling alpha->1 (strong: density is close to spec)")
    p.add_argument("--lam-zero", type=float, default=1.0,
                   help="ridge pulling zero->0 (light)")
    p.add_argument("--mass-links", default="1,2,3,4,5,6",
                   help="which URDF links get a density scale (default all 6)")
    return p.parse_args()


def main():
    args = parse_args()
    import pinocchio as pin
    from scipy.optimize import least_squares

    rows = []
    for path in args.csv:
        rows.extend(list(csv.DictReader(open(path))))
    n = len(rows)
    if n == 0:
        print("no poses"); sys.exit(1)
    Q = {j: np.array([float(r[f"q{j}"]) for r in rows]) for j in ARMJ}
    TAU = {j: np.array([float(r[f"tau{j}"]) for r in rows]) for j in SOLVE_ZERO}
    print(f"Loaded {n} static poses from {len(args.csv)} file(s)")

    model = pin.buildModelFromUrdf(args.urdf)
    data = model.createData()
    m0 = np.array([model.inertias[i].mass for i in range(1, model.njoints)])  # 6 link masses
    mass_links = [] if args.no_mass else [int(x) for x in args.mass_links.split(",")]

    nz = len(SOLVE_ZERO)
    na = len(mass_links)

    def unpack(x):
        zeros = {j: x[k] for k, j in enumerate(SOLVE_ZERO)}
        alpha = np.ones(len(m0))
        for k, li in enumerate(mass_links):
            alpha[li - 1] = x[nz + k]
        return zeros, alpha

    def set_mass(alpha):
        for i in range(1, model.njoints):
            ii = model.inertias[i]
            ii.mass = alpha[i - 1] * m0[i - 1]   # lever (COM) unchanged -> first moment scales

    def predict(zeros, alpha):
        set_mass(alpha)
        pred = {j: np.empty(n) for j in SOLVE_ZERO}
        for s in range(n):
            qv = np.array([SIGN[k] * (Q[k][s] - zeros.get(k, 0.0)) for k in ARMJ])
            qv[0] = 0.0   # j1 vertical: gravity-irrelevant
            g = pin.computeGeneralizedGravity(model, data, qv)
            for j in SOLVE_ZERO:
                pred[j][s] = SIGN[j] * g[j - 1]
        return pred

    def residual(x):
        zeros, alpha = unpack(x)
        pred = predict(zeros, alpha)
        r = [pred[j][s] - TAU[j][s] for s in range(n) for j in SOLVE_ZERO]
        r += [np.sqrt(args.lam_zero) * x[k] for k in range(nz)]           # zero -> 0
        r += [np.sqrt(args.lam_mass) * (x[nz + k] - 1.0) for k in range(na)]  # alpha -> 1
        return np.array(r)

    def torque_rms(x):
        zeros, alpha = unpack(x)
        pred = predict(zeros, alpha)
        e = np.array([pred[j][s] - TAU[j][s] for s in range(n) for j in SOLVE_ZERO])
        return float(np.sqrt(np.mean(e ** 2)))

    x0 = np.array([0.0] * nz + [1.0] * na)
    rms0 = torque_rms(x0)
    lb = [-1.0] * nz + [0.5] * na
    ub = [1.0] * nz + [1.5] * na
    sol = least_squares(residual, x0, bounds=(lb, ub), xtol=1e-12, ftol=1e-12)
    zeros, alpha = unpack(sol.x)
    rms1 = torque_rms(sol.x)

    print(f"\n  torque-match RMS:  {rms0:.3f} Nm  ->  {rms1:.3f} Nm")
    print("\n  joint   zero(rad)   zero(deg)")
    for j in SOLVE_ZERO:
        print(f"   j{j}     {zeros[j]:+.4f}     {np.degrees(zeros[j]):+6.2f}")
    if mass_links:
        print("\n  link   alpha   density-dev   m_urdf(kg) -> m_fit(kg)")
        for li in mass_links:
            a = alpha[li - 1]
            print(f"   L{li}    {a:.3f}    {(a-1)*100:+5.1f}%      {m0[li-1]:.3f} -> {a*m0[li-1]:.3f}")
        print(f"\n  (alpha 偏离 1 越多 = 该连杆密度与官方差越多;强正则下只有数据明确要求才会动)")

    if args.save:
        out = {
            "urdf": args.urdf,
            "sign": {str(j): SIGN[j] for j in ARMJ},
            "zero": {str(j): (zeros[j] if j in zeros else 0.0) for j in ARMJ},
            "alpha": {str(li): float(alpha[li - 1]) for li in range(1, len(m0) + 1)},
            "m0_urdf": {str(li): float(m0[li - 1]) for li in range(1, len(m0) + 1)},
            "n_poses": n,
            "rms_before": rms0, "rms_after": rms1,
            "lam_mass": args.lam_mass, "lam_zero": args.lam_zero,
            "note": "q_urdf = SIGN*(q_ours - zero); link mass = alpha*m0; gravity via Pinocchio.",
        }
        json.dump(out, open(args.save, "w"), indent=2)
        print(f"\n  saved -> {args.save}")


if __name__ == "__main__":
    main()
