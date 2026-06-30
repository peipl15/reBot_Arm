#!/usr/bin/env python3
"""Solve joint zero-offsets from the slow BIDIRECTIONAL sweep CSVs (calib_gravity_sweep.py).

Far more data than the static poses: ~20k samples densely covering the j2/j3/j4
ranges in both directions. During slow motion the dynamics reduce to
    tau_meas[j] = scale[j] * SIGN[j] * g_urdf(q_urdf)[j] + fric[j]*sign(v[j])
(inertial/Coriolis negligible at <0.3 rad/s; Coulomb friction flips with v and is
fit per joint -> the bidirectional passes let it separate cleanly from gravity).

  q_urdf = SIGN .* (q_ours - zero)        SIGN data-derived (j2/j3/j4 flipped)
  scale[2,3]: the DM-J4340P feedback-torque decode is ~1.5x high (cancels in MIT);
              j4 (DM-J4310) decode already correct -> scale fixed 1.

Fits zero[2,3,4,5] + scale[2,3] + fric[2,3,4,5]. j1 vertical / j6 negligible.

Usage:
  python3 scripts/fit_zero_sweep.py outputs/gravity_calib_*.csv --save configs/zero_calib.json
"""
from __future__ import annotations
import argparse, csv, glob, json, sys
import numpy as np

URDF = ("/home/yinzi/code/reBotArm_control_py/urdf/"
        "reBot-DevArm_fixend_description/urdf/reBot-DevArm_fixend.urdf")
ARMJ = [1, 2, 3, 4, 5, 6]
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: 1.0, 6: 1.0}
SWEPT = [2, 3, 4, 5]
SCALE_JOINTS = [2, 3]          # DM-J4340P decode-scale unknowns (others fixed 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+")
    p.add_argument("--urdf", default=URDF)
    p.add_argument("--min-vel", type=float, default=0.04, help="rad/s; only moving samples")
    p.add_argument("--max-per-joint", type=int, default=600, help="subsample cap per joint")
    p.add_argument("--save", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    import pinocchio as pin
    from scipy.optimize import least_squares

    files = []
    for pat in args.csv:
        files.extend(sorted(glob.glob(pat)))
    rows = []
    for f in files:
        rows.extend(list(csv.DictReader(open(f))))
    if not rows:
        print("no rows"); sys.exit(1)
    Q = {j: np.array([float(r[f"q{j}"]) for r in rows]) for j in ARMJ}
    V = {j: np.array([float(r[f"v{j}"]) for r in rows]) for j in SWEPT}
    T = {j: np.array([float(r[f"tau{j}"]) for r in rows]) for j in SWEPT}
    n = len(rows)

    # moving-sample indices per joint (subsampled)
    idx = {}
    for j in SWEPT:
        sel = np.where(np.abs(V[j]) > args.min_vel)[0]
        if len(sel) > args.max_per_joint:
            sel = sel[np.linspace(0, len(sel) - 1, args.max_per_joint).astype(int)]
        idx[j] = sel
    print(f"Loaded {n} samples from {len(files)} files; moving samples per joint: "
          + " ".join(f"j{j}={len(idx[j])}" for j in SWEPT))

    model = pin.buildModelFromUrdf(args.urdf)
    data = model.createData()

    # param layout: zero[2,3,4,5], scale[2,3], fric[2,3,4,5]
    NZ, NS, NF = len(SWEPT), len(SCALE_JOINTS), len(SWEPT)

    def unpack(x):
        zero = {j: x[k] for k, j in enumerate(SWEPT)}
        scale = {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
        for k, j in enumerate(SCALE_JOINTS):
            scale[j] = x[NZ + k]
        fric = {j: x[NZ + NS + k] for k, j in enumerate(SWEPT)}
        return zero, scale, fric

    # cache gravity per used sample index (recompute each eval since zero changes q)
    used = sorted(set().union(*[set(idx[j].tolist()) for j in SWEPT]))

    def grav_at(zero):
        g = {}
        for s in used:
            qv = np.array([SIGN[k] * (Q[k][s] - zero.get(k, 0.0)) for k in ARMJ])
            qv[0] = 0.0
            g[s] = pin.computeGeneralizedGravity(model, data, qv)
        return g

    def residual(x):
        zero, scale, fric = unpack(x)
        g = grav_at(zero)
        r = []
        for j in SWEPT:
            for s in idx[j]:
                pred = scale[j] * SIGN[j] * g[s][j - 1] + fric[j] * np.sign(V[j][s])
                r.append(pred - T[j][s])
        return np.array(r)

    def rms(x):
        return float(np.sqrt(np.mean(residual(x) ** 2)))

    x0 = np.array([0.0] * NZ + [1.0] * NS + [0.0] * NF)
    lb = [-1.0] * NZ + [0.3] * NS + [-3.0] * NF
    ub = [1.0] * NZ + [3.0] * NS + [3.0] * NF
    r0 = rms(x0)
    sol = least_squares(residual, x0, bounds=(lb, ub), xtol=1e-12, ftol=1e-12)
    zero, scale, fric = unpack(sol.x)
    r1 = rms(sol.x)

    print(f"\n  RMS:  {r0:.3f} -> {r1:.3f} Nm")
    print("\n  joint   zero(rad)  zero(deg)   scale   friction(Nm)")
    for j in SWEPT:
        print(f"   j{j}    {zero[j]:+.4f}   {np.degrees(zero[j]):+6.2f}     "
              f"{scale[j]:.2f}    {fric[j]:+.3f}")

    if args.save:
        out = {"urdf": args.urdf, "method": "bidirectional sweep + friction",
               "sign": {str(j): SIGN[j] for j in ARMJ},
               "zero": {str(j): (zero[j] if j in zero else 0.0) for j in ARMJ},
               "torque_scale": {str(j): scale[j] for j in SWEPT},
               "friction": {str(j): fric[j] for j in SWEPT},
               "n_samples": int(sum(len(idx[j]) for j in SWEPT)), "rms": r1}
        json.dump(out, open(args.save, "w"), indent=2)
        print(f"\n  saved -> {args.save}")


if __name__ == "__main__":
    main()
