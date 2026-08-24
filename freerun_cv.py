#!/usr/bin/env python3
"""No-DA baseline: score the free-running nature run on the same withheld obs.

In: the nature-run npz, the real observations and a split spec.  Out: a
freerun_cv[_tag].npz on the run_da_realobs schema, with forecast == analysis.
"""
# offline, no docker: the free run is already stored in the nature-run npz
# splits come from realobs.split_obs, seeded on (seed, window), so this and the DA
# run get byte-identical withheld sets without sharing any state
#     python freerun_cv.py --w0 128 --nwin 28 --split random80 --tag rnd
#     python freerun_cv.py --w0 128 --nwin 28 --split instrument --tag inst
import os
import sys

import numpy as np

import dgcpm_osse as osse
import realobs as ro

ENKF = os.environ.get("DGCPM_ENKF_ROOT",
                      os.path.dirname(os.path.abspath(__file__)))
NATURE = os.path.join(ENKF, "nature_run_fortran.npz")


def getarg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    w0 = getarg("--w0", 128)
    nwin = getarg("--nwin", 28)
    qc = getarg("--qc", "standard", str)
    split = getarg("--split", "random80", str)
    frac = getarg("--frac", 0.8, float)
    seed = getarg("--seed", 20240502)
    assim_inst = tuple(getarg("--assim", "EMMA", str).split(","))
    debias = "--debias" in sys.argv
    trainw = getarg("--trainwin", 120)
    tag = getarg("--tag", "", str)

    d = np.load(NATURE, allow_pickle=True)
    S, L_cell, MLT_cell = d["states"], d["L"], d["MLT"]
    nT, nP = int(d["nTheta"]), int(d["nPhi"])
    nwin = min(nwin, len(S) - w0)
    w_abs = np.arange(w0, w0 + nwin)

    obs = ro.load_obs(qc=qc)
    bias = ro.estimate_obs_bias(obs, w_train=(0, trainw)) if debias else None

    print("=== FREE RUN (no DA) scored on withheld real obs ===")
    print("windows %d..%d  qc=%s  split=%s%s  bias=%s"
          % (w0, w0 + nwin, qc, split,
             ("  assimilate=%s" % ",".join(assim_inst))
             if split == "instrument" else "", "yes" if debias else "none"))

    ind = ro.split_independence(obs, w_abs, L_cell, MLT_cell, mode=split,
                                frac=frac, seed=seed, assim_inst=assim_inst)
    print()

    cv_r, cv_b, ob_r, ob_b, na, nv = [], [], [], [], [], []
    fields = np.empty((nwin, nT * nP), np.float32)
    res_cv, res_ob, res_w, res_i, res_L, res_M, res_y = [], [], [], [], [], [], []

    for w in range(nwin):
        wa = int(w_abs[w])
        x = osse.to_log(S[wa])                    # free-run state, no DA ever
        o = ro.attach_H(ro.window_obs(obs, wa, bias), L_cell, MLT_cell)
        oa, ov = ro.split_obs(o, wa, mode=split, frac=frac, seed=seed,
                              assim_inst=assim_inst)
        rv, rb = ro.residuals(x, ov), ro.residuals(x, oa)
        rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else np.nan
        mn = lambda v: float(np.mean(v)) if len(v) else np.nan
        cv_r.append(rms(rv)); cv_b.append(mn(rv))
        ob_r.append(rms(rb)); ob_b.append(mn(rb))
        na.append(len(oa)); nv.append(len(ov))
        fields[w] = x.astype(np.float32)
        res_cv.append(rv); res_ob.append(rb)
        res_w.append(np.full(len(ov), wa)); res_i.append(ov.inst); res_L.append(ov.L)
        res_M.append(ov.MLT); res_y.append(ov.y)
        print("w%3d  n=%3d/%3d  free-vs-assim=%.3f  free-vs-WITHHELD=%.3f "
              "(bias %+.3f)" % (wa, na[-1], nv[-1], ob_r[-1], cv_r[-1], cv_b[-1]))

    # same npz schema as run_da_realobs, with forecast==analysis==free run
    out = dict(cv_rmse_f=np.array(cv_r), cv_rmse_a=np.array(cv_r),
               cv_bias_f=np.array(cv_b), cv_bias_a=np.array(cv_b),
               omb_rmse=np.array(ob_r), oma_rmse=np.array(ob_r),
               omb_bias=np.array(ob_b), oma_bias=np.array(ob_b),
               n_assim=np.array(na), n_with=np.array(nv),
               w_abs=w_abs, w0=w0, split=split, qc=qc, debias=debias,
               assim_inst=",".join(assim_inst), frac=frac, seed=seed,
               freerun=True, L=L_cell, MLT=MLT_cell, nTheta=nT, nPhi=nP,
               field_a=fields, field_f=fields,
               res_omb=np.concatenate(res_ob), res_oma=np.concatenate(res_ob),
               res_cv_f=np.concatenate(res_cv), res_cv_a=np.concatenate(res_cv),
               res_cv_w=np.concatenate(res_w), res_cv_inst=np.concatenate(res_i),
               res_cv_L=np.concatenate(res_L), res_cv_MLT=np.concatenate(res_M),
               res_cv_y=np.concatenate(res_y),
               same_cell=ind["same_cell"], loc_weight=ind["weight_median"],
               # kp is a zero placeholder here, NOT the real series; only plotting
               # code could read it, analyse_da_realobs.py never does
               unit_offset=ro.OBS_TO_MODEL_LOG_OFFSET, kp=np.zeros(nwin),
               bias=np.array([0.0 if not bias else bias[n] for n in ro.INSTRUMENTS]))
    name = "freerun_cv%s.npz" % (("_" + tag) if tag else "")
    path = os.path.join(ENKF, name)
    np.savez_compressed(path, **out)

    r = out["res_cv_f"]
    print("\nsaved", path)
    print("FREE RUN on %d withheld obs: RMSE=%.3f dex  bias=%+.3f dex  "
          "median|r|=%.3f" % (len(r), np.sqrt(np.mean(r ** 2)), r.mean(),
                              np.median(np.abs(r))))
    for name_i, i in ro.INST_ID.items():
        m = out["res_cv_inst"] == i
        if m.sum():
            print("   %-8s n=%5d  RMSE=%.3f  bias=%+.3f"
                  % (name_i, int(m.sum()), np.sqrt(np.mean(r[m] ** 2)), r[m].mean()))


if __name__ == "__main__":
    main()
