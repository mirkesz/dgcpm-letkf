#!/usr/bin/env python3
"""LETKF assimilation of REAL Arase/EMMA/AWDANet densities, scored by cross-validation.

In: the nature-run npz for grid and IC, the real observations, a split spec.  Out:
a da_realobs[_dry][_tag].npz with per-window diagnostics and pooled residuals.
"""
# differs from the OSSE in one decisive way: there is no truth, so each window's
# obs are split and only the WITHHELD set is scored, against the free-running model
# on the identical withheld sets (freerun_cv.py)
# the constraints (bounds, increment cap, bc_rings=2, R=1 dex, k=16, localization)
# are inherited unchanged from the OSSE work and are deliberately not re-tuned here
#     python run_da_realobs.py --dry                     # assembly check, no docker
#     python run_da_realobs.py --w0 128 --nwin 28 --k 16 --split random80 --tag rnd
#     python run_da_realobs.py --w0 128 --nwin 28 --k 16 --split instrument --tag inst
#     python run_da_realobs.py --w0 128 --nwin 28 --k 16 --split random80 --debias --tag rnddb
import os
import sys
from datetime import timedelta

import numpy as np

import dgcpm_engine as e
import dgcpm_osse as osse
import realobs as ro
from dgcpm_restart_io import read_restart
from letkf import letkf_analysis, state_coords

ENKF = os.environ.get("DGCPM_ENKF_ROOT",
                      os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ENKF, "run_control/restartOUT_dgcpm_restart.dat")
NATURE = os.path.join(ENKF, "nature_run_fortran.npz")
# deliberately not settable from the environment: anything that changes the
# trajectory travels through a flag, so it lands in --tag and in the saved npz
WINDOW_S = 3600


def getarg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    dry = "--dry" in sys.argv
    w0 = getarg("--w0", 128)              # storm-centred, as in the OSSE work
    nwin = getarg("--nwin", 28)
    k = getarg("--k", 16)
    qc = getarg("--qc", "standard", str)
    split = getarg("--split", "random80", str)
    frac = getarg("--frac", 0.8, float)
    # --seed ONLY controls the CV split (realobs.split_obs); the ensemble is drawn
    # from the fixed default_rng(0) below, so two runs differing only in --seed get
    # the SAME ensemble realization -- documented here, deliberately not fixed
    seed = getarg("--seed", 20240502)
    assim_inst = tuple(getarg("--assim", "EMMA", str).split(","))
    debias = "--debias" in sys.argv
    trainw = getarg("--trainwin", 120)    # bias training = windows 0..trainw
    # inherited constraints; do not re-tune
    kppert = getarg("--kppert", 0.15, float)
    sp_noise = getarg("--spread0", 0.12, float)   # small-scale IC perturbation
    sp_sine = getarg("--spreadL", 0.25, float)    # broad-scale (L-dependent)
    ceil_dex = getarg("--ceil", 1.0, float)
    bc_rings = getarg("--bcrings", 2)
    infl = getarg("--inflation", 1.08, float)
    rtps = getarg("--rtps", 0.0, float)
    repr_dex = getarg("--repr", 1.0, float)   # == the calibrated R = 1.0 dex
    rinfl = getarg("--rinfl", 1.0, float)     # extra sensitivity multiplier
    loc_L = getarg("--locL", 0.8, float)
    loc_MLT = getarg("--locMLT", 3.0, float)
    freerun = "--freerun" in sys.argv
    tag = getarg("--tag", "", str)
    # fixed seed 0, NOT --seed: spread0, spreadL and kppert below are identical in
    # every run whatever --seed is set to; see the note at --seed above
    rng = np.random.default_rng(0)

    if w0 < 1:
        raise SystemExit("--w0 must be >= 1 (the IC is the nature state at w0-1)")

    # grid, initial condition, Kp
    d = np.load(NATURE, allow_pickle=True)
    S, L_cell, MLT_cell = d["states"], d["L"], d["MLT"]
    nT, nP = int(d["nTheta"]), int(d["nPhi"])
    nwin = min(nwin, len(S) - w0)
    w_abs = np.arange(w0, w0 + nwin)

    kt, kv = e.load_kp_series()
    kp = np.array([e.kp_at(kt, kv, e.NATURE_START + timedelta(hours=int(a)))
                   for a in w_abs])

    # observations
    obs = ro.load_obs(qc=qc)
    bias = None
    if debias:
        # bias must be trained on windows disjoint from the evaluation range
        if trainw > w0:
            raise SystemExit("--trainwin must be <= --w0 (disjoint training)")
        bias = ro.estimate_obs_bias(obs, w_train=(0, trainw))

    sL, sMLT = state_coords(L_cell, MLT_cell)

    def get_window(i):
        o = ro.attach_H(ro.window_obs(obs, int(w_abs[i]), bias), L_cell, MLT_cell)
        return ro.split_obs(o, int(w_abs[i]), mode=split, frac=frac,
                            seed=seed, assim_inst=assim_inst)

    # initial ensemble: nature state at the start of w0 + perturbations
    # (S[w0-1] is a spun-up DGCPM state with the right Kp history, not "truth")
    bg = osse.to_log(S[w0 - 1])
    Lg = np.repeat(L_cell[:, None], nP, axis=1).flatten(order="F")
    ens0 = np.empty((nT * nP, k))
    for m in range(k):
        ens0[:, m] = bg + sp_noise * rng.standard_normal(nT * nP) \
            + sp_sine * np.sin(Lg + rng.uniform(0, 6))
    kp_pert = rng.normal(0, kppert, k)

    bounds = osse.state_bounds(L_cell, nP, ceil_margin_dex=ceil_dex)
    incr_cap = osse.increment_cap(L_cell, nP)
    bmask = osse.boundary_mask(nT, nP, bc_rings) if bc_rings else None
    lo, hi = (np.asarray(b, float)[:, None] for b in bounds)
    cap = np.asarray(incr_cap, float)[:, None]

    def constrain(X):
        Xn = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(Xn, lo, hi), int((Xn < lo).sum()), int((Xn > hi).sum())

    # banner
    print("=== REAL-OBSERVATION LETKF ===")
    print("windows %d..%d (%d h from %s), k=%d, grid %dx%d"
          % (w0, w0 + nwin, nwin,
             e.NATURE_START + timedelta(hours=int(w0)), k, nT, nP))
    print("IC: nature state at w%d (start of w%d) + perturbations" % (w0 - 1, w0))
    print("Kp: REAL series, %.1f..%.1f (per-member sd %.2f)"
          % (kp.min(), kp.max(), kppert))
    print("obs: qc=%s  unit bridge=+%.1f dex  bias correction=%s"
          % (qc, ro.OBS_TO_MODEL_LOG_OFFSET,
             ("%s (trained on w0..%d)" % (
                 {kk: round(vv, 3) for kk, vv in bias.items()}, trainw))
             if bias else "NONE (primary configuration)"))
    print("split: %s%s   R = rinfl^2*(sig_inst^2 + %.2f^2) dex^2, rinfl=%.1f"
          % (split, ("  assimilate=%s" % ",".join(assim_inst))
             if split == "instrument" else "  frac=%.2f" % frac,
             repr_dex, rinfl))
    print("constraints: ceil=sat(L)+%.1f dex, incr_cap=8->5 dex, bc_rings=%d, "
          "inflation=%.2f, rtps=%.2f%s"
          % (ceil_dex, bc_rings, infl, rtps,
             "   [FREE RUN -- no assimilation]" if freerun else ""))
    na = [len(get_window(i)[0]) for i in range(min(5, nwin))]
    nv = [len(get_window(i)[1]) for i in range(min(5, nwin))]
    print("obs/window assim%s withheld%s ..." % (na, nv))

    # propagator
    if dry:
        print("\n[DRY] surrogate propagator: relax toward the nature run with a\n"
              "      PERSISTENT per-member density offset, so the ensemble keeps\n"
              "      a spread and the LETKF is actually exercised. Checks\n"
              "      assembly, splits, R, constraints and diagnostics WITHOUT\n"
              "      docker. The numbers it produces are NOT a result.")
        dry_off = rng.normal(0, 0.45, k)          # member-persistent bias [dex]
        dry_rate = 0.35 * (1.0 + 0.5 * kp_pert / max(kppert, 1e-9))

        def propagate(ens_log, kpv, w):
            tgt = osse.to_log(S[w_abs[w]])[:, None] + dry_off[None, :]
            return ens_log + np.clip(dry_rate, 0.05, 0.9)[None, :] * (tgt - ens_log)
    else:
        template = read_restart(TEMPLATE)
        propagate = osse.make_fortran_propagator(
            template, WINDOW_S, e.NATURE_START + timedelta(hours=int(w0)),
            kp_pert=kp_pert, bounds=bounds)

    # the cycle
    D = dict(cv_rmse_f=[], cv_rmse_a=[], cv_bias_f=[], cv_bias_a=[],
             omb_rmse=[], oma_rmse=[], omb_bias=[], oma_bias=[],
             n_assim=[], n_with=[], spread=[], spread_f=[], n_ceil=[],
             n_floor=[], max_state=[], pp_f=[], pp_a=[])
    res_omb, res_oma, res_cvf, res_cva = [], [], [], []
    res_w, cv_inst, cv_L, cv_MLT, cv_y = [], [], [], [], []
    fields_a = np.empty((nwin, nT * nP), np.float32)
    fields_f = np.empty((nwin, nT * nP), np.float32)

    ens = ens0.copy()
    for w in range(nwin):
        Xf = propagate(ens, kp[w], w)
        Xf, nfl, nce = constrain(Xf)
        xf = Xf.mean(1)
        oa, ov = get_window(w)

        omb = ro.residuals(xf, oa)
        cvf = ro.residuals(xf, ov)

        if freerun or len(oa) == 0:
            Xa = Xf
            nfl2 = nce2 = 0
        else:
            R = ro.obs_error_var(oa, repr_dex=repr_dex, rinfl=rinfl)
            Xa = letkf_analysis(Xf, oa.y, R, Xf[oa.idx, :], oa.L, oa.MLT,
                                sL, sMLT, loc_L=loc_L, loc_MLT=loc_MLT,
                                inflation=infl, floor=-np.inf)
            if rtps > 0.0:
                ma = Xa.mean(1, keepdims=True)
                sf, sa = (Xf.std(1, ddof=1, keepdims=True),
                          Xa.std(1, ddof=1, keepdims=True))
                Xa = ma + (Xa - ma) * np.where(
                    sa > 1e-12, (rtps * sf + (1 - rtps) * sa) / np.maximum(sa, 1e-12), 1.0)
            Xa = Xf + np.clip(Xa - Xf, -cap, cap)      # increment cap
            if bmask is not None:
                Xa[bmask, :] = Xf[bmask, :]            # model owns the outer rings
            Xa, nfl2, nce2 = constrain(Xa)
        xa = Xa.mean(1)
        ens = Xa

        oma = ro.residuals(xa, oa)
        cva = ro.residuals(xa, ov)

        rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else np.nan
        mn = lambda v: float(np.mean(v)) if len(v) else np.nan
        D["omb_rmse"].append(rms(omb));  D["oma_rmse"].append(rms(oma))
        D["omb_bias"].append(mn(omb));   D["oma_bias"].append(mn(oma))
        D["cv_rmse_f"].append(rms(cvf)); D["cv_rmse_a"].append(rms(cva))
        D["cv_bias_f"].append(mn(cvf));  D["cv_bias_a"].append(mn(cva))
        D["n_assim"].append(len(oa));    D["n_with"].append(len(ov))
        # both spreads: spread_f << prev spread -> the model collapses it;
        # spread << spread_f -> the analysis collapses it (opposite remedies)
        D["spread_f"].append(float(Xf.std(1).mean()))
        D["spread"].append(float(Xa.std(1).mean()))
        D["n_floor"].append(nfl + nfl2); D["n_ceil"].append(nce + nce2)
        D["max_state"].append(float(Xa.max()))
        D["pp_f"].append(osse.plasmapause_L(xf, L_cell, nP))
        D["pp_a"].append(osse.plasmapause_L(xa, L_cell, nP))
        res_omb.append(omb); res_oma.append(oma)
        res_cvf.append(cvf); res_cva.append(cva)
        res_w.append(np.full(len(ov), w_abs[w])); cv_inst.append(ov.inst)
        cv_L.append(ov.L); cv_MLT.append(ov.MLT); cv_y.append(ov.y)
        fields_a[w] = xa.astype(np.float32); fields_f[w] = xf.astype(np.float32)

        # spread/OmB near 1 = healthy; << 1 = filter blind to its own error
        sr = D["spread"][-1] / D["omb_rmse"][-1] if D["omb_rmse"][-1] > 0 else np.nan
        print("w%3d Kp=%.1f  n=%3d/%3d  OmB=%.3f OmA=%.3f | CV f=%.3f a=%.3f  "
              "Lpp_a=%.2f  ceil=%d sprd %.3f->%.3f s/e=%.2f"
              % (w_abs[w], kp[w], D["n_assim"][-1], D["n_with"][-1],
                 D["omb_rmse"][-1], D["oma_rmse"][-1], D["cv_rmse_f"][-1],
                 D["cv_rmse_a"][-1], D["pp_a"][-1], D["n_ceil"][-1],
                 D["spread_f"][-1], D["spread"][-1], sr))

    out = {kk: np.array(vv) for kk, vv in D.items()}
    out.update(w_abs=w_abs, kp=kp, k=k, w0=w0, split=split, qc=qc,
               debias=debias, repr_dex=repr_dex, rinfl=rinfl, kppert=kppert,
               inflation=infl, rtps=rtps, freerun=freerun,
               assim_inst=",".join(assim_inst), frac=frac, seed=seed,
               unit_offset=ro.OBS_TO_MODEL_LOG_OFFSET,
               bias=np.array([0.0 if not bias else bias[n]
                              for n in ro.INSTRUMENTS]),
               L=L_cell, MLT=MLT_cell, nTheta=nT, nPhi=nP,
               field_a=fields_a, field_f=fields_f,
               res_omb=np.concatenate(res_omb) if res_omb else np.empty(0),
               res_oma=np.concatenate(res_oma) if res_oma else np.empty(0),
               res_cv_f=np.concatenate(res_cvf), res_cv_a=np.concatenate(res_cva),
               res_cv_w=np.concatenate(res_w), res_cv_inst=np.concatenate(cv_inst),
               res_cv_L=np.concatenate(cv_L), res_cv_MLT=np.concatenate(cv_MLT),
               res_cv_y=np.concatenate(cv_y))

    name = "da_realobs%s%s.npz" % ("_dry" if dry else "", ("_" + tag) if tag else "")
    path = os.path.join(ENKF, name)
    np.savez_compressed(path, **out)

    def nrms(v):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        return float(np.sqrt(np.mean(v ** 2))) if len(v) else np.nan

    print("\nsaved", path)
    print("--- pooled over all windows (log10 dex) ---")
    print("assimilated obs : OmB=%.3f  OmA=%.3f   (%.1f%% reduction; must be >0)"
          % (nrms(out["res_omb"]), nrms(out["res_oma"]),
             100 * (1 - nrms(out["res_oma"]) / nrms(out["res_omb"]))))
    print("WITHHELD obs    : forecast=%.3f  analysis=%.3f"
          % (nrms(out["res_cv_f"]), nrms(out["res_cv_a"])))
    print("                  mean bias  fcst=%+.3f  anal=%+.3f"
          % (out["res_cv_f"].mean(), out["res_cv_a"].mean()))
    print("ceiling hits %d in %d/%d windows | max state %.2f | mean spread %.3f"
          % (out["n_ceil"].sum(), int((out["n_ceil"] > 0).sum()), nwin,
             out["max_state"].max(), out["spread"].mean()))
    sr = out["spread"] / np.where(out["omb_rmse"] > 0, out["omb_rmse"], np.nan)
    print("spread/OmB = %.2f (min %.2f).  <0.2 means the ensemble does NOT span "
          "its own error\n              and the LETKF cannot correct it -- see "
          "OSSE_kp_ignorance_results.md sec 3.1;\n              the remedy there "
          "was calibrating the perturbation, here --rtps is the untested one."
          % (np.nanmean(sr), np.nanmin(sr)))
    print("\nNow score the FREE RUN on the identical withheld sets:")
    print("  python freerun_cv.py --w0 %d --nwin %d --split %s%s%s"
          % (w0, nwin, split, " --debias" if debias else "",
             (" --assim " + ",".join(assim_inst)) if split == "instrument" else ""))


if __name__ == "__main__":
    main()
