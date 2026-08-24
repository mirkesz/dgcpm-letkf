#!/usr/bin/env python3
"""Reproduce the cross-validation summary numbers from the saved .npz files.

In: freerun_cv_*.npz and da_realobs_*.npz in the working directory.  Out: the skill
table, per-phase and per-L breakdowns and numerical-health lines on stdout.
"""
# skill is split into cycling (earlier analyses that survived propagation) and the
# current window's increment, because in the instrument split nearly all of it is
# the former and a single total would hide that
# CIs use a block bootstrap over whole windows: per-hour obs are correlated along
# an orbit, so resampling individual obs would understate the interval
# the ceiling diagnostic is reported on the ANALYSIS MEAN, not on the inflated
# cell x member x 2 instance count
# measured: stage1 +58.7%, stage2 +10.8% (CI +4.6..+20.3), stage3 +53.9%
import numpy as np

import dgcpm_osse as osse

STAGES = [("1  random 80/20    ", "freerun_cv_rnd.npz",   "da_realobs_rnd.npz"),
          ("2  EMMA -> Arase   ", "freerun_cv_inst.npz",  "da_realobs_inst.npz"),
          ("3  random, debiased", "freerun_cv_rnddb.npz", "da_realobs_rnddb.npz")]
PHASES = [("A pre-storm w128-131", 128, 132), ("B storm w132-146", 132, 147),
          ("C w147-149", 147, 150), ("D recovery w150-155", 150, 156)]
LBANDS = [(1.3, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 6.0), (6.0, 10.0)]


def rms(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else np.nan


def block_bootstrap(rf, ra, w, n=20000, seed=7):
    """95% CI on the skill %, resampling whole windows."""
    rng = np.random.default_rng(seed)
    uw = np.unique(w)
    idx = {u: np.where(w == u)[0] for u in uw}
    sk = np.empty(n)
    for i in range(n):
        j = np.concatenate([idx[p] for p in rng.choice(uw, len(uw), replace=True)])
        sk[i] = 100 * (1 - rms(ra[j]) / rms(rf[j]))
    return np.percentile(sk, [2.5, 97.5])


def main():
    print("=" * 78)
    print("REAL-OBSERVATION LETKF — cross-validation summary (log10 dex)")
    print("=" * 78)
    print("%-20s %7s %7s %7s | %8s %8s %8s   %s"
          % ("stage", "free", "DAfcst", "DAanal", "total", "cycling", "analysis",
             "95% CI"))
    keep = []
    for tag, fn, dn in STAGES:
        try:
            F, D = (np.load(fn, allow_pickle=True), np.load(dn, allow_pickle=True))
        except FileNotFoundError:
            print("%-20s  (not run)" % tag); continue
        keep.append((tag, F, D))
        rf, rff, ra, w = (F["res_cv_f"], D["res_cv_f"], D["res_cv_a"],
                          D["res_cv_w"])
        tot = 100 * (1 - rms(ra) / rms(rf))
        cyc = 100 * (1 - rms(rff) / rms(rf))
        lo, hi = block_bootstrap(rf, ra, w)
        print("%-20s %7.3f %7.3f %7.3f | %+7.1f%% %+7.1f%% %+7.1f%%   %+.1f..%+.1f"
              % (tag, rms(rf), rms(rff), rms(ra), tot, cyc, tot - cyc, lo, hi))
    print("\n  'cycling'  = free -> DA FORECAST: earlier analyses that survived "
          "propagation.\n  'analysis' = DA forecast -> DA ANALYSIS: this window's "
          "increment at the\n               validation points. In the instrument "
          "split this is ~0 — the\n               information arrives via the "
          "model, not via the analysis operator.")

    for tag, F, D in keep:
        rf, ra, w, Lo = (F["res_cv_f"], D["res_cv_a"], D["res_cv_w"],
                         D["res_cv_L"])
        print("\n" + "-" * 78)
        print("stage %s   split=%s  debias=%s   n_withheld=%d"
              % (tag.strip(), D["split"], bool(D["debias"]), len(ra)))
        print("  OmB=%.3f -> OmA=%.3f (%+.1f%%)   bias %+.3f -> %+.3f"
              % (rms(D["res_omb"]), rms(D["res_oma"]),
                 100 * (1 - rms(D["res_oma"]) / rms(D["res_omb"])),
                 rf.mean(), ra.mean()))
        print("  by phase:")
        for lab, a, b in PHASES:
            m = (w >= a) & (w < b)
            if m.sum():
                print("    %-22s n=%4d  %.3f -> %.3f  %+.1f%%"
                      % (lab, m.sum(), rms(rf[m]), rms(ra[m]),
                         100 * (1 - rms(ra[m]) / rms(rf[m]))))
        print("  by L:")
        for a, b in LBANDS:
            m = (Lo >= a) & (Lo < b)
            if m.sum() >= 20:
                print("    L %4.1f-%4.1f  n=%4d  %.3f -> %.3f  %+.1f%%   "
                      "bias %+.3f -> %+.3f"
                      % (a, b, m.sum(), rms(rf[m]), rms(ra[m]),
                         100 * (1 - rms(ra[m]) / rms(rf[m])),
                         rf[m].mean(), ra[m].mean()))
        # numerical health
        L, nP = D["L"], int(D["nPhi"])
        _, hi_b = osse.state_bounds(L, nP, ceil_margin_dex=1.0)
        atc = D["field_a"] >= (hi_b[None, :] - 1e-6)
        sr = D["spread"] / np.where(D["omb_rmse"] > 0, D["omb_rmse"], np.nan)
        print("  numerics: ceiling instances=%d (cell x member x 2) BUT analysis "
              "mean at ceiling %d/%d = %.2f%%;"
              % (D["n_ceil"].sum(), atc.sum(), atc.size, 100 * atc.mean()))
        print("            max state %.3f vs ceiling %.3f -> bounded.  "
              "spread/OmB mean=%.2f min=%.2f%s"
              % (D["field_a"].max(), hi_b.max(), np.nanmean(sr), np.nanmin(sr),
                 "  <- under-dispersed, results are LOWER BOUNDS"
                 if np.nanmean(sr) < 0.7 else ""))
        z = D["w_abs"][D["n_assim"] == 0]
        if len(z):
            print("            windows with ZERO assimilated obs: %s"
                  % list(map(int, z)))
        moved = int((np.abs(D["cv_rmse_f"] - D["cv_rmse_a"]) > 0.01).sum())
        print("            windows where the analysis moved the withheld score "
              ">0.01 dex: %d/%d" % (moved, len(D["w_abs"])))

    # structure vs constant offset
    d = {t.strip()[0]: (F, D) for t, F, D in keep}
    if "1" in d and "3" in d:
        g1 = rms(d["1"][0]["res_cv_f"]) - rms(d["1"][1]["res_cv_a"])
        g3 = rms(d["3"][0]["res_cv_f"]) - rms(d["3"][1]["res_cv_a"])
        print("\n" + "=" * 78)
        print("STRUCTURE vs CONSTANT OFFSET")
        print("  absolute error removed:  raw obs %.3f dex   debiased obs %.3f dex"
              % (g1, g3))
        print("  => %.0f%% of the gain is spatial/temporal STRUCTURE, "
              "%.0f%% is the constant\n     saturation-level offset of sec 1.2."
              % (100 * g3 / g1, 100 * (1 - g3 / g1)))


if __name__ == "__main__":
    main()
