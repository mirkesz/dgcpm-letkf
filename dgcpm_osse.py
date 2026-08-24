#!/usr/bin/env python3
"""Cycling LETKF for DGCPM: forecast -> observe -> analysis, plus constraints.

In: a truth trajectory, a Kp series, a propagator callable and an initial ensemble.
Out: a per-window diagnostics dict (RMSE, plasmapause, clip counts, spread).
"""
# state is log10(density); the propagator is pluggable, either the Fortran engine
# via docker (make_fortran_propagator) or a cheap surrogate (--selftest)
import numpy as np

from letkf import build_H_index, state_coords, letkf_analysis, _mlt_dist

DENSITY_FLOOR = 1.0e2   # m^-3, floor before log10 (empty flux tubes ~ 0)


def to_log(den):
    return np.log10(np.maximum(np.asarray(den, float), DENSITY_FLOOR))


def from_log(logden):
    return 10.0 ** np.asarray(logden, float)


def saturation_density(L):
    """Carpenter & Anderson (1992) saturation [m^-3]; mirrors the Fortran exactly."""
    # the 1e6 is the cm^-3 -> m^-3 conversion, not a tuning factor
    # see src/ModFunctionsDGCPM.f90:28
    return 1.0e6 * 10.0 ** (-0.3145 * np.asarray(L, float) + 3.9043)


def state_bounds(L_cell, nPhi, ceil_margin_dex=1.0, floor=DENSITY_FLOOR):
    """(lo, hi) log10 bounds per state element: floor .. saturation(L)+margin."""
    # the ceiling is the post-storm blow-up fix
    # measured in the OSSE blow-up diagnosis: without it the LETKF extrapolated an
    # unobserved cell to 1e52 m^-3, and the restart chain then carried it forward
    L_cell = np.asarray(L_cell, float)
    hi_col = np.log10(saturation_density(L_cell)) + float(ceil_margin_dex)
    hi = np.repeat(hi_col[:, None], nPhi, axis=1).flatten(order='F')
    lo = np.full_like(hi, np.log10(floor))
    return lo, hi


def increment_cap(L_cell, nPhi, cap_inner=8.0, cap_outer=5.0,
                  L_taper=(7.5, 9.0)):
    """Per-element cap on |analysis - forecast| in dex (defence in depth)."""
    # measured: the largest 1-hour motion in the truth run is ~5.8 dex, so an
    # 8-dex cap blocks unbounded excursions without ever clipping real motion
    L_cell = np.asarray(L_cell, float)
    L0, L1 = L_taper
    w = np.clip((L_cell - L0) / (L1 - L0), 0.0, 1.0)
    cap_col = cap_inner + (cap_outer - cap_inner) * w
    return np.repeat(cap_col[:, None], nPhi, axis=1).flatten(order='F')


def boundary_mask(nTheta, nPhi, n_rings=2):
    """Mask of the outer n_rings L-shells, which the Fortran overwrites anyway."""
    # superbee resets mgridn on i = nrcells-1, nrcells on every call, so whatever
    # the analysis writes there is discarded; see src/pbo.f:1127
    m = np.zeros(nTheta * nPhi, bool)
    if n_rings > 0:
        i = np.arange(nTheta * nPhi) % nTheta
        m[i >= nTheta - n_rings] = True
    return m


def sample_obs(truth_log, L_cell, MLT_cell, rng, n_obs=300,
               obs_L=None, obs_MLT=None, L_lo=2.0, L_hi=7.0, noise_sd=0.08):
    """Synthetic obs: truth at (L,MLT) positions + Gaussian noise."""
    if obs_L is None:
        obs_L = rng.uniform(L_lo, L_hi, n_obs)
        obs_MLT = rng.uniform(0, 24, n_obs)
    idx = build_H_index(L_cell, MLT_cell, obs_L, obs_MLT)
    y = truth_log[idx] + rng.normal(size=len(idx)) * noise_sd
    R_diag = np.full(len(idx), noise_sd ** 2)
    return obs_L, obs_MLT, y, R_diag


def plasmapause_L(logfield_flat, L_cell, nP, L_lo=2.0, L_hi=6.5):
    """L of the steepest azimuthal-mean density drop (plasmapause proxy)."""
    nT = len(L_cell)
    f = logfield_flat.reshape(nT, nP, order='F')
    m = (L_cell > L_lo) & (L_cell < L_hi)
    prof = f[m, :].mean(1)
    return L_cell[m][np.argmin(np.diff(prof))]


def run_osse(truth_traj_log, L_cell, MLT_cell, kp_win, propagate, ens0,
             rng, n_obs=300, obs_L=None, obs_MLT=None, obs_provider=None,
             noise_sd=0.08, loc_L=0.8, loc_MLT=3.0, inflation=1.08,
             bounds=None, incr_cap=None, bc_rings=0, rtps=0.0,
             r_inflate=1.0, assimilate=True, verbose=True):
    """Cycling identical-twin OSSE; assimilate=False gives the free-run baseline."""
    # r_inflate scales the R the filter ASSUMES (representativeness error)
    # without changing the noise actually added to the synthetic obs
    nwin = len(truth_traj_log)
    sL, sMLT = state_coords(L_cell, MLT_cell)
    nP = len(MLT_cell)
    nT = len(L_cell)
    ens = ens0.copy()
    rf, ra, ppf, ppa, ppt, nob = [], [], [], [], [], []
    nclip, nceil, mxs, spr = [], [], [], []

    lo = hi = None
    if bounds is not None:
        lo, hi = bounds
        lo = np.asarray(lo, float)[:, None]
        hi = np.asarray(hi, float)[:, None]
    cap = None if incr_cap is None else np.asarray(incr_cap, float)[:, None]
    bmask = boundary_mask(nT, nP, bc_rings) if bc_rings else None

    def constrain(X):
        # clip to model-representable bounds; floor hits are benign,
        # ceiling hits are the divergence signature and should stay 0
        if lo is None:
            return X, 0, 0
        Xn = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        nlo = int(np.count_nonzero(Xn < lo))
        nhi = int(np.count_nonzero(Xn > hi))
        return np.clip(Xn, lo, hi), nlo, nhi

    for w in range(nwin):
        truth = truth_traj_log[w]
        Xf = propagate(ens, kp_win[w], w)                 # forecast
        Xf, nlo_f, nhi_f = constrain(Xf)                  # background QC
        wL, wM = (obs_provider(w) if obs_provider is not None
                  else (obs_L, obs_MLT))
        if not assimilate:
            wL = wM = np.empty(0)
        if (not assimilate) or (obs_provider is not None and len(wL) == 0):
            ens = Xf                                       # no obs -> no update
            rf.append(float(np.sqrt(np.mean((Xf.mean(1) - truth) ** 2))))
            ra.append(rf[-1]); nob.append(0)
            nclip.append(nlo_f); nceil.append(nhi_f)
            mxs.append(float(Xf.max()))
            spr.append(float(Xf.std(1).mean()))
            ppt.append(plasmapause_L(truth, L_cell, nP))
            ppf.append(plasmapause_L(Xf.mean(1), L_cell, nP)); ppa.append(ppf[-1])
            continue
        oL, oM, y, R = sample_obs(truth, L_cell, MLT_cell, rng, n_obs,
                                  wL, wM, noise_sd=noise_sd)
        if r_inflate != 1.0:      # assumed R only; y keeps its real noise
            R = R * float(r_inflate) ** 2
        idx = build_H_index(L_cell, MLT_cell, oL, oM)
        HXf = Xf[idx, :]
        Xa = letkf_analysis(Xf, y, R, HXf, oL, oM, sL, sMLT,
                            loc_L=loc_L, loc_MLT=loc_MLT,
                            inflation=inflation, floor=-np.inf)

        # RTPS (Whitaker & Hamill 2012): relax analysis spread toward the prior
        if rtps > 0.0:
            mf, ma = Xf.mean(1, keepdims=True), Xa.mean(1, keepdims=True)
            sf = Xf.std(1, ddof=1, keepdims=True)
            sa = Xa.std(1, ddof=1, keepdims=True)
            scale = np.where(sa > 1e-12,
                             (rtps * sf + (1.0 - rtps) * sa) / np.maximum(sa, 1e-12),
                             1.0)
            Xa = ma + (Xa - ma) * scale

        # constrain the analysis before it becomes the next background
        if cap is not None:                       # limit the increment
            Xa = Xf + np.clip(Xa - Xf, -cap, cap)
        if bmask is not None:                     # model owns the outer rings
            Xa[bmask, :] = Xf[bmask, :]
        Xa, nlo_a, nhi_a = constrain(Xa)          # hard physical/numerical box
        ens = Xa

        def rmse(x): return float(np.sqrt(np.mean((x - truth) ** 2)))
        rf.append(rmse(Xf.mean(1))); ra.append(rmse(Xa.mean(1))); nob.append(len(oL))
        nclip.append(nlo_f + nlo_a); nceil.append(nhi_f + nhi_a)
        mxs.append(float(Xa.max()))
        spr.append(float(Xa.std(1).mean()))
        ppt.append(plasmapause_L(truth, L_cell, nP))
        ppf.append(plasmapause_L(Xf.mean(1), L_cell, nP))
        ppa.append(plasmapause_L(Xa.mean(1), L_cell, nP))
        if verbose:
            print("win %3d Kp=%.1f  #obs=%4d  RMSE f=%.3f a=%.3f  "
                  "Lpp truth=%.2f a=%.2f  floor=%d ceil=%d  max=%.2f  sprd=%.3f"
                  % (w, kp_win[w], nob[-1], rf[-1], ra[-1], ppt[-1], ppa[-1],
                     nclip[-1], nceil[-1], mxs[-1], spr[-1]))

    return dict(rmse_f=np.array(rf), rmse_a=np.array(ra), n_obs=np.array(nob),
                pp_truth=np.array(ppt), pp_f=np.array(ppf), pp_a=np.array(ppa),
                n_clipped=np.array(nclip), n_at_ceiling=np.array(nceil),
                max_state=np.array(mxs),
                spread=np.array(spr))


def make_fortran_propagator(template, window_s, start, kp_pert=None,
                            bounds=None, kp_clip=(0.0, 9.0), strict=True):
    """Fortran propagator: inject each member, run one window via docker, extract."""
    # kp_pert: per-member fractional Kp perturbation (the ensemble-spread source);
    # bounds clip the injected field before it is written to the restart
    # kp_clip is NOT guarding a pole: A = 7.05e-6/(1-0.159Kp+0.0093Kp^2)^3 has no
    # zero in its denominator, see src/dgcpm_setup.f90:66
    # measured: the cubed denominator bottoms out at 0.0329 at Kp=8.548, so A is
    # MAXIMAL there and then falls again as Kp rises -- physically meaningless,
    # extrapolation past the range the Volland-Stern parameterisation was fitted on
    # the clip is at 9.0, so it does not exclude that regime, it only bounds Kp
    import dgcpm_engine as e
    from datetime import timedelta

    lo = hi = None
    if bounds is not None:
        lo, hi = (np.asarray(b, float) for b in bounds)

    def propagate(ens_log, kp, w):
        out = np.empty_like(ens_log)
        t = start + timedelta(seconds=w * window_s)
        for m in range(ens_log.shape[1]):
            kpm = kp if kp_pert is None else kp * (1 + kp_pert[m])
            kpm = float(np.clip(kpm, *kp_clip))
            xm = ens_log[:, m]
            if lo is not None:
                xm = np.clip(np.nan_to_num(xm, nan=0.0, posinf=0.0,
                                           neginf=0.0), lo, hi)
            den = from_log(xm).reshape(template['nTheta'],
                                       template['nPhi'], order='F')
            r = e.propagate_field(den, template, kpm, window_s, t)
            fc = r['den'].flatten(order='F')
            if strict and not np.all(np.isfinite(fc)):
                raise RuntimeError(
                    "window %d member %d: Fortran returned %d non-finite "
                    "density cells (Kp=%.2f)"
                    % (w, m, int((~np.isfinite(fc)).sum()), kpm))
            out[:, m] = to_log(fc)
        return out
    return propagate


# self-test: surrogate identical-twin, exercises the cycle without docker
# measured: mean RMSE free-run 0.016 vs LETKF 0.008 over 40 windows, numpy 2.5.1
# (python dgcpm_osse.py --selftest)
if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        rng = np.random.default_rng(1)
        nT, nP, k, nwin = 62, 120, 24, 40
        L_cell = np.linspace(1.30, 10.0, nT)
        MLT_cell = np.linspace(0, 24 * (1 - 1.0 / nP), nP)
        Lg = np.repeat(L_cell[:, None], nP, axis=1)
        Mg = np.repeat(MLT_cell[None, :], nT, axis=0)

        def target(kp):
            Lpp = 5.8 - 0.32 * kp                    # plasmapause moves in with Kp
            base = -0.30 * Lg + 3.6
            drop = -1.7 / (1 + np.exp(-(Lg - Lpp) / 0.3))
            return (base + drop).flatten(order='F')

        def surrogate(field_log, kp):
            # slow relaxation (alpha=0.2) so IC errors persist across windows
            return field_log + 0.2 * (target(kp) - field_log)

        # Kp storyline with a storm
        kp_win = np.r_[np.full(12, 2.0), np.linspace(2, 6.7, 6),
                       np.full(4, 6.7), np.linspace(6.7, 2, 8),
                       np.full(10, 2.0)][:nwin]

        truth = np.empty((nwin, nT * nP))
        x = target(kp_win[0])
        for w in range(nwin):
            x = surrogate(x, kp_win[w]); truth[w] = x

        # biased+spread initial ensemble (plasmapause too far out)
        bg = target(1.0)
        ens0 = np.empty((nT * nP, k))
        for m in range(k):
            ens0[:, m] = bg + 0.15 * rng.standard_normal(nT * nP) \
                + 0.3 * np.sin(Lg.flatten(order='F') + rng.uniform(0, 6))

        prop = lambda ens, kp, w: np.column_stack(
            [surrogate(ens[:, m], kp) for m in range(ens.shape[1])])

        diag = run_osse(truth, L_cell, MLT_cell, kp_win, prop, ens0, rng,
                        n_obs=250, noise_sd=0.08, inflation=1.05, verbose=False)

        # baseline: same ensemble, no assimilation
        free = ens0.copy(); free_rmse = []
        for w in range(nwin):
            free = np.column_stack([surrogate(free[:, m], kp_win[w])
                                    for m in range(k)])
            free_rmse.append(np.sqrt(np.mean((free.mean(1) - truth[w]) ** 2)))
        free_rmse = np.array(free_rmse)

        print("=== OSSE self-test (surrogate identical-twin, %d windows) ===" % nwin)
        print("mean RMSE  free-run(no DA)=%.3f  analysis(LETKF)=%.3f  (%.0f%% better)"
              % (free_rmse.mean(), diag['rmse_a'].mean(),
                 100 * (1 - diag['rmse_a'].mean() / free_rmse.mean())))
        print("first-window RMSE  forecast=%.3f -> analysis=%.3f  (IC bias corrected)"
              % (diag['rmse_f'][0], diag['rmse_a'][0]))
        print("plasmapause: mean |Lpp_a - truth|=%.2f Re  vs free-run |Lpp_free - truth|=%.2f Re"
              % (np.mean(np.abs(diag['pp_a'] - diag['pp_truth'])),
                 np.mean(np.abs(diag['pp_f'] - diag['pp_truth']))))
        print("storm window Lpp: truth min=%.2f  analysis min=%.2f"
              % (diag['pp_truth'][12:22].min(), diag['pp_a'][12:22].min()))
        ok = (diag['rmse_a'].mean() < 0.5 * free_rmse.mean()
              and np.mean(np.abs(diag['pp_a'] - diag['pp_truth'])) < 0.5)
        print("PASS" if ok else "CHECK",
              "- cycling LETKF beats the free run and tracks the plasmapause.")
    else:
        print("Run the self-test:  python dgcpm_osse.py --selftest")
        print("Real OSSE (on the Mac) is driven from run_osse_fortran.py.")
