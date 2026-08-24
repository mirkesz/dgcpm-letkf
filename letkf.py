#!/usr/bin/env python3
"""Localized Ensemble Transform Kalman Filter (Hunt et al. 2007) for DGCPM.

In: forecast ensemble Xf (n,k) of log10 densities, obs y (p,), diagonal R (p,),
HXf (p,k), and the (L, MLT) of state and obs.  Out: the analysis Xa (n,k).
"""
import numpy as np


def _mlt_dist(a, b, period=24.0):
    # MLT wraps at 24 h, so 23:00 and 01:00 are 2 h apart, not 22
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def build_H_index(L_cell, MLT_cell, L_obs, MLT_obs):
    """Nearest-neighbour obs operator: F-order state index per observation."""
    # theta-fastest is not a free choice: it has to match the order the Fortran
    # restart file stores mgridden in, see src/ModIoDGCPM.f90:100
    nTheta = len(L_cell)
    L_obs = np.atleast_1d(L_obs); MLT_obs = np.atleast_1d(MLT_obs)
    iTheta = np.argmin(np.abs(L_cell[None, :] - L_obs[:, None]), axis=1)
    iPhi = np.argmin(_mlt_dist(MLT_cell[None, :], MLT_obs[:, None]), axis=1)
    return iPhi * nTheta + iTheta


def state_coords(L_cell, MLT_cell):
    """(L, MLT) of every state element, F-order (theta fastest)."""
    nTheta, nPhi = len(L_cell), len(MLT_cell)
    Lg = np.repeat(L_cell[:, None], nPhi, axis=1)
    Mg = np.repeat(MLT_cell[None, :], nTheta, axis=0)
    return Lg.flatten(order='F'), Mg.flatten(order='F')


def letkf_analysis(Xf, y, R_diag, HXf,
                   obs_L, obs_MLT, state_L, state_MLT,
                   loc_L=0.7, loc_MLT=2.0, inflation=1.0, cutoff=3.5,
                   floor=0.01):
    """LETKF analysis: Xf (n,k), y (p,), diagonal R (p,), HXf (p,k) -> Xa (n,k)."""
    Xf = np.asarray(Xf, float)
    n, k = Xf.shape
    xbar = Xf.mean(1)
    Xfp = Xf - xbar[:, None]
    ybar = HXf.mean(1)
    Yfp = HXf - ybar[:, None]
    dep = y - ybar

    Xa = np.empty_like(Xf)
    Ik = np.eye(k)

    # one independent k x k solve per grid point: that is what makes the filter
    # local, and why the cost scales with n rather than with p^3
    for g in range(n):
        dL = (obs_L - state_L[g]) / loc_L
        dM = _mlt_dist(obs_MLT, state_MLT[g]) / loc_MLT
        dist = np.sqrt(dL * dL + dM * dM)
        sel = dist < cutoff
        if not sel.any():
            Xa[g] = Xf[g]
            continue
        # localization enters as a weight on R^-1, never as a taper on Pf
        wloc = np.exp(-0.5 * dist[sel] ** 2)
        Yl = Yfp[sel]                           # (pl,k)
        Rinv = wloc / R_diag[sel]               # (pl,)
        C = Yl.T * Rinv                         # (k,pl)
        # Hunt et al. (2007) eq. 20-23; inflation enters as (k-1)/rho, i.e. it
        # widens the prior instead of rescaling the analysis afterwards
        A = (k - 1) * Ik / inflation + C @ Yl   # (k,k)
        evals, evecs = np.linalg.eigh(A)
        evals = np.maximum(evals, 1e-12)    # A is SPD; guard roundoff
        Pa = (evecs / evals) @ evecs.T                       # A^-1
        Wa = (evecs * np.sqrt((k - 1) / evals)) @ evecs.T    # sqrt((k-1)Pa)
        wbar = Pa @ (C @ dep[sel])              # mean update weights
        W = Wa + wbar[:, None]
        Xa[g] = xbar[g] + Xfp[g] @ W

    return np.maximum(Xa, floor)


# self-test: one synthetic identical-twin analysis step in log10 density
# measured: RMSE 0.2029 -> 0.0259 dex (-87.2%) and an exactly 0.00e+00 update in
# the 710 cells beyond the cutoff, numpy 2.5.1 (python letkf.py)
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    nTheta, nPhi, k = 62, 120, 30

    L_cell = np.linspace(1.30, 10.0, nTheta)
    MLT_cell = np.linspace(0, 24 * (1 - 1.0 / nPhi), nPhi)
    sL, sMLT = state_coords(L_cell, MLT_cell)
    n = nTheta * nPhi
    Lg = np.repeat(L_cell[:, None], nPhi, axis=1)
    Mg = np.repeat(MLT_cell[None, :], nTheta, axis=0)

    def logfield(Lpp, plume_amp):
        # radial falloff + plasmapause drop at Lpp + dusk plume
        base = -0.3 * Lg + 3.6
        pp = -1.7 / (1 + np.exp(-(Lg - Lpp) / 0.3))
        plume = plume_amp * np.exp(-((Lg - 5.0) ** 2) / 2.0
                                   - (_mlt_dist(Mg, 18.0) ** 2) / 8.0)
        return (base + pp + plume).flatten(order='F')

    xt = logfield(4.5, 0.6)                     # truth: plasmapause at L=4.5

    # forecast ensemble: biased plasmapause N(5.0, 0.5) + noise
    Xf = np.empty((n, k))
    for m in range(k):
        Xf[:, m] = logfield(rng.normal(5.0, 0.5), rng.normal(0.15, 0.3)) \
            + 0.05 * rng.normal(size=n)

    # observations: truth at random (L,MLT) + 0.08 dex noise
    p = 300
    obs_L = rng.uniform(2.5, 6.5, p)
    obs_MLT = rng.uniform(0, 24, p)
    idx = build_H_index(L_cell, MLT_cell, obs_L, obs_MLT)
    R_diag = np.full(p, 0.08 ** 2)
    y = xt[idx] + rng.normal(size=p) * 0.08

    HXf = Xf[idx, :]
    Xa = letkf_analysis(Xf, y, R_diag, HXf, obs_L, obs_MLT, sL, sMLT,
                        loc_L=0.8, loc_MLT=3.0, inflation=1.05,
                        cutoff=3.5, floor=-np.inf)

    def rmse(x): return np.sqrt(np.mean((x - xt) ** 2))
    xf_mean, xa_mean = Xf.mean(1), Xa.mean(1)

    # a non-zero update out here would mean the localization leaks, which would
    # invalidate every CV score read at withheld obs elsewhere on the grid
    dL = (obs_L[None, :] - sL[:, None]) / 0.8
    dM = _mlt_dist(obs_MLT[None, :], sMLT[:, None]) / 3.0
    mindist = np.sqrt(dL ** 2 + dM ** 2).min(axis=1)
    far = mindist > 3.5

    print('=== LETKF identical-twin analysis test (log10 density) ===')
    print('members k=%d, obs p=%d, far-from-obs cells=%d' % (k, p, far.sum()))
    print('RMSE forecast mean : %.4f dex' % rmse(xf_mean))
    print('RMSE analysis mean : %.4f dex' % rmse(xa_mean))
    print('RMSE reduction     : %.1f%%'
          % (100 * (1 - rmse(xa_mean) / rmse(xf_mean))))
    print('ensemble spread    : fcst=%.3f  anal=%.3f'
          % (Xf.std(1).mean(), Xa.std(1).mean()))
    if far.any():
        print('far-region |xa-xf| max = %.2e  (localization -> ~0)'
              % np.max(np.abs(xa_mean[far] - xf_mean[far])))
    assert rmse(xa_mean) < rmse(xf_mean), 'analysis did not improve RMSE!'
    if far.any():
        assert np.max(np.abs(xa_mean[far] - xf_mean[far])) < 1e-6, \
            'localization leak!'
    print('PASS: analysis reduces error and localization confines the update.')
