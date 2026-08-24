#!/usr/bin/env python3
"""Shared machinery for real obs: unit bridge, windowing, CV split, scoring.

In: the loaded observation dicts, a grid and an absolute window index.  Out: ObsSet
objects in MODEL units, their assimilated/withheld split, and residuals in dex.
"""
# imported by BOTH run_da_realobs.py and freerun_cv.py, which is the only reason
# the DA run and its no-DA baseline can be scored on byte-identical withheld sets
import os
from datetime import timedelta

import numpy as np

import dgcpm_engine as e
import obs_loader as ol
from letkf import build_H_index

ENKF = os.environ.get("DGCPM_ENKF_ROOT",
                      os.path.dirname(os.path.abspath(__file__)))
INSTRUMENTS = ("Arase", "EMMA", "AWDANet")
INST_ID = {n: i for i, n in enumerate(INSTRUMENTS)}

# unit bridge, DERIVED not fitted: instruments publish cm^-3, DGCPM works in m^-3
# and the 1e6 in saturation() IS that conversion, see src/ModFunctionsDGCPM.f90:28
OBS_TO_MODEL_LOG_OFFSET = 6.0
# measured: median log10(model)-log10(obs) = 5.43 over 216 windows on model-filled
# cells (python unit_bridge.py), i.e. 0.57 dex short of +6
# all three independent instruments sit the same amount above DGCPM's saturation(L)
# and with the same L-trend, so that residual is MODEL BIAS (C&A saturation too
# low), not a unit error; estimate_obs_bias() quantifies it on a disjoint period


class ObsSet:
    """All real observations in one assimilation window, in MODEL units."""

    # y is already log10(ne) + the unit offset, never a linear density

    __slots__ = ("L", "MLT", "y", "sig", "inst", "idx")

    def __init__(self, L, MLT, y, sig, inst, idx=None):
        self.L, self.MLT, self.y, self.sig = L, MLT, y, sig
        self.inst, self.idx = inst, idx

    def __len__(self):
        return len(self.y)

    def take(self, m):
        return ObsSet(self.L[m], self.MLT[m], self.y[m], self.sig[m],
                      self.inst[m], None if self.idx is None else self.idx[m])


def load_obs(data_dir=None, qc="standard"):
    return ol.load_all(data_dir or os.path.join(ENKF, "DataAssimilation"), qc=qc)


def window_obs(obs, w_abs, bias=None, window_h=1):
    """ObsSet for absolute window w_abs; y = log10(ne)+offset[-bias] in model units."""
    t0 = e.NATURE_START + timedelta(hours=w_abs)
    t1 = t0 + timedelta(hours=window_h)
    L, M, Y, S, I = [], [], [], [], []
    for name in INSTRUMENTS:
        oL, oM, one, osig = ol.obs_in_window(obs[name], t0, t1, with_sigma=True)
        if len(oL) == 0:
            continue
        y = np.log10(one) + OBS_TO_MODEL_LOG_OFFSET
        if bias:
            y = y - float(bias.get(name, 0.0))
        L.append(oL); M.append(oM); Y.append(y); S.append(osig)
        I.append(np.full(len(oL), INST_ID[name], np.int8))
    if not L:
        z = np.empty(0)
        return ObsSet(z, z, z, z, np.empty(0, np.int8))
    return ObsSet(np.concatenate(L), np.concatenate(M), np.concatenate(Y),
                  np.concatenate(S), np.concatenate(I))


def attach_H(o, L_cell, MLT_cell):
    """Set the nearest-neighbour state index (the LETKF's H) on `o`; returns `o`."""
    o.idx = (build_H_index(L_cell, MLT_cell, o.L, o.MLT)
             if len(o) else np.empty(0, int))
    return o


def split_obs(o, w_abs, mode="random80", frac=0.8, seed=20240502,
              assim_inst=("EMMA",)):
    """(assimilated, withheld) split, seeded on (seed, w_abs) and nothing else."""
    # seeding on (seed, w_abs) alone, with no shared state, is what lets the DA run
    # and the free-run baseline reproduce identical withheld sets in separate runs
    # mode='instrument' assimilates assim_inst and withholds the rest, the stronger
    # independence test (different technique, different region)
    # measured over w128-155: instrument gives same_cell 0.0000 and median
    # localization weight 0.022; random80 gives 81.3% and 0.999, i.e. it scores fit
    # measured: "the rest" is 1656 Arase + 5 AWDANet, so the withheld set of the
    # instrument split is not purely Arase
    n = len(o)
    if n == 0:
        return o, o
    if mode == "instrument":
        keep = np.isin(o.inst, [INST_ID[i] for i in assim_inst])
        return o.take(keep), o.take(~keep)
    if mode != "random80":
        raise ValueError("mode must be random80|instrument")
    rng = np.random.default_rng(int(seed) * 100003 + int(w_abs))
    keep = rng.random(n) < frac
    if keep.all():                      # guarantee a non-empty withheld set
        keep[rng.integers(n)] = False
    return o.take(keep), o.take(~keep)


def split_independence(obs, w_range, L_cell, MLT_cell, mode="random80",
                       frac=0.8, seed=20240502, assim_inst=("EMMA",),
                       loc_L=0.8, loc_MLT=3.0, cutoff=3.5, verbose=True):
    """How independent is the withheld set: same-cell fraction + localization weight."""
    # same_cell ~1 means the score is read at the very state elements the analysis
    # was fitted at; weight ~1 means the update at the validation point is fully
    # constrained by assimilated data -- either one voids a CV claim
    from letkf import _mlt_dist
    same, tot, wts = 0, 0, []
    for w in w_range:
        o = attach_H(window_obs(obs, int(w)), L_cell, MLT_cell)
        a, v = split_obs(o, int(w), mode=mode, frac=frac, seed=seed,
                         assim_inst=assim_inst)
        if len(v) == 0:
            continue
        tot += len(v)
        if len(a) == 0:
            wts.append(np.zeros(len(v)))
            continue
        same += int(np.isin(v.idx, a.idx).sum())
        dL = (a.L[None, :] - v.L[:, None]) / loc_L
        dM = _mlt_dist(a.MLT[None, :], v.MLT[:, None]) / loc_MLT
        wts.append(np.exp(-0.5 * np.sqrt(dL ** 2 + dM ** 2).min(1) ** 2))
    wts = np.concatenate(wts) if wts else np.zeros(0)
    r = dict(n=tot, same_cell=same / max(tot, 1),
             weight_median=float(np.median(wts)) if len(wts) else np.nan)
    if verbose:
        print("split independence (%s): %d withheld obs; %.0f%% share a grid "
              "cell with an assimilated obs; median localization weight to the "
              "nearest assimilated obs = %.3f"
              % (mode, r["n"], 100 * r["same_cell"], r["weight_median"]))
        if r["same_cell"] > 0.5:
            print("  !! WARNING: with nearest-neighbour H this split largely "
                  "measures FIT, not independent skill.\n"
                  "     The analysis was updated at the very state elements the "
                  "score is read from.\n"
                  "     Treat --split instrument as the real independence test.")
    return r


def estimate_obs_bias(obs, w_train=(0, 120), L_max=2.8, filled=1.0e6,
                      nature=None, verbose=True):
    """Per-instrument constant bias for --debias, trained on disjoint windows."""
    # restricted to L<=L_max (inside the plasmapause even at storm peak) and
    # model-filled cells, so plasmapause-position error cannot contaminate it
    d = np.load(nature or os.path.join(ENKF, "nature_run_fortran.npz"),
                allow_pickle=True)
    S, L_cell, MLT_cell = d["states"], d["L"], d["MLT"]
    acc = {n: [] for n in INSTRUMENTS}
    for w in range(w_train[0], min(w_train[1], len(S))):
        o = attach_H(window_obs(obs, w), L_cell, MLT_cell)
        if len(o) == 0:
            continue
        mod = S[w][o.idx]
        ok = (mod > filled) & (o.L <= L_max) & np.isfinite(mod)
        for name, i in INST_ID.items():
            m = ok & (o.inst == i)
            if m.any():
                acc[name].append(o.y[m] - np.log10(mod[m]))
    bias, nrec = {}, {}
    for name in INSTRUMENTS:
        v = np.concatenate(acc[name]) if acc[name] else np.empty(0)
        nrec[name] = len(v)
        bias[name] = float(np.median(v)) if len(v) >= 30 else 0.0
    if verbose:
        print("observation bias, trained on windows %d..%d (L<=%.1f, model>%.0e):"
              % (w_train[0], w_train[1], L_max, filled))
        for name in INSTRUMENTS:
            print("   %-8s n=%6d  b=%+.3f dex%s"
                  % (name, nrec[name], bias[name],
                     "   (n<30 -> not applied)" if nrec[name] < 30 else ""))
    return bias


def score(field_log, o):
    """(rmse, bias, n) of (H field - y) in dex; `o.idx` must be set."""
    if len(o) == 0:
        return np.nan, np.nan, 0
    r = field_log[o.idx] - o.y
    return float(np.sqrt(np.mean(r ** 2))), float(np.mean(r)), len(r)


def residuals(field_log, o):
    return (field_log[o.idx] - o.y) if len(o) else np.empty(0)


def obs_error_var(o, repr_dex=1.0, rinfl=1.0):
    """Diagonal R in dex^2: rinfl^2 * (sigma_instrument^2 + repr_dex^2)."""
    # repr_dex=1.0 is the representativeness error calibrated in the OSSE work
    # (nearest-neighbour H onto a 0.14 Re x 0.2 h cell); inherited, not re-tuned
    return (float(rinfl) ** 2) * (np.asarray(o.sig, float) ** 2
                                  + float(repr_dex) ** 2)


if __name__ == "__main__":
    import sys
    qc = sys.argv[sys.argv.index("--qc") + 1] if "--qc" in sys.argv else "standard"
    obs = load_obs(qc=qc)
    print("OBS_TO_MODEL_LOG_OFFSET = %.1f dex (derived; see module docstring)\n"
          % OBS_TO_MODEL_LOG_OFFSET)
    b = estimate_obs_bias(obs)
    print()
    d = np.load(os.path.join(ENKF, "nature_run_fortran.npz"), allow_pickle=True)
    for w in (128, 138, 141, 150):
        o = attach_H(window_obs(obs, w), d["L"], d["MLT"])
        a, v = split_obs(o, w, "random80")
        a2, v2 = split_obs(o, w, "instrument")
        print("w%3d n=%3d  random80 -> %3d/%3d   EMMA|Arase -> %3d/%3d   "
              "free-run RMSE on withheld=%.3f"
              % (w, len(o), len(a), len(v), len(a2), len(v2),
                 score(np.log10(np.maximum(d["states"][w], 1e2)), v)[0]))
