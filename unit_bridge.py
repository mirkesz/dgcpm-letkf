#!/usr/bin/env python3
"""Measure the obs->model unit offset by comparing the nature run to the real obs.

In: the nature-run npz and the observation directory.  Out: per-instrument arrays
of d = log10(model) - log10(obs), printed overall, per instrument and per L-bin.
"""
# each measurement is mapped onto the nearest model cell with the same H the LETKF
# uses, so the offset is measured through the operator that will consume it
# it is also broken down per L-bin, because an offset that varies with L would
# mean the gap is model bias rather than a unit error
import os
import sys
from datetime import timedelta

import numpy as np

import dgcpm_engine as e
import obs_loader as ol
from letkf import build_H_index

ENKF = os.environ.get("DGCPM_ENKF_ROOT",
                      os.path.dirname(os.path.abspath(__file__)))
NATURE = os.path.join(ENKF, "nature_run_fortran.npz")
INSTRUMENTS = ("Arase", "EMMA", "AWDANet")

# predicted BEFORE measuring so the measurement can refute it: the cm^-3 -> m^-3
# factor, see src/ModFunctionsDGCPM.f90:28
# measured: combined filled-cell median d = 5.432 over 216 windows, a -0.568 dex
# residual against +6.0 (python unit_bridge.py)
EXPECTED_OFFSET_DEX = 6.0


def measure(w0=0, nwin=216, filled_threshold=1.0e6, qc="none", verbose=True):
    """qc='none' by default: the unit bridge is measured on the unfiltered data."""
    # a quality cut that correlates with density would move the measured offset, so
    # the bridge is deliberately read off the raw records
    d = np.load(NATURE, allow_pickle=True)
    S = d["states"]
    L_cell, MLT_cell = d["L"], d["MLT"]
    nwin = min(nwin, len(S) - w0)

    obs = ol.load_all(os.path.join(ENKF, "DataAssimilation"), qc=qc)

    rec = {name: dict(d=[], L=[], mod=[], ne=[]) for name in INSTRUMENTS}
    for i in range(nwin):
        t0 = e.NATURE_START + timedelta(hours=w0 + i)
        state = S[w0 + i]
        for name in INSTRUMENTS:
            oL, oM, one = ol.obs_in_window(obs[name], t0, t0 + timedelta(hours=1))
            if len(oL) == 0:
                continue
            idx = build_H_index(L_cell, MLT_cell, oL, oM)
            mod = state[idx]
            ok = np.isfinite(mod) & (mod > 0) & np.isfinite(one) & (one > 0)
            rec[name]["d"].append(np.log10(mod[ok]) - np.log10(one[ok]))
            rec[name]["L"].append(oL[ok])
            rec[name]["mod"].append(mod[ok])
            rec[name]["ne"].append(one[ok])

    out = {}
    for name in INSTRUMENTS:
        if not rec[name]["d"]:
            out[name] = None
            continue
        out[name] = {kk: np.concatenate(vv) for kk, vv in rec[name].items()}

    if verbose:
        print("=== STEP 0: unit bridge, windows %d..%d, qc=%s ==="
              % (w0, w0 + nwin, qc))
        print("d = log10(model den at the obs cell) - log10(observed ne)\n")
        print("%-8s %8s %8s %8s %8s | %8s %8s %8s"
              % ("inst", "n", "median", "p25", "p75", "n_fill", "med_fill", "iqr_f"))
        allf = []
        for name in INSTRUMENTS:
            r = out[name]
            if r is None:
                print("%-8s      0" % name); continue
            f = r["mod"] > filled_threshold
            allf.append(r["d"][f])
            print("%-8s %8d %8.2f %8.2f %8.2f | %8d %8.3f %8.3f"
                  % (name, len(r["d"]), np.median(r["d"]),
                     np.percentile(r["d"], 25), np.percentile(r["d"], 75),
                     int(f.sum()),
                     np.median(r["d"][f]) if f.any() else np.nan,
                     (np.percentile(r["d"][f], 75) - np.percentile(r["d"][f], 25))
                     if f.any() else np.nan))
        allf = np.concatenate([a for a in allf if len(a)])
        print("\ncombined, model-filled cells only (den > %.0e m^-3): "
              "n=%d  median=%.3f  mean=%.3f  sd=%.3f"
              % (filled_threshold, len(allf), np.median(allf),
                 allf.mean(), allf.std()))
        print("expected from the unit analysis: %.1f dex   -> residual %+0.3f dex"
              % (EXPECTED_OFFSET_DEX, np.median(allf) - EXPECTED_OFFSET_DEX))

        print("\n--- is the offset constant in L? (filled cells only) ---")
        print("%-8s %-6s %8s %8s %8s" % ("inst", "L-bin", "n", "median", "iqr"))
        edges = np.array([1.3, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 10.0])
        for name in INSTRUMENTS:
            r = out[name]
            if r is None:
                continue
            f = r["mod"] > filled_threshold
            for a, b in zip(edges[:-1], edges[1:]):
                m = f & (r["L"] >= a) & (r["L"] < b)
                if m.sum() < 30:
                    continue
                print("%-8s %-6s %8d %8.3f %8.3f"
                      % (name, "%.1f-%.1f" % (a, b), int(m.sum()),
                         np.median(r["d"][m]),
                         np.percentile(r["d"][m], 75) - np.percentile(r["d"][m], 25)))
    return out


def _getarg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


if __name__ == "__main__":
    measure(w0=_getarg("--w0", 0), nwin=_getarg("--nwin", 216),
            qc=_getarg("--qc", "none", str))
