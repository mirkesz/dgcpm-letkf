#!/usr/bin/env python3
"""Load real observations (Arase / EMMA / AWDANet) as (t, L, MLT, ne, sig).

In: a data directory and a qc level (none|standard|strict).  Out: one dict per
instrument with equal-length t, L, MLT, ne [cm^-3] and sig [dex] arrays.
"""
# every instrument publishes cm^-3; the cm^-3 -> m^-3 conversion is deliberately
# NOT applied here but in realobs.OBS_TO_MODEL_LOG_OFFSET, so that the unit
# bridge has exactly one place to audit
import glob
import os
from datetime import datetime

import numpy as np

L_MIN, L_MAX = 1.3, 10.0   # DGCPM grid range; measured L span 1.298 .. 10.000

# what the quality flags actually are: measured, not assumed (print with --notes)
QC_NOTES = """\
EMMA  `flag`: the readme calls it "source flag", and it is **1 in all 84 051
      records**. It therefore carries no quality information and CANNOT be used
      as an accept/reject rule (an `flag==0` rule would discard everything).
      The readme's own list of "information to be used for quality assessment"
      is used instead: `prom` (phase prominence, deg) and `terr` (timing error,
      s). standard: prom>=10 & |terr|<=1  -> keeps 81.5%.
      strict:   prom>=20 & |terr|<=1 & xph>=20 -> keeps 68.5% (57539/84051).
      OPEN QUESTION FOR THE SUPERVISOR: the prominence threshold is a
      distribution-anchored choice (prom p5=8.0, p25=22.4), not a documented
      one. If EMMA publishes a recommended acceptance rule, use it instead.

AWDANet `qf`: values 00..12, counts 73,49,51,43,46,13,12,8,11,5,3,1,1 (n=316).
      The trend decays clearly but it is NOT monotonic: the count rises again at
      qf=02 (49->51), qf=04 (43->46) and qf=08 (8->11). That is the profile of
      an event/trace INDEX, not of a quality RANK, and it is reproduced in the
      source filename (`..._qf_02.vr2.mat`). It is therefore NOT used as an
      accept/reject rule.
      The file's own per-measurement uncertainty columns are used instead:
      `dneq(%)` and `dL(%)`.  standard: dneq<=40 & dL<=40 -> keeps ~86%.
      strict: dneq<=30 -> keeps ~46% (only 316 records exist in total).
      `dneq(%)` is additionally propagated into the observation error.
      OPEN QUESTION FOR THE SUPERVISOR: confirm the meaning of `qf`.

Arase : no per-record quality column in the magcoord files. Only the basic
      physical screen (L in the model grid, ne>0, finite) is applied.
      strict additionally requires L<=9.0 (the outermost model L-shells are
      owned by the superbee boundary condition, see dgcpm_osse.boundary_mask).
"""

# per-instrument 1-sigma MEASUREMENT error in dex; representativeness error
# (~1 dex, nearest-neighbour H onto a 0.14 Re x 0.2 h cell) is added in realobs.py
SIGMA_DEX = {
    "Arase":   0.10,   # upper-hybrid derived n_e, ~20-25%
    # assumption: EMMA's amu/cm^3 equals cm^-3, i.e. a pure-hydrogen plasma; any
    # heavy-ion fraction biases ne high, check: EMMA readme lines 5-9
    "EMMA":    0.15,   # FLR-derived mass density
    "AWDANet": 0.12,   # overridden per-record by the reported dneq(%)
}


def _dex(rel_percent):
    """Relative error in % -> 1-sigma in dex, symmetrised."""
    # clipped at 95% because r -> 1 sends log10((1+r)/(1-r)) to infinity
    r = np.clip(np.asarray(rel_percent, float), 0.0, 95.0) / 100.0
    return 0.5 * np.log10((1.0 + r) / (1.0 - r))


def load_arase(base, qc="standard"):
    """magcoord_ARA_*: cols year month day H M S L MLT mlat ne [cm^-3]."""
    t, L, MLT, ne = [], [], [], []
    l_hi = 9.0 if qc == "strict" else L_MAX
    for fp in sorted(glob.glob(os.path.join(base, "magcoord_ARA_igrf_t89q_*_v2.txt"))):
        for line in open(fp).readlines()[2:]:
            p = line.split()
            if len(p) < 10:
                continue
            try:
                vals = [float(x) for x in p[:10]]
            except ValueError:
                continue
            y, mo, d, h, mi, s, Lv, mltv, _, nev = vals
            if not (L_MIN <= Lv <= l_hi and nev > 0 and np.isfinite(nev)):
                continue
            # measured: no Arase record actually reaches s=60, the clamp is
            # defensive and mirrors the EMMA loader
            t.append(datetime(int(y), int(mo), int(d), int(h), int(mi),
                              min(int(s), 59)))
            L.append(Lv); MLT.append(mltv % 24); ne.append(nev)
    return _pack("Arase", t, L, MLT, ne)


def load_emma(base, qc="standard"):
    """plasma_revised_flr_density_v3_*: cols y m d H M S L rho_0 MLT ... [cm^-3]."""
    t, L, MLT, ne = [], [], [], []
    for fp in sorted(glob.glob(os.path.join(base, "plasma_revised_flr_density_v3_*.txt"))):
        for line in open(fp).readlines()[1:]:
            p = line.split()
            if len(p) < 9:
                continue
            try:
                y, mo, d, h, mi, s = (int(float(x)) for x in p[:6])
                Lv, rho, mltv = float(p[6]), float(p[7]), float(p[8])
            except ValueError:
                continue
            if not (L_MIN <= Lv <= L_MAX and rho > 0 and np.isfinite(rho)):
                continue
            if qc != "none":
                # `flag` (p[18]) is constant 1 -> unusable; see QC_NOTES
                try:
                    xph, prom, terr = float(p[10]), float(p[11]), float(p[17])
                except (ValueError, IndexError):
                    continue
                if qc == "strict":
                    if not (prom >= 20.0 and abs(terr) <= 1.0 and xph >= 20.0):
                        continue
                else:                                    # standard
                    if not (prom >= 10.0 and abs(terr) <= 1.0):
                        continue
            # measured: 256 EMMA records carry s=60 (rounded), which datetime
            # rejects; clamping shifts them by <1 s, well inside the 1 h window
            t.append(datetime(y, mo, d, h, mi, min(s, 59)))
            L.append(Lv); MLT.append(mltv % 24); ne.append(rho)
    return _pack("EMMA", t, L, MLT, ne)


def load_awdanet(base, qc="standard"):
    """plasma_demo_pd_p_*: cols UTC station MLT L neq dL(%) dneq(%) ... [cm^-3]."""
    t, L, MLT, ne, sig = [], [], [], [], []
    for fp in sorted(glob.glob(os.path.join(base, "plasma_demo_pd_p_*.txt"))):
        for line in open(fp):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 7:
                continue
            try:
                tt = datetime.strptime(p[0].replace("UT", " ").split(".")[0],
                                       "%Y-%m-%d %H:%M:%S")
                mltv, Lv, nev = float(p[2]), float(p[3]), float(p[4])
                dL, dne = abs(float(p[5])), abs(float(p[6]))
            except (ValueError, IndexError):
                continue
            if not (L_MIN <= Lv <= L_MAX and nev > 0 and np.isfinite(nev)):
                continue
            if qc == "standard" and not (dne <= 40.0 and dL <= 40.0):
                continue
            if qc == "strict" and not (dne <= 30.0 and dL <= 30.0):
                continue
            t.append(tt); L.append(Lv); MLT.append(mltv % 24); ne.append(nev)
            sig.append(max(float(_dex(dne)), 0.02))   # per-record reported sigma
    return _pack("AWDANet", t, L, MLT, ne, sig)


def _pack(name, t, L, MLT, ne, sig=None):
    n = len(t)
    if sig is None:
        sig = np.full(n, SIGMA_DEX.get(name, 0.15))
    return dict(t=np.array(t), L=np.array(L, float), MLT=np.array(MLT, float),
                ne=np.array(ne, float), sig=np.array(sig, float),
                inst=name)


def load_all(data_dir, qc="standard"):
    """Load all three instruments; qc: none|standard|strict."""
    if qc not in ("none", "standard", "strict"):
        raise ValueError("qc must be none|standard|strict")
    return {
        "Arase":   load_arase(os.path.join(data_dir, "Arase"), qc),
        "EMMA":    load_emma(os.path.join(data_dir, "EMMA"), qc),
        "AWDANet": load_awdanet(os.path.join(data_dir, "AWDANet"), qc),
    }


def obs_in_window(obs, t0, t1, with_sigma=False):
    """Select (L, MLT, ne[, sig]) with t in [t0, t1)."""
    m = (obs["t"] >= t0) & (obs["t"] < t1)
    if with_sigma:
        return obs["L"][m], obs["MLT"][m], obs["ne"][m], obs["sig"][m]
    return obs["L"][m], obs["MLT"][m], obs["ne"][m]


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "DataAssimilation"
    if "--notes" in sys.argv:
        print(QC_NOTES); raise SystemExit
    for qc in ("none", "standard", "strict"):
        print("=== qc = %s ===" % qc)
        for name, o in load_all(base, qc=qc).items():
            if len(o["t"]) == 0:
                print("%-8s : 0 obs" % name); continue
            print("%-8s : %6d obs | t %s .. %s | L %.2f..%.2f | MLT %.1f..%.1f "
                  "| ne %.3g..%.3g | sig %.3f dex"
                  % (name, len(o["t"]), o["t"].min().date(), o["t"].max().date(),
                     o["L"].min(), o["L"].max(), o["MLT"].min(), o["MLT"].max(),
                     o["ne"].min(), o["ne"].max(), np.median(o["sig"])))
        print()
