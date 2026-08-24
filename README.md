# DGCPM + LETKF — plasmasphere data assimilation

An hourly-cycling, 16-member **Localized Ensemble Transform Kalman Filter**
(Hunt, Kostelich & Szunyogh, 2007) built around the **original Fortran
DGCPM/SWMF** model, driven by state injection through the model's own restart
files. It assimilates real electron-density observations from **Arase**, **EMMA**
and **AWDANet**, and is scored by **cross-validation** against a free run of the
same model with no assimilation: part of the observations go in, the rest are
withheld, and only the withheld ones are used for scoring.

**The DGCPM physics is not modified.** The Fortran source is untouched; the
ensemble member is written into a restart file and the model reads it back.

---

## Pipeline

Five layers, in dependency order.

```
LAYER 0  one-off, expensive, NEEDS DOCKER
  in : kp_apr27_may05.dat, Input/restart_cold/dgcpm_restart.dat
  run: dgcpm_engine.nature_run()          216 x 1 h SWMF.exe calls
  out: nature_run_fortran.npz             <- the free run; everything is scored
                                             against it
       |
       +--------------------------------------------+
       v                                            v
LAYER 3  baseline (offline, minutes)     LAYER 4  main experiment (NEEDS DOCKER)
  freerun_cv.py                            run_da_realobs.py
  -> freerun_cv_*.npz                      -> da_realobs_*.npz
       |         28 windows x 16 members = 448 SWMF calls    |
       +----------------------+-----------------------------+
                              v
                    LAYER 5  evaluation (offline, ~30 s)
                      analyse_da_realobs.py

Alongside the main chain, two layers that produce no result but establish
correctness:

LAYER 1  self-tests (offline, seconds)
  letkf.py                    filter mathematics            PASS / FAIL
  dgcpm_osse.py --selftest    cycling loop                  PASS / FAIL
  dgcpm_restart_io.py         restart write -> read round trip

LAYER 2  measurements (offline, minutes)
  obs_loader.py               how many observations survive QC, per instrument
  unit_bridge.py              MEASURES the +6 dex unit bridge (does not assume it)
  realobs.py                  observation bias, split independence diagnostics
```

> ### Layer 3 and layer 4 must run with the same flags
>
> `--w0`, `--nwin`, `--qc`, `--split` and `--seed` must be **identical** between
> `freerun_cv.py` and `run_da_realobs.py`. The split is seeded on
> `seed * 100003 + window`, so identical flags give bit-identical withheld sets.
> **If the flags differ, the two runs score different observations and the
> comparison is meaningless.**

---

## File map

| file | what it does |
|---|---|
| `letkf.py` | the filter mathematics: LETKF analysis, nearest-neighbour H, localization. Self-test: `python letkf.py` |
| `dgcpm_restart_io.py` | reads and writes the Fortran model's restart files — this is how ensemble state gets in and out without touching the Fortran |
| `dgcpm_engine.py` | steps the Fortran DGCPM one window at a time in Docker; Kp series; nature run |
| `dgcpm_osse.py` | the cycling loop plus the physical constraints (ceiling, increment cap, boundary rings). Self-test: `--selftest` |
| `obs_loader.py` | reads the raw Arase / EMMA / AWDANet observations and applies quality control |
| `realobs.py` | unit bridge (+6 dex), per-window observation assembly, cross-validation split, scoring |
| `unit_bridge.py` | measures the unit offset by comparing the nature run against the raw observations |
| `run_da_realobs.py` | the main experiment: assimilation of real observations with cross-validation (`--dry` runs without Docker) |
| `freerun_cv.py` | the no-assimilation baseline, scored on the same withheld observations (offline) |
| `analyse_da_realobs.py` | reproduces the summary numbers from the saved `.npz` files (offline) |

**The chain.** `dgcpm_engine` steps the Fortran model, with state passing through
`dgcpm_restart_io`. Each hour, `run_da_realobs` propagates 16 members;
`obs_loader` + `realobs` supply that hour's observations in model units together
with the assimilated/withheld split; `letkf` computes the analysis under the
constraints from `dgcpm_osse`; and the analysis becomes the next hour's initial
condition. Skill is measured on the withheld observations against the
`freerun_cv` baseline, and summarised by `analyse_da_realobs`.

---

## Method, in brief

- **State**: `log10(density)` on the DGCPM grid — 62 radial (L) x 120 azimuthal
  (MLT) cells = 7440 elements. Errors are quoted in **dex** (one dex = one order
  of magnitude; 0.3 dex is a factor of 2).
- **Observation operator**: nearest-neighbour selection of the grid cell,
  in the same Fortran column-major (`order='F'`) convention the model uses.
- **Localization**: Gaussian on `R^-1` (the Hunt 2007 variant), 0.8 R_E in L and
  3.0 h in MLT, cutoff at 3.5 normalized units. A grid point with no nearby
  observation keeps its forecast unchanged.
- **Unit bridge**: instruments publish cm^-3, the model computes in m^-3, so the
  offset is `log10(10^6) = 6.0` — derived, not fitted. `unit_bridge.py` states the
  expected value up front and then measures it, so the data can contradict it.
- **Cross-validation**: `--split instrument` assimilates EMMA and withholds
  everything else. This is the real independence test — with nearest-neighbour H,
  a random 80/20 split largely measures fit rather than skill, because the
  withheld observation is read from the very grid cell the analysis was fitted at.
  `realobs.split_independence` reports the same-cell fraction and the median
  localization weight, and the code prints a warning when the split is degenerate.
- **Physical constraints** (all inherited from earlier OSSE work, not re-tuned):
  density floor `1e2 m^-3`; analysis ceiling `saturation(L) + 1 dex`; increment cap
  8 dex inside, 5 dex outside; the outer 2 L-shells are left to the model boundary
  condition.

---

## Data (not in this repository)

None of the following is included here. Observational data comes from third
parties and is not ours to redistribute.

| needed | what it is | size |
|---|---|---|
| `nature_run_fortran.npz` | the free run — 216 hourly DGCPM states | ~12 MB |
| `DataAssimilation/` | raw Arase, EMMA and AWDANet observation files | large |
| `kp_apr27_may05.dat` | Kp index, 5-minute cadence, 2024-04-27 … 2024-05-05 | small |
| `Input/restart_cold/dgcpm_restart.dat` | cold-start restart file for the model | small |
| `freerun_cv_*.npz`, `da_realobs_*.npz` | run outputs, needed by `analyse_da_realobs.py` | small |

The Fortran DGCPM itself runs inside a Docker container (SWMF in stand-alone
Plasmasphere mode) named `dgcpm`; it is not part of this repository either.

---

## How to run

```bash
cd ~/Documents/ENKF && source .venv/bin/activate
```

The scripts expect the data files above to sit in the project root. Set
`DGCPM_ENKF_ROOT` if your root is elsewhere.

### Offline — no Docker needed

```bash
# self-tests: these must pass before anything else is meaningful
python clean_simple/letkf.py                    # PASS, RMSE 0.2029 -> 0.0259 dex
python clean_simple/dgcpm_osse.py --selftest    # PASS, free 0.016 -> LETKF 0.008
python clean_simple/dgcpm_restart_io.py         # restart round trip OK

# measurements
python clean_simple/unit_bridge.py              # measures the +6 dex unit bridge
python clean_simple/obs_loader.py DataAssimilation   # observation counts per QC level
python clean_simple/obs_loader.py --notes       # the QC notes and open questions
python clean_simple/realobs.py                  # bias estimate, split independence
```

### Baseline

```bash
python clean_simple/freerun_cv.py --w0 128 --nwin 28 --split instrument --tag inst
```

### Main experiment — needs Docker

```bash
docker start dgcpm

python clean_simple/run_da_realobs.py --w0 128 --nwin 28 --k 16 --qc standard \
    --split instrument --assim EMMA --repr 1.0 --rinfl 1.0 --kppert 0.15 \
    --rtps 0.0 --spread0 0.12 --spreadL 0.25 --tag inst
```

Cost: `nwin x k` = 448 SWMF calls.

### Evaluation

```bash
python clean_simple/analyse_da_realobs.py
```

Prints, per stage: free-run / DA-forecast / DA-analysis RMSE on the withheld
observations, the skill decomposed into a **cycling** term (earlier analyses that
survived propagation) and an **analysis** term (this window's increment at the
validation points), 95% confidence intervals from a block bootstrap over whole
windows, and breakdowns by storm phase and L band.

### Assembly check only — no Docker

```bash
python clean_simple/run_da_realobs.py --dry --nwin 6 --tag TEST
```

This substitutes a cheap surrogate propagator. It exercises the splits, R, the
constraints and the diagnostics, but **the numbers it produces are not a result** —
the code says so at runtime.

---

## Output tags

**Every setting that changes the trajectory must be encoded in `--tag`.**

Otherwise a sensitivity run silently overwrites the baseline it is meant to be
compared against. This has happened: an `rtps=0.8` run clobbered the `rtps=0.0`
stage-2 `.npz`, `.log` and `.png`, and the original raw output was lost.

The saved `.npz` records the full configuration (`k`, `w0`, `split`, `qc`, `seed`,
`rtps`, `inflation`, `rinfl`, `repr_dex`, `kppert`, `frac`, `debias`,
`assim_inst`), so a run can always be identified after the fact — but the file
name is what stops it being overwritten in the first place.

---

## Requirements

- Python and NumPy — no other third-party dependency. The self-tests and the
  evaluation numbers reproduce on both Python 3.11 / NumPy 2.4.4 and
  Python 3.14 / NumPy 2.5.1.
- Docker, with an SWMF/DGCPM container named `dgcpm`, for layers 0 and 4 only.
  Layers 1, 2, 3 and 5 run without it.

Environment variables, both optional, both defaulting to the current layout:

| variable | overrides | default |
|---|---|---|
| `DGCPM_ENKF_ROOT` | project root — all data paths are derived from it | `~/Documents/ENKF` |
| `DGCPM_CONTAINER` | Docker container name | `dgcpm` |

Nothing else is configurable by environment. Window length, ensemble size,
localization scales and every other setting that changes the trajectory are
command-line flags, so that they end up in `--tag` and in the saved `.npz`.

---

## Reference

Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007). *Efficient data
assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter.*
Physica D, 230(1–2), 112–126.
