#!/usr/bin/env python3
"""Host-side driver: steps the Fortran DGCPM/SWMF one window at a time via docker.

In: a restart file or a density field in m^-3, a Kp value, a window length.  Out:
the restart dict the Fortran wrote back, i.e. the propagated state.
"""
# one window = one short SWMF.exe run, restart-in -> restart-out, with Kp passed
# per window as `#KP const <value>`; requires a running container, by default
# `dgcpm` (DGCPM_CONTAINER), under the project root (DGCPM_ENKF_ROOT)
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta

import numpy as np

from dgcpm_restart_io import read_restart, write_restart, state_from_restart

# host paths, all derived from ENKF: only the root and the container name come
# from the environment, so nothing that changes a run can be redirected unseen
ENKF = os.environ.get("DGCPM_ENKF_ROOT", os.path.expanduser("~/Documents/ENKF"))
WORK = os.path.join(ENKF, "letkf_work")
CONTAINER = os.environ.get("DGCPM_CONTAINER", "dgcpm")
RUN_WINDOW_HOST = os.path.join(ENKF, "DGCPM_docker", "run_window.sh")
COLD_RESTART = os.path.join(ENKF, "Input/restart_cold/dgcpm_restart.dat")
KP_FILE = os.path.join(ENKF, "kp_apr27_may05.dat")
NATURE_START = datetime(2024, 4, 27, 0, 0, 0)         # matches the Kp data window

# DGCPM_THROTTLE = seconds of idle per second of compute (thermal relief only;
# sleep never changes the numerical result)
THROTTLE = float(os.environ.get("DGCPM_THROTTLE", "0") or 0)
THROTTLE_MAX = float(os.environ.get("DGCPM_THROTTLE_MAX", "30") or 30)


def load_kp_series(path=KP_FILE):
    """Read the piecewise-constant Kp series (time, value) from kp_apr27_may05.dat."""
    times, vals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            p = line.split()
            if len(p) < 3:
                continue
            try:
                t = datetime.strptime(p[0] + ' ' + p[1], '%Y-%m-%d %H:%M')
                v = float(p[2])
            except ValueError:
                continue
            times.append(t); vals.append(v)
    return np.array(times), np.array(vals)


def kp_at(times, vals, t):
    """Kp value active at time t (last entry with time <= t)."""
    idx = np.searchsorted(times, t, side='right') - 1
    return float(vals[max(idx, 0)])


def gen_param_in(kp, timemax_s, start_dt, output_interval=600.0):
    """Generate the PARAM.in for one window (const Kp, TimeMax=window length)."""
    y, mo, d = start_dt.year, start_dt.month, start_dt.day
    h, mi, s = start_dt.hour, start_dt.minute, start_dt.second
    return f"""#COMPONENTMAP
PS 0 0 1		CompMap runs on 1 PE

#DESCRIPTION
DGCPM window: const Kp={kp}, TimeMax={timemax_s}s		StringDescription

#PLANET
EARTH			NamePlanet

#SAVERESTART
T			DoSaveRestart
-1			DnSaveRestart
{timemax_s:.1f}			DtSaveRestart

#TIMEACCURATE
T			IsTimeAccurate

#STARTTIME
{y}			iYear
{mo}			iMonth
{d}			iDay
{h}			iHour
{mi}			iMinute
{s}			iSecond
0.0			FracSecond

#IDEALAXES

#ROTATION
F			UseRotation

#BEGIN_COMP PS ---------------------------------------------------------------

#TIMESTEP
20.0			DtStep

#KP
const			NameSourceKp
{kp}			ConstKp

#OUTPUT
T			WriteStatic
T			WriteDynamic
{output_interval}			OutputInterval
SHORT			OutputType
DIPOLE			MagneticType

#LOG
T			WriteLogFile

#END_COMP PS -----------------------------------------------------------------

#STOP
-1			MaxIter
{timemax_s:.1f}			TimeMax

#END
"""


def _fsync(path):
    # flush to disk so the Docker-Desktop mount never serves a partial file
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


BIN = "/work/SWMF/bin"   # SWMF run dir inside the container


def _run_container_window():
    # files go via `docker cp` (the macOS mount has write->read latency)
    par = os.path.join(WORK, "PARAM.in")
    rin = os.path.join(WORK, "restartIN.dat")
    rout = os.path.join(WORK, "restartOUT.dat")

    def dk(*args):
        # capture stderr so a dead container is not just "exit status 1"
        p = subprocess.run(["docker", *args], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(
                "docker %s\n  exit=%d\n  stderr: %s\n  stdout: %s"
                % (" ".join(args[:3]), p.returncode,
                   p.stderr.strip() or "(empty)", p.stdout.strip() or "(empty)"))

    dk("exec", CONTAINER, "bash", "-c",
       f"mkdir -p {BIN}/PS/Output {BIN}/PS/restartIN {BIN}/PS/restartOUT && "
       f"rm -f {BIN}/PS/restartOUT/dgcpm_restart.dat")
    dk("cp", par, f"{CONTAINER}:{BIN}/PARAM.in")
    dk("cp", rin, f"{CONTAINER}:{BIN}/PS/restartIN/dgcpm_restart.dat")
    t_exec = time.time()
    try:
        dk("exec", CONTAINER, "bash", "-c",
           f"cd {BIN} && ./SWMF.exe > /tmp/dgcpm_run.log 2>&1; "
           f"test -f PS/restartOUT/dgcpm_restart.dat")
    except subprocess.CalledProcessError:
        # dead branch: dk() raises RuntimeError, never CalledProcessError, so this
        # log rescue never runs; left as-is to keep the behaviour byte-identical
        subprocess.run(["docker", "cp", f"{CONTAINER}:/tmp/dgcpm_run.log", rout + ".log"])
        raise RuntimeError("SWMF.exe produced no restart; see " + rout + ".log")
    t_exec = time.time() - t_exec
    dk("cp", f"{CONTAINER}:{BIN}/PS/restartOUT/dgcpm_restart.dat", rout)
    r = read_restart(rout)
    if THROTTLE > 0:
        time.sleep(min(THROTTLE * t_exec, THROTTLE_MAX))   # cool-down, sleep only
    return r


def propagate_restart(restart_in_path, kp, timemax_s, start_dt):
    """Run one window from an existing restart file. Returns restart dict."""
    os.makedirs(WORK, exist_ok=True)
    rin = os.path.join(WORK, "restartIN.dat")
    par = os.path.join(WORK, "PARAM.in")
    shutil.copy(restart_in_path, rin)
    with open(par, "w") as f:
        f.write(gen_param_in(kp, timemax_s, start_dt)); f.flush(); os.fsync(f.fileno())
    _fsync(rin)
    return _run_container_window()


def propagate_field(den, template, kp, timemax_s, start_dt):
    """Inject a density field on the template geometry, run one window."""
    # den is in m^-3, the units read_restart hands back; no conversion happens
    # anywhere in this pipeline, see dgcpm_restart_io
    os.makedirs(WORK, exist_ok=True)
    rin = os.path.join(WORK, "restartIN.dat")
    par = os.path.join(WORK, "PARAM.in")
    write_restart(rin, template, den)
    with open(par, "w") as f:
        f.write(gen_param_in(kp, timemax_s, start_dt)); f.flush(); os.fsync(f.fileno())
    _fsync(rin)
    return _run_container_window()


def nature_run(n_windows, window_s=3600, start=NATURE_START,
               out_npz=os.path.join(ENKF, "nature_run_fortran.npz")):
    """Chain windows from the cold restart with the real Kp series -> npz."""
    times, vals = load_kp_series()
    prev = COLD_RESTART
    states, tstamps, grid = [], [], None
    for i in range(n_windows):
        t = start + timedelta(seconds=i * window_s)
        kp = kp_at(times, vals, t)
        r = propagate_restart(prev, kp, window_s, t)
        if grid is None:
            grid = dict(nTheta=r['nTheta'], nPhi=r['nPhi'],
                        L=r['L'], MLT=r['MLT'])
        states.append(state_from_restart(r))
        tstamps.append(t.isoformat())
        # keep a private copy of restartOUT so the next window's cp can't clobber it
        prev = os.path.join(WORK, "restartOUT.dat")
        prev_keep = os.path.join(WORK, "chain_prev.dat")
        shutil.copy(prev, prev_keep); prev = prev_keep
        print(f"[{i+1}/{n_windows}] {t}  Kp={kp}  "
              f"den(min/max)={r['den'].min():.3g}/{r['den'].max():.3g}")
    np.savez_compressed(out_npz, states=np.array(states),
                        times=np.array(tstamps), **grid)
    print("saved", out_npz)


if __name__ == "__main__":
    import sys
    if "--nature" in sys.argv:
        n = 216
        for a in sys.argv:
            if a.startswith("--n="):
                n = int(a.split("=")[1])
        nature_run(n)
    else:
        # dry run (no docker): validate Kp parsing + PARAM.in generation
        times, vals = load_kp_series()
        print("Kp series: %d entries, %s .. %s"
              % (len(vals), times[0], times[-1]))
        for hh in (0, 130, 138):
            t = NATURE_START + timedelta(hours=hh)
            print("  window @ +%dh (%s): Kp=%.1f"
                  % (hh, t, kp_at(times, vals, t)))
        print("\n--- sample PARAM.in for window @ +138h (storm peak) ---")
        t = NATURE_START + timedelta(hours=138)
        print(gen_param_in(kp_at(times, vals, t), 3600, t))
        print("DRY-RUN OK (no container called). "
              "Run on the Mac with:  python dgcpm_engine.py --nature --n=216")
