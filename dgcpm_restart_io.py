#!/usr/bin/env python3
"""DGCPM restart-file I/O: inject/extract ensemble state without touching the Fortran.

In: a restart path (read), or a template dict plus a (nTheta,nPhi) density array
(write).  Out: a dict of grid + fields, or a 6-block restart file on disk.
"""
# layout: line 1 is nTheta nPhi, then the blocks vtheta, vphi, den, x, y, oc
# see src/ModIoDGCPM.f90:100 (read) and src/PS_wrapper.f90:703 (write); a save
# file appends pot, vr, vp, n, vol, which the loader ignores
# den is in m^-3, NOT cm^-3: the Fortran saturation() already carries the 1e6
# conversion (see src/ModFunctionsDGCPM.f90:26-28) and nothing on either side of
# this module rescales it
# measured: restartOUT den max 3.37e+09, absurd as cm^-3 but exactly saturation(L)
# at the innermost shell L=1.298 in m^-3
# all arrays are Fortran column-major (theta fastest), same as the LETKF state
import numpy as np


def read_restart(path):
    """Read a DGCPM restart file into a dict with grid + fields."""
    with open(path) as f:
        tok = f.read().split()
    i = 0
    nTheta = int(tok[i]); nPhi = int(tok[i + 1]); i += 2
    n2 = nTheta * nPhi

    def take(n):
        nonlocal i
        a = np.array(tok[i:i + n], dtype=float); i += n
        return a

    vtheta = take(nTheta)
    vphi = take(nPhi)
    den = take(n2).reshape(nTheta, nPhi, order='F')
    x = take(n2).reshape(nTheta, nPhi, order='F')
    y = take(n2).reshape(nTheta, nPhi, order='F')
    oc = take(n2).reshape(nTheta, nPhi, order='F')

    extra = {}
    for name in ('pot', 'vr', 'vp', 'n', 'vol'):    # optional save-file blocks
        if i + n2 <= len(tok):
            extra[name] = take(n2).reshape(nTheta, nPhi, order='F')

    L = 1.0 / np.sin(np.deg2rad(vtheta)) ** 2       # dipole L from co-latitude
    MLT = vphi / 15.0
    return dict(nTheta=nTheta, nPhi=nPhi, vtheta=vtheta, vphi=vphi,
                den=den, x=x, y=y, oc=oc, L=L, MLT=MLT, extra=extra)


def _fmt_flat(a, per_line=5):
    flat = np.asarray(a).flatten(order='F')
    out = []
    for k in range(0, len(flat), per_line):
        out.append(' '.join('%.8E' % v for v in flat[k:k + per_line]))
    return '\n'.join(out)


def write_restart(path, ref, den):
    """Write a 6-block restart: geometry from `ref`, density `den` injected."""
    nTheta, nPhi = ref['nTheta'], ref['nPhi']
    den = np.asarray(den)
    if den.shape != (nTheta, nPhi):
        raise ValueError('den shape %s != grid (%d,%d)'
                         % (den.shape, nTheta, nPhi))
    with open(path, 'w') as f:
        f.write('%d %d\n' % (nTheta, nPhi))
        f.write(' '.join('%.8E' % v for v in ref['vtheta']) + '\n')
        f.write(' '.join('%.8E' % v for v in ref['vphi']) + '\n')
        f.write(_fmt_flat(den) + '\n')
        f.write(_fmt_flat(ref['x']) + '\n')
        f.write(_fmt_flat(ref['y']) + '\n')
        f.write(_fmt_flat(ref['oc']) + '\n')


def state_from_restart(r):
    """LETKF state vector: density flattened in Fortran order."""
    return r['den'].flatten(order='F')


if __name__ == '__main__':
    import sys, tempfile, os
    path = sys.argv[1] if len(sys.argv) > 1 else \
        'Input/restart_cold/dgcpm_restart.dat'
    r = read_restart(path)
    print('read  : grid %d x %d = %d' % (r['nTheta'], r['nPhi'],
                                         r['nTheta'] * r['nPhi']))
    print('        L range   %.3f .. %.3f Re' % (r['L'].min(), r['L'].max()))
    print('        MLT range %.2f .. %.2f h' % (r['MLT'].min(), r['MLT'].max()))
    print('        den min/max/mean %.4g / %.4g / %.4g'
          % (r['den'].min(), r['den'].max(), r['den'].mean()))
    print('        extra blocks present:', list(r['extra'].keys()) or 'none')

    # the analysis reaches the Fortran only through write_restart, so a formatting
    # loss here would silently corrupt every assimilation cycle
    # measured: True on the real cold restart; %.8E keeps ~8 digits, rtol=1e-6
    tmp = os.path.join(tempfile.gettempdir(), 'rt_restart.dat')
    write_restart(tmp, r, r['den'])
    r2 = read_restart(tmp)
    ok = (r2['nTheta'] == r['nTheta'] and r2['nPhi'] == r['nPhi']
          and np.allclose(r2['den'], r['den'], rtol=1e-6, atol=0)
          and np.allclose(r2['vtheta'], r['vtheta'], rtol=1e-6)
          and np.allclose(r2['oc'], r['oc'], rtol=1e-6))
    print('round-trip write->read identical:', ok)
