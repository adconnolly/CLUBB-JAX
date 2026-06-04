"""JAX port of Radiation/simple_rad_module.F90:simple_rad_lba — the LBA prescribed radiative-cooling scheme.

The LBA deep-convection case uses a prescribed radiative-heating-rate table `lba_krad` (33 altitudes × 36 times,
every 600 s) read from `lba_forcings/lba_heights.dat` + `lba_forcings/lba_rad.dat`. Each model step the table is
linearly interpolated in time, then vertically interpolated (zero-extrapolation) to the thermodynamic grid.

Self-contained + differentiable (piecewise-linear in time, linear in the table values). Ports the `simple_rad_lba`
+ `simple_rad_lba_init` routines; the `rad_scheme = 'lba'` case is otherwise blocked by interactive SILHS, but the
radiation scheme itself is faithful and tested here against a literal NumPy transcription.
"""
import os

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.Benchmark_cases.generic_forcings import _zlinterp

_LBA_NZRAD = 33
_LBA_NTIMES = 36
_LBA_DT = 600.0   # seconds between the prescribed radiation times (simple_rad_module.F90)


def load_lba_rad_table(forcings_dir):
    """Read lba_heights.dat (33,) and lba_rad.dat (33 levels × 36 times) — simple_rad_lba_init.

    file_read_2d fills `lba_krad(level, time)` level-major (each level's 36 time values in sequence), so the
    flat token stream reshapes to (33, 36).
    """
    zrad = np.fromstring(open(os.path.join(forcings_dir, "lba_heights.dat")).read().replace(",", " "), sep=" ")
    krad = np.fromstring(open(os.path.join(forcings_dir, "lba_rad.dat")).read().replace(",", " "), sep=" ")
    assert zrad.size == _LBA_NZRAD, f"lba_heights.dat: expected {_LBA_NZRAD}, got {zrad.size}"
    assert krad.size == _LBA_NZRAD * _LBA_NTIMES, f"lba_rad.dat: expected {_LBA_NZRAD*_LBA_NTIMES}, got {krad.size}"
    return zrad, krad.reshape(_LBA_NZRAD, _LBA_NTIMES)


def lba_radhtz(time, lba_krad):
    """Time-interpolate the radiative-heating column radhtz(33) for elapsed `time` [s] (simple_rad_lba time loop).

    time <= 600 -> column 0; time >= 36*600 -> column 35; else linear between the bracketing 600-s columns
    (Fortran 1-based i1=floor(time/600): radhtz = a*krad[:,i2] + (1-a)*krad[:,i1], a=(time-600 i1)/600).
    """
    krad = jnp.asarray(lba_krad)
    if time <= _LBA_DT:
        return krad[:, 0]
    if time >= _LBA_NTIMES * _LBA_DT:
        return krad[:, _LBA_NTIMES - 1]
    i1 = int(time // _LBA_DT)                      # 1-based bracketing index (matches the Fortran loop)
    a = (time - _LBA_DT * i1) / _LBA_DT
    return a * krad[:, i1] + (1.0 - a) * krad[:, i1 - 1]   # 0-based: i2=i1, i1=i1-1


def simple_rad_lba(time, lba_zrad, lba_krad, zt):
    """Radiative heating rate on the thermodynamic grid (simple_rad_module.F90:simple_rad_lba).

    Args:
        time:     elapsed time since start [s] (= time_current - time_initial).
        lba_zrad: source altitudes (33,).
        lba_krad: heating-rate table (33, 36) [K/s].
        zt:       thermodynamic-grid heights (nzt,) for one column.
    Returns:
        radht (nzt,) [K/s].
    """
    radhtz = np.asarray(lba_radhtz(time, lba_krad))
    return _zlinterp(np.asarray(zt), np.asarray(lba_zrad), radhtz)
