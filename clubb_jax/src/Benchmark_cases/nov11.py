"""JAX port of nov11.F90 — the NOV-11 altocumulus case special total-water adjustment.

Mirrors clubb_release/src/Benchmark_cases/nov11.F90: `nov11_altocu_rtm_adjust` — a one-time adjustment of the
total water above cloud. One hour after the initial time (at the single timestep where
`time_initial + 3600 <= time < time_initial + 3600 + dt`), the total water mixing ratio above 2900 m + the
surface elevation is multiplied by 0.89 (a documented "magic number"). The case's large-scale forcing is read
through the generic time-dependent path and its `nov11_altocu_tndcy` subsidence is obsolete (commented out in the
Fortran); only this rtm adjustment is case-specific. prescribe_forcings.F90 calls it for the nov11_altocu case.

Pure-numpy → bit-identical (the case is Morrison/ice FP-limited end-to-end, but this routine mirrors the oracle).
"""

from __future__ import annotations

import numpy as np


def nov11_altocu_rtm_adjust(rtm, gr, time_current: float, time_initial: float, dt: float):
    """One-time above-cloud total-water adjustment (nov11.F90:nov11_altocu_rtm_adjust).

    A no-op except on the single timestep at/after t = time_initial + 3600 s, where
    rtm *= 0.89 for every thermodynamic level above 2900 m + the surface elevation gr.zm[:, 0].
    Returns the (possibly adjusted) rtm array, shape (ngrdcol, nzt).
    """
    if not (time_initial + 3600.0 <= time_current < time_initial + 3600.0 + dt):
        return rtm
    zt = np.asarray(gr.zt, dtype=np.float64)
    z_sfc = np.asarray(gr.zm, dtype=np.float64)[:, 0:1]     # gr%zm(i, 1)
    rtm_a = np.asarray(rtm, dtype=np.float64)
    return np.where(zt > (2900.0 + z_sfc), 0.89 * rtm_a, rtm_a)
