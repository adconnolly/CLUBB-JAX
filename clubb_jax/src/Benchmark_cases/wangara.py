"""JAX port of wangara.F90 — Wangara day-33 surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/wangara.F90: `wangara_tndcy` (no large-scale forcing, no subsidence —
zeros thlm/rtm forcing + wm) and `wangara_sfclyr` (the analytic diurnal surface heat flux, cosine in AEST local
time, with a fixed moisture-flux ratio and ustar=0.13). prescribe_forcings.py imports them, mirroring the Fortran
case dispatch's `use wangara`.

Pure-numpy → bit-identical and differentiable.
"""

from __future__ import annotations

import math

import numpy as np

_SEC_PER_DAY = 86400.0
_PI = math.pi


def wangara_tndcy(state: dict) -> None:
    """Wangara large-scale tendencies (wangara.F90:wangara_tndcy): zero everything — no LS forcing,
    no subsidence. Zeros thlm_forcing/rtm_forcing and wm_zt/wm_zm in place."""
    state['thlm_forcing'][:] = 0.0
    state['rtm_forcing'][:] = 0.0
    state['wm_zt'][:] = 0.0
    state['wm_zm'][:] = 0.0


def wangara_sfclyr(time_current: float, ngrdcol: int) -> tuple:
    """Wangara day 33 analytic surface fluxes (wangara.F90:wangara_sfclyr).

    Cosine formula in AEST local time (UTC+10h). ustar = 0.13.
    """
    time_utc = time_current % _SEC_PER_DAY
    time_est = (time_utc + 36000.0) % _SEC_PER_DAY  # +10h → AEST
    ustar = np.full(ngrdcol, 0.13)
    wpthlp_val = 0.18 * math.cos((time_est - 45000.0) / 36000.0 * _PI)
    wpthlp_sfc = np.full(ngrdcol, wpthlp_val)
    wprtp_sfc  = 1.3e-4 * wpthlp_sfc
    return wpthlp_sfc, wprtp_sfc, ustar
