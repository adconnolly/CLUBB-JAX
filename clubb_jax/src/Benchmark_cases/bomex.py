"""JAX port of bomex.F90 — BOMEX case large-scale forcings + surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/bomex.F90: `bomex_tndcy` (analytic large-scale moisture tendency,
zero heat tendency) and `bomex_sfclyr` (prescribed time-interpolated surface heat/moisture fluxes, fixed
ustar). prescribe_forcings.py imports these, mirroring the Fortran case dispatch's `use bomex`.

Pure-numpy / tracer-transparent → bit-identical and differentiable. The specific-humidity→mixing-ratio
conversions live in their own Fortran-home module (spec_hum_to_mixing_ratio.py).
"""

from __future__ import annotations

import numpy as np

from clubb_jax.src.Benchmark_cases.spec_hum_to_mixing_ratio import (
    flux_spec_hum_to_mixing_ratio, force_spec_hum_to_mixing_ratio,
)


def bomex_tndcy(state: dict) -> None:
    """Analytic large-scale tendencies for BOMEX (bomex.F90:bomex_tndcy).

    Moisture tendency in terms of specific humidity (qtm), then converted
    to mixing ratio. No heat tendency (thlm_forcing = 0).
    """
    zt = state['gr'].zt   # (ngrdcol, nzt)
    rtm = state['rtm']

    qtm_forcing = np.where(
        zt < 300.0, -1.2e-8,
        np.where(zt < 500.0, -1.2e-8 * (1.0 - (zt - 300.0) / 200.0), 0.0)
    )
    state['thlm_forcing'][:] = 0.0
    state['rtm_forcing'][:] = force_spec_hum_to_mixing_ratio(rtm, qtm_forcing)


def bomex_sfclyr(state: dict, time_current: float, ngrdcol: int,
                     rtm_bot: np.ndarray) -> tuple:
    """BOMEX surface fluxes from time-interpolated sfc file (bomex.F90:bomex_sfclyr).

    rtm_bot is the bottom-level total water mixing ratio from read_surface_var_for_bc
    (at z_bot=25 m for the convergence-test BC, else lowest zt level).
    Returns: wpthlp_sfc, wprtp_sfc, ustar  (all shape (ngrdcol,))
    """
    fd = state['_forcings_data']
    sfc = fd['sfc']
    wpthlp_sfc_val = float(np.interp(time_current, sfc['time'], sfc['wpthlp_sfc']))
    wpqtp_sfc_val  = float(np.interp(time_current, sfc['time'], sfc['wpqtp_sfc']))
    ustar_val = 0.28

    wpqtp_sfc = np.full(ngrdcol, wpqtp_sfc_val)
    wpthlp_sfc = np.full(ngrdcol, wpthlp_sfc_val)
    ustar = np.full(ngrdcol, ustar_val)
    wprtp_sfc = flux_spec_hum_to_mixing_ratio(rtm_bot, wpqtp_sfc)
    return wpthlp_sfc, wprtp_sfc, ustar
