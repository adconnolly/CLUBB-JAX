"""JAX port of neutral_case.F90 — the neutral-stratification case surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/neutral_case.F90: `neutral_case_sfclyr` — a prescribed-flux surface
layer (fixed ustar=0.5, a constant w'thl' surface flux that switches off after t=80880 s, zero w'rt') with the
surface momentum fluxes from `sfc_flux:compute_momentum_flux`. prescribe_forcings.py imports it, mirroring the
Fortran case dispatch's `use neutral_case`.

Pure-numpy / tracer-transparent → bit-identical and differentiable.
"""

from __future__ import annotations

import numpy as np

from clubb_jax.src.Benchmark_cases.sfc_flux import compute_momentum_flux


def neutral_case_sfclyr(ngrdcol: int, time, um_sfc, vm_sfc, ubar):
    """Neutral-case surface fluxes (neutral_case.F90:neutral_case_sfclyr).

    ustar = 0.5; wpthlp_sfc = 0.05 until t=80880 s, then 0; wprtp_sfc = 0; the surface momentum fluxes
    upwp_sfc/vpwp_sfc come from compute_momentum_flux. Returns (upwp_sfc, vpwp_sfc, wpthlp_sfc, wprtp_sfc, ustar)
    in the Fortran out-arg order.
    """
    ustar = np.full(ngrdcol, 0.5)
    wpthlp_sfc = np.full(ngrdcol, 0.0 if time > 80880.0 else 0.05)
    wprtp_sfc = np.zeros(ngrdcol)
    upwp_sfc, vpwp_sfc = compute_momentum_flux(um_sfc, vm_sfc, ubar, ustar)
    return upwp_sfc, vpwp_sfc, wpthlp_sfc, wprtp_sfc, ustar
