"""JAX port of ekman.F90 — the Ekman-spiral case surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/ekman.F90: `ekman_sfclyr` — a neutral, dry surface layer (fixed
ustar=0.3, zero w'thl' and w'rt' surface fluxes) with the surface momentum fluxes from
`sfc_flux:compute_momentum_flux`. prescribe_forcings.py imports it, mirroring the Fortran case dispatch's
`use ekman`. (The ekman case also needs sponge-layer damping wired in to be bit-faithful — see
sponge_layer_damping.py.)

Pure-numpy / tracer-transparent → bit-identical and differentiable.
"""

from __future__ import annotations

import numpy as np

from clubb_jax.src.Benchmark_cases.sfc_flux import compute_momentum_flux


def ekman_sfclyr(ngrdcol: int, um_sfc, vm_sfc, ubar):
    """Ekman-case surface fluxes (ekman.F90:ekman_sfclyr).

    ustar = 0.3; wpthlp_sfc = 0; wprtp_sfc = 0; the surface momentum fluxes upwp_sfc/vpwp_sfc come from
    compute_momentum_flux. Returns (upwp_sfc, vpwp_sfc, wpthlp_sfc, wprtp_sfc, ustar) in the Fortran out-arg order.
    """
    ustar = np.full(ngrdcol, 0.3)
    wpthlp_sfc = np.zeros(ngrdcol)
    wprtp_sfc = np.zeros(ngrdcol)
    upwp_sfc, vpwp_sfc = compute_momentum_flux(um_sfc, vm_sfc, ubar, ustar)
    return upwp_sfc, vpwp_sfc, wpthlp_sfc, wprtp_sfc, ustar
