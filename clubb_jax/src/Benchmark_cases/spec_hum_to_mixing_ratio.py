"""JAX port of spec_hum_to_mixing_ratio.F90 — specific-humidity ↔ total-water-mixing-ratio conversions.

Mirrors clubb_release/src/Benchmark_cases/spec_hum_to_mixing_ratio.F90. The two routines convert a flux or a
forcing expressed in terms of specific humidity (q_t) into the equivalent in terms of total water mixing ratio
(r_t), via the linearised relation r_t = q_t/(1 - q_t) ⇒ d/dq_t = (1 + r_t)^2. Used by the case forcing/surface
routines (bomex, gabls2, …) in Benchmark_cases/prescribe_forcings.py, mirroring the Fortran `use
spec_hum_to_mixing_ratio` there.

Pure-arithmetic (numpy-in → numpy-out, type-preserving) → bit-identical and differentiable.
"""

from __future__ import annotations

import numpy as np


def flux_spec_hum_to_mixing_ratio(rtm_sfc: np.ndarray, wpqtp: np.ndarray) -> np.ndarray:
    """w'r_t' = (1 + r_tm)^2 * w'q_t'  (linearised) — spec_hum_to_mixing_ratio.F90:flux_spec_hum_to_mixing_ratio."""
    return (1.0 + rtm_sfc) ** 2 * wpqtp


def force_spec_hum_to_mixing_ratio(rtm: np.ndarray, qtm_forcing: np.ndarray) -> np.ndarray:
    """rtm_forcing = (1 + rtm)^2 * qtm_forcing  (linearised) — spec_hum_to_mixing_ratio.F90:force_spec_hum_to_mixing_ratio."""
    return (1.0 + rtm) ** 2 * qtm_forcing
