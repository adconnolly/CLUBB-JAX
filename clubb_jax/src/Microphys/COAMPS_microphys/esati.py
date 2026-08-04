"""JAX port of COAMPS_microphys/esati.F — saturation vapour pressure over ice.

Mirrors `subroutine esati(t, w1, m)`. The Fortran default (`l_new=.true.`) calls
`esat_new(t, w1, m, l_ice=.true.)`; the legacy Teton table branch is dead. Returns
w1 = saturation vapour pressure over ice [Pa].
"""

from __future__ import annotations

from clubb_jax.src.Microphys.COAMPS_microphys.esat_new import esat_new


def esati(t):
    """Saturation vapour pressure over ice [Pa] at temperature t [K]."""
    return esat_new(t, l_ice=True)
