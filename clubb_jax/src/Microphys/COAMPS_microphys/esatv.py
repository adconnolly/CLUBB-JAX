"""JAX port of COAMPS_microphys/esatv.F — saturation vapour pressure over water.

Mirrors `subroutine esatv(t, w1, m)`. The Fortran default (`l_new=.true.`) calls
`esat_new(t, w1, m, l_ice=.false.)`; the legacy Teton table branch is dead. Returns
w1 = saturation vapour pressure over water [Pa].
"""

from __future__ import annotations

from clubb_jax.src.Microphys.COAMPS_microphys.esat_new import esat_new


def esatv(t):
    """Saturation vapour pressure over water [Pa] at temperature t [K]."""
    return esat_new(t, l_ice=False)
