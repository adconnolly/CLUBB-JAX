"""JAX port of COAMPS_microphys/esat_new.F — Goff-Gratch saturation vapour pressure.

Mirrors `subroutine esat_new(tp, vp, len, l_ice)`. Returns the saturation vapour
pressure `vp` [Pa] as a function of temperature `tp` [K], over ice (l_ice=True below
0 C, over water at/above 0 C) or over water (l_ice=False).

PORTING DEVIATION: the Fortran builds a 0.1 K lookup table (2751 entries, 120.16 K ..
395 K) once and linearly interpolates. This port evaluates the SAME Goff-Gratch (1946
Smithsonian) closed-form directly (the `l_interp=.false.` "exact calculation" branch of
the Fortran) — forward-nearly-identical (differs only by the table's linear-interp
error on a smooth function at 0.1 K resolution) and fully differentiable. Constants and
the over-ice / over-water blend at 273.16 K are copied verbatim from the F77 source.
"""

from __future__ import annotations

from clubb_jax.src.CLUBB_core.tracer_numpy import _xp

_ESBASW = 1013246.0
_TBASW = 373.16
_ESBASI = 6107.1
_TBASI = 273.16


def esat_new(tp, l_ice):
    """Saturation vapour pressure [Pa] at temperature tp [K]. Vectorised, differentiable."""
    xp = _xp
    tem = tp

    def _log10(x):
        return xp.log10(x)

    # SATURATION VAPOR PRESSURE OVER ICE  (Goff-Gratch)
    aa = -9.09718 * (_TBASI / tem - 1.0)
    b = -3.56654 * _log10(_TBASI / tem)
    c = 0.876793 * (1.0 - tem / _TBASI)
    e = _log10(_ESBASI)
    esice = 10.0 ** (aa + b + c + e)
    esice = 0.1 * esice   # microbars -> Pascals

    # SATURATION VAPOR PRESSURE OVER WATER
    aa = -7.90298 * (_TBASW / tem - 1.0)
    b = 5.02808 * _log10(_TBASW / tem)
    c = -1.3816e-07 * (10.0 ** ((1.0 - tem / _TBASW) * 11.344) - 1.0)
    d = 8.1328e-03 * (10.0 ** ((_TBASW / tem - 1.0) * (-3.49149)) - 1.0)
    e = _log10(_ESBASW)
    esh2o = 10.0 ** (aa + b + c + d + e)
    esh2o = 0.1 * esh2o   # microbars -> Pascals

    if l_ice:
        # over ice, but above 0 C use ESH2O
        vp = xp.where(tem >= 273.16, esh2o, esice)
    else:
        vp = esh2o
    return vp
