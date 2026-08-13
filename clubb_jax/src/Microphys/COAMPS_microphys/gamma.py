"""JAX port of COAMPS_microphys/gamma.F — the COAMPS gamma-function approximation.

Mirrors `subroutine gamma(arg, val)`. This is NOT the ANL gamma used by the rest of
CLUBB; the COAMPS driver uses this cruder series form purely to reproduce COAMPS-LES
results (see coamps_microphys_driver_module.F90 comment near the `call gamma` block).

The Fortran reduces arg to (1,2] by pulling out integer factors (fac *= zarg while
zarg>2), then evaluates a 15-term power series 1/Gamma(1+z) ≈ sum_k c_k z^k and returns
fac / series. It is only ever called with fixed scalar constants (3.0, 4.0, ... , and
the bsnow/bgrp-derived magic numbers), so a scalar float implementation is faithful.
"""

from __future__ import annotations

# Series coefficients for 1/Gamma over z in (0,1] region (Fortran DATA c/.../).
_C = (
    1.000000, 0.577210, -0.655870, -0.042000, 0.166530,
    -0.042190, -0.009620, 0.007210, -0.001160, -0.000210,
    0.000120, -0.000020, -0.000001, 0.000001, 2.0e-7,
)


def gamma(arg):
    """COAMPS gamma(arg) -> val (scalar float). Faithful to gamma.F."""
    zarg = float(arg)
    fac = 1.0
    # Reduce zarg into (1, 2] pulling factors into fac.
    while zarg > 2.0:
        zarg = zarg - 1.0
        fac = fac * zarg
    wamma = 0.0
    zpow = 1.0
    for k in range(1, 16):
        zpow = zpow * zarg          # zarg**k
        wamma = wamma + _C[k - 1] * zpow
    return fac / wamma
