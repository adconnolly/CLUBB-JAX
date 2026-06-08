"""Monin-Obukhov surface friction velocity — pure Python port of diag_ustar_module.F90.

Mirrors clubb_release/src/Benchmark_cases/diag_ustar_module.F90, whose single function
`diag_ustar` diagnoses the surface friction velocity u* from the Monin-Obukhov similarity
functions and a specified surface buoyancy flux. The benchmark surface schemes (arm.py,
prescribe_forcings.py, lba.py, arm_97.py) call it through `use diag_ustar_module`.

Two forms:
  _diag_ustar      — direct float port (the concrete-run path; bit-identical to Fortran)
  diag_ustar  — differentiable jnp mirror used only under a jax.grad trace
"""

import math

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.tracer_numpy import _safe_sqrt
# vonk mirrors diag_ustar_module.F90 `use constants_clubb, only: vonk` (= 0.4, bit-identical)
from clubb_jax.src.CLUBB_core.constants_clubb import vonk as _vonk

# Monin-Obukhov similarity constants — diag_ustar_module.F90 local parameters (not in constants_clubb)
_am = 4.8
_bm = 19.3


def _diag_ustar(z: float, bflx: float, wnd: float, z0: float) -> float:
    """Monin-Obukhov ustar — direct port of diag_ustar_module.F90."""
    lnz  = math.log(z / z0)
    klnz = _vonk / lnz
    c1   = math.pi / 2.0 - 3.0 * math.log(2.0)
    ustar = wnd * klnz

    if abs(bflx) > 1.0e-6:
        for _ in range(4):
            lmo  = -(ustar ** 3) / (_vonk * bflx)
            zeta = z / lmo
            if zeta > 0.0:
                if zeta > 1.0e10:
                    ustar = 1.0e-10
                    break
                ustar = _vonk * wnd / (lnz + _am * zeta)
            else:
                x    = math.sqrt(math.sqrt(1.0 - _bm * zeta))
                psi1 = (2.0 * math.log(1.0 + x)
                        + math.log(1.0 + x * x)
                        - 2.0 * math.atan(x)
                        + c1)
                ustar = wnd * _vonk / (lnz - psi1)

    return ustar


def diag_ustar(z, bflx, wnd, z0):
    """Differentiable jnp mirror of ``_diag_ustar`` (used ONLY under a jax.grad trace).

    The two data-dependent branches (``abs(bflx)>1e-6`` gate; stable ``zeta>0`` vs unstable, plus the
    ``zeta>1e10`` clamp) become ``jnp.where`` selections, and the 4-iteration Monin-Obukhov loop is
    unrolled. Forward-identical to ``_diag_ustar`` on the taken branch; finite reverse-mode gradient
    (``bflx`` guarded away from 0 so the unused branch can't poison the grad; ``_safe_sqrt`` for the
    quartic root)."""
    lnz   = jnp.log(z / z0)
    klnz  = _vonk / lnz
    c1    = jnp.pi / 2.0 - 3.0 * jnp.log(2.0)
    ustar = wnd * klnz
    apply = jnp.abs(bflx) > 1.0e-6
    bflx_safe = jnp.where(apply, bflx, 1.0)   # avoid 1/0 in lmo when the MO correction is inactive
    for _ in range(4):
        lmo  = -(ustar ** 3) / (_vonk * bflx_safe)
        # Sign-preserving floor on lmo (REFACTOR B5): in very stable conditions (gabls3) the MO iteration
        # drives ustar->0 so lmo->0 and `z/lmo` would blow the reverse-mode gradient to inf (then 0*inf=nan in
        # the zeta>1e10 clamp). Forward-identical for non-degenerate lmo (e.g. arm, which never hits this).
        _tiny = 1.0e-12
        lmo_safe = jnp.where(jnp.abs(lmo) > _tiny, lmo, jnp.where(lmo >= 0.0, _tiny, -_tiny))
        zeta = z / lmo_safe
        ustar_stable = jnp.where(zeta > 1.0e10, 1.0e-10,
                                 _vonk * wnd / (lnz + _am * zeta))
        x    = _safe_sqrt(_safe_sqrt(1.0 - _bm * zeta))
        psi1 = (2.0 * jnp.log(1.0 + x)
                + jnp.log(1.0 + x * x)
                - 2.0 * jnp.arctan(x)
                + c1)
        ustar_unstable = wnd * _vonk / (lnz - psi1)
        ustar_new = jnp.where(zeta > 0.0, ustar_stable, ustar_unstable)
        ustar = jnp.where(apply, ustar_new, ustar)
    return ustar


__all__ = ["_diag_ustar", "diag_ustar"]
