"""JAX port of COAMPS_microphys/slope.F — hydrometeor slope (size-distribution) factors.

Mirrors `subroutine slope(sr, srcm, ss, sg, qr, qs, qg, rho, sloper, slopes, slopeg,
pcut, lice, lgrpl, len)`. Computes the inverse-slope-derived factors for rain (sr in
metres, srcm in cm), snow (ss), and graupel (sg) from the exponential-distribution
intercepts (sloper/slopes/slopeg = pi*rho_x*N0_x*1e-8). Below the threshold `pcut`
the factor is set to the sentinel -2.0 (COAMPS convention meaning "no hydrometeor").

Faithful, tracer-transparent (jnp under a grad trace, numpy otherwise via `_xp`).
"""

from __future__ import annotations

from clubb_jax.src.CLUBB_core.tracer_numpy import _xp, _safe_sqrt

_CM2M = 100.0


def slope(qr, qs, qg, rho, sloper, slopes, slopeg, pcut, lice, lgrpl):
    """Return (sr, srcm, ss, sg). Inputs are per-level arrays (mixing ratios, density).

    sr   : rain slope factor in metres (srcm * 100)         [-]
    srcm : rain slope factor in cm                          [cm]
    ss   : snow slope factor (in cm units)                  [cm]
    sg   : graupel slope factor (in cm units)               [cm]
    """
    xp = _xp
    # ---- rain ----
    sr_raw = sloper / xp.where(qr > pcut, rho * qr, 1.0)   # safe denom on masked branch
    srcm_full = _safe_sqrt(_safe_sqrt(sr_raw))
    srcm = xp.where(qr > pcut, srcm_full, -2.0)
    sr = xp.where(qr > pcut, srcm_full * _CM2M, -2.0)

    # ---- snow / graupel ----
    if lice:
        ss_raw = slopes / xp.where(qs > pcut, rho * qs, 1.0)
        ss = xp.where(qs > pcut, _safe_sqrt(_safe_sqrt(ss_raw)) * _CM2M, -2.0)
        if lgrpl:
            sg_raw = slopeg / xp.where(qg > pcut, rho * qg, 1.0)
            sg = xp.where(qg > pcut, _safe_sqrt(_safe_sqrt(sg_raw)) * _CM2M, -2.0)
        else:
            sg = xp.full_like(ss, -2.0)
    else:
        ss = xp.full_like(qs, -2.0)
        sg = xp.full_like(qg, -2.0)
    return sr, srcm, ss, sg
