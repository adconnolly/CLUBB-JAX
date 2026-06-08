"""JAX port of gabls3_night.F90 — GABLS3-night surface layer (Monin-Obukhov landflx).

Mirrors clubb_release/src/Benchmark_cases/gabls3_night.F90: the `gabls3_night_sfclyr` surface-flux routine and
its Businger-Dyer Monin-Obukhov `landflx` scheme (plus the stability functions gm1/gh1/fm1/fh1 and the
integrated-stability helper psi_h). Used only by the gabls3_night case; prescribe_forcings.py imports
`gabls3_night_sfclyr`, mirroring the Fortran case dispatch's `use gabls3_night`.

Tracer-transparent: concrete runs keep the exact per-column scalar `landflx` loop (bit-identical); under a
jax.grad trace the surface layer routes to the vectorized differentiable mirror `landflx`.
"""

from __future__ import annotations

import math

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.tracer_numpy import _is_tracer_arg, _safe_sqrt
from clubb_jax.src.CLUBB_core.constants_clubb import ep1  # landflx: `use constants_clubb, only: ep1`
from clubb_jax.src.Benchmark_cases.sfc_flux import compute_momentum_flux

_PI = math.pi


# Businger-Dyer stability functions (gabls3_night.F90: gm1/gh1/fm1/fh1) + the integrated psi_h.
def gm1(x): return (1.0 - 15.0 * x) ** 0.25
def gh1(x): return math.sqrt(abs(1.0 - 9.0 * x)) / 0.74
def fm1(x): return (2.0 * math.log((1.0 + x) / 2.0)
                     + math.log((1.0 + x * x) / 2.0)
                     - 2.0 * math.atan(x) + _PI / 2.0)
def fh1(x): return 2.0 * math.log((1.0 + 0.74 * x) / 2.0)
def psi_h(x, xlmo): return (-5.0 * x) / xlmo


def _landflx_scalar(th, ts, qh, qs, uh, vh, h, z0):
    """Port of gabls3_night.F90:landflx for a single column.

    Returns (shf, lhf, vel, ustar) — all in natural CLUBB units.
    """
    zody = math.log(h / z0)
    vel  = math.sqrt(max(0.5, uh ** 2 + vh ** 2))
    r    = 9.81 / ts * (th * (1.0 + ep1 * qh) - ts * (1.0 + ep1 * qs)) * h / vel ** 2

    if r < 0.0:
        # Unstable: 3 explicit Businger-Dyer iterations
        xsi = 0.0
        for _ in range(3):
            xm  = gm1(xsi);  xh = gh1(xsi)
            fm  = zody - fm1(xm)
            fh  = 0.74 * (zody - fh1(xh))
            xsi = r / fh * fm ** 2
            xsi = -abs(xsi)
    else:
        # Stable: quadratic formula
        a = 4.8 ** 2 * r - 6.35
        b = (2.0 * r * 4.8 - 1.0) * zody
        c = r * zody ** 2
        disc = b * b - 4.0 * a * c
        disc = max(0.0, disc)
        xsi1 = (-b + math.sqrt(disc)) / (2.0 * a)
        xsi2 = (-b - math.sqrt(disc)) / (2.0 * a)
        xsi  = max(xsi1, xsi2)
        fm   = zody + 4.8 * xsi
        fh   = zody + 7.8 * xsi   # 1.0 * (...)

    vel   = math.sqrt(uh ** 2 + vh ** 2)
    ustar = 0.4 / fm * vel

    xsi = max(1e-5, xsi) if xsi >= 0.0 else min(-1e-5, xsi)
    xlmo = h / xsi
    denom = math.log(h / 0.25) - psi_h(h, xlmo) + psi_h(0.25, xlmo)
    shf = 0.4 * ustar * (ts - th) / denom
    lhf = 0.4 * ustar * (qs - qh) / denom
    return shf, lhf, vel, ustar


def landflx(th, ts, qh, qs, uh, vh, h, z0):
    """Differentiable, vectorized jnp mirror of ``_landflx_scalar`` (used ONLY under a jax.grad trace).

    The data-dependent unstable (r<0, 3 Businger-Dyer iterations) vs stable (r>=0, quadratic) branches are
    both computed and selected with ``jnp.where``; clip-sqrt → ``_safe_sqrt``; ``1/(2a)`` and ``1/vel`` are
    guarded so the unselected branch can't poison the gradient. Forward-identical to the scalar version on
    the taken branch (the ``(1-15*xsi)**0.25`` base stays >=1 since xsi<=0, so the fractional power is safe)."""
    zody = jnp.log(h / z0)
    vel0 = _safe_sqrt(jnp.maximum(0.5, uh ** 2 + vh ** 2))           # >= sqrt(0.5), grad-safe
    r = 9.81 / ts * (th * (1.0 + ep1 * qh) - ts * (1.0 + ep1 * qs)) * h / vel0 ** 2
    unstable = r < 0.0

    # unstable: 3 explicit Businger-Dyer iterations (xsi forced <= 0 each step)
    xsi_u = jnp.zeros_like(r)
    fm_u = zody
    fh_u = 0.74 * zody
    for _ in range(3):
        xm = (1.0 - 15.0 * xsi_u) ** 0.25                           # base = 1-15*xsi_u >= 1 (xsi_u<=0)
        xh = _safe_sqrt(jnp.abs(1.0 - 9.0 * xsi_u)) / 0.74
        fm_u = zody - (2.0 * jnp.log((1.0 + xm) / 2.0) + jnp.log((1.0 + xm * xm) / 2.0)
                       - 2.0 * jnp.arctan(xm) + jnp.pi / 2.0)
        fh_u = 0.74 * (zody - 2.0 * jnp.log((1.0 + 0.74 * xh) / 2.0))
        xsi_u = -jnp.abs(r / fh_u * fm_u ** 2)

    # stable: quadratic root
    a = 4.8 ** 2 * r - 6.35
    b = (2.0 * r * 4.8 - 1.0) * zody
    c = r * zody ** 2
    sq = _safe_sqrt(jnp.maximum(0.0, b * b - 4.0 * a * c))
    a_safe = jnp.where(jnp.abs(a) > 1.0e-30, a, 1.0e-30)            # guard 1/(2a)
    xsi_s = jnp.maximum((-b + sq) / (2.0 * a_safe), (-b - sq) / (2.0 * a_safe))
    fm_s = zody + 4.8 * xsi_s
    fh_s = zody + 7.8 * xsi_s

    xsi = jnp.where(unstable, xsi_u, xsi_s)
    fm  = jnp.where(unstable, fm_u, fm_s)
    vel = _safe_sqrt(uh ** 2 + vh ** 2)
    ustar = 0.4 / fm * vel

    xsi = jnp.where(xsi >= 0.0, jnp.maximum(1.0e-5, xsi), jnp.minimum(-1.0e-5, xsi))
    xlmo = h / xsi
    denom = jnp.log(h / 0.25) - (-5.0 * h) / xlmo + (-5.0 * 0.25) / xlmo
    shf = 0.4 * ustar * (ts - th) / denom
    lhf = 0.4 * ustar * (qs - qh) / denom
    return shf, lhf, vel, ustar


def gabls3_night_sfclyr(state: dict, time_current: float, ngrdcol: int,
                            um_bot, vm_bot, thlm_bot, rtm_bot, z_bot) -> tuple:
    """GABLS3 night surface fluxes (gabls3_night.F90:gabls3_night_sfclyr).

    Reads thlm_sfc, rtm_sfc from gabls3_night_sfc.in (time-dependent).
    Uses landflx (Monin-Obukhov) for heat/moisture fluxes and ustar.
    upwp/vpwp from sfc file if available, else from momentum flux.
    """
    z0 = 0.15
    fd = state['_forcings_data']
    sfc = fd.get('sfc') or {}
    times = sfc.get('time', np.array([0.0, 1e9]))

    ts_arr = sfc.get('thlm_sfc')
    qs_arr = sfc.get('rtm_sfc')
    ts_val = float(np.interp(time_current, times, ts_arr)) if ts_arr is not None else float(thlm_bot[0])
    qs_val = float(np.interp(time_current, times, qs_arr)) if qs_arr is not None else float(rtm_bot[0])

    # Tracer dispatch (REFACTOR B5): landflx is a Businger-Dyer MO scheme with a data-dependent r<0 branch
    # and float()s — concrete runs keep the exact per-column loop (bit-identical), the jax.grad trace uses the
    # vectorized differentiable mirror landflx.
    if not _is_tracer_arg([thlm_bot, um_bot, vm_bot, rtm_bot]):
        wpthlp_sfc = np.zeros(ngrdcol)
        wprtp_sfc  = np.zeros(ngrdcol)
        ubar       = np.zeros(ngrdcol)
        ustar      = np.zeros(ngrdcol)
        for i in range(ngrdcol):
            shf, lhf, vel, us = _landflx_scalar(
                float(thlm_bot[i]), ts_val, float(rtm_bot[i]), qs_val,
                float(um_bot[i]), float(vm_bot[i]), float(z_bot[i]), z0)
            wpthlp_sfc[i] = shf
            wprtp_sfc[i]  = lhf
            ubar[i]       = vel
            ustar[i]      = us
    else:
        wpthlp_sfc, wprtp_sfc, ubar, ustar = landflx(
            thlm_bot, ts_val, rtm_bot, qs_val, um_bot, vm_bot, jnp.asarray(z_bot), z0)

    # Momentum flux
    upwp_arr = sfc.get('upwp_sfc')
    vpwp_arr = sfc.get('vpwp_sfc')
    l_input_xpwp = upwp_arr is not None

    if l_input_xpwp:
        upwp_sfc = np.full(ngrdcol, float(np.interp(time_current, times, upwp_arr)))
        vpwp_sfc = np.full(ngrdcol, float(np.interp(time_current, times, vpwp_arr)))
    else:
        upwp_sfc, vpwp_sfc = compute_momentum_flux(um_bot, vm_bot, ubar, ustar)

    return wpthlp_sfc, wprtp_sfc, ustar, upwp_sfc, vpwp_sfc
