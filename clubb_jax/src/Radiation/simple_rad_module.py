"""Simplified (Larson-style) radiation — pure Python port of simple_rad_module.F90.

Mirrors clubb_release/src/Radiation/simple_rad_module.F90:
  simple_rad        — the exponential-LWP longwave parameterization (`simple_rad`) with the optional
                      above-cloud-inversion correction, plus its helpers (`liq_water_path`,
                      `_linear_interp`, `_inversion_height`)
  simple_rad_bomex  — the prescribed BOMEX radiative-cooling profile

(`simple_rad_lba` lives in its own Radiation/simple_rad_lba.py.) radiation.py's dispatch
(radiation_module.F90) `use`s this module for the simplified-radiation path. Tracer-transparent so the
rcm→radht→thlm_forcing chain stays on the whole-driver autodiff graph.
"""

import os

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.tracer_numpy import _xp, _is_tracer_arg, _safe_pow
from clubb_jax.src.CLUBB_core.constants_clubb import Cp as _CP  # mirror simple_rad_module.F90 `use constants_clubb, only: Cp`

_EPS = np.finfo(np.float64).eps
_LS_DIV = 3.75e-6

# ── LBA prescribed radiative-cooling table (simple_rad_lba / simple_rad_lba_init) ──
_LBA_NZRAD = 33
_LBA_NTIMES = 36
_LBA_DT = 600.0   # seconds between the prescribed radiation times (simple_rad_module.F90)


def simple_rad_bomex(zt: np.ndarray):
    """Compute BOMEX simplified radiation heating rate."""
    return np.where(
        zt < 1500.0,
        -2.315e-5,
        np.where(
            zt < 2500.0,
            -2.315e-5 + 2.315e-5 * (zt - 1500.0) / 1000.0,
            0.0,
        ),
    )


def liq_water_path(ngrdcol: int, nzm: int, nzt: int,
                    rho: np.ndarray, rcm: np.ndarray,
                    invrs_dzt: np.ndarray) -> np.ndarray:
    """Compute liquid water path on momentum levels (cumulative from the top down).

    Tracer-transparent (REFACTOR B5): vectorized reverse-cumsum that sums in the SAME top-down order as the
    original in-place loop (`lwp[k] = lwp[k+1] + contrib[k]`), so it is bit-identical for concrete runs while
    routing through jnp under a jax.grad trace (the loop's `lwp[:,k] = <tracer>` would otherwise sever)."""
    contrib = rcm * rho / invrs_dzt                                   # (ngrdcol, nzt)
    rev_cumsum = _xp.flip(_xp.cumsum(_xp.flip(contrib, axis=1), axis=1), axis=1)  # lwp[:, :nzt], top-down order
    zeros_top = _xp.zeros((ngrdcol, 1), dtype=np.float64)             # lwp[:, nzm-1] = 0 (top boundary)
    return _xp.concatenate([rev_cumsum, zeros_top], axis=1)           # (ngrdcol, nzm)


def _linear_interp(x: float, x_high: float, x_low: float,
                   y_high: float, y_low: float) -> float:
    denom = x_high - x_low
    if abs(denom) < _EPS:
        return 0.5 * (y_high + y_low)
    return y_low + (x - x_low) * (y_high - y_low) / denom


def _inversion_height(rtm, zt, nzt, thresh=8.0e-3):
    """Differentiable jnp mirror of the inversion-height loop in `simple_rad` (used only under a
    jax.grad trace). k_iso = first level (from the bottom) with rtm <= thresh; z_i is rtm=thresh linearly
    interpolated between k_iso-1 and k_iso. Forward-identical to the loop: k_iso==0 -> 0; all-above -> use the
    top two levels; the |denom|<eps fallback -> midpoint. The integer index is data-dependent (piecewise
    constant), the interpolated VALUE carries the gradient w.r.t. rtm."""
    above = rtm > thresh                                          # (ngrdcol, nzt)
    k_iso = jnp.sum(jnp.cumprod(above.astype(rtm.dtype), axis=1), axis=1).astype(jnp.int32)  # leading-True count
    k = jnp.clip(k_iso, 1, nzt - 1)                               # gatherable: k-1>=0 and k<nzt
    km1 = k - 1
    r_hi = jnp.take_along_axis(rtm, k[:, None], axis=1)[:, 0]
    r_lo = jnp.take_along_axis(rtm, km1[:, None], axis=1)[:, 0]
    z_hi = jnp.take_along_axis(zt, k[:, None], axis=1)[:, 0]
    z_lo = jnp.take_along_axis(zt, km1[:, None], axis=1)[:, 0]
    denom = r_hi - r_lo
    small = jnp.abs(denom) < _EPS
    denom_safe = jnp.where(small, 1.0, denom)                     # avoid 0/0 in the masked grad
    z_interp = jnp.where(small, 0.5 * (z_hi + z_lo),
                         z_lo + (thresh - r_lo) * (z_hi - z_lo) / denom_safe)
    return jnp.where(k_iso == 0, 0.0, z_interp)                   # k_iso==0 -> z_i=0


def simple_rad(
    gr,
    ngrdcol: int,
    rho: np.ndarray,
    rho_zm: np.ndarray,
    rtm: np.ndarray,
    rcm: np.ndarray,
    exner: np.ndarray,
    f0: float,
    f1: float,
    kappa: float,
    l_rad_above_cloud: bool,
    l_sample: bool,
    sw=None,
):
    """Port of simple_rad from simple_rad_module.F90."""
    nzm = gr.zm.shape[1]
    nzt = gr.zt.shape[1]

    lwp = liq_water_path(ngrdcol, nzm, nzt, rho, rcm, gr.invrs_dzt)

    if f1 > _EPS:
        frad_lw = f0 * _xp.exp(-kappa * lwp) + f1 * _xp.exp(-kappa * (lwp[:, 0:1] - lwp))
    else:
        frad_lw = f0 * _xp.exp(-kappa * lwp)

    if l_rad_above_cloud:
        # Above-cloud LW correction. Block-level tracer dispatch (REFACTOR B5): the inversion-height search is
        # a data-dependent threshold-crossing loop on rtm and the correction uses boolean-mask in-place writes
        # + fractional powers — so concrete runs keep the EXACT original (bit-identical) while a jax.grad trace
        # takes a jnp mirror (jnp inversion finder, mask-MULTIPLY instead of boolean-index, _safe_pow for the
        # dz**(1/3)/(4/3) cloud-top inf-grad).
        if not _is_tracer_arg([rtm, frad_lw]):
            z_i = np.zeros(ngrdcol, dtype=np.float64)
            for i in range(ngrdcol):
                k_iso = 0
                while k_iso < nzt and rtm[i, k_iso] > 8.0e-3:
                    k_iso += 1
                if k_iso == 0 or k_iso > nzt:
                    z_i[i] = 0.0
                    continue
                if k_iso == nzt:
                    k_iso = nzt - 1
                z_i[i] = _linear_interp(
                    8.0e-3, rtm[i, k_iso], rtm[i, k_iso - 1],
                    gr.zt[i, k_iso], gr.zt[i, k_iso - 1],
                )
            dz = gr.zm - z_i[:, np.newaxis]
            heaviside = np.where(dz < -_EPS, 0.0, np.where(dz > _EPS, 1.0, 0.5))
            pos = heaviside > 0.0
            if np.any(pos):
                dz_pos = np.maximum(dz[pos], 0.0)
                z_i_broad = np.broadcast_to(z_i[:, np.newaxis], (ngrdcol, nzm))[pos]
                frad_lw[pos] += (
                    rho_zm[pos] * _CP * _LS_DIV * heaviside[pos]
                    * (0.25 * (dz_pos ** (4.0 / 3.0)) + z_i_broad * (dz_pos ** (1.0 / 3.0)))
                )
            if l_sample and sw is not None:
                sw.update("z_inversion", z_i)
        else:
            z_i = _inversion_height(rtm, gr.zt, nzt)
            dz = jnp.asarray(gr.zm) - z_i[:, None]
            heaviside = jnp.where(dz < -_EPS, 0.0, jnp.where(dz > _EPS, 1.0, 0.5))
            dz_pos = jnp.maximum(dz, 0.0)
            correction = (
                jnp.asarray(rho_zm) * _CP * _LS_DIV * heaviside
                * (0.25 * _safe_pow(dz_pos, 4.0 / 3.0) + z_i[:, None] * _safe_pow(dz_pos, 1.0 / 3.0))
            )
            frad_lw = frad_lw + correction

    radht_lw = (
        (1.0 / exner) * (-1.0 / (_CP * rho))
        * (frad_lw[:, 1:] - frad_lw[:, :-1]) * gr.invrs_dzt
    )

    return frad_lw, radht_lw


# ---------------------------------------------------------------------------
# LBA prescribed radiative-cooling scheme (simple_rad_lba / simple_rad_lba_init)
# ---------------------------------------------------------------------------
#
# The LBA deep-convection case uses a prescribed radiative-heating-rate table `lba_krad` (33 altitudes × 36
# times, every 600 s) read from `lba_forcings/lba_heights.dat` + `lba_forcings/lba_rad.dat`. Each model step
# the table is linearly interpolated in time, then vertically interpolated (zero-extrapolation) to the
# thermodynamic grid. Self-contained + differentiable; the `rad_scheme='lba'` case is otherwise blocked by
# interactive SILHS, but the radiation scheme itself is faithful and tested (`tests/test_simple_rad_lba.py`).

def simple_rad_lba_init(forcings_dir):
    """Read lba_heights.dat (33,) and lba_rad.dat (33 levels × 36 times) — simple_rad_lba_init.

    file_read_2d fills `lba_krad(level, time)` level-major (each level's 36 time values in sequence), so the
    flat token stream reshapes to (33, 36).
    """
    zrad = np.fromstring(open(os.path.join(forcings_dir, "lba_heights.dat")).read().replace(",", " "), sep=" ")
    krad = np.fromstring(open(os.path.join(forcings_dir, "lba_rad.dat")).read().replace(",", " "), sep=" ")
    assert zrad.size == _LBA_NZRAD, f"lba_heights.dat: expected {_LBA_NZRAD}, got {zrad.size}"
    assert krad.size == _LBA_NZRAD * _LBA_NTIMES, f"lba_rad.dat: expected {_LBA_NZRAD*_LBA_NTIMES}, got {krad.size}"
    return zrad, krad.reshape(_LBA_NZRAD, _LBA_NTIMES)


def lba_radhtz(time, lba_krad):
    """Time-interpolate the radiative-heating column radhtz(33) for elapsed `time` [s] (simple_rad_lba time loop).

    time <= 600 -> column 0; time >= 36*600 -> column 35; else linear between the bracketing 600-s columns
    (Fortran 1-based i1=floor(time/600): radhtz = a*krad[:,i2] + (1-a)*krad[:,i1], a=(time-600 i1)/600).
    """
    krad = jnp.asarray(lba_krad)
    if time <= _LBA_DT:
        return krad[:, 0]
    if time >= _LBA_NTIMES * _LBA_DT:
        return krad[:, _LBA_NTIMES - 1]
    i1 = int(time // _LBA_DT)                      # 1-based bracketing index (matches the Fortran loop)
    a = (time - _LBA_DT * i1) / _LBA_DT
    return a * krad[:, i1] + (1.0 - a) * krad[:, i1 - 1]   # 0-based: i2=i1, i1=i1-1


def simple_rad_lba(time, lba_zrad, lba_krad, zt):
    """Radiative heating rate on the thermodynamic grid (simple_rad_module.F90:simple_rad_lba).

    Args:
        time:     elapsed time since start [s] (= time_current - time_initial).
        lba_zrad: source altitudes (33,).
        lba_krad: heating-rate table (33, 36) [K/s].
        zt:       thermodynamic-grid heights (nzt,) for one column.
    Returns:
        radht (nzt,) [K/s].
    """
    # simple_rad_module.F90:simple_rad_lba calls interpolation.F90:zlinterp_fnc for the vertical remap.
    from clubb_jax.src.CLUBB_core.interpolation import zlinterp_fnc
    radhtz = np.asarray(lba_radhtz(time, lba_krad))
    return zlinterp_fnc(np.asarray(zt), np.asarray(lba_zrad), radhtz)


__all__ = ["simple_rad_bomex", "simple_rad",
           "simple_rad_lba_init", "lba_radhtz", "simple_rad_lba"]
