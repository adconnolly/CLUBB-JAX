"""JAX implementations of clip_explicit.F90 clipping routines.

Implements:
  clip_variance          - clip xp2 to [threshold_lo, (optional) threshold_hi]
  clip_skewness_core     - clip wp3 via Skewness-of-w limit (the pure clip math)
  clip_skewness          - clip_skewness_core + the wp3_cl budget (Fortran clip_skewness)
  clip_covars_denom      - clip wprtp/wpthlp/upwp/vpwp after wp2/wp3 solve
  clip_rcm               - clip cloud water rcm <= rtm

(fill_holes_wp2_from_horz_tke lives in fill_holes.py, mirroring its Fortran home fill_holes.F90.)
"""

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import (
    zero_threshold, max_mag_correlation, max_mag_correlation_flux,
)
from clubb_jax.src.CLUBB_core.advance_helper_module import smooth_heaviside_peskin
from clubb_jax.src.CLUBB_core.tracer_numpy import _asarray

_F64_EPS = jnp.finfo(jnp.float64).eps


def clip_covar(
    wpxp: jnp.ndarray,
    wp2: jnp.ndarray,
    xp2: jnp.ndarray,
    max_mag_corr: float = max_mag_correlation,
) -> jnp.ndarray:
    """clip_explicit.F90:clip_covar — covariance clipping applied after the solve.

    Per the Fortran (clip_covariance): clip_wprtp/clip_wpthlp use max_mag_correlation_flux;
    all other covariances use max_mag_correlation (the default here). Both equal 0.99.

    Clips wpxp at interior levels k=1..nzm-2 (Python 0-based) to within the Cauchy-Schwarz
    bound; boundaries (k=0, k=nzm-1) are left unchanged:
        xpyp_bound   = max_mag_corr * sqrt(wp2 * xp2)
        wpxp_clipped = clip(wpxp, -xpyp_bound, xpyp_bound)

    Args:
        wpxp, wp2, xp2: (ngrdcol, nzm)
        max_mag_corr: max magnitude of correlation (0.99 for wprtp/wpthlp)
    """
    xpyp_bound = max_mag_corr * jnp.sqrt(wp2 * xp2)   # (ngrdcol, nzm)
    wpxp_clipped = jnp.clip(wpxp, -xpyp_bound, xpyp_bound)
    # Preserve boundaries unchanged
    wpxp_clipped = wpxp_clipped.at[:, 0].set(wpxp[:, 0])
    wpxp_clipped = wpxp_clipped.at[:, -1].set(wpxp[:, -1])
    return wpxp_clipped


def clip_variance(xp2, threshold_lo, threshold_hi=None):
    """clip_explicit.F90:clip_variance — clamp xp2 to [threshold_lo, threshold_hi].

    Operates on levels k=0..nzm-2 (Python 0-based), i.e., all levels except the
    top boundary.  Bottom boundary (k=0) IS included (matches Fortran charlass fix).

    Args:
        xp2:          (ngrdcol, nzm)
        threshold_lo: (ngrdcol, nzm) or scalar — minimum value
        threshold_hi: optional scalar — maximum value (not clipped at top level either)

    Returns:
        xp2: (ngrdcol, nzm) with holes filled at levels 0..nzm-2
    """
    nzm = xp2.shape[1]
    # Build a mask for levels 0..nzm-2 (skip top level k=nzm-1)
    k_idx = jnp.arange(nzm)[None, :]          # (1, nzm)
    mask = k_idx < (nzm - 1)                  # True for k=0..nzm-2

    xp2_clipped = jnp.where(mask, jnp.maximum(threshold_lo, xp2), xp2)
    if threshold_hi is not None:
        xp2_clipped = jnp.where(mask, jnp.minimum(threshold_hi, xp2_clipped), xp2_clipped)
    return xp2_clipped


def clip_skewness_core(wp3, wp2_zt, zt, sfc_elevation, Skw_max_mag,
                           l_use_wp3_lim_with_smth_Heaviside=True):
    """clip_explicit.F90:clip_skewness_core — limit |Sk_w| = |wp3|/wp2_zt^(3/2).

    The pure clipping math (no timestep / no budget stats); the `clip_skewness`
    wrapper below adds the `wp3_cl` budget around it, mirroring the Fortran
    clip_skewness → clip_skewness_core split.

    ARM flag: l_use_wp3_lim_with_smth_Heaviside=True.

    Args:
        wp3:                (ngrdcol, nzt)
        wp2_zt:             (ngrdcol, nzt) — wp2 interpolated to zt
        zt:                 (ngrdcol, nzt) — thermodynamic level altitudes [m]
        sfc_elevation:      (ngrdcol,) — surface elevation [m]
        Skw_max_mag:        (ngrdcol,) — max skewness magnitude (default 4.5)
        l_use_wp3_lim_with_smth_Heaviside: bool (True for ARM)

    Returns:
        wp3: (ngrdcol, nzt) clipped
    """
    _WP3_MAX = 100.0          # absolute limit on |wp3|

    wp2_zt_cubed = wp2_zt ** 3   # (ngrdcol, nzt)

    if l_use_wp3_lim_with_smth_Heaviside:
        # Peskin smooth Heaviside with smth_range=0.6 (note: different from tau code's 1.0)
        zagl_thresh = (zt - sfc_elevation[:, None]) / 100.0 - 1.0
        H_zagl = smooth_heaviside_peskin(zagl_thresh, 0.6)
        skw_sq = Skw_max_mag[:, None] ** 2
        wp3_lim_sqd = wp2_zt_cubed * (
            H_zagl * skw_sq + (1.0 - H_zagl) * 0.0021 * skw_sq
        )
    else:
        # Default sharp threshold at 100 m AGL
        zagl = zt - sfc_elevation[:, None]
        skw_sq = Skw_max_mag[:, None] ** 2
        wp3_lim_sqd = jnp.where(
            zagl <= 100.0,
            0.0021 * skw_sq * wp2_zt_cubed,
            skw_sq * wp2_zt_cubed,
        )

    # Clip: if |wp3| > sqrt(wp3_lim_sqd), keep sign but set magnitude to limit
    exceed = wp3 ** 2 > wp3_lim_sqd
    wp3_clipped = jnp.where(exceed, jnp.sign(wp3) * jnp.sqrt(wp3_lim_sqd), wp3)

    # Absolute limit |wp3| <= 100
    wp3_clipped = jnp.clip(wp3_clipped, -_WP3_MAX, _WP3_MAX)
    return wp3_clipped


def clip_skewness(wp3, wp2_zt, zt, sfc_elevation, Skw_max_mag, dt,
                      l_use_wp3_lim_with_smth_Heaviside=True,
                      stats_writer=None, l_sample=False):
    """clip_explicit.F90:clip_skewness — the budget wrapper around `clip_skewness_core`:
    snapshot wp3, clip it via `clip_skewness_core`, and record the `wp3_cl` clip tendency
    `(wp3_clipped - wp3_pre)/dt` (the Fortran's begin/finalize "wp3_cl" budget). Returns the
    clipped wp3. The budget write is a no-op unless `l_sample` and a `stats_writer` are given."""
    wp3_pre = wp3
    wp3_clipped = clip_skewness_core(
        wp3, wp2_zt, zt, sfc_elevation, Skw_max_mag, l_use_wp3_lim_with_smth_Heaviside)
    if l_sample and stats_writer is not None:
        stats_writer.update("wp3_cl",
            (_asarray(wp3_clipped, dtype=np.float64) - _asarray(wp3_pre, dtype=np.float64)) / dt)
    return wp3_clipped


def clip_covars_denom(wprtp, wpthlp, upwp, vpwp, wp2, rtp2, thlp2, up2, vp2,
                          l_tke_aniso=True):
    """clip_explicit.F90:clip_covars_denom — Cauchy-Schwarz clip after wp2/wp3 solve.

    Clips wprtp/wpthlp/upwp/vpwp to stay within correlation bound [-0.99, 0.99]
    at interior levels k=1..nzm-2 (Python 0-based).

    ARM flags: l_tke_aniso=True, l_linearize_pbl_winds=False, l_predict_upwp_vpwp=True,
               sclr_dim=0.

    Args:
        wprtp, wpthlp, upwp, vpwp: (ngrdcol, nzm)
        wp2, rtp2, thlp2, up2, vp2: (ngrdcol, nzm)
        l_tke_aniso: if True, clip upwp/vpwp against up2/vp2; else against wp2

    Returns:
        wprtp, wpthlp, upwp, vpwp: clipped (ngrdcol, nzm)
    """
    # clip_wprtp / clip_wpthlp → max_mag_correlation_flux; momentum fluxes → max_mag_correlation (default).
    wprtp_new  = clip_covar(wprtp,  wp2, rtp2,  max_mag_correlation_flux)
    wpthlp_new = clip_covar(wpthlp, wp2, thlp2, max_mag_correlation_flux)

    if l_tke_aniso:
        upwp_new = clip_covar(upwp, wp2, up2)
        vpwp_new = clip_covar(vpwp, wp2, vp2)
    else:
        upwp_new = clip_covar(upwp, wp2, wp2)
        vpwp_new = clip_covar(vpwp, wp2, wp2)

    return wprtp_new, wpthlp_new, upwp_new, vpwp_new


def clip_rcm(rcm, rtm):
    """clip_explicit.F90:clip_rcm — ensure rcm <= rtm.

    Clips rcm to max(zero_threshold, rtm - eps_f64) wherever rcm > rtm.
    Applied at all levels (k=0..nzt-1).

    Args:
        rcm: (ngrdcol, nzt) — cloud water mixing ratio [kg/kg]
        rtm: (ngrdcol, nzt) — total water mixing ratio [kg/kg]

    Returns:
        rcm: (ngrdcol, nzt) clipped
    """
    rcm_clipped = jnp.where(
        rtm < rcm,
        jnp.maximum(zero_threshold, rtm - _F64_EPS),
        rcm,
    )
    return rcm_clipped
