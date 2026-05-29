"""JAX implementations of clip_explicit.F90 clipping routines.

Implements:
  clip_variance_jax          - clip xp2 to [threshold_lo, (optional) threshold_hi]
  clip_skewness_jax          - clip wp3 via Skewness-of-w limit
  clip_covars_denom_jax      - clip wprtp/wpthlp/upwp/vpwp after wp2/wp3 solve
  clip_rcm_jax               - clip cloud water rcm <= rtm
  fill_holes_wp2_from_horz_tke_jax - TKE-conserving wp2 hole-fill
"""

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import zero_threshold
from clubb_jax.src.CLUBB_core.advance_xm_wpxp_module import clip_covar_jax
from clubb_jax.src.CLUBB_core.mixing_length import _smooth_heaviside_peskin_jax

_F64_EPS = jnp.finfo(jnp.float64).eps


def clip_variance_jax(xp2, threshold_lo, threshold_hi=None):
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


def clip_skewness_jax(wp3, wp2_zt, zt, sfc_elevation, Skw_max_mag,
                      l_use_wp3_lim_with_smth_Heaviside=True):
    """clip_explicit.F90:clip_skewness_core — limit |Sk_w| = |wp3|/wp2_zt^(3/2).

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
        H_zagl = _smooth_heaviside_peskin_jax(zagl_thresh, 0.6)
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


def clip_covars_denom_jax(wprtp, wpthlp, upwp, vpwp, wp2, rtp2, thlp2, up2, vp2,
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
    wprtp_new  = clip_covar_jax(wprtp,  wp2, rtp2)
    wpthlp_new = clip_covar_jax(wpthlp, wp2, thlp2)

    if l_tke_aniso:
        upwp_new = clip_covar_jax(upwp, wp2, up2)
        vpwp_new = clip_covar_jax(vpwp, wp2, vp2)
    else:
        upwp_new = clip_covar_jax(upwp, wp2, wp2)
        vpwp_new = clip_covar_jax(vpwp, wp2, wp2)

    return wprtp_new, wpthlp_new, upwp_new, vpwp_new


def clip_rcm_jax(rcm, rtm):
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


def fill_holes_wp2_from_horz_tke_jax(wp2, up2, vp2, threshold, lower_k, upper_k):
    """fill_holes.F90:fill_holes_wp2_from_horz_tke — TKE-conserving wp2 hole-fill.

    Where wp2 < threshold AND (up2 > threshold OR vp2 > threshold), borrows TKE
    from up2 and vp2 proportionally to fill holes in wp2.

    ARM call: lower_hf_level=1 (Fortran 1-based), upper_hf_level=nzm-2 (Fortran 1-based)
              → Python lower_k=0..1? Let's accept 0-based Python range.

    Args:
        wp2, up2, vp2: (ngrdcol, nzm)
        threshold:     scalar
        lower_k, upper_k: Python 0-based inclusive index range

    Returns:
        wp2, up2, vp2: (ngrdcol, nzm) modified within [lower_k, upper_k]
    """
    nzm = wp2.shape[1]
    k_idx = jnp.arange(nzm)[None, :]   # (1, nzm)
    in_range = (k_idx >= lower_k) & (k_idx <= upper_k)

    # Operate only where a hole exists AND there is available TKE
    has_hole    = wp2 < threshold
    has_tke_up  = up2 > threshold
    has_tke_vp  = vp2 > threshold
    do_fill     = in_range & has_hole & (has_tke_up | has_tke_vp)

    missing     = threshold - wp2            # > 0 where has_hole
    up2_avail   = jnp.maximum(up2 - threshold, 0.0)
    vp2_avail   = jnp.maximum(vp2 - threshold, 0.0)
    total_avail = up2_avail + vp2_avail      # > 0 where (has_tke_up | has_tke_vp)

    # --- Case 1: not enough TKE ---
    case1 = do_fill & (missing >= total_avail)
    wp2_c1  = wp2 + total_avail
    up2_c1  = jnp.minimum(up2, threshold)
    vp2_c1  = jnp.minimum(vp2, threshold)

    # --- Case 2: enough TKE (missing < total_avail) ---
    case2 = do_fill & (missing < total_avail)

    # Fortran epsilon threshold for deciding if a component is "zero"
    eps_thr = _F64_EPS * 1000.0

    case2a = case2 & (jnp.abs(up2_avail) < eps_thr)   # no up2 avail → take all from vp2
    case2b = case2 & (~case2a) & (jnp.abs(vp2_avail) < eps_thr)  # no vp2 avail → take all from up2
    case2c = case2 & (~case2a) & (~case2b)              # both available → proportional

    ratio  = jnp.where(total_avail > 0.0, missing / total_avail, 0.0)

    # case2a: up2 unchanged, vp2 -= missing
    up2_2a = up2
    vp2_2a = vp2 - missing

    # case2b: up2 -= missing, vp2 unchanged
    up2_2b = up2 - missing
    vp2_2b = vp2

    # case2c: proportional
    up2_2c = threshold + up2_avail * (1.0 - ratio)
    vp2_2c = threshold + vp2_avail * (1.0 - ratio)

    up2_c2 = jnp.where(case2a, up2_2a, jnp.where(case2b, up2_2b, up2_2c))
    vp2_c2 = jnp.where(case2a, vp2_2a, jnp.where(case2b, vp2_2b, vp2_2c))

    # --- Assemble ---
    wp2_new = jnp.where(case1, wp2_c1, jnp.where(case2, threshold, wp2))
    up2_new = jnp.where(case1, up2_c1, jnp.where(case2, up2_c2, up2))
    vp2_new = jnp.where(case1, vp2_c1, jnp.where(case2, vp2_c2, vp2))

    return wp2_new, up2_new, vp2_new
