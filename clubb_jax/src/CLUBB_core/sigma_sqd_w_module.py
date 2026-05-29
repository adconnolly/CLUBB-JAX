"""JAX implementation of sigma_sqd_w_module.F90:compute_sigma_sqd_w."""

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import (
    rt_tol,
    thl_tol,
    w_tol,
    w_tol_sqd,
    zero_threshold,
)
from clubb_jax.src.CLUBB_core.grid_class import zm2zt2zm_jax

_ONE_HUNDRED = 100.0


def compute_sigma_sqd_w_jax(
    gamma_Skw_fnc,      # (ngrdcol, nzm)
    wp2,                # (ngrdcol, nzm)
    thlp2,              # (ngrdcol, nzm)
    rtp2,               # (ngrdcol, nzm)
    up2,                # (ngrdcol, nzm)
    vp2,                # (ngrdcol, nzm)
    wpthlp,             # (ngrdcol, nzm)
    wprtp,              # (ngrdcol, nzm)
    upwp,               # (ngrdcol, nzm)
    vpwp,               # (ngrdcol, nzm)
    l_predict_upwp_vpwp: bool,
    gr,
):
    """PDF width parameter sigma_sqd_w (sigma_sqd_w_module.F90:compute_sigma_sqd_w).

    Returns sigma_sqd_w (ngrdcol, nzm).

    Eq: sigma_sqd_w = gamma_Skw_fnc * (1 - max(corr_wx^2 for all x))
    with vertical zm→zt→zm smoothing applied.
    """
    # Correlation denominators with numerical stability regularization
    denom_thl = jnp.sqrt(wp2 * thlp2) + _ONE_HUNDRED * w_tol * thl_tol
    denom_rtp = jnp.sqrt(wp2 * rtp2)  + _ONE_HUNDRED * w_tol * rt_tol

    # Maximum squared w-x correlation over rt and thl
    corr_thl_sqd = (wpthlp / denom_thl) ** 2
    corr_rtp_sqd = (wprtp  / denom_rtp) ** 2
    max_corr = jnp.maximum(corr_thl_sqd, corr_rtp_sqd)

    if l_predict_upwp_vpwp:
        denom_up = jnp.sqrt(up2 * wp2) + _ONE_HUNDRED * w_tol_sqd
        denom_vp = jnp.sqrt(vp2 * wp2) + _ONE_HUNDRED * w_tol_sqd
        corr_up_sqd = (upwp / denom_up) ** 2
        corr_vp_sqd = (vpwp / denom_vp) ** 2
        max_corr = jnp.maximum(max_corr, jnp.maximum(corr_up_sqd, corr_vp_sqd))

    sigma_sqd_w_tmp = gamma_Skw_fnc * (1.0 - jnp.minimum(max_corr, 1.0))

    # Smooth in vertical: zm → zt → zm with zero_threshold floor
    return zm2zt2zm_jax(sigma_sqd_w_tmp, gr, zm_min=zero_threshold)
