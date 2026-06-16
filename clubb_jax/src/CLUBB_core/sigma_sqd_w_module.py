"""JAX port of ``src/CLUBB_core/sigma_sqd_w_module.F90``."""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.Skx_module import Skx_func, compute_gamma_Skw
from clubb_jax.src.CLUBB_core.clubb_constants import (
    l_gamma_Skw,
    one,
    one_hundred,
    rt_tol,
    thl_tol,
    w_tol,
    w_tol_sqd,
    zero_threshold,
)
from clubb_jax.src.CLUBB_core.grid_class import zm2zt2zm, zt2zm
from clubb_jax.src.derived_types import Grid


@partial(
    jax.jit,
    static_argnames=(
        "nzm",
        "nzt",
        "ngrdcol",
        "l_predict_upwp_vpwp",
    ),
)
def compute_sigma_sqd_w(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr: Grid,
    wp3,
    wp2,
    thlp2,
    rtp2,
    up2,
    vp2,
    wpthlp,
    wprtp,
    upwp,
    vpwp,
    clubb_params,
    l_predict_upwp_vpwp: bool,
):
    """Compute the variable sigma_sqd_w (PDF width parameter)."""

    if nzm > 1:
        wp3_zm = zt2zm(nzm, nzt, ngrdcol, gr, wp3)
    else:
        # Skip interpolation if nzm == 1, this only occurs for testing purposes
        wp3_zm = wp3

    Skw_zm = Skx_func(
        nzm, ngrdcol, wp2, wp3_zm,
        w_tol, clubb_params,
    )

    gamma_Skw_fnc = compute_gamma_Skw(
        nzm, ngrdcol, Skw_zm, clubb_params,
        l_gamma_Skw,
    )

    #----------------------------------------------------------------
    # Compute sigma_sqd_w with new formula from Vince
    #----------------------------------------------------------------

    # Find the maximum value of <w'x'>^2 / ( <w'^2> * <x'^2> ) for all
    # variables x that are Double Gaussian PDF responder variables.  This
    # includes rt and theta-l.  When l_predict_upwp_vpwp is enabled, u and v are
    # also calculated as part of the PDF, and they are included as well.
    # Additionally, when sclr_dim > 0, passive scalars (sclr) are also included.
    max_corr_w_x_sqd = jnp.maximum(
        (
            wpthlp
            / (
                jnp.sqrt(wp2 * thlp2)
                + one_hundred * w_tol * thl_tol
            )
        ) ** 2,
        (
            wprtp
            / (
                jnp.sqrt(wp2 * rtp2)
                + one_hundred * w_tol * rt_tol
            )
        ) ** 2,
    )

    if l_predict_upwp_vpwp:
        max_corr_w_x_sqd = jnp.maximum(
            max_corr_w_x_sqd,
            jnp.maximum(
                (
                    upwp
                    / (
                        jnp.sqrt(up2 * wp2)
                        + one_hundred * w_tol_sqd
                    )
                ) ** 2,
                (
                    vpwp
                    / (
                        jnp.sqrt(vp2 * wp2)
                        + one_hundred * w_tol_sqd
                    )
                ) ** 2,
            ),
        )

    sigma_sqd_w_tmp = gamma_Skw_fnc * (one - jnp.minimum(max_corr_w_x_sqd, one))

    if nzm > 1:
        # Smooth in the vertical using interpolation.
        sigma_sqd_w = zm2zt2zm(
            nzm, nzt, ngrdcol, gr, sigma_sqd_w_tmp,
            zero_threshold,
        )
    else:
        # Skip interpolation if nzm == 1, this only occurs for testing purposes
        sigma_sqd_w = sigma_sqd_w_tmp

    return sigma_sqd_w
