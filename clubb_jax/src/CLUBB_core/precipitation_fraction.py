"""JAX port of precipitation_fraction.F90.

Sets overall precipitation fraction as well as the precipitation fraction in
each PDF component. The routine order mirrors the Fortran module:

  precip_fraction
  component_precip_frac_weighted
  component_precip_frac_specify
  precip_frac_assert_check

JAX adaptation: Fortran output and inout arguments are returned. Divisions in
unselected ``jnp.where`` branches use safe denominators so JIT/AD does not carry
NaNs from paths that the Fortran branch structure would not execute.
"""

import numpy as np

import jax.numpy as jnp
from jax import lax

from clubb_jax.src.CLUBB_core.clubb_constants import (
    cloud_frac_min,
    eps,
    iupsilon_precip_frac_rat,
)


precip_frac_calc_type = 2
_MAX_HM_IP_COMP_MEAN = 0.0025
_PRECIP_FRAC_TOL_COEF = 0.1


def precip_fraction(
    gr,
    nzt,
    ngrdcol,
    hydromet_dim,
    hydromet,
    cloud_frac,
    cloud_frac_1,
    l_mix_rat_hm,
    l_frozen_hm,
    hydromet_tol,
    cloud_frac_2,
    ice_supersat_frac,
    ice_supersat_frac_1,
    ice_supersat_frac_2,
    mixt_frac,
    clubb_params,
    err_info,
    stats,
):
    """Determine overall and per-component precipitation fractions."""
    del nzt, ngrdcol

    hydromet = jnp.asarray(hydromet, dtype=jnp.float64)
    cloud_frac = jnp.asarray(cloud_frac, dtype=jnp.float64)
    cloud_frac_1 = jnp.asarray(cloud_frac_1, dtype=jnp.float64)
    cloud_frac_2 = jnp.asarray(cloud_frac_2, dtype=jnp.float64)
    ice_supersat_frac = jnp.asarray(ice_supersat_frac, dtype=jnp.float64)
    ice_supersat_frac_1 = jnp.asarray(ice_supersat_frac_1, dtype=jnp.float64)
    ice_supersat_frac_2 = jnp.asarray(ice_supersat_frac_2, dtype=jnp.float64)
    mixt_frac = jnp.asarray(mixt_frac, dtype=jnp.float64)
    hydromet_tol = jnp.asarray(hydromet_tol, dtype=jnp.float64)
    l_mix_rat_hm = jnp.asarray(l_mix_rat_hm, dtype=bool)
    l_frozen_hm = jnp.asarray(l_frozen_hm, dtype=bool)

    any_frozen_hm = jnp.any(l_frozen_hm)
    warm_precip_frac_tol = _PRECIP_FRAC_TOL_COEF * jnp.max(cloud_frac, axis=1)
    frozen_precip_frac_tol = _PRECIP_FRAC_TOL_COEF * jnp.maximum(
        jnp.max(cloud_frac, axis=1),
        jnp.max(ice_supersat_frac, axis=1),
    )
    precip_frac_tol = jnp.maximum(
        jnp.where(any_frozen_hm, frozen_precip_frac_tol, warm_precip_frac_tol),
        cloud_frac_min,
    )
    precip_frac_tol_2d = precip_frac_tol[:, None]

    precip_frac_base = jnp.where(
        any_frozen_hm,
        jnp.maximum(cloud_frac, ice_supersat_frac),
        cloud_frac,
    )
    if int(gr.grid_dir_indx) > 0:
        precip_frac = lax.cummax(precip_frac_base, axis=1, reverse=True)
    else:
        precip_frac = lax.cummax(precip_frac_base, axis=1)

    has_hydromet = jnp.any(hydromet >= hydromet_tol[None, None, :], axis=2)
    precip_frac = jnp.where(
        has_hydromet,
        jnp.maximum(precip_frac, precip_frac_tol_2d),
        0.0,
    )

    if precip_frac_calc_type == 1:
        precip_frac_1, precip_frac_2 = component_precip_frac_weighted(
            gr,
            hydromet_dim,
            l_frozen_hm,
            hydromet_tol,
            hydromet,
            precip_frac,
            cloud_frac_1,
            cloud_frac_2,
            ice_supersat_frac_1,
            ice_supersat_frac_2,
            mixt_frac,
            precip_frac_tol,
        )
    elif precip_frac_calc_type == 2:
        clubb_params = jnp.asarray(clubb_params, dtype=jnp.float64)
        if clubb_params.ndim == 1:
            upsilon_precip_frac_rat = clubb_params[iupsilon_precip_frac_rat]
        else:
            upsilon_precip_frac_rat = clubb_params[:, iupsilon_precip_frac_rat][:, None]
        precip_frac_1, precip_frac_2 = component_precip_frac_specify(
            hydromet_dim,
            hydromet_tol,
            upsilon_precip_frac_rat,
            hydromet,
            precip_frac,
            mixt_frac,
            precip_frac_tol,
        )
    else:
        raise ValueError("Invalid option to calculate precip_frac_1 and precip_frac_2.")

    one_minus_mixt_frac = 1.0 - mixt_frac
    mixt_frac_safe = jnp.where(mixt_frac != 0.0, mixt_frac, 1.0)
    one_minus_mixt_frac_safe = jnp.where(
        one_minus_mixt_frac != 0.0,
        one_minus_mixt_frac,
        1.0,
    )

    for ivar in range(int(hydromet_dim)):
        hydromet_i = hydromet[:, :, ivar]
        hydromet_present = hydromet_i >= hydromet_tol[ivar]
        l_mix_rat_hm_i = l_mix_rat_hm[ivar]

        boost_component_1 = (
            l_mix_rat_hm_i
            & hydromet_present
            & (
                hydromet_i
                > mixt_frac * precip_frac_1 * _MAX_HM_IP_COMP_MEAN
            )
        )
        precip_frac_1_limited = jnp.maximum(
            jnp.minimum(
                hydromet_i / (mixt_frac_safe * _MAX_HM_IP_COMP_MEAN),
                1.0,
            ),
            precip_frac_tol_2d,
        )
        precip_frac_1 = jnp.where(
            boost_component_1,
            precip_frac_1_limited,
            precip_frac_1,
        )

        boost_component_2 = (
            l_mix_rat_hm_i
            & hydromet_present
            & (
                hydromet_i
                > one_minus_mixt_frac
                * precip_frac_2
                * _MAX_HM_IP_COMP_MEAN
            )
        )
        precip_frac_2_limited = jnp.maximum(
            jnp.minimum(
                hydromet_i / (one_minus_mixt_frac_safe * _MAX_HM_IP_COMP_MEAN),
                1.0,
            ),
            precip_frac_tol_2d,
        )
        precip_frac_2 = jnp.where(
            boost_component_2,
            precip_frac_2_limited,
            precip_frac_2,
        )

    precip_frac = (
        mixt_frac * precip_frac_1
        + (1.0 - mixt_frac) * precip_frac_2
    )
    precip_frac = jnp.where(
        has_hydromet,
        jnp.minimum(jnp.maximum(precip_frac, precip_frac_tol_2d), 1.0),
        precip_frac,
    )

    stats = stats.update("precip_frac_tol", precip_frac_tol)

    return err_info, precip_frac, precip_frac_1, precip_frac_2, precip_frac_tol, stats


def component_precip_frac_weighted(
    gr,
    hydromet_dim,
    l_frozen_hm,
    hydromet_tol,
    hydromet,
    precip_frac,
    cloud_frac_1,
    cloud_frac_2,
    ice_supersat_frac_1,
    ice_supersat_frac_2,
    mixt_frac,
    precip_frac_tol,
):
    """Set precipitation fraction in each component of the PDF."""
    del hydromet_dim

    hydromet = jnp.asarray(hydromet, dtype=jnp.float64)
    hydromet_tol = jnp.asarray(hydromet_tol, dtype=jnp.float64)
    precip_frac = jnp.asarray(precip_frac, dtype=jnp.float64)
    cloud_frac_1 = jnp.asarray(cloud_frac_1, dtype=jnp.float64)
    cloud_frac_2 = jnp.asarray(cloud_frac_2, dtype=jnp.float64)
    ice_supersat_frac_1 = jnp.asarray(ice_supersat_frac_1, dtype=jnp.float64)
    ice_supersat_frac_2 = jnp.asarray(ice_supersat_frac_2, dtype=jnp.float64)
    mixt_frac = jnp.asarray(mixt_frac, dtype=jnp.float64)
    precip_frac_tol = jnp.asarray(precip_frac_tol, dtype=jnp.float64)

    any_frozen_hm = jnp.any(jnp.asarray(l_frozen_hm, dtype=bool))
    one_minus_mixt_frac = 1.0 - mixt_frac
    weighted_pfrac_1_base = jnp.where(
        any_frozen_hm,
        jnp.maximum(
            mixt_frac * cloud_frac_1,
            mixt_frac * ice_supersat_frac_1,
        ),
        mixt_frac * cloud_frac_1,
    )
    weighted_pfrac_2_base = jnp.where(
        any_frozen_hm,
        jnp.maximum(
            one_minus_mixt_frac * cloud_frac_2,
            one_minus_mixt_frac * ice_supersat_frac_2,
        ),
        one_minus_mixt_frac * cloud_frac_2,
    )
    if int(gr.grid_dir_indx) > 0:
        weighted_pfrac_1 = lax.cummax(weighted_pfrac_1_base, axis=1, reverse=True)
        weighted_pfrac_2 = lax.cummax(weighted_pfrac_2_base, axis=1, reverse=True)
    else:
        weighted_pfrac_1 = lax.cummax(weighted_pfrac_1_base, axis=1)
        weighted_pfrac_2 = lax.cummax(weighted_pfrac_2_base, axis=1)

    has_hydromet = jnp.any(hydromet >= hydromet_tol[None, None, :], axis=2)
    precip_frac_tol_2d = precip_frac_tol[:, None]
    mixt_frac_safe = jnp.where(mixt_frac != 0.0, mixt_frac, 1.0)
    one_minus_mixt_frac_safe = jnp.where(
        one_minus_mixt_frac != 0.0,
        one_minus_mixt_frac,
        1.0,
    )
    weighted_pfrac_sum = weighted_pfrac_1 + weighted_pfrac_2
    weighted_pfrac_sum_safe = jnp.where(
        weighted_pfrac_sum > 0.0,
        weighted_pfrac_sum,
        1.0,
    )

    precip_frac_1 = jnp.where(
        weighted_pfrac_sum > 0.0,
        weighted_pfrac_1
        * (precip_frac / weighted_pfrac_sum_safe)
        / mixt_frac_safe,
        0.0,
    )

    precip_frac_1_limit = jnp.minimum(1.0, precip_frac / mixt_frac_safe)
    precip_frac_1 = jnp.where(
        has_hydromet & (precip_frac_1 > precip_frac_1_limit),
        precip_frac_1_limit,
        jnp.where(
            has_hydromet
            & (precip_frac_1 > 0.0)
            & (precip_frac_1 < precip_frac_tol_2d),
            precip_frac_tol_2d,
            jnp.where(has_hydromet, precip_frac_1, 0.0),
        ),
    )

    precip_frac_2 = jnp.where(
        has_hydromet,
        jnp.maximum(
            (precip_frac - mixt_frac * precip_frac_1)
            / one_minus_mixt_frac_safe,
            0.0,
        ),
        0.0,
    )

    precip_frac_1_if_2_gt_1 = (
        precip_frac - one_minus_mixt_frac
    ) / mixt_frac_safe
    precip_frac_2_if_1_gt_1 = (
        precip_frac - mixt_frac
    ) / one_minus_mixt_frac_safe
    precip_frac_2_if_1_lt_tol = precip_frac_tol_2d * (
        (precip_frac / precip_frac_tol_2d - mixt_frac)
        / one_minus_mixt_frac_safe
    )
    precip_frac_1_after_2_gt_1 = jnp.where(
        precip_frac_1_if_2_gt_1 > 1.0,
        1.0,
        jnp.where(
            (precip_frac_1_if_2_gt_1 > 0.0)
            & (precip_frac_1_if_2_gt_1 < precip_frac_tol_2d),
            precip_frac_tol_2d,
            precip_frac_1_if_2_gt_1,
        ),
    )
    precip_frac_2_after_2_gt_1 = jnp.where(
        precip_frac_1_if_2_gt_1 > 1.0,
        precip_frac_2_if_1_gt_1,
        jnp.where(
            (precip_frac_1_if_2_gt_1 > 0.0)
            & (precip_frac_1_if_2_gt_1 < precip_frac_tol_2d),
            precip_frac_2_if_1_lt_tol,
            1.0,
        ),
    )

    precip_frac_1_if_2_lt_tol = (
        precip_frac - one_minus_mixt_frac * precip_frac_tol_2d
    ) / mixt_frac_safe
    precip_frac_1_after_2_lt_tol = jnp.where(
        precip_frac_1_if_2_lt_tol > 1.0,
        1.0,
        jnp.where(
            (precip_frac_1_if_2_lt_tol > 0.0)
            & (precip_frac_1_if_2_lt_tol < precip_frac_tol_2d),
            precip_frac_tol_2d,
            precip_frac_1_if_2_lt_tol,
        ),
    )
    precip_frac_2_after_2_lt_tol = jnp.where(
        precip_frac_1_if_2_lt_tol > 1.0,
        precip_frac_2_if_1_gt_1,
        jnp.where(
            (precip_frac_1_if_2_lt_tol > 0.0)
            & (precip_frac_1_if_2_lt_tol < precip_frac_tol_2d),
            precip_frac_2_if_1_lt_tol,
            precip_frac_tol_2d,
        ),
    )

    precip_frac_2_gt_1 = has_hydromet & (precip_frac_2 > 1.0)
    precip_frac_2_lt_tol = (
        has_hydromet
        & (~precip_frac_2_gt_1)
        & (precip_frac_2 > 0.0)
        & (precip_frac_2 < precip_frac_tol_2d)
    )
    precip_frac_1 = jnp.where(
        precip_frac_2_gt_1,
        precip_frac_1_after_2_gt_1,
        jnp.where(precip_frac_2_lt_tol, precip_frac_1_after_2_lt_tol, precip_frac_1),
    )
    precip_frac_2 = jnp.where(
        precip_frac_2_gt_1,
        precip_frac_2_after_2_gt_1,
        jnp.where(precip_frac_2_lt_tol, precip_frac_2_after_2_lt_tol, precip_frac_2),
    )

    return precip_frac_1, precip_frac_2


def component_precip_frac_specify(
    hydromet_dim,
    hydromet_tol,
    upsilon_precip_frac_rat,
    hydromet,
    precip_frac,
    mixt_frac,
    precip_frac_tol,
):
    """Calculate the precipitation fraction in each PDF component."""
    del hydromet_dim

    hydromet = jnp.asarray(hydromet, dtype=jnp.float64)
    hydromet_tol = jnp.asarray(hydromet_tol, dtype=jnp.float64)
    precip_frac = jnp.asarray(precip_frac, dtype=jnp.float64)
    mixt_frac = jnp.asarray(mixt_frac, dtype=jnp.float64)
    precip_frac_tol = jnp.asarray(precip_frac_tol, dtype=jnp.float64)[:, None]
    upsilon_precip_frac_rat = jnp.asarray(
        upsilon_precip_frac_rat,
        dtype=jnp.float64,
    )

    has_hydromet = jnp.any(hydromet >= hydromet_tol[None, None, :], axis=2)
    one_minus_mixt_frac = 1.0 - mixt_frac
    mixt_frac_safe = jnp.where(mixt_frac != 0.0, mixt_frac, 1.0)
    one_minus_mixt_frac_safe = jnp.where(
        one_minus_mixt_frac != 0.0,
        one_minus_mixt_frac,
        1.0,
    )

    precip_frac_1_one = jnp.where(
        precip_frac <= mixt_frac,
        precip_frac / mixt_frac_safe,
        1.0,
    )
    precip_frac_2_one_initial = (
        precip_frac - mixt_frac
    ) / one_minus_mixt_frac_safe
    precip_frac_1_one_recalc = (
        precip_frac - one_minus_mixt_frac * precip_frac_tol
    ) / mixt_frac_safe
    precip_frac_2_one_if_1_gt_1 = (
        precip_frac - mixt_frac
    ) / one_minus_mixt_frac_safe
    precip_frac_2_one_if_1_lt_tol = precip_frac_tol * (
        (precip_frac / precip_frac_tol - mixt_frac)
        / one_minus_mixt_frac_safe
    )
    precip_frac_1_one_checked = jnp.where(
        precip_frac_1_one_recalc > 1.0,
        1.0,
        jnp.where(
            precip_frac_1_one_recalc < precip_frac_tol,
            precip_frac_tol,
            precip_frac_1_one_recalc,
        ),
    )
    precip_frac_2_one_checked = jnp.where(
        precip_frac_1_one_recalc > 1.0,
        precip_frac_2_one_if_1_gt_1,
        jnp.where(
            precip_frac_1_one_recalc < precip_frac_tol,
            precip_frac_2_one_if_1_lt_tol,
            precip_frac_tol,
        ),
    )
    precip_frac_2_one = jnp.where(
        precip_frac <= mixt_frac,
        0.0,
        jnp.where(
            (precip_frac_2_one_initial > 1.0)
            & (
                jnp.abs(precip_frac - 1.0)
                < jnp.abs(precip_frac + 1.0) / 2.0 * eps
            ),
            1.0,
            jnp.where(
                precip_frac_2_one_initial < precip_frac_tol,
                precip_frac_2_one_checked,
                precip_frac_2_one_initial,
            ),
        ),
    )
    precip_frac_1_one = jnp.where(
        precip_frac <= mixt_frac,
        precip_frac_1_one,
        jnp.where(
            (precip_frac_2_one_initial > 1.0)
            & (
                jnp.abs(precip_frac - 1.0)
                < jnp.abs(precip_frac + 1.0) / 2.0 * eps
            ),
            1.0,
            jnp.where(
                precip_frac_2_one_initial < precip_frac_tol,
                precip_frac_1_one_checked,
                precip_frac_1_one,
            ),
        ),
    )

    precip_frac_1_zero = jnp.where(
        precip_frac <= one_minus_mixt_frac,
        0.0,
        (precip_frac - one_minus_mixt_frac) / mixt_frac_safe,
    )
    precip_frac_2_zero = jnp.where(
        precip_frac <= one_minus_mixt_frac,
        precip_frac / one_minus_mixt_frac_safe,
        1.0,
    )
    precip_frac_2_zero_recalc = (
        precip_frac - mixt_frac * precip_frac_tol
    ) / one_minus_mixt_frac_safe
    precip_frac_1_zero_if_2_gt_1 = (
        (precip_frac - 1.0) + mixt_frac
    ) / mixt_frac_safe
    precip_frac_1_zero_if_2_lt_tol = (
        (precip_frac - precip_frac_tol) / mixt_frac_safe
        + precip_frac_tol
    )
    precip_frac_2_zero_checked = jnp.where(
        precip_frac_2_zero_recalc > 1.0,
        1.0,
        jnp.where(
            precip_frac_2_zero_recalc < precip_frac_tol,
            precip_frac_tol,
            precip_frac_2_zero_recalc,
        ),
    )
    precip_frac_1_zero_checked = jnp.where(
        precip_frac_2_zero_recalc > 1.0,
        precip_frac_1_zero_if_2_gt_1,
        jnp.where(
            precip_frac_2_zero_recalc < precip_frac_tol,
            precip_frac_1_zero_if_2_lt_tol,
            precip_frac_1_zero,
        ),
    )
    precip_frac_1_zero_initial = (
        precip_frac - one_minus_mixt_frac
    ) / mixt_frac_safe
    precip_frac_1_zero = jnp.where(
        precip_frac <= one_minus_mixt_frac,
        precip_frac_1_zero,
        jnp.where(
            (precip_frac_1_zero_initial > 1.0)
            & (
                jnp.abs(precip_frac - 1.0)
                < jnp.abs(precip_frac + 1.0) / 2.0 * eps
            ),
            1.0,
            jnp.where(
                precip_frac_1_zero_initial < precip_frac_tol,
                precip_frac_1_zero_checked,
                precip_frac_1_zero,
            ),
        ),
    )
    precip_frac_2_zero = jnp.where(
        precip_frac <= one_minus_mixt_frac,
        precip_frac_2_zero,
        jnp.where(
            (precip_frac_1_zero_initial > 1.0)
            & (
                jnp.abs(precip_frac - 1.0)
                < jnp.abs(precip_frac + 1.0) / 2.0 * eps
            ),
            precip_frac_2_zero,
            jnp.where(
                precip_frac_1_zero_initial < precip_frac_tol,
                precip_frac_2_zero_checked,
                precip_frac_2_zero,
            ),
        ),
    )

    precip_frac_1_general = (
        upsilon_precip_frac_rat * precip_frac / mixt_frac_safe
    )
    precip_frac_1_general = jnp.where(
        precip_frac_1_general > 1.0,
        1.0,
        jnp.where(
            precip_frac_1_general < precip_frac_tol,
            precip_frac_tol,
            precip_frac_1_general,
        ),
    )
    precip_frac_2_general = (
        precip_frac - mixt_frac * precip_frac_1_general
    ) / one_minus_mixt_frac_safe

    precip_frac_1_if_2_gt_1 = (
        precip_frac - one_minus_mixt_frac
    ) / mixt_frac_safe
    precip_frac_2_if_1_gt_1 = (
        precip_frac - mixt_frac
    ) / one_minus_mixt_frac_safe
    precip_frac_2_if_1_lt_tol = precip_frac_tol * (
        (precip_frac / precip_frac_tol - mixt_frac)
        / one_minus_mixt_frac_safe
    )
    precip_frac_1_after_2_gt_1 = jnp.where(
        precip_frac_1_if_2_gt_1 > 1.0,
        1.0,
        jnp.where(
            precip_frac_1_if_2_gt_1 < precip_frac_tol,
            precip_frac_tol,
            precip_frac_1_if_2_gt_1,
        ),
    )
    precip_frac_2_after_2_gt_1 = jnp.where(
        precip_frac_1_if_2_gt_1 > 1.0,
        precip_frac_2_if_1_gt_1,
        jnp.where(
            precip_frac_1_if_2_gt_1 < precip_frac_tol,
            precip_frac_2_if_1_lt_tol,
            1.0,
        ),
    )

    precip_frac_1_if_2_lt_tol = (
        precip_frac - one_minus_mixt_frac * precip_frac_tol
    ) / mixt_frac_safe
    precip_frac_1_after_2_lt_tol = jnp.where(
        precip_frac_1_if_2_lt_tol > 1.0,
        1.0,
        jnp.where(
            precip_frac_1_if_2_lt_tol < precip_frac_tol,
            precip_frac_tol,
            precip_frac_1_if_2_lt_tol,
        ),
    )
    precip_frac_2_after_2_lt_tol = jnp.where(
        precip_frac_1_if_2_lt_tol > 1.0,
        precip_frac_2_if_1_gt_1,
        jnp.where(
            precip_frac_1_if_2_lt_tol < precip_frac_tol,
            precip_frac_2_if_1_lt_tol,
            precip_frac_tol,
        ),
    )
    precip_frac_1_general = jnp.where(
        precip_frac_2_general > 1.0,
        precip_frac_1_after_2_gt_1,
        jnp.where(
            precip_frac_2_general < precip_frac_tol,
            precip_frac_1_after_2_lt_tol,
            precip_frac_1_general,
        ),
    )
    precip_frac_2_general = jnp.where(
        precip_frac_2_general > 1.0,
        precip_frac_2_after_2_gt_1,
        jnp.where(
            precip_frac_2_general < precip_frac_tol,
            precip_frac_2_after_2_lt_tol,
            precip_frac_2_general,
        ),
    )

    l_upsilon_one = (
        jnp.abs(upsilon_precip_frac_rat - 1.0)
        < jnp.abs(upsilon_precip_frac_rat + 1.0) / 2.0 * eps
    )
    l_upsilon_zero = (
        jnp.abs(upsilon_precip_frac_rat)
        < jnp.abs(upsilon_precip_frac_rat) / 2.0 * eps
    )
    precip_frac_1 = jnp.where(
        l_upsilon_one,
        precip_frac_1_one,
        jnp.where(l_upsilon_zero, precip_frac_1_zero, precip_frac_1_general),
    )
    precip_frac_2 = jnp.where(
        l_upsilon_one,
        precip_frac_2_one,
        jnp.where(l_upsilon_zero, precip_frac_2_zero, precip_frac_2_general),
    )

    precip_frac_1 = jnp.where(has_hydromet, precip_frac_1, 0.0)
    precip_frac_2 = jnp.where(has_hydromet, precip_frac_2, 0.0)

    return precip_frac_1, precip_frac_2


def precip_frac_assert_check(
    hydromet,
    hydromet_tol,
    mixt_frac,
    precip_frac,
    precip_frac_1,
    precip_frac_2,
    precip_frac_tol,
):
    """Assertion check for the precipitation fraction code."""
    hydromet = np.asarray(hydromet, dtype=np.float64)
    hydromet_tol = np.asarray(hydromet_tol, dtype=np.float64)
    mixt_frac = np.asarray(mixt_frac, dtype=np.float64)
    precip_frac = np.asarray(precip_frac, dtype=np.float64)
    precip_frac_1 = np.asarray(precip_frac_1, dtype=np.float64)
    precip_frac_2 = np.asarray(precip_frac_2, dtype=np.float64)
    precip_frac_tol = float(np.asarray(precip_frac_tol, dtype=np.float64))

    has_hydromet = np.any(hydromet >= hydromet_tol, axis=-1)
    cloudy_ok = (
        (precip_frac >= precip_frac_tol)
        & (precip_frac <= 1.0)
        & ~((precip_frac_1 > 0.0) & (precip_frac_1 < precip_frac_tol - eps))
        & (precip_frac_1 >= 0.0)
        & (precip_frac_1 <= 1.0)
        & ~((precip_frac_2 > 0.0) & (precip_frac_2 < precip_frac_tol - eps))
        & (precip_frac_2 >= 0.0)
        & (precip_frac_2 <= 1.0)
    )
    clear_ok = (
        (np.abs(precip_frac) <= eps)
        & (np.abs(precip_frac_1) <= eps)
        & (np.abs(precip_frac_2) <= eps)
    )
    mixture_ok = (
        precip_frac
        - (mixt_frac * precip_frac_1 + (1.0 - mixt_frac) * precip_frac_2)
    ) <= eps

    return bool(np.all(np.where(has_hydromet, cloudy_ok, clear_ok)) and np.all(mixture_ok))


__all__ = [
    "precip_frac_calc_type",
    "precip_fraction",
    "component_precip_frac_weighted",
    "component_precip_frac_specify",
    "precip_frac_assert_check",
]
