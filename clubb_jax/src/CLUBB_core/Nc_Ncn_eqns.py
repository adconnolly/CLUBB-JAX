"""JAX port of Nc_Ncn_eqns.F90.

The source routines convert between cloud droplet concentration, Nc, and
simplified cloud nuclei concentration, Ncn, using the chi/Ncn PDF parameters.

JAX adaptation: unselected `where` branches still evaluate, so divisions use
local safe denominators only where the Fortran branch would have skipped the
division.
"""

import jax.scipy.special as jsp
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.clubb_constants import (
    chi_tol,
    cloud_frac_min,
    eps,
    Ncn_tol,
    sqrt_2,
)


def Ncnm_to_Nc_in_cloud(
    mu_chi_1,
    mu_chi_2,
    mu_Ncn_1,
    mu_Ncn_2,
    sigma_chi_1,
    sigma_chi_2,
    sigma_Ncn_1,
    sigma_Ncn_2,
    sigma_Ncn_1_n,
    sigma_Ncn_2_n,
    corr_chi_Ncn_1_n,
    corr_chi_Ncn_2_n,
    mixt_frac,
    cloud_frac_1,
    cloud_frac_2,
):
    """Calculate in-cloud mean cloud droplet concentration from Ncn parameters."""
    cloud_frac = mixt_frac * cloud_frac_1 + (1.0 - mixt_frac) * cloud_frac_2

    Ncm = Ncnm_to_Ncm(
        mu_chi_1,
        mu_chi_2,
        mu_Ncn_1,
        mu_Ncn_2,
        sigma_chi_1,
        sigma_chi_2,
        sigma_Ncn_1,
        sigma_Ncn_2,
        sigma_Ncn_1_n,
        sigma_Ncn_2_n,
        corr_chi_Ncn_1_n,
        corr_chi_Ncn_2_n,
        mixt_frac,
    )

    cloud_frac_safe = jnp.where(cloud_frac > cloud_frac_min, cloud_frac, 1.0)
    return jnp.where(cloud_frac > cloud_frac_min, Ncm / cloud_frac_safe, mu_Ncn_1)


def Nc_in_cloud_to_Ncnm(
    mu_chi_1,
    mu_chi_2,
    sigma_chi_1,
    sigma_chi_2,
    mixt_frac,
    Nc_in_cloud,
    cloud_frac_1,
    cloud_frac_2,
    const_Ncnp2_on_Ncnm2,
    const_corr_chi_Ncn,
):
    """Calculate overall mean simplified cloud nuclei concentration from in-cloud Nc."""
    cloud_frac = mixt_frac * cloud_frac_1 + (1.0 - mixt_frac) * cloud_frac_2

    Ncnm = Nc_in_cloud
    Ncm = Nc_in_cloud * cloud_frac

    Ncnm_varying = Ncm_to_Ncnm(
        mu_chi_1,
        mu_chi_2,
        sigma_chi_1,
        sigma_chi_2,
        mixt_frac,
        Ncm,
        const_Ncnp2_on_Ncnm2,
        const_corr_chi_Ncn,
        Nc_in_cloud,
    )

    l_varying_Ncn = (
        (cloud_frac > cloud_frac_min)
        & (jnp.abs(const_corr_chi_Ncn * const_Ncnp2_on_Ncnm2) > eps)
    )
    return jnp.where(l_varying_Ncn, Ncnm_varying, Ncnm)


def Ncnm_to_Ncm(
    mu_chi_1,
    mu_chi_2,
    mu_Ncn_1,
    mu_Ncn_2,
    sigma_chi_1,
    sigma_chi_2,
    sigma_Ncn_1,
    sigma_Ncn_2,
    sigma_Ncn_1_n,
    sigma_Ncn_2_n,
    corr_chi_Ncn_1_n,
    corr_chi_Ncn_2_n,
    mixt_frac,
):
    """Calculate overall mean cloud droplet concentration from Ncn parameters."""
    Ncm = (
        mixt_frac
        * bivar_NL_chi_Ncn_mean(
            mu_chi_1,
            mu_Ncn_1,
            sigma_chi_1,
            sigma_Ncn_1,
            sigma_Ncn_1_n,
            corr_chi_Ncn_1_n,
        )
        + (1.0 - mixt_frac)
        * bivar_NL_chi_Ncn_mean(
            mu_chi_2,
            mu_Ncn_2,
            sigma_chi_2,
            sigma_Ncn_2,
            sigma_Ncn_2_n,
            corr_chi_Ncn_2_n,
        )
    )

    return Ncm


def Ncm_to_Ncnm(
    mu_chi_1,
    mu_chi_2,
    sigma_chi_1,
    sigma_chi_2,
    mixt_frac,
    Ncm,
    const_Ncnp2_on_Ncnm2,
    const_corr_chi_Ncn,
    Ncnm_val_denom_0,
):
    """Calculate overall mean simplified cloud nuclei concentration from overall Nc."""
    denominator = (
        mixt_frac
        * bivar_Ncnm_eqn_comp(
            mu_chi_1,
            sigma_chi_1,
            const_Ncnp2_on_Ncnm2,
            const_corr_chi_Ncn,
        )
        + (1.0 - mixt_frac)
        * bivar_Ncnm_eqn_comp(
            mu_chi_2,
            sigma_chi_2,
            const_Ncnp2_on_Ncnm2,
            const_corr_chi_Ncn,
        )
    )

    denominator_safe = jnp.where(denominator > 0.0, denominator, 1.0)
    return jnp.where(denominator > 0.0, Ncm / denominator_safe, Ncnm_val_denom_0)


def bivar_NL_chi_Ncn_mean(
    mu_chi_i,
    mu_Ncn_i,
    sigma_chi_i,
    sigma_Ncn_i,
    sigma_Ncn_i_n,
    corr_chi_Ncn_i_n,
):
    """Evaluate the per-component normal-lognormal Nc integral."""
    chi_and_Ncn_constant = (sigma_chi_i <= chi_tol) & (sigma_Ncn_i <= Ncn_tol)
    chi_constant = sigma_chi_i <= chi_tol
    Ncn_constant = sigma_Ncn_i <= Ncn_tol

    constant_value = jnp.where(mu_chi_i > 0.0, mu_Ncn_i, 0.0)

    sigma_chi_i_safe = jnp.where(sigma_chi_i > chi_tol, sigma_chi_i, 1.0)
    Ncn_constant_value = (
        mu_Ncn_i
        * 0.5
        * jsp.erfc(-(mu_chi_i / (sqrt_2 * sigma_chi_i_safe)))
    )

    both_vary_value = (
        0.5
        * mu_Ncn_i
        * jsp.erfc(
            -(1.0 / sqrt_2)
            * ((mu_chi_i / sigma_chi_i_safe) + corr_chi_Ncn_i_n * sigma_Ncn_i_n)
        )
    )

    return jnp.where(
        chi_and_Ncn_constant,
        constant_value,
        jnp.where(
            chi_constant,
            constant_value,
            jnp.where(Ncn_constant, Ncn_constant_value, both_vary_value),
        ),
    )


def bivar_Ncnm_eqn_comp(
    mu_chi_i,
    sigma_chi_i,
    const_Ncnp2_on_Ncnm2,
    const_corr_chi_Ncn,
):
    """Calculate one PDF component's denominator term in the Ncnm equation."""
    constant_value = jnp.where(mu_chi_i > 0.0, 1.0, 0.0)
    sigma_chi_i_safe = jnp.where(sigma_chi_i > chi_tol, sigma_chi_i, 1.0)

    varying_value = (
        0.5
        * jsp.erfc(
            -(1.0 / sqrt_2)
            * (
                (mu_chi_i / sigma_chi_i_safe)
                + const_corr_chi_Ncn * jnp.sqrt(const_Ncnp2_on_Ncnm2)
            )
        )
    )

    return jnp.where(sigma_chi_i <= chi_tol, constant_value, varying_value)
