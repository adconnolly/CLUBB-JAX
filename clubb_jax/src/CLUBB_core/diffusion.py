"""JAX port of `src/CLUBB_core/diffusion.F90`.

Module diffusion computes the eddy diffusion terms for all of the
time-tendency (prognostic) equations in the CLUBB parameterization.  Most of
the eddy diffusion terms are solved for completely implicitly, and therefore
become part of the left-hand side of their respective equations.  However, wp2
and wp3 have an option to use a Crank-Nicholson eddy diffusion scheme, which
has both implicit and explicit components.

Function diffusion_zt_lhs handles the eddy diffusion terms for the variables
located at thermodynamic grid levels.  These variables are: wp3 and all
hydrometeor species.  The variables um and vm also use the Crank-Nicholson
eddy-diffusion scheme for their turbulent advection term.

Function diffusion_zm_lhs handles the eddy diffusion terms for the variables
located at momentum grid levels.  The variables are: wprtp, wpthlp, wp2, rtp2,
thlp2, rtpthlp, up2, vp2, wpsclrp, sclrprtp, sclrpthlp, and sclrp2.
"""

from __future__ import annotations

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.clubb_constants import l_upwind_Kh_dp_term
from clubb_jax.src.CLUBB_core.grid_class import ddzm, ddzt


@partial(jax.jit, static_argnames=("nzm", "nzt", "ngrdcol"))
def diffusion_zt_lhs(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr,
    K_zm,
    K_zt,
    nu,
    invrs_rho_ds_ztzxt,
    rho_ds_zm,
):
    """Vertical eddy diffusion of var_zt: implicit portion of the code.

    The variable "var_zt" stands for a variable that is located at
    thermodynamic grid levels.

    The d(var_zt)/dt equation contains an eddy diffusion term:

      + d [ ( K_zm + nu ) * d(var_zt)/dz ] / dz.

    This term is usually solved for completely implicitly, such that:

      + d [ ( K_zm + nu ) * d( var_zt(t+1) )/dz ] / dz.

    However, when a Crank-Nicholson scheme is used, the eddy diffusion term has
    both implicit and explicit components.  This function computes only the
    general implicit form.  For a Crank-Nicholson scheme, the left-hand side
    result of this function will have to be multiplied by 1/2.

    Note: When the implicit term is brought over to the left-hand side, the
    sign is reversed and the leading "+" in front of the term is changed to a
    "-".

    The values of var_zt are found on the thermodynamic levels, while the
    values of K_zm are found on the momentum levels.  The derivatives d/dz of
    var_zt are taken over the intermediate momentum levels.  At the
    intermediate momentum levels, d(var_zt)/dz is multiplied by (K_zm + nu).
    Then the derivative of the whole mathematical expression is taken over the
    central thermodynamic level.

      --var_zt------------------------------------------------- t(k+1)

      ==========d(var_zt)/dz==(K_zm+nu)======================== m(k+1)

      --var_zt-------------------d[(K_zm+nu)*d(var_zt)/dz]/dz-- t(k)

      ==========d(var_zt)/dz==(K_zm+nu)======================== m(k)

      --var_zt------------------------------------------------- t(k-1)

    Boundary Conditions:

    This function is set up to use zero-flux boundary conditions at both the
    lower boundary level and the upper boundary level.  Many equations in the
    model use fixed-point boundary conditions instead; the parent routine then
    ignores or overwrites the boundary rows.

    JAX adaptation: the Fortran out-argument `lhs` is returned as an array with
    diagonal order [superdiagonal, main diagonal, subdiagonal].  Fortran 1-based
    grid indices are represented by 0-based Python slices.
    """
    K_zm_nu = K_zm + nu[:, None]

    if l_upwind_Kh_dp_term:
        # calculate the dKh_zt/dz
        rho_K_zm_nu = rho_ds_zm * K_zm_nu
        ddzm_rho_K_zm_nu = ddzm(nzm, nzt, ngrdcol, gr, rho_K_zm_nu)
        drhoKdz_zt = -invrs_rho_ds_ztzxt * ddzm_rho_K_zm_nu

        # extra terms with upwind scheme
        gd = gr.grid_dir
        zero = jnp.zeros_like(drhoKdz_zt)
        min_drho = jnp.minimum(gd * drhoKdz_zt, zero)
        max_drho = jnp.maximum(gd * drhoKdz_zt, zero)

        # k = 1 (bottom level); lower boundary level
        super_upwind_bot = gd * min_drho[:, :1] * gr.invrs_dzm[:, 1:2]
        main_upwind_bot = -gd * min_drho[:, :1] * gr.invrs_dzm[:, 1:2]
        sub_upwind_bot = jnp.zeros((ngrdcol, 1), dtype=jnp.float64)

        # Most of the interior model; normal conditions.
        super_upwind_int = gd * min_drho[:, 1:-1] * gr.invrs_dzm[:, 2:-1]
        main_upwind_int = (
            -gd * min_drho[:, 1:-1] * gr.invrs_dzm[:, 2:-1]
            + gd * max_drho[:, 1:-1] * gr.invrs_dzm[:, 1:-2]
        )
        sub_upwind_int = -gd * max_drho[:, 1:-1] * gr.invrs_dzm[:, 1:-2]

        # k = nzt (top level); upper boundary level.
        # Only relevant if zero-flux boundary conditions are used.
        super_upwind_top = jnp.zeros((ngrdcol, 1), dtype=jnp.float64)
        main_upwind_top = gd * max_drho[:, -1:] * gr.invrs_dzm[:, nzm - 2:nzm - 1]
        sub_upwind_top = -gd * max_drho[:, -1:] * gr.invrs_dzm[:, nzm - 2:nzm - 1]

        lhs_upwind = jnp.stack(
            [
                jnp.concatenate([super_upwind_bot, super_upwind_int, super_upwind_top], axis=1),
                jnp.concatenate([main_upwind_bot, main_upwind_int, main_upwind_top], axis=1),
                jnp.concatenate([sub_upwind_bot, sub_upwind_int, sub_upwind_top], axis=1),
            ],
            axis=0,
        )

        K_zt_nu = K_zt + nu[:, None]

        # k = 1 (bottom level); lower boundary level.
        # Only relevant if zero-flux boundary conditions are used.
        # These k=1 lines currently do not have any effect on model results.
        # This k=1 level of this "lhs" array is not fed into the final LHS
        # matrix that will be used to solve for the next timestep.
        common_bot = gr.invrs_dzt[:, :1] * K_zt_nu[:, :1] * gr.invrs_dzm[:, 1:2]
        super_bot = -common_bot + lhs_upwind[0, :, :1]
        main_bot = common_bot + lhs_upwind[1, :, :1]
        sub_bot = lhs_upwind[2, :, :1]

        # Most of the interior model; normal conditions.
        common_int = gr.invrs_dzt[:, 1:-1] * K_zt_nu[:, 1:-1]
        super_int = (
            -common_int * gr.invrs_dzm[:, 2:-1]
            + lhs_upwind[0, :, 1:-1]
        )
        main_int = (
            common_int * (gr.invrs_dzm[:, 2:-1] + gr.invrs_dzm[:, 1:-2])
            + lhs_upwind[1, :, 1:-1]
        )
        sub_int = (
            -common_int * gr.invrs_dzm[:, 1:-2]
            + lhs_upwind[2, :, 1:-1]
        )

        # k = nzt (top level); upper boundary level.
        # Only relevant if zero-flux boundary conditions are used.
        common_top = gr.invrs_dzt[:, -1:] * K_zt_nu[:, -1:] * gr.invrs_dzm[:, nzm - 2:nzm - 1]
        super_top = lhs_upwind[0, :, -1:]
        main_top = common_top + lhs_upwind[1, :, -1:]
        sub_top = -common_top + lhs_upwind[2, :, -1:]
    else:
        # k = 1 (bottom level); lower boundary level.
        # Only relevant if zero-flux boundary conditions are used.
        # These k=1 lines currently do not have any effect on model results.
        # This k=1 level of this "lhs" array is not fed into the final LHS
        # matrix that will be used to solve for the next timestep.
        common_bot = (
            gr.invrs_dzt[:, :1] * invrs_rho_ds_ztzxt[:, :1]
            * K_zm_nu[:, 1:2] * rho_ds_zm[:, 1:2] * gr.invrs_dzm[:, 1:2]
        )
        super_bot = -common_bot
        main_bot = common_bot
        sub_bot = jnp.zeros((ngrdcol, 1), dtype=jnp.float64)

        # Most of the interior model; normal conditions.
        scale_int = gr.invrs_dzt[:, 1:-1] * invrs_rho_ds_ztzxt[:, 1:-1]

        # Thermodynamic superdiagonal: [ x var_zt(k+1,<t+1>) ]
        super_int = (
            -scale_int * K_zm_nu[:, 2:-1]
            * rho_ds_zm[:, 2:-1] * gr.invrs_dzm[:, 2:-1]
        )

        # Thermodynamic subdiagonal: [ x var_zt(k-1,<t+1>) ]
        sub_int = (
            -scale_int * K_zm_nu[:, 1:-2]
            * rho_ds_zm[:, 1:-2] * gr.invrs_dzm[:, 1:-2]
        )

        # Thermodynamic main diagonal: [ x var_zt(k,<t+1>) ]
        main_int = -(super_int + sub_int)

        # k = nzt (top level); upper boundary level.
        # Only relevant if zero-flux boundary conditions are used.
        common_top = (
            gr.invrs_dzt[:, -1:] * invrs_rho_ds_ztzxt[:, -1:]
            * K_zm_nu[:, nzm - 2:nzm - 1]
            * rho_ds_zm[:, nzm - 2:nzm - 1]
            * gr.invrs_dzm[:, nzm - 2:nzm - 1]
        )
        super_top = jnp.zeros((ngrdcol, 1), dtype=jnp.float64)
        main_top = common_top
        sub_top = -common_top

    superdiag = jnp.concatenate([super_bot, super_int, super_top], axis=1)
    maindiag = jnp.concatenate([main_bot, main_int, main_top], axis=1)
    subdiag = jnp.concatenate([sub_bot, sub_int, sub_top], axis=1)
    return jnp.stack([superdiag, maindiag, subdiag], axis=0)


@partial(jax.jit, static_argnames=("nzm", "nzt", "ngrdcol"))
def diffusion_zm_lhs(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    gr,
    K_zt,
    K_zm,
    nu,
    invrs_rho_ds_zm,
    rho_ds_zt,
):
    """Vertical eddy diffusion of var_zm: implicit portion of the code.

    The variable "var_zm" stands for a variable that is located at momentum
    grid levels.

    The d(var_zm)/dt equation contains an eddy diffusion term:

      + d [ ( K_zt + nu ) * d(var_zm)/dz ] / dz.

    This term is usually solved for completely implicitly, such that:

      + d [ ( K_zt + nu ) * d( var_zm(t+1) )/dz ] / dz.

    However, when a Crank-Nicholson scheme is used, the eddy diffusion term has
    both implicit and explicit components.  This function computes only the
    general implicit form.  For a Crank-Nicholson scheme, the left-hand side
    result of this function will have to be multiplied by 1/2.

    Note: When the implicit term is brought over to the left-hand side, the
    sign is reversed and the leading "+" in front of the term is changed to a
    "-".

    The values of var_zm are found on the momentum levels, while the values of
    K_zt are found on the thermodynamic levels.  The derivatives d/dz of
    var_zm are taken over the intermediate thermodynamic levels.  At the
    intermediate thermodynamic levels, d(var_zm)/dz is multiplied by
    (K_zt + nu).  Then the derivative of the whole mathematical expression is
    taken over the central momentum level.

      ==var_zm================================================= m(k+1)

      ----------d(var_zm)/dz--(K_zt+nu)------------------------ t(k)

      ==var_zm===================d[(K_zt+nu)*d(var_zm)/dz]/dz== m(k)

      ----------d(var_zm)/dz--(K_zt+nu)------------------------ t(k-1)

      ==var_zm================================================= m(k-1)

    Boundary Conditions:

    This function is set up to use zero-flux boundary conditions at both the
    lower boundary level and the upper boundary level.  Many equations in the
    model use fixed-point boundary conditions instead; the parent routine then
    ignores or overwrites the boundary rows.

    JAX adaptation: the Fortran out-argument `lhs` is returned as an array with
    diagonal order [superdiagonal, main diagonal, subdiagonal].  Fortran 1-based
    grid indices are represented by 0-based Python slices.
    """
    K_zt_nu = K_zt + nu[:, None]

    if l_upwind_Kh_dp_term:
        # calculate the dKh_zm/dz
        rho_K_zt_nu = rho_ds_zt * K_zt_nu
        ddzt_rho_K_zt_nu = ddzt(nzm, nzt, ngrdcol, gr, rho_K_zt_nu)
        drhoKdz_zm = -invrs_rho_ds_zm * ddzt_rho_K_zt_nu

        # extra terms with upwind scheme
        gd = gr.grid_dir
        zero = jnp.zeros_like(drhoKdz_zm)
        min_drho = jnp.minimum(gd * drhoKdz_zm, zero)
        max_drho = jnp.maximum(gd * drhoKdz_zm, zero)

        # k = 1 (bottom level); lower boundary level
        super_upwind_bot = gd * min_drho[:, :1] * gr.invrs_dzt[:, :1]
        main_upwind_bot = -gd * min_drho[:, :1] * gr.invrs_dzt[:, :1]
        sub_upwind_bot = jnp.zeros((ngrdcol, 1), dtype=jnp.float64)

        # Most of the interior model; normal conditions.
        super_upwind_int = gd * min_drho[:, 1:-1] * gr.invrs_dzt[:, 1:]
        main_upwind_int = (
            -gd * min_drho[:, 1:-1] * gr.invrs_dzt[:, 1:]
            + gd * max_drho[:, 1:-1] * gr.invrs_dzt[:, :-1]
        )
        sub_upwind_int = -gd * max_drho[:, 1:-1] * gr.invrs_dzt[:, :-1]

        # k = nzm (top level); upper boundary level.
        # Only relevant if zero-flux boundary conditions are used.
        super_upwind_top = jnp.zeros((ngrdcol, 1), dtype=jnp.float64)
        main_upwind_top = gd * max_drho[:, -1:] * gr.invrs_dzt[:, -1:]
        sub_upwind_top = -gd * max_drho[:, -1:] * gr.invrs_dzt[:, -1:]

        lhs_upwind = jnp.stack(
            [
                jnp.concatenate([super_upwind_bot, super_upwind_int, super_upwind_top], axis=1),
                jnp.concatenate([main_upwind_bot, main_upwind_int, main_upwind_top], axis=1),
                jnp.concatenate([sub_upwind_bot, sub_upwind_int, sub_upwind_top], axis=1),
            ],
            axis=0,
        )

        K_zm_nu = K_zm + nu[:, None]

        # k = 1; lower boundary level at surface.
        # Only relevant if zero-flux boundary conditions are used.
        # These k=1 lines currently do not have any effect on model results.
        # This k=1 level of this "lhs" array is not fed into the final LHS
        # matrix that will be used to solve for the next timestep.
        common_bot = gr.invrs_dzm[:, :1] * K_zm_nu[:, :1] * gr.invrs_dzt[:, :1]
        super_bot = -common_bot + lhs_upwind[0, :, :1]
        main_bot = common_bot + lhs_upwind[1, :, :1]
        sub_bot = lhs_upwind[2, :, :1]

        # Most of the interior model; normal conditions.
        common_int = gr.invrs_dzm[:, 1:-1] * K_zm_nu[:, 1:-1]
        super_int = (
            -common_int * gr.invrs_dzt[:, 1:]
            + lhs_upwind[0, :, 1:-1]
        )
        main_int = (
            common_int * (gr.invrs_dzt[:, 1:] + gr.invrs_dzt[:, :-1])
            + lhs_upwind[1, :, 1:-1]
        )
        sub_int = (
            -common_int * gr.invrs_dzt[:, :-1]
            + lhs_upwind[2, :, 1:-1]
        )

        # k = nzm (top level); upper boundary level.
        # Only relevant if zero-flux boundary conditions are used.
        common_top = gr.invrs_dzm[:, -1:] * K_zm_nu[:, -1:] * gr.invrs_dzt[:, -1:]
        super_top = lhs_upwind[0, :, -1:]
        main_top = common_top + lhs_upwind[1, :, -1:]
        sub_top = -common_top + lhs_upwind[2, :, -1:]
    else:
        # k = 1; lower boundary level at surface.
        # Only relevant if zero-flux boundary conditions are used.
        # These k=1 lines currently do not have any effect on model results.
        # This k=1 level of this "lhs" array is not fed into the final LHS
        # matrix that will be used to solve for the next timestep.
        common_bot = (
            gr.invrs_dzm[:, :1] * invrs_rho_ds_zm[:, :1]
            * K_zt_nu[:, :1] * rho_ds_zt[:, :1] * gr.invrs_dzt[:, :1]
        )
        super_bot = -common_bot
        main_bot = common_bot
        sub_bot = jnp.zeros((ngrdcol, 1), dtype=jnp.float64)

        # Most of the interior model; normal conditions.
        scale_int = gr.invrs_dzm[:, 1:-1] * invrs_rho_ds_zm[:, 1:-1]

        # Momentum superdiagonal: [ x var_zm(k+1,<t+1>) ]
        super_int = (
            -scale_int * K_zt_nu[:, 1:]
            * rho_ds_zt[:, 1:] * gr.invrs_dzt[:, 1:]
        )

        # Momentum subdiagonal: [ x var_zm(k-1,<t+1>) ]
        sub_int = (
            -scale_int * K_zt_nu[:, :-1]
            * rho_ds_zt[:, :-1] * gr.invrs_dzt[:, :-1]
        )

        # Momentum main diagonal: [ x var_zm(k,<t+1>) ]
        main_int = -(super_int + sub_int)

        # k = nzm (top level); upper boundary level.
        # Only relevant if zero-flux boundary conditions are used.
        common_top = (
            gr.invrs_dzm[:, -1:] * invrs_rho_ds_zm[:, -1:]
            * K_zt_nu[:, -1:] * rho_ds_zt[:, -1:] * gr.invrs_dzt[:, -1:]
        )
        super_top = jnp.zeros((ngrdcol, 1), dtype=jnp.float64)
        main_top = common_top
        sub_top = -common_top

    superdiag = jnp.concatenate([super_bot, super_int, super_top], axis=1)
    maindiag = jnp.concatenate([main_bot, main_int, main_top], axis=1)
    subdiag = jnp.concatenate([sub_bot, sub_int, sub_top], axis=1)
    return jnp.stack([superdiag, maindiag, subdiag], axis=0)


__all__ = [
    "diffusion_zt_lhs",
    "diffusion_zm_lhs",
]
