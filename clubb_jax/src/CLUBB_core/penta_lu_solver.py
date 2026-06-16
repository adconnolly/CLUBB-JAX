"""JAX port of `src/CLUBB_core/penta_lu_solver.F90`."""

from __future__ import annotations

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("ndim", "ngrdcol"))
def penta_lu_solve_single_rhs_multiple_lhs(ndim: int, ngrdcol: int, lhs, rhs):
    """Written for single RHS and multiple LHS."""
    del ngrdcol

    # LHS is stored in band diagonal form:
    #   lhs[0, i, k] = Fortran lhs(-2,i,k), second superdiagonal
    #   lhs[1, i, k] = Fortran lhs(-1,i,k), first superdiagonal
    #   lhs[2, i, k] = Fortran lhs( 0,i,k), diagonal
    #   lhs[3, i, k] = Fortran lhs( 1,i,k), first subdiagonal
    #   lhs[4, i, k] = Fortran lhs( 2,i,k), second subdiagonal
    upper_2_band = lhs[0].T
    upper_1_band = lhs[1].T
    diag_band = lhs[2].T
    lower_1_band = lhs[3].T
    lower_2_band = lhs[4].T
    rhs_t = rhs.T

    lower_diag_invrs_0 = 1.0 / diag_band[0]
    upper_1_0 = lower_diag_invrs_0 * upper_1_band[0]
    upper_2_0 = lower_diag_invrs_0 * upper_2_band[0]
    lower_1_0 = jnp.zeros_like(lower_diag_invrs_0)
    lower_2_0 = jnp.zeros_like(lower_diag_invrs_0)

    lower_1_1 = lower_1_band[1]
    lower_2_1 = jnp.zeros_like(lower_diag_invrs_0)
    lower_diag_invrs_1 = 1.0 / (
        diag_band[1] - lower_1_1 * upper_1_0
    )
    upper_1_1 = lower_diag_invrs_1 * (
        upper_1_band[1] - lower_1_1 * upper_2_0
    )
    upper_2_1 = lower_diag_invrs_1 * upper_2_band[1]

    def lu_step(carry, x):
        upper_1_km1, upper_1_km2, upper_2_km1, upper_2_km2 = carry
        upper_2_lhs, upper_1_lhs, diag_lhs, lower_1_lhs, lower_2_lhs = x
        lower_2_k = lower_2_lhs
        lower_1_k = lower_1_lhs - lower_2_k * upper_1_km2
        lower_diag_invrs_k = 1.0 / (
            diag_lhs
            - lower_2_k * upper_2_km2
            - lower_1_k * upper_1_km1
        )
        upper_1_k = lower_diag_invrs_k * (
            upper_1_lhs - lower_1_k * upper_2_km1
        )
        upper_2_k = lower_diag_invrs_k * upper_2_lhs
        return (
            upper_1_k,
            upper_1_km1,
            upper_2_k,
            upper_2_km1,
        ), (
            lower_diag_invrs_k,
            lower_1_k,
            lower_2_k,
            upper_1_k,
            upper_2_k,
        )

    _, (
        lower_diag_invrs_rest,
        lower_1_rest,
        lower_2_rest,
        upper_1_rest,
        upper_2_rest,
    ) = jax.lax.scan(
        lu_step,
        (upper_1_1, upper_1_0, upper_2_1, upper_2_0),
        (
            upper_2_band[2:],
            upper_1_band[2:],
            diag_band[2:],
            lower_1_band[2:],
            lower_2_band[2:],
        ),
    )

    lower_diag_invrs = jnp.concatenate(
        [
            lower_diag_invrs_0[None],
            lower_diag_invrs_1[None],
            lower_diag_invrs_rest,
        ],
        axis=0,
    )
    lower_1 = jnp.concatenate(
        [lower_1_0[None], lower_1_1[None], lower_1_rest],
        axis=0,
    )
    lower_2 = jnp.concatenate(
        [lower_2_0[None], lower_2_1[None], lower_2_rest],
        axis=0,
    )
    upper_1 = jnp.concatenate(
        [upper_1_0[None], upper_1_1[None], upper_1_rest],
        axis=0,
    )
    upper_2 = jnp.concatenate(
        [upper_2_0[None], upper_2_1[None], upper_2_rest],
        axis=0,
    )

    soln_0 = lower_diag_invrs[0] * rhs_t[0]
    soln_1 = lower_diag_invrs[1] * (rhs_t[1] - lower_1[1] * soln_0)

    def forward_substitution(carry, x):
        soln_km2, soln_km1 = carry
        rhs_k, lower_diag_invrs_k, lower_1_k, lower_2_k = x
        soln_k = lower_diag_invrs_k * (
            rhs_k - lower_2_k * soln_km2 - lower_1_k * soln_km1
        )
        return (soln_km1, soln_k), soln_k

    _, soln_rest = jax.lax.scan(
        forward_substitution,
        (soln_0, soln_1),
        (rhs_t[2:], lower_diag_invrs[2:], lower_1[2:], lower_2[2:]),
    )
    soln = jnp.concatenate([soln_0[None], soln_1[None], soln_rest], axis=0)

    soln_ndim_minus_1 = (
        soln[ndim - 2] - upper_1[ndim - 2] * soln[ndim - 1]
    )

    def backward_substitution(carry, x):
        soln_kp1, soln_kp2 = carry
        soln_k_old, upper_1_k, upper_2_k = x
        soln_k = soln_k_old - upper_1_k * soln_kp1 - upper_2_k * soln_kp2
        return (soln_k, soln_kp1), soln_k

    _, soln_reversed = jax.lax.scan(
        backward_substitution,
        (soln_ndim_minus_1, soln[ndim - 1]),
        (
            soln[: ndim - 2][::-1],
            upper_1[: ndim - 2][::-1],
            upper_2[: ndim - 2][::-1],
        ),
    )
    return jnp.concatenate(
        [
            soln_reversed[::-1],
            soln_ndim_minus_1[None],
            soln[ndim - 1 : ndim],
        ],
        axis=0,
    ).T


@partial(jax.jit, static_argnames=("ndim", "nrhs", "ngrdcol"))
def penta_lu_solve_multiple_rhs_lhs(
    ndim: int, nrhs: int, ngrdcol: int, lhs, rhs
):
    """Written for multiple RHS and multiple LHS."""
    del nrhs, ngrdcol

    # LHS is stored in band diagonal form:
    #   lhs[0, i, k] = Fortran lhs(-2,i,k), second superdiagonal
    #   lhs[1, i, k] = Fortran lhs(-1,i,k), first superdiagonal
    #   lhs[2, i, k] = Fortran lhs( 0,i,k), diagonal
    #   lhs[3, i, k] = Fortran lhs( 1,i,k), first subdiagonal
    #   lhs[4, i, k] = Fortran lhs( 2,i,k), second subdiagonal
    upper_2_band = lhs[0].T
    upper_1_band = lhs[1].T
    diag_band = lhs[2].T
    lower_1_band = lhs[3].T
    lower_2_band = lhs[4].T
    rhs_t = jnp.transpose(rhs, (1, 0, 2))

    lower_diag_invrs_0 = 1.0 / diag_band[0]
    upper_1_0 = lower_diag_invrs_0 * upper_1_band[0]
    upper_2_0 = lower_diag_invrs_0 * upper_2_band[0]
    lower_1_0 = jnp.zeros_like(lower_diag_invrs_0)
    lower_2_0 = jnp.zeros_like(lower_diag_invrs_0)

    lower_1_1 = lower_1_band[1]
    lower_2_1 = jnp.zeros_like(lower_diag_invrs_0)
    lower_diag_invrs_1 = 1.0 / (
        diag_band[1] - lower_1_1 * upper_1_0
    )
    upper_1_1 = lower_diag_invrs_1 * (
        upper_1_band[1] - lower_1_1 * upper_2_0
    )
    upper_2_1 = lower_diag_invrs_1 * upper_2_band[1]

    def lu_step(carry, x):
        upper_1_km1, upper_1_km2, upper_2_km1, upper_2_km2 = carry
        upper_2_lhs, upper_1_lhs, diag_lhs, lower_1_lhs, lower_2_lhs = x
        lower_2_k = lower_2_lhs
        lower_1_k = lower_1_lhs - lower_2_k * upper_1_km2
        lower_diag_invrs_k = 1.0 / (
            diag_lhs
            - lower_2_k * upper_2_km2
            - lower_1_k * upper_1_km1
        )
        upper_1_k = lower_diag_invrs_k * (
            upper_1_lhs - lower_1_k * upper_2_km1
        )
        upper_2_k = lower_diag_invrs_k * upper_2_lhs
        return (
            upper_1_k,
            upper_1_km1,
            upper_2_k,
            upper_2_km1,
        ), (
            lower_diag_invrs_k,
            lower_1_k,
            lower_2_k,
            upper_1_k,
            upper_2_k,
        )

    _, (
        lower_diag_invrs_rest,
        lower_1_rest,
        lower_2_rest,
        upper_1_rest,
        upper_2_rest,
    ) = jax.lax.scan(
        lu_step,
        (upper_1_1, upper_1_0, upper_2_1, upper_2_0),
        (
            upper_2_band[2:],
            upper_1_band[2:],
            diag_band[2:],
            lower_1_band[2:],
            lower_2_band[2:],
        ),
    )

    lower_diag_invrs = jnp.concatenate(
        [
            lower_diag_invrs_0[None],
            lower_diag_invrs_1[None],
            lower_diag_invrs_rest,
        ],
        axis=0,
    )
    lower_1 = jnp.concatenate(
        [lower_1_0[None], lower_1_1[None], lower_1_rest],
        axis=0,
    )
    lower_2 = jnp.concatenate(
        [lower_2_0[None], lower_2_1[None], lower_2_rest],
        axis=0,
    )
    upper_1 = jnp.concatenate(
        [upper_1_0[None], upper_1_1[None], upper_1_rest],
        axis=0,
    )
    upper_2 = jnp.concatenate(
        [upper_2_0[None], upper_2_1[None], upper_2_rest],
        axis=0,
    )

    soln_0 = lower_diag_invrs[0, :, None] * rhs_t[0]
    soln_1 = lower_diag_invrs[1, :, None] * (
        rhs_t[1] - lower_1[1, :, None] * soln_0
    )

    def forward_substitution(carry, x):
        soln_km2, soln_km1 = carry
        rhs_k, lower_diag_invrs_k, lower_1_k, lower_2_k = x
        soln_k = lower_diag_invrs_k[:, None] * (
            rhs_k
            - lower_2_k[:, None] * soln_km2
            - lower_1_k[:, None] * soln_km1
        )
        return (soln_km1, soln_k), soln_k

    _, soln_rest = jax.lax.scan(
        forward_substitution,
        (soln_0, soln_1),
        (rhs_t[2:], lower_diag_invrs[2:], lower_1[2:], lower_2[2:]),
    )
    soln = jnp.concatenate([soln_0[None], soln_1[None], soln_rest], axis=0)

    soln_ndim_minus_1 = (
        soln[ndim - 2] - upper_1[ndim - 2, :, None] * soln[ndim - 1]
    )

    def backward_substitution(carry, x):
        soln_kp1, soln_kp2 = carry
        soln_k_old, upper_1_k, upper_2_k = x
        soln_k = (
            soln_k_old
            - upper_1_k[:, None] * soln_kp1
            - upper_2_k[:, None] * soln_kp2
        )
        return (soln_k, soln_kp1), soln_k

    _, soln_reversed = jax.lax.scan(
        backward_substitution,
        (soln_ndim_minus_1, soln[ndim - 1]),
        (
            soln[: ndim - 2][::-1],
            upper_1[: ndim - 2][::-1],
            upper_2[: ndim - 2][::-1],
        ),
    )
    return jnp.transpose(
        jnp.concatenate(
            [
                soln_reversed[::-1],
                soln_ndim_minus_1[None],
                soln[ndim - 1 : ndim],
            ],
            axis=0,
        ),
        (1, 0, 2),
    )


def penta_lu_solve(ndim: int, ngrdcol: int, lhs, rhs):
    """Generic pentadiagonal LU solve interface."""
    if rhs.ndim == 2:
        return penta_lu_solve_single_rhs_multiple_lhs(ndim, ngrdcol, lhs, rhs)
    if rhs.ndim == 3:
        return penta_lu_solve_multiple_rhs_lhs(
            ndim, rhs.shape[2], ngrdcol, lhs, rhs
        )
    raise ValueError("rhs must be rank-2 or rank-3 for penta_lu_solve")


__all__ = [
    "penta_lu_solve",
    "penta_lu_solve_single_rhs_multiple_lhs",
    "penta_lu_solve_multiple_rhs_lhs",
]
