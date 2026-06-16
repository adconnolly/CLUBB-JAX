"""JAX port of `src/CLUBB_core/tridiag_lu_solver.F90`."""

from __future__ import annotations

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("ndim",))
def tridiag_lu_solve_single_rhs_lhs(ndim: int, lhs, rhs):
    """Written for single RHS and single LHS."""
    del ndim

    # LHS is stored in band diagonal form:
    #   lhs[0, k] = Fortran lhs(-1,k), first superdiagonal
    #   lhs[1, k] = Fortran lhs( 0,k), diagonal
    #   lhs[2, k] = Fortran lhs( 1,k), first subdiagonal
    upper_band = lhs[0]
    diag_band = lhs[1]
    lower_band = lhs[2]

    lower_diag_invrs_0 = 1.0 / diag_band[0]
    upper_0 = lower_diag_invrs_0 * upper_band[0]

    def lu_step(upper_prev, x):
        diag_k, lower_k, upper_k_lhs = x
        lower_diag_invrs_k = 1.0 / (diag_k - lower_k * upper_prev)
        upper_k = lower_diag_invrs_k * upper_k_lhs
        return upper_k, (lower_diag_invrs_k, upper_k)

    upper_last, (lower_diag_invrs_mid, upper_mid) = jax.lax.scan(
        lu_step,
        upper_0,
        (diag_band[1:-1], lower_band[1:-1], upper_band[1:-1]),
    )

    lower_diag_invrs_last = 1.0 / (
        diag_band[-1] - lower_band[-1] * upper_last
    )
    lower_diag_invrs = jnp.concatenate(
        [
            lower_diag_invrs_0[None],
            lower_diag_invrs_mid,
            lower_diag_invrs_last[None],
        ],
        axis=0,
    )
    upper = jnp.concatenate([upper_0[None], upper_mid], axis=0)

    soln_0 = lower_diag_invrs[0] * rhs[0]

    def forward_substitution(soln_prev, x):
        lower_diag_invrs_k, lower_k, rhs_k = x
        soln_k = lower_diag_invrs_k * (rhs_k - lower_k * soln_prev)
        return soln_k, soln_k

    _, soln_rest = jax.lax.scan(
        forward_substitution,
        soln_0,
        (lower_diag_invrs[1:], lower_band[1:], rhs[1:]),
    )
    soln = jnp.concatenate([soln_0[None], soln_rest], axis=0)

    def backward_substitution(soln_next, x):
        soln_k_old, upper_k = x
        soln_k = soln_k_old - upper_k * soln_next
        return soln_k, soln_k

    _, soln_reversed = jax.lax.scan(
        backward_substitution,
        soln[-1],
        (soln[:-1][::-1], upper[::-1]),
    )
    return jnp.concatenate([soln_reversed[::-1], soln[-1:]], axis=0)


@partial(jax.jit, static_argnames=("ndim", "ngrdcol"))
def tridiag_lu_solve_single_rhs_multiple_lhs(ndim: int, ngrdcol: int, lhs, rhs):
    """Written for single RHS and multiple LHS."""
    del ndim, ngrdcol

    # LHS is stored in band diagonal form:
    #   lhs[0, i, k] = Fortran lhs(-1,i,k), first superdiagonal
    #   lhs[1, i, k] = Fortran lhs( 0,i,k), diagonal
    #   lhs[2, i, k] = Fortran lhs( 1,i,k), first subdiagonal
    upper_band = lhs[0].T
    diag_band = lhs[1].T
    lower_band = lhs[2].T
    rhs_t = rhs.T

    lower_diag_invrs_0 = 1.0 / diag_band[0]
    upper_0 = lower_diag_invrs_0 * upper_band[0]

    def lu_step(upper_prev, x):
        diag_k, lower_k, upper_k_lhs = x
        lower_diag_invrs_k = 1.0 / (diag_k - lower_k * upper_prev)
        upper_k = lower_diag_invrs_k * upper_k_lhs
        return upper_k, (lower_diag_invrs_k, upper_k)

    upper_last, (lower_diag_invrs_mid, upper_mid) = jax.lax.scan(
        lu_step,
        upper_0,
        (diag_band[1:-1], lower_band[1:-1], upper_band[1:-1]),
    )

    lower_diag_invrs_last = 1.0 / (
        diag_band[-1] - lower_band[-1] * upper_last
    )
    lower_diag_invrs = jnp.concatenate(
        [
            lower_diag_invrs_0[None],
            lower_diag_invrs_mid,
            lower_diag_invrs_last[None],
        ],
        axis=0,
    )
    upper = jnp.concatenate([upper_0[None], upper_mid], axis=0)

    soln_0 = lower_diag_invrs[0] * rhs_t[0]

    def forward_substitution(soln_prev, x):
        lower_diag_invrs_k, lower_k, rhs_k = x
        soln_k = lower_diag_invrs_k * (rhs_k - lower_k * soln_prev)
        return soln_k, soln_k

    _, soln_rest = jax.lax.scan(
        forward_substitution,
        soln_0,
        (lower_diag_invrs[1:], lower_band[1:], rhs_t[1:]),
    )
    soln = jnp.concatenate([soln_0[None], soln_rest], axis=0)

    def backward_substitution(soln_next, x):
        soln_k_old, upper_k = x
        soln_k = soln_k_old - upper_k * soln_next
        return soln_k, soln_k

    _, soln_reversed = jax.lax.scan(
        backward_substitution,
        soln[-1],
        (soln[:-1][::-1], upper[::-1]),
    )
    return jnp.concatenate([soln_reversed[::-1], soln[-1:]], axis=0).T


@partial(jax.jit, static_argnames=("ndim", "nrhs", "ngrdcol"))
def tridiag_lu_solve_multiple_rhs_lhs(
    ndim: int, nrhs: int, ngrdcol: int, lhs, rhs
):
    """Written for multiple RHS and multiple LHS."""
    del ndim, nrhs, ngrdcol

    # LHS is stored in band diagonal form:
    #   lhs[0, i, k] = Fortran lhs(-1,i,k), first superdiagonal
    #   lhs[1, i, k] = Fortran lhs( 0,i,k), diagonal
    #   lhs[2, i, k] = Fortran lhs( 1,i,k), first subdiagonal
    upper_band = lhs[0].T
    diag_band = lhs[1].T
    lower_band = lhs[2].T
    rhs_t = jnp.transpose(rhs, (1, 0, 2))

    lower_diag_invrs_0 = 1.0 / diag_band[0]
    upper_0 = lower_diag_invrs_0 * upper_band[0]

    def lu_step(upper_prev, x):
        diag_k, lower_k, upper_k_lhs = x
        lower_diag_invrs_k = 1.0 / (diag_k - lower_k * upper_prev)
        upper_k = lower_diag_invrs_k * upper_k_lhs
        return upper_k, (lower_diag_invrs_k, upper_k)

    upper_last, (lower_diag_invrs_mid, upper_mid) = jax.lax.scan(
        lu_step,
        upper_0,
        (diag_band[1:-1], lower_band[1:-1], upper_band[1:-1]),
    )

    lower_diag_invrs_last = 1.0 / (
        diag_band[-1] - lower_band[-1] * upper_last
    )
    lower_diag_invrs = jnp.concatenate(
        [
            lower_diag_invrs_0[None],
            lower_diag_invrs_mid,
            lower_diag_invrs_last[None],
        ],
        axis=0,
    )
    upper = jnp.concatenate([upper_0[None], upper_mid], axis=0)

    soln_0 = lower_diag_invrs[0, :, None] * rhs_t[0]

    def forward_substitution(soln_prev, x):
        lower_diag_invrs_k, lower_k, rhs_k = x
        soln_k = lower_diag_invrs_k[:, None] * (
            rhs_k - lower_k[:, None] * soln_prev
        )
        return soln_k, soln_k

    _, soln_rest = jax.lax.scan(
        forward_substitution,
        soln_0,
        (lower_diag_invrs[1:], lower_band[1:], rhs_t[1:]),
    )
    soln = jnp.concatenate([soln_0[None], soln_rest], axis=0)

    def backward_substitution(soln_next, x):
        soln_k_old, upper_k = x
        soln_k = soln_k_old - upper_k[:, None] * soln_next
        return soln_k, soln_k

    _, soln_reversed = jax.lax.scan(
        backward_substitution,
        soln[-1],
        (soln[:-1][::-1], upper[::-1]),
    )
    return jnp.transpose(
        jnp.concatenate([soln_reversed[::-1], soln[-1:]], axis=0),
        (1, 0, 2),
    )


def tridiag_lu_solve(ndim: int, lhs, rhs):
    """Generic tridiagonal LU solve interface."""
    if lhs.ndim == 2 and rhs.ndim == 1:
        return tridiag_lu_solve_single_rhs_lhs(ndim, lhs, rhs)
    if lhs.ndim == 3 and rhs.ndim == 2:
        return tridiag_lu_solve_single_rhs_multiple_lhs(
            ndim, lhs.shape[1], lhs, rhs
        )
    if lhs.ndim == 3 and rhs.ndim == 3:
        return tridiag_lu_solve_multiple_rhs_lhs(
            ndim, rhs.shape[2], lhs.shape[1], lhs, rhs
        )
    raise ValueError("Unsupported lhs/rhs ranks for tridiag_lu_solve")


__all__ = [
    "tridiag_lu_solve",
    "tridiag_lu_solve_single_rhs_lhs",
    "tridiag_lu_solve_single_rhs_multiple_lhs",
    "tridiag_lu_solve_multiple_rhs_lhs",
]
