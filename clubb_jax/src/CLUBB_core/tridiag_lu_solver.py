"""Tridiagonal LU solver — JAX port of tridiag_lu_solver.F90.

Faithful port of CLUBB_core/tridiag_lu_solver.F90
  tridiag_lu_solve_single_rhs_multiple_lhs  (the common path).

Algorithm: Thomas / LU decomposition:
  1. LU decomposition  — forward sweep (sequential, uses lax.scan)
  2. Forward substitution  (sequential, lax.scan)
  3. Backward substitution (reverse sequential, lax.scan)

LHS band storage (matching diffusion.py output):
  lhs[0, :, k] = superdiagonal  (Fortran lhs(-1, i, k))
  lhs[1, :, k] = main diagonal  (Fortran lhs( 0, i, k))
  lhs[2, :, k] = subdiagonal    (Fortran lhs( 1, i, k))

Array layout: levels on last axis, columns on middle axis.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit, lax


def tridiag_lu_solve_jax(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve a tridiagonal system lhs * soln = rhs via LU decomposition.

    Faithful port of Fortran tridiag_lu_solve_single_rhs_multiple_lhs.

    Fortran band index map:
      lhs(-1, i, k) = lhs[0, i-1, k-1]  superdiagonal
      lhs( 0, i, k) = lhs[1, i-1, k-1]  main diagonal
      lhs( 1, i, k) = lhs[2, i-1, k-1]  subdiagonal

    Args:
        lhs: shape (3, ngrdcol, ndim).
        rhs: shape (ngrdcol, ndim).

    Returns:
        soln: shape (ngrdcol, ndim).
    """
    sup = lhs[0]   # (ngrdcol, ndim)  superdiagonal
    mid = lhs[1]   # (ngrdcol, ndim)  main diagonal
    sub = lhs[2]   # (ngrdcol, ndim)  subdiagonal

    # Transpose to (ndim, ngrdcol) so the scan axis is the level axis.
    sup_t = sup.T   # (ndim, ngrdcol)
    mid_t = mid.T
    sub_t = sub.T
    rhs_t = rhs.T   # (ndim, ngrdcol)

    # ---- LU decomposition ----
    # k = 0
    l_inv_0 = 1.0 / mid_t[0]         # (ngrdcol,)
    upper_0 = l_inv_0 * sup_t[0]      # (ngrdcol,)

    # k = 1 .. ndim-2  (sequential scan)
    def lu_step(carry, x):
        upper_prev = carry             # (ngrdcol,)
        mid_k, sub_k, sup_k = x
        l_inv_k = 1.0 / (mid_k - sub_k * upper_prev)
        upper_k = l_inv_k * sup_k
        return upper_k, (l_inv_k, upper_k)

    final_upper, (l_inv_int, upper_int) = lax.scan(
        lu_step, upper_0,
        (mid_t[1:-1], sub_t[1:-1], sup_t[1:-1]),
    )
    # l_inv_int: (ndim-2, ngrdcol), upper_int: (ndim-2, ngrdcol)

    # k = ndim-1
    l_inv_last = 1.0 / (mid_t[-1] - sub_t[-1] * final_upper)   # (ngrdcol,)

    # Assemble
    l_inv_t = jnp.concatenate(
        [l_inv_0[None], l_inv_int, l_inv_last[None]], axis=0
    )   # (ndim, ngrdcol)
    upper_t = jnp.concatenate(
        [upper_0[None], upper_int], axis=0
    )   # (ndim-1, ngrdcol)

    # ---- Forward substitution: L * y = rhs  (store in soln_t) ----
    soln_0 = l_inv_t[0] * rhs_t[0]   # (ngrdcol,)

    def fwd_step(carry, x):
        soln_prev = carry
        l_inv_k, sub_k, rhs_k = x
        return l_inv_k * (rhs_k - sub_k * soln_prev), None

    soln_t_end, _ = lax.scan(
        fwd_step, soln_0,
        (l_inv_t[1:], sub_t[1:], rhs_t[1:]),
    )
    # Need all intermediate values, not just the last.
    # Re-scan storing outputs:
    def fwd_step_store(carry, x):
        soln_prev = carry
        l_inv_k, sub_k, rhs_k = x
        soln_k = l_inv_k * (rhs_k - sub_k * soln_prev)
        return soln_k, soln_k

    _, soln_int_t = lax.scan(
        fwd_step_store, soln_0,
        (l_inv_t[1:], sub_t[1:], rhs_t[1:]),
    )
    # soln_int_t: (ndim-1, ngrdcol)

    soln_t = jnp.concatenate([soln_0[None], soln_int_t], axis=0)   # (ndim, ngrdcol)

    # ---- Backward substitution: U * x = y ----
    def bwd_step(carry, x):
        soln_next = carry
        soln_k, upper_k = x
        soln_k_new = soln_k - upper_k * soln_next
        return soln_k_new, soln_k_new

    # Scan from k = ndim-2 down to k = 0
    _, soln_updated_rev = lax.scan(
        bwd_step, soln_t[-1],
        (soln_t[:-1][::-1], upper_t[::-1]),
    )
    # soln_updated_rev: (ndim-1, ngrdcol) in order [k=ndim-2, k=ndim-3, ..., k=0]

    soln_final_t = jnp.concatenate(
        [soln_updated_rev[::-1], soln_t[-1:]], axis=0
    )   # (ndim, ngrdcol)

    return soln_final_t.T   # (ngrdcol, ndim)


# JIT-compiled production version.
# Called eagerly (per prognostic variable, per timestep), the solver redefines its nested scan bodies
# (lu_step/fwd_step/bwd_step) on every call, so JAX's scan compile-cache misses and XLA recompiles each
# step → unbounded compile-cache growth (a co-cause of the rico OOM, Iter290). Jitting the pure
# (lhs, rhs) -> soln entry point makes every call hit the jit cache by input aval (one compile per distinct
# grid size, then reused), bounding memory and removing the per-step recompiles. jit is value-preserving and
# composes with grad, so the solve stays bit-identical and differentiable. Both names are jitted so callers
# importing either hit the cache.
tridiag_lu_solve_jax = jit(tridiag_lu_solve_jax)
tridiag_lu_solve = tridiag_lu_solve_jax


__all__ = ["tridiag_lu_solve_jax", "tridiag_lu_solve"]
