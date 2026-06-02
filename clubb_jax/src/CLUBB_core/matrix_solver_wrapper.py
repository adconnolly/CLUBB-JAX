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

import jax
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


# JIT-compiled production version
tridiag_lu_solve = jit(tridiag_lu_solve_jax)


__all__ = ["tridiag_lu_solve_jax", "tridiag_lu_solve",
           "penta_lu_solve_jax", "penta_lu_solve"]


def penta_lu_solve_jax(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve a penta-diagonal system lhs * soln = rhs via LU decomposition.

    Faithful port of Fortran penta_lu_solve_single_rhs_multiple_lhs.

    Fortran band index map (lhs(-2:2, ngrdcol, ndim) → Python (5, ngrdcol, ndim)):
      lhs[0] = lhs(-2) = super2  (2nd superdiagonal)
      lhs[1] = lhs(-1) = super1  (1st superdiagonal)
      lhs[2] = lhs( 0) = diag    (diagonal)
      lhs[3] = lhs( 1) = sub1    (1st subdiagonal)
      lhs[4] = lhs( 2) = sub2    (2nd subdiagonal)

    Args:
        lhs: shape (5, ngrdcol, ndim).
        rhs: shape (ngrdcol, ndim).

    Returns:
        soln: shape (ngrdcol, ndim).
    """
    # Transpose to (ndim, ngrdcol) so scan runs over the level axis.
    super2_t = lhs[0].T   # (ndim, ngrdcol)
    super1_t = lhs[1].T
    diag_t   = lhs[2].T
    sub1_t   = lhs[3].T
    sub2_t   = lhs[4].T
    rhs_t    = rhs.T      # (ndim, ngrdcol)
    ndim = lhs.shape[2]

    # ---- LU decomposition ----
    # k=0 (Fortran k=1): no lower bands
    ldi_0 = 1.0 / diag_t[0]
    u1_0  = ldi_0 * super1_t[0]
    u2_0  = ldi_0 * super2_t[0]
    l1_0  = jnp.zeros_like(ldi_0)
    l2_0  = jnp.zeros_like(ldi_0)

    # k=1 (Fortran k=2): one lower band (sub1), no sub2
    l1_1  = sub1_t[1]
    l2_1  = jnp.zeros_like(ldi_0)
    ldi_1 = 1.0 / (diag_t[1] - l1_1 * u1_0)
    u1_1  = ldi_1 * (super1_t[1] - l1_1 * u2_0)
    u2_1  = ldi_1 * super2_t[1]

    # k=2..ndim-1 (Fortran k=3..ndim): full 5-band step
    def lu_scan_step(carry, x):
        u1_km1, u1_km2, u2_km1, u2_km2 = carry
        s2, s1, d, sb1, sb2 = x        # each (ngrdcol,)
        l2  = sb2
        l1  = sb1 - l2 * u1_km2
        ldi = 1.0 / (d - l2 * u2_km2 - l1 * u1_km1)
        u1  = ldi * (s1 - l1 * u2_km1)
        u2  = ldi * s2
        new_carry = (u1, u1_km1, u2, u2_km1)
        return new_carry, (ldi, l1, l2, u1, u2)

    init_carry = (u1_1, u1_0, u2_1, u2_0)
    _, (ldi_rest, l1_rest, l2_rest, u1_rest, u2_rest) = lax.scan(
        lu_scan_step, init_carry,
        (super2_t[2:], super1_t[2:], diag_t[2:], sub1_t[2:], sub2_t[2:]),
    )
    # each _rest: (ndim-2, ngrdcol)

    # Assemble full arrays (ndim, ngrdcol)
    ldi_t = jnp.concatenate([ldi_0[None], ldi_1[None], ldi_rest], axis=0)
    l1_t  = jnp.concatenate([l1_0[None],  l1_1[None],  l1_rest],  axis=0)
    l2_t  = jnp.concatenate([l2_0[None],  l2_1[None],  l2_rest],  axis=0)
    u1_t  = jnp.concatenate([u1_0[None],  u1_1[None],  u1_rest],  axis=0)
    u2_t  = jnp.concatenate([u2_0[None],  u2_1[None],  u2_rest],  axis=0)

    # ---- Forward substitution: L y = rhs ----
    # soln[0] = ldi[0] * rhs[0]
    # soln[1] = ldi[1] * (rhs[1] - l1[1]*soln[0])
    # soln[k] = ldi[k] * (rhs[k] - l2[k]*soln[k-2] - l1[k]*soln[k-1]) for k>=2
    soln_0 = ldi_t[0] * rhs_t[0]
    soln_1 = ldi_t[1] * (rhs_t[1] - l1_t[1] * soln_0)

    def fwd_scan_step(carry, x):
        soln_km2, soln_km1 = carry
        rhs_k, ldi_k, l1_k, l2_k = x
        soln_k = ldi_k * (rhs_k - l2_k * soln_km2 - l1_k * soln_km1)
        return (soln_km1, soln_k), soln_k

    _, soln_rest = lax.scan(
        fwd_scan_step, (soln_0, soln_1),
        (rhs_t[2:], ldi_t[2:], l1_t[2:], l2_t[2:]),
    )
    # soln_rest: (ndim-2, ngrdcol)

    soln_t = jnp.concatenate([soln_0[None], soln_1[None], soln_rest], axis=0)  # (ndim, ngrdcol)

    # ---- Backward substitution: U x = y ----
    # Fortran (1-indexed):
    #   soln[ndim-1] -= upper_1[ndim-1] * soln[ndim]
    #   for k = ndim-2 .. 1: soln[k] -= upper_1[k]*soln[k+1] + upper_2[k]*soln[k+2]
    # Python (0-indexed):
    #   soln[ndim-2] -= u1[ndim-2] * soln[ndim-1]
    #   for k = ndim-3 .. 0: soln[k] -= u1[k]*soln[k+1] + u2[k]*soln[k+2]

    # Single step at k=ndim-2
    soln_nm2 = soln_t[ndim-2] - u1_t[ndim-2] * soln_t[ndim-1]

    # Reverse scan from k=ndim-3 down to k=0
    def bwd_scan_step(carry, x):
        soln_kp1, soln_kp2 = carry
        soln_k_fwd, u1_k, u2_k = x
        soln_k = soln_k_fwd - u1_k * soln_kp1 - u2_k * soln_kp2
        return (soln_k, soln_kp1), soln_k

    # Levels 0..ndim-3 in reverse order (scan processes ndim-3 first)
    _, soln_bwd_rev = lax.scan(
        bwd_scan_step,
        (soln_nm2, soln_t[ndim-1]),
        (soln_t[:ndim-2][::-1], u1_t[:ndim-2][::-1], u2_t[:ndim-2][::-1]),
    )
    # soln_bwd_rev: (ndim-2, ngrdcol), order is [k=ndim-3, ..., k=0]

    # Assemble final solution (k=0..ndim-1)
    soln_final_t = jnp.concatenate(
        [soln_bwd_rev[::-1], soln_nm2[None], soln_t[ndim-1:ndim]], axis=0
    )  # (ndim, ngrdcol)

    return soln_final_t.T  # (ngrdcol, ndim)


# Called eagerly (per prognostic variable, per timestep), these solvers redefine their nested scan
# bodies (lu_step/fwd_step/bwd_step) on every call, so JAX's scan compile-cache misses and XLA recompiles
# each step → unbounded compile-cache growth (a co-cause of the rico OOM, Iter290; ~40 scan recompiles/step
# remained after the parabolic_cylinder fix). Jitting the pure (lhs, rhs) -> soln entry points makes every
# call hit the jit cache by input aval (one compile per distinct grid size, then reused), bounding memory
# and removing the per-step recompiles. jit is value-preserving and composes with grad, so the solves stay
# bit-identical and differentiable. All callers import these names, so rebinding here covers every use.
tridiag_lu_solve_jax = jit(tridiag_lu_solve_jax)
penta_lu_solve_jax = jit(penta_lu_solve_jax)
penta_lu_solve = penta_lu_solve_jax   # back-compat alias (advance_xm_wpxp_module imports this)
