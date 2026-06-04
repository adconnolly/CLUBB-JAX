"""JAX port of matrix_operations.F90 — the linear-algebra helpers CLUBB's PDF/SILHS code needs.

cholesky_factor — a Cholesky factorization with LAPACK-style symmetric equilibration (dpoequ/dlaqsy) and the
Fortran's iterative "add tau to the diagonal" fallback for a non-positive-definite input (matrix_operations.F90:109).
For a correlation matrix (unit diagonal) the equilibration is a no-op (scaling = 1/sqrt(diag) = 1, scond = 1 >
THRESH), so it reduces to a plain lower-triangular Cholesky — which is the use in calc_corr_norm_and_cholesky_factor.

The plain (positive-definite, no-fallback) path is pure-jnp and differentiable. The tau fallback only triggers
on a concrete (non-traced) non-PD input — it is a numerical safeguard and is skipped under a JAX trace, matching
the Fortran design comment that, with chi/eta never perfectly correlated, the fallback "shouldn't happen now".
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_ITERMAX = 10      # matrix_operations.F90:137
_D_COEF = 0.1      # matrix_operations.F90:139
_THRESH = 0.1      # LAPACK dlaqsy equilibration threshold


def _lower_mask(n):
    """Boolean (n,n) mask, True on and below the diagonal."""
    r = jnp.arange(n)
    return r[:, None] >= r[None, :]


def mirror_lower_triangular_matrix(matrix):
    """Symmetrize a lower-triangular matrix by mirroring the lower triangle onto the upper
    (matrix_operations.F90:mirror_lower_triangular_matrix). The Fortran sets
    ``matrix(row,col) = matrix(col,row)`` for every ``row < col``, i.e. it overwrites the strict
    upper triangle with the transpose of the strict lower triangle, leaving the diagonal and the
    lower triangle untouched. Equivalent to ``tril(M) + tril(M, -1)ᵀ``. Pure-jnp → differentiable.

    Args:
        matrix: square ``(nvars, nvars)`` array; only its lower triangle (and diagonal) is used.

    Returns:
        The symmetric ``(nvars, nvars)`` matrix.
    """
    m = jnp.asarray(matrix, dtype=jnp.float64)
    lower = jnp.tril(m)                       # diagonal + strict lower
    return lower + jnp.tril(m, -1).T          # add strict-lower reflected into the upper triangle


def cholesky_factor(a_input):
    """Lower-triangular Cholesky factor of a symmetric positive-definite matrix, with LAPACK-style
    equilibration (matrix_operations.F90:Cholesky_factor).

    Returns (a_scaling, a_cholesky, l_scaled):
      * a_scaling[i] = 1/sqrt(a[i,i])  (the dpoequ symmetric scaling)
      * l_scaled — whether equilibration was applied (dlaqsy equed=='Y'): True iff scond < THRESH or the max
        diagonal is outside the safe floating range.
      * a_cholesky — the factored matrix: its lower triangle (incl. diagonal) holds L (of the scaled matrix
        when l_scaled, else of a_input); its strict upper triangle retains the working matrix's upper values,
        exactly as LAPACK dpotrf('Lower') leaves them (the caller zeroes the upper triangle).
    """
    a = jnp.asarray(a_input, dtype=jnp.float64)
    n = a.shape[0]
    diag = jnp.diagonal(a)

    # dpoequ: symmetric scaling and condition number.
    a_scaling = 1.0 / jnp.sqrt(diag)
    amax = jnp.max(diag)
    amin = jnp.min(diag)
    scond = jnp.sqrt(amin / amax)

    fi = jnp.finfo(jnp.float64)
    small, large = fi.tiny, 1.0 / fi.tiny
    # dlaqsy: equed='N' (no scaling) iff scond>=THRESH and amax in [small, large]; else scale.
    l_scaled = ~((scond >= _THRESH) & (amax >= small) & (amax <= large))

    a_scaled = a_scaling[:, None] * a * a_scaling[None, :]
    a_work = jnp.where(l_scaled, a_scaled, a)

    mask = _lower_mask(n)

    def _factor(m):
        L = jnp.linalg.cholesky(m)                 # lower factor; strict upper = 0
        return jnp.where(mask, L, m)               # restore strict-upper from the working matrix (dpotrf)

    a_cholesky = _factor(a_work)

    # Tau-on-diagonal fallback for a non-PD input (eager/concrete only; skipped under trace).
    try:
        failed = bool(jnp.isnan(jnp.diagonal(a_cholesky)).any())
    except (jax.errors.TracerBoolConversionError, jax.errors.ConcretizationTypeError):
        failed = False
    if failed:
        d_smallest = float(jnp.min(jnp.diagonal(a_work)))
        for it in range(1, _ITERMAX + 1):
            tau = d_smallest * _D_COEF * it
            m = a_work + tau * jnp.eye(n)
            cand = _factor(m)
            if not bool(jnp.isnan(jnp.diagonal(cand)).any()):
                a_cholesky = cand
                break
            a_cholesky = cand

    return a_scaling, a_cholesky, l_scaled
