"""JAX port of diagnose_correlations_module.F90 (active routines).

Mirrors `clubb_release/src/CLUBB_core/diagnose_correlations_module.F90`: `diagnose_correlations` diagnoses a
hydrometeor correlation matrix (Larson et al. 2011, JGR 116 D00T02) to feed SILHS — it swaps the w-variable to
row/col 1 (`rearrange_corr_array`), fills the missing upper-triangle correlations via eq. (15)/(16)
(`diagnose_corr`), then swaps back. Plus the PDF helpers `calc_w_corr`, `calc_varnce`, `calc_mean`.

This is a completeness port (the gated config uses PRESCRIBED correlations via `corr_varnce_module`, not this
dynamic diagnosis; the commented-out `approx_w_corr` w-correlation path is NOT ported, so `l_calc_w_corr=True`
is unsupported — the Fortran leaves the swapped array uninitialised there). Bit-validated against the f2py
`f2py_diagnose_correlations` oracle (`tests/test_diagnose_correlations.py`). Pure jnp → `jax.grad`-able.
"""
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import max_mag_correlation
from clubb_jax.src.CLUBB_core.tracer_numpy import _safe_sqrt


def _rearrange_corr_array(pdf_dim, iipdf_w, corr_array):
    """Swap variable `iipdf_w` (1-based) with variable 1 in the (lower-triangular-stored) correlation matrix.
    Faithful 0-based transcription of the Fortran section assignments (all RHS read the ORIGINAL matrix —
    the in-place writes do not alias the reads)."""
    n = int(pdf_dim)
    w0 = int(iipdf_w) - 1
    C = jnp.asarray(corr_array)
    swap = C[:, 0]                                              # saved original column 1
    out = C
    out = out.at[0:w0 + 1, 0].set(C[w0, w0::-1])               # col1[0..w0] = row w0 reversed (cols w0..0)
    out = out.at[w0 + 1:n, 0].set(C[w0 + 1:n, w0])            # col1[w0+1..] = col w0[w0+1..]
    out = out.at[w0, 0:w0 + 1].set(swap[w0::-1])              # row w0[0..w0] = saved col1 reversed
    out = out.at[w0 + 1:n, w0].set(swap[w0 + 1:n])           # col w0[w0+1..] = saved col1[w0+1..]
    return out


def _diagnose_corr(corr_approx, corr_prescribed):
    """Diagnose the upper-triangle correlations (Larson 2011 eq. 15):
    `c_ij = c_i1*c_j1 + f_ij*sqrt(1-c_i1^2)*sqrt(1-c_j1^2)`, `f_ij = clip(c_ij^prescribed, ±0.99)`,
    for j=2..n-1, i=j+1..n (1-based). `sqrt_sigma2_on_mu2_ip` is unused (always 0 in the driver)."""
    Ca = jnp.asarray(corr_approx)
    Cp = jnp.asarray(corr_prescribed)
    n = Ca.shape[0]
    s_1j = _safe_sqrt(1.0 - Ca[:, 0] ** 2)                     # forward-identical to sqrt(1-c^2) for |c|<=1
    f = jnp.clip(Cp, -max_mag_correlation, max_mag_correlation)
    diagnosed = jnp.outer(Ca[:, 0], Ca[:, 0]) + f * jnp.outer(s_1j, s_1j)
    i_idx = jnp.arange(n)[:, None]
    j_idx = jnp.arange(n)[None, :]
    mask = (i_idx > j_idx) & (j_idx >= 1) & (j_idx <= n - 2)   # upper-tri, cols 2..n-1 (1-based)
    return jnp.where(mask, diagnosed, Ca)


def diagnose_correlations(pdf_dim, iipdf_w, corr_array_pre, l_calc_w_corr=False):
    """Diagnose the correlation matrix for SILHS (Larson et al. 2011). Returns the (pdf_dim, pdf_dim)
    diagnosed `corr_array`. `l_calc_w_corr=True` (the commented-out approx_w_corr path) is unsupported."""
    if l_calc_w_corr:
        raise NotImplementedError("diagnose_correlations: l_calc_w_corr=True (approx_w_corr) is not ported "
                                  "(commented out in the Fortran; it leaves the swapped array uninitialised).")
    pre_swapped = _rearrange_corr_array(pdf_dim, iipdf_w, corr_array_pre)
    swapped = _diagnose_corr(pre_swapped, pre_swapped)         # corr_array_swapped = pre_swapped (not l_calc_w_corr)
    return _rearrange_corr_array(pdf_dim, iipdf_w, swapped)


def calc_w_corr(wpxp, stdev_w, stdev_x, w_tol, x_tol):
    """Correlation of w with x, clipped to ±max_mag_correlation: `wpxp/(max(stdev_x,x_tol)*max(stdev_w,w_tol))`."""
    corr = wpxp / (jnp.maximum(stdev_x, x_tol) * jnp.maximum(stdev_w, w_tol))
    return jnp.clip(corr, -max_mag_correlation, max_mag_correlation)


def calc_varnce(mixt_frac, x1, x2, xm, x1p2, x2p2):
    """Variance of x from its two PDF components: `a*((x1-xm)^2+x1p2) + (1-a)*((x2-xm)^2+x2p2)`."""
    return mixt_frac * ((x1 - xm) ** 2 + x1p2) + (1.0 - mixt_frac) * ((x2 - xm) ** 2 + x2p2)


def calc_mean(mixt_frac, x1, x2):
    """Mean of x from its two PDF components: `a*x1 + (1-a)*x2`."""
    return mixt_frac * x1 + (1.0 - mixt_frac) * x2


def setup_corr_cholesky_mtx(n_variables, corr_matrix):
    """Transposed correlation Cholesky matrix L' from a correlation matrix (Larson 2011 formula 10;
    diagnose_correlations_module.F90:setup_corr_cholesky_mtx).

    Uses the angle parameterization s(j,i)=sqrt(1-corr(j,i)^2): the result is lower-triangular (0-based
    row>=col), with diagonal L'(j,j)=prod_{i<j} s(j,i), first column L'(j,0)=corr(j,0), and off-diagonal
    L'(j,i)=corr(j,i)*prod_{k<i} s(j,k). Pure-jnp (n_variables is static) → differentiable."""
    n = int(n_variables)
    C = jnp.asarray(corr_matrix, dtype=jnp.float64)
    s = _safe_sqrt(1.0 - C ** 2)                       # s[j,i] = sqrt(1 - corr[j,i]^2)
    rows = jnp.arange(n)[:, None]
    cols = jnp.arange(n)[None, :]
    M = jnp.where(rows >= cols, 1.0, 0.0)              # init lower triangle + diagonal to 1

    # Diagonal: M[j,j] = product_{i<j} s[j,i]
    for j in range(1, n):
        prod = 1.0
        for i in range(j):
            prod = prod * s[j, i]
        M = M.at[j, j].set(prod)

    # First column: M[j,0] = corr[j,0]
    for j in range(1, n):
        M = M.at[j, 0].set(C[j, 0])

    # Off-diagonal lower triangle: M[j,i] = corr[j,i] * product_{k<i} s[j,k]
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            prod = 1.0
            for k in range(i):
                prod = prod * s[j, k]
            M = M.at[j, i].set(prod * C[j, i])

    return M


def cholesky_to_corr_mtx_approx(corr_cholesky_mtx_t):
    """Approximate correlation matrix C = L' L'^T from the transposed correlation Cholesky matrix
    (diagnose_correlations_module.F90:cholesky_to_corr_mtx_approx, ref formula 8)."""
    M = jnp.asarray(corr_cholesky_mtx_t, dtype=jnp.float64)
    return M @ M.T


def calc_cholesky_corr_mtx_approx(n_variables, iipdf_w, corr_matrix):
    """Transposed correlation Cholesky matrix and the approximated correlation matrix C=LL'
    (diagnose_correlations_module.F90:calc_cholesky_corr_mtx_approx, Larson 2011). Swaps w to the first
    row/col, builds the angle-Cholesky, swaps back; the approximation is built from the swapped Cholesky then
    swapped back, with the strict upper triangle zeroed. iipdf_w is 1-based. Returns
    (corr_cholesky_mtx, corr_mtx_approx)."""
    n = int(n_variables)
    corr_swap = _rearrange_corr_array(n, iipdf_w, corr_matrix)
    chol_swap = setup_corr_cholesky_mtx(n, corr_swap)
    corr_cholesky_mtx = _rearrange_corr_array(n, iipdf_w, chol_swap)
    approx_swap = cholesky_to_corr_mtx_approx(chol_swap)
    corr_mtx_approx = _rearrange_corr_array(n, iipdf_w, approx_swap)
    # Zero the strict upper triangle (row < col) for conformity.
    rows = jnp.arange(n)[:, None]
    cols = jnp.arange(n)[None, :]
    corr_mtx_approx = jnp.where(rows >= cols, corr_mtx_approx, 0.0)
    return corr_cholesky_mtx, corr_mtx_approx
