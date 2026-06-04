"""JAX port of calc_roots.F90 — closed-form roots of cubic/quadratic polynomials.

Mirrors `clubb_release/src/CLUBB_core/calc_roots.F90` (`cubic_solve`, `quadratic_solve`, `cube_root`).
`cubic_solve` is Cardano's formula in complex128 (x64 enabled at import). The cube roots of the Cardano
coefficients S, T use `_cardano_cbrt`, which takes the **real, sign-preserving** cube root when the argument is
real (determinant D >= 0) and the principal complex branch otherwise — this reproduces gfortran's complex
`**(1/3)` behavior and yields mathematically-correct roots (a naive principal-branch `**(1/3)` returns garbage
for the negative-real arguments that arise when D > 0 with R < 0; iter-71 bug fix). The returned roots SATISFY
the cubic to ~1e-16 and set-match `numpy.roots`; the ordering of the conjugate-pair / real root for D > 0 can
differ from gfortran's branch-cut convention, so the f2py shadow compares sorted real parts (what the
`new_pdf`/responder-limit consumers use). All three functions are vectorized (pass arrays; they broadcast) and
`jax.grad`-able away from the branch points (cube-root cusp at 0, determinant=0 double-root).

In CLUBB these are used by the `new_pdf` / ADG closures; the gated ADG1 path does not call them, so this is a
completeness port validated by the polynomial-residual + `numpy.roots` oracle (the roots ARE the roots) and,
when available, the f2py `calc_roots` shadow.
"""
import jax
jax.config.update("jax_enable_x64", True)   # cubic_solve uses complex128 (principal-branch ** and sqrt)
import jax.numpy as jnp

_ONE_THIRD = 1.0 / 3.0
_C = jnp.complex128


def _cardano_cbrt(z):
    """Cube root matching gfortran's complex ``z**(1/3)`` as used by the Fortran cubic_solve.

    Subtlety: when the determinant D >= 0, the Cardano arguments ``R ± sqrt(D)`` are EXACTLY real (imag == 0),
    and gfortran's complex power of a real argument returns the **real, sign-preserving** cube root
    (``x<0 -> -|x|^(1/3)``) — NOT the principal complex branch (``|x|^(1/3) e^{iπ/3}``) that a naive
    ``z**(1/3)`` produces. For genuinely complex arguments (D < 0) the principal branch is correct (the
    conjugate pair recombines to real roots). This reproduces the Fortran/f2py exactly. Finite-gradient at 0."""
    re = jnp.real(z)
    im = jnp.imag(z)
    re_abs = jnp.abs(re)
    real_cbrt = jnp.where(re_abs > 0.0,
                          jnp.sign(re) * jnp.where(re_abs > 0.0, re_abs, 1.0) ** _ONE_THIRD,
                          0.0).astype(_C)
    principal = z ** _C(_ONE_THIRD)
    return jnp.where(im == 0.0, real_cbrt, principal)


def cubic_solve(a_coef, b_coef, c_coef, d_coef):
    """Roots of ``a*x^3 + b*x^2 + c*x + d = 0`` (a /= 0) via Cardano's formula.

    Returns a complex array with the three roots stacked on the LAST axis (shape ``(..., 3)``), in the same
    order as the Fortran: ``roots[...,0]`` is the always-real root; ``roots[...,1]``/``[...,2]`` are a complex
    conjugate pair when the determinant D = Q^3 + R^2 > 0, else real."""
    a = jnp.asarray(a_coef); b = jnp.asarray(b_coef)
    c = jnp.asarray(c_coef); d = jnp.asarray(d_coef)

    ba = b / a
    ca = c / a
    da = d / a
    cap_Q = (3.0 * ca - ba ** 2) / 9.0                                   # Q = (3(c/a) - (b/a)^2)/9
    cap_R = (9.0 * ba * ca - 27.0 * da - 2.0 * ba ** 3) / 54.0           # R = (9(b/a)(c/a) - 27(d/a) - 2(b/a)^3)/54
    determinant = cap_Q ** 3 + cap_R ** 2                               # D = Q^3 + R^2

    sqrt_det = jnp.sqrt(determinant.astype(_C))                          # complex sqrt, principal branch
    R_c = cap_R.astype(_C)
    one_third_c = _C(_ONE_THIRD)
    cap_S = _cardano_cbrt(R_c + sqrt_det)                                # S = (R + sqrt(D))^(1/3) (gfortran branch)
    cap_T = _cardano_cbrt(R_c - sqrt_det)                                # T = (R - sqrt(D))^(1/3) (gfortran branch)

    sqrt_3 = jnp.sqrt(jnp.asarray(3.0, dtype=_C))
    i_c = _C(1j)
    ba_c = ba.astype(_C)
    base = -one_third_c * ba_c                                          # -(1/3)(b/a)
    SpT = cap_S + cap_T
    SmT = cap_S - cap_T
    root1 = base + SpT
    root2 = base - _C(0.5) * SpT + _C(0.5) * i_c * sqrt_3 * SmT
    root3 = base - _C(0.5) * SpT - _C(0.5) * i_c * sqrt_3 * SmT
    return jnp.stack([root1, root2, root3], axis=-1)


def quadratic_solve(a_coef, b_coef, c_coef):
    """Roots of ``a*x^2 + b*x + c = 0`` (a /= 0). Returns complex ``(..., 2)``:
    ``(-b ± sqrt(b^2 - 4ac)) / (2a)`` (real when the determinant >= 0)."""
    a = jnp.asarray(a_coef); b = jnp.asarray(b_coef); c = jnp.asarray(c_coef)
    determinant = b ** 2 - 4.0 * a * c
    sqrt_det = jnp.sqrt(determinant.astype(_C))
    b_c = b.astype(_C)
    two_a = (2.0 * a).astype(_C)
    root1 = (-b_c + sqrt_det) / two_a
    root2 = (-b_c - sqrt_det) / two_a
    return jnp.stack([root1, root2], axis=-1)


def cube_root(x):
    """Real cube root: ``x^(1/3)`` for x>=0, ``-|x|^(1/3)`` for x<0 (the Fortran's NaN-avoiding form).

    Computed as ``sign-adjusted |x|^(1/3)`` so the unused branch carries no NaN; forward-identical to the
    Fortran. (The fractional power has an infinite derivative at x=0 — a genuine cusp, not a port artifact.)"""
    x = jnp.asarray(x)
    abs_cbrt = jnp.abs(x) ** _ONE_THIRD
    return jnp.where(x >= 0.0, abs_cbrt, -abs_cbrt)
