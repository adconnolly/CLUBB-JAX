"""JAX port of LY93_pdf.F90 — the Lewellen & Yoh (1993) binormal PDF parameters.

iiPDF_LY93 is an alternative PDF closure (the gated CLUBB config uses ADG1), so this is a completeness port.
calc_params_LY93 gives the two PDF-component means and variances of a variable x from its overall mean,
variance, skewness, and the mixture fraction (Lewellen & Yoh 1993, Eqs. 14–18). Pure-jnp → differentiable.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# grad-safe sqrt(max(x,0)) — the canonical tracer-toolkit helper.
from clubb_jax.src.CLUBB_core.pdf_utilities import _safe_sqrt as _ssqrt


def _scbrt(x):
    """Real cube root of x>=0 with a finite gradient at x=0 (cbrt'(0)=inf otherwise)."""
    xp = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.cbrt(xp), 0.0)


def calc_params_LY93(xm, xp2, Skx, mixt_frac):
    """PDF-component means/variances of x (Lewellen & Yoh 1993; LY93_pdf.F90:calc_params_LY93).

    sgn = sign(Skx) (with sgn(0)=+1); B_x = sgn·√(xp2)·(|Skx|/(1−mf))^(1/3); then
      mu_x_1 = xm − B_x(1−mf),  mu_x_2 = xm + B_x·mf,
      sigma_x_1_sqd = xp2 − B_x²(1−mf)(1+mf+mf²)/(3 mf),
      sigma_x_2_sqd = xp2 + B_x²(1−mf)²/3.
    All args are arrays of the same shape. Returns (mu_x_1, mu_x_2, sigma_x_1_sqd, sigma_x_2_sqd)."""
    xm = jnp.asarray(xm, dtype=jnp.float64)
    xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    Skx = jnp.asarray(Skx, dtype=jnp.float64)
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64)
    omf = 1.0 - mf

    sgn = jnp.where(Skx >= 0.0, 1.0, -1.0)
    B_x = sgn * _ssqrt(xp2) * _scbrt(jnp.abs(Skx) / omf)

    mu_x_1 = xm - B_x * omf
    mu_x_2 = xm + B_x * mf
    sigma_x_1_sqd = xp2 - B_x ** 2 * omf * (1.0 + mf + mf ** 2) / (3.0 * mf)
    sigma_x_2_sqd = xp2 + B_x ** 2 * omf ** 2 / 3.0
    return mu_x_1, mu_x_2, sigma_x_1_sqd, sigma_x_2_sqd


from jax import lax  # noqa: E402


def calc_mixt_frac_LY93(Sk_max, itermax=60):
    """LY93 mixture fraction (LY93_pdf.F90:calc_mixt_frac_LY93, Eq. 21). For Sk_max <= 0.84, mixt_frac = 3/4;
    otherwise solve mixt_frac^6 = Sk_max^2 (1 - mixt_frac) by bisection on [1/2, 1] to tolerance 1e-4. The JAX
    port replicates the Fortran's exact bisection, freezing each point once |expr| < 1e-4 (a fixed `lax.scan`;
    itermax=60 exceeds the ~16 steps the 1e-4 tolerance needs). Pure-jnp → differentiable."""
    Sk_max = jnp.asarray(Sk_max, dtype=jnp.float64)
    tol = 1.0e-4
    use_iter = Sk_max > 0.84

    def step(carry, _):
        mf, low, high, done = carry
        expr = mf ** 6 - Sk_max ** 2 * (1.0 - mf)
        hit = jnp.abs(expr) < tol
        new_high = jnp.where(expr > 0.0, mf, high)
        new_low = jnp.where(expr < 0.0, mf, low)
        new_done = done | hit
        new_mf = jnp.where(new_done, mf, 0.5 * (new_low + new_high))
        new_low = jnp.where(done, low, new_low)
        new_high = jnp.where(done, high, new_high)
        return (new_mf, new_low, new_high, new_done), None

    half = 0.5 * jnp.ones_like(Sk_max)
    one_ = jnp.ones_like(Sk_max)
    init = (0.75 * jnp.ones_like(Sk_max), half, one_, jnp.zeros_like(Sk_max, dtype=bool))
    (mf_final, _, _, _), _ = lax.scan(step, init, None, length=itermax)
    return jnp.where(use_iter, mf_final, 0.75)


def LY93_driver(wm, rtm, thlm, wp2, rtp2, thlp2, Skw, Skrt, Skthl):
    """LY93 PDF driver (LY93_pdf.F90:LY93_driver): mixture fraction from Sk_max = max(|Skw|,|Skrt|,|Skthl|),
    then calc_params_LY93 for w, rt, thl with that shared mixt_frac. Returns
    (mu_w_1, mu_w_2, mu_rt_1, mu_rt_2, mu_thl_1, mu_thl_2,
     sigma_w_1_sqd, sigma_w_2_sqd, sigma_rt_1_sqd, sigma_rt_2_sqd, sigma_thl_1_sqd, sigma_thl_2_sqd, mixt_frac)."""
    Skw = jnp.asarray(Skw, dtype=jnp.float64)
    Skrt = jnp.asarray(Skrt, dtype=jnp.float64)
    Skthl = jnp.asarray(Skthl, dtype=jnp.float64)
    Sk_max = jnp.maximum(jnp.maximum(jnp.abs(Skw), jnp.abs(Skrt)), jnp.abs(Skthl))
    mixt_frac = calc_mixt_frac_LY93(Sk_max)
    mu_w_1, mu_w_2, sig_w_1, sig_w_2 = calc_params_LY93(wm, wp2, Skw, mixt_frac)
    mu_rt_1, mu_rt_2, sig_rt_1, sig_rt_2 = calc_params_LY93(rtm, rtp2, Skrt, mixt_frac)
    mu_thl_1, mu_thl_2, sig_thl_1, sig_thl_2 = calc_params_LY93(thlm, thlp2, Skthl, mixt_frac)
    return (mu_w_1, mu_w_2, mu_rt_1, mu_rt_2, mu_thl_1, mu_thl_2,
            sig_w_1, sig_w_2, sig_rt_1, sig_rt_2, sig_thl_1, sig_thl_2, mixt_frac)
