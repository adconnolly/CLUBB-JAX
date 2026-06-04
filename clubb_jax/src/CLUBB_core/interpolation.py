"""JAX port of interpolation.F90 — CLUBB's vertical interpolation primitives.

lin_interpolate_two_points — straight linear interpolation between two known (height, value) points.
mono_cubic_interp — Steffen (1990) monotone cubic interpolation (with the optional non-monotone quintic
variant), used to interpolate a field to an arbitrary altitude between grid levels. Both pure-jnp in their
float arguments → differentiable; the integer k-level arguments select interpolate-vs-extrapolate / boundary
branches and are treated as static Python control flow (they are concrete grid indices, not differentiated).
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_EPS = 1.0e-10   # constants_clubb.F90 eps


def lin_interpolate_two_points(height_int, height_high, height_low, var_high, var_low):
    """Linear interpolation of a variable to height_int, given its values at height_high and height_low
    (interpolation.F90:lin_interpolate_two_points):
      var = (height_int - height_low)/(height_high - height_low) * (var_high - var_low) + var_low.
    """
    return ((height_int - height_low) / (height_high - height_low)) * (var_high - var_low) + var_low


def mono_cubic_interp(z_in, km1, k00, kp1, kp2, zm1, z00, zp1, zp2, fm1, f00, fp1, fp2,
                      l_quintic_poly_interp=False):
    """Steffen (1990) monotone cubic interpolation to altitude z_in between z00 and zp1
    (interpolation.F90:mono_cubic_interp). km1/k00/kp1/kp2 are integer grid indices selecting the branch:
    when km1 <= k00 a cubic (or quintic) is fit using the slopes at the surrounding levels with Steffen's
    monotonicity limiter; otherwise a linear extrapolation is used. Returns the interpolated field value.
    Differentiable in z_in and the field/height values."""
    coef1 = 1.0
    coef2 = 1.0

    def _lim(sa, sb, p):
        # Steffen's limited derivative: (sign(sa)+sign(sb)) * min(|sa|, |sb|, |p|/2 * coef2).
        return (coef1 * (jnp.sign(sa) + jnp.sign(sb))
                * jnp.minimum(jnp.minimum(jnp.abs(sa), jnp.abs(sb)), coef2 * 0.5 * jnp.abs(p)))

    if km1 <= k00:
        hm1 = z00 - zm1
        h00 = zp1 - z00
        hp1 = zp2 - zp1

        if km1 == k00:
            s00 = (fp1 - f00) / (zp1 - z00)
            sp1 = (fp2 - fp1) / (zp2 - zp1)
            dfdx00 = s00
            pp1 = (s00 * hp1 + sp1 * h00) / (h00 + hp1)
            dfdxp1 = _lim(s00, sp1, pp1)
        elif kp1 == kp2:
            sm1 = (f00 - fm1) / (z00 - zm1)
            s00 = (fp1 - f00) / (zp1 - z00)
            p00 = (sm1 * h00 + s00 * hm1) / (hm1 + h00)
            dfdx00 = _lim(sm1, s00, p00)
            dfdxp1 = s00
        else:
            sm1 = (f00 - fm1) / (z00 - zm1)
            s00 = (fp1 - f00) / (zp1 - z00)
            sp1 = (fp2 - fp1) / (zp2 - zp1)
            p00 = (sm1 * h00 + s00 * hm1) / (hm1 + h00)
            pp1 = (s00 * hp1 + sp1 * h00) / (h00 + hp1)
            dfdx00 = _lim(sm1, s00, p00)
            dfdxp1 = _lim(s00, sp1, pp1)

        if not l_quintic_poly_interp:
            c1 = (dfdx00 + dfdxp1 - 2.0 * s00) / (h00 ** 2)
            c2 = (3.0 * s00 - 2.0 * dfdx00 - dfdxp1) / h00
            c3 = dfdx00
            c4 = f00
            zprime = z_in - z00
            return c4 + zprime * (c3 + zprime * (c2 + zprime * c1))
        else:
            beta = 120.0 * ((fp1 - f00) - 0.5 * h00 * (dfdx00 + dfdxp1))
            # Linear-interpolation fallback when beta underflows, else the quintic.
            alpha = (6.0 / jnp.where(jnp.abs(beta) < _EPS, 1.0, beta)) * h00 * (dfdxp1 - dfdx00) + 0.5
            zn = (z_in - z00) / h00
            quintic = (((beta / 20.0) * zn - (beta * (1.0 + alpha) / 12.0)) * zn
                       + (beta * alpha / 6.0)) * zn ** 2 * zn + dfdx00 * h00 * zn + f00
            linfall = lin_interpolate_two_points(z00, zp1, zm1, fp1, fm1)
            return jnp.where(jnp.abs(beta) < _EPS, linfall, quintic)
    else:
        # Linear extrapolation.
        wp1 = (z_in - z00) / (zp1 - z00)
        w00 = 1.0 - wp1
        return wp1 * fp1 + w00 * f00


def linear_interp_factor(factor, var_high, var_low):
    """Linear-interpolation coefficient applied to two values (interpolation.F90:linear_interp_factor):
      result = factor * (var_high - var_low) + var_low.
    """
    return factor * (var_high - var_low) + var_low


def zlinterp_fnc(grid_out, grid_src, var_src):
    """Vertical linear interpolation of var_src (on grid_src) onto grid_out
    (interpolation.F90:zlinterp_fnc, "LIN_INT" from WRF-HOC). Values below the lowest / above the highest source
    level are set to zero, and altitude is assumed to increase monotonically — i.e. this is exactly a linear
    interpolation with zero-fill outside the source range. Pure-jnp → differentiable in var_src (and grids)."""
    grid_out = jnp.asarray(grid_out, dtype=jnp.float64)
    grid_src = jnp.asarray(grid_src, dtype=jnp.float64)
    var_src = jnp.asarray(var_src, dtype=jnp.float64)
    return jnp.interp(grid_out, grid_src, var_src, left=0.0, right=0.0)
