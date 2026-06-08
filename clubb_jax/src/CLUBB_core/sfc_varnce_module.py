"""JAX implementation of sfc_varnce_module.F90:calc_sfc_varnce.

ARM flags: l_andre_1978=False (parameter), l_vary_convect_depth=False.
Only the non-Andre, non-vary-depth path is implemented.
"""

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.tracer_numpy import _safe_sqrt, _safe_pow  # REFACTOR B5: finite grad at 0
from clubb_jax.src.CLUBB_core.constants_clubb import (
    eps,
    grav,
    max_mag_correlation_flux,
    rt_tol,
    thl_tol,
    ufmin,
    w_tol_sqd,
    wp2_max,
)

# Local parameter constants from sfc_varnce_module.F90
_Z_CONST = 1.0         # Defined height of 1 m (z_const)


def calc_sfc_varnce(
    upwp_sfc,       # (ngrdcol,)   surface u-momentum flux [m^2/s^2]
    vpwp_sfc,       # (ngrdcol,)   surface v-momentum flux [m^2/s^2]
    wpthlp_zm,      # (ngrdcol, nzm) full zm wpthlp; surface = [:, 0]
    wprtp_sfc,      # (ngrdcol,)   surface moisture flux [kg/kg m/s]
    lhs_splat_wp2,  # (ngrdcol, nzm) splat LHS coefficient
    tau_zm,         # (ngrdcol, nzm) turbulent timescale [s]
    T0,             # scalar        reference temperature [K]
    up2_sfc_coef,   # (ngrdcol,)   tunable parameter
    a_const,        # (ngrdcol,)   tunable parameter
    wp2,            # (ngrdcol, nzm) modified in-place (surface level only)
    up2,            # (ngrdcol, nzm)
    vp2,            # (ngrdcol, nzm)
    thlp2,          # (ngrdcol, nzm)
    rtp2,           # (ngrdcol, nzm)
    rtpthlp,        # (ngrdcol, nzm)
    zm_sfc,         # (ngrdcol,)   zm[:, 0] surface height [m]
    sfc_elevation,  # (ngrdcol,)   actual surface elevation [m]
):
    """Surface variance parameterization (Andre et al. 1978, else-branch).

    Implements sfc_varnce_module.F90:calc_sfc_varnce for the ARM case:
      l_andre_1978 = .false. (fixed parameter)
      l_vary_convect_depth = .false.

    Returns updated (wp2, up2, vp2, thlp2, rtp2, rtpthlp).
    Only the surface level (index 0) of each array is modified.
    """
    # Extract surface-level slices
    wpthlp_sfc = wpthlp_zm[:, 0]   # surface heat flux [K m/s]
    tau_sfc    = tau_zm[:, 0]       # surface timescale [s]
    lhs_sfc    = lhs_splat_wp2[:, 0]

    # 1. Friction velocity squared (magnitude of surface momentum flux)
    #    In Fortran: ustar2 = sqrt(upwp^2 + vpwp^2)  [m^2/s^2]
    ustar2 = _safe_sqrt(upwp_sfc ** 2 + vpwp_sfc ** 2)   # _safe_sqrt: finite grad at zero surface stress

    # 2. Convective velocity (l_vary_convect_depth=False → use z_const=1 m)
    #    wstar = ( (1/T0) * grav * wpthlp_sfc * z_const )^(1/3)  when wpthlp > 0
    # _safe_pow (REFACTOR B5): in STABLE conditions (wpthlp_sfc<0, e.g. gabls3) safe_cubed=max(0,·)=0 and the
    # bare 0**(1/3) has an inf reverse grad that the `where` masks into a nan (0*inf). Forward-identical.
    safe_cubed = jnp.maximum(0.0, (1.0 / T0) * grav * wpthlp_sfc * _Z_CONST)
    wstar = jnp.where(wpthlp_sfc > 0.0, _safe_pow(safe_cubed, 1.0 / 3.0), 0.0)

    # 3. Effective surface velocity (l_vary_convect_depth=False → coef=0.3)
    uf = _safe_sqrt(ustar2 + 0.3 * wstar ** 2)
    uf = jnp.maximum(ufmin, uf)

    # 4. Surface second-order moments (l_vary_convect_depth=False)
    wp2_sfc    = a_const * uf ** 2
    up2_sfc    = up2_sfc_coef * a_const * uf ** 2
    vp2_sfc    = up2_sfc_coef * a_const * uf ** 2
    thlp2_sfc  = 0.4 * a_const * (wpthlp_sfc / uf) ** 2
    rtp2_sfc   = 0.4 * a_const * (wprtp_sfc  / uf) ** 2
    rtpthlp_sfc = 0.2 * a_const * (wpthlp_sfc / uf) * (wprtp_sfc / uf)

    # 5. Clip scalar variances to tolerance floor
    thlp2_sfc = jnp.maximum(thl_tol ** 2, thlp2_sfc)
    rtp2_sfc  = jnp.maximum(rt_tol  ** 2, rtp2_sfc)

    # 6. Minimum wp2 to keep w-rt and w-thl correlations within (-1, 1)
    corr2 = max_mag_correlation_flux ** 2
    min_wp2 = jnp.maximum(
        w_tol_sqd,
        jnp.maximum(
            wprtp_sfc  ** 2 / (rtp2_sfc  * corr2),
            wpthlp_sfc ** 2 / (thlp2_sfc * corr2),
        ),
    )

    # 7. Splat correction for wp2 at the surface
    #    Fortran: if wp2 - tau*lhs*wp2 < min_wp2: use correction = -wp2 + min_wp2
    #             else: correction = tau*lhs*wp2, wp2 += correction
    wp2_after_splat = wp2_sfc - tau_sfc * lhs_sfc * wp2_sfc
    splat_cond = wp2_after_splat < min_wp2
    correction = jnp.where(splat_cond, -wp2_sfc + min_wp2, tau_sfc * lhs_sfc * wp2_sfc)
    wp2_sfc = jnp.where(splat_cond, min_wp2, wp2_sfc + correction)
    up2_sfc = up2_sfc - 0.5 * correction
    vp2_sfc = vp2_sfc - 0.5 * correction

    # 8. Clip wp2 at wp2_max
    wp2_sfc = jnp.minimum(wp2_sfc, wp2_max)

    # 9. Zero out surface variances if lowest zm level is not at the surface
    #    Fortran: if abs(zm_sfc - sfc_elev) > abs(zm_sfc + sfc_elev)*eps/2
    not_at_sfc = jnp.abs(zm_sfc - sfc_elevation) > jnp.abs(zm_sfc + sfc_elevation) * eps / 2.0
    wp2_sfc     = jnp.where(not_at_sfc, w_tol_sqd,      wp2_sfc)
    up2_sfc     = jnp.where(not_at_sfc, w_tol_sqd,      up2_sfc)
    vp2_sfc     = jnp.where(not_at_sfc, w_tol_sqd,      vp2_sfc)
    thlp2_sfc   = jnp.where(not_at_sfc, thl_tol ** 2,   thlp2_sfc)
    rtp2_sfc    = jnp.where(not_at_sfc, rt_tol  ** 2,   rtp2_sfc)
    rtpthlp_sfc = jnp.where(not_at_sfc, 0.0,            rtpthlp_sfc)

    # 10. Write surface level back into full arrays (JAX immutable update)
    wp2     = wp2.at[:, 0].set(wp2_sfc)
    up2     = up2.at[:, 0].set(up2_sfc)
    vp2     = vp2.at[:, 0].set(vp2_sfc)
    thlp2   = thlp2.at[:, 0].set(thlp2_sfc)
    rtp2    = rtp2.at[:, 0].set(rtp2_sfc)
    rtpthlp = rtpthlp.at[:, 0].set(rtpthlp_sfc)

    return wp2, up2, vp2, thlp2, rtp2, rtpthlp
