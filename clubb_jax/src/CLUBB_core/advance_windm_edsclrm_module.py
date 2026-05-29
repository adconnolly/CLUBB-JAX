"""JAX port of advance_windm_edsclrm_module.F90.

Advances horizontal mean winds um/vm and momentum fluxes upwp/vpwp
using eddy-diffusivity closure with Crank-Nicholson time-stepping.

For l_predict_upwp_vpwp=True (ARM case), um/vm are already advanced in
advance_xm_wpxp and this function is a no-op for the wind variables.
"""

import jax
import jax.numpy as jnp
from jax import lax, jit

from clubb_jax.src.CLUBB_core.diffusion import diffusion_zt_lhs_jax
from clubb_jax.src.CLUBB_core.matrix_solver_wrapper import tridiag_lu_solve_jax
from clubb_jax.src.CLUBB_core.advance_xm_wpxp_module import clip_covar_jax, term_ma_zt_lhs_jax
from clubb_jax.src.CLUBB_core.constants_clubb import ic_K10, ic_K10h
from clubb_jax.src.CLUBB_core.grid_class import zm2zt_jax

jax.config.update("jax_enable_x64", True)

# CLUBB eps constant: max(1e-10, machine_eps) = 1e-10 for float64
_EPS = 1.0e-10


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _term_ma_zt_lhs_centered_jax(
    wm_zt: jnp.ndarray,
    gr,
) -> jnp.ndarray:
    """Mean-advection LHS for zt-level variable, centered scheme.

    Faithful port of mean_adv.F90:term_ma_zt_lhs with l_upwind_xm_ma=False,
    ascending grid (grid_dir=+1).

    Interior k=1..nzt-2 (Python), Fortran k=2..nzt-1:
      super[k] = wm_zt[k] * invrs_dzt[k] * weights_zt2zm[k+1, 0]
      main[k]  = wm_zt[k] * invrs_dzt[k] * (weights_zt2zm[k+1, 1] - weights_zt2zm[k, 0])
      sub[k]   = -wm_zt[k] * invrs_dzt[k] * weights_zt2zm[k, 1]

    Lower boundary (k=0, Fortran k=1):
      super[0] = wm_zt[0] * invrs_dzt[0] * weights_zt2zm[1, 0]
      main[0]  = -wm_zt[0] * invrs_dzt[0] * (1 - weights_zt2zm[1, 1])
      sub[0]   = 0

    Upper boundary (k=nzt-1, Fortran k=nzt):
      super[-1] = 0
      main[-1]  = wm_zt[-1] * invrs_dzt[-1] * (1 - weights_zt2zm[-2, 0])
      sub[-1]   = -wm_zt[-1] * invrs_dzt[-1] * weights_zt2zm[-2, 1]
    """
    ngrdcol, nzt = wm_zt.shape
    invrs_dzt = gr.invrs_dzt       # (ngrdcol, nzt)
    w = gr.weights_zt2zm           # (ngrdcol, nzm, 2); index 0=t_above, 1=t_below

    # Interior k=1..nzt-2 (Python)
    fac = wm_zt[:, 1:-1] * invrs_dzt[:, 1:-1]      # (ngrdcol, nzt-2)
    super_int = fac * w[:, 2:-1, 0]                  # w[k_py+1, t_above]
    main_int  = fac * (w[:, 2:-1, 1] - w[:, 1:-2, 0])
    sub_int   = -fac * w[:, 1:-2, 1]

    # Lower boundary k=0
    fac0 = wm_zt[:, 0] * invrs_dzt[:, 0]            # (ngrdcol,)
    sup_bot = fac0 * w[:, 1, 0]
    mid_bot = -fac0 * (1.0 - w[:, 1, 1])
    sub_bot = jnp.zeros((ngrdcol,), dtype=wm_zt.dtype)

    # Upper boundary k=nzt-1
    fac_top = wm_zt[:, -1] * invrs_dzt[:, -1]       # (ngrdcol,)
    sup_top = jnp.zeros((ngrdcol,), dtype=wm_zt.dtype)
    mid_top = fac_top * (1.0 - w[:, -2, 0])
    sub_top = -fac_top * w[:, -2, 1]

    sup = jnp.concatenate([sup_bot[:, None], super_int, sup_top[:, None]], axis=1)
    mid = jnp.concatenate([mid_bot[:, None], main_int,  mid_top[:, None]], axis=1)
    sub = jnp.concatenate([sub_bot[:, None], sub_int,   sub_top[:, None]], axis=1)

    return jnp.stack([sup, mid, sub], axis=0)   # (3, ngrdcol, nzt)


def _calc_xpwp_jax(
    Km_zm_p_nu: jnp.ndarray,    # (ngrdcol, nzm)
    xm: jnp.ndarray,             # (ngrdcol, nzt)
    invrs_dzm: jnp.ndarray,      # (ngrdcol, nzm)
) -> jnp.ndarray:
    """Compute x'w' at interior momentum levels k=1..nzm-2 (Python).

    Faithful port of advance_helper_module.F90:calc_xpwp_2D.
    Formula (Fortran k=2..nzm-1, ascending):
      xpwp[k] = Km_zm_p_nu[k] * invrs_dzm[k] * (xm[k] - xm[k-1])

    xm is on zt-levels (nzt = nzm-1), so:
      xm[k]   for zm index k (1..nzm-2) → xm[:, 1:]  (indices 1..nzm-2)
      xm[k-1] for zm index k            → xm[:, :-1] (indices 0..nzm-3)

    Returns shape (ngrdcol, nzm-2) representing interior levels.
    """
    return Km_zm_p_nu[:, 1:-1] * invrs_dzm[:, 1:-1] * (xm[:, 1:] - xm[:, :-1])


def _windm_rhs_jax(
    lhs_diff: jnp.ndarray,   # (3, ngrdcol, nzt)
    xm: jnp.ndarray,          # (ngrdcol, nzt) - wind component at time t
    xm_tndcy: jnp.ndarray,    # (ngrdcol, nzt) - Coriolis+forcing tendency
    dt: float,
) -> jnp.ndarray:
    """Compute RHS for um or vm tridiagonal solve.

    Faithful port of windm_edsclrm_rhs (ascending grid, l_imp_sfc_momentum_flux=True).

    rhs[k] = 0.5 * explicit_diffusion[k] + xm_tndcy[k] + xm[k]/dt

    For implicit surface flux (l_imp_sfc_momentum_flux=True), no explicit
    surface flux term is added (it is handled implicitly in the LHS).
    """
    invrs_dt = 1.0 / dt

    # Lower boundary k=0 (Fortran k=1)
    # rhs[0] = 0.5*(-lhs_diff[1,:,0]*xm[:,0] - lhs_diff[0,:,0]*xm[:,1]) + tndcy + xm/dt
    rhs_bot = (0.5 * (-lhs_diff[1, :, 0] * xm[:, 0]
                      - lhs_diff[0, :, 0] * xm[:, 1])
               + xm_tndcy[:, 0]
               + invrs_dt * xm[:, 0])[:, None]   # (ngrdcol, 1)

    # Interior k=1..nzt-2 (Fortran k=2..nzt-1, ascending: band 3=sub, band 1=super)
    rhs_int = (0.5 * (-lhs_diff[2, :, 1:-1] * xm[:, :-2]
                      - lhs_diff[1, :, 1:-1] * xm[:, 1:-1]
                      - lhs_diff[0, :, 1:-1] * xm[:, 2:])
               + xm_tndcy[:, 1:-1]
               + invrs_dt * xm[:, 1:-1])          # (ngrdcol, nzt-2)

    # Upper boundary k=nzt-1 (Fortran k=nzt)
    rhs_top = (0.5 * (-lhs_diff[2, :, -1] * xm[:, -2]
                      - lhs_diff[1, :, -1] * xm[:, -1])
               + xm_tndcy[:, -1]
               + invrs_dt * xm[:, -1])[:, None]  # (ngrdcol, 1)

    return jnp.concatenate([rhs_bot, rhs_int, rhs_top], axis=1)   # (ngrdcol, nzt)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def advance_windm_edsclrm_jax(
    um: jnp.ndarray,           # (ngrdcol, nzt) - mean u wind
    vm: jnp.ndarray,           # (ngrdcol, nzt) - mean v wind
    upwp: jnp.ndarray,         # (ngrdcol, nzm) - u'w' momentum flux
    vpwp: jnp.ndarray,         # (ngrdcol, nzm) - v'w' momentum flux
    wp2: jnp.ndarray,          # (ngrdcol, nzm) - w'^2 variance
    up2: jnp.ndarray,          # (ngrdcol, nzm) - u'^2 variance
    vp2: jnp.ndarray,          # (ngrdcol, nzm) - v'^2 variance
    wm_zt: jnp.ndarray,        # (ngrdcol, nzt) - mean vertical velocity
    Kh_zm: jnp.ndarray,        # (ngrdcol, nzm) - scalar eddy diffusivity
    ug: jnp.ndarray,           # (ngrdcol, nzt) - u geostrophic wind
    vg: jnp.ndarray,           # (ngrdcol, nzt) - v geostrophic wind
    um_forcing: jnp.ndarray,   # (ngrdcol, nzt) - prescribed u forcing [m/s/s]
    vm_forcing: jnp.ndarray,   # (ngrdcol, nzt) - prescribed v forcing [m/s/s]
    rho_ds_zm: jnp.ndarray,    # (ngrdcol, nzm) - dry static density on zm
    rho_ds_zt: jnp.ndarray,    # (ngrdcol, nzt) - dry static density on zt
    invrs_rho_ds_zt: jnp.ndarray,  # (ngrdcol, nzt) - 1/rho_ds on zt
    fcor: jnp.ndarray,         # (ngrdcol,) - Coriolis parameter [s^-1]
    clubb_params: jnp.ndarray, # (ngrdcol, nparams)
    nu10: float,               # background momentum diffusivity nu10 [m^2/s]
    dt: float,                 # timestep [s]
    gr,
    l_predict_upwp_vpwp: bool,
    l_upwind_xm_ma: bool,
    l_tke_aniso: bool,
) -> tuple:
    """Advance um/vm and upwp/vpwp via eddy-diffusivity closure.

    Faithful port of advance_windm_edsclrm_module.F90:advance_windm_edsclrm
    for the l_predict_upwp_vpwp=False path (standalone mode, ascending grid).

    For ARM (l_predict_upwp_vpwp=True), um/vm/upwp/vpwp are not advanced
    here (they are handled by advance_xm_wpxp) — returns inputs unchanged.

    Returns:
        (um_new, vm_new, upwp_new, vpwp_new) — all (ngrdcol, nz*)
    """
    if l_predict_upwp_vpwp:
        # ARM path: um/vm already advanced by advance_xm_wpxp; this is a no-op.
        return um, vm, upwp, vpwp

    # ------------------------------------------------------------------ #
    # l_predict_upwp_vpwp=False: full eddy-diffusion wind advancement     #
    # ------------------------------------------------------------------ #

    ngrdcol = um.shape[0]

    # Fortran line 328-330: momentum diffusivity
    Km_zm = Kh_zm * clubb_params[:, ic_K10 - 1:ic_K10]  # (ngrdcol, nzm) broadcast
    Km_zm_p_nu10 = Km_zm + nu10

    # Fortran line 367: Km_zt = zm2zt(Km_zm, zero)
    Km_zt = zm2zt_jax(Km_zm, gr)   # (ngrdcol, nzt)

    # Fortran line 370-372: LHS diffusion
    nu10_arr = jnp.full((ngrdcol,), nu10, dtype=um.dtype)
    lhs_diff = diffusion_zt_lhs_jax(Km_zm, nu10_arr, invrs_rho_ds_zt, rho_ds_zm, gr)
    # shape (3, ngrdcol, nzt): [0=super, 1=main, 2=sub]

    # Fortran line 349-352: LHS mean advection (l_implemented=False standalone)
    if l_upwind_xm_ma:
        lhs_ma_zt = term_ma_zt_lhs_jax(wm_zt, gr)           # (3, ngrdcol, nzt) upwind
    else:
        lhs_ma_zt = _term_ma_zt_lhs_centered_jax(wm_zt, gr) # (3, ngrdcol, nzt) centered

    # Fortran lines 390-401: compute_uv_tndcy
    # um: d(um)/dt = -fcor*vg + fcor*vm + um_forcing
    # vm: d(vm)/dt =  fcor*ug - fcor*um + vm_forcing
    fcor2d = fcor[:, None]   # (ngrdcol, 1) → broadcasts over nzt
    um_tndcy = -fcor2d * vg + fcor2d * vm + um_forcing   # (ngrdcol, nzt)
    vm_tndcy =  fcor2d * ug - fcor2d * um + vm_forcing   # (ngrdcol, nzt)

    # Fortran lines 410-416: wind speed, surface u_star^2
    wind_speed = jnp.maximum(jnp.sqrt(um ** 2 + vm ** 2), _EPS)   # (ngrdcol, nzt)
    k_lb_zm = gr.k_lb_zm   # Python 0-based (0 for ascending)
    k_lb_zt = gr.k_lb_zt   # Python 0-based
    k_ub_zm = gr.k_ub_zm   # Python 0-based (nzm-1 for ascending)

    u_star_sqd = jnp.sqrt(
        upwp[:, k_lb_zm] ** 2 + vpwp[:, k_lb_zm] ** 2
    )   # (ngrdcol,)

    # Fortran lines 444-478: Crank-Nicholson explicit half for upwp and vpwp
    xpwp_u = _calc_xpwp_jax(Km_zm_p_nu10, um, gr.invrs_dzm)   # (ngrdcol, nzm-2)
    upwp_new = upwp.at[:, 1:-1].set(-0.5 * xpwp_u)
    upwp_new = upwp_new.at[:, k_ub_zm].set(0.0)

    xpwp_v = _calc_xpwp_jax(Km_zm_p_nu10, vm, gr.invrs_dzm)
    vpwp_new = vpwp.at[:, 1:-1].set(-0.5 * xpwp_v)
    vpwp_new = vpwp_new.at[:, k_ub_zm].set(0.0)

    # Fortran lines 482-487: windm_edsclrm_lhs
    # lhs = 0.5 * lhs_diff + 1/dt * I + lhs_ma_zt[0..nzt-2] + implicit_sfc_flux
    lhs = 0.5 * lhs_diff
    lhs = lhs.at[1].add(1.0 / dt)

    # Add MA only for k=0..nzt-2 (Fortran k=k_lb_zt..k_ub_zt-1; ascending: k=1..nzt-1)
    lhs = lhs.at[:, :, :-1].add(lhs_ma_zt[:, :, :-1])

    # Implicit surface momentum flux (l_imp_sfc_momentum_flux=True)
    # Fortran lines 2336-2343
    sfc_term = (
        invrs_rho_ds_zt[:, k_lb_zt]
        * gr.invrs_dzt[:, k_lb_zt]
        * rho_ds_zm[:, k_lb_zm]
        * (u_star_sqd / wind_speed[:, k_lb_zt])
    )   # (ngrdcol,)  [grid_dir=+1 for ascending]
    lhs = lhs.at[1, :, k_lb_zt].add(sfc_term)

    # Fortran lines 427-441: windm_edsclrm_rhs for um and vm
    rhs_um = _windm_rhs_jax(lhs_diff, um, um_tndcy, dt)   # (ngrdcol, nzt)
    rhs_vm = _windm_rhs_jax(lhs_diff, vm, vm_tndcy, dt)   # (ngrdcol, nzt)

    # Fortran lines 493-498: windm_edsclrm_solve → tridiag solve for um and vm
    um_new = tridiag_lu_solve_jax(lhs, rhs_um)   # (ngrdcol, nzt)
    vm_new = tridiag_lu_solve_jax(lhs, rhs_vm)   # (ngrdcol, nzt)

    # Fortran lines 613-635: second Crank-Nicholson half (implicit component)
    xpwp_u_new = _calc_xpwp_jax(Km_zm_p_nu10, um_new, gr.invrs_dzm)
    upwp_new = upwp_new.at[:, 1:-1].add(-0.5 * xpwp_u_new)

    xpwp_v_new = _calc_xpwp_jax(Km_zm_p_nu10, vm_new, gr.invrs_dzm)
    vpwp_new = vpwp_new.at[:, 1:-1].add(-0.5 * xpwp_v_new)

    # Fortran lines 670-781: clip_covar for upwp and vpwp
    if l_tke_aniso:
        upwp_new = clip_covar_jax(upwp_new, wp2, up2, 0.99)
        vpwp_new = clip_covar_jax(vpwp_new, wp2, vp2, 0.99)
    else:
        upwp_new = clip_covar_jax(upwp_new, wp2, wp2, 0.99)
        vpwp_new = clip_covar_jax(vpwp_new, wp2, wp2, 0.99)

    return um_new, vm_new, upwp_new, vpwp_new
