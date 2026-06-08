"""JAX port of advance_windm_edsclrm_module.F90.

Advances horizontal mean winds um/vm and momentum fluxes upwp/vpwp
using eddy-diffusivity closure with Crank-Nicholson time-stepping.

For l_predict_upwp_vpwp=True (ARM case), um/vm are already advanced in
advance_xm_wpxp and this function is a no-op for the wind variables.
"""

import jax
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.diffusion import diffusion_zt_lhs_jax
from clubb_jax.src.CLUBB_core.tridiag_lu_solver import tridiag_lu_solve_jax
from clubb_jax.src.CLUBB_core.clip_explicit import clip_covar
from clubb_jax.src.CLUBB_core.mean_adv import term_ma_zt_lhs_jax
from clubb_jax.src.CLUBB_core.constants_clubb import ic_K10, max_mag_correlation
from clubb_jax.src.CLUBB_core.grid_class import zm2zt_jax
from clubb_jax.src.CLUBB_core.advance_helper_module import calc_xpwp

jax.config.update("jax_enable_x64", True)

# CLUBB eps constant: max(1e-10, machine_eps) = 1e-10 for float64
_EPS = 1.0e-10


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# x'w' down-gradient eddy flux: advance_helper_module.calc_xpwp (the Fortran calc_xpwp_2D).
# calc_xpwp returns the full (ngrdcol, nzm) array with zeroed boundaries; the windm/edsclrm
# Crank-Nicholson half-step needs only the interior, so call sites slice [:, 1:-1].


def windm_edsclrm_rhs(
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
# windm_edsclrm_lhs
# ---------------------------------------------------------------------------

def windm_edsclrm_lhs(lhs_diff, lhs_ma_zt, dt, invrs_rho_ds_zt, rho_ds_zm,
                          u_star_sqd, wind_speed, gr, k_lb_zt, k_lb_zm):
    """Assemble the windm/edsclrm tridiagonal LHS — the JAX analog of
    advance_windm_edsclrm_module.F90:windm_edsclrm_lhs (F90:2236). Combines the Crank-Nicholson diffusion
    (0.5*lhs_diff) + the 1/dt accumulation + mean advection (interior levels only) + the implicit
    surface-momentum-flux term at the lower boundary (F90:2336-2343, l_imp_sfc_momentum_flux=True).
    Returns the (3, ngrdcol, nzt) banded LHS."""
    lhs = 0.5 * lhs_diff
    lhs = lhs.at[1].add(1.0 / dt)
    # Add MA only for k=0..nzt-2 (Fortran k=k_lb_zt..k_ub_zt-1; ascending: k=1..nzt-1)
    lhs = lhs.at[:, :, :-1].add(lhs_ma_zt[:, :, :-1])
    # Implicit surface momentum flux (l_imp_sfc_momentum_flux=True)
    sfc_term = (
        invrs_rho_ds_zt[:, k_lb_zt]
        * gr.invrs_dzt[:, k_lb_zt]
        * rho_ds_zm[:, k_lb_zm]
        * (u_star_sqd / wind_speed[:, k_lb_zt])
    )   # (ngrdcol,)  [grid_dir=+1 for ascending]
    lhs = lhs.at[1, :, k_lb_zt].add(sfc_term)
    return lhs


# ---------------------------------------------------------------------------
# windm_edsclrm_solve
# ---------------------------------------------------------------------------

def windm_edsclrm_solve(lhs, rhs_list):
    """Tridiagonal solve of the windm/edsclrm system for each RHS sharing the one assembled LHS —
    the JAX analog of advance_windm_edsclrm_module.F90:windm_edsclrm_solve (a single-LHS multi-RHS solve;
    in the gated path the RHS list is (um, vm)). Returns one solved field per RHS, in the order given."""
    return tuple(tridiag_lu_solve_jax(lhs, rhs) for rhs in rhs_list)


def compute_uv_tndcy(fcor, ug, vg, um, vm, um_forcing, vm_forcing):
    """Coriolis + geostrophic + prescribed-forcing tendencies for the um/vm winds — the JAX analog of
    advance_windm_edsclrm_module.F90:compute_uv_tndcy (F90:2051). Returns (um_tndcy, vm_tndcy):
    d(um)/dt = -fcor*vg + fcor*vm + um_forcing;  d(vm)/dt = fcor*ug - fcor*um + vm_forcing."""
    fcor2d = fcor[:, None]   # (ngrdcol, 1) → broadcasts over nzt
    um_tndcy = -fcor2d * vg + fcor2d * vm + um_forcing
    vm_tndcy = fcor2d * ug - fcor2d * um + vm_forcing
    return um_tndcy, vm_tndcy


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def advance_windm_edsclrm(
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
    # Single term_ma_zt_lhs subroutine, scheme selected by l_upwind_xm_ma (mirrors Fortran)
    lhs_ma_zt = term_ma_zt_lhs_jax(wm_zt, gr, l_upwind_xm_ma)   # (3, ngrdcol, nzt)

    # Fortran lines 390-401: compute_uv_tndcy (Coriolis + geostrophic + forcing)
    um_tndcy, vm_tndcy = compute_uv_tndcy(fcor, ug, vg, um, vm, um_forcing, vm_forcing)  # each (ngrdcol, nzt)

    # Fortran lines 410-416: wind speed, surface u_star^2
    wind_speed = jnp.maximum(jnp.sqrt(um ** 2 + vm ** 2), _EPS)   # (ngrdcol, nzt)
    k_lb_zm = gr.k_lb_zm   # Python 0-based (0 for ascending)
    k_lb_zt = gr.k_lb_zt   # Python 0-based
    k_ub_zm = gr.k_ub_zm   # Python 0-based (nzm-1 for ascending)

    u_star_sqd = jnp.sqrt(
        upwp[:, k_lb_zm] ** 2 + vpwp[:, k_lb_zm] ** 2
    )   # (ngrdcol,)

    # Fortran lines 444-478: Crank-Nicholson explicit half for upwp and vpwp
    xpwp_u = calc_xpwp(Km_zm_p_nu10, um, gr.invrs_dzm)[:, 1:-1]   # (ngrdcol, nzm-2)
    upwp_new = upwp.at[:, 1:-1].set(-0.5 * xpwp_u)
    upwp_new = upwp_new.at[:, k_ub_zm].set(0.0)

    xpwp_v = calc_xpwp(Km_zm_p_nu10, vm, gr.invrs_dzm)[:, 1:-1]
    vpwp_new = vpwp.at[:, 1:-1].set(-0.5 * xpwp_v)
    vpwp_new = vpwp_new.at[:, k_ub_zm].set(0.0)

    # Fortran lines 482-487: windm_edsclrm_lhs (CN diffusion + 1/dt + MA + implicit sfc momentum flux)
    lhs = windm_edsclrm_lhs(lhs_diff, lhs_ma_zt, dt, invrs_rho_ds_zt, rho_ds_zm,
                                u_star_sqd, wind_speed, gr, k_lb_zt, k_lb_zm)

    # Fortran lines 427-441: windm_edsclrm_rhs for um and vm
    rhs_um = windm_edsclrm_rhs(lhs_diff, um, um_tndcy, dt)   # (ngrdcol, nzt)
    rhs_vm = windm_edsclrm_rhs(lhs_diff, vm, vm_tndcy, dt)   # (ngrdcol, nzt)

    # Fortran lines 493-498: windm_edsclrm_solve → tridiag solve for um and vm (shared LHS)
    um_new, vm_new = windm_edsclrm_solve(lhs, (rhs_um, rhs_vm))   # each (ngrdcol, nzt)

    # Fortran lines 613-635: second Crank-Nicholson half (implicit component)
    xpwp_u_new = calc_xpwp(Km_zm_p_nu10, um_new, gr.invrs_dzm)[:, 1:-1]
    upwp_new = upwp_new.at[:, 1:-1].add(-0.5 * xpwp_u_new)

    xpwp_v_new = calc_xpwp(Km_zm_p_nu10, vm_new, gr.invrs_dzm)[:, 1:-1]
    vpwp_new = vpwp_new.at[:, 1:-1].add(-0.5 * xpwp_v_new)

    # Fortran lines 670-781: clip_covar for upwp and vpwp
    if l_tke_aniso:
        upwp_new = clip_covar(upwp_new, wp2, up2, max_mag_correlation)
        vpwp_new = clip_covar(vpwp_new, wp2, vp2, max_mag_correlation)
    else:
        upwp_new = clip_covar(upwp_new, wp2, wp2, max_mag_correlation)
        vpwp_new = clip_covar(vpwp_new, wp2, wp2, max_mag_correlation)

    return um_new, vm_new, upwp_new, vpwp_new
