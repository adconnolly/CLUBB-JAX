"""JAX implementations of mixing_length.F90.

diagnose_lscale_from_tau_jax:
  ARM flags: l_e3sm_config=False, l_smooth_Heaviside_tau_wpxp=True,
             l_smooth_min_max=False (local Fortran constant)

compute_mixing_length_jax:
  Golaz et al. (2002) nonlocal parcel length scale.
  Faithful port of compute_mixing_length (ascending grid only).

calc_lscale_directly_jax:
  Wrapper calling compute_mixing_length_jax once (l_avg_Lscale=False).
"""

import jax
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import (
    Cp,
    em_min,
    ep,
    ep1,
    ep2,
    eps,
    grav,
    Lv,
    Rd,
    min_max_smth_mag,
    rt_tol,
    thl_tol,
    vonk,
    zero_threshold,
    iC_invrs_tau_bkgnd,
    iC_invrs_tau_sfc,
    iC_invrs_tau_shear,
    iC_invrs_tau_N2,
    iC_invrs_tau_N2_wp2,
    iC_invrs_tau_N2_xp2,
    iC_invrs_tau_N2_wpxp,
    iC_invrs_tau_N2_clear_wp3,
    iC_invrs_tau_wpxp_Ri,
    iC_invrs_tau_wpxp_N2_thresh,
    ialtitude_threshold,
    iwpxp_Ri_exp,
    iz_displace,
    iLscale_mu_coef,
    iLscale_pert_coef,
)
from clubb_jax.src.CLUBB_core.grid_class import (
    zt2zm_jax,
    zm2zt_jax,
    zm2zt2zm_jax,
)
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_liq_jax

# Local Fortran constant: smoothing range for Peskin Heaviside
_HEAVISIDE_SMTH_RANGE = 1.0


def _smooth_max_jax(arr, scalar, smth_coef):
    """smooth_max_array_scalar: 0.5*(a + b + sqrt((a-b)^2 + smth_coef^2))."""
    return 0.5 * ((arr + scalar) + jnp.sqrt((arr - scalar) ** 2 + smth_coef ** 2))


def _smooth_heaviside_peskin_jax(x, smth_range):
    """Peskin smooth Heaviside function (advance_helper_module.F90).

    H = 0   if x < -smth_range
    H = 1   if x >  smth_range
    H = 0.5*(1 + x/r + (1/pi)*sin(pi*x/r))  otherwise
    """
    x_over_r = x / smth_range
    mid = 0.5 * (1.0 + x_over_r + jnp.sin(jnp.pi * x_over_r) / jnp.pi)
    return jnp.where(x < -smth_range, 0.0, jnp.where(x > smth_range, 1.0, mid))


def diagnose_lscale_from_tau_jax(
    upwp_sfc,                   # (ngrdcol,)
    vpwp_sfc,                   # (ngrdcol,)
    ddzt_umvm_sqd,              # (ngrdcol, nzm)
    ice_supersat_frac,          # (ngrdcol, nzt)
    em,                         # (ngrdcol, nzm)
    sqrt_em_zt,                 # (ngrdcol, nzt)
    ufmin,                      # scalar
    tau_const,                  # scalar
    sfc_elevation,              # (ngrdcol,)
    Lscale_max,                 # (ngrdcol,)
    clubb_params,               # (ngrdcol, nparams)  1-based indices
    Ri_zm,                      # (ngrdcol, nzm)
    brunt_vaisala_freq_sqd_smth, # (ngrdcol, nzm)
    l_e3sm_config: bool,
    l_smooth_Heaviside_tau_wpxp: bool,
    gr,
):
    """diagnose_Lscale_from_tau for ARM flags.

    Returns:
        invrs_tau_zt       (ngrdcol, nzt)
        invrs_tau_zm       (ngrdcol, nzm)
        invrs_tau_sfc      (ngrdcol, nzm)
        invrs_tau_no_N2_zm (ngrdcol, nzm)
        invrs_tau_bkgnd    (ngrdcol, nzm)
        invrs_tau_shear    (ngrdcol, nzm)
        invrs_tau_N2_iso   (ngrdcol, nzm)
        invrs_tau_wp2_zm   (ngrdcol, nzm)
        invrs_tau_xp2_zm   (ngrdcol, nzm)
        invrs_tau_wp3_zm   (ngrdcol, nzm)
        invrs_tau_wp3_zt   (ngrdcol, nzt)
        invrs_tau_wpxp_zm  (ngrdcol, nzm)
        tau_max_zm         (ngrdcol, nzm)
        tau_max_zt         (ngrdcol, nzt)
        tau_zm             (ngrdcol, nzm)
        tau_zt             (ngrdcol, nzt)
        Lscale             (ngrdcol, nzt)
        Lscale_up          (ngrdcol, nzt)
        Lscale_down        (ngrdcol, nzt)
    """
    zm = gr.zm   # (ngrdcol, nzm)

    # --- Step 1: ustar (l_smooth_min_max=False → hard max) ---
    ustar = jnp.maximum((upwp_sfc ** 2 + vpwp_sfc ** 2) ** 0.25, ufmin)  # (ngrdcol,)

    # --- Step 2: invrs_tau_bkgnd ---
    C_bkgnd = clubb_params[:, iC_invrs_tau_bkgnd - 1]       # (ngrdcol,)
    invrs_tau_bkgnd = C_bkgnd[:, None] / tau_const           # (ngrdcol, nzm)

    # --- Step 3–5: invrs_tau_shear via smoothed norm of ddzt ---
    norm_ddzt_umvm = jnp.sqrt(ddzt_umvm_sqd)                 # (ngrdcol, nzm)
    smooth_norm = zm2zt2zm_jax(norm_ddzt_umvm, gr, zm_min=zero_threshold)  # (ngrdcol, nzm)
    C_shear = clubb_params[:, iC_invrs_tau_shear - 1]
    invrs_tau_shear_smooth = C_shear[:, None] * smooth_norm   # (ngrdcol, nzm)
    # smooth_max(array, zero_threshold, min_max_smth_mag) ≈ max(array, 0)
    invrs_tau_shear = _smooth_max_jax(invrs_tau_shear_smooth, zero_threshold, min_max_smth_mag)

    # --- Step 6: invrs_tau_sfc ---
    C_sfc = clubb_params[:, iC_invrs_tau_sfc - 1]            # (ngrdcol,)
    z_displace = clubb_params[:, iz_displace - 1]             # (ngrdcol,)
    z_eff = zm - sfc_elevation[:, None] + z_displace[:, None]  # (ngrdcol, nzm)
    invrs_tau_sfc = C_sfc[:, None] * (ustar[:, None] / vonk) / z_eff  # (ngrdcol, nzm)

    # --- Step 7: invrs_tau_no_N2_zm ---
    invrs_tau_no_N2_zm = invrs_tau_bkgnd + invrs_tau_sfc + invrs_tau_shear

    # --- Step 8: brunt_freq_pos (l_smooth_min_max=False) ---
    brunt_freq_pos = jnp.sqrt(jnp.maximum(zero_threshold, brunt_vaisala_freq_sqd_smth))

    # --- Step 9: ice_supersat_frac_zm, brunt_freq_out_cloud ---
    ice_supersat_frac_zm = zt2zm_jax(ice_supersat_frac, gr, zm_min=zero_threshold)
    fac = jnp.minimum(1.0, jnp.maximum(zero_threshold, 1.0 - ice_supersat_frac_zm / 0.001))
    brunt_freq_out_cloud = brunt_freq_pos * fac
    # zero below altitude_threshold
    alt_thresh = clubb_params[:, ialtitude_threshold - 1]     # (ngrdcol,)
    brunt_freq_out_cloud = jnp.where(zm < alt_thresh[:, None], 0.0, brunt_freq_out_cloud)

    # --- Step 10: invrs_tau_N2_iso, invrs_tau_wp2_zm, invrs_tau_zm ---
    C_N2     = clubb_params[:, iC_invrs_tau_N2 - 1]          # (ngrdcol,)
    C_N2_wp2 = clubb_params[:, iC_invrs_tau_N2_wp2 - 1]      # (ngrdcol,)

    invrs_tau_N2_iso  = (invrs_tau_bkgnd + invrs_tau_shear
                         + C_N2_wp2[:, None] * brunt_freq_pos)

    invrs_tau_wp2_zm  = (invrs_tau_no_N2_zm
                         + C_N2[:, None] * brunt_freq_pos
                         + C_N2_wp2[:, None] * brunt_freq_out_cloud)

    invrs_tau_zm      = invrs_tau_no_N2_zm + C_N2[:, None] * brunt_freq_pos

    # --- Step 11: xp2 / wpxp (l_e3sm_config=False path) ---
    if l_e3sm_config:
        # Not the ARM path; included for completeness
        invrs_tau_zm      = 0.5 * invrs_tau_zm
        C_N2_xp2 = clubb_params[:, iC_invrs_tau_N2_xp2 - 1]
        invrs_tau_xp2_zm  = (invrs_tau_no_N2_zm
                              + C_N2_xp2[:, None] * brunt_freq_pos
                              + C_sfc[:, None] * 2.0
                              * jnp.sqrt(em) / z_eff)
        bv_safe = jnp.maximum(1.0e-7, brunt_vaisala_freq_sqd_smth)
        ratio = jnp.clip(jnp.sqrt(ddzt_umvm_sqd / bv_safe), 0.3, 1.0)
        invrs_tau_xp2_zm  = ratio * invrs_tau_xp2_zm
        C_N2_wpxp = clubb_params[:, iC_invrs_tau_N2_wpxp - 1]
        invrs_tau_wpxp_zm = (2.0 * invrs_tau_zm
                              + C_N2_wpxp[:, None] * brunt_freq_out_cloud)
    else:
        C_N2_xp2 = clubb_params[:, iC_invrs_tau_N2_xp2 - 1]
        invrs_tau_xp2_zm  = (invrs_tau_no_N2_zm
                              + C_N2[:, None] * brunt_freq_pos
                              + C_N2_xp2[:, None] * brunt_freq_out_cloud)
        C_N2_wpxp = clubb_params[:, iC_invrs_tau_N2_wpxp - 1]
        invrs_tau_wpxp_zm = (invrs_tau_no_N2_zm
                              + C_N2[:, None] * brunt_freq_pos
                              + C_N2_wpxp[:, None] * brunt_freq_out_cloud)

    # --- Step 12: Heaviside for invrs_tau_wpxp ---
    C_N2_thresh = clubb_params[:, iC_invrs_tau_wpxp_N2_thresh - 1]  # (ngrdcol,)

    if l_smooth_Heaviside_tau_wpxp:
        bvf_thresh = brunt_vaisala_freq_sqd_smth / C_N2_thresh[:, None] - 1.0
        H = _smooth_heaviside_peskin_jax(bvf_thresh, _HEAVISIDE_SMTH_RANGE)
    else:
        H = jnp.where(brunt_vaisala_freq_sqd_smth > C_N2_thresh[:, None], 1.0, 0.0)

    # --- Step 13: Ri enhancement of invrs_tau_wpxp above altitude_threshold ---
    C_wpxp_Ri  = clubb_params[:, iC_invrs_tau_wpxp_Ri - 1]   # (ngrdcol,)
    wpxp_Ri_exp = clubb_params[:, iwpxp_Ri_exp - 1]           # (ngrdcol,)
    Ri_pos = jnp.maximum(Ri_zm, 0.0)
    Ri_term = jnp.minimum(C_wpxp_Ri[:, None] * Ri_pos ** wpxp_Ri_exp[:, None], 12.0)
    above = zm > alt_thresh[:, None]
    invrs_tau_wpxp_zm = jnp.where(
        above,
        invrs_tau_wpxp_zm * (1.0 + H * Ri_term),
        invrs_tau_wpxp_zm,
    )

    # --- Step 14: invrs_tau_wp3_zm ---
    C_N2_clear_wp3 = clubb_params[:, iC_invrs_tau_N2_clear_wp3 - 1]
    invrs_tau_wp3_zm = invrs_tau_wp2_zm + C_N2_clear_wp3[:, None] * brunt_freq_out_cloud

    # --- Step 15: tau_max_zm, tau_max_zt (l_smooth_min_max=False) ---
    tau_max_zt = Lscale_max[:, None] / sqrt_em_zt                           # (ngrdcol, nzt)
    tau_max_zm = Lscale_max[:, None] / jnp.sqrt(jnp.maximum(em, em_min))   # (ngrdcol, nzm)

    # --- Step 16: tau_zm, tau_zt ---
    tau_zm = jnp.minimum(1.0 / invrs_tau_zm, tau_max_zm)
    tau_zt_raw = zm2zt_jax(tau_zm, gr)
    tau_zt = jnp.minimum(tau_zt_raw, tau_max_zt)

    # --- Step 17: invrs_tau_zt, invrs_tau_wp3_zt ---
    invrs_tau_zt    = zm2zt_jax(invrs_tau_zm, gr)
    invrs_tau_wp3_zt = zm2zt_jax(invrs_tau_wp3_zm, gr)

    # --- Step 18: Lscale ---
    Lscale      = tau_zt * sqrt_em_zt
    Lscale_up   = jnp.zeros_like(Lscale)
    Lscale_down = jnp.zeros_like(Lscale)

    return (
        invrs_tau_zt,
        invrs_tau_zm,
        invrs_tau_sfc,
        invrs_tau_no_N2_zm,
        invrs_tau_bkgnd,
        invrs_tau_shear,
        invrs_tau_N2_iso,
        invrs_tau_wp2_zm,
        invrs_tau_xp2_zm,
        invrs_tau_wp3_zm,
        invrs_tau_wp3_zt,
        invrs_tau_wpxp_zm,
        tau_max_zm,
        tau_max_zt,
        tau_zm,
        tau_zt,
        Lscale,
        Lscale_up,
        Lscale_down,
        brunt_freq_pos,
        brunt_freq_out_cloud,
    )


# ---------------------------------------------------------------------------
# compute_mixing_length_jax
# Golaz et al. (2002) nonlocal parcel length scale.
# Faithful port of mixing_length.F90:compute_mixing_length.
# Ascending grid only (grid_dir_indx = +1).
# ---------------------------------------------------------------------------

_ZLMIN = 0.1               # minimum Lscale [m]
_LSCALE_SFCLYR_DEPTH = 500.0  # surface-layer depth for lminh [m]
_LV2_COEF = ep * Lv ** 2 / (Rd * Cp)  # ep*Lv²/(Rd*Cp)  [K²]


def _parcel_thv(thl_par, rt_par, exner_j, p_j, thv_ds_j, Lv_coef_j, saturation_formula):
    """Compute thv_par at a single level j (scalar).
    Lewellen-Yoh (1993) condensate formula inside the parcel.
    """
    tl_j = thl_par * exner_j
    rsat_j = sat_mixrat_liq_jax(
        jnp.reshape(p_j, (1, 1)),
        jnp.reshape(tl_j, (1, 1)),
        saturation_formula,
    )[0, 0]
    tl_sqd = tl_j ** 2
    s_j = (rt_par - rsat_j) * tl_sqd / (tl_sqd + _LV2_COEF * rsat_j)
    rc_j = jnp.maximum(s_j, zero_threshold)
    return thl_par + ep1 * thv_ds_j * rt_par + Lv_coef_j * rc_j


def _upward_inner_while(
    k_py,
    tke_0,
    thl_init, rt_init, dCAPE_init,   # parcel state at k_py+1 (initial step result)
    thl_precalc_up,                   # (nzm,) thl_par_j_precalc for upward
    rt_precalc_up,                    # (nzm,)
    exp_mu_dzm,                       # (nzm,) per zm level
    grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,  # (nzt,) per zt level
    dzm, invrs_dzm,                   # (nzm,)
    zt,                               # (nzt,) altitude
    k_ub_zt_py,
    saturation_formula,
):
    """Inner while loop for upward parcel trajectory from k_py+2.

    Fortran pattern:
        j = k + 2
        do while (j < k_ub_zt)
            compute CAPE_incr at j using parcel thl/rt from prev step
            if (tke + CAPE_incr <= 0) exit   ! j stays, early exit
            tke += CAPE_incr; j += 1
        end do
        Lscale_up[k] += zt[j-1] - zt[k]  (full levels crossed)
        + quadratic formula for fractional level

    Returns (Lscale_up_k, j_final, exited_early, tke_exit, dCAPE_exit_prev, dCAPE_exit_j)
    """
    # State: (j, tke, thl_par, rt_par, dCAPE_prev, done, j_last_reached,
    #         tke_at_exit, dCAPE_exit_prev, dCAPE_exit_j)
    init_state = (
        k_py + 2,           # j: next level to test (ascending)
        tke_0,              # remaining TKE
        thl_init,           # thl_par at level k_py+1 (from initial step)
        rt_init,            # rt_par at level k_py+1
        dCAPE_init,         # dCAPE/dz at level k_py+1 (for trapezoidal rule)
        jnp.bool_(False),   # done (early exit flag)
        k_py + 1,           # j_last: last fully-reached level (k_py+1 initially)
        tke_0,              # tke_at_exit (TKE before exhaustion step)
        dCAPE_init,         # dCAPE_exit_prev (dCAPE at j_last at time of exit)
        jnp.float64(0.0),   # dCAPE_exit_j (dCAPE at exit level j)
    )

    def cond_fn(state):
        j, tke, thl, rt, dCAPE_prev, done, j_last, tke_exit, dep, dej = state
        return ~done & (j < k_ub_zt_py)

    def body_fn(state):
        j, tke, thl, rt, dCAPE_prev, done, j_last, tke_exit, dep, dej = state
        # Ascending grid: j_zm = j (zm-level index same as zt-level for ascending)
        thl_new = thl_precalc_up[j] + thl * exp_mu_dzm[j]
        rt_new  = rt_precalc_up[j]  + rt  * exp_mu_dzm[j]
        thv_new = _parcel_thv(thl_new, rt_new, exner[j], p[j], thv_ds[j], Lv_coef[j],
                               saturation_formula)
        dCAPE_j   = grav_on_thvm[j] * (thv_new - thvm[j])
        CAPE_incr = 0.5 * (dCAPE_j + dCAPE_prev) * dzm[j]  # j_zm = j

        new_tke   = tke + CAPE_incr
        exhausted = new_tke <= 0.0
        newly_ex  = exhausted & ~done

        # Preserve exit info at first exhaustion
        tke_exit_out = jnp.where(newly_ex, tke, tke_exit)
        dep_out       = jnp.where(newly_ex, dCAPE_prev, dep)
        dej_out       = jnp.where(newly_ex, dCAPE_j, dej)

        # j increments only when NOT exhausted
        j_out       = jnp.where(exhausted, j, j + 1)
        tke_out     = jnp.where(exhausted, tke, new_tke)
        thl_out     = jnp.where(exhausted, thl, thl_new)
        rt_out      = jnp.where(exhausted, rt,  rt_new)
        dCAPE_out   = jnp.where(exhausted, dCAPE_prev, dCAPE_j)
        j_last_out  = jnp.where(exhausted, j_last, j)  # last fully reached = current j
        done_out    = done | exhausted

        return (j_out, tke_out, thl_out, rt_out, dCAPE_out, done_out,
                j_last_out, tke_exit_out, dep_out, dej_out)

    final = jax.lax.while_loop(cond_fn, body_fn, init_state)
    j_final, tke_final, _, _, _, done_final, j_last, tke_exit, dCAPE_exit_prev, dCAPE_exit_j = final

    exited_early = done_final  # True if TKE exhausted before reaching k_ub_zt
    # j_final = exit level (j < k_ub_zt when exited_early),
    #           or k_ub_zt when loop hit boundary (loop advanced j PAST boundary)
    # j_last  = last fully-traversed level (j_final - 1 when exited early, else k_ub_zt - 1)

    return j_last, exited_early, j_final, tke_exit, dCAPE_exit_prev, dCAPE_exit_j


def _compute_lscale_up_col(
    tke_i_col,            # (nzt,)
    thl_par_1_up,         # (nzt,) initial parcel thl at j_py=1..nzt-1 (padded at 0)
    rt_par_1_up,          # (nzt,)
    dCAPE_dz_1_up,        # (nzt,) initial dCAPE at j_py=1..nzt-1 (padded at 0)
    CAPE_incr_1_up,       # (nzt,) initial CAPE_incr at j_py=1..nzt-1 (padded at 0)
    thl_precalc_up,       # (nzm,) thl_par_j_precalc, index 1..nzt-2 valid
    rt_precalc_up,        # (nzm,)
    exp_mu_dzm,           # (nzm,)
    grav_on_thvm,         # (nzt,)
    Lv_coef,              # (nzt,)
    thv_ds,               # (nzt,)
    exner,                # (nzt,)
    p,                    # (nzt,)
    thvm,                 # (nzt,)
    dzm,                  # (nzm,)
    invrs_dzm,            # (nzm,)
    zt,                   # (nzt,)
    k_ub_zt_py,
    saturation_formula,
    nzt,
):
    """Compute Lscale_up for a single column."""
    zlmin = _ZLMIN

    # Outer scan over starting levels k = 0..nzt-3
    def outer_step(carry, k_py):
        max_alt = carry  # Lscale_up_max_alt (running max of zt + Lscale_up)

        tke_i_k = tke_i_col[k_py]
        tke_0 = tke_i_k + CAPE_incr_1_up[k_py + 1]

        # ---- Case A: TKE exhausted before reaching level k+1 ----
        # kp1_zm = k+1 (ascending), dCAPE at k+1, tke = tke_i[k]
        dCAPE_1_kp1 = dCAPE_dz_1_up[k_py + 1]
        # Avoid division by zero: if dCAPE_1_kp1 = 0, set frac_a = 0
        safe_dCAPE_a = jnp.where(jnp.abs(dCAPE_1_kp1) > 0.0, dCAPE_1_kp1, 1.0)
        frac_a = -jnp.sqrt(jnp.maximum(
            0.0, -2.0 * tke_i_k * dzm[k_py + 1] * dCAPE_1_kp1
        )) / safe_dCAPE_a

        # ---- Case B/C: TKE survives initial step — run inner while loop ----
        j_last, exited_early, j_final, tke_exit, dCAPE_exit_prev, dCAPE_exit_j = (
            _upward_inner_while(
                k_py,
                tke_0,
                thl_par_1_up[k_py + 1],
                rt_par_1_up[k_py + 1],
                dCAPE_dz_1_up[k_py + 1],
                thl_precalc_up, rt_precalc_up,
                exp_mu_dzm,
                grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
                dzm, invrs_dzm,
                zt,
                k_ub_zt_py,
                saturation_formula,
            )
        )

        # Base distance: full levels crossed (zt[j_last] - zt[k_py])
        base_dist = zt[j_last] - zt[k_py]

        # Fractional correction when exited early (TKE exhausted at j_final)
        dCAPE_diff = dCAPE_exit_j - dCAPE_exit_prev
        # Linear case: |diff| * 2 <= |sum| * eps (dCAPE/dz nearly constant)
        linear_case = (jnp.abs(dCAPE_diff) * 2.0
                       <= jnp.abs(dCAPE_exit_j + dCAPE_exit_prev) * eps)
        safe_dCAPE_j = jnp.where(jnp.abs(dCAPE_exit_j) > 0.0, dCAPE_exit_j, 1.0)
        frac_linear = -tke_exit / safe_dCAPE_j

        safe_diff = jnp.where(jnp.abs(dCAPE_diff) > 0.0, dCAPE_diff, 1.0)
        invrs_diff = 1.0 / safe_diff
        disc = (dCAPE_exit_prev ** 2
                - 2.0 * tke_exit * invrs_dzm[j_final] * dCAPE_diff)
        frac_quad = (
            - dCAPE_exit_prev * invrs_diff * dzm[j_final]
            - jnp.sqrt(jnp.maximum(0.0, disc)) * invrs_diff * dzm[j_final]
        )
        frac_inner = jnp.where(linear_case, frac_linear, frac_quad)
        frac_bc = jnp.where(exited_early, frac_inner, 0.0)

        Lscale_up_k = jnp.where(
            tke_0 > 0.0,
            zlmin + base_dist + frac_bc,
            zlmin + frac_a,
        )

        # Smooth-profile constraint: if a lower parcel can rise higher, use that
        k_alt = zt[k_py] + Lscale_up_k
        Lscale_up_k_smooth = jnp.where(k_alt < max_alt,
                                         max_alt - zt[k_py],
                                         Lscale_up_k)
        new_max_alt = jnp.where(k_alt < max_alt, max_alt, k_alt)

        return new_max_alt, Lscale_up_k_smooth

    _, Lscale_up_values = jax.lax.scan(outer_step, jnp.float64(0.0), jnp.arange(nzt - 2))
    # Lscale_up_values: shape (nzt-2,) for k=0..nzt-3
    # Top level (nzt-2 and nzt-1) get zlmin
    pad_top = jnp.full(2, _ZLMIN)
    return jnp.concatenate([Lscale_up_values, pad_top])  # (nzt,)


def _downward_inner_while(
    k_py,
    tke_0,
    thl_init, rt_init, dCAPE_init,   # parcel state at k_py-1 (initial step result)
    thl_precalc_down,                 # (nzm,) thl_par_j_precalc for downward
    rt_precalc_down,                  # (nzm,)
    exp_mu_dzm,                       # (nzm,)
    grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,  # (nzt,)
    dzm, invrs_dzm,                   # (nzm,)
    zt,                               # (nzt,)
    k_lb_zt_py,
    saturation_formula,
):
    """Inner while loop for downward parcel trajectory from k_py-2.

    Fortran:
        j = k - 2
        do while (j >= k_lb_zt)     (ascending: j >= 0 in Python)
            compute CAPE_incr at j
            if (tke - CAPE_incr <= 0) exit
            tke -= CAPE_incr; j -= 1
        end do
        Lscale_down[k] += zt[k] - zt[j+1]  (full levels crossed)
        + quadratic formula for fractional level
    """
    init_state = (
        k_py - 2,           # j: next level to test (descending)
        tke_0,
        thl_init,           # thl at k_py-1
        rt_init,
        dCAPE_init,         # dCAPE at k_py-1 (the dCAPE_j_plus_1 in Fortran)
        jnp.bool_(False),
        k_py - 1,           # j_last_reached (last fully crossed level)
        tke_0,
        dCAPE_init,         # dCAPE_exit_plus1 (= dCAPE_j_plus_1 at exit)
        jnp.float64(0.0),   # dCAPE_exit_j
    )

    def cond_fn(state):
        j, tke, thl, rt, dCAPE_plus1, done, j_last, tex, dep1, dej = state
        return ~done & (j >= k_lb_zt_py)

    def body_fn(state):
        j, tke, thl, rt, dCAPE_plus1, done, j_last, tex, dep1, dej = state
        # Ascending grid: jp1_zm = j+1
        thl_new = thl_precalc_down[j] + thl * exp_mu_dzm[j + 1]
        rt_new  = rt_precalc_down[j]  + rt  * exp_mu_dzm[j + 1]
        thv_new = _parcel_thv(thl_new, rt_new, exner[j], p[j], thv_ds[j], Lv_coef[j],
                               saturation_formula)
        dCAPE_j   = grav_on_thvm[j] * (thv_new - thvm[j])
        # Downward: CAPE_incr uses j+1 spacing (jp1_zm = j+1 ascending)
        CAPE_incr = 0.5 * (dCAPE_j + dCAPE_plus1) * dzm[j + 1]

        new_tke   = tke - CAPE_incr
        exhausted = new_tke <= 0.0
        newly_ex  = exhausted & ~done

        tex_out  = jnp.where(newly_ex, tke, tex)
        dep1_out = jnp.where(newly_ex, dCAPE_plus1, dep1)
        dej_out  = jnp.where(newly_ex, dCAPE_j, dej)

        j_out      = jnp.where(exhausted, j, j - 1)
        tke_out    = jnp.where(exhausted, tke, new_tke)
        thl_out    = jnp.where(exhausted, thl, thl_new)
        rt_out     = jnp.where(exhausted, rt,  rt_new)
        dCAPE_out  = jnp.where(exhausted, dCAPE_plus1, dCAPE_j)
        j_last_out = jnp.where(exhausted, j_last, j)  # last fully crossed = current j
        done_out   = done | exhausted

        return (j_out, tke_out, thl_out, rt_out, dCAPE_out, done_out,
                j_last_out, tex_out, dep1_out, dej_out)

    final = jax.lax.while_loop(cond_fn, body_fn, init_state)
    j_final, _, _, _, _, done_final, j_last, tke_exit, dCAPE_exit_plus1, dCAPE_exit_j = final

    exited_early = done_final
    return j_last, exited_early, j_final, tke_exit, dCAPE_exit_plus1, dCAPE_exit_j


def _compute_lscale_down_col(
    tke_i_col,
    thl_par_1_down, rt_par_1_down, dCAPE_dz_1_down, CAPE_incr_1_down,
    thl_precalc_down, rt_precalc_down,
    exp_mu_dzm, grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
    dzm, invrs_dzm, zt,
    k_ub_zt_py, k_lb_zt_py,
    saturation_formula, nzt,
):
    """Compute Lscale_down for a single column.

    Outer loop: k from k_ub_zt down to k_lb_zt+1 = nzt-1..1 (Python).
    Scanned as i=0..nzt-2, k_py = nzt-1-i.
    """
    zlmin = _ZLMIN

    def outer_step(carry, i):
        min_alt = carry  # Lscale_down_min_alt (running min of zt - Lscale_down)
        k_py = nzt - 1 - i  # descend from top

        tke_i_k = tke_i_col[k_py]
        # Downward initial: CAPE_incr at k-1 (Fortran k-1 = Python k_py-1)
        tke_0 = tke_i_k - CAPE_incr_1_down[k_py - 1]

        # ---- Case A: TKE exhausted before reaching level k-1 ----
        # k_zm = k (ascending: k_zm = k for Lscale_down case, Fortran kp1_zm)
        # Fortran uses k_zm for the boundary term: dzm[k_zm] where k_zm = k (ascending)
        dCAPE_1_km1 = dCAPE_dz_1_down[k_py - 1]
        safe_dCAPE_a = jnp.where(jnp.abs(dCAPE_1_km1) > 0.0, dCAPE_1_km1, 1.0)
        frac_a = jnp.sqrt(jnp.maximum(
            0.0, 2.0 * tke_i_k * dzm[k_py] * dCAPE_1_km1
        )) / safe_dCAPE_a

        # ---- Case B/C: TKE survives initial step ----
        j_last, exited_early, j_final, tke_exit, dCAPE_exit_plus1, dCAPE_exit_j = (
            _downward_inner_while(
                k_py,
                tke_0,
                thl_par_1_down[k_py - 1],
                rt_par_1_down[k_py - 1],
                dCAPE_dz_1_down[k_py - 1],
                thl_precalc_down, rt_precalc_down,
                exp_mu_dzm,
                grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
                dzm, invrs_dzm, zt,
                k_lb_zt_py,
                saturation_formula,
            )
        )

        base_dist = zt[k_py] - zt[j_last]

        dCAPE_diff = dCAPE_exit_j - dCAPE_exit_plus1
        linear_case = (jnp.abs(dCAPE_diff) * 2.0
                       <= jnp.abs(dCAPE_exit_j + dCAPE_exit_plus1) * eps)
        safe_dCAPE_j = jnp.where(jnp.abs(dCAPE_exit_j) > 0.0, dCAPE_exit_j, 1.0)
        frac_linear = tke_exit / safe_dCAPE_j

        safe_diff = jnp.where(jnp.abs(dCAPE_diff) > 0.0, dCAPE_diff, 1.0)
        invrs_diff = 1.0 / safe_diff
        disc = (dCAPE_exit_plus1 ** 2
                + 2.0 * tke_exit * invrs_dzm[j_final + 1] * dCAPE_diff)
        frac_quad = (
            - dCAPE_exit_plus1 * invrs_diff * dzm[j_final + 1]
            + jnp.sqrt(jnp.maximum(0.0, disc)) * invrs_diff * dzm[j_final + 1]
        )
        frac_inner = jnp.where(linear_case, frac_linear, frac_quad)
        frac_bc = jnp.where(exited_early, frac_inner, 0.0)

        Lscale_down_k = jnp.where(
            tke_0 > 0.0,
            zlmin + base_dist + frac_bc,
            zlmin + frac_a,
        )

        k_alt = zt[k_py] - Lscale_down_k
        Lscale_down_k_smooth = jnp.where(k_alt > min_alt,
                                           zt[k_py] - min_alt,
                                           Lscale_down_k)
        new_min_alt = jnp.where(k_alt > min_alt, min_alt, k_alt)

        return new_min_alt, (k_py, Lscale_down_k_smooth)

    init_min_alt = zt[k_ub_zt_py]  # = zt[nzt-1] (top)
    _, (k_indices, Lscale_down_values) = jax.lax.scan(
        outer_step, init_min_alt, jnp.arange(nzt - 1)
    )
    # k_indices: shape (nzt-1,) values nzt-1..1
    # Build Lscale_down array by scatter
    Lscale_down_col = jnp.full(nzt, _ZLMIN)
    Lscale_down_col = Lscale_down_col.at[k_indices].set(Lscale_down_values)
    # k=0 (bottom) keeps zlmin
    return Lscale_down_col


def compute_mixing_length_jax(
    thvm, thlm, rtm, em, Lscale_max, p_in_Pa, exner, thv_ds,
    mu, lmin, saturation_formula, l_implemented, gr,
):
    """JAX port of mixing_length.F90:compute_mixing_length.

    Golaz et al. (2002) nonlocal parcel length scale.
    Ascending grid only (grid_dir_indx = +1).

    Inputs (all float64):
      thvm, thlm, rtm, p_in_Pa, exner, thv_ds: (ngrdcol, nzt)
      em: (ngrdcol, nzm)
      Lscale_max: (ngrdcol,)
      mu: (ngrdcol,) entrainment rate [1/m]
      lmin: scalar [m]
      saturation_formula: int
      l_implemented: bool (True = within host model)
      gr: grid namedtuple
    Returns:
      Lscale, Lscale_up, Lscale_down: (ngrdcol, nzt)
    """
    ngrdcol, nzt = thvm.shape
    nzm = em.shape[1]
    k_ub_zt_py = gr.k_ub_zt   # = nzt - 1 = 132
    k_lb_zt_py = gr.k_lb_zt   # = 0

    # ---- Shared precomputations ----
    tke_i = zm2zt_jax(em, gr)                             # (ngrdcol, nzt)
    grav_on_thvm = grav / thvm                            # (ngrdcol, nzt)
    Lv_coef = Lv / (exner * Cp) - ep2 * thv_ds           # (ngrdcol, nzt)

    # zm-level precomputes (shape nzm)
    exp_mu_dzm  = jnp.exp(-mu[:, None] * gr.dzm)          # (ngrdcol, nzm)
    invrs_dzm_on_mu = gr.invrs_dzm / mu[:, None]           # (ngrdcol, nzm)
    entrain_coef = (1.0 - exp_mu_dzm) * invrs_dzm_on_mu   # (ngrdcol, nzm)

    # ---- Upward precalculations ----
    # thl_par_j_precalc_up[i, j_py] for j_py = 1..nzt-2 (ascending: j_zm = j_py)
    thl_mid = thlm[:, 1:nzt - 1]         # (ngrdcol, nzt-2)
    thl_blw = thlm[:, 0:nzt - 2]
    rt_mid  = rtm[:, 1:nzt - 1]
    rt_blw  = rtm[:, 0:nzt - 2]
    emu_up  = exp_mu_dzm[:, 1:nzt - 1]   # j_zm = j_py = 1..nzt-2
    ec_up   = entrain_coef[:, 1:nzt - 1]

    thl_precalc_up_int = thl_mid - thl_blw * emu_up - (thl_mid - thl_blw) * ec_up
    rt_precalc_up_int  = rt_mid  - rt_blw  * emu_up - (rt_mid  - rt_blw)  * ec_up
    _pad0   = jnp.zeros((ngrdcol, 1))
    _pad2   = jnp.zeros((ngrdcol, 2))
    # Shape (ngrdcol, nzm=nzt+1): index 0 unused, 1..nzt-2 valid, nzt-1..nzt unused
    thl_precalc_up = jnp.concatenate([_pad0, thl_precalc_up_int, _pad2], axis=1)
    rt_precalc_up  = jnp.concatenate([_pad0, rt_precalc_up_int,  _pad2], axis=1)

    # Upward initial step: thl_par_1_up[i, j_py] for j_py = 1..nzt-1
    # j_zm = j_py (ascending), entrain_coef[j_py] used
    ec_init_up  = entrain_coef[:, 1:nzt]  # (ngrdcol, nzt-1)
    thl_diff_up = thlm[:, 1:] - thlm[:, :-1]
    rt_diff_up  = rtm[:, 1:]  - rtm[:, :-1]
    thl_par_1_up_int = thlm[:, 1:] - thl_diff_up * ec_init_up  # (ngrdcol, nzt-1)
    rt_par_1_up_int  = rtm[:, 1:]  - rt_diff_up  * ec_init_up

    tl_par_1_up_int = thl_par_1_up_int * exner[:, 1:]           # (ngrdcol, nzt-1)
    rsat_1_up_int   = sat_mixrat_liq_jax(p_in_Pa[:, 1:], tl_par_1_up_int, saturation_formula)
    tl_sqd_up       = tl_par_1_up_int ** 2
    s_1_up          = ((rt_par_1_up_int - rsat_1_up_int) * tl_sqd_up
                       / (tl_sqd_up + _LV2_COEF * rsat_1_up_int))
    rc_1_up         = jnp.maximum(s_1_up, zero_threshold)
    thv_1_up        = (thl_par_1_up_int + ep1 * thv_ds[:, 1:] * rt_par_1_up_int
                       + Lv_coef[:, 1:] * rc_1_up)
    dCAPE_dz_1_up_int  = grav_on_thvm[:, 1:] * (thv_1_up - thvm[:, 1:])   # (ngrdcol, nzt-1)
    CAPE_incr_1_up_int = 0.5 * dCAPE_dz_1_up_int * gr.dzm[:, 1:nzt]      # j_zm = j_py

    thl_par_1_up  = jnp.concatenate([_pad0, thl_par_1_up_int], axis=1)  # (ngrdcol, nzt)
    rt_par_1_up   = jnp.concatenate([_pad0, rt_par_1_up_int],  axis=1)
    dCAPE_dz_1_up = jnp.concatenate([_pad0, dCAPE_dz_1_up_int], axis=1)
    CAPE_incr_1_up= jnp.concatenate([_pad0, CAPE_incr_1_up_int], axis=1)

    # ---- Downward precalculations ----
    # thl_par_j_precalc_down[i, j_py] for j_py = 0..nzt-2 (ascending: jp1_zm = j+1)
    thl_abv  = thlm[:, 1:]             # (ngrdcol, nzt-1), j+1 = j_py+1
    thl_at_j = thlm[:, :-1]            # j_py = 0..nzt-2
    rt_abv   = rtm[:, 1:]
    rt_at_j  = rtm[:, :-1]
    emu_dn   = exp_mu_dzm[:, 1:nzt]    # jp1_zm = j+1, j_py = 0..nzt-2
    ec_dn    = entrain_coef[:, 1:nzt]

    thl_precalc_dn_int = thl_at_j - thl_abv * emu_dn - (thl_at_j - thl_abv) * ec_dn
    rt_precalc_dn_int  = rt_at_j  - rt_abv  * emu_dn - (rt_at_j  - rt_abv)  * ec_dn
    # Shape (ngrdcol, nzm=nzt+1): index 0..nzt-2 valid, nzt-1 unused
    thl_precalc_down = jnp.concatenate([thl_precalc_dn_int, jnp.zeros((ngrdcol, 2))], axis=1)
    rt_precalc_down  = jnp.concatenate([rt_precalc_dn_int,  jnp.zeros((ngrdcol, 2))], axis=1)

    # Downward initial step: thl_par_1_down[i, j_py] for j_py = 0..nzt-2
    # jp1_zm = j+1 (ascending), entrain_coef[j+1] used
    ec_init_dn   = entrain_coef[:, 1:nzt]   # jp1_zm = j_py+1, j_py = 0..nzt-2
    thl_diff_dn  = thlm[:, :-1] - thlm[:, 1:]
    rt_diff_dn   = rtm[:, :-1]  - rtm[:, 1:]
    thl_par_1_dn_int = thlm[:, :-1] - thl_diff_dn * ec_init_dn   # (ngrdcol, nzt-1)
    rt_par_1_dn_int  = rtm[:, :-1]  - rt_diff_dn  * ec_init_dn

    tl_par_1_dn_int = thl_par_1_dn_int * exner[:, :-1]
    rsat_1_dn_int   = sat_mixrat_liq_jax(p_in_Pa[:, :-1], tl_par_1_dn_int, saturation_formula)
    tl_sqd_dn       = tl_par_1_dn_int ** 2
    s_1_dn          = ((rt_par_1_dn_int - rsat_1_dn_int) * tl_sqd_dn
                       / (tl_sqd_dn + _LV2_COEF * rsat_1_dn_int))
    rc_1_dn         = jnp.maximum(s_1_dn, zero_threshold)
    thv_1_dn        = (thl_par_1_dn_int + ep1 * thv_ds[:, :-1] * rt_par_1_dn_int
                       + Lv_coef[:, :-1] * rc_1_dn)
    dCAPE_dz_1_dn_int  = grav_on_thvm[:, :-1] * (thv_1_dn - thvm[:, :-1])
    CAPE_incr_1_dn_int = 0.5 * dCAPE_dz_1_dn_int * gr.dzm[:, 1:nzt]   # jp1_zm = j_py+1

    # Pad: index nzt-1 unused (level above top)
    thl_par_1_down  = jnp.concatenate([thl_par_1_dn_int,  _pad0], axis=1)  # (ngrdcol, nzt)
    rt_par_1_down   = jnp.concatenate([rt_par_1_dn_int,   _pad0], axis=1)
    dCAPE_dz_1_down = jnp.concatenate([dCAPE_dz_1_dn_int, _pad0], axis=1)
    CAPE_incr_1_down= jnp.concatenate([CAPE_incr_1_dn_int,_pad0], axis=1)

    # ---- Per-column upward/downward computation (vmap over ngrdcol) ----
    # Convert to JAX arrays (needed for dynamic indexing with traced indices)
    zt_single  = jnp.asarray(gr.zt[0])
    dzm_single = jnp.asarray(gr.dzm[0])
    invrs_dzm_single = jnp.asarray(gr.invrs_dzm[0])

    def col_lscale(i):
        Lscale_up_col = _compute_lscale_up_col(
            tke_i[i], thl_par_1_up[i], rt_par_1_up[i],
            dCAPE_dz_1_up[i], CAPE_incr_1_up[i],
            thl_precalc_up[i], rt_precalc_up[i],
            exp_mu_dzm[i], grav_on_thvm[i], Lv_coef[i],
            thv_ds[i], exner[i], p_in_Pa[i], thvm[i],
            dzm_single, invrs_dzm_single, zt_single,
            k_ub_zt_py, saturation_formula, nzt,
        )
        Lscale_down_col = _compute_lscale_down_col(
            tke_i[i], thl_par_1_down[i], rt_par_1_down[i],
            dCAPE_dz_1_down[i], CAPE_incr_1_down[i],
            thl_precalc_down[i], rt_precalc_down[i],
            exp_mu_dzm[i], grav_on_thvm[i], Lv_coef[i],
            thv_ds[i], exner[i], p_in_Pa[i], thvm[i],
            dzm_single, invrs_dzm_single, zt_single,
            k_ub_zt_py, k_lb_zt_py, saturation_formula, nzt,
        )
        return Lscale_up_col, Lscale_down_col

    Lscale_up_all   = jnp.stack([col_lscale(i)[0] for i in range(ngrdcol)])
    Lscale_down_all = jnp.stack([col_lscale(i)[1] for i in range(ngrdcol)])

    # ---- Apply lminh floor and Lscale_max cap ----
    invrs_sfclyr = 1.0 / _LSCALE_SFCLYR_DEPTH
    if l_implemented:
        # Host model: surface layer above *ground* (bottom zm level)
        zm_sfc = gr.zm[:, gr.k_lb_zm]  # (ngrdcol,)
        lminh = (jnp.maximum(0.0, _LSCALE_SFCLYR_DEPTH - (gr.zt - zm_sfc[:, None]))
                 * lmin * invrs_sfclyr)
    else:
        # Standalone: above mean sea level
        lminh = (jnp.maximum(0.0, _LSCALE_SFCLYR_DEPTH - gr.zt)
                 * lmin * invrs_sfclyr)

    Lscale_up   = jnp.maximum(lminh, Lscale_up_all)
    Lscale_down = jnp.maximum(lminh, Lscale_down_all)
    Lscale = jnp.sqrt(Lscale_up * Lscale_down)

    # Upper boundary: Lscale[k_ub] = Lscale[k_ub - 1]
    Lscale = Lscale.at[:, k_ub_zt_py].set(Lscale[:, k_ub_zt_py - 1])

    # Global cap
    Lscale = jnp.minimum(Lscale, Lscale_max[:, None])

    return Lscale, Lscale_up, Lscale_down


def calc_lscale_directly_jax(
    thvm, thlm, rtm, em, Lscale_max, p_in_Pa, exner, thv_ds,
    clubb_params, lmin, saturation_formula, l_implemented, gr,
):
    """JAX port of mixing_length.F90:calc_Lscale_directly.

    l_avg_Lscale = False (Fortran compile-time constant).
    Calls compute_mixing_length_jax once with mean values.

    Inputs:
      clubb_params: (ngrdcol, nparams+1) — 1-indexed
    Returns:
      Lscale, Lscale_up, Lscale_down: (ngrdcol, nzt)
    """
    mu = clubb_params[:, imu]   # imu = 60 (1-indexed, column 0 unused)
    return compute_mixing_length_jax(
        thvm, thlm, rtm, em, Lscale_max, p_in_Pa, exner, thv_ds,
        mu, lmin, saturation_formula, l_implemented, gr,
    )
