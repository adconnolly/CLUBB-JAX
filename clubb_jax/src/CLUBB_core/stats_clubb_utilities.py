"""JAX/Python port of stats_clubb_utilities.F90 — per-timestep statistics accumulation.

Mirrors clubb_release/src/CLUBB_core/stats_clubb_utilities.F90:stats_accumulate, which writes every
diagnostic/prognostic CLUBB variable to the stats output each sampled step. The NetCDF file machinery
(the StatsWriter) lives in io/stats_writer.py (stats_netcdf.F90); this module is the accumulation logic
that computes the derived diagnostics and calls sw.update(...) for each variable.

Called by advance_clubb_core only when sw.l_sample is True.
"""

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.tracer_numpy import _asarray, _xp
from clubb_jax.src.CLUBB_core.constants_clubb import (
    Cp, Lv, eps, rc_tol as _RC_TOL, cloud_frac_min as _CLOUD_FRAC_MIN,
)
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_ice
from clubb_jax.src.CLUBB_core.advance_helper_module import vertical_avg, vertical_integral
from clubb_jax.src.CLUBB_core.numerical_check import calculate_spurious_source


def stats_accumulate(sw, *, nzm, nzt, ngrdcol, dt, gr,
                          l_implemented, l_host_applies_sfc_fluxes,
                          l_stability_correct_tau_zm,
                          um, vm, upwp, vpwp, up2, vp2,
                          thlm, rtm, thlm_before, rtm_before,
                          thlm_forcing, rtm_forcing,
                          wpthlp_sfc, wprtp_sfc, wprtp, wpthlp,
                          wp2, wp3, rtp2, rtp3, thlp2, thlp3, rtpthlp,
                          p_in_Pa, exner, rho, rho_zm, rho_ds_zm, rho_ds_zt,
                          thv_ds_zm, thv_ds_zt, wm_zt, wm_zm, rcm, cloud_frac,
                          ice_supersat_frac,
                          thvm, ug, vg, ddzt_umvm_sqd, stability_correction,
                          Kh_zt, rsat, Kh_zm, em, wp3_on_wp2, wp3_on_wp2_zt,
                          sclrm, sclrp2, sclrprtp, sclrpthlp, sclrm_forcing,
                          wpsclrp, wpedsclrp, edsclrm, edsclrm_forcing,
                          saturation_formula, sclr_dim=0, edsclr_dim=0):
    """Python equivalent of Fortran stats_accumulate in stats_clubb_utilities.F90.

    Calls sw.update(name, val) for every variable written by the Fortran routine.
    Only called when sw.l_sample is True.
    """
    if sw is None:
        return

    # Compute grid layer thicknesses
    grid_dir = float(gr.grid_dir)
    # dzt[i,k] = (zm[i,k+1] - zm[i,k]) * grid_dir  (shape: ngrdcol, nzt)
    dzt = (_asarray(gr.zm)[:, 1:] - _asarray(gr.zm)[:, :-1]) * grid_dir
    # dzm: shape (ngrdcol, nzm)
    dzm = _asarray(gr.dzm) * grid_dir

    # ---- Computed diagnostics ----

    # T_in_K: absolute temperature  T = thlm * exner + (Lv/Cp) * rcm
    if sw.var_on_stats_list("T_in_K") or sw.var_on_stats_list("rsati"):
        T_in_K_acc = thlm * exner + (Lv / Cp) * rcm
        sw.update("T_in_K", T_in_K_acc)
        if sw.var_on_stats_list("rsati"):
            _rsati_acc = _asarray(
                sat_mixrat_ice(jnp.asarray(p_in_Pa), jnp.asarray(T_in_K_acc)),
                dtype=np.float64,
            )
            sw.update("rsati", _rsati_acc)

    # rcm_in_cloud
    if sw.var_on_stats_list("rcm_in_cloud"):
        # Guard the denominator so the cloud_frac==0 elements don't produce a nan in the unused branch
        # (the bare `rcm / cloud_frac` divided by zero everywhere cloud_frac==0 → RuntimeWarning + nan that
        # also poisons reverse-mode gradients through this diagnostic). where-select stays forward-identical.
        cf_safe = _xp.where(cloud_frac > _CLOUD_FRAC_MIN, cloud_frac, 1.0)
        rcm_in_cloud = _xp.where(cloud_frac > _CLOUD_FRAC_MIN, rcm / cf_safe, rcm)
        sw.update("rcm_in_cloud", rcm_in_cloud)

    # shear (zm-level)
    if sw.var_on_stats_list("shear"):
        shear = np.zeros((ngrdcol, nzm), dtype=np.float64)
        # Interior zm levels k=1..nzm-2
        um_diff = um[:, 1:] - um[:, :-1]  # (ngrdcol, nzm-2)
        vm_diff = vm[:, 1:] - vm[:, :-1]
        invrs_dzm_int = _asarray(gr.invrs_dzm)[:, 1:-1]
        shear[:, 1:-1] = (- upwp[:, 1:-1] * um_diff * invrs_dzm_int
                          - vpwp[:, 1:-1] * vm_diff * invrs_dzm_int)
        sw.update("shear", shear)

    # zt-level profile variables (unconditional)
    sw.update("thlm", thlm)
    sw.update("thvm", thvm)
    sw.update("rtm", rtm)
    sw.update("rcm", rcm)
    # Cloud diagnostics (Fortran writes these; their absence makes the diagnostic
    # comparison show a spurious 0 for every cloud case and hides real cloud-PDF
    # divergences during debugging). cloud_frac/ice_supersat_frac are the physics
    # state values, so this also lets the compare verify them.
    sw.update("cloud_frac", cloud_frac)
    sw.update("ice_supersat_frac", ice_supersat_frac)
    sw.update("um", um)
    sw.update("vm", vm)
    sw.update("wm_zt", wm_zt)
    sw.update("ug", ug)
    sw.update("vg", vg)
    sw.update("p_in_Pa", p_in_Pa)
    sw.update("exner", exner)
    sw.update("rho_ds_zt", rho_ds_zt)
    sw.update("thv_ds_zt", thv_ds_zt)
    sw.update("wp3", wp3)
    sw.update("Kh_zt", Kh_zt)
    sw.update("rho", rho)
    sw.update("rsat", rsat)
    sw.update("thlp3", thlp3)
    sw.update("rtp3", rtp3)
    sw.update("wp3_on_wp2_zt", wp3_on_wp2_zt)

    # Scalar species (zt)
    for s in range(sclr_dim):
        sw.update(f"sclr{s+1}m", sclrm[:, :, s])
        sw.update(f"sclr{s+1}m_f", sclrm_forcing[:, :, s])
    for e in range(edsclr_dim):
        sw.update(f"edsclr{e+1}m", edsclrm[:, :, e])
        sw.update(f"edsclr{e+1}m_f", edsclrm_forcing[:, :, e])

    # zm-level profile variables (unconditional)
    sw.update("wm_zm", wm_zm)
    sw.update("ddzt_umvm_sqd", ddzt_umvm_sqd)
    sw.update("wp2", wp2)
    sw.update("rtp2", rtp2)
    sw.update("thlp2", thlp2)
    sw.update("rtpthlp", rtpthlp)
    sw.update("wprtp", wprtp)
    sw.update("wpthlp", wpthlp)
    if l_stability_correct_tau_zm:
        sw.update("stability_correction", stability_correction)
    sw.update("Kh_zm", Kh_zm)
    sw.update("upwp", upwp)
    sw.update("vpwp", vpwp)
    sw.update("vp2", vp2)
    sw.update("up2", up2)
    sw.update("rho_zm", rho_zm)
    sw.update("rho_ds_zm", rho_ds_zm)
    sw.update("thv_ds_zm", thv_ds_zm)
    sw.update("em", em)
    sw.update("wp3_on_wp2", wp3_on_wp2)
    # CFL number: wp3_on_wp2 * dt / dzm
    if sw.var_on_stats_list("wp3_on_wp2_cfl_num"):
        sw.update("wp3_on_wp2_cfl_num", wp3_on_wp2 * dt / dzm)

    # Scalar species (zm)
    for s in range(sclr_dim):
        sw.update(f"sclr{s+1}p2", sclrp2[:, :, s])
        sw.update(f"sclr{s+1}prtp", sclrprtp[:, :, s])
        sw.update(f"sclr{s+1}pthlp", sclrpthlp[:, :, s])
        sw.update(f"wpsclr{s+1}p", wpsclrp[:, :, s])
    for e in range(edsclr_dim):
        sw.update(f"wpedsclr{e+1}p", wpedsclrp[:, :, e])

    # Surface / scalar variables
    zt_arr = _asarray(gr.zt)  # (ngrdcol, nzt)

    # cc: max cloud fraction over zt column
    for i in range(ngrdcol):
        sw.update_col("cc", float(_xp.max(cloud_frac[i, :])), icol=i)

    # z_cloud_base
    if sw.var_on_stats_list("z_cloud_base"):
        for i in range(ngrdcol):
            k = 0
            while rcm[i, k] < _RC_TOL and k < nzt - 1:
                k += 1
            if k == 0:
                z_cb = float(zt_arr[i, 0])
            elif k < nzt - 1:
                # linear interpolation: where rcm crosses rc_tol
                rcm_k, rcm_km1 = float(rcm[i, k]), float(rcm[i, k - 1])
                zt_k, zt_km1 = float(zt_arr[i, k]), float(zt_arr[i, k - 1])
                if abs(rcm_k - rcm_km1) > 1.0e-30:
                    z_cb = ((_RC_TOL - rcm_km1) / (rcm_k - rcm_km1)
                            * (zt_k - zt_km1) + zt_km1)
                else:
                    z_cb = zt_k
            else:
                z_cb = -10.0
            sw.update_col("z_cloud_base", z_cb, icol=i)

    # lwp, vwp (optional)
    if sw.var_on_stats_list("lwp"):
        for i in range(ngrdcol):
            sw.update_col("lwp",
                           vertical_integral(rho_ds_zt[i], rcm[i], dzt[i]),
                           icol=i)
    if sw.var_on_stats_list("vwp"):
        for i in range(ngrdcol):
            sw.update_col("vwp",
                           vertical_integral(rho_ds_zt[i], rtm[i] - rcm[i], dzt[i]),
                           icol=i)

    # Density-weighted vertical averages
    for i in range(ngrdcol):
        sw.update_col("thlm_vert_avg",
                       vertical_avg(rho_ds_zt[i], thlm[i], dzt[i]), icol=i)
        sw.update_col("rtm_vert_avg",
                       vertical_avg(rho_ds_zt[i], rtm[i], dzt[i]), icol=i)
        sw.update_col("um_vert_avg",
                       vertical_avg(rho_ds_zt[i], um[i], dzt[i]), icol=i)
        sw.update_col("vm_vert_avg",
                       vertical_avg(rho_ds_zt[i], vm[i], dzt[i]), icol=i)
        sw.update_col("wp2_vert_avg",
                       vertical_avg(rho_ds_zm[i], wp2[i], dzm[i]), icol=i)
        sw.update_col("up2_vert_avg",
                       vertical_avg(rho_ds_zm[i], up2[i], dzm[i]), icol=i)
        sw.update_col("vp2_vert_avg",
                       vertical_avg(rho_ds_zm[i], vp2[i], dzm[i]), icol=i)
        sw.update_col("rtp2_vert_avg",
                       vertical_avg(rho_ds_zm[i], rtp2[i], dzm[i]), icol=i)
        sw.update_col("thlp2_vert_avg",
                       vertical_avg(rho_ds_zm[i], thlp2[i], dzm[i]), icol=i)

    # Normalized total variation
    if sw.var_on_stats_list("tot_vartn_normlzd_rtm"):
        for i in range(ngrdcol):
            span = abs(rtm[i, -1] - rtm[i, 0])
            if span < eps:
                val = -999.0
            else:
                val = float(_xp.sum(_xp.abs(rtm[i, 1:] - rtm[i, :-1])) / span)
            sw.update_col("tot_vartn_normlzd_rtm", val, icol=i)

    if sw.var_on_stats_list("tot_vartn_normlzd_thlm"):
        for i in range(ngrdcol):
            span = abs(thlm[i, -1] - thlm[i, 0])
            if span < eps:
                val = -999.0
            else:
                val = float(_xp.sum(_xp.abs(thlm[i, 1:] - thlm[i, :-1])) / span)
            sw.update_col("tot_vartn_normlzd_thlm", val, icol=i)

    if sw.var_on_stats_list("tot_vartn_normlzd_wprtp"):
        for i in range(ngrdcol):
            span = abs(wprtp[i, -1] - wprtp[i, 0])
            if span < eps:
                val = -999.0
            else:
                val = float(_xp.sum(_xp.abs(wprtp[i, 1:] - wprtp[i, :-1])) / span)
            sw.update_col("tot_vartn_normlzd_wprtp", val, icol=i)

    # Spurious source (rtm and thlm conservation check)
    k_ub = int(gr.k_ub_zm)  # upper boundary zm index (Python 0-based)
    k_lb = int(gr.k_lb_zm)  # lower boundary zm index (Python 0-based)
    for i in range(ngrdcol):
        if (l_implemented or
                (np.all(_xp.abs(wm_zt[i]) < eps) and np.all(_xp.abs(wm_zm[i]) < eps))):
            rtm_flux_top = float(rho_ds_zm[i, k_ub] * wprtp[i, k_ub])
            if not l_host_applies_sfc_fluxes:
                rtm_flux_sfc = float(rho_ds_zm[i, k_lb] * wprtp_sfc[i])
            else:
                rtm_flux_sfc = 0.0
            rtm_int_before = vertical_integral(rho_ds_zt[i], rtm_before[i], dzt[i])
            rtm_int_after = vertical_integral(rho_ds_zt[i], rtm[i], dzt[i])
            rtm_int_forcing = vertical_integral(rho_ds_zt[i], rtm_forcing[i], dzt[i])
            rtm_spur = calculate_spurious_source(
                rtm_int_after, rtm_int_before, rtm_flux_top, rtm_flux_sfc,
                rtm_int_forcing, dt)

            thlm_flux_top = float(rho_ds_zm[i, k_ub] * wpthlp[i, k_ub])
            if not l_host_applies_sfc_fluxes:
                thlm_flux_sfc = float(rho_ds_zm[i, k_lb] * wpthlp_sfc[i])
            else:
                thlm_flux_sfc = 0.0
            thlm_int_before = vertical_integral(rho_ds_zt[i], thlm_before[i], dzt[i])
            thlm_int_after = vertical_integral(rho_ds_zt[i], thlm[i], dzt[i])
            thlm_int_forcing = vertical_integral(rho_ds_zt[i], thlm_forcing[i], dzt[i])
            thlm_spur = calculate_spurious_source(
                thlm_int_after, thlm_int_before, thlm_flux_top, thlm_flux_sfc,
                thlm_int_forcing, dt)
        else:
            rtm_spur = -9999.0
            thlm_spur = -9999.0
        sw.update_col("rtm_spur_src", rtm_spur, icol=i)
        sw.update_col("thlm_spur_src", thlm_spur, icol=i)



__all__ = ["stats_accumulate"]
