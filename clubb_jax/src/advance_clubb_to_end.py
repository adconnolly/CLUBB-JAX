"""CLUBB time-loop driver utilities.

Mirrors the timestep loop of the Fortran `advance_clubb_to_end` subroutine (clubb_driver.F90): per step it
applies forcings, adds the micro/radiation tendencies, calls the closure advance (advance_clubb_core), then
radiation and microphysics, and accumulates stats. Split out from the previous monolithic Python driver file.
"""

import gc

import numpy as np

from clubb_python import clubb_api
from clubb_jax.src.CLUBB_core import advance_clubb_core_module
from clubb_jax.src.CLUBB_core.advance_helper_module import calculate_thlp2_rad
from clubb_jax.src.CLUBB_core.calc_pressure import calculate_thvm
from clubb_jax.src.CLUBB_core.jax_stats_bridge import JaxStats
from clubb_jax.src.Benchmark_cases.prescribe_forcings import (
    prescribe_forcings_arm,
    prescribe_forcings_generic,
)
from clubb_jax.src.Microphys.microphys_driver import calc_microphys_scheme_tendcies
from clubb_jax.src.derived_types.converters import (
    err_info_from_api,
    err_info_to_api,
)


def _err_code_summary(err_info) -> str:
    if err_info is None:
        return "err_info=None"
    err_info = err_info_from_api(err_info)
    err_code = getattr(err_info, "err_code", None)
    reason_code = getattr(err_info, "reason_code", None)
    reason_messages = err_info.reason_messages_host()
    pieces = []
    if err_code is None:
        pieces.append("err_code=None")
    else:
        pieces.append(f"err_code={err_code}")
    if reason_code is not None:
        pieces.append(f"reason_code={reason_code}")
    if reason_messages:
        pieces.append("message=" + " | ".join(reason_messages))
    return "; ".join(pieces)


def advance_clubb_to_end(state: dict, l_stdout: bool = True, max_steps: int | None = None):
    """Run the CLUBB time loop."""

    dt_main = state['dt_main']
    dt_rad = state['dt_rad']
    time_initial = state['time_initial']
    ifinal = state['ifinal']
    l_stats = state['l_stats']

    rad_interval = int(dt_rad / dt_main)
    n_steps = ifinal if max_steps is None else min(ifinal, max_steps)

    _GC_INTERVAL = 10          # force a cyclic-GC pass every N sampled steps (Iter325 OOM fix)
    _samples_since_gc = 0

    for itime_idx in range(n_steps):
        itime = itime_idx + 1
        time_current = time_initial + (itime - 1) * dt_main

        # ── Stats: begin timestep ───────────────────────────────────────
        l_sample, l_last_sample = _begin_timestep_stats(state, itime_idx)

        # ── Compute thvm ────────────────────────────────────────────────
        _calculate_thvm(state)

        # ── Prescribe forcings (case-specific) ──────────────────────────
        _prescribe_forcings(state, itime, l_sample=l_sample)

        # ── Add microphysical/radiative tendencies to forcings ─────────
        state['rtm_forcing'] = state['rtm_forcing'] + state['rcm_mc']
        state['thlm_forcing'] = state['thlm_forcing'] + state['thlm_mc'] + state['radht']

        # Morrison microphysics source (computed at the END of the previous step; absent on step 1):
        # rtm_forcing += rcm_mc + rvm_mc (vapor+cloud water), thlm_forcing += thlm_mc.
        if state.get('_morr_rcm_mc') is not None:
            state['rtm_forcing'] = (
                state['rtm_forcing'] + state['_morr_rcm_mc'] + state['_morr_rvm_mc']
            )
            state['thlm_forcing'] = state['thlm_forcing'] + state['_morr_thlm_mc']

        # KK second-moment microphysics source (clubb_driver.F90:3348-3353): *_forcing += *_mc, on zm.
        # Computed at the END of the previous step (advance_kk_microphysics); absent on step 1.
        for _f in ('wprtp', 'wpthlp', 'rtp2', 'thlp2', 'rtpthlp'):
            _mc = state.get('_kk_' + _f + '_mc')
            if _mc is not None:
                state[_f + '_forcing'] = state[_f + '_forcing'] + _mc

        # ── Radiation contribution to thlp2 ─────────────────────────────
        _calculate_thlp2_rad(state)

        # ── Advance CLUBB core ──────────────────────────────────────────
        state['l_sample'] = l_sample
        _advance_clubb_core(state)
        if l_stats:
            state['_jax_stats'].to_api()

        # ── Radiation ───────────────────────────────────────────────────
        l_rad_itime = (itime % rad_interval == 0) or (itime == 1)

        if l_rad_itime:
            _advance_radiation(
                state=state,
                time_current=time_current,
                l_sample=(l_stats and l_sample),
            )

        # ── Cloud-droplet sedimentation (clubb_driver.F90:3702-3721) ───
        # Computes the rcm/thlm microphysics tendencies (for next step's forcings)
        # from the post-advance cloud. Only the cloud-sed-only path (no full
        # microphysics) is supported; rcm_mc/thlm_mc are reset to the sed term.
        if state.get('l_cloud_sed', False):
            _cloud_drop_sed(state, l_sample=(l_stats and l_sample))

        # ── Microphysics scheme dispatch (microphys_driver.F90:calc_microphys_scheme_tendcies) ──
        # KK / Morrison per-scheme tendency computation + application, now in its Fortran-home module
        # Microphys/microphys_driver.py (mirror-refactor iter 212).
        calc_microphys_scheme_tendcies(state, time_current)

        # ── Driver-owned stats updates (mirrors Fortran driver) ────────
        if l_stats and l_sample:
            _update_driver_stats(state, time_current)

        # ── Stats: end timestep ─────────────────────────────────────────
        if l_last_sample:
            _end_timestep_stats(state, time_current)

        # ── Update time ─────────────────────────────────────────────────
        time_current = time_initial + itime * dt_main

        # Periodic cyclic-GC collection on sampled steps. The per-step diagnostic
        # JAX arrays produced when l_sample=True form reference cycles (Array↔traceback/aval),
        # which CPython's generational GC does not reclaim promptly inside this tight numeric
        # loop → orphaned device buffers accumulate (~78/sampled step on mpace_a) → OOM on long
        # per-step-stats runs (e.g. compare_runs forces sampling every step). An explicit
        # gc.collect() frees them (verified: live_arrays stays flat). Stats-off runs don't sample,
        # so they never leak and never pay this cost. Amortised over _GC_INTERVAL sampled steps.
        if l_sample:
            _samples_since_gc += 1
            if _samples_since_gc >= _GC_INTERVAL:
                gc.collect()
                _samples_since_gc = 0

        if l_stdout:
            print(f"iteration: {itime:8d} / {ifinal:8d}"
                  f" -- time = {time_current:10.1f} / {state['time_final']:10.1f}")


def _begin_timestep_stats(state: dict, itime_idx: int) -> tuple[bool, bool]:
    """Begin stats collection and create the JAX stats bridge for core updates."""
    if state['l_stats']:
        clubb_api.stats_begin_timestep(itime_idx)
        stats_cfg = clubb_api.get_stats_config()
        l_sample = bool(stats_cfg[7])
        l_last_sample = bool(stats_cfg[8])
        state['_jax_stats'] = JaxStats.from_api(
            ngrdcol=state['ngrdcol'],
            nzm=state['nzm'],
            nzt=state['nzt'],
        )
        return l_sample, l_last_sample

    state['_jax_stats'] = JaxStats.empty(
        l_sample=False,
        names=(),
        ncol=state['ngrdcol'],
        max_nlev=max(state['nzm'], state['nzt'], 1),
        max_events=1,
    )
    return False, False


def _end_timestep_stats(state: dict, time_current: float) -> None:
    """Finalize a sampled stats timestep through the Fortran-backed API."""
    stats_time = float(time_current + state['cfg']['stats_tout'])
    state['err_info'] = err_info_from_api(
        clubb_api.stats_end_timestep(
            stats_time,
            err_info=err_info_to_api(state['err_info']),
        )
    )


def _update_driver_stats(state: dict, time_current: float) -> None:
    """Write driver-owned stats outside the jitted core."""
    # For Morrison cases the Ncm / Nc_in_cloud stats are written INSIDE advance_microphys
    # (advance_microphys_module.F90:425-431). That routine early-returns before those writes
    # whenever time_current < microphys_start_time (line 258), so during the pre-activation
    # spin-up the Fortran leaves both stats at their zero-initialized fill. Mirror that exactly:
    # write zeros in that window instead of the init-time Nc_in_cloud*cloud_frac diagnostic.
    micro_pending = (
        state.get('microphys_scheme', 'none') == 'morrison'
        and time_current < state.get('microphys_start_time', 0.0)
    )
    if micro_pending:
        zero = state['Nc_in_cloud'] * 0.0
        clubb_api.stats_update("Ncm", zero)
        clubb_api.stats_update("Nc_in_cloud", zero)
    else:
        state['Ncm'] = state['Nc_in_cloud'] * state['cloud_frac']
        clubb_api.stats_update("Ncm", state['Ncm'])
        clubb_api.stats_update("Nc_in_cloud", state['Nc_in_cloud'])

    if state.get('_morr_rcm_mc') is not None:
        clubb_api.stats_update("rcm_mc", state['_morr_rcm_mc'])
        clubb_api.stats_update("rvm_mc", state['_morr_rvm_mc'])
        clubb_api.stats_update("thlm_mc", state['_morr_thlm_mc'])
        clubb_api.stats_update("rtm_mc", state['_morr_rcm_mc'] + state['_morr_rvm_mc'])

    for field in ('wprtp_mc', 'wpthlp_mc', 'rtp2_mc', 'thlp2_mc', 'rtpthlp_mc'):
        value = state.get('_kk_' + field)
        if value is not None:
            clubb_api.stats_update(field, value)

    if state.get('hm_metadata') is not None and state.get('hydromet') is not None:
        hm_metadata = state['hm_metadata']
        hydromet = np.asarray(state['hydromet'])
        clubb_api.stats_update("rrm", hydromet[..., int(hm_metadata.iirr)])
        clubb_api.stats_update("Nrm", hydromet[..., int(hm_metadata.iiNr)])
        if state.get('_kk_rrm_mc') is not None:
            clubb_api.stats_update("rrm_mc", state['_kk_rrm_mc'])
            clubb_api.stats_update("Nrm_mc", state['_kk_Nrm_mc'])
        if state.get('_kk_precip_frac') is not None:
            clubb_api.stats_update("precip_frac", state['_kk_precip_frac'])


def _calculate_thvm(state: dict):
    """Update virtual potential temperature diagnostic."""
    state['thvm'] = calculate_thvm(
        nzt=state['nzt'],
        ngrdcol=state['ngrdcol'],
        thlm=state['thlm'],
        rtm=state['rtm'],
        rcm=state['rcm'],
        exner=state['exner'],
        thv_ds_zt=state['thv_ds_zt'],
    )


def _calculate_thlp2_rad(state: dict):
    """Apply radiation contribution to thlp2 forcing when enabled."""
    if not state['l_calc_thlp2_rad']:
        return

    state['thlp2_forcing'] = calculate_thlp2_rad(
        state['ngrdcol'],
        state['nzm'],
        state['nzt'],
        state['gr'],
        state['rcm'],
        state['thlprcp'],
        state['radht'],
        state['clubb_params'],
        state['thlp2_forcing'],
    )


def _cloud_drop_sed(state: dict, l_sample: bool = False):
    """Cloud-droplet sedimentation tendencies (clubb_driver.F90:3702-3721).

    Mirrors the Fortran: reset rcm_mc/thlm_mc, set Ncm = Nc_in_cloud*cloud_frac,
    then add the cloud-sedimentation term. Stores the tendencies in state for the
    next step's forcings.
    """
    from clubb_jax.src.Microphys.cloud_sed_module import cloud_drop_sed

    ncm = state['Nc_in_cloud'] * state['cloud_frac']
    rcm_mc, thlm_mc, fcsed = cloud_drop_sed(
        state['rcm'],
        ncm,
        state['rho_zm'],
        state['rho'],
        state['exner'],
        state['sigma_g'],
        state['gr'],
    )
    state['rcm_mc'] = rcm_mc
    state['thlm_mc'] = thlm_mc
    if l_sample:
        clubb_api.stats_update("sed_rcm", rcm_mc)
        clubb_api.stats_update("Fcsed", fcsed)


def _advance_clubb_core(state: dict):
    """Advance the CLUBB core using the JAX translated core."""
    result = advance_clubb_core_module.advance_clubb_core(
        gr=state['gr'],
        nzm=state['nzm'],
        nzt=state['nzt'],
        ngrdcol=state['ngrdcol'],
        l_implemented=False,
        dt=state['dt_main'],
        fcor=state['fcor'],
        fcor_y=state['fcor_y'],
        sfc_elevation=state['sfc_elevation'],
        hydromet_dim=state['hydromet_dim'],
        sclr_dim=state['sclr_dim'],
        sclr_tol=state['sclr_tol'],
        edsclr_dim=state['edsclr_dim'],
        sclr_idx=state['sclr_idx'],
        thlm_forcing=state['thlm_forcing'],
        rtm_forcing=state['rtm_forcing'],
        um_forcing=state['um_forcing'],
        vm_forcing=state['vm_forcing'],
        sclrm_forcing=state['sclrm_forcing'],
        edsclrm_forcing=state['edsclrm_forcing'],
        wprtp_forcing=state['wprtp_forcing'],
        wpthlp_forcing=state['wpthlp_forcing'],
        rtp2_forcing=state['rtp2_forcing'],
        thlp2_forcing=state['thlp2_forcing'],
        rtpthlp_forcing=state['rtpthlp_forcing'],
        wm_zm=state['wm_zm'],
        wm_zt=state['wm_zt'],
        wpthlp_sfc=state['wpthlp_sfc'],
        wprtp_sfc=state['wprtp_sfc'],
        upwp_sfc=state['upwp_sfc'],
        vpwp_sfc=state['vpwp_sfc'],
        p_sfc=state['p_sfc'],
        wpsclrp_sfc=state['wpsclrp_sfc'],
        wpedsclrp_sfc=state['wpedsclrp_sfc'],
        upwp_sfc_pert=state['upwp_sfc_pert'],
        vpwp_sfc_pert=state['vpwp_sfc_pert'],
        rtm_ref=state['rtm_ref'],
        thlm_ref=state['thlm_ref'],
        um_ref=state['um_ref'],
        vm_ref=state['vm_ref'],
        ug=state['ug'],
        vg=state['vg'],
        p_in_Pa=state['p_in_Pa'],
        rho_zm=state['rho_zm'],
        rho=state['rho'],
        exner=state['exner'],
        rho_ds_zm=state['rho_ds_zm'],
        rho_ds_zt=state['rho_ds_zt'],
        invrs_rho_ds_zm=state['invrs_rho_ds_zm'],
        invrs_rho_ds_zt=state['invrs_rho_ds_zt'],
        thv_ds_zm=state['thv_ds_zm'],
        thv_ds_zt=state['thv_ds_zt'],
        l_mix_rat_hm=state['l_mix_rat_hm'],
        rfrzm=state['rfrzm'],
        wphydrometp=state['wphydrometp'],
        wp2hmp=state['wp2hmp'],
        rtphmp_zt=state['rtphmp_zt'],
        thlphmp_zt=state['thlphmp_zt'],
        host_dx=state['host_dx'],
        host_dy=state['host_dy'],
        clubb_params=state['clubb_params'],
        nu_vert_res_dep=state['nu_vert_res_dep'],
        lmin=state['lmin'],
        mixt_frac_max_mag=state['mixt_frac_max_mag'],
        t0=state['T0'],
        ts_nudge=state['ts_nudge'],
        rtm_min=state['rtm_min'],
        rtm_nudge_max_altitude=state['rtm_nudge_max_altitude'],
        clubb_config_flags=state['flags'],
        stats=state['_jax_stats'],
        um=state['um'],
        vm=state['vm'],
        upwp=state['upwp'],
        vpwp=state['vpwp'],
        up2=state['up2'],
        vp2=state['vp2'],
        up3=state['up3'],
        vp3=state['vp3'],
        thlm=state['thlm'],
        rtm=state['rtm'],
        wprtp=state['wprtp'],
        wpthlp=state['wpthlp'],
        wp2=state['wp2'],
        wp3=state['wp3'],
        rtp2=state['rtp2'],
        rtp3=state['rtp3'],
        thlp2=state['thlp2'],
        thlp3=state['thlp3'],
        rtpthlp=state['rtpthlp'],
        sclrm=state['sclrm'],
        sclrp2=state['sclrp2'],
        sclrp3=state['sclrp3'],
        sclrprtp=state['sclrprtp'],
        sclrpthlp=state['sclrpthlp'],
        wpsclrp=state['wpsclrp'],
        edsclrm=state['edsclrm'],
        rcm=state['rcm'],
        cloud_frac=state['cloud_frac'],
        wpthvp=state['wpthvp'],
        wp2thvp=state['wp2thvp'],
        wp2up=state['wp2up'],
        rtpthvp=state['rtpthvp'],
        thlpthvp=state['thlpthvp'],
        sclrpthvp=state['sclrpthvp'],
        wp2rtp=state['wp2rtp'],
        wp2thlp=state['wp2thlp'],
        uprcp=state['uprcp'],
        vprcp=state['vprcp'],
        rc_coef_zm=state['rc_coef_zm'],
        wp4=state['wp4'],
        wpup2=state['wpup2'],
        wpvp2=state['wpvp2'],
        wp2up2=state['wp2up2'],
        wp2vp2=state['wp2vp2'],
        ice_supersat_frac=state['ice_supersat_frac'],
        um_pert=state['um_pert'],
        vm_pert=state['vm_pert'],
        upwp_pert=state['upwp_pert'],
        vpwp_pert=state['vpwp_pert'],
        pdf_params=state['pdf_params'],
        pdf_params_zm=state['pdf_params_zm'],
        pdf_implicit_coefs_terms=state['pdf_implicit_coefs_terms'],
        err_info=state['err_info'],
    )
    if result is None:
        raise RuntimeError(
            "advance_clubb_core returned without outputs; "
            f"{_err_code_summary(state.get('err_info'))}"
        )

    (
        state['um'],
        state['vm'],
        state['up3'],
        state['vp3'],
        state['thlm'],
        state['rtm'],
        state['rtp3'],
        state['thlp3'],
        state['wp3'],
        state['upwp'],
        state['vpwp'],
        state['up2'],
        state['vp2'],
        state['wprtp'],
        state['wpthlp'],
        state['rtp2'],
        state['thlp2'],
        state['rtpthlp'],
        state['wp2'],
        state['sclrm'],
        state['sclrp3'],
        state['wpsclrp'],
        state['sclrp2'],
        state['sclrprtp'],
        state['sclrpthlp'],
        state['p_in_Pa'],
        state['exner'],
        state['rcm'],
        state['cloud_frac'],
        state['wp2thvp'],
        state['wp2up'],
        state['wpthvp'],
        state['rtpthvp'],
        state['thlpthvp'],
        state['sclrpthvp'],
        state['wp2rtp'],
        state['wp2thlp'],
        state['wpup2'],
        state['wpvp2'],
        state['ice_supersat_frac'],
        state['uprcp'],
        state['vprcp'],
        state['rc_coef_zm'],
        state['wp4'],
        state['wp2up2'],
        state['wp2vp2'],
        state['um_pert'],
        state['vm_pert'],
        state['upwp_pert'],
        state['vpwp_pert'],
        state['edsclrm'],
        state['pdf_params'],
        state['pdf_params_zm'],
        state['pdf_implicit_coefs_terms'],
        state['err_info'],
        state['rcm_in_layer'],
        state['cloud_cover'],
        state['w_up_in_cloud'],
        state['w_down_in_cloud'],
        state['cloudy_updraft_frac'],
        state['cloudy_downdraft_frac'],
        state['wprcp_out'],
        state['invrs_tau_zm'],
        state['Kh_zt'],
        state['Kh_zm'],
        state['thlprcp'],
        state['Lscale'],
        state['_sigma_sqd_w'],
        state['_rc_coef'],
        state['_rcp2_zt'],
        state['_wprtp2'],
        state['_wpthlp2'],
        state['_wprtpthlp'],
        state['_wp2rcp'],
        state['_rtprcp'],
        state['_rcp2'],
        state['_skw_velocity'],
        state['_cloud_frac_zm'],
        state['_ice_supersat_frac_zm'],
        state['_rtm_zm'],
        state['_thlm_zm'],
        state['_rcm_zm'],
        state['_rcm_supersat_adj'],
        state['_wp2sclrp'],
        state['_wpsclrp2'],
        state['_sclrprcp'],
        state['_wpsclrprtp'],
        state['_wpsclrpthlp'],
        state['_jax_stats'],
    ) = result
    if state['err_info'].is_fatal():
        message = (
            "advance_clubb_core returned fatal err_info; "
            f"{_err_code_summary(state['err_info'])}"
        )
        print(message)
        raise RuntimeError(message)


def _prescribe_forcings(state: dict, itime: int, l_sample: bool = False):
    """Set forcings for the current timestep using the JAX prescribe_forcings port."""
    time_current = float(state['time_initial'] + (itime - 1) * state['dt_main'])
    if str(state['runtype']).strip() == "arm":
        prescribe_forcings_arm(state, time_current, l_sample=l_sample)
    else:
        prescribe_forcings_generic(state, time_current, l_sample=l_sample)


def _advance_radiation(
    state: dict,
    time_current: float,
    l_sample: bool = False,
):
    """Advance radiation tendencies for currently supported schemes."""
    from clubb_jax.src.Radiation.radiation import advance_radiation

    advance_radiation(state=state, time_current=time_current, l_sample=l_sample)
