"""CLUBB time-loop driver utilities.

This module contains the timestep advancement logic that was split out from
the previous monolithic Python driver file.
"""

import gc

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.T_in_K_module import calculate_thvm_jax
from clubb_jax.src.CLUBB_core.advance_helper_module import calculate_thlp2_rad_jax
from clubb_jax.src.Benchmark_cases.arm import prescribe_forcings_arm, _Cp as _ARM_Cp, _Lv as _ARM_Lv
from clubb_jax.src.Benchmark_cases.generic_forcings import prescribe_forcings_generic
from clubb_jax.src.CLUBB_core.advance_clubb_core_module import advance_clubb_core as _advance_clubb_core_py
# Tracer-transparent shim (REFACTOR B5): _asarray behaves exactly like np.asarray for concrete
# arrays (normal runs bit-identical) but routes to jnp under a jax.grad trace so the whole-driver
# autodiff graph survives the imperative `state[k] = ...` writebacks. See CLUBB_core/tracer_numpy.py.
from clubb_jax.src.CLUBB_core.tracer_numpy import _asarray


def advance_clubb_to_end(state: dict, l_stdout: bool = True, max_steps: int | None = None):
    """Run the CLUBB time loop."""

    dt_main = state['dt_main']
    dt_rad = state['dt_rad']
    time_initial = state['time_initial']
    ifinal = state['ifinal']
    l_stats = state['l_stats']

    rad_interval = int(dt_rad / dt_main)
    n_steps = ifinal if max_steps is None else min(ifinal, max_steps)
    sw = state.get('stats_writer')  # Python stats writer; None → fall back to Fortran API

    _GC_INTERVAL = 10          # force a cyclic-GC pass every N sampled steps (Iter325 OOM fix)
    _samples_since_gc = 0

    for itime_idx in range(n_steps):
        itime = itime_idx + 1
        time_current = time_initial + (itime - 1) * dt_main
        l_sample = False
        l_last_sample = False

        # ── Stats: begin timestep ───────────────────────────────────────
        if l_stats and sw is not None:
            l_sample, l_last_sample = sw.begin_timestep(itime_idx)

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
            state['rtm_forcing'] = state['rtm_forcing'] + state['_morr_rcm_mc'] + state['_morr_rvm_mc']
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

        # ── KK (Khairoutdinov-Kogan) rain microphysics ─────────────────
        # Staged rollout (Iter155): computes + stores the KK tendencies on live state for
        # shadow-comparison vs the oracle; transport + feedback gated behind l_kk_micro_apply
        # (default off) so the running KK cases are unchanged until the transport stage lands.
        if state.get('microphys_scheme', 'none') == 'khairoutdinov_kogan':
            from clubb_jax.src.Microphys.kk_microphys_step import advance_kk_microphysics
            advance_kk_microphysics(state)

        # ── Morrison (M2005 2-moment) microphysics ─────────────────────
        # Computes the CLUBB-form *_mc (rcm/rvm/thlm + 8 hydrometeors) via morrison_microphys_driver
        # and advances the hydrometeor fields (first-pass Euler; full transport to follow).
        if state.get('microphys_scheme', 'none') == 'morrison' \
                and time_current >= state.get('microphys_start_time', 0.0):
            # The microphysics is skipped until microphys_start_time (microphys_driver.F90:389) — e.g.
            # nov11 has a 60-step spinup; before it, no *_mc and no hydrometeor advance.
            from clubb_jax.src.Microphys.morrison_microphys_step import advance_morrison_microphysics
            advance_morrison_microphysics(state)

        # ── Driver-owned stats updates (mirrors Fortran driver) ────────
        # For Morrison cases the Ncm / Nc_in_cloud stats are written INSIDE advance_microphys
        # (advance_microphys_module.F90:425-431). That routine early-returns before those writes
        # whenever time_current < microphys_start_time (line 258), so during the pre-activation
        # spin-up the Fortran leaves both stats at their zero-initialized fill. Mirror that exactly:
        # write zeros in that window instead of the init-time Nc_in_cloud*cloud_frac diagnostic.
        _micro_pending = (state.get('microphys_scheme', 'none') == 'morrison'
                          and time_current < state.get('microphys_start_time', 0.0))
        if l_stats and l_sample and sw is not None:
            if _micro_pending:
                _zero = state['Nc_in_cloud'] * 0.0
                sw.update("Ncm", _zero)
                sw.update("Nc_in_cloud", _zero)
            else:
                state['Ncm'] = state['Nc_in_cloud'] * state['cloud_frac']
                sw.update("Ncm", state['Ncm'])
                sw.update("Nc_in_cloud", state['Nc_in_cloud'])
            if state.get('_morr_rcm_mc') is not None:   # Morrison tendencies (for diagnosis vs oracle)
                sw.update("rcm_mc", state['_morr_rcm_mc'])
                sw.update("rvm_mc", state['_morr_rvm_mc'])
                sw.update("thlm_mc", state['_morr_thlm_mc'])
                sw.update("rtm_mc", state['_morr_rcm_mc'] + state['_morr_rvm_mc'])
            for _f in ('wprtp_mc', 'wpthlp_mc', 'rtp2_mc', 'thlp2_mc', 'rtpthlp_mc'):
                _v = state.get('_kk_' + _f)
                if _v is not None:
                    sw.update(_f, _v)
            if state.get('hm_metadata') is not None and state.get('hydromet') is not None:
                _hmm = state['hm_metadata']
                sw.update("rrm", np.asarray(state['hydromet'])[..., int(_hmm.iirr)])
                sw.update("Nrm", np.asarray(state['hydromet'])[..., int(_hmm.iiNr)])
                if state.get('_kk_rrm_mc') is not None:
                    sw.update("rrm_mc", state['_kk_rrm_mc'])
                    sw.update("Nrm_mc", state['_kk_Nrm_mc'])
                if state.get('_kk_precip_frac') is not None:
                    sw.update("precip_frac", state['_kk_precip_frac'])

        # ── Stats: end timestep ─────────────────────────────────────────
        if l_stats and l_last_sample and sw is not None:
            stats_time = float(time_current + state['cfg']['stats_tout'])
            sw.end_timestep(stats_time)

        # ── Update time ─────────────────────────────────────────────────
        time_current = time_initial + itime * dt_main

        # Periodic cyclic-GC collection on sampled steps (Iter325). The per-step diagnostic
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

        import os as _os
        if _os.environ.get('CLUBB_LEAK'):
            import jax as _jax, resource as _res
            _rss = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024.0
            print(f"  [LEAK] step {itime}: live_arrays={len(_jax.live_arrays())}  maxRSS={_rss:.0f}MB", flush=True)

        if l_stdout:
            print(f"iteration: {itime:8d} / {ifinal:8d}"
                  f" -- time = {time_current:10.1f} / {state['time_final']:10.1f}")


def _calculate_thvm(state: dict):
    """Update virtual potential temperature diagnostic. Iter65: JAX-only."""
    state['thvm'] = _asarray(calculate_thvm_jax(
        jnp.asarray(state['thlm']),
        jnp.asarray(state['rtm']),
        jnp.asarray(state['rcm']),
        jnp.asarray(state['exner']),
        jnp.asarray(state['thv_ds_zt']),
    ), dtype=np.float64)


def _calculate_thlp2_rad(state: dict):
    """Apply radiation contribution to thlp2 forcing when enabled."""
    if not state['l_calc_thlp2_rad']:
        return

    increment = calculate_thlp2_rad_jax(
        rcm=state['rcm'],
        thlprcp=state['thlprcp'],
        radht=state['radht'],
        clubb_params=state['clubb_params'],
        gr=state['gr'],
    )
    state['thlp2_forcing'] = state['thlp2_forcing'] + _asarray(increment, dtype=np.float64)


def _cloud_drop_sed(state: dict, l_sample: bool = False):
    """Cloud-droplet sedimentation tendencies (clubb_driver.F90:3702-3721).

    Mirrors the Fortran: reset rcm_mc/thlm_mc, set Ncm = Nc_in_cloud*cloud_frac,
    then add the cloud-sedimentation term. Stores the tendencies in state for the
    next step's forcings.
    """
    from clubb_jax.src.Microphys.cloud_sed_module import cloud_drop_sed
    Ncm = state['Nc_in_cloud'] * state['cloud_frac']
    rcm_mc, thlm_mc, Fcsed = cloud_drop_sed(
        state['rcm'], Ncm, state['rho_zm'], state['rho'],
        state['exner'], state['sigma_g'], state['gr'])
    state['rcm_mc'] = rcm_mc
    state['thlm_mc'] = thlm_mc
    if l_sample and state.get('stats_writer') is not None:
        # sed_rcm == rcm_mc for the cloud-sed-only path.
        state['stats_writer'].update("sed_rcm", rcm_mc)
        state['stats_writer'].update("Fcsed", Fcsed)


def _advance_clubb_core(state: dict):
    """Advance the CLUBB core using either the API wrapper or Python port."""
    # Python port with JAX shadow comparisons active for Iteration 4 testing.
    _advance_clubb_core_python(state)
    # Fortran API path (default outside of testing):
    #_advance_clubb_core_api(state)


def _advance_clubb_core_python(state: dict):
    """Advance the CLUBB core using the translated Python port."""
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
    ) = _advance_clubb_core_py(
        gr=state['gr'],
        nzm=state['nzm'],
        nzt=state['nzt'],
        ngrdcol=state['ngrdcol'],
        dt_main=state['dt_main'],
        flags=state['flags'],
        sclr_dim=state['sclr_dim'],
        edsclr_dim=state['edsclr_dim'],
        hydromet_dim=state['hydromet_dim'],
        clubb_params=state['clubb_params'],
        fcor=state['fcor'],
        fcor_y=state['fcor_y'],
        host_dx=state['host_dx'],
        host_dy=state['host_dy'],
        wm_zm=state['wm_zm'],
        wm_zt=state['wm_zt'],
        rho_ds_zt=state['rho_ds_zt'],
        rtm=state['rtm'],
        thlm=state['thlm'],
        rho=state['rho'],
        rfrzm=state['rfrzm'],
        sfc_elevation=state['sfc_elevation'],
        upwp_sfc=state['upwp_sfc'],
        vpwp_sfc=state['vpwp_sfc'],
        wpthlp=state['wpthlp'],
        wprtp_sfc=state['wprtp_sfc'],
        upwp=state['upwp'],
        vpwp=state['vpwp'],
        upwp_sfc_pert=state['upwp_sfc_pert'],
        vpwp_sfc_pert=state['vpwp_sfc_pert'],
        wpsclrp=state['wpsclrp'],
        wpedsclrp_sfc=state['wpedsclrp_sfc'],
        p_sfc=state['p_sfc'],
        thv_ds_zm=state['thv_ds_zm'],
        thv_ds_zt=state['thv_ds_zt'],
        wp2=state['wp2'],
        wp3=state['wp3'],
        thlp2=state['thlp2'],
        rtp2=state['rtp2'],
        rtpthlp=state['rtpthlp'],
        um=state['um'],
        vm=state['vm'],
        p_in_Pa=state['p_in_Pa'],
        exner=state['exner'],
        rcm=state['rcm'],
        ice_supersat_frac=state['ice_supersat_frac'],
        up2=state['up2'],
        vp2=state['vp2'],
        wprtp=state['wprtp'],
        wpthlp_sfc=state['wpthlp_sfc'],
        wp2thvp=state['wp2thvp'],
        wp2up=state['wp2up'],
        rtpthvp=state['rtpthvp'],
        thlpthvp=state['thlpthvp'],
        wpthvp=state['wpthvp'],
        wphydrometp=state['wphydrometp'],
        wp2hmp=state['wp2hmp'],
        rtphmp_zt=state['rtphmp_zt'],
        thlphmp_zt=state['thlphmp_zt'],
        lmin=state['lmin'],
        mixt_frac_max_mag=state['mixt_frac_max_mag'],
        T0=state['T0'],
        ts_nudge=state['ts_nudge'],
        rtm_min=state['rtm_min'],
        rtm_nudge_max_altitude=state['rtm_nudge_max_altitude'],
        um_forcing=state['um_forcing'],
        vm_forcing=state['vm_forcing'],
        thlm_forcing=state['thlm_forcing'],
        rtm_forcing=state['rtm_forcing'],
        wprtp_forcing=state['wprtp_forcing'],
        wpthlp_forcing=state['wpthlp_forcing'],
        rtp2_forcing=state['rtp2_forcing'],
        thlp2_forcing=state['thlp2_forcing'],
        rtpthlp_forcing=state['rtpthlp_forcing'],
        err_info=state['err_info'],
        sclr_tol=state['sclr_tol'],
        thlm_ref=state['thlm_ref'],
        rtm_ref=state['rtm_ref'],
        um_ref=state['um_ref'],
        vm_ref=state['vm_ref'],
        ug=state['ug'],
        vg=state['vg'],
        sclrm_forcing=state['sclrm_forcing'],
        edsclrm_forcing=state['edsclrm_forcing'],
        sclrp2=state['sclrp2'],
        sclrprtp=state['sclrprtp'],
        sclrpthlp=state['sclrpthlp'],
        sclr_idx=state['sclr_idx'],
        pdf_params=state['pdf_params'],
        pdf_params_zm=state['pdf_params_zm'],
        pdf_implicit_coefs_terms=state['pdf_implicit_coefs_terms'],
        nu_vert_res_dep=state['nu_vert_res_dep'],
        sclrm=state['sclrm'],
        sclrpthvp=state['sclrpthvp'],
        up3=state['up3'],
        vp3=state['vp3'],
        um_pert=state['um_pert'],
        vm_pert=state['vm_pert'],
        uprcp=state['uprcp'],
        vprcp=state['vprcp'],
        rc_coef_zm=state['rc_coef_zm'],
        wp4=state['wp4'],
        wpup2=state['wpup2'],
        wpvp2=state['wpvp2'],
        wp2up2=state['wp2up2'],
        wp2vp2=state['wp2vp2'],
        wp2rtp=state['wp2rtp'],
        wp2thlp=state['wp2thlp'],
        upwp_pert=state['upwp_pert'],
        vpwp_pert=state['vpwp_pert'],
        sclrp3=state['sclrp3'],
        cloud_frac=state['cloud_frac'],
        thlp3=state['thlp3'],
        rtp3=state['rtp3'],
        edsclrm=state['edsclrm'],
        wpsclrp_sfc=state['wpsclrp_sfc'],
        l_mix_rat_hm=state['l_mix_rat_hm'],
        rho_ds_zm=state['rho_ds_zm'],
        invrs_rho_ds_zm=state['invrs_rho_ds_zm'],
        invrs_rho_ds_zt=state['invrs_rho_ds_zt'],
        rho_zm=state['rho_zm'],
        l_sample=state.get('l_sample', False),
        l_gamma_Skw=state.get('l_gamma_Skw', True),
        l_advance_xp3=state.get('l_advance_xp3', False),
        l_use_invrs_tau_N2_iso=state.get('l_use_invrs_tau_N2_iso', False),
        order_xm_wpxp=state.get('order_xm_wpxp', 1),
        order_xp2_xpyp=state.get('order_xp2_xpyp', 2),
        order_wp2_wp3=state.get('order_wp2_wp3', 3),
        order_windm=state.get('order_windm', 4),
        debug_level=state['cfg']['debug_level'],
        stats_writer=state.get('stats_writer'),
        wprtp2_carry=state.get('_wprtp2'),
        wpthlp2_carry=state.get('_wpthlp2'),
        wprtpthlp_carry=state.get('_wprtpthlp'),
        sponge_cfg=state.get('sponge'),
    )

def _prescribe_forcings(state: dict, itime: int, l_sample: bool = False):
    """Set forcings for the current timestep.

    ARM: pure-Python port (Iter66).
    Supported non-ARM cases: bomex, fire, generic, neutral, coriolis_test, ekman,
      and any case with l_t_dependent=True and a {runtype}_forcings.in file.
    Unsupported cases: lazy Fortran fallback via clubb_python.clubb_api.
    """
    time_current = state['time_initial'] + (itime - 1) * state['dt_main']

    if state['runtype'] == 'arm':
        prescribe_forcings_arm(state, time_current)
        sw = state.get('stats_writer')
        if l_sample and sw is not None and state.get('rho_zm') is not None:
            rho_zm_sfc = state['rho_zm'][:, 0]
            sw.update("wpthlp_sfc", state['wpthlp_sfc'])
            sw.update("wprtp_sfc", state['wprtp_sfc'])
            sw.update("upwp_sfc", state['upwp_sfc'])
            sw.update("vpwp_sfc", state['vpwp_sfc'])
            sw.update("sh", state['wpthlp_sfc'] * rho_zm_sfc * _ARM_Cp)
            sw.update("lh", state['wprtp_sfc'] * rho_zm_sfc * _ARM_Lv)
            ustar = state.get('ustar', np.zeros(state['ngrdcol']))
            sw.update("ustar", ustar)
            T_sfc = state.get('T_sfc', np.zeros(state['ngrdcol']))
            sw.update("T_sfc", T_sfc)
        return

    # Try the generic Python dispatcher first
    try:
        prescribe_forcings_generic(state, time_current, l_sample=l_sample)
        return
    except NotImplementedError:
        pass  # fall through to Fortran for unsupported cases

    # Fortran fallback for cases not yet ported to Python
    from clubb_python import clubb_api  # lazy import

    (
        state['rtm'],
        state['wm_zm'],
        state['wm_zt'],
        state['ug'],
        state['vg'],
        state['um_ref'],
        state['vm_ref'],
        state['thlm_forcing'],
        state['rtm_forcing'],
        state['um_forcing'],
        state['vm_forcing'],
        state['wprtp_forcing'],
        state['wpthlp_forcing'],
        state['rtp2_forcing'],
        state['thlp2_forcing'],
        state['rtpthlp_forcing'],
        state['wpsclrp'],
        state['sclrm_forcing'],
        state['edsclrm_forcing'],
        state['wpthlp_sfc'],
        state['wprtp_sfc'],
        state['upwp_sfc'],
        state['vpwp_sfc'],
        state['T_sfc'],
        state['p_sfc'],
        state['sens_ht'],
        state['latent_ht'],
        state['wpsclrp_sfc'],
        state['wpedsclrp_sfc'],
        state['err_info'],
    ) = clubb_api.prescribe_forcings(
        gr=state['gr'],
        nzm=state['nzm'],
        nzt=state['nzt'],
        ngrdcol=state['ngrdcol'],
        sclr_dim=state['sclr_dim'],
        edsclr_dim=state['edsclr_dim'],
        runtype=state['runtype'],
        sfctype=state['sfctype'],
        time_current=time_current,
        time_initial=state['time_initial'],
        dt=state['dt_main'],
        um=state['um'],
        vm=state['vm'],
        thlm=state['thlm'],
        p_in_Pa=state['p_in_Pa'],
        exner=state['exner'],
        rho=state['rho'],
        rho_zm=state['rho_zm'],
        thvm=state['thvm'],
        zt_in=state['gr'].zt,
        l_t_dependent=state['l_t_dependent'],
        l_ignore_forcings=state['l_ignore_forcings'],
        l_input_xpwp_sfc=state['l_input_xpwp_sfc'],
        l_modify_bc_for_cnvg_test=state['l_modify_bc_for_cnvg_test'],
        saturation_formula=state['flags'].saturation_formula,
        l_add_dycore_grid=state['flags'].l_add_dycore_grid,
        grid_remap_method=state['flags'].grid_remap_method,
        grid_adapt_in_time_method=state['flags'].grid_adapt_in_time_method,
        rtm=state['rtm'],
        wm_zm=state['wm_zm'],
        wm_zt=state['wm_zt'],
        ug=state['ug'],
        vg=state['vg'],
        um_ref=state['um_ref'],
        vm_ref=state['vm_ref'],
        thlm_forcing=state['thlm_forcing'],
        rtm_forcing=state['rtm_forcing'],
        um_forcing=state['um_forcing'],
        vm_forcing=state['vm_forcing'],
        wprtp_forcing=state['wprtp_forcing'],
        wpthlp_forcing=state['wpthlp_forcing'],
        rtp2_forcing=state['rtp2_forcing'],
        thlp2_forcing=state['thlp2_forcing'],
        rtpthlp_forcing=state['rtpthlp_forcing'],
        wpsclrp=state['wpsclrp'],
        sclrm_forcing=state['sclrm_forcing'],
        edsclrm_forcing=state['edsclrm_forcing'],
        wpthlp_sfc=state['wpthlp_sfc'],
        wprtp_sfc=state['wprtp_sfc'],
        upwp_sfc=state['upwp_sfc'],
        vpwp_sfc=state['vpwp_sfc'],
        T_sfc=state['T_sfc'],
        p_sfc=state['p_sfc'],
        sens_ht=state['sens_ht'],
        latent_ht=state['latent_ht'],
        wpsclrp_sfc=state['wpsclrp_sfc'],
        wpedsclrp_sfc=state['wpedsclrp_sfc'],
        sclr_idx=state['sclr_idx'],
        err_info=state['err_info'],
    )


def _advance_radiation(
    state: dict,
    time_current: float,
    l_sample: bool = False,
):
    """Advance radiation tendencies for currently supported schemes."""
    from clubb_jax.src.Radiation.radiation import advance_radiation

    advance_radiation(state=state, time_current=time_current, l_sample=l_sample)
