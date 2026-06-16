"""Python radiation driver for simplified CLUBB radiation schemes (mirrors Radiation/radiation_module.F90).

Supported schemes:
- none
- simplified
- simplified_bomex
- bugsrad (correlated-k; via clubb_jax/src/Radiation/bugsrad_driver.py)
"""

import numpy as np
import jax.numpy as jnp

# Tracer-transparent shim (REFACTOR B5): _asarray/_xp/_iset behave EXACTLY like numpy for concrete arrays
# (normal runs bit-identical) but route to jnp under a jax.grad trace, so radiation (rcm→radht→thlm_forcing)
# stays on the whole-driver autodiff graph. See CLUBB_core/tracer_numpy.py.
from clubb_jax.src.CLUBB_core.tracer_numpy import _asarray, _iset, _is_tracer_arg
# cos_solar_zen / sunray_sw live in their own Fortran-named modules; radiation_module.F90 `use`s both
# for the SW path.
from clubb_jax.src.Radiation.cos_solar_zen_module import cos_solar_zen
from clubb_jax.src.Radiation.rad_lwsw_module import sunray_sw
# simple_rad / simple_rad_bomex live in simple_rad_module.py (simple_rad_module.F90); the dispatch below
# (radiation_module.F90) `use`s them.
from clubb_jax.src.Radiation.simple_rad_module import simple_rad_bomex, simple_rad
from clubb_jax.src.CLUBB_core.constants_clubb import Cp as _CP  # mirror radiation_module.F90 `use constants_clubb, only: Cp`


def advance_clubb_radiation(state: dict, time_current: float, l_sample: bool = False):
    """Advance radiation tendencies for the current timestep (radiation_module.F90:advance_clubb_radiation).

    Runs the interactive soil/vegetation step (if enabled) before dispatching to the configured radiation
    scheme via `radiation_driver`, mirroring the Fortran advance_clubb_radiation → radiation_driver chain."""
    sw = state.get('stats_writer')

    # Interactive soil/vegetation (gabls3): runs in the radiation wrapper BEFORE the radiation advance
    # (radiation_module.F90:148-157), using the previous step's surface radiative fluxes. The Fortran runs
    # it every step; this lives inside advance_clubb_radiation, which is exact when dt_rad==dt_main (true for
    # gabls3). A future l_soil_veg case with dt_rad>dt_main would need it lifted out of the radiation gate.
    if bool(state['cfg'].get('l_soil_veg', False)):
        _advance_soil_veg_step(state, l_sample=l_sample)

    radiation_driver(state, time_current, l_sample=l_sample)

    if l_sample and sw is not None:
        sw.update("radht", state['radht'])


def radiation_driver(state: dict, time_current: float, l_sample: bool = False):
    """Dispatch to the configured radiation scheme and write state['radht'] (radiation_module.F90:radiation_driver).

    Mirrors the Fortran radiation_driver called by advance_clubb_radiation — selects none / simplified_bomex /
    simplified / bugsrad by `rad_scheme`; the per-scheme work lives in the `_advance_*_radiation` branch helpers
    (the JAX decomposition of radiation_driver's per-scheme inline blocks)."""
    scheme = str(state['rad_scheme']).strip().lower()
    if scheme == "none":
        state['radht'].fill(0.0)
    elif scheme == "simplified_bomex":
        state['radht'][:] = simple_rad_bomex(state['gr'].zt)
    elif scheme == "simplified":
        _advance_simplified_radiation(state, time_current, l_sample=l_sample)
    elif scheme == "bugsrad":
        _advance_bugsrad_radiation(state, time_current, l_sample=l_sample)
    else:
        raise ValueError(
            f"Unsupported rad_scheme={scheme!r} in Python radiation driver. "
            "Supported: none, simplified, simplified_bomex, bugsrad."
        )


def _advance_simplified_radiation(state: dict, time_current: float, l_sample: bool = False):
    """Python port of the simplified branch from radiation_module."""
    cfg = state['cfg']
    gr = state['gr']
    ngrdcol = state['ngrdcol']
    sw = state.get('stats_writer')

    l_sw_radiation = bool(cfg.get('l_sw_radiation', False))

    frad_sw = np.zeros((ngrdcol, gr.nzm), dtype=np.float64)
    radht_sw = np.zeros((ngrdcol, gr.nzt), dtype=np.float64)

    amu0 = _compute_amu0(state, time_current)
    # sunray_sw is a numpy/Python-native sequential-taupath routine (np.exp/math.exp + an in-place flux loop), not
    # tracer-transparent. Radiation feeds only the NEXT timestep's forcing — confirmed iter 518 from the loop order in
    # advance_clubb_to_end: the PREVIOUS step's radht is applied (line 58) BEFORE the core advance (line 78), while
    # THIS step's radht is computed here and stored for next step — so radiation is dead for a single-step whole-driver
    # gradient and skipping it gives the CORRECT (not merely finite) grad. Skip the SW under a jax.grad trace (the
    # detach-under-trace pattern, like BUGSrad); the concrete/forward path is unchanged, so faithfulness is preserved.
    # Fixes the clex9_oct14 grad TracerArrayConversionError (daytime simplified-SW cases), found via _nanhunt (iter 515).
    if l_sw_radiation and amu0 > 0.0 and not _is_tracer_arg([state['rcm'], state['rho']]):
        fs0 = _compute_fs0(cfg, amu0)
        frad_sw = sunray_sw(
            ngrdcol=ngrdcol,
            nzt=gr.nzt,
            rcm=state['rcm'],
            rho=state['rho'],
            xi_abs=amu0,
            dzt=1.0 / gr.invrs_dzt,
            zm=gr.zm,
            zt=gr.zt,
            # sunray_sw gets radius=eff_drop_radius and A=alvdr (radiation_module.F90:507). The JAX was
            # reading the wrong namelist keys (default radius / 0 albedo) — invisible until an active-SW
            # case (nov11: amu0=0.62, alvdr=0.1; jun25's SW is inactive so it never exercised the albedo).
            radius=float(cfg.get('eff_drop_radius', cfg.get('radius', 10.0e-6))),
            A=float(cfg.get('alvdr', cfg.get('A_surface_albedo', 0.0))),
            gc=float(cfg.get('gc', 0.85)),
            Fs0=fs0,
            # the namelist key is `omega` (radiation_module.F90); the JAX read `omega_sw` → default 0.999.
            # The SW absorption ∝ (1−omega), so 0.999 vs nov11's 0.9965 = 3.5× too little absorption.
            omega=float(cfg.get('omega', cfg.get('omega_sw', 0.999))),
            l_center=bool(cfg.get('l_center_rad', True)),
        )
        radht_sw = (
            -(frad_sw[:, 1:] - frad_sw[:, :-1]) * gr.invrs_dzt
            / (state['rho'] * _CP)
        )

    frad_lw, radht_lw = simple_rad(
        gr=gr,
        ngrdcol=ngrdcol,
        rho=state['rho'],
        rho_zm=state['rho_zm'],
        rtm=state['rtm'],
        rcm=state['rcm'],
        exner=state['exner'],
        f0=float(cfg.get('f0', 0.0)),
        f1=float(cfg.get('f1', 0.0)),
        kappa=float(cfg.get('kappa', 0.0)),
        l_rad_above_cloud=bool(cfg.get('l_rad_above_cloud', False)),
        l_sample=l_sample,
        sw=sw,
    )

    frad_total = frad_sw + frad_lw
    state['radht'] = _iset(state['radht'], np.s_[:], radht_sw + radht_lw)  # _iset: tracer-safe under grad
    state['Frad'] = frad_total
    state['Frad_SW'] = frad_sw
    state['Frad_LW'] = frad_lw
    state['radht_SW'] = radht_sw
    state['radht_LW'] = radht_lw

    if l_sample and sw is not None:
        sw.update("Frad", frad_total)
        sw.update("Frad_SW", frad_sw)
        sw.update("Frad_LW", frad_lw)
        sw.update("radht_SW", radht_sw)
        sw.update("radht_LW", radht_lw)


def _advance_soil_veg_step(state: dict, l_sample: bool = False):
    """Advance the gabls3 soil/vegetation temperatures (soil_vegetation.F90, called from radiation_module.F90).
    Uses the SURFACE slice (CLUBB index 0 in JAX) of the previous step's BUGSrad fluxes + the current
    surface turbulent fluxes. Initialises veg/soil temps on first call; persists them in state."""
    from clubb_jax.src.Radiation.soil_vegetation import advance_soil_veg, initialize_soil_veg
    ngrdcol = state['ngrdcol']
    if 'veg_T_in_K' not in state:
        deep, sfc, veg = initialize_soil_veg(ngrdcol)
        state['deep_soil_T_in_K'], state['sfc_soil_T_in_K'], state['veg_T_in_K'] = deep, sfc, veg

    dt = float(state['dt_main'])
    rho_sfc = _asarray(state['rho_zm'], dtype=np.float64)[:, 0]    # rho_zm(:,1) in Fortran
    z = np.zeros(ngrdcol)
    # _asarray (REFACTOR B5): the surface fluxes (wpthlp_sfc/wprtp_sfc) are tracers under a jax.grad trace;
    # advance_soil_veg is jax-compatible, so keep them on the graph (the interactive veg_T it returns feeds the
    # NEXT step's surface flux). np.asarray would sever (TracerArrayConversionError).
    sfc0 = lambda key: _asarray(state[key], dtype=np.float64)[:, 0] if key in state else z
    deep, sfc, veg, _ = advance_soil_veg(
        dt, rho_sfc, sfc0('Frad_SW_up'), sfc0('Frad_SW_down'), sfc0('Frad_LW_down'),
        _asarray(state['wpthlp_sfc'], dtype=np.float64), _asarray(state['wprtp_sfc'], dtype=np.float64),
        _asarray(state['p_sfc'], dtype=np.float64),
        state['deep_soil_T_in_K'], state['sfc_soil_T_in_K'], state['veg_T_in_K'])
    state['deep_soil_T_in_K'] = _asarray(deep)
    state['sfc_soil_T_in_K'] = _asarray(sfc)
    state['veg_T_in_K'] = _asarray(veg)


def _advance_bugsrad_radiation(state: dict, time_current: float, l_sample: bool = False):
    """BUGSrad branch — port of radiation_module.F90 case("bugsrad") (the -Dradoffline path).
    Builds the CLUBB↔BUGSrad grid setup once (cached in state['_bugsrad_setup']), then per step maps
    the CLUBB state onto the radiation grid and calls compute_bugsrad_radiation. T_in_K = thlm·exner +
    Lv·rcm/Cp (T_in_K_module.F90); p on the m-grid via zt2zm; rsm/rim default to 0 (no microphysics)."""
    import jax.numpy as jnp
    from clubb_jax.src.Input_fields.sounding import load_extended_std_atm
    from clubb_jax.src.Radiation.bugsrad_driver import (
        build_rad_grid_setup, compute_bugsrad_radiation)
    from clubb_jax.src.CLUBB_core.grid_class import zt2zm
    from clubb_jax.src.CLUBB_core.constants_clubb import Lv as _LV

    cfg = state['cfg']; gr = state['gr']; ngrdcol = state['ngrdcol']
    sw = state.get('stats_writer')

    # Detach-under-trace (REFACTOR B5). BUGSrad (correlated-k RT, std-atm extension, many bands) is
    # reverse-mode memory-PROHIBITIVE, and radiation runs AFTER the core — so the radht it writes feeds only
    # the NEXT step's forcing, never the current step's prognostics. Under a jax.grad trace we therefore skip
    # it (leaving state['radht'] etc. at their incoming values): EXACT for a single-step gradient, and treats
    # radiation as a detached forcing for multi-step rollouts (the practical choice for heavy RT). The light
    # simplified-LW scheme stays fully differentiable (it is cheap); only BUGSrad is detached.
    if _is_tracer_arg([state['thlm'], state['rcm'], state['rtm']]):
        return

    p_in_Pa = np.asarray(state['p_in_Pa'], dtype=np.float64)
    p_in_Pam = np.asarray(zt2zm(gr.nzm, gr.nzt, gr.ngrdcol, gr, jnp.asarray(p_in_Pa)), dtype=np.float64)

    # build + cache the static radiation-grid layout (std-atm extension + buffer) on first call
    setup = state.get('_bugsrad_setup')
    if setup is None:
        # l_use_default_std_atmosphere=.false. (CGILS/cloud_feedback/astex/twp_ice): the Fortran builds the
        # radiation extended atmosphere (above the model top, T/q/p/o3) from the case's OWN deep sounding +
        # {case}_ozone_sounding.in (convert_snd2extended_atm), not the default US-standard atmosphere. The driver
        # precomputes that into state['_rad_ext_atm'] at init (Iter92). Use it when present; otherwise fall back
        # to the default std atmosphere (the gated cases, flag true, are unaffected).
        ext = state.get('_rad_ext_atm') or load_extended_std_atm()
        zm = np.asarray(gr.zm, dtype=np.float64)[0]
        dzt = np.asarray(gr.dzt, dtype=np.float64)[0]          # zm_grid_spacing (Fortran passes gr%dzt)
        rad_top = float(cfg.get('radiation_top', 50000.0))
        setup = build_rad_grid_setup(zm, dzt, p_in_Pam[0], rad_top, ext)
        state['_bugsrad_setup'] = setup

    rcm = np.asarray(state['rcm'], dtype=np.float64)
    thlm = np.asarray(state['thlm'], dtype=np.float64)
    rtm = np.asarray(state['rtm'], dtype=np.float64)
    exner = np.asarray(state['exner'], dtype=np.float64)
    T_in_K = thlm * exner + _LV * rcm / _CP
    zeros = np.zeros_like(rcm)
    rsm = np.asarray(state.get('rsm', zeros), dtype=np.float64)
    rim = np.asarray(state.get('rim', zeros), dtype=np.float64)

    amu0 = float(_compute_amu0(state, time_current))           # raw cos zenith; bugs_rad masks night (<0.01)
    amu0_arr = np.full(ngrdcol, amu0)
    slr_arr = np.full(ngrdcol, float(cfg.get('slr', 1.0)))
    sol_const = float(cfg.get('sol_const', 1367.0))
    alb = lambda k: np.full(ngrdcol, float(cfg.get(k, 0.1)))

    res = compute_bugsrad_radiation(
        setup, T_in_K, rcm, rtm, rsm, rim,
        np.asarray(state['cloud_frac'], dtype=np.float64),
        np.asarray(state['ice_supersat_frac'], dtype=np.float64),
        p_in_Pa, p_in_Pam, np.asarray(state['rho_zm'], dtype=np.float64), exner,
        amu0_arr, slr_arr, alb('alvdr'), alb('alvdf'), alb('alndr'), alb('alndf'),
        sol_const=sol_const)

    state['radht'][:] = np.asarray(res['radht'])
    state['Frad'] = np.asarray(res['Frad'])
    state['Frad_SW'] = np.asarray(res['Frad_SW_up'] - res['Frad_SW_down'])
    state['Frad_LW'] = np.asarray(res['Frad_LW_up'] - res['Frad_LW_down'])
    state['radht_SW'] = np.asarray(res['radht_SW'])
    state['radht_LW'] = np.asarray(res['radht_LW'])
    # up/down components (surface slices feed soil_vegetation next step)
    state['Frad_SW_up'] = np.asarray(res['Frad_SW_up']); state['Frad_SW_down'] = np.asarray(res['Frad_SW_down'])
    state['Frad_LW_up'] = np.asarray(res['Frad_LW_up']); state['Frad_LW_down'] = np.asarray(res['Frad_LW_down'])

    if l_sample and sw is not None:
        sw.update("Frad", state['Frad'])
        sw.update("Frad_SW", state['Frad_SW'])
        sw.update("Frad_LW", state['Frad_LW'])
        sw.update("radht_SW", state['radht_SW'])
        sw.update("radht_LW", state['radht_LW'])


# ── cos_solar_zen ──────────────────────────────────────────────────────────────


def _compute_amu0(state: dict, time_current: float) -> float:
    """Compute cosine of solar zenith angle for simplified radiation."""
    cfg = state['cfg']
    l_fix = bool(cfg.get('l_fix_cos_solar_zen', False))

    cos_vals = _as_1d_float(cfg.get('cos_solar_zen_values', [0.0]))
    if l_fix:
        if cos_vals.size == 1:
            return float(cos_vals[0])
        times = _as_1d_float(cfg.get('cos_solar_zen_times', [time_current]))
        if times.size != cos_vals.size:
            return float(cos_vals[0])
        idx = int(np.searchsorted(times, time_current, side='left'))
        if idx >= times.size:
            raise ValueError("time_current exceeds provided cos_solar_zen_times range.")
        return float(cos_vals[idx])

    return cos_solar_zen(
        int(cfg.get('day', 1)),
        int(cfg.get('month', 1)),
        int(cfg.get('year', 2000)),
        float(time_current),
        float(cfg.get('lat_vals', 0.0)),
        float(cfg.get('lon_vals', 0.0)),
    )


def _compute_fs0(cfg: dict, amu0: float) -> float:
    fs_values = _as_1d_float(cfg.get('fs_values', [0.0]))
    cos_values = _as_1d_float(cfg.get('cos_solar_zen_values', [0.0]))

    if fs_values.size == 0:
        return 0.0
    if fs_values.size == 1 or cos_values.size != fs_values.size:
        return float(fs_values[0])

    return float(np.interp(amu0, cos_values, fs_values))


def _as_1d_float(value):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr
