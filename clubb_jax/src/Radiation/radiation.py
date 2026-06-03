"""Python radiation driver for simplified CLUBB radiation schemes.

Supported schemes:
- none
- simplified
- simplified_bomex
- bugsrad (correlated-k; via clubb_jax/src/Radiation/bugsrad_driver.py)
"""

import math
import numpy as np
import jax.numpy as jnp

# Tracer-transparent shim (REFACTOR B5): _asarray/_xp/_iset behave EXACTLY like numpy for concrete arrays
# (normal runs bit-identical) but route to jnp under a jax.grad trace, so radiation (rcm→radht→thlm_forcing)
# stays on the whole-driver autodiff graph. See CLUBB_core/tracer_numpy.py.
from clubb_jax.src.CLUBB_core.tracer_numpy import _asarray, _xp, _iset, _is_tracer_arg, _safe_pow
from clubb_jax.src.CLUBB_core.constants_clubb import sec_per_hr, radians_per_deg, rho_lw

_CP = 1004.67
_EPS = np.finfo(np.float64).eps
_LS_DIV = 3.75e-6

# Liou coefficients for cos_solar_zen (cos_solar_zen_module.F90)
_CSZ_C0 =  0.006918
_CSZ_C1 = -0.399912
_CSZ_C2 = -0.006758
_CSZ_C3 = -0.002697
_CSZ_D1 =  0.070257
_CSZ_D2 =  0.000907
_CSZ_D3 =  0.000148


def simple_rad_bomex(zt: np.ndarray):
    """Compute BOMEX simplified radiation heating rate."""
    return np.where(
        zt < 1500.0,
        -2.315e-5,
        np.where(
            zt < 2500.0,
            -2.315e-5 + 2.315e-5 * (zt - 1500.0) / 1000.0,
            0.0,
        ),
    )


def advance_radiation(state: dict, time_current: float, l_sample: bool = False):
    """Advance radiation tendencies for the current timestep."""
    scheme = str(state['rad_scheme']).strip().lower()
    sw = state.get('stats_writer')

    # Interactive soil/vegetation (gabls3): runs in the radiation wrapper BEFORE the radiation advance
    # (radiation_module.F90:148-157), using the previous step's surface radiative fluxes. The Fortran runs
    # it every step; this lives inside advance_radiation, which is exact when dt_rad==dt_main (true for
    # gabls3). A future l_soil_veg case with dt_rad>dt_main would need it lifted out of the radiation gate.
    if bool(state['cfg'].get('l_soil_veg', False)):
        _advance_soil_veg_step(state, l_sample=l_sample)

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

    if l_sample and sw is not None:
        sw.update("radht", state['radht'])


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
    if l_sw_radiation and amu0 > 0.0:
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

    frad_lw, radht_lw = _simple_rad_lw(
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
    from clubb_jax.src.Radiation.bugsrad_driver import (
        load_std_atmosphere, build_rad_grid_setup, compute_bugsrad_radiation)
    from clubb_jax.src.CLUBB_core.grid_class import zt2zm_jax
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
    p_in_Pam = np.asarray(zt2zm_jax(jnp.asarray(p_in_Pa), gr), dtype=np.float64)

    # build + cache the static radiation-grid layout (std-atm extension + buffer) on first call
    setup = state.get('_bugsrad_setup')
    if setup is None:
        ext = load_std_atmosphere()
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


def _inversion_height_jax(rtm, zt, nzt, thresh=8.0e-3):
    """Differentiable jnp mirror of the inversion-height loop in `_simple_rad_lw` (used only under a
    jax.grad trace). k_iso = first level (from the bottom) with rtm <= thresh; z_i is rtm=thresh linearly
    interpolated between k_iso-1 and k_iso. Forward-identical to the loop: k_iso==0 -> 0; all-above -> use the
    top two levels; the |denom|<eps fallback -> midpoint. The integer index is data-dependent (piecewise
    constant), the interpolated VALUE carries the gradient w.r.t. rtm."""
    above = rtm > thresh                                          # (ngrdcol, nzt)
    k_iso = jnp.sum(jnp.cumprod(above.astype(rtm.dtype), axis=1), axis=1).astype(jnp.int32)  # leading-True count
    k = jnp.clip(k_iso, 1, nzt - 1)                               # gatherable: k-1>=0 and k<nzt
    km1 = k - 1
    r_hi = jnp.take_along_axis(rtm, k[:, None], axis=1)[:, 0]
    r_lo = jnp.take_along_axis(rtm, km1[:, None], axis=1)[:, 0]
    z_hi = jnp.take_along_axis(zt, k[:, None], axis=1)[:, 0]
    z_lo = jnp.take_along_axis(zt, km1[:, None], axis=1)[:, 0]
    denom = r_hi - r_lo
    small = jnp.abs(denom) < _EPS
    denom_safe = jnp.where(small, 1.0, denom)                     # avoid 0/0 in the masked grad
    z_interp = jnp.where(small, 0.5 * (z_hi + z_lo),
                         z_lo + (thresh - r_lo) * (z_hi - z_lo) / denom_safe)
    return jnp.where(k_iso == 0, 0.0, z_interp)                   # k_iso==0 -> z_i=0


def _simple_rad_lw(
    gr,
    ngrdcol: int,
    rho: np.ndarray,
    rho_zm: np.ndarray,
    rtm: np.ndarray,
    rcm: np.ndarray,
    exner: np.ndarray,
    f0: float,
    f1: float,
    kappa: float,
    l_rad_above_cloud: bool,
    l_sample: bool,
    sw=None,
):
    """Port of simple_rad from simple_rad_module.F90."""
    nzm = gr.zm.shape[1]
    nzt = gr.zt.shape[1]

    lwp = _liq_water_path(ngrdcol, nzm, nzt, rho, rcm, gr.invrs_dzt)

    if f1 > _EPS:
        frad_lw = f0 * _xp.exp(-kappa * lwp) + f1 * _xp.exp(-kappa * (lwp[:, 0:1] - lwp))
    else:
        frad_lw = f0 * _xp.exp(-kappa * lwp)

    if l_rad_above_cloud:
        # Above-cloud LW correction. Block-level tracer dispatch (REFACTOR B5): the inversion-height search is
        # a data-dependent threshold-crossing loop on rtm and the correction uses boolean-mask in-place writes
        # + fractional powers — so concrete runs keep the EXACT original (bit-identical) while a jax.grad trace
        # takes a jnp mirror (jnp inversion finder, mask-MULTIPLY instead of boolean-index, _safe_pow for the
        # dz**(1/3)/(4/3) cloud-top inf-grad).
        if not _is_tracer_arg([rtm, frad_lw]):
            z_i = np.zeros(ngrdcol, dtype=np.float64)
            for i in range(ngrdcol):
                k_iso = 0
                while k_iso < nzt and rtm[i, k_iso] > 8.0e-3:
                    k_iso += 1
                if k_iso == 0 or k_iso > nzt:
                    z_i[i] = 0.0
                    continue
                if k_iso == nzt:
                    k_iso = nzt - 1
                z_i[i] = _linear_interp(
                    8.0e-3, rtm[i, k_iso], rtm[i, k_iso - 1],
                    gr.zt[i, k_iso], gr.zt[i, k_iso - 1],
                )
            dz = gr.zm - z_i[:, np.newaxis]
            heaviside = np.where(dz < -_EPS, 0.0, np.where(dz > _EPS, 1.0, 0.5))
            pos = heaviside > 0.0
            if np.any(pos):
                dz_pos = np.maximum(dz[pos], 0.0)
                z_i_broad = np.broadcast_to(z_i[:, np.newaxis], (ngrdcol, nzm))[pos]
                frad_lw[pos] += (
                    rho_zm[pos] * _CP * _LS_DIV * heaviside[pos]
                    * (0.25 * (dz_pos ** (4.0 / 3.0)) + z_i_broad * (dz_pos ** (1.0 / 3.0)))
                )
            if l_sample and sw is not None:
                sw.update("z_inversion", z_i)
        else:
            z_i = _inversion_height_jax(rtm, gr.zt, nzt)
            dz = jnp.asarray(gr.zm) - z_i[:, None]
            heaviside = jnp.where(dz < -_EPS, 0.0, jnp.where(dz > _EPS, 1.0, 0.5))
            dz_pos = jnp.maximum(dz, 0.0)
            correction = (
                jnp.asarray(rho_zm) * _CP * _LS_DIV * heaviside
                * (0.25 * _safe_pow(dz_pos, 4.0 / 3.0) + z_i[:, None] * _safe_pow(dz_pos, 1.0 / 3.0))
            )
            frad_lw = frad_lw + correction

    radht_lw = (
        (1.0 / exner) * (-1.0 / (_CP * rho))
        * (frad_lw[:, 1:] - frad_lw[:, :-1]) * gr.invrs_dzt
    )

    return frad_lw, radht_lw


def _liq_water_path(ngrdcol: int, nzm: int, nzt: int,
                    rho: np.ndarray, rcm: np.ndarray,
                    invrs_dzt: np.ndarray) -> np.ndarray:
    """Compute liquid water path on momentum levels (cumulative from the top down).

    Tracer-transparent (REFACTOR B5): vectorized reverse-cumsum that sums in the SAME top-down order as the
    original in-place loop (`lwp[k] = lwp[k+1] + contrib[k]`), so it is bit-identical for concrete runs while
    routing through jnp under a jax.grad trace (the loop's `lwp[:,k] = <tracer>` would otherwise sever)."""
    contrib = rcm * rho / invrs_dzt                                   # (ngrdcol, nzt)
    rev_cumsum = _xp.flip(_xp.cumsum(_xp.flip(contrib, axis=1), axis=1), axis=1)  # lwp[:, :nzt], top-down order
    zeros_top = _xp.zeros((ngrdcol, 1), dtype=np.float64)             # lwp[:, nzm-1] = 0 (top boundary)
    return _xp.concatenate([rev_cumsum, zeros_top], axis=1)           # (ngrdcol, nzm)


def _linear_interp(x: float, x_high: float, x_low: float,
                   y_high: float, y_low: float) -> float:
    denom = x_high - x_low
    if abs(denom) < _EPS:
        return 0.5 * (y_high + y_low)
    return y_low + (x - x_low) * (y_high - y_low) / denom


# ── cos_solar_zen ──────────────────────────────────────────────────────────────

def _gregorian2julian_day(day: int, month: int, year: int) -> int:
    """Julian day number (day of year, 1-based). Port of gregorian2julian_day."""
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if _is_leap_year(year):
        days_in_month[2] = 29
    return sum(days_in_month[:month]) + day


def _is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _compute_current_date(day: int, month: int, year: int, current_time_s: float):
    """Advance start date by current_time_s seconds. Returns (day, month, year, time_in_day_s)."""
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    total_seconds = current_time_s
    total_days = int(total_seconds // 86400)
    time_in_day = total_seconds - total_days * 86400.0

    d, m, y = day, month, year
    remaining = total_days
    while remaining > 0:
        dim = [0, 31, 29 if _is_leap_year(y) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        days_left_in_month = dim[m] - d + 1
        if remaining < days_left_in_month:
            d += remaining
            remaining = 0
        else:
            remaining -= days_left_in_month
            m += 1
            if m > 12:
                m = 1
                y += 1
            d = 1
    return d, m, y, time_in_day


def cos_solar_zen(day: int, month: int, year: int,
                  current_time: float,
                  lat_in_degrees: float, lon_in_degrees: float) -> float:
    """Cosine of solar zenith angle. Port of cos_solar_zen_module.F90."""
    present_day, present_month, present_year, present_time = \
        _compute_current_date(day, month, year, current_time)

    jul_day = _gregorian2julian_day(present_day, present_month, present_year)
    days_in_year = 366 if _is_leap_year(present_year) else 365

    hour = present_time / sec_per_hr
    t = 2.0 * math.pi * (jul_day - 1) / days_in_year

    delta = (_CSZ_C0
             + _CSZ_C1 * math.cos(t)   + _CSZ_D1 * math.sin(t)
             + _CSZ_C2 * math.cos(2*t) + _CSZ_D2 * math.sin(2*t)
             + _CSZ_C3 * math.cos(3*t) + _CSZ_D3 * math.sin(3*t))

    h = int(hour)
    if 0 <= h <= 11:
        zln = 180.0 - hour * 15.0
    elif 12 <= h <= 23:
        zln = 540.0 - hour * 15.0
    else:
        raise ValueError(f"Hour={hour} > 24 in cos_solar_zen")

    longang = abs(lon_in_degrees - zln) * radians_per_deg
    latang = lat_in_degrees * radians_per_deg

    return (math.sin(latang) * math.sin(delta)
            + math.cos(latang) * math.cos(delta) * math.cos(longang))


# ── sunray_sw ──────────────────────────────────────────────────────────────────

def sunray_sw(ngrdcol: int, nzt: int,
              rcm: np.ndarray, rho: np.ndarray,
              xi_abs: float, dzt: np.ndarray,
              zm: np.ndarray, zt: np.ndarray,
              radius: float, A: float, gc: float,
              Fs0: float, omega: float, l_center: bool) -> np.ndarray:
    """Shortwave flux. Port of rad_lwsw_module.F90:sunray_sw (lines 343-755).

    Returns Frad_SW shape (ngrdcol, nzt+1) on momentum levels (bottom-up).
    """
    three_halves = 1.5

    # Per-layer optical depth  tau(i,k) = 1.5 * rcm * rho * dzt / radius / rho_lw
    tau = three_halves * rcm * rho * dzt / radius / rho_lw   # (ngrdcol, nzt)

    # Column total optical depth
    tauc = tau.sum(axis=1)   # (ngrdcol,)

    # Delta-Eddington transformation (Duynkerke eqn.18)
    ff = gc * gc
    gcde = gc / (1.0 + gc)
    omegade = (1.0 - ff) * omega / (1.0 - omega * ff)
    taude = (1.0 - omega * ff) * tau   # (ngrdcol, nzt)

    # Constants (scalar, same for all columns)
    x1 = 1.0 - omegade * gcde
    x2 = 1.0 - omegade
    rk = math.sqrt(3.0 * x2 * x1)
    xi_abs2 = xi_abs * xi_abs
    rk2 = rk * rk
    x3 = 4.0 * (1.0 - rk2 * xi_abs2)
    rp = math.sqrt(3.0 * x2 / x1)
    alpha = 3.0 * omegade * xi_abs2 * (1.0 + gcde * x2) / x3
    beta = 3.0 * omegade * xi_abs * (1.0 + 3.0 * gcde * xi_abs2 * x2) / x3

    rtt = 2.0 / 3.0
    xp23p = 1.0 + rtt * rp
    xm23p = 1.0 - rtt * rp
    ap23b = alpha + rtt * beta
    t1 = 1.0 - A - rtt * (1.0 + A) * rp
    t2 = 1.0 - A + rtt * (1.0 + A) * rp
    t3 = (1.0 - A) * alpha - rtt * (1.0 + A) * beta + A * xi_abs

    # Per-column: column total D-E optical depth, C1, C2
    taucde = (1.0 - omega * ff) * tauc   # (ngrdcol,)
    exmu0 = np.exp(-taucde / xi_abs)
    expk = np.exp(rk * taucde)
    exmk = 1.0 / expk

    c2 = (xp23p * t3 * exmu0 - t1 * ap23b * exmk) / (xp23p * t2 * expk - xm23p * t1 * exmk)
    c1 = (ap23b - c2 * xm23p) / xp23p   # both shape (ngrdcol,)

    # Flux computation on momentum levels: sequential taupath accumulation per column
    Frad_SW = np.zeros((ngrdcol, nzt + 1), dtype=np.float64)

    for i in range(ngrdcol):
        # Top momentum level (k = nzt+1, Python index nzt)
        taupath = 0.5 * taude[i, nzt - 1] if l_center else 0.0

        def _flux(tp):
            F_diff = (-4.0 / 3.0) * Fs0 * (
                rp * (c1[i] * math.exp(-rk * tp) - c2[i] * math.exp(rk * tp))
                - beta * math.exp(-tp / xi_abs)
            )
            F_dir = -Fs0 * xi_abs * math.exp(-tp / xi_abs)
            return F_diff + F_dir

        Frad_SW[i, nzt] = _flux(taupath)

        # Interior levels k = nzt-1 down to 1 (Python indices nzt-1 down to 1)
        for k_py in range(nzt - 1, 0, -1):
            k_fort = k_py + 1  # Fortran 1-based: k goes nzt down to 2
            if l_center:
                # lin_interpolate_two_points(zm[k], zt[k], zt[k-1], taude[k], taude[k-1])
                zm_k = zm[i, k_py]
                zt_k = zt[i, k_py]
                zt_km1 = zt[i, k_py - 1]
                td_k = taude[i, k_py]
                td_km1 = taude[i, k_py - 1]
                denom = zt_k - zt_km1
                if abs(denom) < 1e-300:
                    interp = 0.5 * (td_k + td_km1)
                else:
                    interp = td_km1 + (zm_k - zt_km1) * (td_k - td_km1) / denom
                taupath += interp
            else:
                taupath += taude[i, k_py - 1]
            Frad_SW[i, k_py] = _flux(taupath)

        # Bottom momentum level (k = 1, Python index 0)
        taupath += taude[i, 0]
        Frad_SW[i, 0] = _flux(taupath)

    return Frad_SW


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
