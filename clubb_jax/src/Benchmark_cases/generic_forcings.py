"""Generic prescribe_forcings for non-ARM benchmark cases.

Port of src/Benchmark_cases/prescribe_forcings.F90 and supporting modules.
Each case follows the pattern from the Fortran dispatch:
  1. Zero all forcing arrays
  2. If l_t_dependent: apply_time_dependent_forcings (reads from {case}_forcings.in)
  3. Else: analytic case-specific tndcy
  4. Read bottom-level state for surface BCs
  5. Case-specific sfclyr
  6. Momentum flux, scalar sfc flux, stats
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import math

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Lv
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_liq_jax
from clubb_jax.src.CLUBB_core.grid_class import zt2zm_jax

_EPS64 = np.finfo(np.float64).eps
_BLANK = -999.9
_UBMIN = 0.25   # m/s (compute_ubar ubmin)
_SEC_PER_HR = 3600.0
_SEC_PER_DAY = 86400.0
_PI = math.pi


# ── Unit conversion helpers (spec_hum_to_mixing_ratio.F90) ─────────────────

def _flux_spec_hum_to_mixing_ratio(rtm_sfc: np.ndarray, wpqtp: np.ndarray) -> np.ndarray:
    """w'r_t' = (1 + r_tm)^2 * w'q_t'  (linearised)."""
    return (1.0 + rtm_sfc) ** 2 * wpqtp


def _force_spec_hum_to_mixing_ratio(rtm: np.ndarray, qtm_forcing: np.ndarray) -> np.ndarray:
    """rtm_forcing = (1 + rtm)^2 * qtm_forcing  (linearised)."""
    return (1.0 + rtm) ** 2 * qtm_forcing


# ── Surface BC extraction (prescribe_forcings.F90:read_surface_var_for_bc) ─

_P0_BC = 1.0e5
_KAPPA_BC = 287.04 / 1004.67
_Z_BOT_CNVG = 25.0  # fixed model height for the convergence-test BC


def _fsign(x: np.ndarray) -> np.ndarray:
    """Fortran sign(1.0, x): +1 for x >= 0, -1 for x < 0 (unlike numpy.sign at 0)."""
    return np.where(x >= 0.0, 1.0, -1.0)


def _mono_cubic_interp(z_in, km1, k00, kp1, kp2,
                       zm1, z00, zp1, zp2, fm1, f00, fp1, fp2):
    """Steffen monotone cubic interpolation (interpolation.F90:mono_cubic_interp,
    l_quintic_poly_interp=.false.). Stencils used here always satisfy km1 <= k00."""
    h00 = zp1 - z00
    if km1 == k00:
        hp1 = zp2 - zp1
        s00 = (fp1 - f00) / (zp1 - z00)
        sp1 = (fp2 - fp1) / (zp2 - zp1)
        dfdx00 = s00
        pp1 = (s00 * hp1 + sp1 * h00) / (h00 + hp1)
        dfdxp1 = (_fsign(s00) + _fsign(sp1)) * min(abs(s00), abs(sp1), 0.5 * abs(pp1))
    elif kp1 == kp2:
        hm1 = z00 - zm1
        sm1 = (f00 - fm1) / (z00 - zm1)
        s00 = (fp1 - f00) / (zp1 - z00)
        p00 = (sm1 * h00 + s00 * hm1) / (hm1 + h00)
        dfdx00 = (_fsign(sm1) + _fsign(s00)) * min(abs(sm1), abs(s00), 0.5 * abs(p00))
        dfdxp1 = s00
    else:
        hm1 = z00 - zm1
        hp1 = zp2 - zp1
        sm1 = (f00 - fm1) / (z00 - zm1)
        s00 = (fp1 - f00) / (zp1 - z00)
        sp1 = (fp2 - fp1) / (zp2 - zp1)
        p00 = (sm1 * h00 + s00 * hm1) / (hm1 + h00)
        pp1 = (s00 * hp1 + sp1 * h00) / (h00 + hp1)
        dfdx00 = (_fsign(sm1) + _fsign(s00)) * min(abs(sm1), abs(s00), 0.5 * abs(p00))
        dfdxp1 = (_fsign(s00) + _fsign(sp1)) * min(abs(s00), abs(sp1), 0.5 * abs(pp1))
    c1 = (dfdx00 + dfdxp1 - 2.0 * s00) / (h00 ** 2)
    c2 = (3.0 * s00 - 2.0 * dfdx00 - dfdxp1) / h00
    zprime = z_in - z00
    return f00 + zprime * (dfdx00 + zprime * (c2 + zprime * c1))


def _read_surface_var_for_bc(state: dict) -> dict:
    """Extract bottom-level state values used to compute surface fluxes.

    Mirrors prescribe_forcings.F90:read_surface_var_for_bc.
    - Default (l_modify_bc_for_cnvg_test=False): lowest zt level.
    - Convergence-test (l_modify_bc_for_cnvg_test=True, constant_height_option=2):
      values mono-cubic-interpolated to a fixed height (25 m) on the zt2zm grid.
    Returns dict with um_bot, vm_bot, rtm_bot, thlm_bot, rho_bot, exner_bot, z_bot.
    """
    gr = state['gr']
    if not state.get('l_modify_bc_for_cnvg_test', False):
        return {
            'z_bot':     gr.zt[:, 0].copy(),
            'um_bot':    state['um'][:, 0].copy(),
            'vm_bot':    state['vm'][:, 0].copy(),
            'rtm_bot':   state['rtm'][:, 0].copy(),
            'thlm_bot':  state['thlm'][:, 0].copy(),
            'rho_bot':   state['rho_zm'][:, 0].copy(),
            'exner_bot': (state['p_sfc'] / _P0_BC) ** _KAPPA_BC,
        }

    # l_modify_bc_for_cnvg_test = True, constant_height_option = 2 (interpolation)
    ngrdcol = state['ngrdcol']
    zt = np.asarray(gr.zt)
    zm = np.asarray(gr.zm)
    rho_zm = np.asarray(state['rho_zm'])
    z_bot = _Z_BOT_CNVG

    um_zm    = np.asarray(zt2zm_jax(jnp.asarray(state['um']), gr))
    vm_zm    = np.asarray(zt2zm_jax(jnp.asarray(state['vm']), gr))
    thlm_zm  = np.asarray(zt2zm_jax(jnp.asarray(state['thlm']), gr))
    rtm_zm   = np.asarray(zt2zm_jax(jnp.asarray(state['rtm']), gr))
    exner_zm = np.array(zt2zm_jax(jnp.asarray(state['exner']), gr))
    exner_zm[:, 0] = (state['p_sfc'] / _P0_BC) ** _KAPPA_BC

    out = {k: np.empty(ngrdcol) for k in
           ('z_bot', 'um_bot', 'vm_bot', 'rtm_bot', 'thlm_bot', 'rho_bot', 'exner_bot')}
    out['z_bot'][:] = z_bot

    for i in range(ngrdcol):
        kmin = int(np.argmin(np.abs(zt[i] - z_bot)))  # nearest zt level (0-based)
        if kmin == 0:
            km1, k00, kp1, kp2 = 0, 0, 1, 2
        else:
            km1, k00, kp1, kp2 = kmin - 1, kmin, kmin + 1, kmin + 2
        zi = zm[i]
        out['rho_bot'][i] = rho_zm[i, kmin]
        for key, src in (('um_bot', um_zm), ('vm_bot', vm_zm),
                         ('thlm_bot', thlm_zm), ('rtm_bot', rtm_zm),
                         ('exner_bot', exner_zm)):
            out[key][i] = _mono_cubic_interp(
                z_bot, km1, k00, kp1, kp2,
                zi[km1], zi[k00], zi[kp1], zi[kp2],
                src[i, km1], src[i, k00], src[i, kp1], src[i, kp2])
    return out


# ── Shared surface utilities (sfc_flux.F90) ────────────────────────────────

def _compute_ubar(um_bot: np.ndarray, vm_bot: np.ndarray) -> np.ndarray:
    """Mean surface wind speed, lower-bounded by ubmin=0.25 m/s."""
    return np.maximum(_UBMIN, np.sqrt(um_bot ** 2 + vm_bot ** 2))


def _compute_momentum_flux(um_bot: np.ndarray, vm_bot: np.ndarray,
                           ubar: np.ndarray, ustar: np.ndarray):
    """upwp_sfc = -um * ustar^2 / ubar;  vpwp_sfc = -vm * ustar^2 / ubar."""
    ustar2 = ustar ** 2
    upwp_sfc = -um_bot * ustar2 / ubar
    vpwp_sfc = -vm_bot * ustar2 / ubar
    return upwp_sfc, vpwp_sfc


def _set_sclr_sfc_rtm_thlm(state: dict, wpthlp_sfc: np.ndarray,
                            wprtp_sfc: np.ndarray) -> None:
    """Copy wpthlp/wprtp to scalar surface fluxes (set_sclr_sfc_rtm_thlm)."""
    sclr_idx = state['sclr_idx']
    nzm = state['nzm']
    ngrdcol = state['ngrdcol']
    if state['sclr_dim'] > 0:
        if sclr_idx.iisclr_thl > 0:
            k = sclr_idx.iisclr_thl - 1
            state['wpsclrp'][:, :, k] = wpthlp_sfc[:, np.newaxis] * np.ones((ngrdcol, nzm))
        if sclr_idx.iisclr_rt > 0:
            k = sclr_idx.iisclr_rt - 1
            state['wpsclrp'][:, :, k] = wprtp_sfc[:, np.newaxis] * np.ones((ngrdcol, nzm))


def _stats_surface_update(state: dict, wpthlp_sfc, wprtp_sfc, upwp_sfc,
                          vpwp_sfc, ustar, T_sfc, l_sample: bool) -> None:
    """Mirrors Fortran stats_update calls in the surface section."""
    sw = state.get('stats_writer')
    if not l_sample or sw is None:
        return
    rho_zm_sfc = state['rho_zm'][:, 0]
    sw.update("sh", wpthlp_sfc * rho_zm_sfc * Cp)
    sw.update("lh", wprtp_sfc * rho_zm_sfc * Lv)
    sw.update("wpthlp_sfc", wpthlp_sfc)
    sw.update("wprtp_sfc", wprtp_sfc)
    sw.update("upwp_sfc", upwp_sfc)
    sw.update("vpwp_sfc", vpwp_sfc)
    sw.update("ustar", ustar)
    sw.update("T_sfc", T_sfc)


# ── Time interpolation helper ───────────────────────────────────────────────

def _time_interp(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation of values[..., i] at time t using times array."""
    t = np.clip(t, times[0], times[-1])
    idx = np.searchsorted(times, t, side='right') - 1
    idx = np.clip(idx, 0, len(times) - 2)
    frac = (t - times[idx]) / (times[idx + 1] - times[idx] + 1e-300)
    return (1.0 - frac) * values[..., idx] + frac * values[..., idx + 1]


# ── Generic time-dependent forcing (apply_time_dependent_forcings) ──────────

def _apply_time_dependent_forcings(state: dict, time_current: float) -> None:
    """Apply pre-loaded large-scale forcings for current time (generic cases).

    Reads from state['_forcings_data'] which is loaded at init time.
    Mirrors apply_time_dependent_forcings in time_dependent_input.F90.
    """
    fd = state.get('_forcings_data')
    if fd is None or fd.get('times') is None or len(fd['times']) == 0:
        return

    times = fd['times']
    ngrdcol = state['ngrdcol']

    def _interp_col(key):
        arr = fd.get(key)
        if arr is None:
            return None
        if np.all(np.abs(arr + 999.9) < 1e-3):
            return None
        return _time_interp(times, arr, time_current)  # (nzt,)

    thlm_f = _interp_col('thlm_f')
    rtm_f  = _interp_col('rtm_f')
    sp_hmdty_f = _interp_col('sp_hmdty_f')
    w_zt   = _interp_col('w')
    omega  = _interp_col('omega')
    um_f   = _interp_col('um_f')
    vm_f   = _interp_col('vm_f')
    ug_zt  = _interp_col('ug')
    vg_zt  = _interp_col('vg')

    if thlm_f is not None:
        state['thlm_forcing'][:, :] = thlm_f[np.newaxis, :]
        state['thlm_forcing'][:, -1] = 0.0  # zero at top (Fortran convention)
    if rtm_f is not None:
        state['rtm_forcing'][:, :] = rtm_f[np.newaxis, :]
        state['rtm_forcing'][:, -1] = 0.0
    elif sp_hmdty_f is not None:
        # specific-humidity forcing → rtm forcing: rtm_f = sp_hmdty_f·(1+rtm)² (time_dependent_input.F90:725)
        rtm = np.asarray(state['rtm'], dtype=np.float64)
        state['rtm_forcing'][:, :] = sp_hmdty_f[np.newaxis, :] * (1.0 + rtm) ** 2
        state['rtm_forcing'][:, -1] = 0.0
    # subsidence: either a direct w[m/s] column, or omega (pressure velocity) converted with the current
    # density: wm_zt = -omega/(grav·rho) (time_dependent_input.F90:815/830; mb/hr → ×100/3600 first).
    if omega is not None:
        from clubb_jax.src.CLUBB_core.constants_clubb import grav, sec_per_hr
        fac = (100.0 / sec_per_hr) if fd.get('omega_mb_hr') else 1.0   # pascal_per_mb=100 (constants_clubb.F90)
        rho = np.asarray(state['rho'], dtype=np.float64)
        w_zt = -(omega[np.newaxis, :] * fac) / (grav * rho)
        state['wm_zt'][:, :] = w_zt
        state['wm_zm'][:, :] = np.asarray(
            zt2zm_jax(jnp.asarray(state['wm_zt']), state['gr']), dtype=np.float64)
    elif w_zt is not None:
        state['wm_zt'][:, :] = w_zt[np.newaxis, :]
        # Recompute wm_zm = zt2zm(wm_zt) each step the subsidence forcing updates wm_zt
        # (time_dependent_input.F90:837). Without this, wm_zm stayed at its init value (0 when the
        # subsidence enters only via the forcing, as for jun25_altocu) → the xp2/xpyp mean-advection
        # by subsidence (thlp2_ma etc.) was missing, seeding a cold-cloud divergence. Raw zt2zm is
        # the faithful form (top→0, bottom→interpolated; verified vs the Fortran wm_zm to 4e-10) —
        # do NOT zero the bottom. No-op for cases whose subsidence is set at init (no w forcing column).
        state['wm_zm'][:, :] = np.asarray(
            zt2zm_jax(jnp.asarray(state['wm_zt']), state['gr']), dtype=np.float64)
    # u/v momentum forcing and time/height-dependent geostrophic wind
    # (apply_time_dependent_forcings in time_dependent_input.F90). Missing these
    # broke gabls3_night (um_f / ug vary with height & time).
    if um_f is not None and 'um_forcing' in state:
        state['um_forcing'][:, :] = um_f[np.newaxis, :]
    if vm_f is not None and 'vm_forcing' in state:
        state['vm_forcing'][:, :] = vm_f[np.newaxis, :]
    if ug_zt is not None and state.get('ug') is not None:
        state['ug'][:, :] = ug_zt[np.newaxis, :]
    if vg_zt is not None and state.get('vg') is not None:
        state['vg'][:, :] = vg_zt[np.newaxis, :]


# ── BOMEX ───────────────────────────────────────────────────────────────────

def _bomex_tndcy(state: dict) -> None:
    """Analytic large-scale tendencies for BOMEX (bomex.F90:bomex_tndcy).

    Moisture tendency in terms of specific humidity (qtm), then converted
    to mixing ratio. No heat tendency (thlm_forcing = 0).
    """
    zt = state['gr'].zt   # (ngrdcol, nzt)
    rtm = state['rtm']

    qtm_forcing = np.where(
        zt < 300.0, -1.2e-8,
        np.where(zt < 500.0, -1.2e-8 * (1.0 - (zt - 300.0) / 200.0), 0.0)
    )
    state['thlm_forcing'][:] = 0.0
    state['rtm_forcing'][:] = _force_spec_hum_to_mixing_ratio(rtm, qtm_forcing)


def _bomex_sfclyr(state: dict, time_current: float, ngrdcol: int,
                  rtm_bot: np.ndarray) -> tuple:
    """BOMEX surface fluxes from time-interpolated sfc file (bomex.F90:bomex_sfclyr).

    rtm_bot is the bottom-level total water mixing ratio from read_surface_var_for_bc
    (at z_bot=25 m for the convergence-test BC, else lowest zt level).
    Returns: wpthlp_sfc, wprtp_sfc, ustar  (all shape (ngrdcol,))
    """
    fd = state['_forcings_data']
    sfc = fd['sfc']
    wpthlp_sfc_val = float(np.interp(time_current, sfc['time'], sfc['wpthlp_sfc']))
    wpqtp_sfc_val  = float(np.interp(time_current, sfc['time'], sfc['wpqtp_sfc']))
    ustar_val = 0.28

    wpqtp_sfc = np.full(ngrdcol, wpqtp_sfc_val)
    wpthlp_sfc = np.full(ngrdcol, wpthlp_sfc_val)
    ustar = np.full(ngrdcol, ustar_val)
    wprtp_sfc = _flux_spec_hum_to_mixing_ratio(rtm_bot, wpqtp_sfc)
    return wpthlp_sfc, wprtp_sfc, ustar


# ── Simple zero-forcing cases ───────────────────────────────────────────────

def _zero_forcings(state: dict) -> None:
    """Zero all large-scale forcing arrays (fire/generic/neutral/coriolis_test/ekman)."""
    state['thlm_forcing'][:] = 0.0
    state['rtm_forcing'][:] = 0.0


# ── RICO ─────────────────────────────────────────────────────────────────────

def _rico_sfclyr(state: dict, time_current: float, ngrdcol: int,
                 um_bot, vm_bot, thlm_bot, rtm_bot, rho_bot, exner_bot,
                 z_bot, ubar) -> tuple:
    """RICO surface fluxes — RICO 3D spec (rico.F90:rico_sfclyr, l_use_old_atex=False).

    Uses bulk aerodynamic coefficients C_m_20, C_h_20, C_q_20 scaled to model height.
    T_sfc interpolated from rico_sfc.in.
    """
    C_m_20 = 0.001229
    C_h_20 = 0.001094
    C_q_20 = 0.001133
    z0     = 0.00015
    z_ref  = 20.0

    fd = state['_forcings_data']
    sfc = fd.get('sfc')
    if sfc is not None and 't_sfc' in sfc:
        T_sfc_val = float(np.interp(time_current, sfc['time'], sfc['t_sfc']))
    else:
        T_sfc_val = 299.8
    T_sfc = np.full(ngrdcol, T_sfc_val)

    log_ref_z0 = np.log(z_ref / z0)
    log_z_z0   = np.log(np.maximum(z_bot, z0 * 1.001) / z0)
    scale      = (log_ref_z0 / log_z_z0) ** 2

    Cm = C_m_20 * scale
    Ch = C_h_20 * scale
    Cq = C_q_20 * scale

    ustar = np.full(ngrdcol, 0.3)  # Rico spec constant

    # sat_mixrat_liq_jax expects scalar saturation_formula
    sat_formula = state['flags'].saturation_formula
    p_sfc = state['p_sfc']
    rsat = np.array([float(sat_mixrat_liq_jax(
        float(p_sfc[i]), T_sfc_val, sat_formula)) for i in range(ngrdcol)])

    # compute_wpthlp_sfc: -Cd * ubar * (thlm - T_sfc/exner)
    wpthlp_sfc = -Ch * ubar * (thlm_bot - T_sfc / exner_bot)
    # compute_wprtp_sfc: -Cd * ubar * (rtm - rsat)
    wprtp_sfc = -Cq * ubar * (rtm_bot - rsat)
    # Momentum flux: -Cm * ubar (not using ustar^2/ubar pattern for RICO)
    upwp_sfc = -um_bot * Cm * ubar
    vpwp_sfc = -vm_bot * Cm * ubar

    return wpthlp_sfc, wprtp_sfc, ustar, upwp_sfc, vpwp_sfc, T_sfc


# ── DYCOMS2-RF01 ──────────────────────────────────────────────────────────────

def _dycoms2_rf01_sfclyr(state: dict, time_current: float, ngrdcol: int,
                         rho_bot, ubar=None, thlm_bot=None, rtm_bot=None, exner_bot=None) -> tuple:
    """DYCOMS-II RF01 surface fluxes (dycoms2_rf01.F90:dycoms2_rf01_sfclyr). ustar=0.25 (GCSS spec).
    sfctype=0: prescribed sens_ht/latent_ht → /(rho·Cp), /(rho·Lv). sfctype=1 (fixed SST): bulk flux
    with Cd=0.0011, T_sfc from the sfc file: wpthlp_sfc=-Cd·ubar·(thlm-T_sfc/exner),
    wprtp_sfc=-Cd·ubar·(rtm-rsat). Reads sens_ht/latent_ht/t_sfc from dycoms2_rf01[_fixed_sst]_sfc.in."""
    fd = state['_forcings_data']
    sfc = fd.get('sfc') or {}
    times = sfc.get('time', np.array([0.0]))

    def _interp(key, default):
        arr = sfc.get(key)
        if arr is not None and len(arr) > 0:
            return float(np.interp(time_current, times, arr))
        return default

    T_sfc_val = _interp('t_sfc', 292.0)
    ustar = np.full(ngrdcol, 0.25)
    T_sfc = np.full(ngrdcol, T_sfc_val)

    if int(state.get('sfctype', 0)) == 1:                # fixed-SST bulk flux (dycoms2_rf01_fixed_sst)
        Cd = 0.0011
        sat_formula = state['flags'].saturation_formula
        p_sfc = np.asarray(state['p_sfc'], dtype=np.float64)
        rsat = np.array([float(sat_mixrat_liq_jax(float(p_sfc[i]), T_sfc_val, sat_formula))
                         for i in range(ngrdcol)])
        wpthlp_sfc = -Cd * ubar * (thlm_bot - T_sfc / exner_bot)
        wprtp_sfc  = -Cd * ubar * (rtm_bot - rsat)
    else:                                                # sfctype=0: prescribed sens/latent heat
        sens_ht   = _interp('sens_ht',   16.0)   # W/m^2 — DYCOMS RF01 spec
        latent_ht = _interp('latent_ht', 93.0)
        wpthlp_sfc = sens_ht   / (rho_bot * Cp)
        wprtp_sfc  = latent_ht / (rho_bot * Lv)
    return wpthlp_sfc, wprtp_sfc, ustar, T_sfc


# ── DYCOMS2-RF02 ──────────────────────────────────────────────────────────────

def _dycoms2_rf02_sfclyr(state: dict, time_current: float, ngrdcol: int) -> tuple:
    """DYCOMS-II RF02 surface fluxes (dycoms2_rf02.F90:dycoms2_rf02_sfclyr): time-interpolated
    sens_ht/latent_ht converted with rho_sfc=1.21 (CONSTANT): wpthlp_sfc=sens_ht/(1.21·Cp),
    wprtp_sfc=latent_ht/(1.21·Lv); ustar=0.25. (The sfc file has sens_ht/latent_ht, not wpthlp_sfc.)"""
    from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Lv
    rho_sfc = 1.21
    sfc = (state['_forcings_data'] or {}).get('sfc') or {}
    times = sfc.get('time', np.array([0.0]))
    sh = sfc.get('sens_ht'); lh = sfc.get('latent_ht')
    sens_ht = float(np.interp(time_current, times, sh)) if sh is not None else 0.0
    latent_ht = float(np.interp(time_current, times, lh)) if lh is not None else 0.0
    wpthlp_sfc = np.full(ngrdcol, sens_ht / (rho_sfc * Cp))
    wprtp_sfc  = np.full(ngrdcol, latent_ht / (rho_sfc * Lv))
    ustar      = np.full(ngrdcol, 0.25)
    return wpthlp_sfc, wprtp_sfc, ustar


# ── Wangara ───────────────────────────────────────────────────────────────────

def _wangara_sfclyr(time_current: float, ngrdcol: int) -> tuple:
    """Wangara day 33 analytic surface fluxes (wangara.F90:wangara_sfclyr).

    Cosine formula in AEST local time (UTC+10h). ustar = 0.13.
    """
    time_utc = time_current % _SEC_PER_DAY
    time_est = (time_utc + 36000.0) % _SEC_PER_DAY  # +10h → AEST
    ustar = np.full(ngrdcol, 0.13)
    wpthlp_val = 0.18 * math.cos((time_est - 45000.0) / 36000.0 * _PI)
    wpthlp_sfc = np.full(ngrdcol, wpthlp_val)
    wprtp_sfc  = 1.3e-4 * wpthlp_sfc
    return wpthlp_sfc, wprtp_sfc, ustar


# ── LBA ───────────────────────────────────────────────────────────────────────

def _lba_sfclyr(state: dict, time_current: float, ngrdcol: int,
                z_bot, rho_bot, thlm_bot, ubar) -> tuple:
    """LBA analytic surface fluxes (lba.F90:lba_sfclyr).

    Cosine formula based on elapsed time. Uses diag_ustar (same as ARM).
    """
    from clubb_jax.src.Benchmark_cases.arm import _diag_ustar
    z0   = 0.035
    grav = 9.81
    time_elapsed = time_current - state['time_initial']
    ft = max(0.0, math.cos(0.5 * _PI * (5.25 - time_elapsed / _SEC_PER_HR) / 5.25))

    sh = 270.0 * ft ** 1.5
    lh = 554.0 * ft ** 1.3

    wpthlp_sfc = sh   / (rho_bot * Cp)
    wprtp_sfc  = lh   / (rho_bot * Lv)

    ustar = np.array([
        _diag_ustar(float(z_bot[i]),
                    grav / float(thlm_bot[i]) * float(wpthlp_sfc[i]),
                    float(ubar[i]), z0)
        for i in range(ngrdcol)
    ])
    return wpthlp_sfc, wprtp_sfc, ustar


# ── GABLS2 ───────────────────────────────────────────────────────────────────

def _atex_tndcy(state: dict, time_current: float) -> None:
    """ATEX analytic large-scale tendencies (atex.F90:atex_tndcy). The WHOLE package — subsidence AND
    the thlm/rtm large-scale forcing — is OFF for the first 90 min (`time >= time_initial + 5400`,
    atex.F90:215); then `z_inversion` = the zt level just below the first level where rtm ≤ 6.5e-3, and:
      wm_zt    = -0.0065·z/z_inv (0<z≤z_inv), a linear ramp to 0 over the next 300 m, else 0;
      thlm_forcing = -1.1575e-5·(3 - z/z_inv) for 0<z<z_inv, -2.315e-5·(1-(z-z_inv)/300) in the next
                     300 m, else 0;  rtm_forcing = -1.58e-8·(1 - z/z_inv) for 0<z<z_inv, else 0
    (atex.F90:calc_forcings). NOTE the forcing uses a STRICT z < z_inv where the subsidence uses z ≤ z_inv.
    `wm_zm = zt2zm(wm_zt)` with zeroed surface/top BCs."""
    gr = state['gr']; ngrdcol = state['ngrdcol']
    if time_current < state['time_initial'] + 5400.0:
        state['thlm_forcing'][:] = 0.0
        state['rtm_forcing'][:] = 0.0
        state['wm_zt'][:] = 0.0
        state['wm_zm'][:] = 0.0
        return
    zt = np.asarray(gr.zt, dtype=np.float64)
    rtm = np.asarray(state['rtm'], dtype=np.float64)
    wm_zt = np.zeros_like(zt)
    thlm_f = np.zeros_like(zt)
    rtm_f = np.zeros_like(zt)
    for i in range(ngrdcol):
        j = int(np.argmax(rtm[i] <= 6.5e-3))          # first 0-based level with rtm ≤ 6.5e-3
        z_inv = zt[i, j - 1]                           # the level just below (Fortran zt(z_lev-1))
        z = zt[i]
        wm_zt[i] = np.where((z > 0.0) & (z <= z_inv), -0.0065 * z / z_inv,
                            np.where((z > z_inv) & (z <= z_inv + 300.0),
                                     -0.0065 * (1.0 - (z - z_inv) / 300.0), 0.0))
        thlm_f[i] = np.where((z > 0.0) & (z < z_inv),
                             -1.1575e-5 * (3.0 - z / z_inv),
                             np.where((z > z_inv) & (z <= z_inv + 300.0),
                                      -2.315e-5 * (1.0 - (z - z_inv) / 300.0), 0.0))
        rtm_f[i] = np.where((z > 0.0) & (z < z_inv), -1.58e-8 * (1.0 - z / z_inv), 0.0)
    state['thlm_forcing'][:] = thlm_f
    state['rtm_forcing'][:] = rtm_f
    state['wm_zt'][:] = wm_zt
    wm_zm = np.array(zt2zm_jax(jnp.asarray(wm_zt), gr), dtype=np.float64)
    wm_zm[:, 0] = 0.0
    wm_zm[:, -1] = 0.0
    state['wm_zm'][:] = wm_zm


def _atex_long_tndcy(state: dict, time_current: float) -> None:
    """Long-ATEX analytic large-scale tendencies (atex_long.F90:atex_long_tndcy). Unlike `atex`, the
    subsidence is a FIXED 3-piece profile (no rtm-based inversion, no 90-min gate) and there are nonzero
    thlm/rtm forcings (atex_long.F90:calc_forcings). During the first `spinup = 43200 s` all three
    (thlm_forcing, rtm_forcing, wm_zt) ramp linearly as `· time / spinup`. `wm_zm = zt2zm(wm_zt)`, BCs zeroed.

    wm_zt(z):   -0.00636·z/1050                       0≤z<1050
                -0.00636 - 0.00079·(z-1050)/600       1050≤z<1650
                -0.00715                               z≥1650
    thlm_forcing(z): -3.5805e-5                        0≤z<1400
                     -3.5805e-5 + 1.1935e-5·(z-1400)·0.004      1400≤z<1650
                     -2.3870e-5 - 0.1155e-5·(z-1650)/1350       1650≤z<2990
                     0                                          z≥2990
    rtm_forcing(z):  -1.58e-8·(1 - z/1050)             0≤z<1050 ; else 0
    """
    gr = state['gr']
    zt = np.asarray(gr.zt, dtype=np.float64)
    # subsidence (3-piece)
    wm_zt = np.where(zt < 1050.0, -0.00636 * zt / 1050.0,
                     np.where(zt < 1650.0, -0.00636 - 0.00079 * (zt - 1050.0) / 600.0,
                              -0.00715))
    # theta-l tendency (4-piece)
    thlm_forcing = np.where(zt < 1400.0, -3.5805e-5,
                            np.where(zt < 1650.0, -3.5805e-5 + 1.1935e-5 * (zt - 1400.0) * 0.004,
                                     np.where(zt < 2990.0, -2.3870e-5 - 0.1155e-5 * (zt - 1650.0) / 1350.0,
                                              0.0)))
    # moisture tendency (2-piece)
    rtm_forcing = np.where(zt < 1050.0, -1.58e-8 * (1.0 - zt / 1050.0), 0.0)
    # spin-up ramp (atex_long.F90:155-160)
    spinup = 43200.0
    if time_current < spinup:
        ramp = time_current / spinup
        thlm_forcing = thlm_forcing * ramp
        rtm_forcing = rtm_forcing * ramp
        wm_zt = wm_zt * ramp
    state['thlm_forcing'][:] = thlm_forcing
    state['rtm_forcing'][:] = rtm_forcing
    state['wm_zt'][:] = wm_zt
    wm_zm = np.array(zt2zm_jax(jnp.asarray(wm_zt), gr), dtype=np.float64)
    wm_zm[:, 0] = 0.0
    wm_zm[:, -1] = 0.0
    state['wm_zm'][:] = wm_zm


def _rico_tndcy(state: dict) -> None:
    """RICO analytic large-scale tendencies (rico.F90:rico_tndcy). Does NOT touch `wm` (subsidence is
    init-set and left unchanged). Both forcings are time-independent.

    thlm_forcing = t_tendency / exner, where t_tendency [K/s]:
        z < 4000 :  -2.51/86400 + (0.33/(86400·4000))·z
        4000≤z<5000: -2.18/86400 + (2.18/(86400·1000))·(z-4000)
        z ≥ 5000 :  0
    qtm_forcing (specific-humidity tendency [g/kg/s], then /g_per_kg → kg/kg/s):
        z < 3000 :  -1.0/86400 + (1.345/(86400·3000))·z
        3000≤z<4000: 0.345/86400
        4000≤z<5000: 0.345/86400 + (-0.345/(86400·1000))·(z-4000)
        z ≥ 5000 :  0
    rtm_forcing = (1 + rtm)^2 · qtm_forcing   (spec_hum_to_mixing_ratio:force_spec_hum_to_mixing_ratio).
    """
    gr = state['gr']
    zt = np.asarray(gr.zt, dtype=np.float64)
    exner = np.asarray(state['exner'], dtype=np.float64)
    rtm = np.asarray(state['rtm'], dtype=np.float64)
    day = 86400.0
    # temperature tendency (K/s), 3-piece
    t_tendency = np.where(
        zt < 4000.0, -2.51 / day + ((-2.18 + 2.51) / (day * 4000.0)) * zt,
        np.where(zt < 5000.0, -2.18 / day + (2.18 / (day * (5000.0 - 4000.0))) * (zt - 4000.0),
                 0.0))
    state['thlm_forcing'][:] = t_tendency / exner
    # specific-humidity tendency (g/kg/s), 4-piece, then → kg/kg/s
    qtm_forcing = np.where(
        zt < 3000.0, -1.0 / day + ((0.345 + 1.0) / (day * 3000.0)) * zt,
        np.where(zt < 4000.0, 0.345 / day,
                 np.where(zt < 5000.0, 0.345 / day + (-0.345 / (day * (5000.0 - 4000.0))) * (zt - 4000.0),
                          0.0)))
    qtm_forcing = qtm_forcing / 1000.0                       # g_per_kg
    state['rtm_forcing'][:] = (1.0 + rtm) ** 2 * qtm_forcing  # force_spec_hum_to_mixing_ratio


def _gabls2_tndcy(state: dict, time_current: float) -> None:
    """GABLS2 analytic large-scale tendencies (gabls2.F90:gabls2_tndcy). thlm/rtm forcing = 0; subsidence
    is OFF for the first 26 h then `wm_zt = -0.005·min(zt/1000, 1)`; `wm_zm = zt2zm(wm_zt)` with zeroed
    top/surface BCs. (Does not touch ug/vg/um_ref — they keep their sounding/init values, as in Fortran.)"""
    gr = state['gr']
    zt = np.asarray(gr.zt, dtype=np.float64)
    state['thlm_forcing'][:] = 0.0
    state['rtm_forcing'][:] = 0.0
    if time_current > state['time_initial'] + 93600.0:
        wm_zt = np.where(zt <= 1000.0, -0.005 * (zt / 1000.0), -0.005)
    else:
        wm_zt = np.zeros_like(zt)
    state['wm_zt'][:] = wm_zt
    wm_zm = np.array(zt2zm_jax(jnp.asarray(wm_zt), gr), dtype=np.float64)   # np.array → writable copy
    wm_zm[:, 0] = 0.0
    wm_zm[:, -1] = 0.0
    state['wm_zm'][:] = wm_zm


def _gabls2_sfclyr(state: dict, time_current: float, ngrdcol: int,
                   z_bot, p_sfc_arr, ubar, thlm_bot, rtm_bot, exner_bot) -> tuple:
    """GABLS2 surface fluxes (gabls2.F90:gabls2_sfclyr).

    Analytic T_sfc formula (piecewise cosine/linear in local hours starting at 14h).
    Bulk aerodynamic with C_10 scaled to model height, diag_ustar.
    """
    from clubb_jax.src.Benchmark_cases.arm import _diag_ustar
    C_10 = 0.0013
    z0   = 0.03
    z_ref = 10.0
    p0   = 1.0e5
    Rd_over_Cp = 287.04 / 1004.67
    grav = 9.81

    time_in_hours = 14.0 + (time_current - state['time_initial']) / _SEC_PER_HR

    if time_in_hours <= 17.4:
        T_sfc_C = -10.0 - 25.0 * math.cos(time_in_hours * 0.22 + 0.2)
    elif time_in_hours <= 30.0:
        T_sfc_C = -0.54 * time_in_hours + 15.2
    elif time_in_hours <= 41.9:
        T_sfc_C = -7.0 - 25.0 * math.cos(time_in_hours * 0.21 + 1.8)
    elif time_in_hours <= 53.3:
        T_sfc_C = -0.37 * time_in_hours + 18.0
    elif time_in_hours <= 65.6:
        T_sfc_C = -4.0 - 25.0 * math.cos(time_in_hours * 0.22 + 2.5)
    else:
        T_sfc_C = 4.4
    T_sfc_val = T_sfc_C + 273.15

    T_sfc  = np.full(ngrdcol, T_sfc_val)
    sat_formula = state['flags'].saturation_formula
    rsat = np.array([float(sat_mixrat_liq_jax(float(p_sfc_arr[i]), T_sfc_val, sat_formula))
                     for i in range(ngrdcol)])

    log_ref_z0 = math.log(z_ref / z0)
    log_z_z0   = np.log(np.maximum(z_bot, z0 * 1.001) / z0)
    Cz = C_10 * (log_ref_z0 / log_z_z0) ** 2

    wpthlp_sfc = -Cz * ubar * (thlm_bot - T_sfc / exner_bot)
    wprtp_sfc  = -Cz * ubar * (rtm_bot - rsat)
    wprtp_sfc  = wprtp_sfc * 0.025          # gabls2.F90:299: latent heat flux is 2.5% of its potential

    sstheta = T_sfc * ((p0 / p_sfc_arr) ** Rd_over_Cp)
    bflx_arr = wpthlp_sfc * grav / sstheta
    ustar = np.array([_diag_ustar(float(z_bot[i]), float(bflx_arr[i]),
                                   float(ubar[i]), z0)
                      for i in range(ngrdcol)])
    return wpthlp_sfc, wprtp_sfc, ustar, T_sfc


# ── GABLS3-night: Monin-Obukhov landflx ──────────────────────────────────────

def _gm1(x): return (1.0 - 15.0 * x) ** 0.25
def _gh1(x): return math.sqrt(abs(1.0 - 9.0 * x)) / 0.74
def _fm1(x): return (2.0 * math.log((1.0 + x) / 2.0)
                     + math.log((1.0 + x * x) / 2.0)
                     - 2.0 * math.atan(x) + _PI / 2.0)
def _fh1(x): return 2.0 * math.log((1.0 + 0.74 * x) / 2.0)
def _psi_h(x, xlmo): return (-5.0 * x) / xlmo


def _landflx_scalar(th, ts, qh, qs, uh, vh, h, z0):
    """Port of gabls3_night.F90:landflx for a single column.

    Returns (shf, lhf, vel, ustar) — all in natural CLUBB units.
    """
    ep1 = 0.608  # (Rv/Rd - 1) ≈ 0.608
    zody = math.log(h / z0)
    vel  = math.sqrt(max(0.5, uh ** 2 + vh ** 2))
    r    = 9.81 / ts * (th * (1.0 + ep1 * qh) - ts * (1.0 + ep1 * qs)) * h / vel ** 2

    if r < 0.0:
        # Unstable: 3 explicit Businger-Dyer iterations
        xsi = 0.0
        for _ in range(3):
            xm  = _gm1(xsi);  xh = _gh1(xsi)
            fm  = zody - _fm1(xm)
            fh  = 0.74 * (zody - _fh1(xh))
            xsi = r / fh * fm ** 2
            xsi = -abs(xsi)
    else:
        # Stable: quadratic formula
        a = 4.8 ** 2 * r - 6.35
        b = (2.0 * r * 4.8 - 1.0) * zody
        c = r * zody ** 2
        disc = b * b - 4.0 * a * c
        disc = max(0.0, disc)
        xsi1 = (-b + math.sqrt(disc)) / (2.0 * a)
        xsi2 = (-b - math.sqrt(disc)) / (2.0 * a)
        xsi  = max(xsi1, xsi2)
        fm   = zody + 4.8 * xsi
        fh   = zody + 7.8 * xsi   # 1.0 * (...)

    vel   = math.sqrt(uh ** 2 + vh ** 2)
    ustar = 0.4 / fm * vel

    xsi = max(1e-5, xsi) if xsi >= 0.0 else min(-1e-5, xsi)
    xlmo = h / xsi
    denom = math.log(h / 0.25) - _psi_h(h, xlmo) + _psi_h(0.25, xlmo)
    shf = 0.4 * ustar * (ts - th) / denom
    lhf = 0.4 * ustar * (qs - qh) / denom
    return shf, lhf, vel, ustar


def _gabls3_night_sfclyr(state: dict, time_current: float, ngrdcol: int,
                          um_bot, vm_bot, thlm_bot, rtm_bot, z_bot) -> tuple:
    """GABLS3 night surface fluxes (gabls3_night.F90:gabls3_night_sfclyr).

    Reads thlm_sfc, rtm_sfc from gabls3_night_sfc.in (time-dependent).
    Uses landflx (Monin-Obukhov) for heat/moisture fluxes and ustar.
    upwp/vpwp from sfc file if available, else from momentum flux.
    """
    z0 = 0.15
    fd = state['_forcings_data']
    sfc = fd.get('sfc') or {}
    times = sfc.get('time', np.array([0.0, 1e9]))

    ts_arr = sfc.get('thlm_sfc')
    qs_arr = sfc.get('rtm_sfc')
    ts_val = float(np.interp(time_current, times, ts_arr)) if ts_arr is not None else float(thlm_bot[0])
    qs_val = float(np.interp(time_current, times, qs_arr)) if qs_arr is not None else float(rtm_bot[0])

    wpthlp_sfc = np.zeros(ngrdcol)
    wprtp_sfc  = np.zeros(ngrdcol)
    ubar       = np.zeros(ngrdcol)
    ustar      = np.zeros(ngrdcol)

    for i in range(ngrdcol):
        shf, lhf, vel, us = _landflx_scalar(
            float(thlm_bot[i]), ts_val, float(rtm_bot[i]), qs_val,
            float(um_bot[i]), float(vm_bot[i]), float(z_bot[i]), z0)
        wpthlp_sfc[i] = shf
        wprtp_sfc[i]  = lhf
        ubar[i]       = vel
        ustar[i]      = us

    # Momentum flux
    upwp_arr = sfc.get('upwp_sfc')
    vpwp_arr = sfc.get('vpwp_sfc')
    l_input_xpwp = upwp_arr is not None

    if l_input_xpwp:
        upwp_sfc = np.full(ngrdcol, float(np.interp(time_current, times, upwp_arr)))
        vpwp_sfc = np.full(ngrdcol, float(np.interp(time_current, times, vpwp_arr)))
    else:
        upwp_sfc, vpwp_sfc = _compute_momentum_flux(um_bot, vm_bot, ubar, ustar)

    return wpthlp_sfc, wprtp_sfc, ustar, upwp_sfc, vpwp_sfc


def _gabls3_sfclyr(state: dict, ngrdcol: int, ubar, thlm_bot, rtm_bot, z_bot, exner_bot) -> tuple:
    """GABLS3 (daytime) surface fluxes (gabls3.F90:gabls3_sfclyr). Bulk-aerodynamic heat/moisture fluxes
    using the interactive vegetation temperature `veg_T_in_K` (from soil_vegetation) as the surface temp:
    `wpthlp_sfc=-C_10·ubar·(thlm-veg_T/exner)`, `wprtp_sfc=-C_10·ubar·(rtm-offset)` then ×10, ustar via
    diag_ustar with the veg-based buoyancy flux. C_10=0.00195, offset=9.9e-3, z0=0.15."""
    from clubb_jax.src.Benchmark_cases.arm import _diag_ustar
    from clubb_jax.src.CLUBB_core.constants_clubb import grav
    C_10, offset, z0 = 0.00195, 9.9e-3, 0.15
    veg_T = np.asarray(state.get('veg_T_in_K', np.full(ngrdcol, 300.0)), dtype=np.float64)

    wpthlp_sfc = -C_10 * ubar * (thlm_bot - veg_T / exner_bot)       # compute_wpthlp_sfc, T_sfc=veg_T
    wprtp_sfc = -C_10 * ubar * (rtm_bot - offset)                    # compute_wprtp_sfc, adjustment=offset
    wprtp_sfc = wprtp_sfc * 10.0                                     # gabls3.F90:116

    ustar = np.zeros(ngrdcol)
    for i in range(ngrdcol):
        veg_theta = veg_T[i] / exner_bot[i]
        bflx = float(wpthlp_sfc[i]) * grav / veg_theta
        ustar[i] = _diag_ustar(float(z_bot[i]), bflx, float(ubar[i]), z0)
    return wpthlp_sfc, wprtp_sfc, ustar


# ── ATEX ──────────────────────────────────────────────────────────────────────

def _atex_sfclyr(state: dict, time_current: float, ngrdcol: int,
                 ubar, thlm_bot, rtm_bot, exner_bot) -> tuple:
    """ATEX surface fluxes (atex.F90:atex_sfclyr, sfctype=1).

    C_10=0.0013, adjustment=0.0198293 (fixed), ustar=0.3.
    T_sfc time-interpolated from atex_sfc.in.
    """
    C_10       = 0.0013
    adjustment = 0.0198293

    fd = state['_forcings_data']
    sfc = fd.get('sfc') or {}
    times = sfc.get('time', np.array([0.0, 1e9]))
    T_sfc_arr = sfc.get('t_sfc')
    T_sfc_val = float(np.interp(time_current, times, T_sfc_arr)) if T_sfc_arr is not None else 298.0

    T_sfc   = np.full(ngrdcol, T_sfc_val)
    ustar   = np.full(ngrdcol, 0.3)
    Cd      = np.full(ngrdcol, C_10)
    adj     = np.full(ngrdcol, adjustment)

    # compute_wpthlp_sfc: -Cd * ubar * (thlm - T_sfc/exner)
    wpthlp_sfc = -Cd * ubar * (thlm_bot - T_sfc / exner_bot)
    # compute_wprtp_sfc: -Cd * ubar * (rtm - adjustment)
    wprtp_sfc  = -Cd * ubar * (rtm_bot - adj)
    return wpthlp_sfc, wprtp_sfc, ustar, T_sfc


# ── ATEX-Long ────────────────────────────────────────────────────────────────

def _atex_long_sfclyr(state: dict, time_current: float, ngrdcol: int,
                      ubar, thlm_bot, rtm_bot, exner_bot) -> tuple:
    """ATEX-Long surface fluxes (atex_long.F90:atex_long_sfclyr, l_compute_flux=True).

    C_10=0.0013, adjustment=0.0194664, ustar=0.3. T_sfc from sfc file.
    """
    C_10       = 0.0013
    adjustment = 0.0194664

    fd = state['_forcings_data']
    sfc = fd.get('sfc') or {}
    times = sfc.get('time', np.array([0.0, 1e9]))
    T_sfc_arr = sfc.get('t_sfc')
    T_sfc_val = float(np.interp(time_current, times, T_sfc_arr)) if T_sfc_arr is not None else 298.0

    T_sfc   = np.full(ngrdcol, T_sfc_val)
    ustar   = np.full(ngrdcol, 0.3)
    Cd      = np.full(ngrdcol, C_10)
    adj     = np.full(ngrdcol, adjustment)

    wpthlp_sfc = -Cd * ubar * (thlm_bot - T_sfc / exner_bot)
    wprtp_sfc  = -Cd * ubar * (rtm_bot - adj)
    return wpthlp_sfc, wprtp_sfc, ustar, T_sfc


# ── ARM Variants (arm_0003, arm_97/mc3e) ─────────────────────────────────────

def _arm_variant_sfclyr(state: dict, time_current: float, ngrdcol: int,
                        z_bot, rho_bot, thlm_bot, ubar, z0: float = 0.035) -> tuple:
    """ARM-variant surface fluxes (arm_0003.F90, arm_97.F90:sfclyr).

    Reads sens_ht/latent_ht from sfc file (W/m^2), converts using rho_sfc,
    then computes ustar via diag_ustar (same as ARM).
    """
    from clubb_jax.src.Benchmark_cases.arm import _diag_ustar
    grav = 9.81

    fd = state['_forcings_data']
    sfc = fd.get('sfc') or {}
    times = sfc.get('time', np.array([0.0, 1e9]))

    def _interp_sfc(key, default):
        arr = sfc.get(key)
        if arr is not None and len(arr) > 0:
            return float(np.interp(time_current, times, arr))
        return default

    sens_ht   = _interp_sfc('sens_ht',   0.0)
    latent_ht = _interp_sfc('latent_ht', 0.0)

    wpthlp_sfc = sens_ht   / (rho_bot * Cp)
    wprtp_sfc  = latent_ht / (rho_bot * Lv)

    bflx = grav / thlm_bot * wpthlp_sfc
    ustar = np.array([_diag_ustar(float(z_bot[i]), float(bflx[i]),
                                   float(ubar[i]), z0)
                      for i in range(ngrdcol)])
    return wpthlp_sfc, wprtp_sfc, ustar


# ── Cloud-Feedback / CGILS / ASTEX-A209 ─────────────────────────────────────

def _bulk_aero_sfclyr(state: dict, time_current: float, ngrdcol: int,
                      ubar, thlm_bot, rtm_bot, exner_bot, z_bot,
                      p_sfc_arr, ustar_val: float,
                      C_h_20: float, C_q_20: float,
                      z0: float, z_ref: float) -> tuple:
    """Generic bulk aerodynamic sfclyr used by cloud_feedback, astex_a209.

    Port of cloud_feedback.F90:cloud_feedback_sfclyr (sfctype=1) and
    astex_a209.F90:astex_a209_sfclyr.

    T_sfc interpolated from {case}_sfc.in. Coefficients Ch/Cq scaled to z_bot.
    """
    fd = state['_forcings_data']
    sfc = fd.get('sfc') or {}
    times = sfc.get('time', np.array([0.0, 1e9]))
    T_sfc_arr = sfc.get('t_sfc')
    T_sfc_val = float(np.interp(time_current, times, T_sfc_arr)) if T_sfc_arr is not None else 298.0

    T_sfc = np.full(ngrdcol, T_sfc_val)
    ustar = np.full(ngrdcol, ustar_val)

    log_ref_z0 = math.log(z_ref / z0)
    log_z_z0   = np.log(np.maximum(z_bot, z0 * 1.001) / z0)
    scale      = (log_ref_z0 / log_z_z0) ** 2
    Ch = C_h_20 * scale
    Cq = C_q_20 * scale

    p0    = 1.0e5
    kappa = 287.04 / 1004.67
    exner_sfc = (p_sfc_arr / p0) ** kappa

    sat_formula = state['flags'].saturation_formula
    rsat = np.array([float(sat_mixrat_liq_jax(float(p_sfc_arr[i]), T_sfc_val, sat_formula))
                     for i in range(ngrdcol)])

    wpthlp_sfc = -Ch * ubar * (thlm_bot - T_sfc / exner_sfc)
    wprtp_sfc  = -Cq * ubar * (rtm_bot - rsat)
    return wpthlp_sfc, wprtp_sfc, ustar, T_sfc


# ── Zero-flux altocu cases ────────────────────────────────────────────────────

def _zero_flux_sfclyr(ngrdcol: int) -> tuple:
    """No surface momentum or heat fluxes. ustar = 0.

    Port for nov11_altocu, jun25_altocu, clex9_nov02, clex9_oct14.
    """
    z = np.zeros(ngrdcol)
    return z.copy(), z.copy(), z.copy()  # wpthlp_sfc, wprtp_sfc, ustar


# ── Data loading ────────────────────────────────────────────────────────────

def _linear_fill_blanks_1d(grid: np.ndarray, values: np.ndarray,
                           blank: float = _BLANK) -> np.ndarray:
    """Fill blank sentinels in values by linear interpolation on grid.

    Port of input_reader.F90:linear_fill_blanks.
    Valid = values > blank. If none valid: return blank. Extrapolates at edges.
    """
    valid = values > blank
    if not np.any(valid):
        return np.full_like(values, blank)
    if np.all(valid):
        return values.copy()
    vg = grid[valid]
    vv = values[valid]
    return np.interp(grid, vg, vv, left=float(vv[0]), right=float(vv[-1]))


def _fill_blanks_2d(z_grid: np.ndarray, time_grid: np.ndarray,
                    arr: np.ndarray, blank: float = _BLANK) -> np.ndarray:
    """Fill blanks first along z then along time.

    Port of input_reader.F90:fill_blanks_two_dim_vars.
    arr shape: (nz, ntimes). Modifies in-place and returns result.
    """
    out = arr.copy()
    nz, nt = out.shape
    for it in range(nt):
        out[:, it] = _linear_fill_blanks_1d(z_grid, out[:, it], blank)
    for iz in range(nz):
        out[iz, :] = _linear_fill_blanks_1d(time_grid, out[iz, :], blank)
    return out


def _parse_forcings_file(path: str, zt: np.ndarray) -> dict:
    """Parse a {case}_forcings.in file. Same format as arm_forcings.in.

    Applies fill_blanks_two_dim_vars (port of Fortran) before interpolating
    to model grid, so -999.9 sentinel values are filled by interpolation.
    """
    nzt = zt.shape[0]
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.strip().startswith('!')]
    if not lines:
        return {}

    col_names = [c.strip("'\"") for c in lines[0].split()][1:]  # skip z[m]

    raw_times, raw_z, raw_data = [], [], {n: [] for n in col_names}
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        parts = line.split()
        if len(parts) == 2:
            raw_times.append(float(parts[0]))
            n_lev = int(parts[1])
            z_block = []
            data_block = {n: [] for n in col_names}
            i += 1
            for _ in range(n_lev):
                row = lines[i].split(); i += 1
                z_block.append(float(row[0]))
                for j, n in enumerate(col_names):
                    data_block[n].append(float(row[j + 1]))
            raw_z.append(np.array(z_block))
            for n in col_names:
                raw_data[n].append(np.array(data_block[n]))
        else:
            i += 1

    times = np.array(raw_times)
    ntimes = len(times)

    # Build a common z grid (union of all raw z levels, sorted)
    all_z = np.unique(np.concatenate(raw_z))

    interp = {}
    for name in col_names:
        # Step 1: project each time block onto the common z grid with blanks
        raw_on_common = np.full((len(all_z), ntimes), _BLANK)
        for it in range(ntimes):
            z_t = raw_z[it]
            v_t = raw_data[name][it]
            for iz, z_val in enumerate(z_t):
                # Find matching index in common grid
                idx = np.searchsorted(all_z, z_val)
                if idx < len(all_z) and abs(all_z[idx] - z_val) < 1.0:
                    raw_on_common[idx, it] = v_t[iz]

        # Step 2: fill_blanks_two_dim_vars (z first, then time)
        filled = _fill_blanks_2d(all_z, times, raw_on_common)

        # Step 3: interpolate filled data to model zt grid
        # If still all blank after fill (shouldn't happen), leave as zero
        arr = np.zeros((nzt, ntimes))
        for it in range(ntimes):
            col = filled[:, it]
            if np.all(col <= _BLANK):
                arr[:, it] = 0.0
            else:
                valid = col > _BLANK
                arr[:, it] = np.interp(zt, all_z[valid], col[valid],
                                       left=float(col[valid][0]),
                                       right=float(col[valid][-1]))
        interp[name] = arr

    def _col(prefix):
        # Match a column by name prefix (ignore the [units] suffix), e.g. 'um_f', 'ug'.
        for k in interp:
            if k.split('[')[0].strip() == prefix:
                # A column that is entirely blank (-999.9) in every time block is "not
                # provided" — the fill turns it into 0, but Fortran leaves it unset
                # (keeps the sounding ug/vg). Return None so it is NOT applied.
                if all(np.all(np.abs(np.asarray(b) + 999.9) < 1e-3) for b in raw_data[k]):
                    return None
                return interp[k]
        return None

    # Subsidence may be given as omega (pressure velocity) instead of w (time_dependent_input.F90:798-835).
    # Detect the omega column + its units; the omega→w conversion (needs rho) is deferred to forcing-apply.
    omega_arr, omega_mb_hr = None, False
    for k in interp:
        if k.split('[')[0].strip() == 'omega':
            if not all(np.all(np.abs(np.asarray(b) + 999.9) < 1e-3) for b in raw_data[k]):
                omega_arr, omega_mb_hr = interp[k], ('mb/hr' in k)

    return {
        'times':  times,
        'thlm_f': interp.get('thlm_f[K/s]'),
        # moisture forcing: 'rtm_f' (mixing-ratio) or 'sp_hmdty_f' (specific humidity) — converted at apply
        'rtm_f':  interp.get('rtm_f[kg/kg/s]'),
        'sp_hmdty_f': _col('sp_hmdty_f'),
        # 'w' (subsidence) is STATE like ug/vg — an all-blank column must keep the
        # sounding wm_zt, not be overwritten with 0 (use _col's all-blank guard).
        'w':      _col('w'),
        'omega':  omega_arr,
        'omega_mb_hr': omega_mb_hr,
        'um_f':   _col('um_f'),
        'vm_f':   _col('vm_f'),
        'ug':     _col('ug'),
        'vg':     _col('vg'),
        'um_ref': _col('um_ref'),
        'vm_ref': _col('vm_ref'),
    }


def _parse_sfc_file(path: str) -> dict:
    """Parse a {case}_sfc.in surface flux file."""
    lines = [ln for ln in Path(path).read_text().splitlines()
             if ln.strip() and not ln.strip().startswith('!')]
    if len(lines) < 2:
        return {'time': np.array([0.0]), 'wpthlp_sfc': np.zeros(1),
                'wpqtp_sfc': np.zeros(1), 't_sfc': np.full(1, 300.0)}

    header = lines[0].split()
    # Identify column positions by name
    cols = [c.strip("'\"").lower() for c in header]

    rows = [[float(v) for v in ln.split()] for ln in lines[1:] if ln.split()]
    data = np.array(rows)

    def _get_col(kw):
        for idx, c in enumerate(cols):
            if kw in c:
                return data[:, idx]
        return None

    return {
        'time':       data[:, 0],
        'wpqtp_sfc':  _get_col('wpqtp'),
        'wpthlp_sfc': _get_col('wpthlp'),
        't_sfc':      _get_col('t_sfc'),
        'sens_ht':    _get_col('sens_ht'),
        'latent_ht':  _get_col('latent_ht'),
        # gabls3_night columns
        'thlm_sfc':   _get_col('thlm'),
        'rtm_sfc':    _get_col('rt['),  # matches 'rt[kg/kg]'
        'upwp_sfc':   _get_col('upwp'),
        'vpwp_sfc':   _get_col('vpwp'),
    }


def load_generic_forcings_data(runtype: str, case_dir: str, zt: np.ndarray) -> dict:
    """Load forcing and surface files for a non-ARM benchmark case.

    Looks for {case_dir}/{runtype}_forcings.in and {case_dir}/{runtype}_sfc.in.
    Returns a dict with 'times', 'thlm_f', 'rtm_f', 'w', 'sfc'.
    """
    forcings_path = os.path.join(case_dir, f'{runtype}_forcings.in')
    sfc_path = os.path.join(case_dir, f'{runtype}_sfc.in')

    fd = {'times': None, 'thlm_f': None, 'rtm_f': None, 'w': None, 'sfc': None}

    if runtype == 'mpace_a':
        fd.update(load_mpace_a_forcings(case_dir))       # custom mpace_a_forcings/*.dat (mpace_a.F90)
        return fd

    if os.path.isfile(forcings_path):
        parsed = _parse_forcings_file(forcings_path, zt)
        fd.update(parsed)

    if os.path.isfile(sfc_path):
        fd['sfc'] = _parse_sfc_file(sfc_path)

    return fd


def _fire_sfclyr(state: dict, time_current: float, ngrdcol: int,
                 ubar, thlm_bot, rtm_bot, exner_bot) -> tuple:
    """FIRE surface fluxes — bulk formula, sfctype=1 (fire.F90:fire_sfclyr).

    Cz=0.0013, ustar=0.3, T_sfc from fire_sfc.in; rsat = sat_mixrat_liq(p_sfc, T_sfc).
    wpthlp_sfc = -Cz*ubar*(thlm - T_sfc/exner);  wprtp_sfc = -Cz*ubar*(rtm - rsat).
    """
    sfc = (state.get('_forcings_data') or {}).get('sfc') or {}
    times = sfc.get('time', np.array([0.0]))
    t_sfc_arr = sfc.get('t_sfc')
    T_sfc_val = (float(np.interp(time_current, times, t_sfc_arr))
                 if t_sfc_arr is not None else 288.0)
    Cz = 0.0013
    ustar = np.full(ngrdcol, 0.3)
    T_sfc = np.full(ngrdcol, T_sfc_val)
    sat_formula = state['flags'].saturation_formula
    p_sfc = state['p_sfc']
    rsat = np.array([float(sat_mixrat_liq_jax(float(p_sfc[i]), T_sfc_val, sat_formula))
                     for i in range(ngrdcol)])
    wpthlp_sfc = -Cz * ubar * (thlm_bot - T_sfc / exner_bot)
    wprtp_sfc  = -Cz * ubar * (rtm_bot - rsat)
    return wpthlp_sfc, wprtp_sfc, ustar, T_sfc


# ── MPACE-A (custom Arctic-stratus forcing, mpace_a.F90) ────────────────────

_MPACE_A_NTIMES = 139
_MPACE_A_NLEVELS = 38


def _read_mpace_dat(path: str) -> np.ndarray:
    """Read a mpace_a *.dat file as a flat array of floats (whitespace-delimited, the per_line=5
    formatting is irrelevant since values are read sequentially; file_functions.F90:file_read_1d/2d)."""
    with open(path) as fh:
        return np.array([float(tok) for tok in fh.read().split()], dtype=np.float64)


def load_mpace_a_forcings(case_dir: str) -> dict:
    """Load the 11 mpace_a_forcings/*.dat files (mpace_a.F90:mpace_a_init). 2-D fields are level-major
    (file_read_2d: outer loop over levels) → reshape (nlevels, ntimes)."""
    d = os.path.join(case_dir, 'mpace_a_forcings')
    nt, nl = _MPACE_A_NTIMES, _MPACE_A_NLEVELS
    g1 = lambda name: _read_mpace_dat(os.path.join(d, name))[:nl if name in ('mpace_a_heights.dat',
                                     'mpace_a_press.dat') else nt]
    g2 = lambda name: _read_mpace_dat(os.path.join(d, name)).reshape(nl, nt)
    return dict(file_times=_read_mpace_dat(os.path.join(d, 'mpace_a_times.dat'))[:nt],
                file_heights=_read_mpace_dat(os.path.join(d, 'mpace_a_heights.dat'))[:nl],
                dTdt=g2('mpace_a_dTdt.dat'), dqdt=g2('mpace_a_dqdt_horiz.dat'),
                vertT=g2('mpace_a_verts.dat'), vertq=g2('mpace_a_vertq.dat'),
                um_obs=g2('mpace_a_um_obs.dat'), vm_obs=g2('mpace_a_vm_obs.dat'),
                file_lh=_read_mpace_dat(os.path.join(d, 'mpace_a_lh.dat'))[:nt],
                file_sh=_read_mpace_dat(os.path.join(d, 'mpace_a_sh.dat'))[:nt])


def _mpace_time_select(file_times: np.ndarray, time: float):
    """time_dependent_input.F90:time_select — find before/after bracket + frac. Returns (b, a, ratio),
    0-based. ratio = (time − t[b]) / (t[a] − t[b]). (mpace_a's data covers the full run, so no clamp.)"""
    b = int(np.searchsorted(file_times, time, side='right') - 1)
    b = min(max(b, 0), len(file_times) - 2)
    a = b + 1
    ratio = (time - file_times[b]) / (file_times[a] - file_times[b])
    return b, a, ratio


def _zlinterp(z_out: np.ndarray, z_src: np.ndarray, v_src: np.ndarray) -> np.ndarray:
    """interpolation.F90:zlinterp_fnc — linear interp in height with ZERO extrapolation (below the
    source bottom or above the source top → 0), NOT np.interp's endpoint clamping."""
    out = np.interp(z_out, z_src, v_src)
    return np.where((z_out < z_src[0]) | (z_out > z_src[-1]), 0.0, out)


def _mpace_a_tndcy(state: dict, time_current: float) -> None:
    """mpace_a.F90:mpace_a_tndcy — large-scale tendencies from the custom forcing files. Subsidence was
    deliberately removed (wm=0, Michael Falk 2007); thlm/rtm forcing = (horiz+vert advection) converted;
    um_obs/vm_obs become um_ref/vm_ref for nudging (l_uv_nudge)."""
    from clubb_jax.src.CLUBB_core.constants_clubb import Rd, Cp, grav  # noqa: F401
    sec_per_hr, g_per_kg = 3600.0, 1000.0
    fd = state['_forcings_data']
    gr = state['gr']; ngrdcol = state['ngrdcol']
    b, a, r = _mpace_time_select(fd['file_times'], time_current)
    zt = np.asarray(gr.zt); zh = fd['file_heights']
    p_in_Pa = np.asarray(state['p_in_Pa'], dtype=np.float64)
    p_sfc = 101000.0                                   # HARDCODED in mpace_a.F90:140 (NOT p_sfc_nl=101500)

    def col(field):                                    # time-interp then height-interp to zt, per column
        c = (1.0 - r) * fd[field][:, b] + r * fd[field][:, a]
        return np.stack([_zlinterp(zt[i], zh, c) for i in range(ngrdcol)], axis=0)

    dTdt, vertT = col('dTdt'), col('vertT')
    dqdt, vertq = col('dqdt'), col('vertq')
    um_g, vm_g = col('um_obs'), col('vm_obs')
    exner_fac = (p_sfc / p_in_Pa) ** (Rd / Cp)
    state['thlm_forcing'][:] = (dTdt + vertT) * exner_fac / sec_per_hr
    state['rtm_forcing'][:] = (dqdt + vertq) / g_per_kg / sec_per_hr
    state['wm_zt'][:] = 0.0
    state['wm_zm'][:] = 0.0
    if 'um_ref' in state:
        state['um_ref'][:] = um_g
    if 'vm_ref' in state:
        state['vm_ref'][:] = vm_g


def _mpace_a_sfclyr(state: dict, time_current: float, ngrdcol: int, rho_sfc) -> tuple:
    """mpace_a.F90:mpace_a_sfclyr — surface fluxes from the prescribed sensible/latent heat (time-interp);
    wpthlp_sfc = SH/(ρ·Cp), wprtp_sfc = LH/(ρ·Lv), ustar = 0.25 (fixed)."""
    from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Lv
    fd = state['_forcings_data']
    b, a, r = _mpace_time_select(fd['file_times'], time_current)
    lh = (1.0 - r) * fd['file_lh'][b] + r * fd['file_lh'][a]
    sh = (1.0 - r) * fd['file_sh'][b] + r * fd['file_sh'][a]
    rho_sfc = np.asarray(rho_sfc, dtype=np.float64)
    wpthlp_sfc = np.full(ngrdcol, sh) / (rho_sfc * Cp)
    wprtp_sfc = np.full(ngrdcol, lh) / (rho_sfc * Lv)
    ustar = np.full(ngrdcol, 0.25)
    return wpthlp_sfc, wprtp_sfc, ustar


# ── Main dispatcher ─────────────────────────────────────────────────────────

def prescribe_forcings_generic(state: dict, time_current: float,
                               l_sample: bool = False) -> None:
    """prescribe_forcings for non-ARM cases. Port of prescribe_forcings.F90.

    Supported cases: bomex, fire, generic, neutral, coriolis_test, ekman.
    Cases with l_t_dependent=True and a *_forcings.in file use the generic
    time-dependent framework. Other cases raise NotImplementedError.
    """
    runtype = state['runtype']
    ngrdcol = state['ngrdcol']
    l_t_dependent = state.get('l_t_dependent', False)
    l_ignore_forcings = state.get('l_ignore_forcings', False)

    # ── 1. Zero all forcing arrays ──────────────────────────────────────────
    state['thlm_forcing'][:] = 0.0
    state['rtm_forcing'][:] = 0.0
    state['wprtp_forcing'][:] = 0.0
    state['wpthlp_forcing'][:] = 0.0
    state['rtp2_forcing'][:] = 0.0
    state['thlp2_forcing'][:] = 0.0
    state['rtpthlp_forcing'][:] = 0.0

    # ── 2. Large-scale tendencies ───────────────────────────────────────────
    if runtype == 'mpace_a':
        _mpace_a_tndcy(state, time_current)              # fully custom forcing files (mpace_a.F90)
    elif runtype == 'gabls2':
        _gabls2_tndcy(state, time_current)               # analytic subsidence (off first 26 h)
    elif runtype == 'wangara':
        _zero_forcings(state)                            # wangara.F90: no LS forcing, no subsidence
        state['wm_zt'][:] = 0.0
        state['wm_zm'][:] = 0.0
    elif runtype == 'atex':
        _atex_tndcy(state, time_current)                 # subsidence gated off the first 90 min
    elif runtype == 'atex_long':
        _atex_long_tndcy(state, time_current)            # fixed 3-piece subsidence + thlm/rtm forcing + spinup
    elif l_t_dependent and not l_ignore_forcings:
        _apply_time_dependent_forcings(state, time_current)
    else:
        if runtype == 'bomex':
            _bomex_tndcy(state)
        elif runtype == 'rico':
            _rico_tndcy(state)   # analytic thlm/qtm→rtm forcing; wm is init-set (untouched)
        elif runtype in ('fire', 'generic', 'neutral', 'coriolis_test', 'ekman', 'dycoms2_rf01'):
            # dycoms2_rf01: tndcy zeros thlm/rtm only; wm (subsidence) is set at init and left
            # unchanged (dycoms2_rf01.F90:dycoms2_rf01_tndcy) — so _zero_forcings (no wm touch).
            # (dycoms2_rf01_fixed_sst stays on the fallback — its fixed-SST sfclyr has a latent bug.)
            _zero_forcings(state)
        elif runtype == 'dycoms2_rf02':       # all rf02 variants (nd/so/do/ds) share this runtype
            # dycoms2_rf02.F90:dycoms2_rf02_tndcy — zeros thlm/rtm forcing + zeros wm ONLY at the top
            # level (the rest of wm is init-set, like rf01).
            _zero_forcings(state)
            state['wm_zt'][:, -1] = 0.0
            state['wm_zm'][:, -1] = 0.0
        else:
            raise NotImplementedError(
                f"prescribe_forcings_generic: analytic tndcy not yet ported for "
                f"runtype={runtype!r}. Add to Benchmark_cases/generic_forcings.py."
            )

    # ── 3. Bottom-level state for surface BCs ──────────────────────────────
    bc = _read_surface_var_for_bc(state)
    um_bot    = bc['um_bot']
    vm_bot    = bc['vm_bot']
    rtm_bot   = bc['rtm_bot']
    thlm_bot  = bc['thlm_bot']
    rho_bot   = bc['rho_bot']
    exner_bot = bc['exner_bot']
    z_bot     = bc['z_bot']
    ubar = _compute_ubar(um_bot, vm_bot)

    # ── 4. Case-specific surface fluxes ────────────────────────────────────
    upwp_sfc  = state['upwp_sfc'].copy()
    vpwp_sfc  = state['vpwp_sfc'].copy()
    wpthlp_sfc = state['wpthlp_sfc'].copy()
    wprtp_sfc  = state['wprtp_sfc'].copy()
    T_sfc = state.get('T_sfc', np.zeros(ngrdcol)).copy()
    l_compute_momentum_flux = False
    l_set_sclr = False

    if runtype == 'bomex':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _bomex_sfclyr(state, time_current, ngrdcol, rtm_bot)

    elif runtype == 'rico':
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, upwp_sfc, vpwp_sfc, T_sfc = \
            _rico_sfclyr(state, time_current, ngrdcol,
                         um_bot, vm_bot, thlm_bot, rtm_bot, rho_bot, exner_bot, z_bot, ubar)
        state['upwp_sfc'][:] = upwp_sfc
        state['vpwp_sfc'][:] = vpwp_sfc
        # rico_sfclyr computes momentum flux directly (not via ustar^2/ubar)

    elif runtype == 'dycoms2_rf01':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, T_sfc = \
            _dycoms2_rf01_sfclyr(state, time_current, ngrdcol, rho_bot,
                                 ubar, thlm_bot, rtm_bot, exner_bot)

    elif runtype in ('dycoms2_rf02', 'dycoms2_rf02_do', 'dycoms2_rf02_ds',
                     'dycoms2_rf02_so', 'dycoms2_rf02_nd', 'dycoms2_rf02_morr'):
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _dycoms2_rf02_sfclyr(state, time_current, ngrdcol)

    elif runtype == 'wangara':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _wangara_sfclyr(time_current, ngrdcol)

    elif runtype == 'gabls2':
        l_compute_momentum_flux = False  # gabls2 uses diag_ustar internally
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, T_sfc = _gabls2_sfclyr(
            state, time_current, ngrdcol,
            z_bot, state['p_sfc'], ubar, thlm_bot, rtm_bot, exner_bot)
        upwp_sfc, vpwp_sfc = _compute_momentum_flux(um_bot, vm_bot, ubar, ustar)
        state['upwp_sfc'][:] = upwp_sfc
        state['vpwp_sfc'][:] = vpwp_sfc

    elif runtype == 'gabls3_night':
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, upwp_sfc, vpwp_sfc = _gabls3_night_sfclyr(
            state, time_current, ngrdcol, um_bot, vm_bot, thlm_bot, rtm_bot, z_bot)
        state['upwp_sfc'][:] = upwp_sfc
        state['vpwp_sfc'][:] = vpwp_sfc

    elif runtype == 'gabls3':
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _gabls3_sfclyr(
            state, ngrdcol, ubar, thlm_bot, rtm_bot, z_bot, exner_bot)
        upwp_sfc, vpwp_sfc = _compute_momentum_flux(um_bot, vm_bot, ubar, ustar)
        state['upwp_sfc'][:] = upwp_sfc
        state['vpwp_sfc'][:] = vpwp_sfc

    elif runtype == 'mpace_a':
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _mpace_a_sfclyr(
            state, time_current, ngrdcol, np.asarray(state['rho_zm'])[:, 0])
        upwp_sfc, vpwp_sfc = _compute_momentum_flux(um_bot, vm_bot, ubar, ustar)
        state['upwp_sfc'][:] = upwp_sfc
        state['vpwp_sfc'][:] = vpwp_sfc

    elif runtype == 'atex':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, T_sfc = _atex_sfclyr(
            state, time_current, ngrdcol, ubar, thlm_bot, rtm_bot, exner_bot)

    elif runtype == 'atex_long':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, T_sfc = _atex_long_sfclyr(
            state, time_current, ngrdcol, ubar, thlm_bot, rtm_bot, exner_bot)

    elif runtype in ('arm_0003',):
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _arm_variant_sfclyr(
            state, time_current, ngrdcol, z_bot, rho_bot, thlm_bot, ubar, z0=0.035)

    elif runtype in ('arm_97', 'mc3e', 'arm_3year'):
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _arm_variant_sfclyr(
            state, time_current, ngrdcol, z_bot, rho_bot, thlm_bot, ubar, z0=0.035)

    elif runtype in ('cloud_feedback_s6', 'cloud_feedback_s6_p2k',
                     'cloud_feedback_s11', 'cloud_feedback_s11_p2k',
                     'cloud_feedback_s12', 'cloud_feedback_s12_p2k',
                     'cgils_s6', 'cgils_s6_p2k', 'cgils_s11',
                     'cgils_s11_p2k', 'cgils_s12', 'cgils_s12_p2k'):
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, T_sfc = _bulk_aero_sfclyr(
            state, time_current, ngrdcol,
            ubar, thlm_bot, rtm_bot, exner_bot, z_bot, state['p_sfc'],
            ustar_val=0.3, C_h_20=0.001094, C_q_20=0.001133,
            z0=0.00015, z_ref=20.0)

    elif runtype == 'astex_a209':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, T_sfc = _bulk_aero_sfclyr(
            state, time_current, ngrdcol,
            ubar, thlm_bot, rtm_bot, exner_bot, z_bot, state['p_sfc'],
            ustar_val=0.155, C_h_20=0.001094, C_q_20=0.001133,
            z0=0.00015, z_ref=20.0)

    elif runtype in ('nov11_altocu', 'jun25_altocu',
                     'clex9_nov02', 'clex9_oct14'):
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _zero_flux_sfclyr(ngrdcol)
        upwp_sfc = np.zeros(ngrdcol)
        vpwp_sfc = np.zeros(ngrdcol)
        state['upwp_sfc'][:] = upwp_sfc
        state['vpwp_sfc'][:] = vpwp_sfc

    elif runtype == 'cobra':
        l_compute_momentum_flux = True
        l_set_sclr = True
        # cobra uses sens_ht/latent_ht + T_sfc from sfc file; same as arm variant
        # CO2 scalar flux not yet ported — set to zero for now.
        # cobra.F90: momentum roughness height z0 = 1.75 m (not the ARM 0.035 m).
        wpthlp_sfc, wprtp_sfc, ustar = _arm_variant_sfclyr(
            state, time_current, ngrdcol, z_bot, rho_bot, thlm_bot, ubar, z0=1.75)
        # T_sfc from sfc file
        sfc = state.get('_forcings_data', {}).get('sfc') or {}
        times_sfc = sfc.get('time', np.array([0.0]))
        T_sfc_arr = sfc.get('t_sfc')
        if T_sfc_arr is not None:
            T_sfc = np.full(ngrdcol, float(np.interp(time_current, times_sfc, T_sfc_arr)))

    elif runtype == 'lba':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _lba_sfclyr(state, time_current, ngrdcol,
                                                     z_bot, rho_bot, thlm_bot, ubar)

    elif runtype == 'fire':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, T_sfc = _fire_sfclyr(
            state, time_current, ngrdcol, ubar, thlm_bot, rtm_bot, exner_bot)

    elif runtype == 'generic':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc = np.full(ngrdcol, float(state.get('sens_ht', 0.0)))
        wprtp_sfc  = np.full(ngrdcol, float(state.get('latent_ht', 0.0)))
        ustar      = np.full(ngrdcol, 0.3)

    elif runtype == 'neutral':
        # neutral_case.F90:neutral_case_sfclyr — ustar=0.5, momentum flux via
        # compute_momentum_flux, wpthlp_sfc=0.05 until t=80880 s then 0, wprtp_sfc=0.
        l_set_sclr = True
        ustar      = np.full(ngrdcol, 0.5)
        wpthlp_sfc = np.full(ngrdcol, 0.0 if time_current > 80880.0 else 0.05)
        wprtp_sfc  = np.zeros(ngrdcol)
        upwp_sfc, vpwp_sfc = _compute_momentum_flux(um_bot, vm_bot, ubar, ustar)
        state['upwp_sfc'][:] = upwp_sfc
        state['vpwp_sfc'][:] = vpwp_sfc

    elif runtype == 'ekman':
        # ekman.F90:ekman_sfclyr — ustar=0.3, zero heat/moisture flux, momentum
        # flux via compute_momentum_flux. (Requires sponge-layer damping to be
        # wired in to be bit-faithful — see sponge_layer_damping.py.)
        l_set_sclr = True
        ustar      = np.full(ngrdcol, 0.3)
        wpthlp_sfc = np.zeros(ngrdcol)
        wprtp_sfc  = np.zeros(ngrdcol)
        upwp_sfc, vpwp_sfc = _compute_momentum_flux(um_bot, vm_bot, ubar, ustar)
        state['upwp_sfc'][:] = upwp_sfc
        state['vpwp_sfc'][:] = vpwp_sfc

    elif runtype == 'coriolis_test':
        # coriolis_test.F90 sfclyr unverified; leave the zero-flux stub.
        l_set_sclr = True
        wpthlp_sfc = np.zeros(ngrdcol)
        wprtp_sfc  = np.zeros(ngrdcol)
        ustar      = np.zeros(ngrdcol)
        state['upwp_sfc'][:] = np.zeros(ngrdcol)
        state['vpwp_sfc'][:] = np.zeros(ngrdcol)

    elif l_t_dependent:
        # Generic l_t_dependent case: read surface fluxes from sfc file
        l_compute_momentum_flux = True
        l_set_sclr = True
        sfc = state.get('_forcings_data', {}).get('sfc') or {}
        times = sfc.get('time', np.array([0.0, 1e9]))
        if 'wpthlp_sfc' in sfc:
            wpthlp_sfc = np.full(ngrdcol, float(np.interp(time_current,
                                                            times, sfc['wpthlp_sfc'])))
            wpqtp_sfc  = np.full(ngrdcol, float(np.interp(time_current,
                                                            times, sfc.get('wpqtp_sfc',
                                                                           np.zeros(len(times))))))
            wprtp_sfc = _flux_spec_hum_to_mixing_ratio(rtm_bot, wpqtp_sfc)
            ustar = np.full(ngrdcol, 0.28)
        elif 'sens_ht' in sfc:
            sh = float(np.interp(time_current, times, sfc['sens_ht']))
            lh = float(np.interp(time_current, times, sfc['latent_ht']))
            wpthlp_sfc = sh / (rho_bot * Cp)
            wprtp_sfc  = lh / (rho_bot * Lv)
            ustar = np.full(ngrdcol, 0.28)
        else:
            wpthlp_sfc = np.zeros(ngrdcol)
            wprtp_sfc  = np.zeros(ngrdcol)
            ustar      = np.zeros(ngrdcol)
    else:
        raise NotImplementedError(
            f"prescribe_forcings_generic: sfclyr not yet ported for runtype={runtype!r}. "
            "Add to Benchmark_cases/generic_forcings.py."
        )

    # ── 5. Momentum flux ───────────────────────────────────────────────────
    if l_compute_momentum_flux:
        upwp_sfc, vpwp_sfc = _compute_momentum_flux(um_bot, vm_bot, ubar, ustar)

    # ── 6. Write back surface state ────────────────────────────────────────
    state['wpthlp_sfc'][:] = wpthlp_sfc
    state['wprtp_sfc'][:] = wprtp_sfc
    state['upwp_sfc'][:] = upwp_sfc
    state['vpwp_sfc'][:] = vpwp_sfc
    state['T_sfc'][:] = T_sfc
    state['ustar'] = ustar.copy() if hasattr(ustar, 'copy') else np.asarray(ustar)

    if l_set_sclr:
        _set_sclr_sfc_rtm_thlm(state, wpthlp_sfc, wprtp_sfc)

    # ── 7. Stats ────────────────────────────────────────────────────────────
    _stats_surface_update(state, wpthlp_sfc, wprtp_sfc, upwp_sfc, vpwp_sfc,
                          ustar, T_sfc, l_sample)
