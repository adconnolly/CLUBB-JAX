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

import numpy as np

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Lv, p0, kappa
from clubb_jax.src.CLUBB_core.grid_class import zt2zm_jax
# Tracer-transparency (REFACTOR B5): _asarray/_xp/_iset behave EXACTLY like numpy for concrete arrays
# (normal runs bit-identical) but route to jnp under a jax.grad trace, so the whole-driver autodiff graph
# survives the surface-BC interpolation. See CLUBB_core/tracer_numpy.py.
from clubb_jax.src.CLUBB_core.tracer_numpy import _asarray, _xp, _iset
from clubb_jax.src.CLUBB_core.interpolation import mono_cubic_interp, linear_interp_factor
from clubb_jax.src.Benchmark_cases.spec_hum_to_mixing_ratio import flux_spec_hum_to_mixing_ratio
from clubb_jax.src.Benchmark_cases.sfc_flux import (
    compute_ubar, compute_momentum_flux, set_sclr_sfc_rtm_thlm,
    convert_sens_ht_to_km_s, convert_latent_ht_to_m_s,
)
from clubb_jax.src.Benchmark_cases.gabls3_night import gabls3_night_sfclyr
from clubb_jax.src.Benchmark_cases.bomex import bomex_tndcy, bomex_sfclyr
from clubb_jax.src.Benchmark_cases.dycoms2_rf01 import dycoms2_rf01_sfclyr, dycoms2_rf01_tndcy
from clubb_jax.src.Benchmark_cases.wangara import wangara_sfclyr, wangara_tndcy
from clubb_jax.src.Benchmark_cases.gabls2 import gabls2_tndcy, gabls2_sfclyr
from clubb_jax.src.Benchmark_cases.gabls3 import gabls3_sfclyr
from clubb_jax.src.Benchmark_cases.atex import atex_tndcy, atex_sfclyr
from clubb_jax.src.Benchmark_cases.atex_long import atex_long_tndcy, atex_long_sfclyr
from clubb_jax.src.Benchmark_cases.fire import fire_sfclyr
from clubb_jax.src.Benchmark_cases.dycoms2_rf02 import dycoms2_rf02_sfclyr, dycoms2_rf02_tndcy
from clubb_jax.src.Benchmark_cases.mpace_a import mpace_a_tndcy, mpace_a_sfclyr
from clubb_jax.src.Benchmark_cases.rico import rico_tndcy, rico_sfclyr
from clubb_jax.src.Benchmark_cases.neutral_case import neutral_case_sfclyr
from clubb_jax.src.Benchmark_cases.ekman import ekman_sfclyr
from clubb_jax.src.Benchmark_cases.cobra import cobra_sfclyr
from clubb_jax.src.Benchmark_cases.lba import lba_sfclyr
from clubb_jax.src.Benchmark_cases.arm_97 import arm_97_sfclyr
from clubb_jax.src.Benchmark_cases.arm_0003 import arm_0003_sfclyr
from clubb_jax.src.Benchmark_cases.arm_3year import arm_3year_sfclyr
from clubb_jax.src.Benchmark_cases.nov11 import nov11_altocu_rtm_adjust
from clubb_jax.src.Benchmark_cases.time_dependent_input import apply_time_dependent_forcings, time_select
from clubb_jax.src.Benchmark_cases.arm import arm_sfclyr
from clubb_jax.src.Benchmark_cases.cloud_feedback import cloud_feedback_sfclyr
from clubb_jax.src.Benchmark_cases.astex_a209 import astex_a209_sfclyr


# ── Unit conversion helpers: flux_/force_spec_hum_to_mixing_ratio now live in their Fortran-home
#    module Benchmark_cases/spec_hum_to_mixing_ratio.py (mirror-refactor iter 95), imported above. ──

# ── Surface BC extraction (prescribe_forcings.F90:read_surface_var_for_bc) ─

# read_surface_var_for_bc: `use constants_clubb, only: p0, kappa` (imported above).
_Z_BOT_CNVG = 25.0  # fixed model height for the convergence-test BC


# _fsign / _min3 / _mono_cubic_interp removed (mirror-refactor iter 55): the bottom-BC sounding
# interpolation now calls interpolation.mono_cubic_interp (interpolation.F90:mono_cubic_interp),
# byte-identical to the old local copy (proven 0 diff; at slope==0 the min-term zeros the sign).


def read_surface_var_for_bc(state: dict) -> dict:
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
            'exner_bot': (state['p_sfc'] / p0) ** kappa,
        }

    # l_modify_bc_for_cnvg_test = True, constant_height_option = 2 (interpolation).
    # The nearest-level indices come from the (concrete) grid only, so there is no data-dependent control
    # flow on the traced fields; the field VALUES flow through mono_cubic_interp (pure-jnp, tracer-safe).
    # Build per-column lists and _xp.stack them (jnp under a trace, numpy otherwise → bit-identical) so the
    # autodiff graph survives — the old np.empty()+in-place assignment severed it.
    ngrdcol = state['ngrdcol']
    zt = np.asarray(gr.zt)   # grid: always concrete
    zm = np.asarray(gr.zm)
    rho_zm = _asarray(state['rho_zm'])
    z_bot = _Z_BOT_CNVG

    um_zm    = _asarray(zt2zm_jax(jnp.asarray(state['um']), gr))
    vm_zm    = _asarray(zt2zm_jax(jnp.asarray(state['vm']), gr))
    thlm_zm  = _asarray(zt2zm_jax(jnp.asarray(state['thlm']), gr))
    rtm_zm   = _asarray(zt2zm_jax(jnp.asarray(state['rtm']), gr))
    exner_zm = _asarray(zt2zm_jax(jnp.asarray(state['exner']), gr))
    exner_zm = _iset(exner_zm, np.s_[:, 0], (state['p_sfc'] / p0) ** kappa)

    cols = {k: [] for k in ('um_bot', 'vm_bot', 'rtm_bot', 'thlm_bot', 'rho_bot', 'exner_bot')}
    for i in range(ngrdcol):
        kmin = int(np.argmin(np.abs(zt[i] - z_bot)))  # nearest zt level (0-based), concrete grid
        if kmin == 0:
            km1, k00, kp1, kp2 = 0, 0, 1, 2
        else:
            km1, k00, kp1, kp2 = kmin - 1, kmin, kmin + 1, kmin + 2
        zi = zm[i]
        cols['rho_bot'].append(rho_zm[i, kmin])
        for key, src in (('um_bot', um_zm), ('vm_bot', vm_zm),
                         ('thlm_bot', thlm_zm), ('rtm_bot', rtm_zm),
                         ('exner_bot', exner_zm)):
            cols[key].append(mono_cubic_interp(
                z_bot, km1, k00, kp1, kp2,
                zi[km1], zi[k00], zi[kp1], zi[kp2],
                src[i, km1], src[i, k00], src[i, kp1], src[i, kp2]))

    out = {k: _xp.stack(v) for k, v in cols.items()}
    out['z_bot'] = _xp.full(ngrdcol, z_bot)
    return out


# ── Shared surface utilities: compute_ubar / compute_momentum_flux / set_sclr_sfc_rtm_thlm now live in
#    their Fortran-home module Benchmark_cases/sfc_flux.py (mirror-refactor iter 96), imported above. ──

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


# ── _time_interp + apply_time_dependent_forcings now live in their Fortran-home module
#    Benchmark_cases/time_dependent_input.py (mirror-refactor iter 196), imported above. ──


# ── BOMEX: bomex_tndcy / bomex_sfclyr now live in their Fortran-home module Benchmark_cases/bomex.py
#    (mirror-refactor iter 98), imported above as bomex_tndcy / bomex_sfclyr. ──


# ── Simple zero-forcing cases ───────────────────────────────────────────────

def _zero_forcings(state: dict) -> None:
    """Zero all large-scale forcing arrays (fire/generic/neutral/coriolis_test/ekman)."""
    state['thlm_forcing'][:] = 0.0
    state['rtm_forcing'][:] = 0.0


# ── RICO ─────────────────────────────────────────────────────────────────────

# ── RICO: rico_sfclyr / rico_tndcy now live in their Fortran-home module Benchmark_cases/rico.py
#    (mirror-refactor iter 109), imported above as rico_sfclyr / rico_tndcy. ──


# ── DYCOMS2-RF01: dycoms2_rf01_sfclyr now lives in its Fortran-home module Benchmark_cases/dycoms2_rf01.py
#    (mirror-refactor iter 99), imported above as dycoms2_rf01_sfclyr. ──


# ── DYCOMS2-RF02 ──────────────────────────────────────────────────────────────

# ── DYCOMS2-RF02: dycoms2_rf02_sfclyr now lives in its Fortran-home module Benchmark_cases/dycoms2_rf02.py
#    (mirror-refactor iter 106), imported above as dycoms2_rf02_sfclyr. ──


# ── Wangara: wangara_sfclyr now lives in its Fortran-home module Benchmark_cases/wangara.py
#    (mirror-refactor iter 100), imported above as wangara_sfclyr. ──


# ── LBA: lba_sfclyr now lives in its Fortran-home module Benchmark_cases/lba.py (mirror-refactor iter 186),
#    imported above; the dispatch calls it directly (byte-identical to the former prescribe_forcings duplicate). ──


# ── GABLS2 ───────────────────────────────────────────────────────────────────

# ── ATEX: atex_tndcy / atex_sfclyr now live in their Fortran-home module Benchmark_cases/atex.py
#    (mirror-refactor iter 103), imported above as atex_tndcy / atex_sfclyr. ──


# ── ATEX-Long: atex_long_tndcy / atex_long_sfclyr now live in their Fortran-home module
#    Benchmark_cases/atex_long.py (mirror-refactor iter 104), imported above. ──


# ── GABLS2: gabls2_tndcy / gabls2_sfclyr now live in their Fortran-home module Benchmark_cases/gabls2.py
#    (mirror-refactor iter 101), imported above as gabls2_tndcy / gabls2_sfclyr. ──


# ── GABLS3-night surface layer (landflx + psi_h + gabls3_night_sfclyr) now lives in its Fortran-home module
#    Benchmark_cases/gabls3_night.py (mirror-refactor iter 97), imported above as gabls3_night_sfclyr. ──


# ── GABLS3 (daytime): gabls3_sfclyr now lives in its Fortran-home module Benchmark_cases/gabls3.py
#    (mirror-refactor iter 102), imported above as gabls3_sfclyr. ──


# ── ARM Variants (arm_0003, arm_97/mc3e) ─────────────────────────────────────

def _arm_variant_read_t_dependent(state: dict, time_current: float, ngrdcol: int,
                                  z_bot, rho_bot, thlm_bot, ubar, sfclyr_fn) -> tuple:
    """ARM-variant surface fluxes: read sens_ht/latent_ht from the sfc file (W/m^2) — the `*_read_t_dependent`
    step shared by arm_0003/arm_97/arm_3year/mc3e — then call the case's own `*_sfclyr` (which does the
    kinematic conversion + diag_ustar friction velocity; all four share the arm_97 scheme, z0=0.035)."""
    sfc = (state.get('_forcings_data') or {}).get('sfc') or {}
    times = sfc.get('time', np.array([0.0, 1e9]))

    def _interp_sfc(key, default):
        arr = sfc.get(key)
        if arr is not None and len(arr) > 0:
            return float(np.interp(time_current, times, arr))
        return default

    sens_ht   = _interp_sfc('sens_ht',   0.0)
    latent_ht = _interp_sfc('latent_ht', 0.0)
    return sfclyr_fn(sens_ht, latent_ht, z_bot, rho_bot, thlm_bot, ubar)


# ── Cloud-Feedback / CGILS / ASTEX-A209 ─────────────────────────────────────
# The bulk-aerodynamic drag-law surface schemes now live in their Fortran-home modules
# Benchmark_cases/cloud_feedback.py (cloud_feedback_sfclyr) + astex_a209.py (astex_a209_sfclyr),
# imported above; the dispatch interpolates T_sfc via _interp_sfc_t_sfc then calls them
# (mirror-refactor iter 188; the former prescribe_forcings _bulk_aero_sfclyr duplicate removed).

def _interp_sfc_t_sfc(state: dict, time_current: float) -> float:
    """Time-interpolate T_sfc from the case sfc file (the `*_read_t_dependent` T_sfc step). Default 298 K."""
    sfc = (state.get('_forcings_data') or {}).get('sfc') or {}
    times = sfc.get('time', np.array([0.0, 1e9]))
    T_sfc_arr = sfc.get('t_sfc')
    return float(np.interp(time_current, times, T_sfc_arr)) if T_sfc_arr is not None else 298.0


# ── Zero-flux altocu cases ────────────────────────────────────────────────────

def _zero_flux_sfclyr(ngrdcol: int) -> tuple:
    """No surface momentum or heat fluxes. ustar = 0.

    Port for nov11_altocu, jun25_altocu, clex9_nov02, clex9_oct14.
    """
    z = np.zeros(ngrdcol)
    return z.copy(), z.copy(), z.copy()  # wpthlp_sfc, wprtp_sfc, ustar


# ── Data loading ────────────────────────────────────────────────────────────

# ── linear_fill_blanks / fill_blanks_two_dim_vars now live in their Fortran-home module
#    Input_fields/input_reader.py (mirror-refactor iter 185), imported above. ──


# ── _parse_forcings_file / _parse_sfc_file now live in their Fortran-home module
#    Benchmark_cases/time_dependent_input.py (mirror-refactor iter 197), imported above. ──


# ── load_generic_forcings_data now lives in its Fortran-home module Benchmark_cases/time_dependent_input.py
#    (mirror-refactor iter 200), imported by clubb_driver. ──


# ── FIRE: fire_sfclyr now lives in its Fortran-home module Benchmark_cases/fire.py
#    (mirror-refactor iter 105), imported above as fire_sfclyr. ──


# ── MPACE-A: mpace_a_init / mpace_a_tndcy / mpace_a_sfclyr (+ the _read_mpace_dat / _mpace_time_select
#    helpers) now live in their Fortran-home module Benchmark_cases/mpace_a.py (mirror-refactor iter 107),
#    imported above. The vertical remap routes through interpolation.F90:zlinterp_fnc (mirror, iter 179). ──


# ── ARM branch (prescribe_forcings.F90 `case("arm")`; relocated from arm.py iter 386 to its Fortran-home
#    module — mirrors `use arm` for arm_sfclyr) ────────────────────────────────────────────────────────
_EPS64 = np.finfo(np.float64).eps


def _is_dummy_profile(profile: np.ndarray) -> bool:
    """Return True if any value in profile equals the -999.9 dummy sentinel.

    Mirrors the Fortran check in apply_time_dependent_forcings_from_array:
      .not. any( abs(temp - (-999.9)) < abs(temp + (-999.9)) / 2 * eps )
    """
    return bool(np.any(
        np.abs(profile + 999.9) < np.abs(profile - 999.9) / 2.0 * _EPS64
    ))


def prescribe_forcings_arm(state: dict, time_current: float) -> None:
    """Update state forcing fields for the ARM case — pure Python port.

    Mirrors the ARM branch of prescribe_forcings.F90 when l_t_dependent=True.
    Modifies state in-place.
    """
    fd     = state['_arm_forcings_data']
    ngrdcol = state['ngrdcol']
    nzt    = state['nzt']
    nzm    = state['nzm']

    # ── Reset forcing arrays (prescribe_forcings.F90 lines 311-330) ─────────
    state['rtm_forcing'][:]    = 0.0
    state['thlm_forcing'][:]   = 0.0
    state['wprtp_forcing'][:]  = 0.0
    state['wpthlp_forcing'][:] = 0.0
    state['rtp2_forcing'][:]   = 0.0
    state['thlp2_forcing'][:]  = 0.0
    state['rtpthlp_forcing'][:] = 0.0

    # ── apply_time_dependent_forcings ────────────────────────────────────────
    times  = fd['times']
    before, after, frac = time_select(time_current, times)

    # Time-interpolate each stored profile to current time via interpolation.F90:linear_interp_factor.
    for key, state_key in [
        ('thlm_f', 'thlm_forcing'),
        ('rtm_f',  'rtm_forcing'),
        ('w',      '_wm_zt_tmp'),
    ]:
        prof_before = fd[key][:, before]
        prof_after  = fd[key][:, after]
        profile = linear_interp_factor(frac, prof_after, prof_before)

        if _is_dummy_profile(profile):
            continue

        if key == 'w':
            # wm_name → wm_zt (broadcast to all columns)
            state['wm_zt'][:, :] = profile[np.newaxis, :]
            # Compute wm_zm via zt2zm
            import jax.numpy as jnp
            from clubb_jax.src.CLUBB_core.grid_class import zt2zm_jax
            wm_zt_jax = jnp.asarray(state['wm_zt'])
            state['wm_zm'] = np.asarray(
                zt2zm_jax(wm_zt_jax, state['gr']), dtype=np.float64
            )
        else:
            state[state_key][:, :] = profile[np.newaxis, :]

    # Zero top level (prescribe_forcings.F90 lines 370-373)
    state['rtm_forcing'][:,  -1] = 0.0
    state['thlm_forcing'][:, -1] = 0.0

    # ── read_surface_var_for_bc (l_modify_bc_for_cnvg_test=False) ────────────
    # Fortran uses gr%zt(i,1) which in Python 0-indexed is zt[col, 0]
    z_bot = float(state['gr'].zt[0, 0])   # grid: always concrete

    # ── arm_sfclyr ───────────────────────────────────────────────────────────
    arm_sfclyr(state, time_current, ngrdcol, fd, z_bot)


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
        mpace_a_tndcy(state, time_current)              # fully custom forcing files (mpace_a.F90)
    elif runtype == 'gabls2':
        gabls2_tndcy(state, time_current)               # analytic subsidence (off first 26 h)
    elif runtype == 'wangara':
        wangara_tndcy(state)                          # wangara.F90:wangara_tndcy — no LS forcing, no subsidence
    elif runtype == 'atex':
        atex_tndcy(state, time_current)                 # subsidence gated off the first 90 min
    elif runtype == 'atex_long':
        atex_long_tndcy(state, time_current)            # fixed 3-piece subsidence + thlm/rtm forcing + spinup
    elif l_t_dependent and not l_ignore_forcings:
        apply_time_dependent_forcings(state, time_current)
    else:
        if runtype == 'bomex':
            bomex_tndcy(state)
        elif runtype == 'rico':
            rico_tndcy(state)   # analytic thlm/qtm→rtm forcing; wm is init-set (untouched)
        elif runtype == 'dycoms2_rf01':
            dycoms2_rf01_tndcy(state)      # dycoms2_rf01.F90:dycoms2_rf01_tndcy (zero thlm/rtm; wm init-set)
        elif runtype in ('fire', 'generic', 'neutral', 'coriolis_test', 'ekman'):
            # No per-case Fortran tndcy — prescribe_forcings zeros the LS forcing generically.
            # (dycoms2_rf01_fixed_sst stays on the fallback — its fixed-SST sfclyr has a latent bug.)
            _zero_forcings(state)
        elif runtype == 'dycoms2_rf02':       # all rf02 variants (nd/so/do/ds) share this runtype
            dycoms2_rf02_tndcy(state)      # dycoms2_rf02.F90:dycoms2_rf02_tndcy (zero thlm/rtm + wm top)
        else:
            raise NotImplementedError(
                f"prescribe_forcings_generic: analytic tndcy not yet ported for "
                f"runtype={runtype!r}. Add to Benchmark_cases/prescribe_forcings.py."
            )

    # nov11.F90:nov11_altocu_rtm_adjust — the one-time above-cloud total-water adjustment (prescribe_forcings.F90
    # calls it for this case); a no-op except on the single timestep at t = time_initial + 3600 s.
    if runtype == 'nov11_altocu':
        gr = state['gr']
        state['rtm'] = np.asarray(nov11_altocu_rtm_adjust(
            state['rtm'], gr, time_current, state['time_initial'], state['dt_main']),
            dtype=np.float64)

    # ── 3. Bottom-level state for surface BCs ──────────────────────────────
    bc = read_surface_var_for_bc(state)
    um_bot    = bc['um_bot']
    vm_bot    = bc['vm_bot']
    rtm_bot   = bc['rtm_bot']
    thlm_bot  = bc['thlm_bot']
    rho_bot   = bc['rho_bot']
    exner_bot = bc['exner_bot']
    z_bot     = bc['z_bot']
    ubar = compute_ubar(um_bot, vm_bot)

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
        wpthlp_sfc, wprtp_sfc, ustar = bomex_sfclyr(state, time_current, ngrdcol, rtm_bot)

    elif runtype == 'rico':
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, upwp_sfc, vpwp_sfc, T_sfc = \
            rico_sfclyr(state, time_current, ngrdcol,
                         um_bot, vm_bot, thlm_bot, rtm_bot, rho_bot, exner_bot, z_bot, ubar)
        state['upwp_sfc'] = _iset(state['upwp_sfc'], np.s_[:], upwp_sfc)
        state['vpwp_sfc'] = _iset(state['vpwp_sfc'], np.s_[:], vpwp_sfc)
        # rico_sfclyr computes momentum flux directly (not via ustar^2/ubar)

    elif runtype == 'dycoms2_rf01':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, T_sfc = \
            dycoms2_rf01_sfclyr(state, time_current, ngrdcol, rho_bot,
                                    ubar, thlm_bot, rtm_bot, exner_bot)

    elif runtype in ('dycoms2_rf02', 'dycoms2_rf02_do', 'dycoms2_rf02_ds',
                     'dycoms2_rf02_so', 'dycoms2_rf02_nd', 'dycoms2_rf02_morr'):
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = dycoms2_rf02_sfclyr(state, time_current, ngrdcol)

    elif runtype == 'wangara':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = wangara_sfclyr(time_current, ngrdcol)

    elif runtype == 'gabls2':
        l_compute_momentum_flux = False  # gabls2 uses diag_ustar internally
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, T_sfc = gabls2_sfclyr(
            state, time_current, ngrdcol,
            z_bot, state['p_sfc'], ubar, thlm_bot, rtm_bot, exner_bot)
        upwp_sfc, vpwp_sfc = compute_momentum_flux(um_bot, vm_bot, ubar, ustar)
        state['upwp_sfc'] = _iset(state['upwp_sfc'], np.s_[:], upwp_sfc)
        state['vpwp_sfc'] = _iset(state['vpwp_sfc'], np.s_[:], vpwp_sfc)

    elif runtype == 'gabls3_night':
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, upwp_sfc, vpwp_sfc = gabls3_night_sfclyr(
            state, time_current, ngrdcol, um_bot, vm_bot, thlm_bot, rtm_bot, z_bot)
        state['upwp_sfc'] = _iset(state['upwp_sfc'], np.s_[:], upwp_sfc)
        state['vpwp_sfc'] = _iset(state['vpwp_sfc'], np.s_[:], vpwp_sfc)

    elif runtype == 'gabls3':
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = gabls3_sfclyr(
            state, ngrdcol, ubar, thlm_bot, rtm_bot, z_bot, exner_bot)
        upwp_sfc, vpwp_sfc = compute_momentum_flux(um_bot, vm_bot, ubar, ustar)
        state['upwp_sfc'] = _iset(state['upwp_sfc'], np.s_[:], upwp_sfc)
        state['vpwp_sfc'] = _iset(state['vpwp_sfc'], np.s_[:], vpwp_sfc)

    elif runtype == 'mpace_a':
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = mpace_a_sfclyr(
            state, time_current, ngrdcol, np.asarray(state['rho_zm'])[:, 0])
        upwp_sfc, vpwp_sfc = compute_momentum_flux(um_bot, vm_bot, ubar, ustar)
        state['upwp_sfc'] = _iset(state['upwp_sfc'], np.s_[:], upwp_sfc)
        state['vpwp_sfc'] = _iset(state['vpwp_sfc'], np.s_[:], vpwp_sfc)

    elif runtype == 'atex':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, T_sfc = atex_sfclyr(
            state, time_current, ngrdcol, ubar, thlm_bot, rtm_bot, exner_bot)

    elif runtype == 'atex_long':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, T_sfc = atex_long_sfclyr(
            state, time_current, ngrdcol, ubar, thlm_bot, rtm_bot, exner_bot)

    elif runtype == 'arm_0003':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _arm_variant_read_t_dependent(
            state, time_current, ngrdcol, z_bot, rho_bot, thlm_bot, ubar, arm_0003_sfclyr)

    elif runtype == 'arm_3year':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _arm_variant_read_t_dependent(
            state, time_current, ngrdcol, z_bot, rho_bot, thlm_bot, ubar, arm_3year_sfclyr)

    elif runtype in ('arm_97', 'mc3e'):
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _arm_variant_read_t_dependent(
            state, time_current, ngrdcol, z_bot, rho_bot, thlm_bot, ubar, arm_97_sfclyr)

    elif runtype in ('cloud_feedback_s6', 'cloud_feedback_s6_p2k',
                     'cloud_feedback_s11', 'cloud_feedback_s11_p2k',
                     'cloud_feedback_s12', 'cloud_feedback_s12_p2k',
                     'cgils_s6', 'cgils_s6_p2k', 'cgils_s11',
                     'cgils_s11_p2k', 'cgils_s12', 'cgils_s12_p2k'):
        l_compute_momentum_flux = True
        l_set_sclr = True
        T_sfc_val = _interp_sfc_t_sfc(state, time_current)
        T_sfc = np.full(ngrdcol, T_sfc_val)
        wpthlp_sfc, wprtp_sfc, ustar = cloud_feedback_sfclyr(
            thlm_bot, rtm_bot, z_bot, ubar, state['p_sfc'], T_sfc_val,
            state['flags'].saturation_formula)

    elif runtype == 'astex_a209':
        l_compute_momentum_flux = True
        l_set_sclr = True
        T_sfc_val = _interp_sfc_t_sfc(state, time_current)
        T_sfc = np.full(ngrdcol, T_sfc_val)
        wpthlp_sfc, wprtp_sfc, ustar = astex_a209_sfclyr(
            thlm_bot, rtm_bot, z_bot, ubar, state['p_sfc'], T_sfc_val,
            state['flags'].saturation_formula)

    elif runtype in ('nov11_altocu', 'jun25_altocu',
                     'clex9_nov02', 'clex9_oct14'):
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = _zero_flux_sfclyr(ngrdcol)
        upwp_sfc = np.zeros(ngrdcol)
        vpwp_sfc = np.zeros(ngrdcol)
        state['upwp_sfc'] = _iset(state['upwp_sfc'], np.s_[:], upwp_sfc)
        state['vpwp_sfc'] = _iset(state['vpwp_sfc'], np.s_[:], vpwp_sfc)

    elif runtype == 'cobra':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar = cobra_sfclyr(
            state, time_current, ngrdcol, z_bot, rho_bot, thlm_bot, ubar)
        # T_sfc from sfc file
        sfc = state.get('_forcings_data', {}).get('sfc') or {}
        times_sfc = sfc.get('time', np.array([0.0]))
        T_sfc_arr = sfc.get('t_sfc')
        if T_sfc_arr is not None:
            T_sfc = np.full(ngrdcol, float(np.interp(time_current, times_sfc, T_sfc_arr)))

    elif runtype == 'lba':
        l_compute_momentum_flux = True
        l_set_sclr = True
        time_elapsed = time_current - state['time_initial']
        wpthlp_sfc, wprtp_sfc, ustar = lba_sfclyr(time_elapsed, z_bot, rho_bot, thlm_bot, ubar)

    elif runtype == 'fire':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc, wprtp_sfc, ustar, T_sfc = fire_sfclyr(
            state, time_current, ngrdcol, ubar, thlm_bot, rtm_bot, exner_bot)

    elif runtype == 'generic':
        l_compute_momentum_flux = True
        l_set_sclr = True
        wpthlp_sfc = np.full(ngrdcol, float(state.get('sens_ht', 0.0)))
        wprtp_sfc  = np.full(ngrdcol, float(state.get('latent_ht', 0.0)))
        ustar      = np.full(ngrdcol, 0.3)

    elif runtype == 'neutral':
        l_set_sclr = True
        upwp_sfc, vpwp_sfc, wpthlp_sfc, wprtp_sfc, ustar = neutral_case_sfclyr(
            ngrdcol, time_current, um_bot, vm_bot, ubar)
        state['upwp_sfc'] = _iset(state['upwp_sfc'], np.s_[:], upwp_sfc)
        state['vpwp_sfc'] = _iset(state['vpwp_sfc'], np.s_[:], vpwp_sfc)

    elif runtype == 'ekman':
        l_set_sclr = True
        upwp_sfc, vpwp_sfc, wpthlp_sfc, wprtp_sfc, ustar = ekman_sfclyr(
            ngrdcol, um_bot, vm_bot, ubar)
        state['upwp_sfc'] = _iset(state['upwp_sfc'], np.s_[:], upwp_sfc)
        state['vpwp_sfc'] = _iset(state['vpwp_sfc'], np.s_[:], vpwp_sfc)

    elif runtype == 'coriolis_test':
        # Faithful: prescribe_forcings.F90:837 sets ustar=0 + l_fixed_flux, and coriolis_test_sfc.in prescribes
        # all-zero fixed fluxes (wpqtp/wpthlp/upwp/vpwp = 0), so every surface flux is zero (with ustar=0 the
        # computed momentum flux is 0 too). The analytic Foucault benchmark has no surface forcing.
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
            wprtp_sfc = flux_spec_hum_to_mixing_ratio(rtm_bot, wpqtp_sfc)
            ustar = np.full(ngrdcol, 0.28)
        elif 'sens_ht' in sfc:
            sh = float(np.interp(time_current, times, sfc['sens_ht']))
            lh = float(np.interp(time_current, times, sfc['latent_ht']))
            wpthlp_sfc = convert_sens_ht_to_km_s(sh, rho_bot)
            wprtp_sfc  = convert_latent_ht_to_m_s(lh, rho_bot)
            ustar = np.full(ngrdcol, 0.28)
        else:
            wpthlp_sfc = np.zeros(ngrdcol)
            wprtp_sfc  = np.zeros(ngrdcol)
            ustar      = np.zeros(ngrdcol)
    else:
        raise NotImplementedError(
            f"prescribe_forcings_generic: sfclyr not yet ported for runtype={runtype!r}. "
            "Add to Benchmark_cases/prescribe_forcings.py."
        )

    # ── 5. Momentum flux ───────────────────────────────────────────────────
    if l_compute_momentum_flux:
        upwp_sfc, vpwp_sfc = compute_momentum_flux(um_bot, vm_bot, ubar, ustar)

    # ── 6. Write back surface state ────────────────────────────────────────
    # _iset (REFACTOR B5): under a jax.grad trace the surface fluxes are tracers, so the in-place
    # `state[k][:] = ...` would sever the graph; _iset is functional under trace, in-place otherwise.
    state['wpthlp_sfc'] = _iset(state['wpthlp_sfc'], np.s_[:], wpthlp_sfc)
    state['wprtp_sfc']  = _iset(state['wprtp_sfc'],  np.s_[:], wprtp_sfc)
    state['upwp_sfc']   = _iset(state['upwp_sfc'],   np.s_[:], upwp_sfc)
    state['vpwp_sfc']   = _iset(state['vpwp_sfc'],   np.s_[:], vpwp_sfc)
    state['T_sfc']      = _iset(state['T_sfc'],      np.s_[:], T_sfc)
    state['ustar'] = ustar.copy() if hasattr(ustar, 'copy') else np.asarray(ustar)

    if l_set_sclr:
        set_sclr_sfc_rtm_thlm(state, wpthlp_sfc, wprtp_sfc)

    # ── 7. Stats ────────────────────────────────────────────────────────────
    _stats_surface_update(state, wpthlp_sfc, wprtp_sfc, upwp_sfc, vpwp_sfc,
                          ustar, T_sfc, l_sample)
