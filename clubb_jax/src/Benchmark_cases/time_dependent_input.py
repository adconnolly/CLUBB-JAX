"""JAX port of time_dependent_input.F90 — the generic time-dependent large-scale forcing application.

Mirrors clubb_release/src/Benchmark_cases/time_dependent_input.F90: `apply_time_dependent_forcings` — time-
interpolates the pre-loaded forcing table (thlm_f/T_f, rtm_f/sp_hmdty_f, the subsidence w or omega, the u/v
momentum forcings, the time-dependent geostrophic wind, and the u/v nudging targets) onto the current time and
writes the corresponding `state` forcing arrays. prescribe_forcings.py imports it, mirroring the Fortran
`use time_dependent_input` in prescribe_forcings. The forcing TABLE is parsed at init (prescribe_forcings
_parse_forcings_file, the input_reader-style reader); this module only applies it per timestep.

Pure-numpy / tracer-transparent → bit-identical (the height-coordinate gate cases stay byte-identical).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.grid_class import zt2zm_jax
import os
from pathlib import Path

from clubb_jax.src.Input_fields.input_reader import _BLANK, fill_blanks_two_dim_vars
from clubb_jax.src.Benchmark_cases.mpace_a import mpace_a_init


def time_select(time: float, times: np.ndarray):
    """Find the before/after time bracket + interpolation fraction for `time` (time_dependent_input.F90:time_select).
    Returns (before_idx, after_idx, frac), 0-based. Raises ValueError if `time` is outside [times[0], times[-1]]."""
    if time < times[0] or time > times[-1]:
        raise ValueError(
            f"time {time} outside forcing time range [{times[0]}, {times[-1]}]"
        )
    before = int(np.searchsorted(times, time, side='right')) - 1
    before = min(before, len(times) - 2)
    after  = before + 1
    frac   = (time - times[before]) / (times[after] - times[before])
    return before, after, float(frac)


def _time_interp(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation of values[..., i] at time t using times array."""
    t = np.clip(t, times[0], times[-1])
    idx = np.searchsorted(times, t, side='right') - 1
    idx = np.clip(idx, 0, len(times) - 2)
    frac = (t - times[idx]) / (times[idx + 1] - times[idx] + 1e-300)
    return (1.0 - frac) * values[..., idx] + frac * values[..., idx + 1]



def apply_time_dependent_forcings(state: dict, time_current: float) -> None:
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
    T_f    = _interp_col('T_f')          # absolute-temperature forcing (CGILS cases): thlm_f = T_f/exner
    rtm_f  = _interp_col('rtm_f')
    sp_hmdty_f = _interp_col('sp_hmdty_f')
    w_zt   = _interp_col('w')
    omega  = _interp_col('omega')
    um_f   = _interp_col('um_f')
    vm_f   = _interp_col('vm_f')
    um_ref = _interp_col('um_ref')       # time-dependent u/v nudging targets (CGILS cases; l_uv_nudge)
    vm_ref = _interp_col('vm_ref')
    ug_zt  = _interp_col('ug')
    vg_zt  = _interp_col('vg')

    if thlm_f is not None:
        state['thlm_forcing'][:, :] = thlm_f[np.newaxis, :]
        state['thlm_forcing'][:, -1] = 0.0  # zero at top (Fortran convention)
    elif T_f is not None:
        # Absolute-temperature forcing → thlm forcing: thlm_f = T_f / exner (time_dependent_input.F90:671).
        exner = np.asarray(state['exner'], dtype=np.float64)
        state['thlm_forcing'][:, :] = T_f[np.newaxis, :] / exner
        state['thlm_forcing'][:, -1] = 0.0
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
        from clubb_jax.src.CLUBB_core.constants_clubb import grav, sec_per_hr, pascal_per_mb
        # mb/hr → Pa/s: × pascal_per_mb / sec_per_hr (time_dependent_input.F90:823)
        fac = (pascal_per_mb / sec_per_hr) if fd.get('omega_mb_hr') else 1.0
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
    # Time-dependent nudging targets (time_dependent_input.F90:762-776). Gated cases either don't nudge or
    # have blank um_ref columns (→ _interp_col None), so this is a no-op for them.
    if um_ref is not None and state.get('um_ref') is not None:
        state['um_ref'][:, :] = um_ref[np.newaxis, :]
    if vm_ref is not None and state.get('vm_ref') is not None:
        state['vm_ref'][:, :] = vm_ref[np.newaxis, :]
    if ug_zt is not None and state.get('ug') is not None:
        state['ug'][:, :] = ug_zt[np.newaxis, :]
    if vg_zt is not None and state.get('vg') is not None:
        state['vg'][:, :] = vg_zt[np.newaxis, :]


# ── initialize_t_dependent_forcings: the *_forcings.in / *_sfc.in table parsers (input_reader-style) ──

def _parse_forcings_file(path: str, zt: np.ndarray, p_in_Pa: np.ndarray = None) -> dict:
    """Parse a {case}_forcings.in file. Same format as arm_forcings.in.

    Applies fill_blanks_two_dim_vars (port of Fortran) before interpolating
    to model grid, so -999.9 sentinel values are filled by interpolation.

    The first column is the vertical coordinate, either height (`z[m]`/`Height[m]`) or pressure (`Press[Pa]`).
    For a pressure coordinate the forcing is interpolated against the model pressure profile `p_in_Pa` (the
    Fortran negates both so the interpolation runs on an ascending coordinate; np.interp only needs the SOURCE
    ascending, which ascending pressure already is, so no negation is needed here). The height path is unchanged
    when `p_in_Pa` is None or the coordinate is height — so every height-coordinate (gated) case is byte-identical.
    """
    nzt = zt.shape[0]
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.strip().startswith('!')]
    if not lines:
        return {}

    first_tokens = lines[0].split()
    vcoord_name = first_tokens[0].strip("'\"").split('[')[0].strip().lower()
    l_pressure_coord = vcoord_name in ('press', 'pressure')
    # Target coordinate for the final interpolation: model pressure (Pa) for a pressure file, else height (zt).
    target_coord = np.asarray(p_in_Pa) if (l_pressure_coord and p_in_Pa is not None) else zt
    col_names = [c.strip("'\"") for c in first_tokens][1:]  # skip the vertical-coordinate column

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
        filled = fill_blanks_two_dim_vars(all_z, times, raw_on_common)

        # Step 3: interpolate filled data to model zt grid
        # If still all blank after fill (shouldn't happen), leave as zero
        arr = np.zeros((nzt, ntimes))
        for it in range(ntimes):
            col = filled[:, it]
            if np.all(col <= _BLANK):
                arr[:, it] = 0.0
            else:
                valid = col > _BLANK
                # Interpolate to the model grid: against height (zt) or model pressure (target_coord). np.interp
                # requires the SOURCE (all_z) ascending — true for both ascending height and ascending pressure.
                # ZERO-FILL outside the forcing's range (left=right=0), matching the Fortran reader, which
                # interpolates with `zlinterp_fnc` (interpolation.F90, left=0/right=0) via read_to_grid — NOT
                # constant edge-extrapolation. This was the cloud_feedback step-1 forcing bug (Iter97): its forcing
                # bottom (100731 Pa) is above the model's lowest levels, so those out-of-range levels were
                # edge-extrapolated (≈ −1.6e-5) instead of zeroed. Gated cases' forcings cover the model range
                # (no out-of-range region exercised), so this is byte-identical for them.
                arr[:, it] = np.interp(target_coord, all_z[valid], col[valid], left=0.0, right=0.0)
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
        'T_f':    _col('T_f'),   # absolute-temperature forcing (CGILS) — converted to thlm_f=T_f/exner at apply

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


# ── initialize_t_dependent_input: load the case forcing + sfc tables at init ──

def load_generic_forcings_data(runtype: str, case_dir: str, zt: np.ndarray,
                               p_in_Pa: np.ndarray = None) -> dict:
    """Load forcing and surface files for a non-ARM benchmark case.

    Looks for {case_dir}/{runtype}_forcings.in and {case_dir}/{runtype}_sfc.in.
    Returns a dict with 'times', 'thlm_f', 'rtm_f', 'w', 'sfc'.

    `p_in_Pa` (the model pressure profile on the zt grid) is needed only for forcing files with a `Press[Pa]`
    vertical coordinate (the CGILS / cloud-feedback cases); height-coordinate files ignore it.
    """
    forcings_path = os.path.join(case_dir, f'{runtype}_forcings.in')
    sfc_path = os.path.join(case_dir, f'{runtype}_sfc.in')

    fd = {'times': None, 'thlm_f': None, 'rtm_f': None, 'w': None, 'sfc': None}

    if runtype == 'mpace_a':
        fd.update(mpace_a_init(case_dir))       # custom mpace_a_forcings/*.dat (mpace_a.F90)
        return fd

    if os.path.isfile(forcings_path):
        parsed = _parse_forcings_file(forcings_path, zt, p_in_Pa=p_in_Pa)
        fd.update(parsed)

    if os.path.isfile(sfc_path):
        fd['sfc'] = _parse_sfc_file(sfc_path)

    return fd
