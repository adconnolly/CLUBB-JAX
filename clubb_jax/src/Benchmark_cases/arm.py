"""ARM benchmark case forcing prescriber — pure Python port of arm.F90, sfc_flux.F90,
time_dependent_input.F90, and diag_ustar_module.F90 for the ARM case.

Constants from constants_clubb.F90 (standalone block):
  Cp = 1004.67 J/(kg K), Lv = 2.5e6 J/kg, grav = 9.81 m/s^2, vonk = 0.4
"""

import math
import numpy as np
import jax.numpy as jnp
# Tracer-transparency (REFACTOR B5): only used to detect a jax.grad trace and route the surface-flux
# block through a differentiable jnp mirror. Normal (concrete) runs take the original float path
# unchanged → bit-identical. See CLUBB_core/tracer_numpy.py.
from clubb_jax.src.CLUBB_core.tracer_numpy import _is_tracer_arg, _safe_sqrt

# Physical constants matching constants_clubb.F90 standalone block
_Cp   = 1004.67   # dry air specific heat at const pressure [J/(kg K)]
_Lv   = 2.5e6     # latent heat of vaporization [J/kg]
_grav = 9.81      # gravitational acceleration [m/s^2]
_vonk = 0.4       # von Karman constant [-]

# ARM surface layer constants (arm.F90)
_z0     = 0.035   # momentum roughness height [m]
_rho_sfc = 1.1   # surface density [kg/m^3]

# Monin-Obukhov similarity constants (diag_ustar_module.F90)
_am = 4.8
_bm = 19.3

# Fortran blank / dummy sentinel
_BLANK = -999.9
_EPS64 = np.finfo(np.float64).eps


def _is_dummy_profile(profile: np.ndarray) -> bool:
    """Return True if any value in profile equals the -999.9 dummy sentinel.

    Mirrors the Fortran check in apply_time_dependent_forcings_from_array:
      .not. any( abs(temp - (-999.9)) < abs(temp + (-999.9)) / 2 * eps )
    """
    return bool(np.any(
        np.abs(profile + 999.9) < np.abs(profile - 999.9) / 2.0 * _EPS64
    ))


def _diag_ustar(z: float, bflx: float, wnd: float, z0: float) -> float:
    """Monin-Obukhov ustar — direct port of diag_ustar_module.F90."""
    lnz  = math.log(z / z0)
    klnz = _vonk / lnz
    c1   = math.pi / 2.0 - 3.0 * math.log(2.0)
    ustar = wnd * klnz

    if abs(bflx) > 1.0e-6:
        for _ in range(4):
            lmo  = -(ustar ** 3) / (_vonk * bflx)
            zeta = z / lmo
            if zeta > 0.0:
                if zeta > 1.0e10:
                    ustar = 1.0e-10
                    break
                ustar = _vonk * wnd / (lnz + _am * zeta)
            else:
                x    = math.sqrt(math.sqrt(1.0 - _bm * zeta))
                psi1 = (2.0 * math.log(1.0 + x)
                        + math.log(1.0 + x * x)
                        - 2.0 * math.atan(x)
                        + c1)
                ustar = wnd * _vonk / (lnz - psi1)

    return ustar


def _diag_ustar_jax(z, bflx, wnd, z0):
    """Differentiable jnp mirror of ``_diag_ustar`` (used ONLY under a jax.grad trace).

    The two data-dependent branches (``abs(bflx)>1e-6`` gate; stable ``zeta>0`` vs unstable, plus the
    ``zeta>1e10`` clamp) become ``jnp.where`` selections, and the 4-iteration Monin-Obukhov loop is
    unrolled. Forward-identical to ``_diag_ustar`` on the taken branch; finite reverse-mode gradient
    (``bflx`` guarded away from 0 so the unused branch can't poison the grad; ``_safe_sqrt`` for the
    quartic root)."""
    lnz   = jnp.log(z / z0)
    klnz  = _vonk / lnz
    c1    = jnp.pi / 2.0 - 3.0 * jnp.log(2.0)
    ustar = wnd * klnz
    apply = jnp.abs(bflx) > 1.0e-6
    bflx_safe = jnp.where(apply, bflx, 1.0)   # avoid 1/0 in lmo when the MO correction is inactive
    for _ in range(4):
        lmo  = -(ustar ** 3) / (_vonk * bflx_safe)
        # Sign-preserving floor on lmo (REFACTOR B5): in very stable conditions (gabls3) the MO iteration
        # drives ustar->0 so lmo->0 and `z/lmo` would blow the reverse-mode gradient to inf (then 0*inf=nan in
        # the zeta>1e10 clamp). Forward-identical for non-degenerate lmo (e.g. arm, which never hits this).
        _tiny = 1.0e-12
        lmo_safe = jnp.where(jnp.abs(lmo) > _tiny, lmo, jnp.where(lmo >= 0.0, _tiny, -_tiny))
        zeta = z / lmo_safe
        ustar_stable = jnp.where(zeta > 1.0e10, 1.0e-10,
                                 _vonk * wnd / (lnz + _am * zeta))
        x    = _safe_sqrt(_safe_sqrt(1.0 - _bm * zeta))
        psi1 = (2.0 * jnp.log(1.0 + x)
                + jnp.log(1.0 + x * x)
                - 2.0 * jnp.arctan(x)
                + c1)
        ustar_unstable = wnd * _vonk / (lnz - psi1)
        ustar_new = jnp.where(zeta > 0.0, ustar_stable, ustar_unstable)
        ustar = jnp.where(apply, ustar_new, ustar)
    return ustar


def _time_interp_sfc(time: float,
                     times: np.ndarray,
                     values: np.ndarray) -> float:
    """Linear interpolation with flat extrapolation — mirrors compute_ht_mostr_flux."""
    if time <= times[0]:
        return float(values[0])
    if time >= times[-1]:
        return float(values[-1])
    i = int(np.searchsorted(times, time, side='right')) - 1
    i = min(i, len(times) - 2)
    frac = (time - times[i]) / (times[i + 1] - times[i])
    return float(values[i] + frac * (values[i + 1] - values[i]))


def _time_select(time: float, times: np.ndarray):
    """Return (before_idx, after_idx, frac) — port of time_select in time_dependent_input.F90.

    Raises ValueError if time is outside [times[0], times[-1]].
    """
    if time < times[0] or time > times[-1]:
        raise ValueError(
            f"time {time} outside forcing time range [{times[0]}, {times[-1]}]"
        )
    before = int(np.searchsorted(times, time, side='right')) - 1
    before = min(before, len(times) - 2)
    after  = before + 1
    frac   = (time - times[before]) / (times[after] - times[before])
    return before, after, float(frac)


def load_arm_forcings_data(arm_forcings_file: str,
                           arm_sfc_file: str,
                           zt: np.ndarray) -> dict:
    """Parse ARM forcing files and pre-interpolate to the model zt grid.

    Returns a dict with keys:
      'times'   : np.ndarray (ntimes,) — forcing times [s]
      'thlm_f'  : np.ndarray (nzt, ntimes) — theta_l forcing [K/s]
      'rtm_f'   : np.ndarray (nzt, ntimes) — rt forcing [kg/kg/s]
      'w'       : np.ndarray (nzt, ntimes) — vertical velocity [m/s]
      'sfc_times'   : np.ndarray (ntimes_sfc,) — surface forcing times [s]
      'sens_ht'     : np.ndarray (ntimes_sfc,) — sensible heat flux [W/m^2]
      'latent_ht'   : np.ndarray (ntimes_sfc,) — latent heat flux [W/m^2]

    All dummy (-999.9) columns are stored as-is; the dummy check is deferred to
    prescribe_forcings_arm() to match Fortran behaviour exactly.
    """
    nzt = zt.shape[0]

    # ── Parse arm_forcings.in ────────────────────────────────────────────────
    with open(arm_forcings_file) as fh:
        lines = [ln for ln in fh if not ln.strip().startswith('!')]

    # First non-comment line is the column header
    col_header = lines[0].split()
    # Strip surrounding quotes from names like 'thlm_f[K/s]' → thlm_f[K/s]
    col_names = [c.strip("'\"") for c in col_header]
    # col_names[0] = 'z[m]', rest are forcing variables

    forcing_col_names = col_names[1:]   # e.g. ['thlm_f[K/s]', 'rtm_f[kg/kg/s]', ...]

    # Collect time blocks
    raw_times  = []
    raw_z      = []
    raw_data   = {name: [] for name in forcing_col_names}

    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) == 2:
            # Time header: "time nlevels"
            t_block = float(parts[0])
            n_levels = int(parts[1])
            raw_times.append(t_block)
            z_block  = []
            data_block = {name: [] for name in forcing_col_names}
            i += 1
            for _ in range(n_levels):
                row = lines[i].split()
                z_block.append(float(row[0]))
                for j, name in enumerate(forcing_col_names):
                    data_block[name].append(float(row[j + 1]))
                i += 1
            raw_z.append(np.array(z_block, dtype=np.float64))
            for name in forcing_col_names:
                raw_data[name].append(np.array(data_block[name], dtype=np.float64))
        else:
            i += 1

    times = np.array(raw_times, dtype=np.float64)   # (ntimes,)
    ntimes = len(times)

    # ── fill_blanks_two_dim_vars (along z then along time) ──────────────────
    # For ARM, all useful columns have no blanks, so fill_blanks is a no-op.
    # Dummy columns (-999.9 everywhere) stay -999.9 (default_value = blank_value).

    # ── Vertical interpolation to model grid ────────────────────────────────
    # Matches zlinterp_fnc: linear interp, 0 outside data range.
    interp_data = {}
    for name in forcing_col_names:
        arr = np.zeros((nzt, ntimes), dtype=np.float64)
        for it in range(ntimes):
            z_src = raw_z[it]
            v_src = raw_data[name][it]
            # If all values are the dummy sentinel, leave as -999.9 after interp
            # (zlinterp would produce nonsensical values; we preserve the sentinel)
            if np.all(np.abs(v_src + 999.9) < np.abs(v_src - 999.9) / 2.0 * _EPS64):
                arr[:, it] = _BLANK
            else:
                arr[:, it] = np.interp(zt, z_src, v_src, left=0.0, right=0.0)
        interp_data[name] = arr

    # ── Parse arm_sfc.in ────────────────────────────────────────────────────
    with open(arm_sfc_file) as fh:
        sfc_lines = [ln for ln in fh if not ln.strip().startswith('!')]

    # First non-comment line is the header; rest are data rows
    sfc_times_list  = []
    latent_ht_list  = []
    sens_ht_list    = []
    for ln in sfc_lines[1:]:
        parts = ln.split()
        if len(parts) < 3:
            continue
        sfc_times_list.append(float(parts[0]))
        latent_ht_list.append(float(parts[1]))
        sens_ht_list.append(float(parts[2]))

    sfc_times = np.array(sfc_times_list, dtype=np.float64)
    latent_ht = np.array(latent_ht_list, dtype=np.float64)
    sens_ht   = np.array(sens_ht_list,   dtype=np.float64)

    return {
        'times'     : times,
        'thlm_f'    : interp_data.get('thlm_f[K/s]',     np.full((nzt, ntimes), _BLANK)),
        'rtm_f'     : interp_data.get('rtm_f[kg/kg/s]',  np.full((nzt, ntimes), _BLANK)),
        'w'         : interp_data.get('w[m/s]',           np.full((nzt, ntimes), _BLANK)),
        'sfc_times' : sfc_times,
        'sens_ht'   : sens_ht,
        'latent_ht' : latent_ht,
    }


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
    before, after, frac = _time_select(time_current, times)

    # Time-interpolate each stored profile to current time
    # linear_interp_factor: frac * (v_after - v_before) + v_before
    for key, state_key, is_zt in [
        ('thlm_f', 'thlm_forcing', True),
        ('rtm_f',  'rtm_forcing',  True),
        ('w',      '_wm_zt_tmp',   True),
    ]:
        prof_before = fd[key][:, before]
        prof_after  = fd[key][:, after]
        profile = prof_before + frac * (prof_after - prof_before)

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
    # compute_ht_mostr_flux: linear interp of sfc fluxes (time_current concrete → concrete)
    heat_flx     = _time_interp_sfc(time_current, fd['sfc_times'], fd['sens_ht'])
    moisture_flx = _time_interp_sfc(time_current, fd['sfc_times'], fd['latent_ht'])
    wpthlp_val = heat_flx     / (_rho_sfc * _Cp)   # K m/s
    wprtp_val  = moisture_flx / (_rho_sfc * _Lv)   # m/s

    # The surface-layer scalars depend on um/vm/thlm at the bottom level, so under a jax.grad trace
    # (REFACTOR B5) they are tracers. Dispatch on that: concrete runs keep the EXACT original Python
    # float path (bit-identical); the trace takes the differentiable jnp mirror (_diag_ustar_jax).
    _um0, _vm0, _thlm0 = state['um'][0, 0], state['vm'][0, 0], state['thlm'][0, 0]
    if not _is_tracer_arg([_um0, _vm0, _thlm0]):
        um_bot   = float(_um0)
        vm_bot   = float(_vm0)
        thlm_bot = float(_thlm0)
        ubar  = max(0.25, math.sqrt(um_bot ** 2 + vm_bot ** 2))    # compute_ubar
        bflx  = (_grav / thlm_bot) * wpthlp_val                    # buoyancy flux (m^2/s^3)
        ustar = _diag_ustar(z_bot, bflx, ubar, _z0)                # scalar, ngrdcol=1 for SCM
        state['wpthlp_sfc'][:] = wpthlp_val
        state['wprtp_sfc'][:]  = wprtp_val
        state['upwp_sfc'][:] = -um_bot * ustar ** 2 / ubar         # compute_momentum_flux
        state['vpwp_sfc'][:] = -vm_bot * ustar ** 2 / ubar
        state['ustar'] = np.full(ngrdcol, ustar)
    else:
        um_bot, vm_bot, thlm_bot = _um0, _vm0, _thlm0
        ubar  = jnp.maximum(0.25, jnp.sqrt(um_bot ** 2 + vm_bot ** 2))
        bflx  = (_grav / thlm_bot) * wpthlp_val
        ustar = _diag_ustar_jax(z_bot, bflx, ubar, _z0)
        state['wpthlp_sfc'] = jnp.full(ngrdcol, wpthlp_val)
        state['wprtp_sfc']  = jnp.full(ngrdcol, wprtp_val)
        state['upwp_sfc'] = jnp.full(ngrdcol, -um_bot * ustar ** 2 / ubar)
        state['vpwp_sfc'] = jnp.full(ngrdcol, -vm_bot * ustar ** 2 / ubar)
        state['ustar'] = jnp.full(ngrdcol, ustar)
