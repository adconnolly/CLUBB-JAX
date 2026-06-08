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
from clubb_jax.src.CLUBB_core.tracer_numpy import _is_tracer_arg
# arm.F90 `use`s diag_ustar_module; the surface-flux block calls _diag_ustar/diag_ustar,
# which now live in their own Fortran-named module (re-imported here, also re-exported for the
# other case surface schemes that import them from arm).
from clubb_jax.src.Benchmark_cases.diag_ustar_module import _diag_ustar, diag_ustar
from clubb_jax.src.Benchmark_cases.sfc_flux import (
    compute_ht_mostr_flux, convert_sens_ht_to_km_s, convert_latent_ht_to_m_s)

# Physical constants — mirror the Fortran arm.F90 `use constants_clubb, only: grav` (Cp/Lv via sfc_flux.convert_*_ht)
from clubb_jax.src.CLUBB_core.constants_clubb import grav as _grav

# ARM surface layer constants (arm.F90)
_z0     = 0.035   # momentum roughness height [m]
_rho_sfc = 1.1   # surface density [kg/m^3]

# Fortran blank / dummy sentinel
_BLANK = -999.9
_EPS64 = np.finfo(np.float64).eps



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
    prescribe_forcings_arm() (in prescribe_forcings.py) to match Fortran behaviour exactly.
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


def arm_sfclyr(state: dict, time_current: float, ngrdcol: int, fd: dict, z_bot: float) -> None:
    """arm.F90:arm_sfclyr — GCSS-ARM surface fluxes of heat/moisture/momentum + friction velocity.
    Time-interpolates the prescribed sensible/latent heat to wpthlp_sfc/wprtp_sfc, derives ustar via
    diag_ustar, and the momentum fluxes upwp_sfc/vpwp_sfc, writing them into `state`. (Extracted
    verbatim from the inline block in `prescribe_forcings_arm` [now in prescribe_forcings.py, iter 386],
    mirror-refactor iter 237 — matches the named `*_sfclyr` of arm_97/arm_0003/arm_3year.)
    """
    # compute_ht_mostr_flux: linear interp of sfc fluxes (time_current concrete → concrete)
    heat_flx, moisture_flx = compute_ht_mostr_flux(
        time_current, fd['sfc_times'], fd['sens_ht'], fd['latent_ht'])
    wpthlp_val = convert_sens_ht_to_km_s(heat_flx, _rho_sfc)     # K m/s
    wprtp_val  = convert_latent_ht_to_m_s(moisture_flx, _rho_sfc)  # m/s

    # The surface-layer scalars depend on um/vm/thlm at the bottom level, so under a jax.grad trace
    # (REFACTOR B5) they are tracers. Dispatch on that: concrete runs keep the EXACT original Python
    # float path (bit-identical); the trace takes the differentiable jnp mirror (diag_ustar).
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
        ustar = diag_ustar(z_bot, bflx, ubar, _z0)
        state['wpthlp_sfc'] = jnp.full(ngrdcol, wpthlp_val)
        state['wprtp_sfc']  = jnp.full(ngrdcol, wprtp_val)
        state['upwp_sfc'] = jnp.full(ngrdcol, -um_bot * ustar ** 2 / ubar)
        state['vpwp_sfc'] = jnp.full(ngrdcol, -vm_bot * ustar ** 2 / ubar)
        state['ustar'] = jnp.full(ngrdcol, ustar)
