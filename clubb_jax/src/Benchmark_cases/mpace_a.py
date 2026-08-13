"""JAX port of mpace_a.F90 — M-PACE A (Arctic mixed-phase stratus) custom forcing + surface fluxes.

Mirrors clubb_release/src/Benchmark_cases/mpace_a.F90: the case's custom forcing-file loader (`mpace_a_init`),
`mpace_a_tndcy` (large-scale tendencies time/height-interpolated from the mpace_a_forcings/*.dat files;
subsidence deliberately removed), and `mpace_a_sfclyr` (prescribed sensible/latent heat → kinematic fluxes).
prescribe_forcings.py imports these, mirroring the Fortran case dispatch's `use mpace_a`.

The helper `_mpace_time_select` (time_dependent_input.F90:time_select) is kept here as the mpace_a-local copy
the case uses; the vertical remap routes through the shared interpolation.F90:zlinterp_fnc mirror.

Pure-numpy / tracer-transparent → bit-identical and differentiable.
"""

from __future__ import annotations

import os

import numpy as np

from clubb_jax.src.CLUBB_core.tracer_numpy import _iset  # trace-safe in-place assignment

from clubb_jax.src.CLUBB_core.interpolation import zlinterp_fnc, linear_interp_factor

_MPACE_A_NTIMES = 139
_MPACE_A_NLEVELS = 38


def _read_mpace_dat(path: str) -> np.ndarray:
    """Read a mpace_a *.dat file as a flat array of floats (whitespace-delimited, the per_line=5
    formatting is irrelevant since values are read sequentially; file_functions.F90:file_read_1d/2d)."""
    with open(path) as fh:
        return np.array([float(tok) for tok in fh.read().split()], dtype=np.float64)


def mpace_a_init(case_dir: str) -> dict:
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


def mpace_a_tndcy(state: dict, time_current: float) -> None:
    """mpace_a.F90:mpace_a_tndcy — large-scale tendencies from the custom forcing files. Subsidence was
    deliberately removed (wm=0, Michael Falk 2007); thlm/rtm forcing = (horiz+vert advection) converted;
    um_obs/vm_obs become um_ref/vm_ref for nudging (l_uv_nudge)."""
    from clubb_jax.src.CLUBB_core.constants_clubb import Rd, Cp, grav, sec_per_hr, g_per_kg  # noqa: F401
    fd = state['_forcings_data']
    gr = state['gr']; ngrdcol = state['ngrdcol']
    b, a, r = _mpace_time_select(fd['file_times'], time_current)
    zt = np.asarray(gr.zt); zh = fd['file_heights']
    p_in_Pa = np.asarray(state['p_in_Pa'], dtype=np.float64)
    p_sfc = 101000.0                                   # HARDCODED in mpace_a.F90:140 (NOT p_sfc_nl=101500)

    def col(field):                                    # time-interp then height-interp to zt, per column
        # linear_interp_factor(ratio, after, before) — the exact mpace_a.F90 form (lines 188-202)
        c = np.asarray(linear_interp_factor(r, fd[field][:, a], fd[field][:, b]))
        return np.stack([np.asarray(zlinterp_fnc(zt[i], zh, c)) for i in range(ngrdcol)], axis=0)

    dTdt, vertT = col('dTdt'), col('vertT')
    dqdt, vertq = col('dqdt'), col('vertq')
    um_g, vm_g = col('um_obs'), col('vm_obs')
    exner_fac = (p_sfc / p_in_Pa) ** (Rd / Cp)
    # _iset (not in-place [:] =): trace-safe when a forcing array is a tracer under
    # jax.grad (the generic forcing reset promotes them to jnp). Concrete: in-place.
    state['thlm_forcing'] = _iset(state['thlm_forcing'], np.s_[:], (dTdt + vertT) * exner_fac / sec_per_hr)
    state['rtm_forcing'] = _iset(state['rtm_forcing'], np.s_[:], (dqdt + vertq) / g_per_kg / sec_per_hr)
    state['wm_zt'] = _iset(state['wm_zt'], np.s_[:], 0.0)
    state['wm_zm'] = _iset(state['wm_zm'], np.s_[:], 0.0)
    if 'um_ref' in state:
        state['um_ref'] = _iset(state['um_ref'], np.s_[:], um_g)
    if 'vm_ref' in state:
        state['vm_ref'] = _iset(state['vm_ref'], np.s_[:], vm_g)


def mpace_a_sfclyr(state: dict, time_current: float, ngrdcol: int, rho_sfc) -> tuple:
    """mpace_a.F90:mpace_a_sfclyr — surface fluxes from the prescribed sensible/latent heat (time-interp);
    wpthlp_sfc = SH/(ρ·Cp), wprtp_sfc = LH/(ρ·Lv), ustar = 0.25 (fixed)."""
    from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Lv
    fd = state['_forcings_data']
    b, a, r = _mpace_time_select(fd['file_times'], time_current)
    lh = float(linear_interp_factor(r, fd['file_lh'][a], fd['file_lh'][b]))   # mpace_a.F90:352 form
    sh = float(linear_interp_factor(r, fd['file_sh'][a], fd['file_sh'][b]))
    rho_sfc = np.asarray(rho_sfc, dtype=np.float64)
    wpthlp_sfc = np.full(ngrdcol, sh) / (rho_sfc * Cp)
    wprtp_sfc = np.full(ngrdcol, lh) / (rho_sfc * Lv)
    ustar = np.full(ngrdcol, 0.25)
    return wpthlp_sfc, wprtp_sfc, ustar
