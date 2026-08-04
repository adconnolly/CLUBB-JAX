"""Per-step COAMPS microphysics wiring for the JAX CLUBB loop.

Mirrors the Fortran call sequence (microphys_driver.F90 -> coamps_microphys_driver):
each step builds the per-column COAMPS inputs from the post-advance state, calls the
ported `coamps_microphys_driver` (currently backed by the adjtq no-op stub, so the
tendencies are 0), stores the mean-field microphysics tendencies for the next step's
rtm/thlm forcings, and advances the hydrometeor fields by the (currently zero) *_mc.

BOOTSTRAP: the mean tendencies are stored under the SAME `_morr_*_mc` state keys the
generic driver already consumes (advance_clubb_to_end.py:138), so no shared-driver edit
is needed and the (zero) COAMPS tendencies flow into the forcings exactly like Morrison.
See COAMPS_PORT.md.
"""

from __future__ import annotations

import numpy as np

from clubb_jax.src.Microphys.coamps_microphys_driver_module import coamps_microphys_driver


def advance_coamps_microphysics(state: dict):
    """Compute the COAMPS microphysics tendencies from the post-advance state and store them."""
    hmm = state['hm_metadata']
    cfg = state.get('cfg', {})
    l_ice = bool(cfg.get('l_ice_microphys', True))
    l_grpl = bool(cfg.get('l_graupel', False))

    g = lambda k: np.asarray(state[k], np.float64)
    hydromet = g('hydromet')                       # (ngrdcol, nzt, hydromet_dim)
    iirr, iiNr, iiri, iiNi = int(hmm.iirr), int(hmm.iiNr), int(hmm.iiri), int(hmm.iiNi)
    iirs, iirg = int(hmm.iirs), int(hmm.iirg)

    rtm = g('rtm'); rcm = g('rcm'); thlm = g('thlm')
    exner = g('exner'); rho = g('rho'); pres = g('p_in_Pa')
    wm_zm = g('wm_zm')
    ngrdcol, nzt = rtm.shape
    # grid-mean droplet number diagnosed as Nc_in_cloud * cloud_frac (mirrors Morrison).
    Ncm = g('Nc_in_cloud') * g('cloud_frac')
    dt = float(state['dt_main'])
    sat_formula = int(state['saturation_formula'])
    runtype = str(state.get('runtype', ''))

    # Cloud condensation nuclei — COAMPS carries Nccnm as inout; init to 0 if absent.
    Nccnm = np.asarray(state.get('_coamps_Nccnm', np.zeros((ngrdcol, nzt))), np.float64)

    pick = lambda i: hydromet[..., i] if i >= 0 else np.zeros((ngrdcol, nzt))

    # Accumulate the per-column tendencies (Fortran is called per column `icol`).
    out_keys = ('ritend', 'rrtend', 'rgtend', 'rstend', 'nrmtend', 'ncmtend',
                'nimtend', 'rvm_mc', 'rcm_mc', 'thlm_mc', 'Nccnm')
    acc = {k: np.zeros((ngrdcol, nzt)) for k in out_keys}
    for c in range(ngrdcol):
        res = coamps_microphys_driver(
            runtype, float(state.get('time_current', 0.0)), dt,
            rtm[c], wm_zm[c], pres[c], exner[c], rho[c],
            thlm[c], pick(iiri)[c], pick(iirr)[c], pick(iirg)[c], pick(iirs)[c],
            rcm[c], Ncm[c], pick(iiNr)[c], pick(iiNi)[c],
            sat_formula, Nccnm[c],
            l_ice_microphys=l_ice, l_graupel=l_grpl)
        for k in out_keys:
            acc[k][c] = np.asarray(res[k], np.float64)

    # store the mean-field tendencies for the next step's forcings (reuse Morrison keys).
    state['_morr_rcm_mc'] = acc['rcm_mc']
    state['_morr_thlm_mc'] = acc['thlm_mc']
    state['_morr_rvm_mc'] = acc['rvm_mc']
    state['_coamps_Nccnm'] = acc['Nccnm']

    # Advance each hydrometeor by explicit Euler with its *_mc (zero for the stub).
    new_hm = hydromet.copy()
    for idx, key in ((iirr, 'rrtend'), (iiNr, 'nrmtend'), (iiri, 'ritend'),
                     (iiNi, 'nimtend'), (iirs, 'rstend'), (iirg, 'rgtend')):
        if idx < 0:
            continue
        new_hm[..., idx] = np.maximum(hydromet[..., idx] + acc[key] * dt, 0.0)
    state['hydromet'] = new_hm
    state['Ncm'] = np.maximum(Ncm + acc['ncmtend'] * dt, 0.0)
