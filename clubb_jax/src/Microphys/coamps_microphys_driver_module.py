"""JAX port of coamps_microphys_driver_module.F90 — CLUBB<->COAMPS microphysics wrapper.

Mirrors `subroutine coamps_microphys_driver`. This wraps the COAMPS (NRL Rutledge &
Hobbs bulk) scheme so CLUBB can call it: it sets up the COAMPS field layout, computes
the COAMPS gamma-function constants and slope intercepts, detects the saturated /
in-cloud levels, calls the master routine `adjtq` (currently a no-op stub — see
adjtq.py), and converts the before/after fields into the CLUBB-form `*_mc` tendencies.

BOOTSTRAP status: the full top-level flow is ported and runs; because `adjtq` is a
stub, all returned tendencies are 0 (no microphysics applied). See COAMPS_PORT.md.

Conventions kept from the Fortran:
- CLUBB k=1 is the surface; COAMPS assumes k=1 is the domain top, so the driver builds
  "_flip" (top-down) copies before calling adjtq and un-flips after. `adjtq` does no
  advection/sedimentation, so the flip is a no-op for the stub but is preserved for
  when the process rates are ported.
- Number concentrations are converted MKS <-> CGS at the COAMPS boundary
  (nc3 = Ncm / cm3_per_m3, ni3 = Nim * rho, etc.).

This driver operates on a SINGLE column (1-D arrays of length nzt / nzm), mirroring the
Fortran which is called per column `icol`. The per-column loop lives in
coamps_microphys_step.py.
"""

from __future__ import annotations

import numpy as np

from clubb_jax.src.CLUBB_core.tracer_numpy import _xp, _is_tracer_arg
from clubb_jax.src.CLUBB_core.constants_clubb import (
    Cp, Lv, Lf, Ls, Rv, Rd, p0, T_freeze_K, cm3_per_m3,
)

pi = float(np.pi)
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_liq, sat_mixrat_ice
from clubb_jax.src.CLUBB_core.T_in_K_module import thlm2T_in_K
from clubb_jax.src.Microphys.COAMPS_microphys.gamma import gamma
from clubb_jax.src.Microphys.COAMPS_microphys.adjtq import adjtq_stub

# ── COAMPS local constants (coamps_microphys_driver_module.F90:90-135) ──────────
_BSNOW = 0.11
_BGRP = 0.66
_RHOLIQ = 1000.0
_RHOSNO = 100.0
_RHOGRP = 400.0
_RNZERO = 8.0e6
_GNZERO = 4.0e6


def coamps_microphys_driver(runtype, timea, deltf,
                            rtm, wm_zm, p_in_Pa, exner, rho,
                            thlm, rim, rrm, rgm, rsm,
                            rcm, Ncm, Nrm, Nim,
                            saturation_formula,
                            Nccnm,
                            l_ice_microphys=True, l_graupel=False):
    """Compute the COAMPS microphysics tendencies for one column.

    Inputs are 1-D arrays over the nzt thermodynamic levels (wm_zm is nzm). Returns a
    dict of CLUBB-form tendencies + sedimentation velocities:
      {ritend, rrtend, rgtend, rstend, nrmtend, ncmtend, nimtend,
       rvm_mc, rcm_mc, thlm_mc, Vrr, VNr, Vrs, Vri, Vrg, Nccnm}
    """
    xp = _xp
    under_trace = _is_tracer_arg([rtm, thlm, rcm])
    zeros = xp.zeros_like(rrm)

    # ── icase / snzero (F90:473-527). Only affects adjtq internals + snzero. ──
    ldrizzle = False
    if l_ice_microphys and ldrizzle:
        raise ValueError("l_ice_microphys must be false to use ldrizzle")
    snzero = 2.0e6 if runtype == "mpace_a" else 2.0e7

    # ── vapour, thm, temperature (F90:532-548) ──
    rvm = xp.maximum(rtm - rcm, 0.0)
    thm = thlm + (Lv / (Cp * exner)) * rcm
    T_in_K = thlm2T_in_K(thlm, exner, rcm)

    # ── saturation mixing ratios on the mass grid (F90:571-582) ──
    qsatv = sat_mixrat_liq(p_in_Pa, T_in_K, int(saturation_formula))
    qsati = sat_mixrat_ice(p_in_Pa, T_in_K)

    # ── COAMPS gamma-function constants (F90:615-633) ──
    gm3 = gamma(3.0); gm4 = gamma(4.0); gm5 = gamma(5.0); gm6 = gamma(6.0)
    gm7 = gamma(7.0); gm8 = gamma(8.0); gm9 = gamma(9.0)
    gmbp3 = gamma(_BSNOW + 3.0)
    gmbov2 = gamma(_BSNOW * 0.5 + 2.5)
    gmbov2g = gamma(_BGRP * 0.5 + 2.5)
    # (ex2..ex7g / sloper..slopeg are consumed inside adjtq; see COAMPS_PORT.md.)
    sloper = pi * _RHOLIQ * _RNZERO * 1.0e-8
    slopes = pi * _RHOSNO * snzero * 1.0e-8
    slopeg = pi * _RHOGRP * _GNZERO * 1.0e-8
    _gamma_consts = (gm3, gm4, gm5, gm6, gm7, gm8, gm9, gmbp3, gmbov2, gmbov2g,
                     sloper, slopes, slopeg)  # noqa: F841 (for the ported adjtq)

    # ── in-cloud / saturated-level detection (F90:678-705) ──
    # sat = qv/qsat - 1 (over water above freezing, over ice below when l_ice).
    if l_ice_microphys:
        sat = xp.where(T_in_K >= T_freeze_K, rvm / qsatv - 1.0, rvm / qsati - 1.0)
    else:
        sat = rvm / qsatv - 1.0
    pcut = 1.0e-10
    in_cloud = (sat > 0.0) | (rcm >= pcut) | (rrm >= pcut) | \
               (rsm >= pcut) | (rim >= pcut) | (rgm >= pcut)
    len_saturated = int(np.count_nonzero(np.asarray(in_cloud))) if not under_trace else 1

    # ── build the COAMPS field set (post-conversion) and call adjtq ──
    # nc3/nr3/ncn3 in (m^3/cm^3)*kg^-1; ni3 in #/m^3 (F90:596-610).
    fields = {
        'qc3': rcm, 'qi3': rim, 'qr3': rrm, 'qg3': rgm, 'qs3': rsm,
        'qv3': rvm, 'th3': thm, 'exbm': exner, 'rbm': rho,
        'nc3': Ncm / cm3_per_m3, 'nr3': Nrm / cm3_per_m3,
        'ncn3': Nccnm / cm3_per_m3, 'ni3': Nim * rho,
        'cond': xp.zeros_like(rcm),
    }
    # adjtq is a no-op stub: fields pass through unchanged (len>0 gate mirrored).
    if len_saturated > 0:
        fields = adjtq_stub(fields)

    # ── un-convert numbers and compute tendencies (F90:826-906) ──
    qc3, qi3, qr3, qg3, qs3 = fields['qc3'], fields['qi3'], fields['qr3'], fields['qg3'], fields['qs3']
    qv3, th3, exbm = fields['qv3'], fields['th3'], fields['exbm']
    nc3, nr3, ni3 = fields['nc3'], fields['nr3'], fields['ni3']

    dt = float(deltf)
    rrtend = (qr3 - rrm) / dt
    rgtend = (qg3 - rgm) / dt
    ritend = (qi3 - rim) / dt
    rstend = (qs3 - rsm) / dt
    nrmtend = (nr3 * cm3_per_m3 - Nrm) / dt
    ncmtend = (nc3 * cm3_per_m3 - Ncm) / dt
    nimtend = (ni3 / rho - Nim) / dt
    rvm_mc = (qv3 - rvm) / dt
    rcm_mc = (qc3 - rcm) / dt
    thlm_mc = ((th3 - (Lv / (Cp * exbm)) * qc3) - thlm) / dt

    return {
        'ritend': ritend, 'rrtend': rrtend, 'rgtend': rgtend, 'rstend': rstend,
        'nrmtend': nrmtend, 'ncmtend': ncmtend, 'nimtend': nimtend,
        'rvm_mc': rvm_mc, 'rcm_mc': rcm_mc, 'thlm_mc': thlm_mc,
        # sedimentation velocities (zero for the stub; adjtq fills fallr/n/s/i/g)
        'Vrr': zeros, 'VNr': zeros, 'Vrs': zeros, 'Vri': zeros, 'Vrg': zeros,
        'Nccnm': fields['ncn3'] * cm3_per_m3,
    }
