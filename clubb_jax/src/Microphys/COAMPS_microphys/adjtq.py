"""JAX port of COAMPS_microphys/adjtq.F — the COAMPS master microphysics routine.

============================  BOOTSTRAP STUB  ============================
`adjtq.F` (1609 lines) is the heart of the COAMPS Rutledge & Hobbs bulk scheme:
it advances cloud water/ice/rain/snow/graupel and their number concentrations
through ~40 process-rate equations over one timestep, in a two-pass structure
(warm/collection pass, then the ice deposition/nucleation pass, then autoconversion
and accretion), with an internal saturation adjustment (`qtadj`) between passes.

This module is a **documented no-op passthrough**: it returns the input hydrometeor
fields UNCHANGED and zero fall speeds / process rates. That makes the COAMPS driver
(coamps_microphys_driver_module.py) run end-to-end and exercises all the CLUBB<->COAMPS
wiring (field setup, gamma constants, k-flip, saturation, in-cloud detection, the
tendency = (field_after - field_before)/dt loop). With this stub every *_mc tendency
is 0 — i.e. COAMPS currently applies NO microphysics. Porting adjtq's internals is the
remaining bulk of the work; see COAMPS_PORT.md for the file-by-file checklist.

------------------------------------------------------------------------
adjtq CALL GRAPH (subroutines it invokes, in source order) — the port targets:

  Saturation / adjustment leaves (PORTED as leaf utilities):
    slope    slope.py       inverse-slope factors for rain/snow/graupel
    esatv    esatv.py       sat vapour pressure over water  (via esat_new)
    esati    esati.py       sat vapour pressure over ice    (via esat_new)
    qsatvi   qsatvi.py      sat mixing ratios over water & ice
    gamma    gamma.py       (called from the driver, not adjtq)

  Fall-speed leaves (NOT yet ported):
    tgqr  tgqs  tgqg  tgqi   terminal fall speeds for rain/snow/graupel/ice

  Ice initiation / conversion (NOT yet ported):
    frzh     homogeneous freezing of cloud water (pchomo)
    conice   ice nucleation number (Fletcher/Meyers/Cooper, icon scheme)
    adjmlt   melting adjustment (delta1/2/3 partition)
    qtadj    saturation adjustment of T & qv to qadjw
    nrmcol   number-concentration collection bookkeeping
    nrmtqw   warm-pass number/mass adjustment
    nrmtqi   ice-pass number/mass adjustment

  Process-rate equations eqa*.F (NOT yet ported) — each computes one microphysical
  conversion rate; the `*g` suffix files are the graupel-on variants (l_graupel):
    eqa6  (pcond)  eqa7 (praut)  eqa9 (pracw)  eqa12 (prevp)  eqa15 (pint)
    eqa18 (pdepi)  eqa19 (pconv) eqa21 (psaci) eqa22 (psacw)  eqa25 (psmlt)
    eqa26 (psdep)  eqa27 (pmltse) eqa27r (piacw) eqa28 (psmlti)
    eqa5g eqa7g eqa8g(psacr) eqa9g(pracs) eqa10g eqa11g eqa12g eqa13g eqa14g
    eqa17g eqa18g eqa19g eqa20g eqa21g eqa22g   (graupel variants)
------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings

_WARNED = False


def adjtq_stub(fields: dict) -> dict:
    """No-op passthrough (see module docstring). Returns hydrometeor fields unchanged
    and zero fall speeds. `fields` carries qc3/qi3/qr3/qg3/qs3/qv3/th3/nc3/nr3/ncn3/ni3
    and cond; the returned dict has the same keys plus zeroed fall speeds + snowslope."""
    global _WARNED
    if not _WARNED:
        warnings.warn(
            "COAMPS adjtq is a no-op stub: microphysics tendencies will be 0. "
            "See clubb_jax/src/Microphys/COAMPS_microphys/adjtq.py and COAMPS_PORT.md.",
            RuntimeWarning, stacklevel=2)
        _WARNED = True
    out = dict(fields)   # pass every field through unchanged
    return out
