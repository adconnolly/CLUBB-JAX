#!/usr/bin/env python3
"""test_pressure_coord_forcing.py — validate the pressure-vertical-coordinate forcing path.

CGILS / cloud-feedback forcing files use a `Press[Pa]` vertical coordinate (not `z[m]`). The parser must
interpolate the forcing against the model pressure profile, not height. Tested here against a literal
np.interp-in-pressure reference; a height-coordinate file must be unaffected (interpolated against zt).
"""
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from clubb_jax.src.Benchmark_cases.time_dependent_input import _parse_forcings_file


def _write(tmp, header, blocks):
    """blocks = list of (time, [(coord, [vals...]), ...])."""
    with open(tmp, "w") as fh:
        fh.write("! comment\n")
        fh.write(header + "\n")
        for t, rows in blocks:
            fh.write(f"{t} {len(rows)}\n")
            for coord, vals in rows:
                fh.write(f"{coord} " + " ".join(str(v) for v in vals) + "\n")


def test_pressure_coordinate_interp():
    # Two forcing levels at pressures 90000 and 50000 Pa; one column 'thlm_f', two times.
    p_lev = [90000.0, 50000.0]
    f_t0 = [1.0e-4, 3.0e-4]
    f_t1 = [2.0e-4, 4.0e-4]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "f.in")
        _write(path, "Press[Pa] 'thlm_f[K/s]'",
               [(0.0, [(p_lev[0], [f_t0[0]]), (p_lev[1], [f_t0[1]])]),
                (3600.0, [(p_lev[0], [f_t1[0]]), (p_lev[1], [f_t1[1]])])])
        # Model grid: arbitrary heights, with a pressure profile p_in_Pa decreasing with height.
        zt = np.linspace(100.0, 12000.0, 25)
        p_model = np.linspace(95000.0, 40000.0, 25)   # descending pressure with height
        out = _parse_forcings_file(path, zt, p_in_Pa=p_model)
        got = out['interp']['thlm_f[K/s]'] if 'interp' in out else None
        # The parser returns the dict via _col-style keys; pull the raw interpolated array.
        # (Use the returned 'thlm_f' convenience key if present, else reconstruct.)
        arr = out.get('thlm_f')
        assert arr is not None, f"thlm_f not in parsed output keys={list(out)}"
        # Literal reference: interpolate in PRESSURE (source ascending pressure p_lev sorted), ZERO-FILLED
        # outside the forcing's range (left=right=0) — matching the Fortran reader's `zlinterp_fnc` (Iter97 fix;
        # was edge-extrapolated, which over-applied the forcing at out-of-range levels — the cloud_feedback bug).
        src_p = np.array(p_lev)             # [90000, 50000] -> needs sorting ascending for np.interp
        order = np.argsort(src_p)
        for it, fvals in enumerate((f_t0, f_t1)):
            ref = np.interp(p_model, src_p[order], np.array(fvals)[order], left=0.0, right=0.0)
            assert np.max(np.abs(arr[:, it] - ref)) < 1e-15, f"pressure interp time {it}"
    print("  pressure-coordinate forcing: interpolated against p_in_Pa (vs literal np.interp in pressure)  PASS")


def test_height_coordinate_unchanged():
    # Same data but a z[m] coordinate -> must interpolate against zt, ignoring p_in_Pa.
    z_lev = [500.0, 8000.0]
    fv = [1.0e-4, 3.0e-4]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "f.in")
        _write(path, "z[m] 'thlm_f[K/s]'", [(0.0, [(z_lev[0], [fv[0]]), (z_lev[1], [fv[1]])])])
        zt = np.linspace(100.0, 12000.0, 25)
        p_model = np.linspace(95000.0, 40000.0, 25)
        out_h = _parse_forcings_file(path, zt, p_in_Pa=p_model)       # p_in_Pa present but coord is height
        out_none = _parse_forcings_file(path, zt, p_in_Pa=None)       # no p_in_Pa
        ref = np.interp(zt, np.array(z_lev), np.array(fv), left=0.0, right=0.0)  # zero-fill out of range (Iter97)
        assert np.max(np.abs(out_h['thlm_f'][:, 0] - ref)) < 1e-15, "height interp wrong"
        # p_in_Pa must NOT affect a height-coordinate file (byte-identical to the None call).
        assert np.array_equal(out_h['thlm_f'], out_none['thlm_f']), "p_in_Pa leaked into height path"
    print("  height-coordinate forcing: interpolated against zt, p_in_Pa ignored (byte-identical)  PASS")


def test_T_f_and_um_ref_exposed():
    """The parser must expose the T_f (absolute-T forcing) and um_ref/vm_ref (nudging) columns so the apply
    step can use them — they were previously dropped (T_f never reached the apply step)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "f.in")
        _write(path, "Press[Pa] 'T_f[K/s]' 'um_ref[m/s]' 'vm_ref[m/s]'",
               [(0.0, [(90000.0, [1.0e-4, -7.0, -2.0]), (50000.0, [2.0e-4, -10.0, -3.0])])])
        zt = np.linspace(100.0, 12000.0, 20)
        p_model = np.linspace(95000.0, 40000.0, 20)
        out = _parse_forcings_file(path, zt, p_in_Pa=p_model)
        assert out.get('T_f') is not None, "T_f column not exposed by the parser"
        assert out.get('um_ref') is not None and out.get('vm_ref') is not None, "um_ref/vm_ref not exposed"
        # T_f interpolated in pressure (non-blank, finite, right shape).
        assert out['T_f'].shape == (20, 1) and np.all(np.isfinite(out['T_f'])), "T_f shape/values bad"
        # An all-blank column must still be dropped (None).
        path2 = os.path.join(d, "f2.in")
        _write(path2, "Press[Pa] 'T_f[K/s]' 'um_ref[m/s]'",
               [(0.0, [(90000.0, [1.0e-4, -999.9]), (50000.0, [2.0e-4, -999.9])])])
        out2 = _parse_forcings_file(path2, zt, p_in_Pa=p_model)
        assert out2.get('T_f') is not None and out2.get('um_ref') is None, "blank um_ref should be None"
    print("  parser exposes T_f + um_ref/vm_ref columns (and drops all-blank ones)  PASS")


def test_apply_T_f_conversion():
    """The apply step must convert an absolute-T forcing to a thlm forcing: thlm_forcing = T_f/exner
    (time_dependent_input.F90:671). This is the only forcing branch no gated case exercises."""
    from clubb_jax.src.Benchmark_cases.time_dependent_input import apply_time_dependent_forcings
    nzt = 8
    # T_f profile, two times (constant in time here for a clean check).
    T_f = np.tile(np.linspace(-2e-4, -1e-4, nzt)[:, None], (1, 2))   # (nzt, ntimes)
    exner = np.linspace(1.0, 0.6, nzt)[None, :]                       # (1, nzt)
    fd = {k: None for k in ('thlm_f', 'T_f', 'rtm_f', 'sp_hmdty_f', 'w', 'omega', 'um_f', 'vm_f',
                            'ug', 'vg', 'um_ref', 'vm_ref')}
    fd.update({'times': np.array([0.0, 3600.0]), 'T_f': T_f, 'omega_mb_hr': False})
    state = {'ngrdcol': 1, '_forcings_data': fd,
             'thlm_forcing': np.zeros((1, nzt)), 'rtm_forcing': np.zeros((1, nzt)), 'exner': exner}
    apply_time_dependent_forcings(state, 1800.0)
    expect = T_f[:, 0][None, :] / exner
    expect[:, -1] = 0.0   # Fortran zeroes the top
    assert np.max(np.abs(state['thlm_forcing'] - expect)) < 1e-15, "T_f -> thlm_forcing = T_f/exner failed"
    print("  apply step: T_f absolute-T forcing -> thlm_forcing = T_f/exner (apply branch)  PASS")


def main():
    print("test_pressure_coord_forcing:")
    for t in (test_pressure_coordinate_interp, test_height_coordinate_unchanged, test_T_f_and_um_ref_exposed,
              test_apply_T_f_conversion):
        t()
    print("All pressure-coord forcing checks PASSED")


if __name__ == "__main__":
    main()
