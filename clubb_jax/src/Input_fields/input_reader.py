"""JAX port of input_reader.F90 — the blank-fill helpers for the case input readers.

Mirrors clubb_release/src/Input_fields/input_reader.F90: the routines that fill the `-999.9` blank sentinels in
the time-/height-dependent forcing tables by linear interpolation before the table is regridded —
`linear_fill_blanks` (input_reader.F90:437) and `fill_blanks_two_dim_vars` (input_reader.F90:368). The
generic-forcings file reader (`prescribe_forcings._parse_forcings_file`) imports these, mirroring the Fortran
`use input_reader` in the case forcing readers. (The Fortran `read_x_table`/`read_x_profile` Fortran-file parsers
are I/O infrastructure the JAX replaces with its own `_parse_*_file` readers; only the blank-fill numerics mirror
here.)

Pure-numpy → bit-identical.
"""

from __future__ import annotations

import numpy as np

_BLANK = -999.9   # the missing-value sentinel used in the *_forcings.in / *_sfc.in tables


def linear_fill_blanks(grid: np.ndarray, values: np.ndarray,
                       blank: float = _BLANK) -> np.ndarray:
    """Fill blank sentinels in `values` by linear interpolation on `grid` (input_reader.F90:linear_fill_blanks).

    Valid = values > blank. If none valid: return all-blank. If all valid: return unchanged. Edges are held
    flat (np.interp left/right = first/last valid value).
    """
    valid = values > blank
    if not np.any(valid):
        return np.full_like(values, blank)
    if np.all(valid):
        return values.copy()
    vg = grid[valid]
    vv = values[valid]
    return np.interp(grid, vg, vv, left=float(vv[0]), right=float(vv[-1]))


def fill_blanks_two_dim_vars(z_grid: np.ndarray, time_grid: np.ndarray,
                             arr: np.ndarray, blank: float = _BLANK) -> np.ndarray:
    """Fill blanks in a 2-D (nz, ntimes) table, first along z then along time
    (input_reader.F90:fill_blanks_two_dim_vars). Returns a filled copy.
    """
    out = arr.copy()
    nz, nt = out.shape
    for it in range(nt):
        out[:, it] = linear_fill_blanks(z_grid, out[:, it], blank)
    for iz in range(nz):
        out[iz, :] = linear_fill_blanks(time_grid, out[iz, :], blank)
    return out
