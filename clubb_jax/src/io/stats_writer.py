"""Pure-Python replacement for the Fortran stats NetCDF output infrastructure.

Mirrors stats_netcdf.F90 semantics:
  - begin_timestep(itime)  → set l_sample / l_last_sample, reset buffers
  - update(name, val)      → accumulate into buffer (ADD, not SET)
  - begin_budget(name, v)  → subtract v from buffer (budget start)
  - finalize_budget(name,v)→ add v to buffer, count sample (budget end)
  - end_timestep(t)        → average buffer by nsamples, write NetCDF record
  - finalize()             → close NetCDF file

Variable shapes in the Python/JAX codebase are (ngrdcol, nz) for profile
variables and (ngrdcol,) for surface variables.  The NetCDF dimensions in the
output file are (time, nz, col) for profiles and (time, col) for surface
variables, matching the Fortran-produced reference files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import netCDF4 as nc
    _HAS_NC = True
except ImportError:
    _HAS_NC = False


# Grid type identifiers (matching Fortran GRID_* enum)
GRID_ZT = "zt"
GRID_ZM = "zm"
GRID_SFC = "sfc"
GRID_LH_ZT = "lh_zt"
GRID_LH_SFC = "lh_sfc"
GRID_RAD_ZT = "rad_zt"
GRID_RAD_ZM = "rad_zm"

_PROFILE_GRIDS = {GRID_ZT, GRID_ZM, GRID_LH_ZT, GRID_RAD_ZT, GRID_RAD_ZM}


def _parse_registry(registry_path: str, sclr_dim: int = 0, edsclr_dim: int = 0) -> Dict:
    """Parse standard_stats.in and return {name: (grid, units, long_name)}."""
    registry = {}
    text = Path(registry_path).read_text()
    pattern = re.compile(
        r'entry\s*\(\s*\d+\s*\)\s*=\s*"([^|"]+)\|([^|"]+)\|([^|"]+)\|([^"]+)"'
    )
    for m in pattern.finditer(text):
        name = m.group(1).strip()
        grid = m.group(2).strip()
        units = m.group(3).strip()
        long_name = m.group(4).strip()
        # Skip template entries that need expansion (contain '*')
        if '*' in name:
            continue
        registry[name] = (grid, units, long_name)

    # Expand sclr entries if sclr_dim > 0
    for base in ("sclrm", "sclrm_f", "sclrp2", "sclrprtp", "sclrpthlp", "wpsclrp"):
        if base in registry:
            g, u, ln = registry.pop(base)
            for i in range(1, sclr_dim + 1):
                suf = "m" if base == "sclrm" else base[4:]  # crude
                key = f"sclr{i}{suf}"
                registry[key] = (g, u, ln)

    # Expand edsclr entries if edsclr_dim > 0
    for base in ("edsclrm", "edsclrm_f", "wpedsclrp"):
        if base in registry:
            g, u, ln = registry.pop(base)
            for i in range(1, edsclr_dim + 1):
                suf = "m" if base == "edsclrm" else base[6:]
                key = f"edsclr{i}{suf}"
                registry[key] = (g, u, ln)

    return registry


def _format_time_units(day: int, month: int, year: int, time_initial: float) -> str:
    """Produce the UDUNITS reference-time string matching Fortran format_date."""
    # Compute HH:MM:SS from time_initial in seconds from midnight
    t = int(round(time_initial))
    hh = t // 3600
    mm = (t % 3600) // 60
    ss = t % 60
    return f"seconds since {year:04d}-{month:02d}-{day:02d} {hh:02d}:{mm:02d}:{ss:02d}.0"


class StatsWriter:
    """Accumulate CLUBB statistics and write them to a NetCDF file.

    This class mirrors the Fortran stats_type runtime object and the API
    functions in stats_netcdf.F90:  begin_timestep, stats_update,
    stats_begin_budget, stats_finalize_budget, end_timestep, finalize.
    """

    def __init__(
        self,
        registry_path: str,
        output_path: str,
        nzt: int,
        nzm: int,
        ngrdcol: int,
        zt: np.ndarray,
        zm: np.ndarray,
        stats_tsamp: float,
        stats_tout: float,
        dt_main: float,
        day: int,
        month: int,
        year: int,
        time_initial: float,
        clubb_params_vals: Optional[np.ndarray] = None,
        param_names: Optional[list] = None,
        sclr_dim: int = 0,
        edsclr_dim: int = 0,
    ):
        self.nzt = nzt
        self.nzm = nzm
        self.ngrdcol = ngrdcol
        self.dt_main = dt_main

        # Sampling cadence (matching Fortran stats_init_api)
        self.stats_nsamp = max(1, round(stats_tsamp / dt_main))
        self.stats_nout = max(1, round(stats_tout / dt_main))
        self.samples_per_write = max(1, self.stats_nout // self.stats_nsamp)

        # Parse the variable registry
        self.registry = _parse_registry(registry_path, sclr_dim, edsclr_dim)

        # Per-variable accumulation buffers and sample counts
        # Shapes: (ngrdcol, nz) for profiles, (ngrdcol, 1) for sfc
        self._buffer: Dict[str, np.ndarray] = {}
        self._nsamples: Dict[str, np.ndarray] = {}
        self._in_budget: set = set()

        # Sampling flags
        self.l_sample = False
        self.l_last_sample = False
        self.enabled = True
        self._time_index = 0  # 1-based NetCDF record counter

        # Allocate buffers for all registered variables
        for name, (grid, _, _) in self.registry.items():
            if grid in _PROFILE_GRIDS:
                nz = nzt if grid in (GRID_ZT, GRID_LH_ZT, GRID_RAD_ZT) else nzm
                self._buffer[name] = np.zeros((ngrdcol, nz), dtype=np.float64)
                self._nsamples[name] = np.zeros((ngrdcol, nz), dtype=np.int32)
            else:
                self._buffer[name] = np.zeros((ngrdcol, 1), dtype=np.float64)
                self._nsamples[name] = np.zeros((ngrdcol, 1), dtype=np.int32)

        # NetCDF file setup
        self._ncid: Optional[object] = None
        self._varids: Dict[str, object] = {}
        self._time_varid: Optional[object] = None

        if _HAS_NC and output_path:
            self._open_netcdf(
                output_path, zt, zm, day, month, year, time_initial,
                clubb_params_vals, param_names,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_timestep(self, itime: int) -> Tuple[bool, bool]:
        """Determine sampling flags and reset accumulators for itime (1-based).

        Mirrors stats_begin_timestep_api in stats_netcdf.F90, which the Fortran
        driver calls with the 1-based loop index ``do itime = 1, ifinal``. Callers
        must therefore pass a 1-based step index. Returns (l_sample, l_last_sample).
        """
        if not self.enabled:
            self.l_sample = False
            self.l_last_sample = False
            return False, False

        # Fortran: mod(itime, stats_nsamp) == 0
        self.l_sample = (itime % self.stats_nsamp == 0)
        # Fortran: mod(itime, stats_nout) == 0
        self.l_last_sample = (itime % self.stats_nout == 0)

        # Reset buffers at the start of each output window (Fortran: mod(itime-1, nout) == 0).
        if (itime - 1) % self.stats_nout == 0:
            for name in self._buffer:
                self._buffer[name][:] = 0.0
                self._nsamples[name][:] = 0
            self._in_budget.clear()

        return self.l_sample, self.l_last_sample

    def update(self, name: str, value: np.ndarray) -> None:
        """Accumulate a sample for the named variable (mirrors stats_update_2d).

        value shape: (ngrdcol, nz) for profile vars, (ngrdcol,) or scalar for sfc.
        """
        if not self.l_sample:
            return
        if name not in self._buffer:
            return  # variable not in registry — silently ignore
        val = np.asarray(value, dtype=np.float64)

        buf = self._buffer[name]
        cnt = self._nsamples[name]

        if buf.ndim == 2 and buf.shape[1] > 1:
            # Profile variable: value is (ngrdcol, nz)
            buf[:, :] += val
            cnt[:, :] += 1
        else:
            # Surface variable: value is (ngrdcol,) or scalar
            if val.ndim == 0:
                buf[:, 0] += float(val)
                cnt[:, 0] += 1
            elif val.ndim == 1:
                buf[:, 0] += val
                cnt[:, 0] += 1
            else:
                buf[:, 0] += val[:, 0]
                cnt[:, 0] += 1

    def update_col(self, name: str, value: float, icol: int = 0) -> None:
        """Update a single surface value for column icol (mirrors stats_update_scalar)."""
        if not self.l_sample:
            return
        if name not in self._buffer:
            return
        self._buffer[name][icol, 0] += float(value)
        self._nsamples[name][icol, 0] += 1

    def begin_budget(self, name: str, val_before: np.ndarray) -> None:
        """Start a budget window by subtracting the current state from the buffer.

        Mirrors stats_begin_budget in stats_netcdf.F90: a second begin without an
        intervening finalize is a no-op (the Fortran warns and returns).
        """
        if not self.l_sample:
            return
        if name not in self._buffer:
            return
        if name in self._in_budget:      # "stats budget begin twice" → no-op
            return
        val = np.asarray(val_before, dtype=np.float64)
        buf = self._buffer[name]
        if buf.ndim == 2 and buf.shape[1] > 1:
            buf[:, :] -= val
        else:
            if val.ndim <= 1:
                buf[:, 0] -= val.ravel()[0] if val.ndim == 0 else val
            else:
                buf[:, 0] -= val[:, 0]
        self._in_budget.add(name)

    def update_budget(self, name: str, value: np.ndarray) -> None:
        """Add an intermediate budget contribution (mirrors stats_update_budget).

        No-op unless the variable is inside an active begin/finalize window, and
        never increments the sample counter — exactly the Fortran semantics.
        """
        if not self.l_sample:
            return
        if name not in self._buffer:
            return
        if name not in self._in_budget:  # "stats budget update without begin" → no-op
            return
        val = np.asarray(value, dtype=np.float64)
        buf = self._buffer[name]
        if buf.ndim == 2 and buf.shape[1] > 1:
            buf[:, :] += val
        else:
            if val.ndim <= 1:
                buf[:, 0] += val.ravel()[0] if val.ndim == 0 else val
            else:
                buf[:, 0] += val[:, 0]

    def finalize_budget(self, name: str, val_after: np.ndarray,
                        l_count_sample: bool = True) -> None:
        """Close a budget window by adding the final state and optionally counting the sample.

        l_count_sample=False mirrors Fortran's stats_finalize_budget(..., l_count_sample=.false.)
        which adds to the buffer but does NOT increment the sample counter. A finalize without a
        matching begin is a no-op (the Fortran warns and returns).
        """
        if not self.l_sample:
            return
        if name not in self._buffer:
            return
        if name not in self._in_budget:  # "stats budget finalize without begin" → no-op
            return
        val = np.asarray(val_after, dtype=np.float64)
        buf = self._buffer[name]
        cnt = self._nsamples[name]
        if buf.ndim == 2 and buf.shape[1] > 1:
            buf[:, :] += val
            if l_count_sample:
                cnt[:, :] += 1
        else:
            if val.ndim <= 1:
                buf[:, 0] += val.ravel()[0] if val.ndim == 0 else val
            else:
                buf[:, 0] += val[:, 0]
            if l_count_sample:
                cnt[:, 0] += 1
        self._in_budget.discard(name)

    def end_timestep(self, t_out: float) -> None:
        """Average accumulators by sample count and write one NetCDF record."""
        if not self.l_last_sample:
            return
        if not self.enabled:
            return
        self._time_index += 1
        if self._ncid is not None and self._time_varid is not None:
            self._time_varid[self._time_index - 1] = t_out

        if self._ncid is None:
            return

        for name, buf in self._buffer.items():
            cnt = self._nsamples[name]
            if np.all(cnt <= 0):
                continue  # no samples — leave as _FillValue (0.0)
            grid = self.registry[name][0]
            # Compute average where sampled
            avg = np.where(cnt > 0, buf / np.maximum(cnt, 1), 0.0)
            varid = self._varids.get(name)
            if varid is None:
                continue
            t = self._time_index - 1  # 0-based index
            if grid in _PROFILE_GRIDS:
                # avg shape: (ngrdcol, nz) → write as (nz, ngrdcol) = NetCDF (zt/zm, col)
                varid[t, :, :] = avg.T
            else:
                # sfc: avg shape (ngrdcol, 1) → write as (ngrdcol,)
                varid[t, :] = avg[:, 0]

        # Periodically flush dirty HDF5 chunks to disk so the cache does not grow
        # unbounded over a long run. Without this, a per-step-stats run of a case with
        # a large variable set (e.g. Morrison: ~500 vars × many levels) accumulates
        # hundreds of records' worth of un-flushed chunks and OOMs the process mid-write
        # (→ a truncated/corrupt NetCDF). Every-20-records bounds the dirty cache while
        # keeping the sync I/O negligible next to the physics. (Iter322 — mpace_a OOM.)
        if self._time_index % 20 == 0:
            self._ncid.sync()

        # Reset flags
        self.l_sample = False
        self.l_last_sample = False

    def var_on_stats_list(self, name: str) -> bool:
        """Return True if name is in the stats registry (mirrors var_on_stats_list)."""
        return name in self.registry

    def get_stats_config(self):
        """Return a config array with [enabled, ..., l_sample, l_last_sample]."""
        cfg = [int(self.enabled)] + [0] * 6 + [int(self.l_sample), int(self.l_last_sample)]
        return cfg

    def finalize(self) -> None:
        """Flush and close the NetCDF file."""
        if self._ncid is not None:
            try:
                self._ncid.close()
            except Exception:
                pass
            self._ncid = None

    # ------------------------------------------------------------------
    # NetCDF helpers
    # ------------------------------------------------------------------

    def _open_netcdf(
        self,
        output_path: str,
        zt: np.ndarray,
        zm: np.ndarray,
        day: int,
        month: int,
        year: int,
        time_initial: float,
        clubb_params_vals: Optional[np.ndarray],
        param_names: Optional[list],
    ) -> None:
        """Create and define the output NetCDF file (mirrors stats_open_netcdf)."""
        ds = nc.Dataset(output_path, "w", format="NETCDF4_CLASSIC")
        self._ncid = ds

        # Dimensions
        ds.createDimension("time", None)  # unlimited
        ds.createDimension("col", self.ngrdcol)
        ds.createDimension("zt", self.nzt)
        ds.createDimension("zm", self.nzm)

        # We always create lh_zt with the same size as zt for compatibility
        ds.createDimension("lh_zt", self.nzt)

        # Coordinate variables
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = _format_time_units(day, month, year, time_initial)
        self._time_varid = tv

        cv = ds.createVariable("col", "f8", ("col",))
        cv.units = "count"
        cv[:] = np.arange(1, self.ngrdcol + 1, dtype=np.float64)

        zmv = ds.createVariable("zm", "f8", ("zm",))
        zmv.units = "m"
        zmv[:] = np.asarray(zm).ravel()[:self.nzm]

        ztv = ds.createVariable("zt", "f8", ("zt",))
        ztv.units = "m"
        ztv[:] = np.asarray(zt).ravel()[:self.nzt]

        lhv = ds.createVariable("lh_zt", "f8", ("lh_zt",))
        lhv.units = "m"
        lhv[:] = np.asarray(zt).ravel()[:self.nzt]

        # clubb_params metadata
        if clubb_params_vals is not None and param_names is not None:
            nparam = len(param_names)
            ds.createDimension("param", nparam)
            pstrlen = 28
            ds.createDimension("param_strlen", pstrlen)

            pv = ds.createVariable("param", "f8", ("param",))
            pv.units = "count"
            pv[:] = np.arange(1, nparam + 1, dtype=np.float64)

            pnv = ds.createVariable("param_name", "S1", ("param", "param_strlen"))
            for i, pn in enumerate(param_names):
                padded = pn.ljust(pstrlen)[:pstrlen]
                pnv[i, :] = np.array(list(padded), dtype="S1")

            cpv = ds.createVariable("clubb_params", "f8", ("param", "col"))
            cpv.long_name = "clubb_params"
            vals = np.asarray(clubb_params_vals, dtype=np.float64)
            # vals shape: (ngrdcol, nparam) → transpose to (nparam, ngrdcol)
            cpv[:, :] = vals.T

        # All registered stats variables
        grid_to_dim = {
            GRID_ZT: ("time", "zt", "col"),
            GRID_ZM: ("time", "zm", "col"),
            GRID_LH_ZT: ("time", "lh_zt", "col"),
            GRID_SFC: ("time", "col"),
            GRID_LH_SFC: ("time", "col"),
            GRID_RAD_ZT: ("time", "zt", "col"),
            GRID_RAD_ZM: ("time", "zm", "col"),
        }
        for name, (grid, units, long_name) in self.registry.items():
            dims = grid_to_dim.get(grid)
            if dims is None:
                continue
            v = ds.createVariable(name, "f8", dims, fill_value=0.0)
            v.long_name = long_name
            v.units = units
            self._varids[name] = v
