# Running the JAX driver without f2py / `clubb_release`

## TL;DR

The **physics** runs 100 % in JAX — zero Fortran calls per timestep. f2py is still required at runtime for one
reason: the active standalone path delegates the **statistics output subsystem** (and a couple of smaller things)
to the compiled Fortran through `clubb_python.clubb_api`. Importing `clubb_python` eagerly imports the
`clubb_f2py` `.so`, so the whole f2py build is pulled in even though the closure math never touches it.

A pure-Python replacement for that subsystem **already exists but is not wired in**
(`clubb_jax/src/io/stats_writer.py::StatsWriter`). Finishing the decoupling is mostly a wiring job, plus closing one
metadata gap and vendoring the input data files.

There are **two independent dependencies** on `clubb_release`, and they must be addressed separately:

1. **The compiled f2py runtime** (`clubb_f2py.so` via `clubb_python`) — the subject of "why does it need f2py".
2. **The input *data* files** (`clubb_release/input/{case_setups,tunable_parameters,stats}` + sounding files) —
   plain text the driver reads at init. Removing f2py does **not** remove this; running with no `clubb_release`
   directory at all means vendoring these too.

---

## Why f2py is required today

Every `from clubb_python ...` import triggers `clubb_python/__init__.py` → `import clubb_api` → `import clubb_f2py`
(the compiled extension). So any single touch-point below forces the f2py build to exist. The touch-points in the
**active standalone path** (`clubb_standalone.py` → `clubb_case_initalization.init_clubb_case` → `advance_clubb_to_end`):

| Area | File:line (current) | What it calls | Pure-Python replacement |
|---|---|---|---|
| Stats init | `clubb_case_initalization.py:801` | `clubb_api.init_stats(...)`, `get_stats_config()` | `StatsWriter(...)` ctor |
| Stats per-step | `advance_clubb_to_end.py:157,182,202-286` | `stats_begin_timestep`, `stats_update`, `stats_end_timestep` | `StatsWriter.begin_timestep/update/end_timestep` |
| Stats replay bridge | `CLUBB_core/jax_stats_bridge.py:151,159,386-432` | `get_stats_config`, `get_stats_var_meta`, `stats_update*`, `stats_*_budget` | flush `JaxStats` into `StatsWriter` |
| Stats finalize | `clubb_case_initalization.py:977` | `clubb_api.finalize_stats(...)` | `StatsWriter.finalize()` |
| Radiation stats | `Radiation/radiation.py:15,59,127-196` | `clubb_api.stats_update(...)` | `state['stats_writer'].update(...)` (cf. `radiation_module.py`) |
| Forcing (some cases) | `Benchmark_cases/prescribe_forcings.py:158` | lazy `from clubb_python import clubb_api` | port the case's forcing to the JAX path |
| Type converters | `derived_types/converters.py:8-15` | `from clubb_python.derived_types import ErrInfo, Grid, NuVertResDep, pdf_params, SclrIdx` | the pure-Python mirrors in `clubb_jax/src/derived_types/` |
| err_info / debug (driver path) | `clubb_driver.py:541-1229` | `init_err_info`, `set_debug_level`, `get_err_code`, `reset_err_code`, `cleanup_err_info`, `finalize_stats` | `derived_types/err_info.py` + `StatsWriter` |

The dominant one is **stats**: the Fortran `stats_*` package owns the variable registry (which variables are
enabled, their grid `zt`/`zm`, units, long-names), the per-step accumulate/average/budget machinery, and the NetCDF
writing. The JAX core computes values and replays them into that Fortran object via `JaxStats.to_api()`.

Note `clubb_driver.py` (the `run_clubb` entry) is **not** on the standalone path — the standalone uses
`clubb_case_initalization.init_clubb_case`. Its `clubb_api` calls only matter if you also want to decouple that
entry point.

---

## What is already built (merged from `formatting_and_jitting`)

- **`clubb_jax/src/io/stats_writer.py::StatsWriter`** — a complete pure-Python mirror of `stats_netcdf.F90`:
  `begin_timestep / update / update_col / begin_budget / finalize_budget / end_timestep / finalize /
  get_stats_config / var_on_stats_list`, NetCDF output via `netCDF4`. Its ctor takes essentially the same arguments
  already assembled for `clubb_api.init_stats` (`registry_path`, `output_path`, `ncol`, `stats_tsamp/tout`,
  `dt_main`, `day/month/year`, `time_initial`, `nzt/zt`, `nzm/zm`, `sclr_dim`, `edsclr_dim`). **It is never
  instantiated** in `src/` today (`clubb_driver.py:1185` even passes `stats_writer=None`).
- **Pure-Python derived types** — `clubb_jax/src/derived_types/{config_flags,grid_class,pdf_params,sclr_idx,
  nu_vert_res_dep,err_info,err_info_codes}.py`.
- **Pure-JAX init readers** — `clubb_jax/src/io/{namelist,sounding,surface,grid_file}.py` are **already used** by
  `init_clubb_case` (lines 33-41). So namelist/sounding/surface/grid reading is no longer f2py-bound; only the data
  *files* are still read from `clubb_release/input/`.
- **`Radiation/radiation_module.py`** — a radiation path that records stats through `state['stats_writer']` instead
  of `clubb_api` (the active driver still imports the older `Radiation/radiation.py`).

---

## Migration steps to drop f2py

1. **Instantiate `StatsWriter` in `init_clubb_case`.** Replace the `clubb_api.init_stats(...)` block
   (`clubb_case_initalization.py:796-827`) with `state['stats_writer'] = StatsWriter(registry_path=...,
   output_path=..., ...)` using the args already computed there. Keep `l_stats=False` runs writer-free.
2. **Route the per-step stats through the writer.** In `advance_clubb_to_end.py`, replace `clubb_api.stats_*`
   (lines 157,182,202-286) and the `JaxStats.to_api()` replay with `state['stats_writer']` calls. Cleanest: add a
   `JaxStats.to_writer(stats_writer)` next to `to_api()` (`jax_stats_bridge.py:386`) that calls
   `update / begin_budget / finalize_budget` instead of the `clubb_api.stats_*` equivalents (the method names map
   1:1; only the budget names differ — `stats_begin_budget`→`begin_budget`, `stats_finalize_budget`→
   `finalize_budget`).
3. **Build stats metadata without `get_stats_var_meta`.** `JaxStats.from_api` (`jax_stats_bridge.py:151-159`)
   currently pulls the enabled-variable list + metadata from `clubb_api`. Source it from
   `StatsWriter.registry` instead (already parsed from the registry file).
4. **Swap radiation stats.** Point the driver at `Radiation/radiation_module.py` (writer-based), or change
   `Radiation/radiation.py`'s `clubb_api.stats_update` calls to `state['stats_writer'].update`.
5. **Repoint `derived_types/converters.py`** imports from `clubb_python.derived_types` to the pure-Python
   `clubb_jax.src.derived_types.*` mirrors — or drop the converters entirely once nothing constructs the API
   (Fortran) types.
6. **Port the lazy forcing** at `prescribe_forcings.py:158` to the JAX forcing path so it no longer imports
   `clubb_api`.
7. **err_info / debug level** — use `derived_types/err_info.py` in place of `clubb_api.init_err_info /
   get_err_code / reset_err_code / cleanup_err_info / set_debug_level` (only needed for the `clubb_driver.py`
   `run_clubb` entry; the standalone path carries `err_info` as the pure-Python type already).
8. **Guard the import.** Once the calls are gone, drop the module-level `from clubb_python import clubb_api` in
   `advance_clubb_to_end.py:12`, `clubb_case_initalization.py:8`, `Radiation/radiation.py:15`. `test_standalone_jax.py`
   /`test_no_dead_imports.py` already assert the JAX driver references no `clubb_python`; they will gate this.

### The one real gap — stats metadata

The Fortran per-variable metadata (grid `zt`/`zm`, units, long-name for ~800 stats variables) is **hardcoded across
`stats_init_zt.F90` / `stats_init_zm.F90` / …**, not in the stats `.in` file. `StatsWriter._parse_registry`
(`stats_writer.py:44`) expects an **enriched** registry of the form `entry(N) = "name|grid|units|long_name"` — which
the stock `clubb_release/input/stats/standard_stats.in` does **not** provide (it only lists enabled names). So a
prerequisite is generating that enriched registry once (e.g., dump `get_stats_var_meta` for every variable from a
single Fortran run, or transcribe the `stats_init_*` tables) and shipping it under `clubb_jax/`. Until then the
pure-Python writer can only emit variables whose metadata is captured in the registry it is handed.

---

## Removing the `clubb_release` *directory* entirely (data files)

Independent of f2py. The init path resolves inputs against
`_CLUBB_RELEASE_ROOT = parents[2]/"clubb_release"` (`clubb_driver.py:15`,
`clubb_case_initalization.py:50`) and reads:

- `input/case_setups/<case>_model.in`, `_forcings.in`, `_sounding.in`, `_ozone_sounding.in`
  (`clubb_case_initalization.py:225-268,963-965`; `clubb_driver.py:225-265,1197-1209`)
- `input/tunable_parameters/tunable_parameters.in` (`clubb_driver.py:799`)
- `input/stats/standard_stats.in` (`clubb_case_initalization.py:216`)
- sounding files resolved relative to `clubb_release` (`Input_fields/sounding.py:391`)

To run with no `clubb_release`: copy the needed `input/` subtrees under `clubb_jax/` (or a configurable data root),
add a `CLUBB_JAX_DATA_ROOT` override, and update the two `_*_root()` helpers to prefer it. This is plain file
relocation — no Fortran involved.

---

## Net

- **Remove f2py:** finish wiring `StatsWriter` + pure-Python derived types/err_info into the standalone path
  (steps 1-8) and generate the enriched stats registry. The scaffolding is already in the tree.
- **Remove the `clubb_release` directory too:** additionally vendor the `input/` data files and add a data-root
  override.
