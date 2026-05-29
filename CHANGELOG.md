# CLUBB-JAX Changelog

Append-only record of work completed. For project design, conventions, and what's next, see `DESIGN.md`.

---

### 2026-05-29 — Directory restructure: CLUBB-JAX repo

- Moved `clubb_jax/` out of `clubb_release/` into the top-level `CLUBB-JAX/` directory
- `clubb_release/` is now a git submodule pointing to `larson-group/clubb_release` master
- Test scripts (`run_scm.py`, `compare_runs.py`) moved into `clubb_jax/run_scripts/`; `clubb_release/` is unmodified upstream
- `clubb_jax/src/clubb_standalone.py`: uses `_CLUBB_RELEASE_ROOT` to locate Fortran input files from the sibling submodule
- Verified tests pass against a fresh clone of `clubb_release` master

### 2026-05-29 — `clubb_jax/src/` mirrors Fortran `src/` layout (Refactor Iters 1–3)

- Restructured `clubb_jax/` so every JAX module sits at the same relative path as its Fortran oracle
- Removed backward-compat shim directories (`jax_core/`, `benchmark_cases/`, `io/`)
- All primary consumers updated to import from canonical `src/` paths

### 2026-05-29 — Port check_clubb_settings and check_parameters to Python (Iter 72)

- `numerical_check.py`: `check_clubb_settings_jax` (10 validation checks, fatal + warning), `check_parameters_jax` (all range checks)
- `src/clubb_standalone.py` now has **zero `from clubb_python import clubb_api` imports**

### 2026-05-29 — Port parameterization_check and init routines (Iters 69–70)

- `numerical_check.py`: `parameterization_check_jax` (NaN/Inf + negativity checks, 35 arrays)
- `parameters_tunable.py`: `init_clubb_params_jax`, `calc_derrived_params_jax` — bit-exact
- `model_flags.py`: `get_default_config_flags_jax` — all 88 flags

### 2026-05-28 — Pure-Python stats writer (Iter 67)

- `io/stats_writer.py`: `StatsWriter` mirrors `stats_netcdf.F90` (begin/update/budget/end_timestep, accumulation, NetCDF output)
- All `clubb_api.stats_*` calls removed; ARM per-timestep Fortran calls: **ZERO**

### 2026-05-28 — Bug fix: ice_supersat_frac (Iter 68)

- Missing `ice_supersat_frac = cloud_frac.copy()` after Block U PDF closure caused cascade failure at timestep 214 in 225-step runs

### 2026-05-28 — Port ARM forcings to pure Python (Iter 66)

- `Benchmark_cases/arm.py`: `prescribe_forcings_arm`, `load_arm_forcings_data`, `_diag_ustar` (Monin-Obukhov, 4 iterations)
- Last per-timestep Fortran call removed from `advance_clubb_to_end.py` for ARM

### 2026-05-28 — Initialization ports: hydrostatic, rcm_sat_adj, calculate_thvm (Iters 62–65)

- `calc_pressure.py`: `hydrostatic_jax`, `init_pressure_jax` (sequential upward integration via `jax.lax.scan`)
- `saturation.py`: `rcm_sat_adj_jax` (bisection, 100 iterations, vectorized over `(ngrdcol, nzt)`)
- `advance_clubb_to_end.py`: `calculate_thvm` now uses `calculate_thvm_jax`

### 2026-05-28 — Remove all Fortran calls from advance loop (Iters 56–65)

- Replaced `set_lscale_max`, upwind TA terms, `pdf_params` Fortran object, `sat_mixrat_liq`, `thlm2t_in_k`, `calc_lscale_directly`, and all non-ARM conditional Fortran paths
- ARM state path: **ZERO Fortran calls** after Iter 59

### 2026-05-27 — Remove Fortran oracle calls from advance loop (Iters 46–55)

- Removed Fortran `advance_xm_wpxp`, `advance_xp2_xpyp`, `advance_wp2_wp3`, `advance_windm_edsclrm`, `pdf_closure_driver` calls
- Removed all shadow comparison infrastructure; JAX values primary
- compare_runs.py: PASS (100 timesteps, 0 prognostic failures)

### 2026-05-27 — JAX drives all prognostic state (Iters 34–45)

- All 16 prognostic variables carried forward from JAX each timestep
- Replaced Fortran clip/fill_holes calls with JAX equivalents; cross-timestep ADG1 state passing
- compare_runs.py: PASS (30 timesteps, 0 prognostic failures)

### 2026-05-27 — ADG1 PDF closure (Iters 25–33)

- `adg1_adg2_3d_luhar_pdf.py`: full ADG1 closure — w-closure, responder params, all higher-order moments (wp2xp, wpxp2, wp2xp2, wp4, wprtp2, wpthlp2, wprtpthlp), virtual temperature fluxes

### 2026-05-27 — Pre-advance diagnostics (Iters 15–24)

- Ported: Skw, thvm, BV, Ri, Cx, Lscale/tau, splat, sfc_varnce, sigma_sqd_w, clip functions, fill_holes
- All overriding Fortran from Iter 38

### 2026-05-27 — Full advance functions (Iters 10–14)

- `advance_xp2_xpyp`, `advance_xm_wpxp`, `advance_wp2_wp3`, `advance_windm_edsclrm`, `advance_xp3` — all machine epsilon vs Fortran

### 2026-05-27 — Core operators and LHS/RHS terms (Iters 1–9)

- Grid interpolation, diffusion LHS, tridiagonal solver, MA/DP1/xp2_xpyp/TA LHS and RHS terms
- Unit test suite established (solver, diffusion, penta-solver, Fortran oracle)
