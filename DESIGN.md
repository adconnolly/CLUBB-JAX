# CLUBB-JAX Design

Goal: translate the CLUBB turbulence parameterization from Fortran to JAX for differentiable,
composable use in ML and autodiff workflows.

---

## Repository Structure

```
CLUBB-JAX/
├── clubb_jax/              ← JAX translation (this repo)
│   ├── src/
│   │   ├── CLUBB_core/     ← physics modules, one file per Fortran oracle
│   │   ├── Benchmark_cases/
│   │   ├── Input_fields/
│   │   ├── Radiation/
│   │   ├── io/             ← stats_writer.py (pure Python NetCDF output)
│   │   ├── clubb_standalone.py
│   │   └── advance_clubb_to_end.py
│   ├── run_scripts/        ← test infrastructure
│   │   ├── compare_runs.py ← Fortran vs JAX comparison
│   │   └── run_scm.py      ← single-case runner
│   ├── tests/              ← unit tests
│   └── clubb_standalone.py ← entry point (python -m clubb_jax.clubb_standalone)
└── clubb_release/          ← git submodule: larson-group/clubb_release (master)
    ├── src/                ← Fortran source oracle
    ├── input/              ← case setups, namelists, sounding files
    ├── clubb_python_api/   ← f2py compiled wrappers (not in submodule git)
    ├── bin/ install/       ← compiled Fortran binaries (not in submodule git)
    └── run_scripts/        ← upstream Fortran test scripts (not modified)
```

`clubb_jax/` and `clubb_release/` are siblings. The JAX package works against upstream
`clubb_release` master — it has no dependency on any custom branch.

---

## How to Test

**Prerequisites:** `clubb_release/bin/clubb_standalone` and `clubb_release/clubb_python_api/*.so`
must be present (compiled artifacts, not in git — copy from a build or compile from source).

```bash
# From CLUBB-JAX/ or any directory:

# Quick JAX-only run (3 timesteps, ~20s):
python clubb_jax/run_scripts/run_scm.py arm -jax -max_iters 3

# Full Fortran-vs-JAX regression (30 timesteps):
python clubb_jax/run_scripts/compare_runs.py --case arm --max-iters 30

# Unit tests (pure JAX, no Fortran needed):
python clubb_jax/tests/test_solver.py
python clubb_jax/tests/test_diffusion.py
python clubb_jax/tests/test_penta_solver.py
```

`compare_runs.py` runs Fortran and JAX independently, then diffs their stats NetCDF files.
All **PROGNOSTIC** variables must PASS (rel tol 1e-6). Diagnostic timing differences are expected.

**Current status:** 0 prognostic failures at 30, 100, and 225 timesteps (ARM case, ADG1 PDF).

4 persistent diagnostic-only differences (not fixable without matching Fortran FP ordering):
- `rtm_spur_src`: ~2e-16 (machine epsilon — FP cancellation)
- `thlm_spur_src`: ~2e-11 (sign-opposite cancellation residual)
- `rtp2_pd`: ~7e-27 (FP noise in positive-definite correction)
- `up2_pd`: ~1e-17 (machine epsilon, rel just over 1e-6 threshold)

---

## Critical Conventions

**Band ordering:** Both Fortran and JAX use `lhs[0=super, 1=main, 2=sub]`. No flip needed
between diffusion output and solver input.

**Grid weights (`weights_zm2zt`):** Shape `(ngrdcol, nzt, 2)`. `[:,k,0]` = M_ABOVE (weight for
`zm[k]`), `[:,k,1]` = M_BELOW (weight for `zm[k+1]`). Fortran 1-indexed `m_above=1, m_below=2`.

**JAX x64 mode:** `jax.config.update("jax_enable_x64", True)` called at module load in
`advance_clubb_core_module.py`. All arrays must stay float64.

**Index mapping (Fortran 1-based → Python 0-based):**
Interior loop `k=2..nzm-1` in Fortran → Python `[:,1:-1]` on zm-level arrays.

**`clubb_params` indexing:** Shape `(ngrdcol, 102)`, 0-based. Access as `clubb_params[:, iC2rt - 1]`.

---

## What Has Been Built

Each JAX module mirrors its Fortran oracle at the same relative path under `src/CLUBB_core/`.

| JAX Module | Fortran Oracle | Status |
|---|---|---|
| `grid_class.py` | `grid_class.F90` | `zm2zt`, `zt2zm`, `ddzm`, `ddzt`, `zm2zt2zm`, `zt2zm2zt` — unit tests pass |
| `diffusion.py` | `diffusion.F90` | `diffusion_zt/zm_lhs`, `xpyp_term_ta_pdf_lhs/rhs` (centered + upwind) — ≤ machine epsilon |
| `matrix_solver_wrapper.py` | `tridiag_lu_solver.F90` | `tridiag_lu_solve_jax` — bit-exact |
| `advance_xp2_xpyp_module.py` | `advance_xp2_xpyp_module.F90` | Full solve for rtp2/thlp2/rtpthlp/up2/vp2 — machine epsilon |
| `advance_xm_wpxp_module.py` | `advance_xm_wpxp_module.F90` | Full solve for wprtp/rtm/wpthlp/thlm/upwp/um/vpwp/vm — machine epsilon |
| `advance_wp2_wp3_module.py` | `advance_wp2_wp3_module.F90` | Full solve for wp2/wp3/wp2_zt — machine epsilon |
| `advance_windm_edsclrm_module.py` | `advance_windm_edsclrm_module.F90` | No-op for ARM (l_predict_upwp_vpwp=True) — bit-exact |
| `advance_xp3_module.py` | advance_xp3 + Skx_module | rtp3/thlp3/up3/vp3 (ADG1 path) — machine epsilon |
| `advance_helper_module.py` | `advance_helper_module.F90` | Skw, thvm, BV, Ri, Lscale/tau, splat, Cx — machine epsilon |
| `sfc_varnce_module.py` | `sfc_varnce_module.F90` | Surface second-order moments — sub-machine precision |
| `sigma_sqd_w_module.py` | `sigma_sqd_w_module.F90` | σ²_w PDF width parameter — bit-exact |
| `fill_holes.py` | `fill_holes.F90` | `fill_holes_vertical`, `fill_holes_wp2_from_horz_tke` — machine epsilon |
| `clip_explicit.py` | `clip_explicit.F90` | `clip_variance`, `clip_skewness`, `clip_covar`, `clip_rcm`, `clip_covars_denom` — bit-exact |
| `adg1_adg2_3d_luhar_pdf.py` | `adg1_adg2_3d_luhar_pdf.F90` + `pdf_closure_module.F90` | Full ADG1 PDF closure — machine epsilon |
| `mixing_length.py` | `mixing_length.F90` | `diagnose_lscale_from_tau` + `compute_mixing_length` (Golaz 2002 nonlocal parcel) — machine epsilon |
| `saturation.py` | `saturation.F90` | `sat_mixrat_liq` (Flatau/Bolton), `rcm_sat_adj` (bisection) — machine epsilon |
| `T_in_K_module.py` | `T_in_K_module.F90` | `calculate_thvm` — bit-exact |
| `calc_pressure.py` | `calc_pressure.F90` + `hydrostatic_module.F90` | `hydrostatic`, `init_pressure` via `jax.lax.scan` |
| `parameters_tunable.py` | `parameters_tunable.F90` | `init_clubb_params`, `calc_derrived_params` — bit-exact |
| `model_flags.py` | `model_flags.F90` | `get_default_config_flags` — all 88 flags |
| `numerical_check.py` | `numerical_check.F90` | `parameterization_check`, `check_clubb_settings`, `check_parameters` |
| `Benchmark_cases/arm.py` | `arm.F90`, `prescribe_forcings.F90`, `time_dependent_input.F90`, `sfc_flux.F90`, `diag_ustar_module.F90` | Full ARM forcing (Monin-Obukhov, time-interpolated) |
| `io/stats_writer.py` | `stats_netcdf.F90` | Pure Python NetCDF stats output (StatsWriter) — bit-exact |
| `advance_clubb_core_module.py` | `advance_clubb_core_module.F90` | Full ARM timestep — **zero Fortran calls** |
| `src/clubb_standalone.py` | `clubb_standalone.F90` | Case initialization — **zero Fortran API imports** |

**ARM per-timestep Fortran calls: ZERO.** All prognostic state, diagnostics, forcings, and
stats output are pure JAX/Python.

---

## Remaining Work

1. `advance_clubb_to_end.py`: Fortran `prescribe_forcings` fallback for non-ARM cases
2. `pdf_closure_driver` for `ipdf_pre_advance_fields` path (not used by ARM)
3. `calc_lscale_directly` for `not l_diag_Lscale_from_tau` path (ported but not regression-tested)

---

## Agent Working Rules

1. **Read `DESIGN.md` in full** at the start of every session — it contains current state, conventions, and what's next.
2. **Append to `CHANGELOG.md`** at the end of each session — one entry summarising what changed. Do not read the full changelog history.
3. Read the Fortran source for the target function in `clubb_release/src/` — it is the oracle.
4. Implement in the appropriate `clubb_jax/src/CLUBB_core/` file (path mirrors the Fortran oracle).
5. Export from `clubb_jax/src/CLUBB_core/__init__.py`.
6. **If the target function is in the ARM timestep path:** add a shadow comparison block in
   `src/CLUBB_core/advance_clubb_core_module.py` that runs both Fortran and JAX on the same
   inputs and prints `max |JAX - Fortran|` via a `report_*_stats()` call registered in
   `src/advance_clubb_to_end.py`. Target ≤ machine epsilon before removing the Fortran call.
   **If the target function is in a non-ARM branch** (e.g. `ipdf_pre_advance_fields`,
   `l_upwind_xpyp_ta`, non-ADG1 PDF path): identify or create a test case that exercises that
   branch — ARM will not enter it, so `compare_runs.py` alone is insufficient. Verify accuracy
   directly via a standalone script or unit test before removing the Fortran call.
7. Run `python clubb_jax/run_scripts/compare_runs.py --max-iters 30` — must show 0 prognostic
   failures. This is a necessary check for ARM regressions but **not sufficient** for non-ARM
   paths, which are not exercised by this test.
8. Update the **Remaining Work** section above and append to `CHANGELOG.md`.
