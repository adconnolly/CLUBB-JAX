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
The regression tests require `clubb_python_api` to be present (for the Fortran comparison run
and for non-ARM `prescribe_forcings`). The JAX driver itself has zero module-level Fortran imports
since Iter 73 — ARM runs do not need `clubb_python_api` to be importable at all.

```bash
# From CLUBB-JAX/ or any directory:

# Quick smoke test — JAX driver, no Fortran comparison run (~20s):
python clubb_jax/run_scripts/run_scm.py arm -jax -max_iters 3

# Full Fortran-vs-JAX regression for one case (30 timesteps):
python clubb_jax/run_scripts/compare_runs.py --case arm --max-iters 30

# Multi-case regression dashboard (generalises the test beyond ARM):
python clubb_jax/run_scripts/compare_cases.py --max-iters 30
python clubb_jax/run_scripts/compare_cases.py --cases arm,bomex,wangara --max-iters 15

# Whole unit-test suite in ONE command (Iter305) — runs every clubb_jax/tests/test_*.py, reports pass/fail
# per file + a summary; exit 0 iff all green. Tests vs an unavailable oracle SKIP cleanly (so it is portable).
python clubb_jax/run_scripts/run_all_tests.py            # all 21 files (~12 min: bugsrad+standalone are slow)
python clubb_jax/run_scripts/run_all_tests.py -k solver  # only files matching "solver"

# Divergence-onset classifier (Iter311) — after a compare_runs.py run, classify WHEN/HOW each failing
# prognostic diverges: bit-faithful / FP-growth (gradual, undamped-oscillator-style) / JUMP@N (discrete
# onset → rule out a term/threshold bug). Generalises the per-step decomposition done by hand on every case.
python clubb_jax/run_scripts/diagnose_divergence.py rico            # only failing prognostics
python clubb_jax/run_scripts/diagnose_divergence.py rico --var thlp2  # one variable, pass or fail

# Unit tests (pure JAX, no Fortran needed):
python clubb_jax/tests/test_solver.py
python clubb_jax/tests/test_diffusion.py
python clubb_jax/tests/test_penta_solver.py
python clubb_jax/tests/test_penta_faithful.py      # penta_lu solve == Fortran-order numpy replica (0 ULP eager)
python clubb_jax/tests/test_f2py_advance_xm_wpxp.py # f2py advance_xm_wpxp .so directly callable (oracle unblocked; needs clubb_python_api)
python clubb_jax/tests/test_differentiability.py   # jax.grad through the building blocks
python clubb_jax/tests/test_pdf_utilities.py       # lognormal<->normal moments — BIT-TO-BIT vs f2py
python clubb_jax/tests/test_kk_autoconversion.py   # KK rate functions vs quadrature/scipy
python clubb_jax/tests/test_kk_rico_oracle.py      # KK autoconv END-TO-END vs Fortran rico rrm_auto
python clubb_jax/tests/test_fill_holes_mean.py     # rtm_cl/thlm_cl mean-field fill (Iter186 fix guard)
python clubb_jax/tests/test_morrison_special.py    # Morrison POLYSVP/DERF1/GAMMA vs scipy (Iter190-192)
python clubb_jax/tests/test_morrison_rates.py      # Morrison warm-rain + ice rates vs nov11_altocu oracle (Iter193-201)
python clubb_jax/tests/test_morrison_differentiable.py  # Morrison rate library is jax.grad-able (Iter203)
python clubb_jax/tests/test_bugsrad.py             # 17 BUGSrad tests: RT machinery (vs Fortran replicas ≤2e-13) + bugs_rad + bugsrad_driver + advance_radiation dispatch (invariants) (Iter255-269)
python clubb_jax/tests/test_soil_vegetation.py     # soil_vegetation force-restore surface BC (gabls3 l_soil_veg): BIT-EXACT vs Fortran replica (Iter270)
python clubb_jax/tests/test_standalone_jax.py      # "entirely in JAX": ARM runs with clubb_python BLOCKED (find_spec import-blocker) (Iter280)
```

**★ Unit-test sweep (Iter298): all 15 JAX-only test files PASS** (run each `python clubb_jax/tests/<t>.py`). Tests that
compare against the f2py oracle (`test_pdf_utilities`, `test_Nc_Ncn_eqns`) now **SKIP that part gracefully when
`clubb_f2py` is not built** (try/except ImportError → SKIP+return, like `test_penta_solver`'s "no Fortran API") instead
of hard-failing — so the suite is a clean regression gate in any environment. **Convention: an f2py-oracle import in a
test must be guarded; the JAX-only assertions (or oracle-stats comparisons) should still run/skip cleanly without it.**

**★ Oracle-protection convention (Iter218).** `run_scm.py <case> -jax` writes its stats to the DEFAULT
`clubb_release/output/<case>_stats.nc` — the SAME path as the Fortran oracle. Running a JAX case there
CLOBBERS the oracle (a 478MB Fortran file became a 6MB JAX file → all rate tests went nan). **Always
pass `-out_dir <somewhere_else>` when running a JAX case that has a stored oracle**, and regenerate a
clobbered oracle with `run_scm.py <case> -legacy`. `compare_runs.py` is safe (it manages its own dirs).

**f2py oracle caveat (Iter115).** Most CLUBB_core routines are f2py-verifiable bit-to-bit, but some
wrappers FPE-trap (core dump) in this environment — `f2py_precip_fraction` crashes inside
`precip_fraction_` itself (a build-level `-ffpe-trap` plus the wrapper's internally-constructed grid),
uncatchable from Python. When an f2py wrapper core-dumps, fall back to the case-stats oracle (below).
Note the stats oracle has a **tol-boundary timing confound**: a routine called mid-timestep sees inputs
that differ slightly from the END-of-step stored stats, so discrete tolerance tests (e.g.
`any(hydromet>=tol)`) can flip at boundaries — validate on the well-resolved interior (values
comfortably above the tolerance), where the match is exact.

**Testing strategy for unported subsystems via case-stats oracles (Iter113).** The f2py API exposes
no microphysics, so a ported microphysics rate function can't be shadow-tested in-loop. But a Fortran
SCM run writes BOTH the PDF component moments (its inputs) AND the process rates (its outputs) to the
stats file — so the rate function is verifiable in isolation by feeding the Fortran's OWN moments into
the JAX rate and comparing to the Fortran's OWN rate output. `test_kk_rico_oracle.py` does this for KK
autoconversion: it reads `chi_1/2`, `stdev_chi_1/2`, `mu_Ncn`, `mixt_frac`, `rho` from
`clubb_release/output/rico_fort/rico_stats.nc`, computes `KK_auto_upscaled_mean`, and matches the
Fortran `rrm_auto` (significant points to ~1e-6, median 4.7e-7). This **decouples** rate-function
correctness from the (unported) hydrometeor-PDF-setup correctness — validate each independently. Generate
the oracle stats with `run_scm.py rico -legacy -out_dir clubb_release/output/rico_fort`; the test skips
if absent. **rico specifics:** N_cn is constant (`sigma_Ncn=0`, l_const_Nc_in_cloud) and `corr_chi_Ncn=0`,
so only the const_x2 bivar path is exercised — log moments/correlations are NOT needed to match rrm_auto.
**Two rico oracle runs (Iter133):** the canonical **10-step** `rico_fort` (the auto/accr/evap rate tests are
tuned to it — a longer run adds near-dispatch-boundary points that nudge the autoconv sig-tol past 5e-6) AND a
**250-step** `rico_long_fort` for tests that need *developed rain* — the 10-step run has only sub-micron drops
whose (positive) KK sedimentation velocities all clip to 0, so the sedimentation test needs the longer run
(`run_scm.py rico -legacy -max_iters 250 -out_dir clubb_release/output/rico_long_fort`). **Grid-staggered
oracle stats (Iter133):** a stat may be stored on a DIFFERENT grid than its natural home — e.g. `Vrr`/`VNr` are
written as `zt2zm(hydromet_vel_zt)` on MOMENTUM levels (zm, 58) while the velocity is computed on zt (57); to
validate, apply the same bit-faithful `zt2zm` to the JAX zt-level output (a grid can be reconstructed from the
stats `zt`/`zm` coordinate vars, treating time as the ngrdcol batch). Check `ds[var].dimensions` before comparing.

**Hydrometeor budget-stat bookkeeping (Iter137): a `*_ts`/`*_ta` stat is EXPLICIT+IMPLICIT.** The
turbulent-advection (`rrm_ta`) and turbulent-sedimentation (`rrm_ts`) budget stats are split across two
routines: `microphys_rhs` stores the EXPLICIT half via `stats_begin_budget` (the Crank-Nicholson explicit
½-diffusion for ta; the `term_turb_sed_rhs` of Vhmphmp_expc for ts), and `microphys_solve` ADDS the IMPLICIT
half via `stats_finalize_budget` (`-lhs·hmm`). So the stored `rrm_ts` = explicit + implicit — computing only
`-sed_turb_lhs·hmm` will NOT match it (this is why a naive implicit-only check failed by ~30×). By contrast
`rrm_sd`/`rrm_ma` are plain `stats_update(-lhs·hmm)`. For verifying these operators, prefer the conservation
contract over the in-loop stat.

**Testing convention (Iter136): conservation-contract oracle for flux-form transport operators.** A
discretized flux-divergence operator (e.g. sedimentation `sed_centered_diff_lhs`) can be verified
RIGOROUSLY and oracle-free by its conservation law: the column-mass-weighted (Σ rho_ds_zt·dzt··)
tendency must equal the net boundary flux (for sedimentation, the surface flux rho_ds_zm[0]·V[0]·hm[0],
with no flux through the top). This holds to machine precision (5.6e-15) regardless of the within-step/
end-of-step **timing confound** that makes the stored budget stats (`rrm_sd` etc.) unusable except at
trivially-clean points. Use it when the in-loop budget stat is timing-confounded. **Gotcha:** a new
microphysics module must call `jax.config.update("jax_enable_x64", True)` at import (like
`advance_clubb_core_module.py`) — without it jnp arrays default to **float32**, which silently passes most
relative-error tests but breaks a conservation contract (cancellation) at ~1e-5; the operator can be exactly
correct (matches a float64 flux-divergence reference to 7e-16) yet the contract fails purely from float32.

**Differentiability / composability status (Iter104, extended Iter122).** The project goal is a
*differentiable, composable* JAX CLUBB. `test_differentiability.py` verifies (with finite-difference
correctness checks) that the pure-JAX building blocks — saturation (`sat_mixrat_liq/ice`), the
`lax.scan`-based tridiagonal solver (grad w.r.t. both rhs and the matrix coefficients), the erf-based
PDF cloud-fraction core, Brunt-Vaisala — support `jax.grad` and **compose** into differentiable
pipelines. **The composed KK microphysics drivers (auto/accr/evap) are differentiable end-to-end OVER THE FULL
rico array** (Iter122 single-point; Iter127 full-array, edge-case robust) — `jax.grad` flows through
Nc→Ncnm→log moments→the analytic PDF integral (incl. the D_v parabolic cylinder function),
finite-diff-correct to rel ~1e-9. **Differentiability-hardening convention:** extreme-subsaturation
points give a vanishing erfc denominator (~1e-170) whose square underflows to 0, so the bare quotient
gradient is 0/0=nan, and `jnp.where` masking does NOT help (its VJP computes the nan first, then nan·0=nan).
The fix is AT the operation, all FORWARD-preserving (verified by the f2py bit-to-bit + moment-preservation +
quadrature tests): a `jax.custom_jvp` safe-division (`Nc_Ncn_eqns._safe_div`), a double-where `_safe_sqrt`
and a guarded Rmax denominator (`setup_clubb_pdf_params` — no-precip points give 0/0), a double-where
`_pos_pow` for hydrometeor-mean powers at 0 (`PDF_integrals_means`, e.g. mu_rr^(1/3) at mu_rr=0), and a D_v
argument clamp to ±50 (`PDF_integrals_means._dvc` — the dispatch only SELECTS these forms when |s_c|<=49, so
clamping the unused extreme argument stops the large-negative-z series overflow without changing any selected/
tested case). With all of these, **`compute_kk_microphysics` is FULLY differentiable** (w.r.t. both the rrm
field AND the chi PDF moments; finite-diff-correct to rel ~1e-10), the whole composed microphysics step.
**End-to-end `jax.grad` through a whole CLUBB timestep is NOT yet available** — the precise blockers
(Iter179 audit of `advance_clubb_core_module.py`, 4904 lines):
1. **The orchestration round-trips through NumPy: ~520 `state[..] = np.asarray(..)` writebacks** (sever the
   autodiff graph) + **~50 in-place index mutations** (`arr[i]=v`, break jnp immutability). These are coupled
   (the refactor is all-or-nothing — keeping jnp in the state dict immediately breaks the in-place mutations),
   so it cannot be done incrementally, and it risks all 15 bit-faithful cases. (353 `jnp.asarray` INPUT
   conversions are harmless for grad.)
2. **The monotonic flux limiter (`mono_flux_limiter.py`) is pure NumPy with Python `for` loops** —
   non-differentiable. It triggers only for atex/gabls3_night; a differentiable forward pass on the other 13
   cases could skip it.
3. **`mixing_length.py` (the Golaz parcel-ascent Lscale) uses `lax.while_loop` (mixing_length.py:367,:553) with
   dynamic stop** → supports **forward-mode AD (`jax.jvp`, tested) but NOT reverse-mode (`jax.grad` raises)**.
   A bit-exact transform to bounded `lax.scan`-with-done-mask exists (the body already freezes via `jnp.where`
   once `done`; run `length=k_ub_zt`, frozen iters are no-ops) — but it's **inside the outer per-start-level scan
   (mixing_length.py:474), so it becomes a NESTED scan that loses the while_loop's early-exit → O(nzt²) forward
   pass** (a perf REGRESSION for all 15 faithful cases, which don't need grad). Net: ruled out (Iter180) — it
   wouldn't unlock the full grad (1+2 remain) and would slow the bit-faithful path.
**Component-level differentiability + composability — the practical 'differentiable composable' claim — IS done
and tested** (`tests/test_differentiability.py`, 14 tests: saturation, tridiag solver, **penta solver (Iter295)**,
**fill_holes (Iter295)**, PDF cloud_frac, Brunt-Vaisala, composability, KK rate drivers, KK autoconv-over-array, ADG1
w-PDF closure, ADG1 FULL pdf-driver (Iter185), KK covar driver, mixing-length forward-mode). **★ The Iter290-291 core
jits (parabolic_cylinder D_v, tridiag+penta solvers, fill_holes — the OOM/recompile fix) all PRESERVE differentiability
— confirmed by this suite (penta grad rel 4.7e-9, fill_holes reverse-grad rel 1.4e-10; jit composes with grad).** **The gap is the GLUE, not the physics (Iter181 audit):** the CORE physics modules — the ADG1 PDF
closure (`adg1_adg2_3d_luhar_pdf.py`) and the prognostic solvers (`advance_xm_wpxp`/`advance_wp2_wp3`) — are
while_loop/numpy-free (only static unrolled loops) → REVERSE-mode differentiable (ADG1 w-closure tested, rel 1.4e-9).
Only the 3 blockers above (orchestration numpy glue, the numpy flux limiter, the mixing_length while_loop) stand
between component-differentiability and full-timestep grad.

**"Entirely in JAX" — import-clean (Iter279); but NOT all faithful cases are runtime-clean (Iter281 correction).**
The JAX driver no longer imports `clubb_python` at module level: `model_flags.py` uses the pure-JAX `ConfigFlags`
(field-identical), and the lone `clubb_api` import in `advance_clubb_core_module.py` is LAZY. So **cases with ported
forcings run with ZERO Fortran import/call** — verified by running them with `clubb_python` blocked via a sys.meta_path
import-blocker (`tests/test_standalone_jax.py`: 13 cases incl. atex/atex_long/rico run with `clubb_python` BLOCKED). **★ But "faithful" ≠
"entirely-in-JAX" — Iter282 audit** (init each case + call `prescribe_forcings_generic`; NotImplementedError ⇒
fallback): **ENTIRELY-IN-JAX FORCINGS (19, NO fallback remaining)** = arm, cobra, bomex, fire, neutral, ekman, gabls3_night,
jun25_altocu, mpace_a, gabls3, gabls2, wangara, dycoms2_rf01, dycoms2_rf02_nd, dycoms2_rf02_so, dycoms2_rf01_fixed_sst
(sfctype=1 fixed-SST sfclyr branch), atex (90-min subsidence gate), atex_long (3-piece subsidence + 4-piece thlm/2-piece
rtm forcing + 43200 s spin-up), **rico (Iter289: 3-piece `t_tendency`/exner thlm + 4-piece specific-humidity `qtm_forcing`
→ `rtm_forcing = (1+rtm)²·qtm` per spec_hum_to_mixing_ratio.F90; `wm` is init-set, untouched)**. **FORTRAN FALLBACK: NONE.**
**★ "entirely-in-JAX forcings" ≠ "bit-faithful full run":** rico's forcings are pure JAX (3-step blocked run PASS).
A 30-step rico run used to OOM (SIGKILL/137); **FIXED Iter290** (see the jit-recompilation note below) — rico now
completes 30 steps (163 s, bounded memory). It is **still NOT bit-faithful** (15 prognostics, rel 3e-6–6e-4): KK-FP-limited,
the drift seeded by the parabolic-cylinder handoff band (~2e-4) — NOT a forcing bug (the KK rate tests are bit-identical).
**★★ jit-recompilation → unbounded compile-cache → OOM (the Iter290 root-cause + convention):** an **eager `lax.scan`
whose body CLOSES OVER a concrete (non-tracer) array bakes that array's VALUES into the scan's jaxpr as literal
constants**, so XLA recompiles every timestep when the values change → the compile cache grows without bound → OOM on
long runs (`JAX_LOG_COMPILES=1` showed ~137 scan recompiles/step for rico, dominated by `jit(scan)`). **Two fixes, both
value-preserving + differentiable:** (1) `parabolic_cylinder.dv_parabolic_cylinder` (KK D_v series/asymptotic scans
close over `z`/`w`) — wrapped in `jax.jit` so `v`/`z` become tracers, the captured arrays hoist to scan operands, and
the whole D_v graph compiles ONCE (eliminated ~1152 of 2165 compiles); (2) `matrix_solver_wrapper.{tridiag,penta}_lu_solve_jax`
(nested scan bodies `lu_step`/`fwd_step`/… are redefined each call → eager scan-cache miss) — jitted in place so each
solve hits the jit cache by aval (one compile per distinct grid size, then reused; bounded their recompiles 120+→1).
Net: rico 2165→597 total compiles, 30-step OOM gone, ~1.5× faster. **Verified bit-faithful: arm (tridiag) + bomex (penta)
compares 0 prognostic failures; solver unit tests 6/6; KK rate tests bit-identical (D_v tolerances unchanged).** **General
rule: any per-timestep eager `lax.scan` (or function containing one) should be `jax.jit`-wrapped at a stable entry point;
diagnose with `JAX_LOG_COMPILES=1` and `grep -c "Compiling jit(scan)"` — a count that grows each step is this bug.**
**Iter291 finished the job:** the residual ~9 scan-recompiles/step were `fill_holes_vertical_jax` (called ~7-9×/step for
rtm/thlm/rtp2/thlp2/up2/vp2/wp2; its sliding-window `fori_loop` / global-fill bodies close over `rho_dz`/`threshold`).
Jitted with the int control args static (`lower_k`/`upper_k`/`fill_holes_type`/`grid_dir_indx`), `threshold` traced.
**Net over Iter290+291: rico per-step scan-recompiles 137/step → effectively 0 (only 1 `jit(scan)` compile for the
whole 12-step run; total compiles 2165→381; 12-step runtime 107s→45s).** The compile cache is now BOUNDED for every
case → arbitrarily long runs are safe and faster. Re-verified bit-faithful: arm + bomex compares 0 prognostic failures,
13-case standalone PASS. (Mixing_length's while_loops/scans and calc_pressure also close over arrays but compile to
`while`/non-scan and didn't show as a per-step leak; left as-is.) Multi-case standalone runs (13 in one process) still
keep `jax.clear_caches()`+`gc.collect()` between cases (Iter289) as a belt-and-suspenders against first-compile spikes.
**★ test_standalone_jax.py gotcha (Iter288):** the `clubb_release/` checkout contains an *unrelated* `clubb_jax/`
scaffold (different naming) that shadows our package if `clubb_release` precedes `jax_root` on `sys.path` →
`ModuleNotFoundError: clubb_jax.src`. The test now inserts `jax_root` LAST so it wins at `sys.path[0]`.
**★ Variants share a runtype FIELD distinguished by another flag** (rf01 vs rf01_fixed_sst by `sfctype` 0/1; rf02
nd/so/do/ds all 'dycoms2_rf02') — key off `state['runtype']` + the distinguishing flag, NOT the case-file name, and
re-run the affected variant after a port/revert (a revert keyed on the case-file name silently no-ops → Iter284
rf01_fixed_sst regression went uncaught until Iter286). Making one entirely-in-JAX = wire its tndcy (zero/`_zero_forcings` if
the subsidence is init-set; or analytic) + VERIFY via the STANDALONE test (clubb_python-blocked) — a plain compare can
PASS via the fallback (false positive). The fallback-hidden sfclyr often carries a bug: gabls2's wprtp ×0.025
(gabls2.F90:299); rf02's sfclyr reading sens_ht/latent_ht (not wpthlp_sfc) → `/(1.21·Cp)`/`/(1.21·Lv)` (Iter285);
rf01_fixed_sst's fixed-SST bug (unfixed). **★ Match `state['runtype']`, not the case-file name — variants share a
runtype (rf02 nd/so/do/ds all = 'dycoms2_rf02').** z_bot is `gr.zt[0]` = Fortran `gr%zt(i,1)`. **Reentrancy (Iter281):** `init_clubb_case` calls `reset_clubb_core_state()` (resets `_prev_adg1_j25`) so
multiple cases can run in one process. The Fortran is still essential as (a) the compiled bit-comparison oracle and
(b) the porting source reference.

**Radiation subsystem differentiability (Iter275):** the BUGSrad correlated-k RT (`bugs_rad`, jitted) and the
soil_vegetation surface budget are `jax.grad`-able — grad of a radiative loss (TOA OLR + SW heating) w.r.t.
temperature/cloud-water is finite AND nonzero (`|dL/dT|=3.8`, `|dL/dqc|=1.2e5`), even through cloudg's float32
`sngl` truncation (a grad-transparent cast). Guarded by `test_bugs_rad_differentiable` + `test_soil_veg_differentiable`.

**★ Bit-faithful: 18 cases (Iter299).** All 18 `microphys_scheme="none"` cases are accounted for — 17 bit-faithful +
coriolis_test (FP-limited) — PLUS **mpace_a (18th, Iter299)**, the FIRST Morrison (`l_ice_microphys=.true.`) case made
bit-faithful: it stays clear/sub-saturated so the only Morrison signal is the clear-air `thlm_mc`, which is the Fortran's
**single-precision `thlm↔T_in_K` round-trip residual** — now reproduced (see below). The OTHER Morrison cases (nov11,
dycoms2_rf02_morr) have ACTIVE microphysics → the nov11-class ice-onset/sed FP floor PLUS a single-precision floor:
**VERIFIED (Iter300) that the entire WRF M2005 (`module_mp_graupel.F90`) is real*4** — every declaration is a bare `REAL`
(default REAL(4)), even in the PRECdouble build. **★ dycoms2_rf02_morr — the single-precision hypothesis was TESTED and
DISPROVEN (Iter303), correcting the Iter302 over-optimistic "single-precision-limited" reclassification.** Its N=2 compare
fails by ~3.5e-6–2.75e-5 (rtm/wprtp/wpthlp/thlp2/rtpthlp; hydrometeors rrm/Nrm PASS), and the Fortran M2005 interface
does build the tendencies through `r4` temporaries. BUT a blanket-float32 M2005 experiment (env-gated `_MPDT=float32`,
all 98 internal casts → float32; verified float64-default bit-identical first) did **NOT** fix it — failures unchanged,
and crucially `rtm`'s error was **identical (3.47e-6) in float32 and float64**, i.e. NOT precision-sensitive. So dycoms's
divergence is NOT simply the M2005 working precision; matching would need the Fortran's EXACT mixed pattern (`r4` internal
+ `real(core_rknd)` interface differences, not blanket float32) AND there's a precision-independent component (likely the
near-singular sed `(qr/nr)^⅓` and/or cloud feedback). **CONCLUSION: dycoms stays not-bit-faithful — single-precision is a
floor but not the sole/dominant cause; a clean fix is not in reach.** (Experiment fully reverted, no residual.) nov11 is
NOT single-precision-limited (step-6 seed is the float64 DYNAMICS, before microphysics activates at step 60). COAMPS
(unported), KK (do/ds, oracle-epss-limited), and SILHS (unported) remain. **★ LESSON: TEST precision hypotheses with an
env-gated blanket-float32 experiment (backup the file, verify float64-default is bit-identical, then flip) before
claiming a case is "single-precision-fixable" — Iter302's read-only reclassification was wrong.** **★ dycoms SEED
PRECISELY LOCALIZED (Iter304) — investigation CONCLUDED (3 iterations, no clean fix):** budget-decompose of the N=2
rtm failure → the seed is `rcm_mc` (cloud-water tendency) off by ~3% (abs ~5e-8) at the SHARP cloud-top CF3D edge
(k≈49, z≈781 m, cloud_frac 0.57→0); `thlm_mc` inherits the SAME 3% (via `thlm_mc=(ten['T']−Lv/Cp·rcm_mc)/exner`), and
both feed rtm/thlm/the 2nd moments. `rcm`/`cloud_frac` themselves are bit-exact (the structure matches); the cloud
sedimentation matches and is FOLDED INTO `rcm_mc` (the JAX `rcm_sd_mg_morr` diagnostic stat reads 0 — a recording gap,
NOT a prognostic bug, since the `rcm_mc` total matches to ~5e-8). So it's a tiny near-singular-edge residual at the
sharp cloud top (the in-cloud ÷CF3D / grid-mean ×CF3D conversion + M2005 process at cloud_frac→0) — consistent with the
~30-iteration Iter219-250 "FP/discretization" verdict. **dycoms stays not-bit-faithful; STOP investigating it.**
**★ KEY LESSON (Iter299): the Fortran M2005 interface keeps `T_in_K`/`rcm_r4` in SINGLE
precision (`real(...)` = default REAL(4), even in the PRECdouble build) — `morrison_microphys_module.F90:399,416,793`.
So `thlm_mc = (T_in_K2thlm(real(T_in_K),exner,real(rcm_r4)) − thlm)/dt` carries a ~1e-7 single-precision round-trip
residual that is NONZERO even with zero microphysics tendencies. The algebraically-equal double-precision form gives 0.
`module_mp_graupel.py:morrison_microphys_driver` now replicates the `float32` casts → mpace_a bit-faithful. Faithfulness
means matching the oracle's PRECISION, not just its formula.**

**mpace_a — BIT-FAITHFUL (Iter299, 18th case).** Custom forcings/surface/init ported to pure JAX (Iter276-277):
`generic_forcings.py` (load_mpace_a_forcings + _mpace_a_tndcy/_sfclyr/_zlinterp/_mpace_time_select) replaces the Fortran
`clubb_api` fallback; `_initialize_em_profile` gained `"mpace_a":(2000,1.0)`; `_mpace_a_tndcy` uses the hardcoded
p_sfc=101000 (mpace_a.F90:140). The case is Morrison (`l_ice_microphys`) but stays clear/sub-saturated → all M2005 rates
0 → the ONLY microphysics signal is the clear-air `thlm_mc`, which CLUBB feeds (lagged) into the next step's `thlm_forcing`.
**The fix (Iter299):** that `thlm_mc` is the Fortran's **single-precision `thlm↔T_in_K` round-trip residual** (~2.8e-7);
`module_mp_graupel.py` now replicates the `real*4` casts (see the KEY LESSON above) → 30-step compare PASSES.
**★ Diagnostic journey (a cautionary tale):** Iter293-296 chased a phantom "forcing-time discrepancy" — the recorded
`thlm_forcing` STAT (8.464e-6) ≠ the raw LS forcing (8.1815e-6) because the STAT = raw + lagged `thlm_mc`. Iter297 settled
it with an **isolated oracle debug build** (`compile.py -install <scratch> -debug`, NOT the reference binary; print the raw
`dTdt_hoc_grid`, capture, revert source) — proving the LS-forcing port was faithful all along. **LESSON: when a recorded
`*_forcing` stat disagrees but the case uses microphysics, check `stat == raw_forcing + lagged *_mc` FIRST; an isolated
oracle debug build is the decisive tool (and is safe) when static analysis stalls.**
**★ Conventions: a
`-jax` run dying
with a Fortran `error stop`/no-Python-traceback is the unported-case `clubb_api` fallback (`_prescribe_forcings`) —
port the case's tndcy/sfclyr to `generic_forcings.py`; custom `.dat` files are level-major (read-all-tokens + reshape
(nlevels,ntimes)); a case's .F90 may HARDCODE a constant overriding the namelist (grep for literal assignments);
validate a ported forcing by per-level NetCDF diff of `*_forcing`.**

`compare_runs.py` runs Fortran and JAX independently, then diffs their stats NetCDF files.
All **PROGNOSTIC** variables must PASS (rel tol 1e-6). Diagnostic timing differences are expected.
`compare_cases.py` wraps it over a list of cases and prints one pass/fail line per case — use
this to track bit-faithfulness as cases beyond ARM are brought up. **Resource note (Iter95):** each
JAX run holds substantial memory; **do not launch multiple `compare_cases`/`compare_runs` jobs in
parallel** — concurrent JAX processes OOM-kill each other (the run dies with `rc=1` partway, looking
like a spurious "JAX run failed"). Run long verifications **sequentially** (one `compare_cases` job,
or one case at a time). A 60-step run that fails only under parallelism but passes standalone is this,
not a physics bug. **Bisecting tip:** when a case
diverges, run `compare_runs.py --case X --max-iters N` then `diagnose_divergence.py X` (Iter311) — it
reports, per failing prognostic, the physical onset step and classifies it as FP-growth (gradual) vs
JUMP@N (a *sudden* jump from machine-eps past the floor at one step). A JUMP signals a discrete
branch/threshold being crossed (e.g. cloud/precip onset where `rcm`/`rrm` first becomes nonzero) —
worth ruling out as a term bug; gradual growth is the accumulating-FP signature. The classifier keys
the onset off the bit-exact floor (ABS_TOL), so it is not masked when a field's global |ref| is large.

**Current status (bit-faithful, 0 prognostic failures, 30 steps):** ARM (also 100/225), BOMEX,
dycoms2_rf01, wangara, atex, gabls2, gabls3_night, fire, neutral, ekman, cobra, dycoms2_rf02_nd,
dycoms2_rf01_fixed_sst, atex_long, dycoms2_rf02_so, jun25_altocu, **gabls3**, **mpace_a (Iter299, the first Morrison
case)** — **18 cases**. `compare_runs.py` auto-forces per-step output, so the comparison reflects physics for all.
**★ Full-gate verification:** `compare_cases.py` covers all 18 (gabls3 = BUGSrad-path guard ~3 min; mpace_a = Morrison
single-precision `thlm_mc` guard) — this is the generalized (beyond-ARM) regression gate. Run it after any change to
shared/core code (it confirmed the Iter290-292 core jits are bit-faithful across every case).

**★★ DURABILITY (Iter312): the 30-step gate MASKS late-activating events — run the gate at 100+ steps periodically.**
A 100-step `compare_cases.py` run surfaced that **atex** failed (16 prognostics) while passing at 30, because its
analytic large-scale thlm/rtm forcing is GATED on `time >= time_initial + 5400` (atex.F90:215; a 90-min spinup) =
**step 91** — past the gate. `diagnose_divergence.py` classified it `JUMP@step91`, and a budget decomp localized it to
`rtm_forcing` (Fortran nonzero at 77 levels, JAX exactly 0): the JAX `_atex_tndcy` had ported only the subsidence, not
`calc_forcings` (the thlm/rtm forcing). **FIXED** (generic_forcings.py) — atex now durable to 200 steps; all 18 pass at
100. **Convention: a case whose forcing/microphysics/event activates at a known time (`time >= …` gates, microphys_start_time,
ice onset) needs verification PAST that step — the 30-step gate is necessary but not sufficient. The durability run is
`compare_cases.py --max-iters 100` (then `diagnose_divergence.py CASE` on any failure: JUMP = a discrete event the gate
missed, often a late forcing/threshold; FP-growth = accumulation).**

**★ Iter313 completed the time-gate audit** — found ALL `time >= …` activations across the benchmark cases and verified
each in a bit-faithful case is reached + bit-faithful: **atex** @step91 (fixed Iter312), **gabls2** subsidence @`time>93600`
= **step1560 → VERIFIED bit-faithful (1580-step compare PASSES)**, **neutral** heat-flux-off @`time>80880` = step50
(covered by the 100-step run, passes; note neutral's `time_initial=77880`). **atex_long** (continuous spinup ramp,
`time<43200`) is NOT a discrete event — but a 740-step run showed it **FP/chaos-limited past ~step 305**: every forcing
coefficient was verified bit-exact term-by-term (3-piece wm, 4-piece thlm, 2-piece rtm; atex.F90/atex_long.F90:calc_forcings),
the |Δ| growth is exponential (Lyapunov, not linear-systematic), and the sign tally FLIPS across steps (a fixed
coefficient bug would not) → genuine chaotic amplification in the 192-hour turbulent run, NOT a bug. **So atex_long's
bit-faithful ceiling is ~305 steps (a known FP/chaos limit — do not treat a >305-step failure as a regression).**
`diagnose_divergence.py` now also prints a **sign tally** at the gate-cross step (Iter313): balanced/flipping → FP/chaos;
strongly one-sided & persistent → a systematic term/coefficient bug.

**★ mpace_a durability (Iter321) — robust, NOT an atex-class late event.** mpace_a (the Morrison case) was the last
bit-faithful case unverified past 100. Checked: **bit-faithful at 150 steps (PASS), physics finite/robust through 250**
(ran the loop with stats off, all prognostics finite). Crucially **cloud_frac=0, rcm=0 even at step 250** — the 72-hour
Arctic-stratus case spins up slowly and forms NO cloud in the testable horizon, so the Morrison single-precision floor is
never triggered (rates stay 0, only the clear-air round-trip matters). So mpace_a's gate-passing is not a masked late
event; it is durably clear-air-bit-faithful. **★ Testing constraint discovered:** a `compare_runs` per-step-stats run of
a Morrison case OOMs between 150–250 steps (corrupt output NetCDF) — NOT a model issue (the physics-only loop with stats
off ran fine to 250). **Iter322 narrowed it:** the `StatsWriter` is already incremental (one record/timestep, buffer reset
each window — does not accumulate records); adding a periodic `ds.sync()` (Iter322, every 20 records — a good-practice
flush that bounds the HDF5 dirty cache) did NOT fix it. So the OOM is a per-`l_sample`-call accumulation in the per-step
diagnostic path. **★ Iter323 ROOT-CAUSED it (env-gated `CLUBB_LEAK=1` instrumentation, since reverted): a JAX/XLA-backend
buffer leak of ~85 diagnostic profile arrays PER STEP** (`jax.live_arrays()` grows by exactly 85/step on bomex; small
profiles → ~0.06 MB/step there, fine to 360; Morrison's much larger var set → ~36 MB/step → OOM ~step 150-250). **Ruled
out exhaustively:** NOT jit recompilation (2140 compiles all at startup, ZERO per-step — confirmed via `JAX_LOG_COMPILES`);
NOT the `StatsWriter` (update/begin_budget/finalize_budget all `np.asarray`-materialise immediately, writes are incremental);
NOT the HDF dirty cache (the `ds.sync()` didn't help); NOT a Python container (a clean `gc.get_objects()` scan found no
list/dict holding them). So the ~85 budget-diagnostic arrays the jitted core returns ONLY when `l_sample=True` are retained
by the XLA backend across iterations (their Python refcount appears to drop, but the device buffers are not released —
likely XLA's CPU buffer pool / a jit-dispatch keep-alive). **A held-array, NOT recompilation; the Iter290 jit fix holds.**
Low priority (no current need for long Morrison compare runs: mpace_a's cloud onset is >250 steps, and
nov11/dycoms2_rf02_morr are FP-limited at short steps). **Workaround: run `advance_clubb_to_end` with
`state['stats_writer']=None` and inspect the state dict directly (Iter321), or sample at the case default stats interval.
Fix direction for a future JAX-savvy iteration: trace why the l_sample diagnostic jit-outputs are not released (try
materialising/`del`-ing every returned diagnostic, or computing only registry-present diagnostics).**

**★★ FULL-LENGTH verification (Iter314) — two classes of bit-faithful case.** Ran the 11 short/medium cases (≤600 steps)
to their `time_final`. **9 are bit-faithful for their ENTIRE run** (a strong claim beyond the 30-step gate): dycoms2_rf01
(240), cobra (300), bomex (360), neutral (360), dycoms2_rf02_nd (360), dycoms2_rf02_so (360), wangara (480), **atex (480,
confirms the Iter312 fix)**, dycoms2_rf01_fixed_sst (540). **Two are CHAOS-HORIZON-limited:** fire (gate-cross ~step147,
case is 180) and jun25_altocu (~step200, case is 600) — both FP-growth (no JUMP), sign tally flips across steps, |Δ|
grows exponentially → genuine chaotic FP amplification, NOT bugs (verified by the same flip/exponential criteria as
atex_long). **★ KEY FRAMEWORK: full-length bit-faithfulness is achievable for non-chaotic cases but PHYSICALLY IMPOSSIBLE
for chaotic turbulence** — two bit-identical-start runs of a chaotic system diverge after the Lyapunov time (butterfly
effect), so a >horizon failure is physics, not a code bug. Chaos horizons seen so far: fire ~147, jun25_altocu ~200,
atex_long ~305; all other tested cases are either fully faithful or faithful well past 100. **The diagnostic that
separates a BUG from CHAOS is JUMP-vs-FP-growth + sign-flip** (atex was a JUMP at step91 = bug; fire/jun25/atex_long are
FP-growth+flipping = chaos). **The 100-step `compare_cases.py` durability gate sits within every case's chaos horizon
(all 18 passed it at 100 except the atex bug), so it is the right practical durability metric** — full-length runs only
add value for the non-chaotic cases (now 9 confirmed) and for catching discrete (JUMP) late events.

**★ Iter315 — DIURNAL SOLAR transition audit (a new discrete-event class beyond `time>=` gates).** Diurnal cases compute
a time-varying solar zenith (`cos_solar_zen`), so the SW radiation turns OFF at sunset / ON at sunrise when `amu0`
crosses the daytime threshold — a discrete event the daytime-only 30/100-step gate never reaches. **gabls3** (starts noon
43200s, lat 51.97°N, July 1 2006) has **sunset at step ~480** (amu0 0.017→-0.014) and sunrise ~step 930. Verified: the
night threshold is `amu0 >= 0.01` in BOTH (`bugs_rad.F:611` `bitx = amu0_loc >= 0.01` ↔ JAX bugs_rad.py `day = amu0>=0.01`;
`if nday==0 goto 1000` = skip SW ↔ JAX zeroes SW), and amu0 changes ~0.002/step near the crossing so a sub-tol amu0
difference can't shift the integer crossing step. **A 510-step compare PASSES (0 prognostic failures); radht_SW goes
1.29e-5 (day) → exactly 0 (night) at the crossing, matched bit-for-bit (|d|=0 at night).** Sunrise uses the identical
amu0≥0.01 mechanism (symmetric) → faithful by the same reasoning. **Convention: for a diurnal case (`l_sw_radiation` +
computed `cos_solar_zen`, not `l_fix_cos_solar_zen`), verify PAST the first solar transition — compute the amu0 zero/0.01
crossing step from lat/lon/date and run just past it.** No bug (the SW day/night mask is bit-faithful).

**Testing dimension — GRID TYPE (Iter145; ★ corrected Iter306).** Most bit-faithful cases use `grid_type=1`
(evenly-spaced), but **`dycoms2_rf02_so` is `grid_type=3` (grid read from file) and IS bit-faithful** — so a
non-uniform grid is NOT a blocker by itself. The uniform grid makes every zt↔zm weight exactly 0.5 and every metric
constant, so it does NOT exercise the stretched-grid paths, where weights are non-uniform and `dzt≠dzm` — exposing bugs
(e.g. `(1-w)` vs direct weights, `invrs_dzt`/`invrs_dzm` swaps; the Iter151 rico `weights_zm2zt` order bug). **rico is the
only `grid_type=2` (formula-stretched) case** — but its grid (zt) is now VERIFIED bit-exact vs the Fortran (Iter306, all
57 levels, max diff 0.0), so rico's residual is NOT the grid (it's the precip-onset-timing FP — see BLOCKED_CASES).
**Stale claims corrected (Iter306): dycoms2_rf02_do is `grid_type=1` (NOT 2), `_so` is `grid_type=3`.** **A bug that
vanishes when a case is switched to `grid_type=1` is a stretched-grid handling bug** — the namelist-swap A/B is still a
key localiser; future grid_type=2/3 cases must be tested explicitly.

**f2py IS usable (Iter147) — the `.so` is callable directly.** The `clubb_python` Python *wrapper*
(`advance_xm_wpxp.py`) is out of sync with its compiled `clubb_f2py.*.so` (it passes `wp3, kh_zt` where the
`.so` now expects `wp3_on_wp2, wp3_on_wp2_zt, kh_zt`), so `test_call_tree_advance_xm_wpxp` raises a TypeError.
But `clubb_f2py.f2py_advance_xm_wpxp.__doc__` gives the exact `.so` signature, so the routine can be called
**directly** (bypassing the broken wrapper) for the definitive **input-matched comparison**: push the UDTs via
`clubb_api.set_fortran_{grid,nu_vert_res_dep,implicit_coefs,err_info}` (see `tests/test_call_tree_advance_xm_wpxp.py`
for the grid/UDT setup), capture rico's matched step-1 `advance_xm_wpxp_jax` inputs (eager), call the `.so` with
the introspected arg order, and diff the Fortran vs JAX outputs bit-to-bit. This splits "an input diverges" from
"the assembly/solve differs" — the one test the namelist A/B can't do.

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
| `pdf_utilities.py` | `pdf_utilities.F90` | `mean_L2N`, `stdev_L2N` (lognormal→normal moments) — **bit-to-bit vs f2py** (rel 0.0); `corr_NL2NN`, `corr_LL2NN` (corr→normal) — vs Monte-Carlo (Iter109); `corr_NN2NL`, `corr_NN2LL` (the inverses, normal→linear), `calc_corr_chi_x`/`calc_corr_eta_x` (corr(chi/eta,x) from corr(rt,x)/corr(thl,x)) — **bit-to-bit vs f2py**; `calc_corr_rt_x`/`calc_corr_thl_x` (the inverses) — round-trip exact (Iter118-119). The hydrometeor/Ncn-PDF inputs to the KK rate functions |
| `precipitation_fraction.py` | `precipitation_fraction.F90` | `precip_fraction` — overall (downward cumulative-max) + per-component (`component_precip_frac_specify`, upsilon split, the fixed `precip_frac_calc_type=2`) precipitation fraction + max_hm limiter (Iter115). **Bit-exact** vs the rico stats oracle on the well-resolved precip region (pf 0, pf1/pf2 ~1e-17). f2py wrapper FPE-traps in this env → stats oracle used |
| `setup_clubb_pdf_params.py` | `setup_clubb_pdf_params.F90` | `calc_comp_mu_sigma_hm` — in-precip component means/stdevs (mu_hm_1/2, sigma_hm_1/2) of a precipitating hydrometeor via a mean+variance-preserving quadratic solve (omicron/zeta, emergency bounds, 4 branches) (Iter116). Verified via its **mean+variance-preservation contract** (machine-eps) — not f2py-exposed, and the stats oracle is unusable (its stored inputs/outputs aren't within-step consistent: the defining mean identity fails in the stats). `compute_mean_stdev` + `norm_transform_mean_stdev` (Iter131) — the `setup_pdf_parameters` orchestration that stacks the per-PDF-variable component moments into the `(ngrdcol,nzt,pdf_dim)` arrays the rate functions index (iiPDF order [chi, eta, w, Ncn, hydrometeors]; rico KK pdf_dim=6 = [chi,eta,w,Ncn,rr,Nr]) and transforms the lognormal vars (Ncn + hydrometeors) to normal/log space (mean_L2N/stdev_L2N). The `kk_microphys_driver` now assembles the rr/Nr linear+log moments through these instead of inline calc_comp + _hm_log_moments — verified **bit-identical** (rico oracle unchanged: auto 4.7e-7, accr 6.1e-9, evap 2.7e-6) and differentiable. The below-tolerance `-huge` sentinel is replaced by a finite mu floor (differentiability convention). `tests/test_calc_comp_mu_sigma_hm.py` |
| `Nc_Ncn_eqns.py` | `Nc_Ncn_eqns.F90` | `Nc_in_cloud_to_Ncnm` (+ `Ncm_to_Ncnm`, `bivar_Ncnm_eqn_comp`) — cloud-nuclei mean <Ncn> (the autoconversion mu_Ncn input) from the in-cloud <Nc> and the chi PDF, via the erfc PDF integral (Iter117). **Bit-to-bit vs f2py** (worst rel 2.4e-14, erfc-impl level) over both branches; reproduces rico Ncnm exactly |
| `corr_varnce_module.py` | `corr_varnce_module.F90` | `set_corr_arrays_to_default` (Iter132) — builds the prescribed in-cloud/below-cloud normal-space PDF-variable correlation arrays from the fixed 12×12 default tables (`corr_array_n_{cloud,below}_def`), via the column-major (`order='F'`) reshape that reproduces Fortran storage. For the KK PDF [chi,eta,w,Ncn,rr,Nr] the 6×6 block matches the hand-extracted Fortran values; the KK driver now DERIVES corr(chi,rr)=0.788/corr(chi,Nr)=0.675/corr(rr,Nr)=0.821 from it (`kk_prescribed_correlations`) instead of hardcoding — rico oracle bit-identical. cloud==below for all rate entries (differ only in chi-eta). The SILHS-oriented `calc_corr_norm_and_cholesky_factor` adjustments + Cholesky are deferred (not needed by the rate means). `init_pdf_hydromet_arrays` + `HmMetadata` + `kk_hm_metadata` (Iter135) — the hydrometeor-PDF metadata setup (oracle `init_pdf_hydromet_arrays_api`:455): per-hydrometeor names/tolerances/`l_mix_rat_hm`/`l_frozen_hm`, the in-precip variance ratio (intrcpt+slope·max(dx,dy); rico override slope=0/intrcpt=1.25 -> 1.25), and the PDF-variable indices (0-based: iiPDF_rr=4, iiPDF_Nr=5, pdf_dim=6, matching `setup_clubb_pdf_params.IIPDF_NCN+1`). Verified vs rico's known KK config. The data structure the hydrometeor advance + setup_pdf_parameters consume. `tests/test_corr_varnce.py` |
| `Microphys/KK_microphys/KK_utilities.py` | `KK_microphys/KK_utilities.F90` + `KK_microphys_module.F90:1177` | `G_T_p` (drop-growth coefficient G(T,p), Rogers&Yau) + `kk_evap_coef` (= 3 C_evap G_T_p ((4/3)π rho_lw)^(2/3) (1+Beta_Tl r_sl)/r_sl) (Iter114). Validated via the rico rrm_evap oracle (magnitude correct, T_liq=thlm·exner) |
| `fill_holes.py` | `fill_holes.F90` | `fill_holes_vertical`, `fill_holes_wp2_from_horz_tke` — machine epsilon |
| `clip_explicit.py` | `clip_explicit.F90` | `clip_variance`, `clip_skewness`, `clip_covar`, `clip_rcm`, `clip_covars_denom` — bit-exact |
| `adg1_adg2_3d_luhar_pdf.py` | `adg1_adg2_3d_luhar_pdf.F90` + `pdf_closure_module.F90` | Full ADG1 PDF closure — machine epsilon |
| `mixing_length.py` | `mixing_length.F90` | `diagnose_lscale_from_tau` + `compute_mixing_length` (Golaz 2002 nonlocal parcel) — machine epsilon |
| `saturation.py` | `saturation.F90` | `sat_mixrat_liq` (Flatau/Bolton), `rcm_sat_adj` (bisection) — machine epsilon |
| `sponge_layer_damping.py` | `sponge_layer_damping.F90` | `initialize_tau_sponge_damp` + `sponge_damp_xm` (xm fields rtm/thlm/uv) — wired into `advance_clubb_core`; ekman means bit-faithful. xp2/xp3 sponge not yet ported |
| `Microphys/cloud_sed_module.py` | `Microphys/cloud_sed_module.F90` | `cloud_drop_sed` (Stokes-regime cloud-droplet sedimentation, `l_cloud_sed`) — bit-faithful (`sed_rcm` ~1e-11); wired into the driver loop. Unblocked atex_long + dycoms2_rf02_so (Iter100) |
| `Microphys/KK_microphys/kk_microphys_driver.py` | (assembly) + `KK_microphys_module.F90:1196` | `kk_autoconversion_mean` (Iter120) + `kk_accretion_mean`/`kk_evaporation_mean` (Iter121) — the three KK mass-tendency entry points (validated vs rico auto 4.7e-7, accr 6.1e-9, evap 3.3e-6); `kk_microphys_adjust` (Iter124) — the tendency assembly: rates → (rrm_mc, Nrm_mc, rvm_mc, rcm_mc, thlm_mc) with the source/evap over-depletion limiters (validated vs rico: rcm_mc exact incl. source adj; thlm_mc self-consistent); `compute_kk_microphysics` (Iter125) — the FULL standalone step (hydromet fields + PDF state → tendencies), composing the in-precip moments + Ncnm + all rates + adjust. Runs; no-rain rcm_mc machine-exact vs rico (accr/evap-from-fields await a running rico — timing confound) |
| `Microphys/KK_microphys/KK_Nrm_tendencies.py` | `KK_microphys/KK_Nrm_tendencies.F90` | `KK_Nrm_auto_mean` (= rrm_auto/((4/3)π·rho_lw·r_0³), r_0=25µm), `KK_Nrm_evap_local_mean` (KK00 Eq.23, ν=1) (Iter124); `KK_Nrm_evap_upscaled_mean` (Iter125 — reuses `trivar_NLL_mean_eq` with exps 1,−2/3,5/3; validated vs rico Nrm_evap median 3.2e-6). **All KK rates now ported+validated** (rrm auto/accr/evap, Nrm auto/evap, mvr) |
| `kk_microphys_driver.py::kk_sedimentation` | `KK_microphys_module.F90:1542` | KK mean sedimentation velocities Vrr (rain mass) / VNr (rain number) from the mean volume radius (KK00 Eq.37): `Vrr=-(0.012·mvr_µm−0.2)`, `VNr=-(0.007·mvr_µm−0.1)`, clipped ≤0, top-level zero-flux BC (Iter133). **Bit-exact vs the rico oracle** — fed the Fortran's own `mvrr` through `kk_sedimentation` then the bit-faithful `zt2zm` (the Fortran stores `Vrr=zt2zm(hydromet_vel_zt)` on momentum levels), matches the stored `Vrr`/`VNr` to \|Δ\|max 1.1e-16 on the rain points. Validated against a 250-step rico run (`rico_long_fort`) where rain develops; differentiable. This is the V_hm sedimentation-velocity input `advance_hydrometeor` needs |
| `Microphys/advance_microphys_module.py` | `advance_microphys_module.F90` | `sed_centered_diff_lhs` + `lhs_budget_term` (Iter136) — the implicit, centered-difference MEAN-SEDIMENTATION transport operator (oracle :2188) for the d<hm>/dt equation, returning the 3 LHS bands (super/main/sub), + the generic `-lhs·hmm` budget tendency. Flux-form: <hm> interpolated to momentum levels (zt2zm weights), ×rho_ds_zm·<V_hm>, differenced over the central zt level; surface hm = zt-level-0 value (no weight), top is no-flux (super=0). The turbulent-/mean-advection LHS reuse the bit-faithful `diffusion_zt_lhs`/`term_ma_zt_lhs`. **Verified two ways:** (i) the rigorous oracle-free CONSERVATION CONTRACT — column-mass-weighted Σ tendency = surface flux rho_ds_zm[0]·V[0]·hm[0] — to **5.6e-15**; (ii) exactly reproduces rico's `rrm_sd` budget (=0) at the 14k developed-rain points where the KK velocity clips to 0. (The active-sedimentation points sit at marginal rrm → the within-step/end-of-step timing confound, same as accr/evap.) `term_turb_sed_lhs` (Iter137) — the TURBULENT-sedimentation implicit LHS (oracle :2683, centered branch): verified branch-by-branch IDENTICAL to `sed_centered_diff_lhs` with the momentum-level implicit covariance `Vhmphmp_impc` (= zt2zm of `kk_sed_vel_covars`' Vrrprrp_impc) replacing the velocity, so it delegates to it. The full composition `kk_sed_vel_covars → zt2zm → term_turb_sed_lhs` satisfies the conservation contract to **3.1e-15** on the real rico Vhmphmp_impc. `microphys_lhs` (Iter138) — assembles the FULL implicit LHS tridiagonal (oracle :1564): `1/dt + ½·diffusion_zt_lhs(K_hm,nu) (+ the k=1 lower-BC re-set, identical to its own bottom row) + term_ma_zt_lhs(wm_zt, upwind) + sed_centered_diff_lhs(V_hm) + term_turb_sed_lhs(Vhmphmp_impc)`, reusing the bit-faithful `diffusion_zt_lhs_jax`/`term_ma_zt_lhs_jax`. **Verified:** the assembly == the component sum (bit-exact band/sign/1-dt bookkeeping); the turb-adv (eddy-diffusion) part conserves mass to 1.3e-23; the mean-advection budget reproduces rico `rrm_ma`/`Nrm_ma` to **9.9e-14** at robust-rrm points (clean `stats_update` oracle). `term_turb_sed_rhs` + `microphys_rhs` + `advance_one_hydrometeor` (Iter139) — the EXPLICIT RHS (`hmm/dt + microphysics source − ½·diffusion·hmm [Crank-Nicholson explicit] + term_turb_sed_rhs [flux-divergence of ρ_ds·Vhmphmp_expc]`) and the capstone one-step advance (assemble LHS+RHS, `tridiag_lu_solve`). **Verified:** the FULL turbulent-sed tendency (explicit `term_turb_sed_rhs` + implicit `−term_turb_sed_lhs·hmm`) reproduces rico `rrm_ts` to median **4.5e-11** (95% < 1e-6) — confirming `rrm_ts` = explicit+implicit; `microphys_rhs` == its component sum; `term_turb_sed_rhs` conserves; `advance_one_hydrometeor` round-trips `lhs·soln == rhs` (<1e-18) and physically removes rain mass via sedimentation. **The per-hydrometeor transport solve is now COMPLETE** (minus `fill_holes`, already ported, + the multi-hydrometeor orchestration). `calculate_K_hm` (Iter140) — the hydrometeor eddy diffusivity (oracle :3236, `l_use_non_local_diff_fac=.false.`): `c_K_hm·Kh_zm·(√hydrometp2/max(zt2zm(hm),tol))·(1+|Skw_zm|)`, capped so `|corr(w,hm)|≤1`, K=0 at boundaries. Consumes the WITHIN-step hydrometeor field so the in-loop `K_hm_rr` stat is timing-confounded (≈2%); verified instead by exact formula transcription (vs a hand reference reusing the bit-faithful zt2zm/ddzt, incl. cap + BC) + differentiability. `tests/test_kk_rico_oracle.py` |
| `Microphys/KK_microphys/KK_upscaled_turbulent_sed.py` | `KK_microphys/KK_upscaled_turbulent_sed.F90` | `kk_sed_vel_covars` (Iter134) — the rain sed-velocity covariances `<V_rr'r_r'>=0.012·10⁶·<r_r'R_vr'>`, `<V_Nr'N_r'>=0.007·10⁶·<N_r'R_vr'>`, written semi-implicitly as `coefA·<x>+termB` (impc/expc). The `<x'R_vr'>` are bivariate-lognormal covariances: vs `bivar_LL_mean` the x1-variance term carries (α²+2α) and the cross term (α+1) (the extra differenced-variable factor); 4-way variance dispatch reusing the ported `bivar_LL_mean_const_all/x1`; the Nr side reuses the rr machinery with rr↔Nr + exponents swapped. exps α=1/3, β=−1/3. **Bit-faithful-to-the-gate vs the rico oracle** `rr_KK_mvr_covar_zt`/`Nr_KK_mvr_covar_zt` (rel **4.5e-11**), with NO timing confound (overall means reconstructed within-step as `a·f_p1·μ_rr_1+…`); differentiable. Feeds the sed-turbulence LHS (`term_turb_sed_lhs`) in advance_microphys. `tests/test_kk_rico_oracle.py` |
| `Microphys/KK_microphys/{parabolic_cylinder,PDF_integrals_means,KK_upscaled_means}.py` | `Microphys/KK_microphys/{KK_utilities,PDF_integrals_means,KK_upscaled_means}.F90` + `parameters_KK.F90` | **Complete upscaled-KK analytic means library** (Iter108/110/111/112) — all 4 upscaled means: D_v parabolic cylinder fn (1F1 series + optimally-truncated asymptotic); `bivar_NL_mean`(+3 const)/`bivar_NL_mean_eq` for auto/accr; `trivar_NLL_mean`(+5 const)/`trivar_NLL_mean_eq` (8-way) for evap (chi×r_r×N_r, chi<0 half); `bivar_LL_mean`(+2 const)/`bivar_LL_mean_eq` for mvr (r_r,N_r both lognormal); wrappers `KK_auto/accr/evap/mvr_upscaled_mean`. Verified vs scipy.pbdv + brute-force quadrature (NL 7e-15, accr 1.3e-11, trivar 3.3e-11, LL 1.5e-15); all differentiable. NOT yet wired (needs hydromet PDF moments first). See Remaining Work §KK |
| `Microphys/Morrison_microphys/module_mp_graupel.py` | `Microphys/Morrison_microphys/module_mp_graupel.F90` | **Morrison 2-moment port — STARTED Iter190-191.** `polysvp_jax` (saturation vapor pressure, Flatau 1992 "V1.7" coeffs; distinct from CLUBB-core's Flatau fit) — bit-exact (rel 0.0) vs a Fortran-Horner replica, physically vs Goff-Gratch (<2e-3). `derf1_jax` (erf, Ooura table; array-capable, per-element block gather) — vs `scipy.special.erf` to **2.2e-16**. `gamma_jax` (Γ via Cody, all 4 branches incl. negative-arg reflection + integer reduction + Stirling) — vs `scipy.special.gamma` to **7.6e-15**. **Special-function layer DONE (3/3).** `kk_warm_rain_rates` (Iter193-194) — KK(2000) BULK autoconversion PRC + accretion PRA + the number companions (IRAIN=0 default; distinct from the CLUBB KK scheme's upscaled rates), validated vs the nov11_altocu oracle: **PRC/NPRC/NPRC1 median ~2e-7** (bit-faithful), PRA/NPRA median ~4% (qr timing-confound). `rain_slope`/`cloud_slope` (Iter194) — the gamma-distribution slopes LAMR/N0RR + PGAM/LAMC (via gamma_jax). `rain_evap_rate` (PRE, Iter195) — full Rutledge-Hobbs ventilated diffusion, vs oracle median 7.1%. **WARM-RAIN COMPLETE.** `_gamma_slope`+`ice/snow/graupel_slope` (Iter196, generic LAM=(ρπ·n/q)^⅓). `ice_deposition` (Iter196) — full Harrington-1995 ice/snow/graupel vapor deposition+sublimation, vs oracle PRD 1.5%/PRDS 2.4%/EPRD 3.6%/EPRDS 4.0%. `snow_collection_rates`+`ice_autoconv_to_snow` (Iter197) — PSACWS/PRAI (5.4%/4.7%) + PRCI (1.3%). `snow_self_aggregation`+`deposition_nucleation` (Iter198) — NSAGG (8.3%) + MNUCCD Cooper (10.2%). `rain_immersion_freezing`+`sublimation_number_rates` (Iter199) — MNUCCR Bigg (4.2%) + NSUBI/NSUBS (0.2%/1.3%). `cloud_contact_immersion_freezing` (MNUCCC, Iter200) — bit-exact (8.4e-6). `rain_self_collection` (NRAGG, Iter201, 10.4%) + number companions NPSACWS/NPRAI/NPRCI validated. `tests/test_morrison_{special,rates}.py` (9+19). Morrison runs float64. Entry=`M2005MICRO_GRAUPEL`. **DRIVER GLUE COMPLETE (Iter205-209):** conservation limiters (`conserve_qc/qi/qr/qni`), `saturation_adjustment_pcc` (PCC), `to_in_cloud`/`tendency_to_grid_mean`/`neg_fix_number` (the CF3D subgrid conversion), `rain_fall_speed` (UMR/UNR), and `rain_sedimentation` (the CFL sub-stepped upwind flux-divergence loop + downward fall-speed propagation, `lax.scan`+`lax.fori_loop`, all differentiable). **★ Grid-orientation convention (CRITICAL):** the CLUBB↔M2005 interface FLIPS the vertical index (`microphysics.F90:1944 m=nz-k`) — M2005's KTE (its "top of model") maps to the JAX grid's **surface**. So in JAX-grid order (index 0=surface, nzt-1=top) the sedimentation has "above k"=k+1, the top cell (-1) takes no inflow, and rain exits at index 0; the literal Fortran index transcription would send rain UPward. Validated by the conservation contract (no stored fall-flux in the oracle): ρ-weighted column conserved aloft, centroid descends, surface outflux when the blob reaches the bottom. **Known jit constraint:** NSTEP=`int(max(RGVM·dt/dz+1,1))` is data-dependent → fine eagerly; a jitted driver needs fixed-max NSTEP + masking or `lax.while_loop`. **M2005 STEP ASSEMBLY (Iter210-211):** `m2005_cold_tendencies` (cold T<273.15, "CONSERVATION OF WATER" limiters QC/QI/QR/QNI/QG :3801-3960 + the 12 mass/number tendency assignments :3963-4007, the ~30 ice+rain rates; IGRAUP 0/1) **and** `m2005_warm_tendencies` (warm T≥273.15, :2318-2440 — no ice growth: melting PSMLT/PGMLT + EVPMS/EVPMG + warm rain + the number-melting sub-calcs). A column selects per-level via the T mask. **★ Both verified by the WATER-CONSERVATION CONTRACT** (a new oracle-free convention): every rate is a +source/−sink pair across exactly two species → with pcc=0 the six mass tendencies sum to **exactly 0 (1.4e-20 cold / 3.4e-21 warm)**; any sign/term transcription error breaks it (cold IGRAUP=1 leaves only PRACI unpartnered, the Fortran's faithful route-to-nonexistent-graupel). `m2005_step_tendencies` (Iter212) composes both: select branch by `T>=273.15` then apply the post-assembly PCC saturation adjustment (`qv−=PCC, T+=PCC·Lv/Cp, qc+=PCC`); verified on a MIXED-T column (Σ mass tendencies 7.2e-21). **★ Thermo in the CLUBB build is CONSTANT** (XXLV=Lv, XXLS=Ls=2.834e6, CPM=Cp, XLF=Ls−Lv — the `#ifdef CLUBB` branch :1541; the T-dependent WRF forms are disabled), so the constant-based PCC is faithful. **The M2005 single-column tendency step is COMPLETE.** `compute_m2005_rates` (Iter213) is the rate ORCHESTRATION keystone — composes the ported rate functions in the faithful dependency order (deposition nucleation → ice deposition → evap → warm rain → collection → numbers) into the dict the assembly consumes; verified end-to-end by the water-conservation contract (`compute_rates → m2005_cold_tendencies` conserves to 4.96e-24). The full driver chain is now **in-cloud fields → `compute_m2005_rates` → `m2005_step_tendencies` → ×CF3D (`tendency_to_grid_mean`) → integrate**. **Minor collection rates (Iter214):** oracle-scoped which are nonzero for nov11 (purely cold, no graupel/melting) → only 4: `cloud_ice_collect_droplets` (PSACWI), `rain_ice_collision_snow` (PIACRS/PRACIS), `rain_accrete_snow` (PRACS). Validated PSACWI 1.4%/PIACRS 4.3%/PRACIS 5.5% vs oracle; PRACS (∝rain×snow, double-confounded) by Fortran-replica transcription + hand-calc. **★ The top-level T-split is `IF(T3D≥TMELT)@1806…ELSE@2766`** — PSACWI/PIACRS/PRACIS are cold (wired into `compute_m2005_rates`); **PRACS is WARM-branch (:2103)** so 0 at cold levels, deferred to the warm orchestration. **The nov11 cold-branch rate set is COMPLETE.** Still-zero-for-nov11: QMULT*, graupel block, warm melting (port when a case needs them). **★ `m2005_driver` (Iter215) is the COMPLETE callable single-column step** — saturation → in-cloud (÷CF3D) → `compute_m2005_rates` → `m2005_step_tendencies` → ×CF3D; **runs end-to-end on the real nov11 stored fields (6200 pts): finite, water-conserving to 8.97e-24, physical.** PRACS added warm-gated (`where(T>=TMELT,…,0)`). **New convention:** a rate with a negative-power number dependence (PRC ∝ nc^-1.79) must guard number>0 — the in-cloud ÷CF3D edge produces degenerate qc>0/nc=0 points the per-column Fortran avoids (caught a real inf). **Sedimentation (Iter216):** `ice_fall_speed`/`snow_fall_speed` + `morrison_sedimentation` (rain+ice+snow, mass+number, the SHARED-NSTEP CFL coupling RGVM=max over all species:4749) — nov11 sediments rain+snow+ice (rsm_sd 2.87e-8 is the largest); conservation-contract verified (rain>snow>ice fall ordering). **★ CLUBB-Morrison INTERFACE (Iter217): `morrison_microphys_driver`** — the bridge: `m2005_driver` → post-process fields (DUM=field+ten·dt) → `morrison_sedimentation` → clip → `hydromet_mc=(field_final−field_initial)/dt` (the Iter204 form), rcm_mc/rvm_mc raw, `thlm_mc=(T_in_K2thlm(T_final,exner,rcm_final)−thlm)/dt`, thlm=(T−Lv/Cp·rcm)/exner. Field-update order: sed fall speeds use POST-process fields, sed is a column loop AFTER the ×CF3D exit (grid-mean). **Runs on a real nov11 rain-bearing column: all 12 *_mc finite, physical, every species evolves.** **★★ WIRED + RUNS (Iter218): nov11_altocu runs end-to-end in JAX with Morrison (10 steps stable; ARM still bit-faithful).** `morrison_hm_metadata` (8 fields, bulk → pdf_dim 4 via `l_hydromet_pdf=False`); `advance_morrison_microphysics` (Microphys/morrison_microphys_step.py — per-step driver call + first-pass Euler hydromet advance); gate + hydromet-setup + forcing + per-step call all gated by `microphys_scheme=='morrison'` (KK/none untouched). **Diagnosis (Iter219): nov11 STEP 1 is fully bit-faithful** (`compare_runs --case nov11_altocu`); step-2 seed was the **missing cloud-water sedimentation** (M2005 folds it into rcm_mc via `QC3DTEN += QCSTEN` :4885 — NOT the separate cloud_drop_sed). Added `cloud_fall_speed` (Stokes/viscosity) + cloud in `morrison_sedimentation`; validated 0.3% vs `rcm_sd_mg_morr`; rcm_mc median 71%→1.5%. **Iter220:** the cloud-sed outlier was phantom-cloud (rcm>0 & Ncm=0, ~40% of oracle cloudy points) → added the `nc>0` guard to `cloud_fall_speed` (cloud sed now matches `rcm_sd_mg_morr` exactly). **★ Fixed a real wiring bug: the JAX `Ncm` was 0 in the Morrison loop** (it's DIAGNOSED `Ncm = Nc_in_cloud·cloud_frac`, oracle ratio 1.0 — but was set AFTER the morrison call); now diagnosed inside `advance_morrison_microphysics` → matches Fortran (1.472e8). **Iter221: thlm_mc fix** — it must use the INPUT cloud water (rcm_r4 ≈ input rcm; M2005's QC3D is NOT integrated by QC3DTEN, only ÷CF3D/×CF3D + subsat cleanup, :416/:794), NOT the sed-updated rcm_f → thlm_mc=ten['T']/exner. **Halved the step-2 thlm error** (2.23e-2→1.07e-2, back to baseline). **Iter222-223:** confirmed `clubb_driver.F90:3337` does `rtm_forcing += rcm_mc + rvm_mc` (CF3D=liquid cloud_frac). **★ Solved the 184-strong-heating-point puzzle:** M2005 integrates the fields at the end (`QC3D+=QC3DTEN·dt`, `T3D+=T3DTEN·dt`, :4911-4929), so the faithful **`thlm_mc=(ten['T']−Lv/Cp·rcm_mc)/exner`** (rcm_mc=process+cloud sed); the PCC parts cancel, the cloud-sed rcm change remains = the strong heating at cloud-top mixed-phase (cloud sediments out → rcm_mc<0 → −Lv/Cp·rcm_mc>0). Validated: standalone thlm_mc matches the oracle EXACTLY at max (5.39e-4) + 0.5% at strong points. **Reverted the Iter221 input-rcm "fix" (a dead end).** Remaining: the live-state cloud-sed rcm_mc accuracy (faithful thlm_mc amplifies it at strong points → live compare thlm 2.23e-2). **Convention:** compare a running case's *_mc to the oracle via `-out_dir` + direct stats diff (per-step compare_runs samples *_mc mid-step → timing-confounded maxRef=0). **Iter224:** fixed the sedimentation `dzq = gr.dzm[:,1:]` (= delta_zt = dzm(k+1), the momentum-level spacing, NOT dzt; microphys_driver.F90:419) — matters only on stretched grids (nov11 grid_type=2). Cloud fall speed (UMC=0.011 m/s) + cloud sed verified correct. The step-2 seed is likely the **missing Morrison 2nd-moment source** `update_xp2_mc` (microphys_driver.F90:553, `l_morr_xp2_mc` → rtp2_mc/thlp2_mc/wprtp_mc/wpthlp_mc/rtpthlp_mc, the analog of the KK covar driver) — the failing step-2 fields are the 2nd moments. **Stats-timing:** the oracle writes Morrison *_mc with a one-record delay → compare overall distributions, not record-aligned. **Iter225: ★ isolated the nov11 step-2 seed.** Empirically: zeroing the cloud-sed part of rcm_mc drops the compare step-2 errors to rtm 5.33e-8 / thlm 1.06e-2 (baseline) — my cloud sed is the dominant error. **The oracle's microphysics rcm_mc is 0 for the FIRST 60 STEPS** (per-step stats, not averaged; the fresh compare Fortran also shows 0), activating only at step 60 when ICE forms — yet cloud is active the whole time (rcm grows 3.2e-4→4.3e-4 from dynamics). My driver fires from step 1 (too early). **Ruled out:** GOTO 200 subsat-skip (needs cloud<QSMALL too), LTRUE gate (cloud sets it), FC propagation (faithful), stats averaging, NSTEP, dzq. `update_xp2_mc` (2nd-moment source) is rain-evap-driven → 0 at step 1, not the seed. **★ SOLVED (Iter226): `microphys_start_time = 64800` in nov11's namelist** — the microphysics is SKIPPED for the first 60 steps (case starts 61200; Fortran gates at microphys_driver.F90:389 `if time_current < microphys_start_time return`). My driver fired from step 1 → spurious cloud sed = the whole rtm seed. Fixed: read `microphys_start_time` into state, gate the per-step morrison call on `time_current >= microphys_start_time` (gated to morrison, ARM/16 cases untouched). **rtm step-2 now PASSES.** **Convention:** a microphysics case diverging only after N steps where the oracle's rates are all 0 early = the `microphys_start_time` spinup signature, not a rate bug. **Iter227: the post-spinup nov11 seed is the SHORTWAVE radiation** (not the stretched grid). thlm budget decomposition → radht FAILS (rel 7%); radht_LW fine (0.27%), **radht_SW 71% off**. nov11 has `l_sw_radiation=.true.`, amu0=0.4329 (fixed via `l_fix_cos_solar_zen`); jun25 (faithful) has SW inactive → the SW path was untested. **Fixed a real cfg-key bug:** `sunray_sw` gets `radius=eff_drop_radius`, `A=alvdr` (radiation_module.F90:507) but the JAX read `radius`/`A_surface_albedo` (defaults). Confirmed amu0/Fs0/rho_lw/tau all match. **★ FIXED (Iter228): the SW single-scattering albedo cfg key** — the JAX read `omega_sw` (default 0.999) but the namelist key is `omega` (nov11: 0.9965); absorption ∝ (1−omega) → (1−0.999)/(1−0.9965)=1/3.5, exactly the absorption deficit. Read `cfg['omega']`. **★ nov11 STEP 2 IS NOW BIT-FAITHFUL** (0 prognostic failures, radht machine-precision). The omega+albedo+radius cfg-key fixes (Iter227-228) fully resolved the untested simplified-SW path. **Convention:** simplified-rad namelist keys are `eff_drop_radius`/`alvdr`/`omega` (NOT radius/A_surface_albedo/omega_sw) — cfg-key bugs invisible until an active-SW case; decompose `radht` into radht_LW/radht_SW. **★ nov11 is BIT-FAITHFUL THROUGH STEP 5 (Iter229-230)** — all fields machine-precision; diverges at **step 6** (the ice-cloud onset). **CORRECTION (Iter230): nov11 is `grid_type=1` (UNIFORM), NOT stretched** (`nzmax=176`, deltaz constant — like jun25/ARM); the Iter224/229 "stretched grid" notes were wrong (the dzq=dzm fix is still faithful, just a no-op on uniform). Same wp3/2nd-moment solve as jun25 (bit-faithful) → not a grid bug. **Real chain (opposite of the Iter229 wp3-first guess): `ice_supersat_frac` → `bv_mixed` (the in/out-of-cloud Brunt-Väisälä blend `bv_moist + exp(−bv_efold·isf)·(bv_dry−bv_moist)`, mixing_length.py:148 / advance_helper:250) → Lscale (1.2e-2) → tau → wp3/Skw/up2/vp2** (wp3 is DOWNSTREAM). Root = the ICE path (`l_ice_microphys=.true.`, jun25 lacks it): `ice_supersat_frac` (calc_ice_cloud_frac_component) first activates below-freezing at step 6. JAX port verified FAITHFUL (`_ice_cf60`, `sat_mixrat_ice_jax`, all constants T_FREEZE_K/FLATAU_ICE_A/EP/MIN_T_C exact). **★ CLOSED (Iter231): nov11 step 6 is FP-LIMITED (rico/coriolis class), NOT a bug.** Full-chain code review proved every link bit-faithful: the amplifier `brunt_freq_out_cloud = brunt_freq_pos·min(1,max(0, 1−isf/0.001))` (mixing_length.py:150 ↔ F90:2347-2350) matches exactly; `l_smooth_min_max=.false.` is a Fortran `parameter` (so the JAX hard min/max is right); the ice_supersat_frac/sat_mixrat_ice/constants are faithful (Iter230). The `/0.001` ramp is a 1000× amplifier (d(fac)/d(isf)=−1000): TWO faithful stages compound — the ice_supersat_frac erf at near-zero scalar variance (machine-eps→4.5e-5) THEN the ramp (4.5e-5→4.5e-2) → invrs_tau_wp2_zm → up2/vp2. Seed = cancellation-amplified FP floor in the tiny scalar variance (rtp2 ~4.5e-11) at the ice-cloud EDGE (JAX/Fortran agree on cloud structure — no discrete edge mismatch). **No faithful fix exists** (changing the ramp/erf would reduce faithfulness). **Milestone: the Morrison+ice+SW-radiation case is faithful up to the FP floor.** The hydromet transport (microphysics activates at step 60) can't be bit-validated in the full run (step-6 FP precedes it) → relies on the rate/transport unit tests. nov11 joins rico/coriolis in §FP-limited cases. **★ WARM Morrison transport test (Iter232): `dycoms2_rf02_morr`** (`l_ice_microphys=.false.`, active from step 1, grid_type=2) is the case for validating the hydromet TRANSPORT (nov11's FP-limit blocks it). Step 1 dynamics+cloud+radiation are bit-faithful, but the warm RAIN is ~1/17 of the Fortran. **★ FIXED (Iter233): a sedimentation AXIS bug** — `_sediment`/`_fall_speed_propagate` indexed the vertical on axis 0, but the run passes `(ngrdcol, nzt)` (vertical=axis 1); for ngrdcol=1 the interior-flux term collapsed and every cell got the top-outflow formula → **94% of the rain mass destroyed**. (1-D `(nzt,)` columns conserve, so the unit tests passed; nov11 never sedimented in a full run — gated off until step 60, FP-limited at step 6.) Fix: both helpers now act on the LAST axis. **Result: 2-D conserves to 1.0000; dycoms step-1 rain matches the Fortran (sum 99.6%, was 1/19); rcm/cloud_frac/radht step-2 errors fell from 20%/26%/10% → ~6e-5.** **★ FIXED (Iter235): the missing rain-number size limiter SIZEFIX_NR** (F90:1881-1892) — after sedimentation (mass falls faster than number, UMR>UNR), some levels had LAMR>LAMMAXR (drops down to 4 µm); the Fortran resets `NR=LAMMAXR³·QR/(π·ρw)` so no drop < 20 µm, the JAX didn't → Nrm ~19% high. Added `_sizefix_rain_number` on the post-sed `Nrm_f`. **Result: dycoms step-1 rrm AND Nrm match the Fortran to 99.6%.** **★ Remaining residual DIAGNOSED (Iter236): the cloud-top deficit is the MISSING CLUBB hydrometeor transport** — the Fortran rrm budget shows that above the cloud (no auto/sed) the rain is entirely `rrm_ta` (turbulent advection), and at the cloud top `rrm_auto` and `rrm_sd_morr` cancel so `rrm_bt ≈ rrm_ta` — the cloud-top rain is sustained by transport. The JAX applies only the microphysics `*_mc` (auto+sed)+Euler, NOT the CLUBB mean+turbulent transport (advance_xm_wpxp) of the hydrometeors. A real architectural gap (the 16 faithful cases are non-precipitating; nov11 FP-limited before microphysics; dycoms is the first to need it). **NEXT MAJOR PIECE: the hydrometeor transport** (`advance_microphys_module.F90` → `advance_hydrometeor` + `microphys_lhs` implicit solve; NOT advance_xm_wpxp). **Adversarial review (Iter237): the turbulent advection is a DOWN-GRADIENT diffusion `<w'hm'>=−K·d<hm>/dz`** (`calculate_K_hm`, :3234), empirically `wprrp=−0.75·d(rrm)/dz` exactly on dycoms (K=`c_K_hm`=0.75), zeroed above cloud top (`l_prevent_hm_ta_above_cloud`). **★ RESOLVED (Iter238): the effective diffusivity is `K_hm + nu_hm`** (F90:914) — `nu_hm`=1.5 m²/s background × the grid factor mult_factor_zt (=0.5 for dycoms → 0.75); K_hm (variable, =0 when no hydrometeor variance, so step-1 K=nu_hm=0.75). Faithful chain: `hydrometp2 = ratio_hmp2_on_hmm2·hm²` (ratio=hydrometp2_in/hm_in²) → `K_hm = c_K_hm·Kh_zm·√ratio·(1+|Skw|)` capped at the corr limit → `wphydrometp = −(K_hm+nu_hm)·d(hm)/dz` (Crank-Nicholson), zeroed above cloud top. **★ Iter239: the transport machinery ALREADY EXISTS** — `advance_microphys_module.py` (`calculate_K_hm`, `microphys_lhs/rhs`, `advance_one_hydrometeor`), bit-faithful on rico, used by the KK path. **Wired Morrison through it** (l_sed=False, nu_hm from state). **★ Iter240: fixed K_hm** — the variance is `hydrometp2 = ((ratio+1)/precip_frac−1)·hm²` (`_hydrometp2_zt`, setup_clubb_pdf_params:449); bulk → precip_frac≤0 → pf=1 → `hydrometp2=ratio·hm²=1.25·hm²` (ratio=1.25 rr/Nr) — NOT 0 (the stored rrp2=0 is a different quantity). `K_hm=calculate_K_hm(.., hydrometp2)` (large, capped). Result: dycoms cloud-top rrm 1/9–1/400 → **0.16–0.79**; bulk total 90.3% → **94.1%** (rrm)/96.3% (Nrm). **Iter241: confirmed precip_frac=1.0 (bulk, setup_clubb_pdf_params:400) → hydrometp2=1.25·hm² IS right; l_sed=False IS faithful (Fortran does Morrison sed in M2005, rrm_sd=0). Fixed the hydrometp2 placement to `1.25·zt2zm(hm)²` (advance_microphys:907; was zt2zm(hm²)).** **★★ dycoms FIXED to ~100% (Iter246):** the "irreducible residual" was a real 2× K_hm bug — the transport precip_frac was wrong (used the DEFAULT 1.0 instead of the value `precipitation_fraction`'s max-in-precip-mean limiter computes, ~0.33). `hydrometp2 = ((ratio_ip+1)/precip_frac − 1)·hm²` so precip_frac=0.33 → hydrometp2≈5·hm² (not 1.25·hm²) → K_hm doubles. Fix: compute the real precip_frac (the KK function) from the PDF fields + `hmp2_ip_on_hmm2_ip`. **Result: rrm total 94%→100.6%, max-level rel 0.40→0.099, cloud-top 0.51→0.97.** Dynamics bit-faithful throughout. **★ VERIFIED bit-exact (Iter249): the dycoms K_hm transport matches the oracle's stored `K_hm_rr`/`K_hm_Nr` (advance_microphys:310-311) to ratio 1.000 at every interior level** (only the rain-bottom level straddles the eps cap → branch-flip). So calculate_K_hm + the precip_frac/hydrometp2 computation are FAITHFUL; the prior "very-top overshoot / rel 0.099" was STALE (pre-Iter248). Current step-1 rrm is faithful to 0.985–1.005 across the rain layer; the sub-tol tails (<tol) cause no feedback. **The true residual is the Morrison SEDIMENTATION at the sharp rain-top:** the JAX `rrm_mc` source matches <0.1% in the lower rain but its negative (sed-removal) tendency is ~1.8% too weak at the sharp upper edge (first-order upwind flux where the fall speed ∝(qr/nr)^⅓ is FP-sensitive) → upper rain ~0.4% high, accumulating to 16 prognostic failures by step 15. dycoms is transport-faithful but **sharp-edge-sed-discretization-limited (FP class)** — NOT the transport, NOT the 2nd-moment source (`l_morr_xp2_mc=.false.`, correctly off in both — dycoms does not override the default), NOT RGVM/NSTEP. **★ Convention: the oracle WRITES K_hm to stats (`K_hm_<hm>`) — use it to bit-validate the hydrometeor diffusivity directly; the stored `<hm>p2`/`rrp2` is the POST-advance hydrometp2 (a timing confound), NOT the pre-advance value fed to K_hm.** **★ LESSON (convention): NEVER trust a default-vs-computed value — a Fortran line that SETS x=default may be overwritten later; verify the actual computed quantity. An unverified precip_frac=1 cost 12 iterations of "deep residual" chasing a 2× K_hm bug.** **★ DONE (Iter250): the pre-rate slope clamps for ALL 5 species.** The Fortran clamps every gamma slope PRE-RATE (warm 1881-2002 / cold 2816-2968) — resetting NR3D/NC3D/NI3D/NS3D/NG3D so the rates use in-bounds numbers; NO post-sed clamp (:4509 only scales the diagnostic stat). Added `_sizefix_exp_number` (exponential species, N=LAM³·Q/CONS) + `_sizefix_cloud_number` (gamma form) + `_size_clamp_numbers`, wired pre-rate into `m2005_driver` (reset folded into the number tendencies). **The rain post-sed clamp is KEPT too** — the stored stats reflect the Fortran's NEXT-step pre-rate clamp on the unclamped post-sed output, so a per-step driver must clamp its OUTPUT to match (removing it regresses dycoms Nrm 1.0→1.15). No-op for dycoms (slopes in-bounds); `test_size_slope_clamps`. **★ dycoms CLOSED as FP/discretization-limited (joins rico/coriolis/nov11):** thorough adversarial review confirmed the Morrison sed faithful — upwind stencil (cell k gains falout[k+1], loses falout[k]; top loses only, F90:4767-4860), `dzq=dzm[:,1:]`=delta_zt(k)=dzm(k+1) (faithful on the stretched grid; dycoms grid_type=2 is the FIRST case to exercise non-constant dzq), RGVM/NSTEP, fall-speed propagation, size clamps (don't fire). The ~1.8% residual at the sharp rain-top is the first-order upwind flux where fall speed ∝(qr/nr)^⅓ is FP-sensitive — irreducible. K_hm transport VERIFIED bit-exact. dycoms is transport-faithful, ~100% by mass, NOT bit-faithful. **New test strategy (generalize beyond ARM/nov11): for Morrison transport, run a WARM active-from-step-1 case (dycoms2_rf02_morr) + check sedimentation ∫ρ·q·dz conservation on a 2-D (ngrdcol, nzt) column. Transport/sedimentation helpers MUST act on the vertical=last axis. Bit-validate K_hm against the oracle's stored `K_hm_<hm>` directly. Slope clamps are PRE-RATE (affect rate inputs), not post-sed.** |
| `T_in_K_module.py` | `T_in_K_module.F90` | `calculate_thvm` — bit-exact |
| `calc_pressure.py` | `calc_pressure.F90` + `hydrostatic_module.F90` | `hydrostatic`, `init_pressure` via `jax.lax.scan` |
| `parameters_tunable.py` | `parameters_tunable.F90` | `init_clubb_params`, `calc_derrived_params` — bit-exact |
| `model_flags.py` | `model_flags.F90` | `get_default_config_flags` — all 88 flags |
| `numerical_check.py` | `numerical_check.F90` | `parameterization_check`, `check_clubb_settings`, `check_parameters` |
| `Benchmark_cases/arm.py` | `arm.F90`, `prescribe_forcings.F90`, `time_dependent_input.F90`, `sfc_flux.F90`, `diag_ustar_module.F90` | Full ARM forcing (Monin-Obukhov, time-interpolated) |
| `io/stats_writer.py` | `stats_netcdf.F90` | Pure Python NetCDF stats output (StatsWriter) — bit-exact |
| `advance_clubb_core_module.py` | `advance_clubb_core_module.F90` | Full ARM timestep — **zero Fortran calls** |
| `src/clubb_standalone.py` | `clubb_standalone.F90` | Case initialization — **zero Fortran API imports** |
| `src/derived_types/` | `clubb_python_api/clubb_python/derived_types/` | Pure-Python mirrors: ConfigFlags, ErrInfo, SclrIdx, Grid, pdf_parameter, implicit_coefs_terms |
| `src/Radiation/radiation.py` | `radiation_module.F90`, `cos_solar_zen_module.F90`, `rad_lwsw_module.F90` | `cos_solar_zen`, `sunray_sw`, `simple_rad` — **zero Fortran imports** |
| `advance_clubb_to_end.py` | (orchestration) | Stats + forcing loop — **zero module-level Fortran imports**; `prescribe_forcings` uses lazy import for non-ARM |

**ARM per-timestep Fortran calls: ZERO.** All prognostic state, diagnostics, forcings, and
stats output are pure JAX/Python.

**Module-level Fortran dependency status:**
- `clubb_standalone.py`: zero (`derived_types` now local)
- `advance_clubb_to_end.py`: zero (lazy import only in non-ARM `_prescribe_forcings`)
- `radiation.py`: zero (`cos_solar_zen` and `sunray_sw` ported to pure Python)

---

## Cross-case bit-faithfulness status (vs Fortran, `compare_cases.py`)

"Runs" ≠ "bit-faithful". A case can run end-to-end in JAX yet diverge from the Fortran
oracle. Verified status (prognostic, rel tol 1e-6):

| Case | Status | Notes |
|---|---|---|
| arm | ✅ PASS (225 steps) | bit-faithful reference |
| wangara | ✅ PASS (30) | land case; uses `l_modify_bc_for_cnvg_test` 25 m BC |
| gabls2 | ✅ PASS (30) | land case |
| bomex | ✅ PASS (30) | fully bit-faithful (Iter79 BC + Iter80 thv + Iter81 rc_coef_zm) |
| dycoms2_rf01 | ✅ PASS (30) | fully bit-faithful (Iter80 thv + Iter81 rc_coef_zm) |
| atex | ✅ PASS (30) | bit-faithful since Iter84 (xm monotonic flux limiter wired in) |
| gabls2 | ✅ PASS (30) | bit-faithful (instantaneous output) |
| gabls3_night | ✅ PASS (30) | bit-faithful since Iter86 (um_f/ug time-dependent wind forcing) |
| fire | ✅ PASS (30) | bit-faithful since Iter87 (bulk surface scheme, sfctype=1) |
| neutral | ✅ PASS (30/55) | bit-faithful since Iter91 (neutral_case_sfclyr: ustar=0.5 + momentum flux) |
| ekman | ✅ PASS (30) | bit-faithful since Iter94 (sponge damping Iter93 + `ice_supersat_frac` at the cold 10 km top, Iter94) |
| cobra | ✅ PASS (40) | bit-faithful since Iter94 ice fix (Iter87-89 surface/wind/subsidence + the cold-cloud `ice_supersat_frac`); confirmed Iter96. Its step-14 cloud onset (T=266-270 K) was the SAME ice bug, not FP-boundary |
| dycoms2_rf02_nd | ✅ PASS (30) | bit-faithful (Iter96). "_nd" = **no drizzle** — a standard stratocumulus case, NOT the drizzle variant; was mislabeled blocked |
| dycoms2_rf01_fixed_sst | ✅ PASS (30) | bit-faithful (Iter98). dycoms2_rf01 variant with fixed SST surface |
| atex_long | ✅ PASS (30) | bit-faithful (Iter100). Unblocked by the `cloud_drop_sed` port (`l_cloud_sed`) |
| dycoms2_rf02_so | ✅ PASS (30) | bit-faithful (Iter100). Unblocked by `cloud_drop_sed`; do/ds variants still need drizzle microphysics |
| jun25_altocu | ✅ PASS (30) | bit-faithful (Iter188). Cold-cloud altocumulus + "simplified" radiation; unblocked by the per-step `wm_zm` (subsidence) recompute fix |
| gabls3 | ✅ PASS (30) | **bit-faithful (Iter273-274, 17th case)** — 0 prognostic failures at the full 30-step gate. Full BUGSrad correlated-k radiation + interactive soil_vegetation + gabls3 surface flux + omega subsidence. **bugs_rad is jitted (Iter274)** — fixes the eager-dispatch ~700 MB/call OOM-after-6-steps + ~2.4× faster (~6 s/step, JAX 30-step run 194 s); configured run is 1440 steps (24 h) |

**★ Full 48-case coverage survey (Iter153).** Ran a systematic smoke survey (`run_scm.py -jax -max_iters 3`)
over ALL 48 `*_model.in` cases, categorising each as RUNS / UNSUPPORTED-feature / ERROR. Result: **20 RUNS,
28 UNSUPPORTED, 0 hard crashes.** This replaces the ARM-centric view with a complete map and identifies the
coverage levers. **Methodology note:** the full `compare_runs.py` survey MUST be run sequentially with no
competing foreground/background compute — atex/fire/dycoms2_rf01 first showed spurious "JAX run failed (rc=1)"
that were pure **`timeout` artifacts under background-task contention** (the JAX run compiles + runs in ~100s,
exceeding a 150s timeout when several run at once); re-run clean, all PASS. Never trust a single contended run.
- **20 RUNS (supported features):** arm, atex, atex_long, bomex, cobra, coriolis_test, dycoms2_rf01,
  dycoms2_rf01_fixed_sst, dycoms2_rf02_do, dycoms2_rf02_ds, dycoms2_rf02_nd, dycoms2_rf02_so, ekman, fire,
  gabls2, gabls3_night, jun25_altocu, neutral, rico, wangara. Of these, all the previously-listed cases PASS;
  **rico (grid_type=2) is bit-faithful for steps 1-4 and now FP-limited from step 5** (Iter186 fixed the step-1
  rtm_cl fill_holes seed + removed the unfaithful IC floor — the step-17 abort is gone; remaining residual is
  near-zero rt-flux clip amplifying FP rtp2 diffs at the stretched dry top, see the rico section).
  **jun25_altocu is now bit-faithful (Iter188, 16th case)** — the per-step `wm_zm` subsidence fix. dycoms2_rf02_do/ds
  are KK oracle-limited; coriolis_test is FP-limited (zeroed-closure oscillator).
- **28 UNSUPPORTED — feature-blocker breakdown (the coverage levers, in priority order):**
  - **`morrison` microphysics — 19 cases** (cgils ×6 + cgils_p2k ×6→ actually cgils_s{6,11,12}{,_p2k}, clex9 ×2,
    cloud_feedback ×6, arm_3year, dycoms2_rf02_morr, nov11_altocu). **THE dominant lever** — porting Morrison
    2-moment unblocks ~40% of all cases. **STARTED Iter190** (`Microphys/Morrison_microphys/module_mp_graupel.py`):
    Fortran is `morrison_microphys_module.F90` (1523, CLUBB interface) + `Morrison_microphys/microphysics.F90` (9820,
    SAM-CLUBB wrapper: micro_proc/micro_init/satadj_liquid) + `module_mp_graupel.F90` (6699, the WRF 2-moment graupel
    scheme — the process rates). Playbook = KK's: port the self-contained special functions first (validatable
    standalone), THEN the process rates (validatable via a Morrison case-stats oracle — feed the Fortran's own state
    into the JAX rate, like the rico KK oracle). **Special-function layer DONE (3/3, Iter190-192):** `polysvp_jax`
    (saturation vapor pressure, bit-exact vs a Fortran-Horner replica); `derf1_jax` (erf, vs scipy to 2.2e-16);
    `gamma_jax` (Γ via Cody, all branches, vs scipy to 7.6e-15). **PROCESS RATES — STARTED Iter193.** The CLUBB entry is
    **`M2005MICRO_GRAUPEL`** (module_mp_graupel.F90:1047; NOT the SAM `micro_proc`/`satadj_liquid`, those are SAM-driver
    only). **Morrison oracle established:** `run_scm.py nov11_altocu -legacy` → `output/nov11_altocu_stats.nc` has the
    rates PRC/PRA/PRE + state rcm/Ncm/rrm/rho — feed the Fortran's own state into the JAX rate and compare (rico
    KK-oracle method). **WARM-RAIN rates COMPLETE (Iter193-195):** `kk_warm_rain_rates` (PRC/NPRC/NPRC1 median ~2e-7
    bit-faithful, PRA/NPRA ~4% qr-confound — BULK KK, IRAIN=0); `rain_slope`/`cloud_slope` (LAMR/N0RR + PGAM/LAMC via
    gamma_jax); `rain_evap_rate` (PRE, full Rutledge-Hobbs ventilated diffusion MU/DV/SC/ARN/AB/QVS/EPSR — oracle median
    7.1% double timing-confound + exact transcription). **ICE block STARTED (Iter196):** `_gamma_slope` +
    `ice/snow/graupel_slope` (all 4 slopes now done); `ice_deposition` (full Harrington-1995 PRD/EPRD/PRDS/EPRDS/PRDG/
    EPRDG with the SUM_DEP limiter — oracle PRD 1.5%/PRDS 2.4%/EPRD 3.6%/EPRDS 4.0%, comprehensively validating LAMI/LAMS/
    QVI/ABI). **Done (Iter197-198):** `snow_collection_rates` (PSACWS 5.4% / PRAI 4.7%), `ice_autoconv_to_snow` (PRCI 1.3%),
    `snow_self_aggregation` (NSAGG 8.3%), `deposition_nucleation` (MNUCCD Cooper 10.2%), `rain_immersion_freezing`
    (MNUCCR Bigg 4.2%), `sublimation_number_rates` (NSUBI 0.2%/NSUBS 1.3% — formula isolated by feeding oracle EPRD).
    `cloud_contact_immersion_freezing` (Iter200, MNUCCC bit-exact 8.4e-6 — log-space form preserved),
    `rain_self_collection` (Iter201, NRAGG 10.4%) + the number companions NPSACWS/NPRAI/NPRCI validated.
    **★ PROCESS-RATE COVERAGE ESSENTIALLY COMPLETE** (warm-rain + the full ice block, all oracle-validated). The
    remaining minor rates (PRACS rain-snow, melting ~0 for cold nov11, graupel — nov11 has none) + ALL the driver-glue
    pieces are COUPLED to the full assembly (the oracle stores post-limiter/post-PCC values, so they validate only
    end-to-end vs the *_mc tendencies). **M2005 driver assembly STARTED (Iter202):** `assemble_q_tendencies` (the faithful in-cloud rate-sum formula
    :3974-3988). **★ Characterization finding:** the CLUBB-output tendency `rim_mc` is NOT the single-call rate sum —
    the driver scales the in-cloud tendency by a cloud-fraction weighting **CF3D** (`T3DTEN*=CF3D`, gated on
    `cloud_frac_thresh`, :4385; grid-mean = in-cloud × CF3D — and CF3D is NOT the liquid cloud_frac, which is 0 at cold
    ice points, implied ~0.45) + the conservation limiters (oracle stores post-limiter rates) + clipping. So the *_mc
    validate ONLY END-TO-END. **★ CONCLUSIVE (Iter204): `rim_mc = (hydromet_final − hydromet_initial)/dt`**
    (morrison_microphys_module.F90:786) — the NET field change over the full Morrison step (integration + clipping +
    CF3D + sub-stepping), NOT the rate sum (explicitly excludes nothing — it's the whole-step delta). And
    `hydromet_initial` (pre-microphysics) ≠ the oracle's stored end-of-step field (CLUBB advection is between), so the
    *_mc CANNOT be reconstructed from the oracle's stored fields at all. **The ONLY Morrison validation path is to port
    the full M2005 driver, wire it into the JAX CLUBB loop (like the KK scheme), and diff via `compare_runs --case
    nov11_altocu`** — the same end-to-end gate as every faithful case. The rate LIBRARY (~24 functions, validated +
    **differentiable**) is the reusable deliverable. **Driver glue (Iter205-207):** `conserve_qc/qi/qr/qni`
    (conservation limiters) + `saturation_adjustment_pcc` (PCC) + `to_in_cloud`/`tendency_to_grid_mean`/`neg_fix_number`
    (the CF3D in-cloud↔grid-mean conversion — the piece explaining rim_mc=in-cloud_tend×CF3D). `rain_fall_speed` (Iter208,
    UMR/UNR). Remaining driver glue: the SEDIMENTATION flux-divergence sub-step loop (NSTEP CFL + downward fall-speed
    propagation + upwind flux, :4747-4825) + tendency integration (q+=qten·dt); then wire into the JAX CLUBB loop +
    `compare_runs --case nov11_altocu` (nov11_altocu/mpace_a/clex9 are morrison-only → candidates for a 17th faithful case). **Double-confound
    convention:** evap/deposition rates depending on the saturation deficit (qv−qvs) carry a ~2× confound (the field is
    created AND the deficit relaxes during the step) — BUT supersaturation-LIMITED deposition (SUM_DEP) validates tighter
    (the limiter ties it to the deficit). **Oracle-validation convention (Morrison):** auto rates
    (f(qc,nc)) validate tightly standalone (qc/nc within-step-stable); accr/evap/ice rates that depend on the rain/ice
    fields created *during* the step carry a timing confound (~few %) — validate the FORMULA by the median, defer tight
    bit-faithfulness to a running Morrison case.
    **✓ PRECISION RESOLVED (Iter191): Morrison runs FLOAT64.** The inner WRF graupel scheme (`module_mp_graupel.F90`)
    uses bare `REAL` (not `core_rknd`), but the CLUBB build compiles with **`-fdefault-real-8`** (gfortran) / `-r8`
    (nvhpc) — see `compile/config/linux_x86_64_gfortran.bash:42` — promoting all default `REAL` (and unsuffixed literals
    like EPS=1.19e-7) to real8. So Morrison is double-precision in the standard build (the same one that makes the 16
    cases bit-faithful to ~1e-15), and the JAX float64 ports CAN be bit-faithful. (Reproduce the single-precision
    *literals* as their decimal real8 values — the branch thresholds match.)
  - **`bugsrad` radiation — 5 cases** (arm_97, astex_a209, gabls3, mc3e, …).
  - **`coamps` microphysics — 2 cases** (arm_0003, mpace_b/mpace_a).
  - **SILHS interactive sampling — 2+ cases** (lba, twp_ice, mpace_b_silhs, rico_silhs — `lh_microphys_type=interactive`).
  KK (khairoutdinov_kogan) microphysics is partially built (`advance_one_hydrometeor` complete) but not yet
  wired per-step; the runnable-but-non-KK cases don't exercise it.

**✓ RESOLVED (Iter156): `pdf_params` zero-init blocker.** `pdf_parameter` is a NamedTuple; the canonical
post-advance zt-level PDF moments are Block-U locals (`_chi1_60/_schi1_60/_cf1_60/_rc1_60/_mf60/_thl1_60/
_issf1_60` at ~:4264-4297). Added `pdf_params = pdf_params._replace(chi_1/2, stdev_chi_1/2, cloud_frac_1/2,
rc_1/2, mixt_frac, thl_1/2, ice_supersat_frac_1/2=…)` right after the ice-supersat computation (:4297).
**Faithful timing confirmed:** `pdf_params` is `intent(inout)` in `advance_clubb_core_api` (clubb_driver.F90:3392)
→ the Fortran microphysics (3618, AFTER advance_clubb_core) uses the POST-advance PDF, which is exactly Block-U
"_60". After the fix `state['pdf_params'].chi_1` is nonzero and the KK step's autoconversion produces
rcm_mc≈8.2e-9 (rf02_do oracle ≈9.2e-9). **No regression: ARM + bomex still PASS** (the 15 cases don't read these
pdf_params fields). The KK rates are now FUNCTIONAL; transport + feedback application (`l_kk_micro_apply`) is the
remaining stage.

**✓ TRANSPORT WIRED (Iter157): full KK microphysics runs.** `kk_microphys_step.py`'s `l_kk_micro_apply` branch
composes the validated transport pieces (Skw_zm via `skx_func_jax(wp2, zt2zm(wp3), …)`, `kk_sedimentation`,
`kk_sed_vel_covars`, `calculate_K_hm`, `advance_one_hydrometeor` + `fill_holes` for rrm & Nrm; nu=`nu_vert_res_dep
.nu_hm`, w_above=`weights_zt2zm[:,:,0]`). Enabled for KK cases (`clubb_standalone.py`
`l_kk_micro_apply=(scheme=='khairoutdinov_kogan')`). dycoms2_rf02_do runs **30 steps stably**, rrm matches the
oracle magnitude (~1.9e-6 @10 steps vs ~2.0e-6).

**✓ COMPOSITION DIFFERENCE LOCALIZED (Iter158): the missing second-moment covar driver.** Step-aligned compares:
**step 1 — ALL dynamics machine-exact** (microphysics feeds the NEXT step's forcings); **step 2 — the MEANS pass
(rtm 3.4e-7, thlm 2.7e-8) but the SECOND MOMENTS fail (rtp2 1.4e-4, rtpthlp 1.9e-4, thlp2 9e-6, wpthlp 1.4e-5).**
So the mean tendencies (rcm_mc/thlm_mc) + transport are ~bit-faithful; the gap is the SECOND-MOMENT microphysics
source. The Fortran `calc_microphys_scheme_tendcies` also outputs `wprtp_mc/wpthlp_mc/rtp2_mc/thlp2_mc/
rtpthlp_mc` from **`KK_upscaled_covar_driver`** (KK_upscaled_covariances.F90, 2574 lines + `PDF_integrals_covars`)
— the covariances of the auto/accr/evap rates with w/rt/thl. The JAX has NONE of it (no `*covar*` microphysics
module) and never adds these to the second-moment forcings. **THE COVAR LAYER IS NOW FULLY PORTED (Iter159-170):**
(1) `PDF_integrals_covar.py` — both integral families complete + validated: `trivar_NNL_covar` (base +7 const-variants
+ `trivar_NNL_covar_eq` dispatch, SUPERSATURATED chi>0, +s_c) and `quadrivar_NNLL_covar` (base +11 const-variants +
`quadrivar_NNLL_covar_eq` dispatch, SUBSATURATED chi<0, `_signed_pow(-σ_x2,α)`, +s_cc; the `_eq` dispatch implements
the (r_r,β)↔(N_r,γ) symmetry via an x3↔x4 swap). MC-validated bases; machine-exact dispatch/variant limits.
(2) `KK_upscaled_covariances.py` — all 9 `covar_{rt,thl,x}_KK_{auto,accr,evap}` (shared `_covar_x_comp(...,s)` /
`_covar_x_evap_comp` with the ADG1 `x'=(1/(2c))(eta'∓chi')`, s=+1 rt / s=−1 thl; w-covars are direct; accr/evap w-forms
carry the `−(1−precip_frac)·(mu_x−x_mean)·tndcy` out-of-precip term) **+ `KK_upscaled_covar_driver` (Iter170)** which
sums the 9 → the 5 `_mc` with `L=Lv/(Cp·exner)`: `wprtp_mc=−w_tot`, `wpthlp_mc=L·w_tot`, `rtp2_mc=−2·rt_tot`,
`thlp2_mc=2L·thl_tot`, `rtpthlp_mc=L·rt_tot−thl_tot`. All finite+differentiable.
**MC validation convention (oracle-free):** supersaturated (trivar_NNL, chi>0) integrand = `chi^α` over full range;
subsaturated (quadrivar/evap, chi<0) integrand = `−(−chi)^α` over chi<0 only — both Cov_i(x1, ·) via sampling the
component normal/lognormal distribution. The covar FUNCTIONS/driver validate vs the rico `*_mc` stats.
**✓ THE KK rt/thl COVAR IS PHYSICALLY CORRECT — dycoms_do/ds/rico covar is `epss=1e-4`-parab-LIMITED (Iter251-253).**
**★★ PROVEN with oracle numbers (Iter310), upgrading the Iter253 inference.** Built a standalone harness
(`clubb_jax/tools/parab_harness/`, links only ACM-850 `parab`+`AiryFunction`, no CLUBB rebuild) that evaluates
`Dv_fnc` at BOTH tolerances. Captured the ACTUAL `do`-run `_dvc(order, arg)` pairs (env-gated dump + `JAX_DISABLE_JIT`;
24 unique, order ∈ [−4.47,−2.0], arg ∈ [−37,37]) and compared three ways: **JAX vs the oracle's `epss=1e-15` (true) `D_v`
= median 1.08e-14, max 2.92e-14 — the JAX is bit-faithful to the TRUE function**; **JAX vs the oracle's `epss=1e-4`
(the SCM-run value) = median 9.38e-8, max 9.95e-7, EXACTLY equal to the oracle's own 1e-4-vs-1e-15 gap.** So the entire
JAX-vs-run `D_v` discrepancy IS the run's epss=1e-4 truncation (only for arg>0, the chi<0 half; arg<0 is bit-identical),
amplified ~16× by the covar near-cancellation → the observed 1e-5–1e-4 do/ds covar failures. **do/ds is non-bit-faithful
ONLY because the JAX is MORE accurate than the deliberately-low-accuracy oracle run** — a bit-faithful match would
require porting `parab` at epss=1e-4 to reproduce the artifact.
**★★ Iter316 — DID the faithful port (the `parab` port is NOT 3385 lines for do/ds).** Traced (instrumented harness)
that for the do/ds regime — a=−order−0.5∈[1.5,4] (always ≥1.5 since order≤−2), x=arg∈[31,37] — the oracle's `parab`
dispatch goes STRAIGHT to **`expax`** (the Poincaré-type asymptotic, `Parabolic.f90:3259`), no recursion/Airy/quadrature.
`expax` (mode=0) is a self-contained ~125-line convergent series (sums U/V/U'/V' per-term recurrences, truncates when
all four `err < epss`; converges in 2-3 terms at x~32). **Ported it to JAX** (`Microphys/KK_microphys/parabolic_expax.py`,
`expax_U`, a fixed-max masked `fori_loop` reproducing the `DO WHILE` truncation — jit/vmap/grad-safe) — **bit-exact (rel
0.0) vs the harness epss=1e-4 over a∈[1.5,3.5], arg∈[15,49]** (`tests/test_parabolic_expax.py`). **Wired into `_dvc`**
(PDF_integrals_means.py): for arg ≥ 15 (the expax dispatch region; the boundary is x≥(30−a)/2.5≈11.4 for a≥1.5, so 15 is
safe), reproduce `Dv_fnc = U(a,x)` via `expax_U` instead of the high-accuracy DLMF dv. **Result: do's N=2 covar source
(rtp2) `max|Δ|` dropped from ~1e-5 to 1.36e-12 (~7 orders) — the epss=1e-4 mismatch is ELIMINATED.** No regression: the
only bit-faithful KK case `dycoms2_rf02_so` makes ZERO `_dvc` calls (verified) and PASSES at N=30; only do/ds & rico (both
already non-faithful) have dvc calls in this region. **do is still not gate-passing** — a residual ~1.36e-12 (just over the
1e-12 abs floor) amplifies past the gate by N=5; it is either the XLA-vs-ifx libm difference in `expax` (the JAX matches
the *ifx* harness to 2.9e-16, but the *run* uses XLA exp/log) or dvc calls with arg in [12,15) still on the DLMF dv.
**NEXT: characterize the ~1e-12 residual (libm vs sub-threshold args) to try to fully close do/ds.** The port is the
faithful choice (the prior high-accuracy dv was an improvisation); do/ds is now libm/near-bit-exact-limited, not
epss-limited.
**★ Iter317 CORRECTION — the Iter316 "7 orders" claim was rtp2-ONLY; the expax fix is necessary but NOT sufficient.**
Captured do's step-1 dvc args (232 calls): **112 with arg≥15 (expax, matched bit-exact), 120 with arg≤0 (V-dominated,
already bit-exact), ZERO in (0,15)** → no sub-threshold/unported-branch args; the dv is now fully correct. So **rtp2's
residual IS libm (1.4e-12, FP-growth)** — that side is done. BUT decomposing do's N=5 thlp2 (the dominant failure, JUMP to
6.0e-6, SYSTEMATIC sign jax<fort 31/118): the culprit is **`thlp2_mc` (the THL covar source) still ~1.87× the oracle**
(jax −7.21e-6 vs fort −3.85e-6, |Δ|3.36e-6) — the expax fix cut the Iter253 "16×" thl-covar error to ~2× but a SYSTEMATIC
residual remains, and it is NOT the dv (all dv calls are routed; rtp2 confirms the dv is correct). So a SECOND thl-side
mismatch exists in the covar assembly (`KK_upscaled_covariances._covar_x_comp`/`_covar_x_evap_comp` use the same dv for
rt s=+1 and thl s=−1, so the asymmetry is in a thl-specific factor or a larger thl cancellation, not the parabolic
cylinder). **NEXT: find the residual thl-covar mismatch (decompose Cov(thl,KK) auto/accr/evap components).** Honest status:
do/ds rt-side is libm-limited; the thl-side has a separate ~2× systematic covar residual — do/ds remains non-bit-faithful.
**★★ Iter318 RESOLVED the thl residual — it is irreducible cloud-top-EDGE libm, NOT a bug.** (1) Verified ALL three thl
covar formulas (auto/accr/evap, `covar_thl_KK_{auto,accr,evap}`) match the Fortran term-by-term — `(1/(2c))(tri∓biv_a1) +
(mu_thl−thlm+mu_chi/(2c))·biv_a`, the s=−1 path of `_covar_x_comp`; no formula/precip_frac/factor bug. (2) Per-step:
`wpthlp_mc` (w-projection of KK_thl) is BIT-EXACT at step 1, but `thlp2_mc` (thl-projection) is off — the only difference
is the `1/(2cthl)` factor and the `mu_chi/(2cthl)` term. (3) **Decisive — the thlp2_mc jax/fort ratio is EXACTLY 1.0000
at k=62–74 (the bulk cloud/rain layer) and only diverges at k=75–78 (the 4 sharp cloud-top-EDGE levels), growing 2.12→2.61.**
So thlp2_mc is bit-exact in the bulk; the residual is confined to the extreme-s_c edge where s_c~32 (the do/ds gap region),
the covar's `exp(−0.25 s_c²)`≈exp(−256) factors carry XLA-vs-ifx libm (~1e-14), and the severe `(tri−biv_a1)` thl
cancellation amplifies it to ~2×. **The expax port (Iter316) made do/ds BULK-bit-exact and removed the dominant epss
artifact; the residual is irreducible cloud-top-edge libm (XLA exp/log ≠ ifx), same class as rico/nov11 sharp-edge FP —
NOT fixable without improvising the cancellation.** do/ds is now edge-libm-limited (was epss-limited); the rt-side and the
bulk thl-side are bit-exact. **CONCLUDED: do/ds non-bit-faithfulness is irreducible (oracle low-accuracy dv faithfully
reproduced + XLA-vs-ifx edge libm).**
**★ Iter319 — `ds` differs from `do`; both irreducibly covar-cancellation-FP-limited (regression-checked, do/ds CLOSED).**
(1) The expax change has been in 3 iters; ran the FULL 18-case gate → **all PASS, GATE_EXIT=0** (no regression; only the
KK case dycoms2_rf02_so is bit-faithful and it makes 0 dvc calls). (2) rico is expax-NEUTRAL — same args, its precip-onset
JUMP@step6 FP is unchanged (no unlock, no regression). (3) **ds has the SAME dvc args as do** (112 arg≥15 expax-matched,
120 arg≤0, 0 in (0,15)) and bit-faithful step-1 dynamics (rtm 1e-13, chi_1 1.6e-13, stdev_chi_1 1e-12), YET its thlp2_mc
diverges BROADLY (ratio 0.22→0.75 across the whole cloud, jax<fort) — unlike do (bulk bit-exact, edge-only). Same dv,
same formula, same bit-faithful inputs → **ds's thl `(tri−biv_a1)` covar cancellation is far more ill-conditioned than
do's**, amplifying the irreducible ~1e-12 PDF-input/XLA-vs-ifx-libm differences to a broad 2–5×, where do's mild
cancellation leaves the bulk bit-exact. So do is edge-cancellation-FP-limited and ds is broadly-cancellation-FP-limited;
**both are irreducible covar-near-cancellation FP** (the expax port removed the dominant epss artifact and made do
bulk-bit-exact — the faithful contribution; the residual is the ill-conditioned cancellation amplifying machine-floor
differences, not fixable without improvising). **do/ds CLOSED.** **★ ROOT CAUSE (Iter253): the Fortran computes `D_v` to only ~1e-4 accuracy.** `Dv_fnc` (KK_utilities.F90:143)
sets the `parab` (Algorithm 850) tolerance `epss=1e-15` only if `l_high_accuracy_parab_cyl_fnc=.true.`, else
`epss=1e-4`; the DEFAULT is `.false.` (Parabolic.f90:20) and no case overrides it. The KK covar is a product of
NESTED near-cancellations (assembly `(1/2crt)(biv_a1−mu_chi·biv_a)`, ~2e-3 of the terms, AND inside `tri`,
`Γ(α+2)Dv₂−r·Γ(α+1)Dv₁`) that amplify the ~1e-4 `D_v` error to ~16× at the cloud edge; the RATES are plain means
(no cancellation) so they stay bit-faithful at ~1e-4. The faithful fix needs the 3385-line `parab` ported with the
exact epss=1e-4 truncation — impractical/low-ROI. The JAX is MORE accurate. The Iter251-252 evidence below stands: A live `dycoms2_rf02_do` run fails on rtp2/rtpthlp/thlp2 only (1e-5–1e-4, from the covar source);
the `_w` covariances are bit-exact but `_rt`/`_thl` (eta-based) differ ~16×. **Decisive isolation (Iter252):** a
brute-force 2-component MC of `Cov(rt, auto)` (transform rt'=(chi'+eta')/(2crt)) matches the JAX to **0.04%** but the
oracle to 16× — and the MC is VALIDATED (`Cov(w,auto)` MC matches the exact oracle a_w to 0.3%). Reconstructing
`covar_rt_KK_auto` with the oracle's OWN inputs reproduces the JAX (3.73e-15), not the oracle (5.89e-14) → not an
input issue. **The JAX is correct:** `_dvc` matches scipy `pbdv` to 1e-14 (r=5→48, Dv 1e4→1e255), `bivar_NL_mean_eq`
matches quadrature 1e-10, `const_x3` matches an MC 2e-4, the formula matches F90 term-by-term. The const_x3/bivar
forms multiply `exp(-0.25 r²)`(~1e-19) by huge `Dv`(~1e21), and a_rt is a near-cancellation residual (3.7e-15 from
1.65e-12 terms) → a ~3% imprecision in the Fortran's Dv at extreme cloud-edge args is amplified 16×. **dycoms_do/ds/
rico's covar source joins the FP/numerically-limited class** (the JAX is MORE accurate; matching the oracle would
require replicating the Fortran's Dv artifact). The Iter171 rico match (Fortran inputs) was real but didn't exercise
the live cancellation at the cloud edge. **★ Convention: validate a "faithful" analytic covariance against a
brute-force MC AND check its special-function callee vs scipy — distinguishes a JAX bug from a Fortran artifact.** **Oracle tip:** the
Fortran also stores the **9 individual covariances** `{rt,thl,w}_KK_{auto,accr,evap}_covar_zt` on zt — per-component
oracles (no sum/grid) that pinpoint a bug to one covar function. **`mu_eta` cancels** in a covariance (x_mean=mu_x1),
so it needn't be stored; mu_w=w_1, sigma_w=√varnce_w_1, sigma_eta=stdev_eta_1. **★ Found+fixed the `_signed_pow`
even-exponent parity bug (Iter171):** the rt/thl-evap covar is the only path using `trivar_NLL_mean` at α+1=2 (even);
the evap rate MEANS only use α=1 (odd) so it never showed. `_signed_pow(base,exp)=sign(base)|base|^exp` is odd-only
(−σ^exp for all exp); Fortran `(-σ)**exp`=`(−1)^exp σ^exp` is +σ² for even. Fixed to `where(base<0, cos(π·exp),
1)·|base|^exp` (cos(π·exp)=(−1)^exp, identical for odd → means unchanged, correct for even). **Lesson: a `signed`/
parity helper validated only at odd (or only-even) exponents is unverified at the other parity — exercise both.**
**✓ THE COVAR DRIVER IS WIRED (Iter172) — the `_mc` are bit-faithful in a live run.** `_compute_kk_covar_mc`
(kk_microphys_step.py) builds the ~70 inputs and calls the jit-compiled driver each KK step; the 5 zm `_mc`
(zt2zm + boundary-zero) feed the next step's `*_forcing` (advance_clubb_to_end.py, mirror clubb_driver.F90:3348).
Sourcing: pdf_params (extended `_replace` populates w_1/2, varnce_w_1/2, rt_1/2, stdev_eta_1/2, crt_1/2, cthl_1/2,
corr_chi_eta_1/2 from Block-U "_60" locals; corr_w_chi/corr_w_eta = 0 for ADG1, pdf_closure:1037), the prescribed
normal-space correlations (`kk_prescribed_correlations` now returns the eta/w rows of the 6×6 default array), and
`prereqs` (rr/Nr moments + process tendencies + coefs + Ncnm). **Live dycoms2_rf02_do vs Fortran:** the JAX `_mc`
match the Fortran `_mc` stats — wprtp_mc/wpthlp_mc **bit-exact (3e-7)**, thlp2_mc/rtpthlp_mc **median-exact (~7e-12)**;
rtp2_mc negligibly tiny (9e-14 = 8.6e-6 of rtp2 — auto-only/cancellation FP-noise when there's no rain). The covar
input PDF moments (crt/cthl/rt/chi/eta/w) are all bit-faithful to Fortran (~1e-11). **Performance:** the eager
16-branch-dispatch + parabolic-cylinder driver is ~100s/step on a 160-level grid → jit it (`_covar_driver_jit`,
~60s compile then ~10s/step).
**✓ dycoms2_rf02_do STEP 1 IS FULLY BIT-FAITHFUL (Iter174).** The Iter173 rrm-transport residual was the
**fill_holes threshold bug**: the JAX hydrometeor `fill_holes_vertical` (kk_microphys_step `_advance_hm`) used
`threshold = hydromet_tol` (~1e-10) but the Fortran `fill_holes_driver_api` uses `zero_threshold (=0)`, running the
vertical fill (for "r..." mixing ratios) ONLY when `any(hydromet<0)`. The JAX over-filled tiny-POSITIVE values below
tol at the cloud edge (where the Fortran did nothing) — diverging the tiny rrm field (~5e-7, near its 1e-10 tol) but
not the large Nrm field. Fix: threshold `hm_tol→0.0`, `lower_k 1→0` (the Fortran `begin=1`/1-based = the surface),
`dz=gr.dzt`. With threshold 0 the unconditional call is a no-op when there are no negatives — the faithful form.
Result: rrm/Nrm rel **1.7e-12/1.3e-10** and ALL dynamics+second moments machine-exact for step 1.
**★ dycoms2_rf02_do step 2+ "diverges" only where the JAX OUT-PERFORMS the Fortran oracle (Iter175).** The JAX is
CORRECT; the FORTRAN is inaccurate at the rt-covar. Root chain: step-1 wprtp_mc/wpthlp_mc are bit-exact but **rtp2_mc
is 16× off** (the rt covariance in dycoms's no-rain/deep-cloud regime), feeding the step-2 rtp2 forcing (3.4e-4
per-step impact > gate) → second moments diverge → cascade. The rt covar is an extreme ~**850× CANCELLATION** of two
~5.6e-26 terms. **A 20M-sample Monte-Carlo of the FULL covariance (no analytic formula) = 3.72e-15; the JAX matches
it (rel 1.18e-3 = MC noise) while the FORTRAN is 15× off (5.89e-14)** — the Fortran's ACM-850 `Dv` at the α+1 order
(~1e-3-accurate) gets amplified 850× by the cancellation. The JAX `Dv` is 1.6e-14 (vs scipy) so it gives the true
value. All JAX sub-functions independently validated at the exact inputs (Dv −3.47/−4.47 vs scipy 1.6e-14/1.4e-14;
bivar α/α+1 vs quadrature 1.5e-10/7.5e-11; trivar vs MC 1.2e-3). **Matching the oracle would require degrading the JAX
to reproduce the Fortran's error — not meaningful.** NB the pdf_closure hydrometeor→thv coupling
(`l_liq_ice_loading_test`) is default `.false.` → gated OFF (not the cause).
**Convention (Iter175):** when a KK second-moment covar "diverges" from the Fortran, **MC the FULL covariance directly**
— the extreme cancellations make the Fortran oracle itself unreliable there; the JAX may be the correct one. Do NOT
re-chase the Dv/bivar/trivar (all validated).
**Secondary (not the gate-blocker, but real):** the active-rain rrm/Nrm transport carries ~2.7% once sedimentation/K_hm
turn on (step 2; fringe relMax ~13). Suspect the cross-hydrometeor `fill_holes_hydromet_api` (Fortran runs it first when
`any(hydromet<0)`; JAX skips it) or the within-step K_hm/velocity timing (Iter140 ~2%). **Given the rt-covar FP-limit,
dycoms is likely at its practical limit (step-1 faithful) — the higher-leverage path is Morrison (19 cases).**
**Debug convention (Iter173):** the JAX never wrote rrm/Nrm/`*_mc` stats → they read 0 in compare_runs ("rel 1.0
FAIL" = unwritten, NOT a physics divergence). Now written (gated on `hm_metadata`, KK-only). Always confirm a stat
is JAX-written (`grep update("<name>"`) before trusting a compare_runs FAIL on it. The 15 non-KK cases are untouched
(gate; ARM/bomex/dycoms2_rf01 PASS).

**★ KK microphysics per-step wiring plan (Iter154 — the immediate lever; unblocks dycoms2_rf02_do/ds).**
The KK cases dycoms2_rf02_do/ds are uniform-grid (dynamics already bit-faithful like the other dycoms), so
wiring KK should make them bit-faithful — a more immediate win than Morrison. All rate + transport pieces are
oracle-validated (`tests/test_kk_rico_oracle.py`, 16 tests PASS). The orchestration to add (gated on
`microphys_scheme == 'khairoutdinov_kogan'`, so the 15 non-KK cases stay bit-identical — they use `'none'`),
mirroring Fortran `clubb_driver.F90` (`pdf_hydromet_microphys_prep` → `calc_microphys_scheme_tendcies` →
`advance_microphys`):
  1. `precip_fraction(hydromet, cloud_frac, cloud_frac_1, cloud_frac_2, …)` → `precip_frac, precip_frac_1/2`
     (`precip_frac_tol = max(0.1·max_k cloud_frac, 0.005)`).
  2. `compute_kk_microphysics(rrm, Nrm, <chi_1/2, stdev_chi_1/2, mixt_frac, thl_1/2, cloud_frac_1/2 from
     state['pdf_params']>, Nc_in_cloud, precip_frac*, rho, T_liq=thlm·exner, p_in_Pa, exner, rcm, dt,
     l_return_vel_prereqs=True)` → `(rrm_mc, Nrm_mc, rvm_mc, rcm_mc, thlm_mc), prereqs`. The Iter154
     `l_return_vel_prereqs` flag returns `prereqs` = `mvr` + the rr/Nr component moments (linear+normal) that
     the velocity functions need (the previously-missing link).
  3. Velocities (zt-level → zt2zm to momentum): `kk_sedimentation(prereqs['mvr'])` → `V_hm`;
     `kk_sed_vel_covars(precip_frac*·mu_rr/Nr, mvr, <component moments from prereqs>, …)` →
     `Vhmphmp_impc/_expc`. `calculate_K_hm(Kh_zm, hydrometp2, hm, Skw_zm, …)` → `K_hm`.
  4. `advance_one_hydrometeor(dt, hmm=rrm, hmm_tndcy=rrm_mc, K_hm, nu, wm_zt, zt2zm(V_hm),
     zt2zm(Vhmphmp_impc/expc), rho_ds_zm, invrs_rho_ds_zt, gr, w_above)` → new rrm; same for Nrm. Then
     `fill_holes`. Feed `rcm_mc`/`thlm_mc` into next step's `rtm_forcing`/`thlm_forcing` (already wired,
     advance_clubb_to_end.py:67-68).
  **Validation:** rico is FP-limited at the dynamics (k51, Iter153) so won't go bit-faithful, but dycoms2_rf02_do/ds
  should; validate ARM unaffected (gate skips KK), then the rf02_do/ds compares. **Caveat:** the accr/evap rate
  contributions from the rrm/Nrm fields carry the `calc_comp_mu_sigma_hm` within-step timing confound — only a
  running case fully validates them (the static rico oracle is pre-developed-rain). **Then: Morrison microphysics**
  (the 19-case lever).

**Testing convention (Iter85-86): compare INSTANTANEOUS, time-aligned output.** Cases with
`stats_tout > stats_tsamp` output **window-averaged** records, and `stats_tsamp > dt_main` (e.g.
gabls3_night: dt=10, tsamp=60) **sub-samples**. In both cases the JAX vs Fortran records are
misaligned in time (e.g. gabls2 JAX [600,1200] vs Fortran [1140,1740]), producing huge spurious
"failures" unrelated to physics — gabls2's "intermittent fail at 10/20/30, pass at 15" and
gabls3_night's apparent "blow-up" were both this artifact. **`compare_runs.py` now auto-reads
`dt_main` and forces `stats_tsamp = stats_tout = dt_main`** for both runs, so every comparison is
true per-step and time-aligned (use `--tout N` to override). `compare_cases.py` relies on this.
Always diagnose a "failure" of an averaged/sub-sampled case this way before assuming a physics bug.
The averaged-output path of `io/stats_writer.py` remains a known lower-priority (diagnostic-only)
discrepancy vs Fortran; the physics it averages is correct.

**Testing convention (Iter93): allclose gate floor.** `compare_runs.py` flags a field when
`|Δ| > ABS_TOL + REL_TOL·|ref|` (numpy.allclose convention; ABS_TOL=1e-12, REL_TOL=1e-6) rather than
pure relative error. A **dry case** (e.g. ekman) leaves moisture moments (`rtp2`, `rtpthlp`, `wprtp`,
`wpthlp`) at physical zero, where f64 roundoff (|Δ|~1e-24 vs |ref|~1e-24) reports enormous *relative*
error — false positives. The absolute floor is far below any real-scale prognostic difference, so it
never masks a true divergence (e.g. ekman's `wp2` |Δ|~8e-5 still fails). When a near-zero field
"fails," check its `maxAbsDiff` column before assuming a physics bug.

**Testing convention (Iter97): unwritten diagnostic stats read as 0 — not physics.** Many CLUBB
stats variables are registered (so they exist in the JAX NetCDF) but the JAX never calls
`sw.update(...)` for them, so they stay at their **0 default**. In a comparison these show as a huge
"diag FAIL" (Fortran writes the real value, JAX writes 0) that is purely an output gap, NOT a physics
divergence — e.g. `cloud_frac`, `cloud_frac_1/2`, `wprcp`, `wp2rcp`, `ice_supersat_frac` were all
unwritten (Iter97 added `cloud_frac` and `ice_supersat_frac`; **Iter309 added `precip_frac`** for the
KK path — it was reading 0-everywhere and briefly looked like a `precip_frac=5e-3` vs `0` physics bug
in rico before the all-zero check exposed it as the same red herring). **Before chasing a cloud/PDF diagnostic
that is exactly 0 in JAX, grep `clubb_jax/src` for `update("<name>"` — if there is no write, it is a
red herring.** Verify the underlying physics via a quantity that IS written and consumed (e.g.
`bv_freq_sqd_mixed`, `chi_1`, `stdev_chi_1`, `rsat`, `rsati`, `mixt_frac`).

**Convention: time-dependent forcings.** Cases with `l_t_dependent=.true.` and
`l_ignore_forcings=.false.` read `{case}_forcings.in`, which may provide — besides `thlm_f`,
`rtm_f`, `w` — the columns `um_f`, `vm_f` (u/v momentum forcing), `ug`, `vg` (geostrophic wind,
height- AND time-dependent), and `um_ref`, `vm_ref`. `generic_forcings.py` ports the full set; a
case that diverges only in `um`/`vm` with `um_f`/`vm_f`/`ug`/`vg` present is the signature of a
missing time-dependent wind forcing. **A column that is entirely blank (-999.9) in every time block
is "not provided" — Fortran keeps the sounding value for state fields (`ug`, `vg`, `w` subsidence),
so JAX must NOT apply the filled-to-0 column** (`_parse_forcings_file._col` returns `None` for
all-blank columns; `'w'`/`'ug'`/`'vg'`/`'um_f'`/`'vm_f'` all use it). Additive forcings (`thlm_f`,
`rtm_f`) default to 0 so an all-blank one is harmless either way.
A constant offset in a field above the sounding top is instead the signature of a sounding
extrapolation/blank-handling difference.

**Surface scheme by `sfctype`:** `sfctype=1` means compute fluxes from a **bulk formula** (drag
coefficient × wind × air-sea contrast), NOT prescribed `sens_ht`/`latent_ht`. Each case has its own
drag coefficient and momentum roughness `z0` (fire `Cz=0.0013`; cobra uses `diag_ustar` with
`z0=1.75 m` — verify `z0` against the case's `.F90`, it is NOT always the ARM 0.035 m).

**Critical convention discovered (Iter 79):** Cases with `l_modify_bc_for_cnvg_test = .true.`
in their `*_model.in` (e.g. bomex, wangara) do **not** take surface-BC physical quantities
(`rtm_bot`, `um_bot`, …) from the lowest zt level. They take them at a fixed height
**z_bot = 25 m** via `mono_cubic_interp` (Steffen, `interpolation.F90`) applied to the
`zt2zm`-interpolated fields (`constant_height_option = 2`, hard-coded in
`prescribe_forcings.F90:998`). `exner_zm[0]` is overridden to `(p_sfc/p0)^kappa` first.
Ported in `generic_forcings.py:_read_surface_var_for_bc` + `_mono_cubic_interp`. The old code
used `rtm[:,0]` (20 m) → BOMEX surface moisture flux was wrong by ~3e-5, growing to ~2e-2.

**Fixed (Iter80): cloud-regime thv-moment round-trip.** The thv buoyancy moments
`wpthvp`/`rtpthvp`/`thlpthvp` (advance_clubb_core_module Iter33) were assembled from a lossy
`zt→zm→zt` round-trip of the rc-flux moments instead of the native-zt `_wprcp_zt_61` etc.
Fortran computes `wprcp` and `wpthvp` on the same pdf grid in one pass
(`pdf_closure_module.F90:1130-1155`). The round-trip smooths sharp cloud-top `wprcp` gradients
(negligible for ARM's small cloud fraction; ~5e-4 for thick cloud). This was the dominant
divergence for dycoms2_rf01 (cloudy from step 1) and BOMEX (at cloud onset). **General lesson:**
any quantity Fortran computes natively on the pdf grid must be assembled on that grid in JAX —
do not round-trip through the other grid; the error is invisible on ARM but grows with cloud water.

**Fixed (Iter81): stale `rc_coef_zm` in the post-advance path.** `rc_coef_zm` (used in
`diagnose_upxp`/`upthvp`'s cloud term `rc_coef_zm*uprcp`) is initialized to zeros and carried
across timesteps, but the post-advance JAX path never refreshed it (assigned only in the skipped
pre-advance Block G and the stats-only Iter69 block). So the momentum buoyancy cloud term was
always `0*uprcp` — invisible on ARM (uprcp≈0) but a growing error for cloudy cases. Fix: assign
`rc_coef_zm = zt2zm(rc_coef)` (k_ub zeroed) in the post-advance override block. This fully closed
bomex and dycoms2_rf01. **General convention:** every carried (post-advance) diagnostic must be
refreshed each timestep in the post-advance path — `grep` for any field assigned only in Block G
or a stats-only block but consumed before the next PDF call.

**Fixed (Iter94): `ice_supersat_frac` at cold/deep domain tops.** The post-advance Block U used the
warm-cloud shortcut `ice_supersat_frac = cloud_frac`, valid only where every level is above freezing.
`l_calc_ice_supersat_frac` is hardcoded `.true.` in Fortran, and `calc_ice_cloud_frac_component`
(`pdf_closure_module.F90:2490`) computes, for `tl ≤ T_freeze_K`, the PDF fraction supersaturated
w.r.t. **ice** (`chi` vs `chi_at_ice_sat = crt·(rsat_ice − rsatl)`), not the liquid `cloud_frac`.
ekman's 10 km domain has a cold top (T≈203 K) that is ice-supersaturated (`ice_supersat_frac`=1 in
Fortran, 0 in the JAX shortcut). That fed `brunt_vaisala_freq_sqd_mixed` (via
`exp(−bv_efold·ice_supersat_frac)·(bv_dry − bv_moist)`); the wrong ≈0 N² passed the `sqrt(max(0,·))`
**splat** clip, so JAX applied spurious wp2→up2/vp2 splatting at the top, seeding a step-2 moment
divergence that grew ~2–4×/step. **General lesson — diagnosing a "moment-only, top-of-domain" drift:**
when the means match but moments diverge at the domain top, suspect a quantity that is identically
zero in shallow/warm cases — `ice_supersat_frac`, the splat Brunt-Väisälä term, or another upper-region
diagnostic — before assuming FP-boundary. Bisect per-step (a sudden machine-eps→1e-7 jump at one step
is a branch/threshold), then per-level and per-component (`bv_freq_sqd_dry/moist/mixed/splat` and
`ice_supersat_frac` are all in the stats output).

**Fixed (Iter102): uv nudging (`l_uv_nudge`) was unported.** The JAX passed `ts_nudge`/`um_ref` into
`advance_clubb_core` but never applied the nudging. Added `um/vm -= (um/vm − um/vm_ref)·(dt/ts_nudge)`
after the uv sponge (faithful to `advance_xm_wpxp_module.F90:1126-1151`, under `l_predict_upwp_vpwp`
+ `l_uv_nudge`). For coriolis_test (`ts_nudge=dt`) this fully resets um/vm to their zero reference each
step, fixing the spurious um drift (was 2.8e-7 vs Fortran's ~0). **No-op for the 15 faithful cases**
(none set `l_uv_nudge`). General lesson: a parameter threaded into `advance_clubb_core` but never
consumed (grep for its use) is the signature of an unapplied physics term — the same class as the
Iter81 stale `rc_coef_zm`.

**Open (next major feature): xm monotonic flux limiter — `mono_flux_limiter.F90`.**
ATEX's step-1 `um` failure was traced to the `um_mfl` budget term (1.808e-4, vs 0 in JAX): the
**monotonic flux limiter for the mean fields** (`l_mono_flux_lim_um/vm/rtm/thlm`, all default
`.true.` in `configurable_model_flags.in`). JAX does **not** apply this limiter (only computes its
`mean_w_up/down` stats). It is a no-op for the 5 bit-faithful cases (their `um_mfl`/`rtm_mfl`≈0 —
never triggers), but it triggers for strong-shear/stable BLs: **atex** (um_mfl from step 1) and
**gabls3_night** (passes to step ~15, then blows up — the limiter exists precisely to prevent such
flux-driven instabilities). Porting it is the highest-leverage next step (unblocks ≥2 cases).

**Iter83: module ported** → `clubb_jax/src/CLUBB_core/mono_flux_limiter.py`
(`monotonic_turbulent_flux_limit()`). Faithful to the SCM standalone path (ascending grid,
`l_implemented=.false.`, `l_mfl_xm_imp_adj=.true.`, `l_force_descending_solves=.false.`). Detects
non-monotonic turbulent advection of `xm`, limits `wpxp` (sequential k-loop), and re-solves `xm`
implicitly (`term_ma_zt_lhs_jax` + 1/dt; tridiag solve), plus the domain-top spike-fix. **Iter84: wired in** after the um/vm/rtm/thlm solves in advance_clubb_core (before `clip_covar`,
matching the Fortran order), reusing one `calc_turb_adv_range` per step. Per-field params:
rtm `(rt_tol², rt_tol_mfl)`, thlm `(thl_tol², thl_tol_mfl)`, um/vm `(w_tol_sqd, w_tol)`; xm_forcing =
the field's solve forcing (um/vm include the Coriolis tendency). **atex is now bit-faithful (30
steps).** No-op for arm/bomex/dycoms2_rf01/wangara (verified PASS at 30) and for gabls2/gabls3_night
(their Fortran `um_mfl`=0 — their failures are unrelated to the mfl). Descending grid / host-model
paths and the bit-exactness of `calc_turb_adv_range` for stable BLs were not needed here and are
unverified.

**Known unported flag paths (Iter95 adversarial scan).** The JAX port implements the path each flag
takes for the 10 bit-faithful cases; these alternate paths are NOT ported and will break any case that
sets them. Grep the JAX source for the flag name to find the assumption:
- ~~`l_ho_nontrad_coriolis=True`~~ — **ported Iter103** (4 terms: upwp forcing `+= fcor_y·(up2−wp2)`,
  wp2 RHS `+= 2·fcor_y·upwp`, wp3 RHS `+= 3·fcor_y·wp2up`, up2 RHS `−= 2·fcor_y·upwp`). Verified
  bit-faithful at step 0; coriolis_test's Foucault oscillator works but FP-accumulates at long runs.
- `l_andre_1978=True` / `l_vary_convect_depth=True` — Andre-1978 / varying-convective-depth surface
  variance in `sfc_varnce_module.py` (only the non-Andre, fixed-depth path ported).
- `l_use_wp3_lim_with_smth_Heaviside=False` — alternate wp3 limiter in `clip_explicit.py`.
- `l_lmm_stepping=True` — LMM time stepping (only standard stepping ported).
- non-default `fill_holes_type` — `fill_holes.py` raises `NotImplementedError`.
When a new case fails, first diff its `*_model.in` and parameter/flags files against a known-faithful
case (and the defaults) — a non-default flag here is the likely cause, not a subtle numerical bug.

**Triage of other runnable-but-failing cases** (see `compare_cases.py` `BLOCKED_CASES`):
dycoms2_rf02_* (drizzle/rf02 physics); arm_97 (morrison + bugsrad + SILHS); mc3e/arm_0003 (COAMPS
microphysics / `l_predict_Nc`). These need substantial unported physics, not subtle fixes.
(cobra became faithful via the Iter94 ice fix — its cold cloud onset was the same bug, see status table.)

## Remaining Work

**★ ACHIEVABLE-STATE ASSESSMENT (Iter254) — read this before picking the next piece.** The non-subsystem
bit-faithful frontier is **SATURATED**: 16 cases bit-faithful (ARM re-verified 30-step PASS Iter254); rico/coriolis
FP-limited (variance-floor / Foucault oscillator). Every remaining gain requires a LARGE subsystem port with
**poor ROI**, because the cases they unblock are themselves numerically-limited:
- **Microphysics cases are numerically/FP/epss-limited, and the JAX is often MORE accurate than the low-accuracy
  Fortran defaults** (e.g. dycoms_do/ds/rico KK covar = `epss=1e-4`-parab-limited Iter253; dycoms_morr = sharp-edge
  upwind-sed FP-limited Iter250; nov11 = ice-edge FP-limited Iter231). Porting MORE microphysics (COAMPS→arm_0003)
  yields more numerically-limited cases, not bit-faithful ones.
- **BUGSrad (~7000 lines correlated-k radiation) unblocks ~18 cases — but most are ALSO Morrison** (cgils/
  cloud_feedback/arm_97/mc3e) → numerically-limited. The ONLY clean (microphysics-free) win is **gabls3**, which
  also needs `l_soil_veg` (only 250 lines). So BUGSrad is ~20+ iterations for ~1 clean case (gabls3). **STARTED
  Iter255**, `clubb_jax/src/Radiation/BUGSrad/`: planck, newexp, rayle, bugsrad_physconst, gascon, cloudg, comscp,
  two_rt_lw, two_rt_sw, gases-helpers, gases-tables, gases-dispatch, bugs_lwr, bugs_swr, bugs_rad +
  `Radiation/bugsrad_driver.py` (16 pieces, validated vs Fortran-formula replicas to ≤2e-13 + invariants;
  `tests/test_bugsrad.py`). The whole RT MACHINERY + the CLUBB↔BUGSrad grid interface are done — solvers + gas
  absorption + cloud/Rayleigh optics + both band drivers + the `bugs_rad` orchestration with heating rates
  (`rate=−grav·0.01/cp·(Fnet[l]−Fnet[l+1])/dpl`, LW conserves by telescoping) + `bugsrad_driver` (std-atm load,
  `determine_extended_atmos_bounds`, the top-down vertical-flip + buffer + std-atm extension grid map, radht flip
  back) + **the JAX radiation dispatch** (`radiation.py:advance_radiation` case "bugsrad" → `_advance_bugsrad_radiation`,
  Iter269: builds+caches the rad-grid setup, computes T_in_K/p_in_Pam/amu0 per step, writes radht/Frad to state).
  The full BUGSrad path runs end-to-end in the JAX driver (`tests/test_bugsrad.py::test_bugsrad_radiation_dispatch`).
  **`soil_vegetation.py` ported (Iter270, BIT-EXACT)** — the gabls3 `l_soil_veg` lower BC (Deardorff/Duynkerke
  force-restore surface energy budget; `tests/test_soil_vegetation.py`). It runs in the radiation wrapper BEFORE the
  radiation advance (radiation_module.F90:152), using the PREVIOUS step's surface fluxes `Frad_*(:,1)` (JAX index
  `[:,0]`); the updated `veg_T_in_K` is the surface temperature for the gabls3 surface-flux path (`prescribe_forcings`).
  **★ gabls3 now RUNS end-to-end in the JAX driver (Iter271)** — BUGSrad + soil_veg + the `_gabls3_sfclyr` surface
  flux are all wired (`generic_forcings.py` runtype `gabls3`, `radiation.py:_advance_soil_veg_step`, unblocked in
  `_check_unsupported_features`). **The compiled Fortran DOES produce a bugsrad gabls3 reference**, so `compare_runs
  --case gabls3` is the gate. **★★ gabls3 is BIT-FAITHFUL (Iter273, the 17th case)** — `compare_runs --case gabls3`
  PASSES all 16 prognostics (rel ~1e-9–1e-12 at 3 steps). Two fixes got it there: (Iter272) gabls3 prescribes
  subsidence as `omega[Pa/s]` not `w[m/s]` and moisture as `sp_hmdty_f` — added `wm_zt=-omega/(grav·rho)` +
  `rtm_f=sp_hmdty_f·(1+rtm)²` to `generic_forcings.py`; (Iter273) the BUGSrad heating rate must use **constants_clubb
  grav/Cp (9.81/1004.67), NOT BUGSrad's physconst (9.80665/1004.0)** — the Fortran driver passes constants_clubb
  (bugsrad_driver.F90:357); fixed in `bugsrad_driver.py`. **★ `bugs_rad` is JITTED (Iter274, cached per scalar combo)
  — the eager 18-band×k dispatch leaked ~700 MB/call → OOM (EXIT=137) after ~6 steps; jit gives bounded memory + ~6 s/step.
  Bit-exact to ~1e-13 (XLA reordering). gabls3 now passes the full 30-step gate** (configured run is 1440 steps / 24 h).
  **★ A `Killed`/EXIT=137 with no traceback = OOM, not a NaN; run `PYTHONUNBUFFERED=1 python -u` + a repeated-call
  `resource.getrusage().ru_maxrss` probe to localize. jit any per-timestep heavy-op-count JAX subsystem to avoid the
  eager device-buffer leak.** **★
  Convention: prognostic drift with bit-exact forcing diagnostics ⇒ check `wm_zt`/`wm_zm`; subsidence may be `omega`
  not `w`.** **★ Faithfulness: pass the constants the Fortran CALLER passes (constants_clubb), not a subsystem's own
  physconst.** **★ CLUBB
  BUGSrad build flags: `-Dradoffline -Dnooverlap -DCLUBB` (compile.bash:173) → nnp=nlm (no ghost layer); the simple
  `two_rt_lw/sw` called twice (cloudy→all-sky, clear→clear-sky) ARE used (not the max/random `_sel/_iter`); overlap
  coeffs b1..b4 and the snow field qril are UNUSED; the cloud-fraction effect enters ONLY via `cwrho/cirho =
  den·1000·q·acld` weighting, done inside `bugs_rad` (`den=ppl·100/(287·tl)`, `rmix=ql/(1−ql)`).** ★ Validate a composed driver by
  an INVARIANT (no-cloud → all-sky fluxes == clear-sky, bit-exact) when the Fortran can't be run for a bit-oracle. **★ Convention:
  PARSE large Fortran `data`/coefficient tables from the source file rather than hand-transcribe (mechanical +
  validatable by shape/count/spot-check). Gotchas: implied-DO bound names are UPPERCASE; `data((` has no space.**
  Remaining: the `gases` 18-band dispatch (thin select-case over helpers+tables) → band drivers `bugs_lwr/swr` →
  `bugs_rad` → `bugsrad_driver` → soil_veg → wire gabls3. See Iter255-263 CHANGELOG. **★ BUGSrad faithfulness convention (NEW): the radiation
  has DELIBERATE single-precision / approximate steps the JAX must replicate (not "improve"), same class as the KK
  epss=1e-4 parab: (1) `pi=acos(-1.)` is the float32 value of π in cloudg (`float(np.arccos(np.float32(-1.0)))`);
  (2) cloudg's absorption term wraps its expression in `sngl(...)` → truncate to float32. ★ CAVEAT (Iter260): the
  `newexp` fast-exp the two_rt solvers `use` is guarded by `#ifdef usenewexp`, which the oracle gfortran build does
  NOT define (CPPDEFS has no -Dusenewexp) → the solvers use the INTRINSIC exp (jnp.exp); `newexp.py` is faithful but
  UNUSED by the standard build. ALWAYS check `#ifdef`/CPPDEFS before assuming a `use`d module is active.**
- **SILHS (rico_silhs/mpace_b/lba) uses interactive Latin-Hypercube random sampling → not bit-reproducible** vs the
  Fortran RNG. Not a bit-faithfulness target.
- **Differentiability (secondary goal)** is component-level DONE; full-timestep grad is blocked by the all-or-nothing
  orchestration-numpy refactor + the numpy `mono_flux_limiter` (whose `calc_turb_adv_range` core is inherently
  discrete) + the `mixing_length` while_loop (perf-ruled-out Iter180). The mono_flux_limiter is the only bounded
  "entirely-in-JAX" piece, but it's risky (2 faithful cases) for a discrete-core differentiability gain.
**Conclusion:** the project is at its practical bit-faithful ceiling for the tractable scope; full 48-case
completion is gated by Fortran numerical limits (some are the Fortran's own imprecision) + impractical ports.
Future iterations: pick gabls3 (BUGSrad+soil/veg, the one clean win) only as a deliberate multi-iteration push, or
consolidate/verify. **Do NOT chase the numerically-limited microphysics cases as "bugs" — they are characterized.**

**Per-case blocker survey (Iter101).** Of 48 cases, 15 are bit-faithful. The other 33 each need a
major unported SUBSYSTEM (verified by smoke-testing each — the `_check_unsupported_features` message
lists the exact blocker). Grouped by what unblocks them:
- **KK microphysics (`khairoutdinov_kogan`) only** → rico, dycoms2_rf02_do, dycoms2_rf02_ds. The most
  **self-contained** unblock (no bugsrad/SILHS). 3 cases.
- **morrison microphysics only** → nov11_altocu, mpace_a, clex9_nov02/oct14. (mpace_b: morrison.)
- **morrison + bugsrad** → cgils_s6/s11/s12 (+p2k), cloud_feedback_s6/s11/s12 (+p2k), arm_97, mc3e. ~18.
- **KK + bugsrad** → astex_a209.
- **bugsrad + l_soil_veg** (no microphysics) → gabls3.
- **SILHS** (`lh_microphys_type='interactive'`) → rico_silhs, mpace_b_silhs, lba (+morrison+lba rad),
  arm_97, mc3e.
- **COAMPS microphysics** → arm_0003.
- ~~jun25_altocu~~ — **BIT-FAITHFUL (Iter188), the 16th case.** coriolis_test (nontraditional Coriolis, FP-limited)
  is the only runnable non-subsystem case left besides rico (FP-limited). **jun25 root cause (Iter188):** despite its
  "steep radiation" reputation, the radiation was bit-exact — the seed was the **stale per-step `wm_zm`**. The decisive
  steps: (1) decouple-the-oracle exonerated the radiation (fed Fortran's own cloud into the JAX `_simple_rad_lw`, radht
  matched to 4.4e-18); (2) budget-decomposing thlp2 found `thlp2_ma`=0 and `wm_zm`=0 (vs Fortran 4e-3) while `wm_zt`
  was faithful; (3) `prescribe_forcings` updated `wm_zt` per-step but never recomputed `wm_zm`, so the xp2/xpyp
  mean-advection by subsidence was missing → cold-cloud cascade. Fixed: recompute `wm_zm = zt2zm_jax(wm_zt)` when the
  forcing updates `wm_zt` (generic_forcings.py, faithful to time_dependent_input.F90:837 — raw zt2zm, do NOT zero the
  bottom). **Lesson (a new convention):** a per-step-forced field with a grid-staggered partner (`_zt`↔`_zm`) must have
  the partner recomputed when it's updated; a field reassigned in `prescribe_forcings` whose counterpart is computed
  only at init is the bug signature (same class as the Iter81 stale `rc_coef_zm`). And: exonerate the obvious subsystem
  by decouple-the-oracle BEFORE assuming it's the seed — "steep radiation" was a red herring.

**Microphysics-port roadmap (the prerequisite for ~25 cases).** The JAX `advance_clubb_core` was built
for ARM (`hydromet_dim = 0`, hardcoded in `clubb_standalone.py:746`); there is **NO hydrometeor
(rrm/Nrm) prognostic infrastructure**. The required order:
  1. **Hydrometeor infrastructure first** — set `hydromet_dim` from the case, init rrm/Nrm from the
     sounding, and advect them (the `hydromet`/`wphydrometp`/`K_hm` machinery; reuse advance_xm_wpxp /
     advance_windm_edsclrm where possible). Until a case can RUN, the process rates can't be tested
     in-context (working rule 6), so do NOT port KK/morrison rates before this.
     - **Verifiability constraint (Iter105):** the f2py API (`clubb_f2py`) exposes 226 CLUBB_core
       functions but **ZERO microphysics** — so KK/morrison/hydromet routines can only be verified by
       full-case comparison, which needs the *whole* subsystem complete. The advance_clubb_core side
       of the coupling is small — the hydrometeor contributions to the thv buoyancy moments
       (`pdf_closure_module.F90:1168-1180`): `{wpthvp,wp2thvp,thlpthvp,rtpthvp} -= thv_ds·{wphydrometp,
       wp2hmp,thlphmp,rtphmp}` summed over mixing-ratio hydrometeors (`l_mix_rat_hm`). It is a no-op
       for the 15 cases (hydromet_dim=0), but the wp2thvp/wp2hmp grid (the JAX keeps wp2thvp on zt,
       the Fortran hmp covariances are zm) must be resolved against a RUNNING rico before porting —
       a speculative port can't be verified and would risk a silent grid error. So defer it to when
       rico runs end-to-end.
  2. **KK microphysics** — unblocks rico + dycoms2_rf02_do/ds, the smallest subsystem. **Iter106
     scoping (concrete):**
     - ALL KK cases use `l_local_kk=.false.` → the **UPSCALED** (analytic PDF-integrated) path, NOT
       the simpler local one. That path is large: `KK_microphys/` ≈ 15 files (`KK_upscaled_means/
       covariances/variances`, `PDF_integrals_*`, special functions `AiryFunction`, `Parabolic`).
       Each upscaled rate (`KK_auto/accr/evap_upscaled_mean`) is an analytic function of **16
       hydrometeor-PDF moments** (chi/Ncn means+sigmas+correlations per component) — so it CANNOT be
       verified in isolation; it needs the full hydrometeor PDF first.
     - **Oracle (good news):** a Fortran case run writes the rates and their state to stats —
       `rrm_auto`, `rrm_accr`, `rrm_evap`, `rrm_sd`, plus `rrm`, `Nrm`, `rtm_mc`, `rcm_mc`. So the KK
       port IS verifiable by full-case comparison (`compare_runs.py --case rico`) once the subsystem
       is complete — no f2py needed (the API exposes no microphysics).
     - **Order within KK:** (a) hydrometeor infra (hydromet_dim=2, rrm/Nrm init=0, hm_metadata);
       (b) the hydrometeor PDF (chi/Ncn/rr component moments) — the inputs to the rate functions;
       (c) the upscaled rate functions + PDF integrals; (d) hydromet mean advance (rrm/Nrm transport);
       (e) the advance_clubb_core thv coupling (the 4 no-op-for-now terms, grid to resolve when rico
       runs). **Note:** rico's cloud is inactive for the first steps (`rcm=rrm=0`), so the very early
       steps are a cloud-free dry case — a possible incremental-verification foothold.
     - **DONE so far (Iter108+110+111): step (c), all three process rates (auto + accr + evap).**
       Accretion (Iter110): `KK_accr_upscaled_mean` — same `bivar_NL_mean_eq` dispatch as auto but y=r_r
       (rr_tol=1e-10), exps alpha=beta=1.15, constant coef 67, per-component precip_frac factor.
       Evaporation (Iter111): the TRIVARIATE family — `trivar_NLL_mean` + 5 const variants +
       `trivar_NLL_mean_eq` (8-way dispatch reusing const_x1x2/const_x2 with swapped x2,x3 args) +
       `KK_evap_upscaled_mean` (exps chi=1, r_r=1/3, N_r=2/3; integrates over the chi<0 SUBSATURATED
       half — hence (-sigma_x1)^alpha and Dv(.,+s_cc); KK_evap_coef=3·C_evap·G_T_p passed in like the
       other coefs). Verified: trivar general vs 3-D-reduced quadrature rel 3.3e-11; KK_evap composes to a
       correctly NEGATIVE rate (removes rain) and is differentiable.
       Mean volume radius (Iter112): `KK_mvr_upscaled_mean` — the bivar_LL form (r_r,N_r both lognormal,
       exps 1/3, -1/3, coef ((4/3)π·rho_lw)^(-1/3)); `bivar_LL_mean`(+2 const)/`bivar_LL_mean_eq` (4-way).
       Verified vs quadrature (LL general rel 1.5e-15), gives a physical ~80 µm radius, differentiable.
       **The upscaled-KK analytic MEANS library is now complete** (all 4 means). Nrm tendencies (Iter124-125)
       and the mean sedimentation velocities (`kk_sedimentation`, Iter133, bit-exact vs rico Vrr/VNr) are now
       also ported. The upscaled-turbulent-sed COVARIANCES (`KK_sed_vel_covars`, the `<V_hm'h_m'>` impc/expc
       terms) are now also ported+validated (Iter134, bit-faithful vs rico rr/Nr_KK_mvr_covar_zt). **All KK
       rate/velocity/covariance functions the rrm/Nrm advance needs are now ported.** Still unported: the
       upscaled VARIANCES (`KK_upscaled_variances.F90`) and SILHS local means — neither is needed for the
       upscaled rico rrm/Nrm mass + sedimentation advance.
     - **ALL THREE mass-tendency rates VALIDATED END-TO-END vs the Fortran rico oracle (Iter113-114).**
       Feeding the Fortran's own PDF moments (from rico_stats.nc) into the JAX rates + the linear→log
       conversions (`mean_L2N`/`stdev_L2N`/`corr_NL2NN`/`corr_LL2NN`, sigma2_on_mu2=(σ/μ)²) reproduces:
       `rrm_auto` to median **4.7e-7** (const_x2 path — rico's N_cn is constant); `rrm_accr` to median **6.1e-9**
       (general bivar + corr_chi_rr); `rrm_evap` to median **3.3e-6** (trivariate + 6 correlations + the
       thermodynamic `kk_evap_coef` at T_liq=thlm·exner) with 11/12 points <1e-4 (one variance-tolerance-boundary
       point sigma_rr~rr_tol differs — a dispatch edge, not a rate error). `test_kk_rico_oracle.py`. This proves
       the rate-function math is bit-faithful-to-the-gate against the actual Fortran microphysics, independent of
       the PDF setup. **Consequence:** the rate functions + log conversions + G_T_p/evap-coef are DONE+validated;
       the remaining gap to a running rico is purely the hydrometeor INFRASTRUCTURE + PDF-moment SETUP (so the JAX
       produces chi/Ncn/rr/Nr component moments itself) + advance/coupling — NOT the rate math.
       `Microphys/KK_microphys/` (new JAX package): `parabolic_cylinder.py` (the parabolic cylinder
       function D_v — the only transcendental in the KK means, oracle `KK_utilities.F90::Dv_fnc` /
       ACM Alg. 850; implemented via the 1F1 series DLMF 12.4 for z≲5.75 and the optimally-truncated
       descending asymptotic DLMF 12.9 for z>5.75); `PDF_integrals_means.py` (`bivar_NL_mean` + the 3
       const variants); `KK_upscaled_means.py` (`bivar_NL_mean_eq` 4-way dispatch + `KK_auto_upscaled_mean`
       + params: KK_auto_rc_exp=2.47, KK_auto_Nc_exp=-1.79, kk_auto_coef(rho), chi_tol/Nc_tol/parab_cyl_max_input).
       **Verified two independent ways** (`tests/test_kk_autoconversion.py`, no Fortran binary needed since
       the API exposes no microphysics): D_v vs `scipy.special.pbdv` (series rel<1e-8, asym rel<1e-6) and
       the closed forms vs brute-force 1-D quadrature of their defining NL integral (general rel **7e-15**,
       const_x2 rel 9e-16). `KK_auto_upscaled_mean` is differentiable (jax.grad rel 1e-10). Caveats logged
       in `parabolic_cylinder.py`: (i) the z∈[5,6.5] series↔asym handoff band carries worst-case rel ~2e-4
       for the steepest exponent (physically suppressed by exp(-s_c²/4)); (ii) the large-NEGATIVE-z growing
       branch (z≲-6, the `V(a,x)` half of Alg. 850) is NOT yet ported — the consuming integral's exp(-s_c²/4)
       tames it, so a future revision should evaluate that scaled combination directly.
     - **DONE (Iter109): the linear→log moment conversions, step (b) building block.**
       `CLUBB_core/pdf_utilities.py`: `mean_L2N`, `stdev_L2N`, `corr_NL2NN`, `corr_LL2NN` (oracle
       `pdf_utilities.F90`). mean_L2N/stdev_L2N are **bit-to-bit exact vs the f2py API** (`f2py_mean_l2n`,
       `f2py_stdev_l2n`; rel 0.0 — `pdf_utilities` IS a CLUBB_core module so it is in the API, unlike the
       microphysics); the two corr conversions (not exposed) are verified vs Monte-Carlo correlated-lognormal
       sampling (|Δ|<1e-3). `tests/test_pdf_utilities.py`. These convert the rico stats' LINEAR Ncn moments
       into the LOG moments (mu_Ncn_n, sigma_Ncn_n) and the corr_chi_Ncn_n that the autoconversion kernel needs.
     - **Step (b) progress (Iter115-116):** (i) `precip_fraction` (Iter115, bit-exact vs rico) → precip_frac_1/2.
       (ii) `calc_comp_mu_sigma_hm` (Iter116, `setup_clubb_pdf_params.py`) → the in-precip component means/stdevs
       mu_hm_1/2, sigma_hm_1/2 (and hm_1/2) via a mean+variance-preserving quadratic solve (4 branches, emergency
       bounds). **rico's actual params (corrected Iter123):** `hmp2_ip_on_hmm2_ip` is a CASE override — rico sets
       `slope%rr=0, intrcpt%rr=1.25` (in rico_setup.txt), so the ratio is **1.25**, NOT the default
       `0.54+2.12e-5·max(host_dx,host_dy)` (host_dx=1e6 is irrelevant here). With omicron=0.5, zeta=0 → R=0.625,
       NON-emergency (the Iter116 "21.74/emergency" claim was wrong). The prescribed normal-space correlations are
       the default-array values (constant cloud=below): corr_chi_Ncn_n=0.09, corr_chi_rr_n=0.788, corr_chi_Nr_n=0.675,
       corr_rr_Nr_n=0.821. **Verification convention:** this routine is not f2py-exposed AND the stats oracle can't
       verify it — even the defining identity <hm>=a·f_p1·mu_1+(1-a)·f_p2·mu_2 fails across the stored stats (rel
       ~0.7), because rrm is end-of-step while precip_frac/mu_rr are within-step. Verified instead via the routine's
       CONTRACT (it preserves <hm> and <hm'^2>) — which REQUIRES the precip_fraction invariant pf=a·pf1+(1-a)·pf2
       (else Rmax≠ratio and variance is not preserved). **Consequence (Iter123):** the full hydromet-PDF chain from
       the rrm field (precip_frac→hydrometp2→calc_comp_mu_sigma_hm→rate) is NOT stats-validatable for accr/evap;
       only a RUNNING rico (within-step-consistent state) can. The autoconversion chain (Ncn only, no rr moments)
       IS fully validated (Iter120, median 4.7e-7).
       (iii) `Nc_in_cloud_to_Ncnm` (Iter117, `Nc_Ncn_eqns.py`) → the cloud-nuclei mean <Ncn> (= mu_Ncn, the
       autoconversion input), bit-to-bit vs f2py. (iv) the correlation conversions `corr_NN2NL`/`corr_NN2LL` +
       `calc_corr_chi_x` (Iter118, `pdf_utilities.py`, bit-to-bit vs f2py) — `calc_corr_chi_x` produces
       corr(chi,hm) from corr(rt,hm)/corr(thl,hm), the chi-side of the prescribed correlations.
       (v) `compute_mean_stdev` + `norm_transform_mean_stdev` (Iter131, `setup_clubb_pdf_params.py`) → the
       MOMENT-ASSEMBLY orchestration: stacks chi/eta/w/Ncn/hydrometeor component means+stdevs into the
       `(ngrdcol,nzt,pdf_dim)` arrays and transforms the lognormal vars to log space. The KK driver now routes
       its rr/Nr moments through these (bit-identical to the old inline path; rico oracle unchanged).
       (vi) `set_corr_arrays_to_default` (Iter132, `corr_varnce_module.py`) → the PRESCRIBED normal-space
       correlation arrays (in-cloud/below-cloud) from the fixed default tables; the KK driver now derives its
       rate correlations (chi-rr/chi-Nr/rr-Nr) from these instead of hardcoding (rico oracle unchanged). For
       rico cloud==below for every rate entry, so the rc-based selection is a no-op. Remaining step (b): the
       full per-gridbox `calc_corr_norm_and_cholesky_factor` (rc-based cloud/below selection + Cholesky) is only
       needed for SILHS sampling, not the upscaled-rate means — so step (b) for the NON-SILHS rico rates is
       effectively COMPLETE; the gating prerequisite is now the hydrometeor INFRASTRUCTURE (next).
     - **Hydrometeor INFRASTRUCTURE (the gating prerequisite for a RUNNING rico) — STARTED Iter135.**
       `hm_metadata` (the per-hydrometeor names/tols/flags/ratios + PDF indices, `init_pdf_hydromet_arrays`)
       is now ported+verified, the full transport solve (Iter136-139, `advance_one_hydrometeor`) + `calculate_K_hm`
       (Iter140), and (Iter140) the INIT wiring: `_check_unsupported_features` allows KK and `clubb_standalone`
       sets `hydromet_dim=2` + the hydromet mean array (rrm/Nrm=0) + `hm_metadata` (gated on KK; ARM 30-step
       regression still PASS). **rico now RUNS end-to-end in JAX (Iter141): 30 timesteps, no crash.** The Iter140
       step-2 blocker (rtm → −4.16e-11 at the dry domain top, where rico's sounding rt→0 above ~9000 m and the
       model top is 10000 m) was a missing **rt_tol floor**: the Fortran rico `rtm` min is exactly 1e-8 = rt_tol,
       so `clubb_standalone` now floors `rtm = max(rtm, rt_tol)` at init (no-op for the 15 cases; ARM still PASS).
       **rico is NOT YET bit-faithful** — `compare_runs --case rico` is the working end-to-end oracle.
       **Diagnosis (Iter142):** at step 1 the MEANS pass (rtm 8.5e-7) but ALL second-order MOMENTS fail ~1e-5,
       localised to the **trade inversion (zm≈1500 m, k=17-18)** where wp2 drops sharply 0.34→0.047 over one
       level (far sharper than ARM's well-mixed BL). The moment-budget terms each diverge ~1e-7 and ACCUMULATE
       to ~1e-5 in the prognostics. **Refined (Iter143) — REAL systematic difference, NOT FP:** the cascade
       root is the FLUX solve (`advance_xm_wpxp`) in rico's STRONG-turbulence regime (init em=1.0 →
       wp2=up2=vp2=2/3, vs the 15 faithful cases' em_min): `wpthlp`/`upwp`/`vpwp` diverge ~5e-8 abs at low-mid
       BL (k=4-6) where ARM is MACHINE-PRECISE. `upwp`/`vpwp` diverge with NO buoyancy term (rico clear at
       step 1) → it's the momentum/heat-flux solve at high wp2; `wprtp` stays exact (≈0). The flux error feeds
       the moments, amplified by Skw=wp3/wp2^1.5 → the ADG1 PDF → ~8e-5 by step 3. VERIFIED bit-exact (NOT the
       cause): tau_zm/Lscale (1.8e-16), sigma_sqd_w, wpthvp, the em/`l_tke_aniso`-partition init (byte-identical
       to Fortran). RULED OUT (cumulative): cloud, flags(=ARM), params(=ARM), wp3-limiter (both paths ported),
       splat, em-init, partition, `l_damp_wp2_using_em`, tau/Lscale. **Refined again (Iter144) — the rt/thl
       ASYMMETRY pins it to a thl-specific value, NOT strong-turbulence per se.** At step 1 everything **rt** is
       machine-exact (`wprtp` 4.5e-15, `rtp2` 1.5e-12, `rtpthvp` 2e-10) while everything **thl** diverges
       (`wpthlp` 4.9e-8, `thlp2` 1e-7, `wp2` 1.1e-5) — the two flux solves share lhs (`_sh11`)/ADG1 `lhs_ta`, so
       the error enters via a thl-specific input and tracks the **thl inversion gradient** (rt is well-mixed
       there). Step-1 buoyancy is ZERO (`thlpthvp=[0,0,0,0]`, rico clear) so it is NOT the thv coupling.
       **RULED OUT this iteration (with direct instrumentation, all reverted):** (1) the **monotonic flux
       limiter** — it DOES fire in JAX and matches the Fortran `rtm_mfl`/`thlm_mfl` budgets exactly (the zero
       `*_mfl` stats were just un-written diagnostics); the flux-solve output diverges 6.4e-3 at k18 but the mfl
       clamps both paths to the same bound. (2) the **penta solver** — rico's captured thl lhs/rhs solved EAGER
       == a pure-numpy Fortran-order `penta_lu` replica to **0 ULP** (`tests/test_penta_faithful.py`); jit adds
       only ~5.7e-14 (XLA FMA). (3) **XLA FP semantics** — EAGER (`jax.disable_jit`) rico gives the IDENTICAL
       divergence (`wp2` 1.124e-5) as jit. (4) **thl-specific assembly inputs** all machine-exact: `thlm_forcing`
       6.8e-21, `wpthlp_forcing` 0, C6thl const. So the locus is the lhs/rhs **assembly or init `thlm`/`wpthlp`**
       (the only unverified thl inputs; rt's init interp is faithful → likely a subtle wpxp-row assembly term).
       **ROOT CAUSE ISOLATED (Iter145) — the STRETCHED GRID (`grid_type=2`).** rico is the ONLY case with
       `grid_type=2` (a stretched zt grid from `deep_convection_128lev_27km_zt_grid.grd`); ALL 15 bit-faithful
       cases use `grid_type=1` (uniform). **Decisive A/B test:** temporarily set rico's namelist to
       `grid_type=1` (uniform, deltaz=40) → `compare_runs` prognostic failures **16 → 2**, every core var
       MACHINE-EXACT (wp2 6.8e-13, thlp2 5.3e-14, wpthlp 9.8e-14). So the entire divergence is **stretched-grid
       handling**; the rt/thl asymmetry, the inversion, "strong turbulence" were all symptoms of the stretched
       grid (rt is well-mixed so its near-uniform-value interpolation is least sensitive; thl/momentum carry
       the gradient). **f2py IS usable (Iter147 correction):** `test_call_tree_advance_xm_wpxp` fails only
       because the `clubb_python` wrapper is out of sync with its `.so` (passes `wp3, kh_zt` vs the `.so`'s
       `wp3_on_wp2, wp3_on_wp2_zt, kh_zt`); `clubb_f2py.f2py_advance_xm_wpxp` is callable DIRECTLY with the
       `__doc__`-introspected arg order — the input-matched comparison is the next step (see the f2py note in
       the testing section). The `grid_type=1` namelist swap is the other localiser. **Fixed (Iter145):** two `(1-w)` interpolation-weight bugs
       (the JAX computed the below-weight as `1 - w_above`; Fortran computes BOTH weights directly —
       grid_class.F90:2265/2269/2621/2625 — identical on uniform grids, ~1 ULP off on stretched): `zm2zt_jax`/
       `zt2zm_jax` (CLUBB_core/grid_class.py) + `_calc_zm2zt_weights` (derived_types/grid_class.py). **Verified
       safe** (ARM PASS, uniform-rico still machine-exact) but **NOT the dominant cause** — rico's divergence is
       bit-identical after the fix (the `(1-w)` residual is only ~1.1e-16). **Still to find:** the larger
       stretched-grid term producing ~1e-5. Ruled out: wm (8.7e-17), grid coords (0 ULP), interior metrics
       (exact), tau/Kh (≤1e-13). Density/pressure diverge ~1e-9 at the TOP only (hydrostatic integration). The
       `_ma` budget terms are the largest (thlm_ma 2.9e-5) but are post-advance diagnostics, not the cause.
       **BUG LOCALISED to `advance_xm_wpxp_jax` (Iter148) via the f2py input-matched comparison.** Captured
       rico's matched step-1 advance_xm_wpxp inputs (env-gated `$XMWP_CAP` hooks) + the RAW JAX output (pre-
       sponge/nudge), built the Fortran grid+UDTs, called `clubb_f2py.f2py_advance_xm_wpxp` directly with the
       SAME inputs (`run_scripts/compare_xm_wpxp_f2py.py`). With **`l_implemented=False`** (standalone — else
       the Fortran skips the xm mean-advection), f2py(JAX inputs) vs JAX output reproduces the EXACT rico full-
       run divergence (thlm 2.4e-6, um 2.5e-6, wpthlp 4.9e-8@k6, upwp 3.1e-7@k4, **wprtp machine-exact**). So
       given IDENTICAL inputs `advance_xm_wpxp_jax` ≠ Fortran → **the bug is IN that routine's assembly/solve**,
       not upstream. Flux divergence is at k4-6 (fine BL grid, large invrs_dzm), NOT the inversion → not the
       mfl. The penta solver is bit-faithful (Iter144) ⇒ the lhs/rhs ASSEMBLY differs on the stretched grid.
       **★ BUG FOUND & FIXED (Iter151) — the Iter150 "un-inspectable compiled-Fortran" conclusion was WRONG.**
       The decisive tool was the set of **individually-exposed f2py LHS-term routines** (`f2py_xpyp_term_ta_pdf_lhs`,
       `f2py_term_ma_zm_lhs`, `f2py_diffusion_zm_lhs` — see `run_scripts/cmp_terms_f2py.py`): feed the SAME captured
       inputs to each Fortran term routine AND the JAX term, diff directly (no solve, no clip, no shared-bug confound).
       Result: TA bit-exact (0.0), diffusion FP (1e-17), but **`term_ma_zm_lhs` differed 1.25e-6 (rel 7.8%) at the
       superdiagonal**. Root cause: **`weights_zm2zt` was stored with the two columns SWAPPED vs Fortran.** Fortran
       `calc_zm2zt_weights` (grid_class.F90:2621/2625): `m_above=(zt[k]-zm[k])/total` (idx0), `m_below=(zm[k+1]-zt[k])/total`
       (idx1); the JAX `derived_types/grid_class.py::_calc_zm2zt_weights` stored `idx0=dist_upper, idx1=dist_lower`
       (the OPPOSITE). The interp `zm2zt_api` compensated (correct OUTPUT) so it was invisible there, but the LHS term
       routines (`term_ma_zm_lhs_jax`, `xpyp_term_ta_pdf_lhs_jax` in `diffusion.py`) index `weights_zm2zt[:,:,m_above]`
       DIRECTLY, so they picked the wrong physical weight. **On uniform grids both weights are 0.5 → invisible → why all
       15 grid_type=1 cases always passed and only rico (grid_type=2) failed.** Fix: store `weights_zm2zt` in EXACT Fortran
       convention + flip `zm2zt_api`'s pairing to match (interp output invariant; uniform cases bit-identical → ARM/bomex
       still PASS, 23/23 unit tests pass). **After the fix rico step-1 is machine-exact: `advance_xm_wpxp` output vs the
       f2py oracle thlm 2.4e-6→1.1e-13, um→3.6e-15, wpthlp→3.7e-14; the full step-1 compare PASSES all 16 prognostics.**
       NOTE the physics interpolations (`CLUBB_core/grid_class.py::zm2zt_jax/zt2zm_jax`) were ALREADY Fortran-faithful — the
       bug was ONLY the `derived_types` stored-weight column order consumed by the term routines.
       **★ rico step-1 seed FIXED (Iter186) — the missing rtm/thlm fill_holes + the unfaithful IC floor.** The
       Iter152 "cubic-interp at k51" hypothesis was WRONG. Decomposing the rtm budget from the per-step stats showed
       every rtm-budget term machine-exact at k51 at step 1 EXCEPT **`rtm_cl`** (Fortran −4.5555e-11/s, JAX 0); ×dt(300s)
       = the entire 1.367e-8 seed. `rtm_cl` is the **`fill_holes_vertical` applied to the MEAN field rtm/thlm after the
       xm solve** (advance_xm_wpxp_module.F90:4974-5018, gated `fill_holes_type/=0 & solve_type/=um/vm`, threshold
       `rt_tol`/`thl_tol`, zt-level full range) — the JAX filled the variances but never rtm/thlm. AND the Iter141 IC
       hack `rtm=max(rtm,rt_tol)` **pre-floored** the dry top so the fill never fired. **The Iter141 claim "Fortran
       floors the IC to 1e-8" is FALSE** — Fortran's `rtm_old` entering step 1 is the bare sounding (~0/2e-19 at the dry
       top); the per-step fill raises it to 1e-8 each step, mass-conservingly pulling ~4.5e-11 from k51 (the topmost
       moist level). **Fix:** added the rtm/thlm fill_holes after MFL/before clip_covar (Fortran order MFL→pos_def→
       fill_holes→clip), REMOVED the IC floor (the fill keeps rtm≥rt_tol each step → no step-2 abort). **Result:** step 1
       FULLY bit-faithful (rtm 1.37e-8→**6.3e-15**), steps 1-4 bit-faithful (rtm/thlm/um/vm/wp2/rtp2/up2/vp2 pass the
       whole run); failures now start at **step 5** (was abort at 17). **Remaining = FP limit, not a bug:** the leading
       residual is the near-zero rt flux/covariance at the **stretched-grid dry top (k53)** — rtm/rtp2 there are
       bit-faithful (~1e-16) but `clip_covar`'s `|wprtp|≤√(wp2·rtp2)` amplifies FP-level rtp2 diffs (1e-16) into ~1e-11
       (same class as coriolis/jun25). **PROVEN FP-limited (Iter187), not assumed — 3 diagnostics:** (1) wprtp budget
       @k53 is machine-exact except `wprtp_cl` (a *conditional* clip change, NOT a systematic missing term — the JAX
       correctly implements clip_covar + `l_enable_relaxed_clipping`); (2) the divergence is a DISCRETE jump at step 5
       (1.2e-14→1.7e-11) = the clip-bound crossing, not gradual; (3) the dry-top rtp2 is machine-zero (4e-16 vs 7.9e-16,
       both at the rt_tol²=1e-16 floor) — matching ≈0 moisture moments to rel-1e-6 is impossible. Lesson: budget-decompose
       AND check the reference magnitude (≈ tol floor ⇒ FP) before concluding FP vs bug. No regression (arm/ekman/neutral/bomex/atex PASS). **Convention:** the `rtm_cl`/
       `thlm_cl` budget = fill_holes_vertical on the mean field; a mean field whose IC dips below its tol (a stretched
       dry top) MUST rely on this per-step fill, NOT an IC floor — an IC floor masks the donor-level mass transfer.
       General method that cracked it: **decompose the prognostic's BUDGET terms from the per-step stats** (`*_bt`,
       `*_ma`, `*_ta`, `*_cl`, `*_forcing`, …) to pin a seed to one term, before chasing the solve.
       **Prior context (Iter152):** RULED OUT the mfl (bit-faithful via `cmp_mfl_f2py.py`: rt 3.5e-18, thl 5.7e-14 —
       lle/hle are 0-based in JAX, +1 for f2py), the IC monotone-cubic interp (`_steffen_interp_1d`==`f2py_mono_cubic_
       interp`, d=0.0). General tools: **individual f2py term/routine oracles** fed captured JAX inputs (`cmp_terms_f2py.py`,
       `cmp_mfl_f2py.py`); `output/{case}_compare_{fort,jax}/{case}_stats.nc` per-step profiles — diff by LEVEL/STEP.
       Still needed for a full rico pass (now FP-limited, likely impractical): (a) ✓ init wiring (done, rico runs); (b) the rest of the
       `advance_hydrometeor` transport solve
       (`microphys_lhs` assembly DONE Iter138 = ½·diffusion_zt_lhs + term_ma_zt_lhs + sed_centered_diff_lhs +
       term_turb_sed_lhs + 1/dt; STILL NEEDED: `microphys_rhs` = hmm/dt + microphysics source + the
       Crank-Nicholson explicit ½-diffusion + the explicit turb-sed `term_turb_sed_rhs` (Vhmphmp_expc); the
       tridiag solve — ALL DONE Iter139 as `advance_one_hydrometeor`) + `fill_holes` (already ported); the
       per-hydrometeor transport solve is now COMPLETE. STILL NEEDED: (a) the init/loop wiring
       (hydromet_dim=2, rrm/Nrm init, the multi-hydrometeor + Ncm loop); (c) the microphysics call
       each step (precip_fraction →
       compute_kk_microphysics) wired into the timestep loop; (d) the advance_clubb_core thv coupling (the 4
       no-op-for-now buoyancy terms). All KK RATE/VELOCITY/COVARIANCE math is done — the remaining work is
       infrastructure + transport + wiring, end-to-end verifiable via `compare_runs.py --case rico` once it
       runs. The chi component moments (mu_chi/sigma_chi) come from the existing ADG1 PDF closure.
  3. **bugsrad radiation** (`Radiation/BUGSrad/`) — highest case count but huge; most of its cases
     ALSO need morrison, so it pairs with (4).
  4. **morrison microphysics**, then **SILHS**, then **COAMPS**.
2. **coriolis_test (Iter95: NOT a near-term target).** It is an analytic Foucault-pendulum benchmark:
   its `coriolis_test_parameters.in` **zeroes nearly all closure constants** (C1=C4=C7=C8=C14=0), and
   its purpose is to test the **nontraditional Coriolis terms** (`l_ho_nontrad_coriolis`), which are
   enabled only via a special flags file and are **not ported** in JAX (only the `l_ho_nontrad_coriolis=False`
   path exists). With default flags it is far from faithful (um off by ~370, wp2/up2/vp2 gross). The
   special init (harmonic-oscillator `em`, half-step leapfrog `upwp`) and `fcor_y` ARE ported. Its
   surface is `wpthlp_sfc=sens_ht`, `wprtp_sfc=latent_ht` (both 0 by default), `ustar=0` — the existing
   zero-flux stub is already correct. Upstream `model.in` has a **stale `parameter_file` path** (points
   at `tunable_parameters/` but the file is in `tunable_parameters_coriolis_cases/`); `run_scm.py` now
   resolves this via a basename fallback (Iter95). **Iter102-103: the uv-nudging term and the
   nontraditional Coriolis terms are now ported and verified bit-faithful at step 0** (with
   `--override l_ho_nontrad_coriolis=.true.`); the case's Foucault oscillator works. It still cannot
   pass the 30-step gate because the zeroed-closure (undamped) benchmark accumulates FP noise in upwp
   (rel ~4e-6 by step 10) — a genuine FP-limitation (the undamped oscillator does not decay errors).
   **PROVEN by budget decomposition (Iter189), not assumed:** step 1 is fully faithful (`upwp_bt` total tendency
   machine-eps 1.5e-17; the `upwp_nct`=0 is an UNWRITTEN stat, the nct term IS in upwp_bt which matches); at step 2
   every budget term is machine-eps (~1e-13) with FP sign-flips in the near-zero buoyancy/pressure terms, and
   vpwp/vm are EXACTLY 0 (1-D oscillator). NO step-1 systematic seed → no hidden bug (unlike jun25/rico, whose seeds
   showed at step 1). Physics complete; no further faithful work. (NB jun25, once grouped with coriolis as "FP", was
   a real bug — the stale wm_zm, fixed Iter188; coriolis is genuinely FP. Method: budget-decompose + check whether the
   value diverges while ALL tendency terms stay machine-eps ⇒ undamped FP amplification, not a missing term.)
3. xp2/xp3 sponge (`sponge_damp_xp2`/`xp3` for wp2/wp3/up2_vp2) — not yet ported (blocks any case
   enabling those; none of the current target cases do).
4. `pdf_closure_driver` for `ipdf_pre_advance_fields` path (not used by ARM)
5. `calc_lscale_directly` ✓ — verified working via ARM override test; `imu` import bug fixed

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
