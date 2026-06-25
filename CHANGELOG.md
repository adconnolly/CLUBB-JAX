# CLUBB-JAX Changelog

Condensed work record. The full per-iteration history is recoverable from git; this file keeps the human-readable
arc. For design, the correctness standard, and conventions see `DESIGN.md`; for the per-file port map see
`TRANSLATION_STATUS.md`. Append one concise entry per session at the bottom of "Recent work".

---

## Numerical-accuracy refactor — final status

The refactor relaxed the **bit-faithfulness** gate to a **tiered numerical-accuracy standard** (DESIGN.md
"Correctness standard") to favor differentiability, simplicity, and accuracy. Outcome:

- **Differentiable (entirely in JAX): all 19 cases.** Whole-driver reverse-mode `jax.grad` through one
  `advance_clubb_to_end` step is finite + finite-difference-correct — gated by `compare_grad.py`. Achieved by a
  tracer-transparent toolkit (`CLUBB_core/tracer_numpy.py`: `_asarray`/`_xp`/`_iset` route to jnp under a trace,
  exactly numpy otherwise → normal runs bit-identical; `_safe_sqrt`/`_safe_pow` for clip-sqrt/fractional-pow), plus
  block-level tracer dispatch and diagnostic-skip / detach-under-trace for post-core diagnostics.
- **Faithful (vs Fortran): 20/20 DEFAULT_CASES PASS Tier-C** (19 strictly bit-faithful + mpace_a within Tier-C).
  Every differentiability change was forward-identical → zero faithfulness regression. The accuracy-lowering
  contrivances were removed (`parabolic_expax`, Morrison `real*4`, BUGSrad `sngl`/float32-π) → strictly more
  accurate there.
- **Tooling added:** tiered `validation.py`/`validate_case.py`, Tier-A `invariants.py`/`test_invariants.py`,
  golden-trajectory regression (`golden.py`/`update_golden.py`), `compare_grad.py`/`probe_driver_grad.py`,
  `mirror_audit.py`, `diagnose_divergence.py`.

---

## Condensed history (chronological)

**Phase 1 — Incremental port (2026-05-27 → 05-28, Iters 1–65).** Built the port routine-by-routine with in-loop
shadow comparison (run JAX beside the Fortran oracle, match to machine epsilon, remove the Fortran call): core grid
operators + LHS/RHS terms, the four advance functions (xm_wpxp / xp2_xpyp / wp2_wp3 / windm_edsclrm), the ADG1 PDF
closure, pre-advance diagnostics, and the initialization ports (hydrostatic, rcm_sat_adj, calculate_thvm). End
state: **zero Fortran calls in the advance loop**.

**Phase 2 — Cross-case faithfulness (→ 20 bit-faithful cases).** Extended beyond ARM by porting each case's
forcing/surface scheme and fixing per-case seeds (BC/thv/rc_coef, time-dependent wind forcing, bulk surface
schemes, cloud-droplet sedimentation, the cold-cloud `ice_supersat_frac`, the per-step `wm_zm` recompute). Added
gabls3 (BUGSrad correlated-k radiation + interactive soil_vegetation), the Morrison cases (mpace_a), and the
clex9 altocumulus pair. Characterized the FP-/oracle-limited cases (rico, coriolis_test, nov11_altocu,
dycoms2_rf02_morr/do/ds) as not-bugs.

**Phase 3 — Completeness loop (in-scope routine coverage).** Ported every remaining in-scope, oracle-validatable,
self-contained routine and unit-tested each: the KK PDF-integral mixed-moment machinery, BUGSrad cloud-overlap,
`ice_dfsn`, the GFDL droplet-activation CLUBB side, `inverse_hydrostatic`, all alternative PDF closures
(ADG2/LY93/3-D Luhar/new/TSDADG/new-hybrid, f2py end-to-end), `remapping_module` (both methods), and all benchmark
surface/forcing schemes. Brought in the CGILS/cloud_feedback family — Press[Pa]-sounding→altitude +
absolute-`T[K]`→θ init + the case-specific BUGSrad extended atmosphere + the forcing out-of-range zero-fill fix —
so cgils_s11/s12 reach Tier-C PASS and are differentiable.

**Phase 4 — Numerical-accuracy refactor.** Relaxed the gate to the tiered standard, removed the accuracy-lowering
contrivances, and made the whole driver `jax.grad`-differentiable (the tracer-transparent toolkit + conventions).
See "final status" above.

**Phase 5 — Mirror-refactor loop (2026-05-29 → 06-08, iters to ~1354).** Made the JAX *file and routine names*
mirror the `.F90` oracle without changing any numbers (every step byte-identical): relocated inlined subroutines to
their Fortran-home modules, un-inlined the pdf_closure / advance_xp2_xpyp / advance_xm_wpxp drivers, retired the
`_jax` suffix for single-subroutine mirrors and all shadow-comparison/iteration-tag scaffolding, deduplicated
constants against `constants_clubb`/`parameters_KK`, and aligned routine casing to the exact Fortran. Built
`mirror_audit.py` (a reproducible 7-dimension name/file/dir diff → PASS) with standing test guards
(`test_mirror_audit`, dead-import/function/config/param-roundtrip, source-grounded excusal tripwires). Ran a
behavioral-validation campaign: ~100 named mirrors now carry direct f2py-oracle bit-shadow unit tests, and every
load-bearing data structure (the 102 tunable params, the 67 config flags, the physical constants, the derived-type
field layouts) is directly pinned. Converged to a single deliberately-deferred routine, `pdf_closure_driver_zm`.

---

## Recent work

### 2026-06-20 — Microphysics import repair + config-flag variant sweep vs Fortran
- **Microphysics repair (independent of f2py).** rico (KK) and mpace_a (Morrison) couldn't run at all — the
  formatting/JIT refactor (5d77cd2) deleted `fill_holes_hydromet_clip_jax` and `morrison_hm_metadata` but left
  their callers, then the surviving callers of `Skx_func` and `fill_holes_vertical` still used the old arg order
  (arrays landing in jit static slots). Restored both functions and updated the KK/Morrison call sites to the
  current `(nz, ngrdcol, ...)` signatures. All 14 faithful cases now init + run 3 steps f2py-blocked, thlm finite.
- **Flag-variant verification harness.** Added `run_scripts/compare_flag_variants.py`: flips one config flag to a
  non-default value, runs JAX and Fortran with that flags file, diffs, classifies MATCH / DIFF / JAX_FAIL_LOUD.
  Swept 16 closure flags × {bomex, fire, gabls3_night} (cumulus / stratocumulus / stable): **the 10 genuinely-ported
  alternate paths are bit-faithful to Fortran in every regime; 0 DIFF.**
- **Guard fix the sweep found.** `l_vert_avg_closure=true` had diverged on bomex (123 vars) — the JAX has no code
  for it (vertically-averaged closure unported), and the standalone init path had drifted off `clubb_driver`'s
  unported-flag guard, so it silently ran the default closure. Mirrored `clubb_driver`'s unported closure/numerics
  guards into `clubb_case_initalization._check_unsupported_features`; the 6 unported swept flags now fail-loud in
  all three regimes, default cases unaffected.

### 2026-06-19 — Eliminate f2py from the standalone JAX path (stats decoupling)
- **What:** the standalone driver (`clubb_standalone` → `init_clubb_case` → `advance_clubb_to_end`) now runs with
  **`clubb_python`/`clubb_f2py` hard import-blocked**. The last f2py tie was the **stats subsystem**; it now uses the
  pure-Python `src/io/stats_writer.py::StatsWriter` (which parses the already-enriched `standard_stats.in`
  `name|grid|units|long_name` registry — no metadata regeneration needed).
- **Changes:** `StatsWriter` gained `update_budget` + Fortran-exact `l_in_budget` guards on begin/update/finalize
  (mirroring `stats_netcdf.F90`); `JaxStats` gained `from_writer`/`to_writer` and made its `clubb_api` import lazy;
  `init_clubb_case` builds `state['stats_writer']`; per-step replay, `Radiation/radiation.py`, `prescribe_forcings`,
  and `derived_types/converters.py` (now optional-import) route to the writer / pure-Python mirrors.
- **Latent bug fixed:** the driver passed a **0-based** step index to the stats sampler while the Fortran standalone
  loops `do itime = 1, ifinal` (**1-based**) — invisible when `stats_tout==stats_tsamp` (a record/step) but it shifts
  every multi-sample averaging window. Exposed by gabls2 (`tout=600`, `tsamp=60` → 10-sample average). Now passes
  1-based `itime`.
- **Validation:** f2py-blocked run completes end-to-end; **arm, bomex, fire, wangara, gabls2 all bit-faithful** to
  the Fortran oracle (0 vars over threshold). *Still f2py-coupled (out of scope here):* the `clubb_driver.py`
  `run_clubb` entry (err_info), and the `clubb_release/input/` data files. See `running_without_clubb_release.md`.

### 2026-06-19 — JAX-vs-Fortran speed assessment + persistent JIT cache (`main`)
- **Environment reconstruction.** This checkout arrived with no jax, an empty `clubb_release` submodule, and no
  compiled oracle. Rebuilt all of it: jax venv (jax 0.10.2 + numpy + netCDF4 + tabulate) at
  `/burg/home/ac5006/scratch/jaxenv`; `git submodule update --init clubb_release`; compiled the intel `ifx` +
  f2py oracle (`compile.py -python -precision double` → `install/intel_PRECdouble_PYTHON`). Two build blockers
  fixed: (1) cmake 3.28 module had a GLIBCXX mismatch → used a pip-wheel cmake; (2) every `ifx` Fortran link
  failed on `for_alloc_allocatable_handle` because a stale `parallel_studio_xe_2020` ifort runtime shadowed the
  HPCKit `libifcore` on `LIBRARY_PATH` → prepend `HPCKit/.../compiler/lib/intel64_lin`. The module's
  netcdf-fortran is ifort-built (ifx can't read its `.mod`), so CMake correctly FetchContent-builds netcdf-fortran
  with ifx. (`run_bindiff_all.py` needs `tabulate`.)
- **Comparison runs.** `run_jax_vs_fortran_cases.py` + `run_scm.py -jax`: ARM/30 steps is **bit-faithful** (PASS,
  0 vars over 1e-7). Speed: **Fortran 2.0 s vs JAX cold 133–147 s** — confirmed compile-dominated (244 XLA
  compiles via `JAX_LOG_COMPILES`), matching the DESIGN performance finding.
- **Fix applied (DESIGN Next Steps, step 1).** Enabled JAX's **persistent on-disk compilation cache** at import in
  `advance_clubb_core_module.py` (default `~/.cache/clubb_jax_jit`; `CLUBB_JAX_NO_JIT_CACHE=1` to disable,
  `JAX_COMPILATION_CACHE_DIR` to relocate). Cross-process cache hit → **warm 70–78 s (~47% faster)**; numerically
  transparent (ARM still bit-faithful, grad unaffected — cached executable is byte-identical). The residual ~70 s
  is tracing+MLIR-lowering of the 244 leaves; the whole-step-jit / `lax.scan` restructuring remains the open lever.

### 2026-06-19 — `formatting_and_jitting` branch (WIP)
- **Branch scope:** a formatting / Fortran-comment-copying pass; **JIT-friendly derived types** (`derived_types/`
  ConfigFlags/Grid/pdf_parameter made jit-static); a new **pure-JAX `src/io/` init path**
  (`namelist.py`/`sounding.py`/`surface.py`/`grid_file.py` + `derived_types/converters.py`); reintroduced the JAX
  `prescribe_forcings`; **removed `clubb_api` usage except for stats**; shrank stats buffers (memory/runtime win).
- **Status:** 12/20 DEFAULT_CASES bit-PASS; 7 cases fail-loud on features the new init path hasn't rewired
  (Morrison microphysics, `l_cloud_sed`, `l_soil_veg`, sponge); dycoms2_rf01 runs fine standalone (its
  `compare_cases` "crash" was node contention, not a bug). **Whole-step jit is NOT done** — only ~23 leaf
  functions are jitted while `advance_clubb_core`/the timestep loop stay eager, so first-call JIT compilation
  dominates runtime (full analysis + fix direction in DESIGN.md "Remaining Work → `formatting_and_jitting`").
- **This session:** fixed **35** signature-drift unit tests (refactor added leading `nzm/nzt/ngrdcol/gr` args,
  reordered args, `static_argnums`) + 2 src fixes (`set_sfc_value_of_flux_profiles` zeros `wpedsclrp`
  unconditionally; `Input_fields/sounding.py` stale `calculate_thvm` call). **12 tests still failing** — 6
  structural-audit guards (new `io/` files + the stats-`clubb_python` keep trip the mirror/100%-JAX guards) and 6
  genuine numerical/grad/oracle (see DESIGN.md next-step queue).

### 2026-06-08 — Documentation condensation
- Reviewed + corrected `TRANSLATION_STATUS.md` against the live repo and condensed it (392 → 218 lines): dropped
  the per-iteration narrative, replaced the stale "Total 202 / ported 128" headline with the current
  `mirror_audit.py` framing (PASS, 1 DEFERRED, 296 source files scoped out by subsystem), kept the Fortran↔JAX
  mapping tables with one-line notes.
- Condensed `DESIGN.md` (1214 → ~280 lines): cut ~600 lines of iter-NNN narrative blockquotes; preserved the
  operational core (repo structure, test instructions, verification oracles, gates, divergence/durability/grid-type
  guidance, differentiability conventions, the tiered correctness standard, critical conventions, remaining work,
  agent rules); pointed "What Has Been Built" at `TRANSLATION_STATUS.md`.
- Condensed this changelog (5547 → ~110 lines): collapsed 256 dated iteration entries into the phase summaries
  above (full detail remains in git).
- Proposed a condensed `CLAUDE.md` in `CLAUDE_PROPOSAL.md` for review (fixes stale refs: `generic_forcings.py` →
  `prescribe_forcings.py`, `radiation.py` → `radiation_module.py`, test-file count).

### 2026-06-24 — GPU enablement + whole-step JIT + CPU/GPU/Fortran scaling
- **Whole-step JIT (the open performance lever, closed).** Wrapped `advance_clubb_core` in one `jax.jit`
  (`advance_clubb_core_jit`; driver uses it on non-sampled steps, opt out `CLUBB_JAX_NO_WHOLE_STEP_JIT=1` —
  sampled steps stay eager since fusing the per-step `stats.update()` writes balloons the compile to minutes;
  `l_sample` gates only diagnostics, not physics, so stats-off jit ≡ the proven bit-faithful stats-on path).
  XLA now fuses the ~23 leaves into a single dispatch/step instead of hundreds of eager primitive pjits with a host
  round-trip between each. Static args = the shape/branch scalars (`nzm/nzt/ngrdcol`, `*_dim`, `l_implemented`, `clubb_config_flags`
  — an unregistered NamedTuple whose fields must stay static); everything else is a traced array or registered
  pytree (`gr`/`JaxStats`/`pdf_*`/`err_info`/`nu_vert_res_dep`/`sclr_idx`). **Validated: bit-faithful** (`compare_runs
  arm` Result[bit] PASS, 0 prognostic failures, Tier-C PASS) and **grad-transparent** (grad probe identical
  eager-vs-jit; only blocker is the pre-existing `fill_holes_sliding_window` dynamic-`fori_loop` reverse-mode
  limit). Per-step at matched stats: **GPU 2727 → 408 ms/step (6.7×)** at ngrdcol=1.
- **GPU enabled.** Installed CUDA-12 jaxlib (`jax-cuda12-plugin`/`pjrt` 0.10.2) into the jaxenv; the cluster's
  system `cuda12.8` on `LD_LIBRARY_PATH` shadows the bundled cuSPARSE (→ silent CPU fallback) so `jaxenv.sh` strips
  `cuda*` from `LD_LIBRARY_PATH`. 1× Tesla V100S exposed by SLURM (`--gres=gpu`).
- **Scaling documented** (`benchmark_backends.py`; env-gated `CLUBB_JAX_BENCH=1` per-step timing in
  `advance_clubb_to_end`; Fortran from its `CLUBB-TIMER`). ARM steady ms/step, stats off, sweeping the column
  axis `-multicol`: ngrdcol 1/8/64/256/1024 → **GPU** 72/70/71/101/113, **CPU** 8/15/63/110/335, **Fortran**
  0.6/1.9/17/94/529. **GPU per-step is nearly flat** (single-kernel, launch-latency bound) so its throughput
  scales ~linearly with column count, while Fortran (serial column loop) and JAX-CPU grow ~linearly. **JAX-GPU
  overtakes JAX-CPU at ngrdcol≈256 and Fortran at ngrdcol≈256–1024; at 1024 columns JAX-GPU is 4.7× faster than
  Fortran and 3.0× faster than JAX-CPU.** For 1-few-column short runs Fortran still wins on absolute latency; for
  ensemble/batch (the ML/autodiff use case) JAX-GPU is the fastest *and* the only differentiable backend. Full
  table + analysis in DESIGN.md "Performance, GPU, and …". Open levers: the per-step eager glue
  (forcings/radiation/microphysics + output host-transfer outside the core jit) and `lax.scan` over the timestep.

### 2026-06-25 — Single/double precision toggle + float32 GPU benchmark (`precision-flag`)
- **Precision toggle.** New `src/CLUBB_core/clubb_precision.py::configure_jax_precision()` centralizes the JAX `jax_enable_x64`
  decision (the analog of Fortran's compile-time `-precision single|double`), gated on env `CLUBB_JAX_PRECISION`
  (default `double`). Replaced the 54 hard-coded `jax.config.update("jax_enable_x64", True)` sites with a call to
  it, so precision is consistent process-wide (an inconsistent per-module setting would let the last import win).
  `double` is **byte-identical** to before — arm `compare_runs` Result[bit] PASS post-refactor (0 prognostic
  failures, Tier-C PASS). `single` runs and stays finite; vs double after 10 arm steps it diverges at float32 level
  (~1e-7 means, ~1e-5–1e-4 second moments/fluxes) → not bit-faithful (expected), for perf/memory exploration only.
- **Benchmark gains precision + GPU memory.** `benchmark_backends.py` takes `--precision double single`; BENCH_JSON
  now reports `precision` and `peak_mem_bytes` (`jax.devices()[0].memory_stats()`), and a second table prints peak
  GPU memory.
- **Finding: float32 is a MEMORY win, not a speed win for CLUBB (V100S, ARM).** f32 is ~10–30% *slower* than f64
  across ngrdcol 1–1024 (70→90 ms f64 vs 92→103 ms f32) but uses ~½ the device memory (503→273 MiB at 1024).
  CLUBB's step is launch/overhead-bound (many small tridiagonal solves + elementwise, no large GEMMs), so the
  V100S fp32 throughput edge never engages; the benefit is ~2× column capacity per GPU. Double stays the default.

### 2026-06-25 — Per-step phase profiling + dead-code (thvm) removal (`optimize_performance`)
- **Profiled where the per-step time goes** (opt-in `CLUBB_JAX_BENCH_PHASES=1` in `advance_clubb_to_end`;
  `block_until_ready` at pre-core/core/post boundaries; phase-timed totals run high — see methodology note). ARM
  steady ms/step (post-cleanup) — GPU: glue 11/15/19 + **core 52/60/58** + post ~0 at ngrdcol 1/256/1024; CPU: glue
  5/15 + **core 3/71** + post ~0 at 1/256. **The jitted core dominates the GPU step (~55 ms) and is flat in ngrdcol
  → kernel-launch bound, not FLOP-bound** (proof: the same core is 3 ms on CPU at ngrdcol=1 vs ~52 ms GPU — the gap
  is launch latency of the many small serial vertical-solver kernels). The pre-core glue is host-bound forcings
  (device↔host round-trips + per-step `zt2zm`/surface dispatches).
- **Reframed frontier:** small-ngrdcol GPU floor = core *kernel count* → real lever is a **parallel vertical
  solver** (cyclic reduction vs serial Thomas; FLOPs are free when launch-bound) and/or `lax.scan` over timesteps;
  the glue is a secondary, tractable win (on-device forcings). Large-ngrdcol GPU win already holds.
- **Dead code removed (and a real win at scale).** The driver computed `state['thvm']` every step but it is read
  nowhere (the core recomputes thvm internally and never received the driver's copy — a port vestige). Removed the
  per-step `_calculate_thvm` call + helper + import; ARM stays **bit-faithful** (`compare_runs` Result[bit] PASS,
  0 prognostic failures). Plain-bench effect (the redundant compute scaled with the grid): **JAX-GPU 101→83 ms at
  256 cols, 113→87 at 1024; JAX-CPU 63→42 at 64, 110→87 at 256**; ~0 at ngrdcol=1 (async dispatch already hid it).
  Refreshed the DESIGN scaling table — **GPU now 6.1× faster than Fortran at 1024 cols** (was 4.7×).
- **Methodology note:** the `BENCH_PHASES` `block_until_ready` boundaries *serialize* work the plain async bench
  overlaps, so phase-timed totals overstate absolutes (thvm looked like 26 ms/step under phase timing but cost ~0
  at ngrdcol=1 in async plain-bench). Use phase timing for the *proportional* split, plain bench for absolutes.
