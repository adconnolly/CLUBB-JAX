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

### 2026-06-24/25 — Performance & GPU optimization arc (whole-step JIT → GPU → precision → profiling)
Condensed from five iterations on `main`/`optimize_performance`/`precision-flag`. Full per-step detail in
DESIGN.md "Performance, GPU, and …"; the per-iteration commits are in git (`58a3d19`, `368abba`, `c5631e0`,
`c33b0a2`).

- **Whole-step JIT** — wrapped `advance_clubb_core` in one `jax.jit` (`advance_clubb_core_jit`), fusing the ~23
  eager leaves into a single dispatch/step instead of hundreds of primitive pjits with a host round-trip between
  each. Static args = the shape/branch scalars (`nzm/nzt/ngrdcol`, `*_dim`, `l_implemented`, `clubb_config_flags`
  — an unregistered NamedTuple whose fields must stay static); everything else is a traced array or registered
  pytree. Gated to non-sampled steps (`CLUBB_JAX_NO_WHOLE_STEP_JIT=1` to disable): fusing the per-step stats
  writes balloons the compile, and `l_sample` gates only diagnostics, so stats-off jit ≡ the bit-faithful stats-on
  physics. **Bit-faithful** (`compare_runs arm` Result[bit] PASS) + **grad-transparent**. ~6.7× per-step at
  ngrdcol=1, matched stats.
- **GPU enabled** — CUDA-12 jaxlib (`jax-cuda12-plugin`/`pjrt` 0.10.2) in the jaxenv; the cluster `cuda12.8` on
  `LD_LIBRARY_PATH` shadows the bundled cuSPARSE (silent CPU fallback) so the env loader strips `cuda*` (tracked
  template `jaxenv.sh.example` → copy to a git-ignored `jaxenv.sh`, set `_JAXENV`, `source`). 1× Tesla V100S via SLURM.
- **Precision toggle** — `CLUBB_core/clubb_precision.py::configure_jax_precision()` (analog of Fortran
  `-precision`), env `CLUBB_JAX_PRECISION` (default `double`), replacing the 54 hard-coded `jax_enable_x64=True`
  sites. `double` byte-identical (still bit-faithful); `single` runs/finite but diverges at float32 level (not
  bit-faithful, expected). **Finding: float32 is a memory win, not a speed win** — ~10–30% *slower* on the V100S
  (CLUBB is launch/overhead-bound, no large GEMMs) but ~½ the device memory (≈2× column capacity).
- **Dead code:** removed the per-step `_calculate_thvm` (the driver computed `state['thvm']`, read nowhere — the
  core recomputes it; a port vestige). Bit-faithful; sped the plain bench at scale (the redundant compute scaled
  with the grid).
- **Profiling pinned the GPU floor** (`CLUBB_JAX_BENCH_PHASES=1` phase split + nz sweep + XLA HLO dump): the
  jitted **core dominates the GPU step (~55 ms) and is flat in BOTH ngrdcol and nz** → bound by a *fixed kernel
  count* (HLO: **1332 fusion kernels + 296 lax.scan while-loops ≈ ~1600 serial GPU kernels**), not solver depth or
  FLOPs (the same core is 3 ms on CPU at ngrdcol=1). **Refutes a parallel vertical solver** (the while-loops are a
  flat-in-nz minority; the 1332 elementwise/reduction fusions dominate). The small-batch GPU floor is inherent to
  the algorithm's many sequential ops; it is not the GPU's use case.

- **Scaling (ARM, steady ms/step, stats off, double; post-thvm-cleanup):**

  | ngrdcol | JAX-GPU | JAX-CPU | Fortran |
  |--------:|--------:|--------:|--------:|
  |       1 |    66.5 |     8.2 |     0.6 |
  |      64 |    75.7 |    41.5 |    16.8 |
  |     256 |    82.9 |    86.5 |    95.7 |
  |    1024 |    87.3 |   355.2 |   531.4 |

  GPU per-step ~flat (launch-bound) → throughput scales ~linearly with columns; CPU/Fortran grow ~linearly.
  **JAX-GPU overtakes both at ngrdcol≈256; at 1024 it is 6.1× faster than Fortran and 4.1× faster than JAX-CPU.**
  Short 1-few-column runs: Fortran wins on latency. Batch/ensemble (the ML/autodiff use case): **JAX-GPU is the
  fastest and the only differentiable backend.**

- **Remaining lever — `lax.scan` over timesteps** (would remove the ~11–19 ms/step Python/dispatch/host glue for
  long runs + enable memory-efficient multi-step `jax.grad`; the ~50 ms core launches still recur per step).
  **Blocker (scoped this iteration):** the case forcings reset arrays via numpy *in-place* mutation
  (`state['rtm_forcing'][:] = 0.0`) and `arm_sfclyr` is state-dependent — both work today only because the arrays
  are concrete, but under a full `lax.scan` trace every carry array is a tracer, so the forcings must first be
  rewritten as pure (no in-place mutation) functions and the evolving state assembled into a scan carry. That is a
  multi-iteration refactor, not a one-shot change.

### 2026-06-25 — Cross-regime faithfulness validation of the performance changes
- Confirmed the cumulative performance work (whole-step JIT, GPU enablement, precision toggle, thvm removal)
  preserved correctness across **three diverse regimes**: **arm** (continental closure), **bomex** (shallow
  cumulus), **gabls3_night** (stable BL + BUGSrad radiation + interactive soil) — all `compare_runs` **Result[bit]
  PASS**, 0 prognostic failures, Tier-C PASS. Differentiability is grad-transparent (arm probe identical
  eager-vs-jit; `jax.grad(jit(f)) == jax.grad(f)` for the forward-identical core). Together with the
  by-construction argument (double-mode whole-step jit is the same XLA computation, just fused) this establishes no
  faithfulness/differentiability regression from the optimization arc. (The full 20-case `compare_cases` sweep was
  started but is ~1.3 min/case on the eager per-step-stats path; the 3-regime targeted check + by-construction
  reasoning gives equivalent confidence far faster.)

### 2026-06-26 — Performance campaign closed (accepted as practical optimum)
- **Decision (user):** accept the current state as the practical performance optimum; stop the optimization loop.
  The implementation is **jittable** (whole-step JIT), **differentiable** (grad-transparent, validated),
  **faithful** (arm/bomex/gabls3_night bit-faithful + by construction), GPU-accelerated (**6.1× vs Fortran at
  ngrdcol=1024**), and the **CPU/GPU/Fortran scaling is documented** (DESIGN "Performance, GPU, …" + SCALING.md).
- **Deferred (documented future work, poor near-term ROI):** `lax.scan` over timesteps — the only remaining lever.
  It would remove the ~11–19 ms/step Python/dispatch/host glue on long runs (~20%) and enable memory-efficient
  multi-step `jax.grad`, but NOT the dominant ~50 ms GPU core (a proven inherent kernel-launch floor), and it
  requires rewriting the forcings from numpy in-place resets to pure functions + a scan carry (a sizeable,
  bit-faithfulness-risking refactor). Tracked in task #5 / DESIGN frontier. The batch/ensemble use case — where
  the differentiable GPU port already beats Fortran — does not need it.

### 2026-06-26 — JAX is the default driver; new `-cpu`/`-gpu` backend & device-count flags
- **`run_scm.py` now runs JAX by default** (the model this repo is). The `-jax` flag is **removed**; the no-flag
  default branch is the JAX standalone driver. The compiled Fortran oracle — previously the no-flag default — is now
  selected explicitly with **`-fortran`** (= `install/latest/clubb_standalone`), `-legacy` (= `bin/`), or `-exe PATH`.
  Updated all in-repo callers: dropped `-jax` from compare_cases/compare_runs/validate_case/update_golden and the JAX
  side of benchmark_backends/compare_flag_variants/run_jax_vs_fortran_cases; added `-fortran` to the bare-default
  Fortran side of the latter three; updated test_invariants/test_full_timestep_grad and two debug docstrings. Mirrored
  the same change into the untracked profiling copy `nvtx_run_scm.py`. (`clubb_release/` is the upstream submodule and
  was left untouched.)
- **New `-cpu [N]` / `-gpu [N]` flags** (mutually exclusive; JAX runs only) select the JAX compute backend and cap
  device use, via `run_scm.py::apply_jax_device_flags`: `-cpu` sets `JAX_PLATFORMS=cpu` and, when N>0, pins the
  process to N cores with `os.sched_setaffinity` (inherited by the child); `-gpu` clears `JAX_PLATFORMS` and, when
  N>0, sets `CUDA_VISIBLE_DEVICES=0..N-1`. Bare `-cpu`/`-gpu` uses all devices; neither follows the environment.
- **Why a flag was needed / what "implicit parallelism" is (now documented in DESIGN "Backend & device control").**
  The JAX driver runs through XLA, whose CPU backend executes every op on an Eigen thread pool sized to the logical-
  core count — so even `ngrdcol=1` spreads each kernel across **all** cores with no code-level threading (measured
  ~2080 % CPU). Verified empirically that jaxlib 0.10.2 **ignores** `XLA_FLAGS` eigen/host-device knobs and
  `OMP_NUM_THREADS` (~2300 % regardless); only OS CPU affinity caps it (4 cores → ~337 %, 8 → ~644 %, linear) — hence
  the `sched_setaffinity` approach. Trade-offs of capping (longer wall time vs freeing cores / avoiding the two-runs
  OOM hazard; near-free at small ngrdcol since the step is launch-bound, 1:1 throughput cost at large ngrdcol; GPU is
  single-device today so `-gpu N>1` only restricts visibility) are written up in the same DESIGN section.
- **Validated end-to-end:** default JAX run with `-cpu 4` pins to 4 cores and completes (arm, 2 steps); `-gpu 1`
  sets `CUDA_VISIBLE_DEVICES=0` and completes; `-jax` now errors (unrecognized); `-cpu`+`-fortran` is rejected.

### 2026-06-26 — Added OPTIONS.md (full flag + env-var reference)
- New top-level **`OPTIONS.md`** catalogues every user-facing knob: all `run_scm.py` flags (grouped: driver
  selection, `-cpu`/`-gpu` backend control, case files, grid, time stepping, output/stats, `-multicol`, overrides/
  debug) and every environment variable (`CLUBB_JAX_PRECISION`, `CLUBB_JAX_NO_WHOLE_STEP_JIT`,
  `CLUBB_JAX_NO_JIT_CACHE`, `JAX_COMPILATION_CACHE_DIR`, `JAX_PLATFORMS`, `CLUBB_JAX_CPU`, `CUDA_VISIBLE_DEVICES`,
  `CLUBB_JAX_BENCH`, `CLUBB_JAX_BENCH_PHASES`, `JAX_LOG_COMPILES`, `JAX_DISABLE_JIT`), plus a section on the
  comparison/test scripts (`compare_runs`/`compare_cases`/`compare_grad`/`benchmark_backends`/`run_all_tests` flags +
  `FORTRAN_EXE`/`FORTRAN_OUT_ROOT`/`JAX_OUT_ROOT`). Env-var semantics confirmed by grepping the actual
  `os.environ` reads. CLAUDE.md and DESIGN.md now point to it.

### 2026-07-30 — New case: COMBLE (marine cold-air outbreak), added to both drivers + compared
- **Added `comble` as a genuinely new case** (13 Mar 2020 Norwegian Sea cold-air outbreak, from the ARM COMBLE-MIP
  DEPHY-SCM forcing V2.4 — not one of the existing benchmarks). Converter `make_comble_case.py` translates the DEPHY
  netCDF → `comble_{sounding,forcings,sfc,model}.in`: geostrophic `ug/vg` forcing (time+height), no subsidence/adv
  tendencies (`forc_wap=forc_wa=0`), SST time series (247→279 K) driving `sfctype=1` ocean bulk fluxes, Morrison
  mixed-phase (fixed Nc=20 cm⁻³, like mpace_a), 74.5 °N, 50 m grid to 6 km.
- **Both drivers now dispatch `comble`.** Surface fluxes routed to the generic bulk-ocean routine
  `cloud_feedback_sfclyr` (sfctype=1) via a new `comble` branch in **both** `prescribe_forcings.F90` (surface select)
  and `prescribe_forcings.py`; large-scale forcing goes through the generic `l_t_dependent` file path. Note the two
  runtype selects: the LS-forcing select (1) is skipped when `l_t_dependent .and. .not. l_ignore_forcings`, so only
  the surface select needed a branch; the TKE-init select default (`em=em_min`) is used on both sides. The Fortran
  oracle was **incrementally recompiled** with Intel `ifx` (ninja, LD_LIBRARY_PATH `libimf`/`libifcore` fix).
- **JAX vs Fortran agree** (in-process `clubb_driver.run_clubb`, morrison-capable path — NOT the standalone
  `run_scm.py`, whose `clubb_case_initalization` init supports only microphys `none`). After 20 steps: max|Δthlm|=8e-4 K
  over 248–286 K, |Δrtm|=6e-8, |Δwp2|=3e-4 over 0–0.16; rcm/cloud_frac exactly 0 (pre-cloud spin-up). Nonzero-but-tiny
  = genuine JAX↔Fortran FP divergence, within the tiered "faithful" standard.
- Env note: the machine-specific `jaxenv` was wiped; rebuilt a CPU venv (jax 0.10.2 + netCDF4 + ninja/cmake) off
  anaconda3-2023.09. COMBLE case files + the two-driver `comble` branches are uncommitted (submodule + JAX src).

### 2026-07-31 — tune-to-obs: differentiable core hardened for coefficient gradients (branch tune-to-obs)
- **De-risk found the whole-driver `jax.grad` w.r.t. tunable coefficients was broken** (NaN for c_K/gamma_coef,
  detached zero for C1/C11) through a multi-step trajectory — a pre-existing gap (stock `probe_driver_grad` fails on
  `thlm` at 1 step too; `um`/momentum path was fine). Coeffs enter via a traced `state['clubb_params']` array
  (`clubb_params[:, i<name>]`), so no plumbing was needed — only the singular/detaching ops.
- **Four root-cause fixes, all forward-bit-identical at default params** (arm stays `Result[bit]` PASS vs Fortran,
  0 prognostic failures): `sfc_varnce_module` `_safe_sqrt`×3 + `_safe_pow` (wstar cube-root); `fill_holes`
  `fill_holes_wp2_from_horz_tke` safe division denominator; `advance_wp2_wp3_module` drop the `C1_varying`/
  `C11_varying` compute-shortcut where (it zeroed dC1/dC11 at C==Cb) for the smooth Skw form; `Skx_module` same for
  `gamma_coef`. Result: `probe_coeff_grad.py` (new) shows all 5 coeffs finite + FD-correct through 30 steps
  (worst rel 8e-3, C11 FD-step noise).
- **Route-around:** grad uses `fill_holes_type=global_fill` (1); the default `sliding_window` (2) has a
  non-differentiable dynamic-bound `fori_loop` — a separate follow-up. Next: the Adam tuning loop + obs target.
