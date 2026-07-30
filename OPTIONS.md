# CLUBB-JAX Options Reference

Every user-facing knob for running CLUBB-JAX, in one place: **command-line flags** of the run scripts and the
**environment variables** that control precision, JIT, the compute backend, and diagnostics. For *why* things
behave as they do (the whole-step JIT, the GPU scaling, the implicit CPU parallelism) see `DESIGN.md`; for the
correctness gates see `CLAUDE.md`.

> **TL;DR.** `python clubb_jax/run_scripts/run_scm.py <case>` runs the **JAX driver by default** (no flag needed).
> Pick the backend with `-cpu [N]` / `-gpu [N]`; run the Fortran oracle with `-fortran`. Precision, JIT, and
> caching are environment-controlled (`CLUBB_JAX_PRECISION`, `CLUBB_JAX_NO_*`).

---

## 1. `run_scm.py` — command-line flags

`python clubb_jax/run_scripts/run_scm.py <case_name> [options]`

The single positional argument `case_name` (e.g. `arm`, `bomex`, `gabls3`) is **required**. Flags use a single
leading dash (Fortran-CLUBB convention), e.g. `-max_iters`, not `--max-iters`.

### 1.1 Driver selection (which model runs)

The **JAX driver is the default** — give no executable-selector flag and JAX runs. At most one of these may be set.

| Flag | Runs | Notes |
|---|---|---|
| *(none)* | **JAX standalone driver** (`python -m clubb_jax.src.clubb_standalone`) | The default. The model this repo is. |
| `-fortran` | Compiled Fortran oracle `clubb_release/install/latest/clubb_standalone` | Was the old no-flag default; now explicit. |
| `-legacy` | Legacy Fortran build `clubb_release/bin/clubb_standalone` | From `compile.bash`. |
| `-exe PATH` | A specific compiled executable | Overrides `-legacy`. |
| `-python` | f2py Python driver (`clubb_python_driver.clubb_standalone`) | Needs `clubb_release/clubb_python_api/`. |
| `-driver_test` | `install/latest/clubb_driver_test` | Fortran driver-test binary. |

> The old `-jax` flag is **retired** — JAX is the default, so it is unnecessary. Passing `-jax` now errors.

### 1.2 Backend & device control (JAX runs only)

Mutually exclusive; valid only on a JAX (default) run — combining with `-fortran`/`-legacy`/`-exe`/`-python`/
`-driver_test` is an error. The optional integer caps how many devices are used. With **neither** flag the backend
follows the environment (see `JAX_PLATFORMS` / `jaxenv.sh` below).

| Flag | Effect |
|---|---|
| `-cpu` | Force CPU backend (`JAX_PLATFORMS=cpu`), use **all** available cores. |
| `-cpu N` | CPU backend, **pin the process to N cores** via `os.sched_setaffinity` (inherited by the child). |
| `-gpu` | Force GPU backend, use **all** visible GPUs. |
| `-gpu N` | GPU backend with `CUDA_VISIBLE_DEVICES=0..N-1`. |

**Why a cap is needed and how it works.** The JAX driver runs through XLA, whose CPU backend executes every op on
an Eigen thread pool sized to the logical-core count — so even a 1-column run spreads each kernel across **all**
cores with no code-level threading (measured ~2080 % CPU on a 32-core node). In jaxlib 0.10.2 the `XLA_FLAGS` /
`OMP_NUM_THREADS` knobs are **ignored** by the CPU runtime, so OS CPU affinity (`-cpu N`) is the only reliable cap
(verified linear: 4 cores → ~337 %, 8 → ~644 %). The GPU backend currently uses a **single device**, so `-gpu N>1`
only restricts visibility (to coexist with other jobs); it does not yet shard columns. Trade-offs of capping are in
`DESIGN.md` → "Backend & device control".

### 1.3 Case configuration files

| Flag | Default | Purpose |
|---|---|---|
| `-config DIR` | `clubb_release/input/tunable_parameters` | Directory holding `tunable_parameters.in`, `configurable_model_flags.in`, `silhs_parameters.in`. |
| `-params FILE` | from `-config` (or the case's `parameter_file`) | Tunable-parameters file (overrides `-config`). |
| `-flags FILE` | from `-config` | Configurable model-flags file (overrides `-config`). |
| `-silhs_params FILE` | from `-config` | SILHS parameters file (overrides `-config`). |

### 1.4 Grid

| Flag | Default | Purpose |
|---|---|---|
| `-zt_grid FILE` | unused | Use a zt grid file (`grid_type=2`). Mutually exclusive with `-zm_grid`. |
| `-zm_grid FILE` | unused | Use a zm grid file (`grid_type=3`). |
| `-nzmax NUM` | — | Max vertical levels; required when a `-zt_grid`/`-zm_grid` is given (no effect otherwise). |

### 1.5 Time stepping & iteration

| Flag | Default | Purpose |
|---|---|---|
| `-max_iters NUM` | from model file | Cap the number of timesteps (shortens `time_final`; never extends it). |
| `-dt_main SECONDS` | from model file | Main timestep. |
| `-dt_rad SECONDS` | from model file | Radiation timestep. |

### 1.6 Output & stats

| Flag | Default | Purpose |
|---|---|---|
| `-stats FILE` | `input/stats/standard_stats.in` | Fields to output; `-stats none` disables stats entirely. |
| `-tout SECONDS` | from model file | Stats output interval; `-tout 0` disables stats output. |
| `-out_dir DIR` | `clubb_jax/output/` (JAX) or `clubb_release/output/` (Fortran) | Output directory. JAX and the oracle default to **separate** trees so a JAX run can never clobber the oracle. |

### 1.7 Multi-column (the GPU data-parallel axis)

| Flag | Purpose |
|---|---|
| `-multicol N` | Duplicate the column into an `N`-column ensemble (legacy `dup_tweak` mode). The throughput axis on which JAX-GPU wins. |
| `-multicol SPEC` | Hypergrid spec forwarded to `create_multi_col_params.py -hr`, e.g. `-multicol C8/0.2:0.8/4`. |

### 1.8 Overrides & debug

| Flag | Purpose |
|---|---|
| `-override "K1=v1,K2=v2,..."` | Patch arbitrary namelist keys, e.g. `-override FLAG1=true,C2=2.0`. Keys not present are appended. |
| `-debug NUM` | CLUBB runtime-check level (0 = none … 3). Default from the model file. |

> `nvtx_run_scm.py` is an NVTX-annotated copy of `run_scm.py` for GPU profiling (Nsight). It accepts the **same**
> flags (including the default-JAX behavior and `-cpu`/`-gpu`).

---

## 2. Environment variables

Set these **before** launching (they are read at process start / first JAX op). The `-cpu`/`-gpu` flags above
override the backend env vars for that run.

### 2.1 Precision

| Variable | Default | Values | Effect |
|---|---|---|---|
| `CLUBB_JAX_PRECISION` | `double` | `double` / `single` (aliases: `float32`, `f32`, `32`, `real4`, `sp`) | Selects `jax_enable_x64`. `double` is bit-faithful; `single` (float32) runs and stays finite but is **not** bit-faithful — it is a memory/throughput exploration toggle (~½ the device memory; ~10–30 % *slower* on V100S since CLUBB is launch-bound, not FLOP-bound). |

### 2.2 JIT & compilation cache

| Variable | Default | Effect |
|---|---|---|
| `CLUBB_JAX_NO_WHOLE_STEP_JIT` | unset (whole-step JIT **on**) | Set to any non-empty value to disable the fused whole-step `jax.jit` and fall back to the eager leaf-by-leaf path (much slower; for debugging). |
| `CLUBB_JAX_NO_JIT_CACHE` | unset (persistent cache **on**) | Set to disable the on-disk JIT compilation cache (every process re-pays the first-step compile). |
| `JAX_COMPILATION_CACHE_DIR` | `~/.cache/clubb_jax_jit` | Relocate the persistent JIT cache directory. |

> The persistent cache is content-addressed on lowered HLO, so a code change transparently invalidates stale
> entries — the cached executable is byte-identical to a fresh compile (faithfulness/grad preserved by construction).

### 2.3 Compute backend (see also the `-cpu`/`-gpu` flags)

| Variable | Default | Effect |
|---|---|---|
| `JAX_PLATFORMS` | unset → GPU if available, else CPU | `cpu` forces the CPU backend. `-cpu`/`-gpu` set/clear this for you. |
| `CLUBB_JAX_CPU` | unset | Read **only by `jaxenv.sh`**: `CLUBB_JAX_CPU=1 source jaxenv.sh` exports `JAX_PLATFORMS=cpu`. |
| `CUDA_VISIBLE_DEVICES` | all GPUs | Standard CUDA device mask; `-gpu N` sets it to `0..N-1`. |
| `LD_LIBRARY_PATH` | (machine) | `jaxenv.sh` strips every `cuda*` entry so jaxlib loads its **bundled** CUDA libs (the cluster's system cuSPARSE is incompatible → otherwise a silent CPU fallback). |

### 2.4 Benchmarking & profiling

| Variable | Default | Effect |
|---|---|---|
| `CLUBB_JAX_BENCH` | unset | `=1` enables in-loop per-step wall timing in `advance_clubb_to_end` and prints a `BENCH_JSON {...}` line (parsed by `benchmark_backends.py`). |
| `CLUBB_JAX_BENCH_PHASES` | unset | `=1` adds phase timing (pre-core glue / jitted core / post-core) with `block_until_ready` boundaries. Read for the *proportional split*, not absolutes (the syncs serialize work the plain bench overlaps). |

### 2.5 Stock JAX diagnostics (not CLUBB-specific, but useful here)

| Variable | Effect |
|---|---|
| `JAX_LOG_COMPILES=1` | Log every XLA compile — used to diagnose runaway per-step recompilation (`grep -c "Compiling jit(scan)"`; a count that grows each step is the cache-blowup bug, see DESIGN "Operational gotchas"). |
| `JAX_DISABLE_JIT=1` | Run eagerly (op-by-op). Used by the f2py term-comparison debug scripts under `run_scripts/debug/`. |

---

## 3. Test / comparison scripts (flags + env)

These wrap `run_scm.py`. Common flags use `--double-dash` (argparse style).

| Script | Key flags | Env |
|---|---|---|
| `compare_runs.py` | `--case`, `--max-iters`, `--tout`, `--override`, `--tier {bit,physical}`, `--fortran-exe`, `--fortran-out-root`, `--jax-out-root` | `FORTRAN_EXE`, `FORTRAN_OUT_ROOT`, `JAX_OUT_ROOT` (defaults for the matching flags) |
| `compare_cases.py` | `--max-iters`, `--cases {default,tier_c,...}`, `--tier {bit,physical}`, `--list`, `--survey` | — |
| `compare_grad.py` | `--cases`, `--strict` | — |
| `benchmark_backends.py` | `--case`, `--steps`, `--ngrdcol N [N ...]`, `--backends {jax-gpu,jax-cpu,fortran}`, `--precision {double,single}`, `--out` | sets `CLUBB_JAX_BENCH`, `CLUBB_JAX_PRECISION`, `JAX_PLATFORMS` per run internally |
| `run_all_tests.py` | `-k SUBSTR`, `--timeout SEC`, `-j/--jobs N` | — |
| `run_scm.py -jax` *(retired)* | use the default (no flag) instead | — |

> **Do not run multiple `compare_*` jobs concurrently** — unpinned JAX processes grab every core and OOM-kill each
> other (looks like a spurious `rc=1`). If you must share a node, cap each with `-cpu N` (via the run it drives) or
> run them serially. See `DESIGN.md` → "Backend & device control" for the trade-offs.

---

## 4. Pointers

- **`jaxenv.sh.example`** — copy to git-ignored `jaxenv.sh`, set `_JAXENV`, `source` it (GPU by default;
  `CLUBB_JAX_CPU=1 source jaxenv.sh` for CPU). Handles the CUDA `LD_LIBRARY_PATH` fix.
- **`DESIGN.md`** — architecture, the performance/GPU arc, the "Backend & device control" trade-offs, correctness
  standard, and operational gotchas.
- **`CLAUDE.md`** — the test/gate commands and conventions.
