# CLUBB-JAX GPU Profiling

Scripts and notes for profiling the JAX driver with NVIDIA Nsight tools.

---

## Prerequisites

```bash
pip install nvtx          # NVTX Python bindings
# nsys ships with CUDA Toolkit / Nsight Systems — verify it's on PATH:
nsys --version
```

---

## `profile_nvtx.py` — NVTX range demo

Wraps each timestep's five main phases in named `nvtx.annotate()` ranges so the
Nsight Systems timeline shows coloured bands with GPU gaps between them.

| NVTX range | Colour | What it covers |
|---|---|---|
| `jit_warmup` | red | Step 1 only — XLA JIT compilation |
| `forcings` | blue | `calculate_thvm` + `prescribe_forcings` |
| `advance_clubb_core` | green | Closure physics — bulk of GPU work |
| `radiation` | orange | BUGSrad / simplified rad (every `dt_rad`) |
| `microphysics` | purple | KK / Morrison tendencies |

`jax.block_until_ready()` is inserted at the end of each phase so the host
synchronises with the device before the range closes — this is what makes GPU
idle gaps visible as blank space between bands in the Nsight timeline.

### Step 1 — capture a profile

Run from the **repo root** (`CLUBB-JAX/`):

```bash
mkdir -p profiling/reports

nsys profile \
    --trace=cuda,nvtx \
    --output profiling/reports/arm_profile \
    python3 profiling/profile_nvtx.py arm --max-iters 5
```

This produces `profiling/reports/arm_profile.nsys-rep`.

Other useful flags:

```bash
# Skip the JIT warmup step from the capture window (cleaner steady-state view):
nsys profile \
    --trace=cuda,nvtx \
    --capture-range=nvtx \
    --nvtx-capture="step_0002" \
    --output profiling/reports/arm_no_jit \
    python3 profiling/profile_nvtx.py arm --max-iters 6

# Disable stats output to remove NetCDF I/O from the profile:
python3 profiling/profile_nvtx.py arm --max-iters 5 --no-stats
```

### Step 2 — open in the Nsight Systems GUI

```
File → Open → profiling/reports/arm_profile.nsys-rep
```

Look for:
- **NVTX row** — coloured bands for each phase.
- **CUDA row** — kernel launches within each band.
- **Blank space inside a band** — GPU idle (host is computing or doing Python
  overhead before the next kernel launch). This is the gap to eliminate.

### Step 3 — CLI stats summary (no GUI needed)

```bash
nsys stats profiling/reports/arm_profile.nsys-rep
```

This prints tables for CUDA kernel time, NVTX range durations, and memory
transfers — useful for a quick text comparison across runs.

---

## What to look for (GPU gaps 101)

A "GPU gap" is time the device spends idle between kernel launches.  Common
causes in JAX workloads:

| Symptom | Likely cause |
|---|---|
| Large gap at start of `advance_clubb_core` | First-call XLA compilation (only step 1) |
| Gap between every kernel inside `forcings` | NumPy→JAX array copies (`jnp.asarray`) |
| Periodic spike every N steps | Python GC or `block_until_ready` sync |
| Gap before `radiation` only on some steps | `dt_rad > dt_main` — radiation runs every N steps |

---

## Adding your own ranges

```python
import nvtx

with nvtx.annotate("my_section", color=0xFF_00_FF_00):
    result = my_jax_function(x)
    jax.block_until_ready(result)   # sync so range end is accurate
```

Colours are `0xAA_RR_GG_BB` (alpha, red, green, blue) as 32-bit integers.

---

## Folder layout

```
profiling/
├── README.md           ← this file
├── profile_nvtx.py     ← NVTX harness (run under nsys)
└── reports/            ← .nsys-rep output files land here (git-ignored)
```
