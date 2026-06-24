#!/usr/bin/env python3
"""
Profile CLUBB-JAX with NVTX ranges for Nsight Systems.

Wraps the five main phases of each timestep in named NVTX ranges so that
GPU gaps (time between kernel launches) become visible in the Nsight timeline.

Quick-start
-----------
1.  Install the nvtx Python package (already included in most CUDA environments):
        pip install nvtx

2.  Run under Nsight Systems (captures a .nsys-rep file you can open in the GUI):
        nsys profile \
            --trace=cuda,nvtx \
            --output profiling/arm_profile \
            python clubb_jax/run_scripts/profile_nvtx.py arm --max-iters 5

3.  Or generate a SQLite report for CLI analysis:
        nsys stats profiling/arm_profile.nsys-rep

4.  Open profiling/arm_profile.nsys-rep in the Nsight Systems GUI and look at the
    NVTX row — coloured bands mark each phase; gaps between CUDA kernels inside a
    band are GPU idle time.

Design notes
------------
- Only the JAX driver path is supported here; no Fortran oracle call.
- jax.block_until_ready() is called at the end of each timestep phase so that
  async dispatch does not blur the NVTX boundaries.  Remove it to measure
  throughput without synchronisation overhead.
- The first iteration is always slow (XLA JIT compilation).  The NVTX range
  "jit_warmup" wraps that first step so it stands out in the timeline.
- Keep --max-iters small (5-10) for a quick look; increase for a steady-state
  throughput picture.
"""
from __future__ import annotations

import argparse
import os
import sys

# ── path setup (mirrors run_scm.py) ──────────────────────────────────────────
RUN_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
JAX_ROOT    = os.path.normpath(os.path.join(RUN_SCRIPTS, "../.."))
CLUBB_ROOT  = os.path.normpath(os.path.join(JAX_ROOT, "clubb_release"))
for p in [JAX_ROOT, CLUBB_ROOT, os.path.join(CLUBB_ROOT, "clubb_python_api")]:
    if p not in sys.path:
        sys.path.insert(0, p)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import nvtx
except ImportError:
    sys.exit(
        "nvtx package not found.  Install it with:\n"
        "    pip install nvtx\n"
        "then re-run."
    )

import gc
import jax
import jax.numpy as jnp

from clubb_jax.src import clubb_driver
from clubb_jax.src.CLUBB_core import advance_clubb_core_module
from clubb_jax.src.CLUBB_core.advance_helper_module import calculate_thlp2_rad
from clubb_jax.src.CLUBB_core.calc_pressure import calculate_thvm
from clubb_jax.src.Benchmark_cases.prescribe_forcings import (
    prescribe_forcings_arm,
    prescribe_forcings_generic,
)
from clubb_jax.src.Microphys.microphys_driver import calc_microphys_scheme_tendcies
from clubb_jax.src.CLUBB_core.jax_stats_bridge import JaxStats


# ── colour palette (Nsight renders NVTX ranges with these) ───────────────────
_COLOURS = {
    "warmup":    0xFF_FF_00_00,   # red   – JIT compilation step
    "forcings":  0xFF_00_AA_FF,   # blue
    "core":      0xFF_00_CC_00,   # green – closure advance (most GPU work)
    "radiation": 0xFF_FF_88_00,   # orange
    "microphys": 0xFF_AA_00_FF,   # purple
    "stats":     0xFF_88_88_88,   # grey
}


def _block(state: dict, keys: list[str]) -> None:
    """Call jax.block_until_ready on selected state arrays.

    This forces a host↔device sync so the NVTX range end aligns with the
    actual kernel completion, making gaps visible.  Remove for throughput runs.
    """
    for k in keys:
        v = state.get(k)
        if v is not None and hasattr(v, "block_until_ready"):
            jax.block_until_ready(v)


def _run_one_step(state: dict, itime: int, time_current: float) -> None:
    """One instrumented timestep — mirrors advance_clubb_to_end.py."""

    dt_rad       = state["dt_rad"]
    dt_main      = state["dt_main"]
    rad_interval = int(dt_rad / dt_main)
    l_stats      = state["l_stats"]

    # ── stats begin ──────────────────────────────────────────────────────────
    with nvtx.annotate("stats_begin", color=_COLOURS["stats"]):
        if l_stats:
            sw = state["stats_writer"]
            l_sample, l_last_sample = sw.begin_timestep(itime)
        else:
            l_sample = l_last_sample = False
        state["l_sample"] = l_sample

    # ── forcings ─────────────────────────────────────────────────────────────
    with nvtx.annotate("forcings", color=_COLOURS["forcings"]):
        calculate_thvm(state)

        case = state.get("case_name", "")
        if case == "arm":
            prescribe_forcings_arm(state, itime, l_sample=l_sample)
        else:
            prescribe_forcings_generic(state, itime, l_sample=l_sample)

        state["rtm_forcing"]  = state["rtm_forcing"]  + state["rcm_mc"]
        state["thlm_forcing"] = state["thlm_forcing"] + state["thlm_mc"] + state["radht"]
        calculate_thlp2_rad(state)

        _block(state, ["rtm_forcing", "thlm_forcing"])

    # ── CLUBB core (closure advance — bulk of GPU work) ──────────────────────
    with nvtx.annotate("advance_clubb_core", color=_COLOURS["core"]):
        advance_clubb_core_module.advance_clubb_core(state)
        if l_stats:
            state["_jax_stats"].to_writer(state["stats_writer"])
        _block(state, ["wp2", "wp3", "wpthlp", "wprtp", "thlp2", "rtp2"])

    # ── radiation ────────────────────────────────────────────────────────────
    l_rad_itime = (itime % rad_interval == 0) or (itime == 1)
    if l_rad_itime:
        with nvtx.annotate("radiation", color=_COLOURS["radiation"]):
            from clubb_jax.src.advance_clubb_to_end import _advance_radiation
            _advance_radiation(
                state=state,
                time_current=time_current,
                l_sample=(l_stats and l_sample),
            )
            _block(state, ["radht"])

    # ── microphysics ─────────────────────────────────────────────────────────
    with nvtx.annotate("microphysics", color=_COLOURS["microphys"]):
        calc_microphys_scheme_tendcies(state, time_current)
        _block(state, ["rcm_mc", "thlm_mc"])

    # ── stats end ────────────────────────────────────────────────────────────
    with nvtx.annotate("stats_end", color=_COLOURS["stats"]):
        if l_stats and l_sample:
            from clubb_jax.src.advance_clubb_to_end import (
                _update_driver_stats,
                _end_timestep_stats,
            )
            _update_driver_stats(state, time_current)
        if l_last_sample:
            from clubb_jax.src.advance_clubb_to_end import _end_timestep_stats
            _end_timestep_stats(state, time_current)


def run_profiled(case_name: str, max_iters: int, l_stats: bool) -> None:
    """Initialise the JAX driver and run `max_iters` instrumented steps."""

    print(f"[profile_nvtx] initialising case '{case_name}' ...")
    state = clubb_driver.init_clubb_case(case_name, l_stats=l_stats)

    dt_main      = state["dt_main"]
    time_initial = state["time_initial"]
    n_steps      = min(state["ifinal"], max_iters)

    print(f"[profile_nvtx] running {n_steps} steps  (dt={dt_main}s)")
    print("[profile_nvtx] step 1 includes XLA JIT compilation — marked 'jit_warmup'")

    for itime_idx in range(n_steps):
        itime        = itime_idx + 1
        time_current = time_initial + (itime - 1) * dt_main

        if itime_idx == 0:
            # wrap the first (JIT) step in a distinct colour so it's obvious
            with nvtx.annotate("jit_warmup", color=_COLOURS["warmup"]):
                _run_one_step(state, itime, time_current)
        else:
            with nvtx.annotate(f"step_{itime:04d}", color=0xFF_20_20_20):
                _run_one_step(state, itime, time_current)

        gc.collect()
        print(f"  step {itime:4d}/{n_steps}")

    print("[profile_nvtx] done.")


def main() -> None:
    p = argparse.ArgumentParser(description="CLUBB-JAX NVTX profiling harness")
    p.add_argument("case_name", help="Case name (e.g. arm, bomex, dycoms2_rf01)")
    p.add_argument("--max-iters", type=int, default=5,
                   help="Number of timesteps to profile (default: 5)")
    p.add_argument("--no-stats", action="store_true",
                   help="Disable stats output (removes NetCDF I/O from profile)")
    args = p.parse_args()

    run_profiled(
        case_name  = args.case_name,
        max_iters  = args.max_iters,
        l_stats    = not args.no_stats,
    )


if __name__ == "__main__":
    main()
