#!/usr/bin/env python3
"""CLUBB standalone frontend — mirrors ``clubb_release/src/clubb_standalone.F90``.

Like the Fortran ``program clubb_standalone``, this is a minimalist frontend: read the
namelist filename from the command line and hand off to ``run_clubb`` (clubb_driver.py,
mirroring ``clubb_driver.F90``), which does init -> advance -> cleanup.

Usage:
    python -m clubb_jax.src.clubb_standalone input/case_setups/bomex_model.in
    python -m clubb_jax.src.clubb_standalone input/case_setups/bomex_model.in --quiet
"""
import sys
import time

from clubb_jax.src.clubb_driver import run_clubb


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python -m clubb_jax.src.clubb_standalone <namelist_path> [--quiet] [--max_steps=N]")
        print("  namelist_path: path to *_model.in file")
        print("  --quiet: suppress per-timestep output")
        print("  --max_steps=N: cap the number of timesteps")
        sys.exit(0 if sys.argv[1:] and sys.argv[1] in ('-h', '--help') else 1)

    namelist_path = sys.argv[1]
    l_stdout = '--quiet' not in sys.argv
    max_steps = None
    for arg in sys.argv[2:]:
        if arg.startswith('--max_steps='):
            max_steps = int(arg.split('=', 1)[1])

    t0 = time.time()
    state = run_clubb(namelist_path, l_stdout=l_stdout, max_steps=max_steps)
    elapsed = time.time() - t0

    print(f"Completed {state['ifinal']} timesteps in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
