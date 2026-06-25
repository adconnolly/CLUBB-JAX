#!/bin/bash
# Source this to put the CLUBB-JAX python env (jax 0.10.2) on PATH with a working
# GPU backend. The pip-installed CUDA 12.9 libs (jax-cuda12-plugin) are shadowed
# by the cluster's system CUDA dirs on LD_LIBRARY_PATH (cuda12.8 ships an
# incompatible cuSPARSE → "Unable to load cuSPARSE" → silent CPU fallback), so we
# strip every cuda* entry and let jaxlib dlopen its own bundled libs.
#
#   source clubb_jax/run_scripts/jaxenv.sh        # GPU (default backend = gpu)
#   CLUBB_JAX_CPU=1 source clubb_jax/run_scripts/jaxenv.sh   # force CPU backend
#
# Idempotent; safe to source repeatedly.

_JAXENV=/burg/home/ac5006/scratch/jaxenv
export PATH="$_JAXENV/bin:$PATH"
export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -i 'cuda' | paste -sd:)"

if [ -n "$CLUBB_JAX_CPU" ]; then
    export JAX_PLATFORMS=cpu
else
    unset JAX_PLATFORMS
fi
