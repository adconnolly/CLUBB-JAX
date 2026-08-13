#!/usr/bin/env python3
"""Regression probe: whole-driver multi-step jax.grad must stay FINITE.

Guards the fix for the multi-step reverse-mode NaN (bare sqrt of a quantity that
hits exactly 0 — w-variance wp2, zero shear ddzt_umvm_sqd, zero Lscale product —
has an inf reverse-mode gradient). dycoms2 is the trigger: its quiescent free
troposphere above the Sc inversion has wp2==0 / zero shear. Uses the coarse
dycoms2_200 fixture (200 levels) so the grad compile is fast.

Run:  python clubb_jax/tests/probe_multistep_grad.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from clubb_jax.src.clubb_driver import init_clubb_case
from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end
from clubb_jax.src.CLUBB_core.parameter_indices import ic_K

CASE = "/burg-archive/glab/users/ac5006/CLUBB-JAX/clubb_jax/output/dycoms2_200_compare_jax/dycoms2_200.in"

def multistep_grad(K):
    s = init_clubb_case(CASE); s['l_stats'] = False; s['stats_writer'] = None
    s['flags'] = s['flags']._replace(fill_holes_type=1)   # global_fill (differentiable)
    base = np.asarray(s['clubb_params'], np.float64)
    def loss(ck):
        st = dict(s); st['clubb_params'] = jnp.asarray(base).at[:, ic_K].set(ck)
        st['l_stats'] = False; st['stats_writer'] = None
        advance_clubb_to_end(st, l_stdout=False, max_steps=K)
        return jnp.sum(jnp.asarray(st['thlm']) ** 2)
    return float(jax.grad(loss)(float(base[0, ic_K])))

if __name__ == "__main__":
    ok = True
    for K in (2, 4, 6):          # K>=4 was NaN before the sqrt-cusp fixes
        g = multistep_grad(K)
        fin = np.isfinite(g)
        ok &= fin
        print(f"K={K}: d(sum thlm^2)/d(c_K) = {g:.6e}  finite={fin}")
    assert ok, "multi-step grad is NaN/inf — the sqrt-cusp differentiability fix regressed"
    print("PASS: multi-step whole-driver grad is finite")
