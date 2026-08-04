#!/usr/bin/env python3
"""Validate the Morrison Nc gradient on a genuine cloud case (mpace_a).
Loss = sum(rcm^2) (cloud water, directly Nc-sensitive via autoconversion)."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from clubb_jax.src.clubb_driver import init_clubb_case
from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end
REPO = "/burg-archive/glab/users/ac5006/CLUBB-JAX"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
W = int(sys.argv[2]) if len(sys.argv) > 2 else 10
state = init_clubb_case(f"{REPO}/clubb_jax/output/mpace_a_compare_jax/mpace_a.in")
state['l_stats'] = False; state['stats_writer'] = None
state['flags'] = state['flags']._replace(fill_holes_type=1)
advance_clubb_to_end(state, l_stdout=False, max_steps=W)
print("after warmup: rcm_max=%.3e cloud_frac_max=%.3e" % (
    float(np.max(state['rcm'])), float(np.max(state['cloud_frac']))))
nc0 = np.asarray(state['Nc_in_cloud'], np.float64)
def _s(x): return x if isinstance(x, jax.core.Tracer) else jnp.asarray(x)
def loss(scale):
    s = dict(state); s['Nc_in_cloud'] = jnp.asarray(nc0) * scale
    s['l_stats'] = False; s['stats_writer'] = None
    advance_clubb_to_end(s, l_stdout=False, max_steps=N)
    return 1.0e6 * jnp.sum(_s(s['rcm']) ** 2)   # cloud water; scaled for readability
f0, g0 = jax.value_and_grad(loss)(jnp.asarray(1.0))
print(f"N={N} loss={float(f0):.6e} d/dNc={float(g0):.6e} finite={np.isfinite(float(g0))}")
eps = 1e-3
fd = (float(loss(jnp.asarray(1.0+eps))) - float(loss(jnp.asarray(1.0-eps)))) / (2*eps)
print(f"FD={fd:.6e} rel={abs(fd-float(g0))/(abs(fd)+abs(float(g0))+1e-30):.2e}")
