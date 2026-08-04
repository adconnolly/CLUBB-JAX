#!/usr/bin/env python3
"""Does Nc_in_cloud actually change the mc3e output at a cloudy state?
Warm W steps (once), then run N steps at several Nc scales; report cloud/rain/thermo."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from clubb_jax.src.clubb_driver import init_clubb_case
from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end
REPO = "/burg-archive/glab/users/ac5006/CLUBB-JAX"
W = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
state = init_clubb_case(f"{REPO}/clubb_jax/output/mc3e_compare_jax/mc3e.in")
state['l_stats'] = False; state['stats_writer'] = None
if os.environ.get("GFILL"):
    state['flags'] = state['flags']._replace(fill_holes_type=1)   # global_fill (differentiable)
print("fill_holes_type =", int(state['flags'].fill_holes_type))
advance_clubb_to_end(state, l_stdout=False, max_steps=W)
nc0 = np.asarray(state['Nc_in_cloud'], np.float64)
irr = int(state['hm_metadata'].iirr)
print(f"WARMUP={W}: rcm_max={np.max(state['rcm']):.3e} cf_max={np.max(state['cloud_frac']):.3e} "
      f"rrm_max={np.max(state['hydromet'][...,irr]):.3e}")
for sc in (0.3, 1.0, 3.0):
    s = dict(state); s['Nc_in_cloud'] = nc0 * sc
    s['l_stats'] = False; s['stats_writer'] = None
    advance_clubb_to_end(s, l_stdout=False, max_steps=N)
    print(f"  Nc*{sc:4.1f}: sum|rcm|={np.sum(np.abs(s['rcm'])):.6e} "
          f"sum|rrm|={np.sum(np.abs(s['hydromet'][...,irr])):.6e} "
          f"sum thlm^2={np.sum(s['thlm']**2):.10e}")
