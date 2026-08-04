#!/usr/bin/env python3
"""Find when mc3e forms cloud: advance in chunks, report rcm/cloud_frac max + time."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from clubb_jax.src.clubb_driver import init_clubb_case
from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end
REPO = "/burg-archive/glab/users/ac5006/CLUBB-JAX"
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 300
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 20
state = init_clubb_case(f"{REPO}/clubb_jax/output/mc3e_compare_jax/mc3e.in")
state['l_stats'] = False; state['stats_writer'] = None
dt = float(state['dt_main'])
done = 0
print(f"dt={dt}s  tracking rcm_max/cloud_frac_max per {CHUNK} steps")
while done < TOTAL:
    advance_clubb_to_end(state, l_stdout=False, max_steps=CHUNK)
    done += CHUNK
    hr = done * dt / 3600.0
    rc = float(np.max(state['rcm'])); cf = float(np.max(state['cloud_frac']))
    rr = float(np.max(state['hydromet'][..., int(state['hm_metadata'].iirr)]))
    print(f"  step {done:4d} (+{hr:5.2f} h): rcm_max={rc:.3e} cf_max={cf:.3e} rrm_max={rr:.3e}")
    if rc > 1e-7:
        print(f"  >>> CLOUD at step {done} (~{hr:.2f} h into the IOP)");
