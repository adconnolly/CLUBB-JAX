import sys, numpy as np
from clubb_jax.src.clubb_driver import init_clubb_case
from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end
W=int(sys.argv[1])
s=init_clubb_case("/burg-archive/glab/users/ac5006/CLUBB-JAX/clubb_jax/output/mpace_a_compare_jax/mpace_a.in")
s['l_stats']=False; s['stats_writer']=None
irr=int(s['hm_metadata'].iirr); dt=float(s['dt_main'])
advance_clubb_to_end(s, l_stdout=False, max_steps=W)
print(f"mpace_a W={W} (+{W*dt/3600:.1f}h): rcm_max={np.max(s['rcm']):.3e} cf_max={np.max(s['cloud_frac']):.3e} rrm_max={np.max(s['hydromet'][...,irr]):.3e}")
