"""Input-matched JAX-vs-Fortran comparison of the mono flux limiter (Iter152).

RESULT (Iter152): with the SAME inputs the JAX mono_flux_limiter == f2py to MACHINE PRECISION for
BOTH rt (3.5e-18) and thl (5.7e-14) → **the JAX mfl is bit-faithful on the stretched grid; it is NOT
the rico rt seed.** The seed is therefore an upstream rt-specific INPUT to advance_xm_wpxp (both that
routine and the mfl are now proven faithful given identical inputs).

CRITICAL HARNESS NOTE: the JAX `low_lev_effect`/`high_lev_effect` are 0-BASED level indices; the
Fortran f2py uses them as 1-BASED array indices → **must pass `lle+1, hle+1`**. Without the +1 the
f2py mfl fires spuriously (thl xm off by 0.9) — a harness artifact, not a JAX bug.

Run after:  MFLCAP=1 JAX_DISABLE_JIT=1 run_scm.py rico -max_iters 1
"""
import sys, numpy as np
from pathlib import Path
API = Path('/glade/work/adac/Claude/CLUBB-JAX/clubb_release/clubb_python_api')
sys.path.insert(0, str(API)); sys.path.insert(0, str(API / 'clubb_python'))
import clubb_f2py
from clubb_python import clubb_api
from clubb_python.derived_types.err_info import ErrInfo

T = '/glade/work/adac/Claude/CLUBB-JAX/.tmp_claude'
d = np.load(f'{T}/xmwp_in.npz', allow_pickle=True)
m = np.load(f'{T}/mfl_rt_in.npz'); mo = np.load(f'{T}/mfl_rt_out.npz')
zt = np.asarray(d['gr_zt'], np.float64); zm = np.asarray(d['gr_zm'], np.float64)
ng, nzt, nzm = zt.shape[0], zt.shape[1], zm.shape[1]

clubb_api.init_err_info(ng)
cf = clubb_api.get_default_config_flags(); clubb_api.init_config_flags(cf)
gr, _ = clubb_api.setup_grid(nzmax=nzm, ngrdcol=ng, sfc_elevation=np.zeros(ng),
    l_implemented=True, l_ascending_grid=True, grid_type=2, deltaz=np.full(ng, 40.0),
    zm_init=np.zeros(ng), zm_top=np.full(ng, zm[0, -1]),
    momentum_heights=np.asfortranarray(zm), thermodynamic_heights=np.asfortranarray(zt),
    err_info=ErrInfo(ngrdcol=ng))
F = lambda a: np.asfortranarray(np.asarray(a, np.float64))
Fi = lambda a: np.asfortranarray(np.asarray(a, np.int32))

for tag, st in [('thl', 1), ('rt', 2)]:   # solve_type: mono_flux_thlm=1, mono_flux_rtm=2
    m = np.load(f'{T}/mfl_{tag}_in.npz'); mo = np.load(f'{T}/mfl_{tag}_out.npz')
    xm_f, wpxp_f = clubb_f2py.f2py_monotonic_turbulent_flux_limit(
        st, 300.0, F(m['xm_old']), F(m['xp2']), F(m['wm_zt']), F(m['xm_forcing']),
        F(m['rho_ds_zm']), F(m['rho_ds_zt']), F(m['invrs_rho_ds_zm']), F(m['invrs_rho_ds_zt']),
        float(m['xp2_thr']), float(m['xm_tol']), int(m['l_implemented']),
        Fi(m['lle'] + 1), Fi(m['hle'] + 1),   # 0-based JAX -> 1-based Fortran
        int(m['tridiag_solve_method']), int(m['l_upwind_xm_ma']), int(m['l_spikefix']),
        F(m['xm_in']), F(m['wpxp_in']))
    xm_j = np.asarray(mo['xm_out'])[0]; wpxp_j = np.asarray(mo['wpxp_out'])[0]
    dx = np.abs(np.asarray(xm_f)[0] - xm_j); dw = np.abs(np.asarray(wpxp_f)[0] - wpxp_j)
    print(f"mfl {tag}: xm max|d|={dx.max():.3e}@k{int(dx.argmax())}  wpxp max|d|={dw.max():.3e}@k{int(dw.argmax())}")
