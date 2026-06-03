"""Term-by-term LHS comparison: JAX vs Fortran f2py oracle (Iter151).

Feeds the SAME captured inputs to each individually-exposed Fortran LHS-term routine
(f2py_xpyp_term_ta_pdf_lhs / f2py_term_ma_zm_lhs / f2py_diffusion_zm_lhs) and the JAX,
then diffs. This removes BOTH confounds of the residual/solve test: no shared-bug (the
Fortran term is an independent oracle) and no clipping/solver FP. Isolates which wpxp-row
matrix term diverges on the stretched grid.

Run after: TACAP=1 ... run_scm.py rico  (writes .tmp_claude/ta_term.npz)
       and  XMWP_CAP=1 ... run_scm.py rico (writes xmwp_in.npz)
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
ta = np.load(f'{T}/ta_term.npz')
def F(n): return np.asfortranarray(np.asarray(d[n], np.float64))
flg = {k: v for k, v in d['_flags']}
zt = np.asarray(d['gr_zt'], np.float64); zm = np.asarray(d['gr_zm'], np.float64)
ng, nzt, nzm = zt.shape[0], zt.shape[1], zm.shape[1]
nu6 = float(np.asarray(d['nu6']).flat[0])

clubb_api.init_err_info(ng)
cf = clubb_api.get_default_config_flags(); clubb_api.init_config_flags(cf)
gr, _ = clubb_api.setup_grid(nzmax=nzm, ngrdcol=ng, sfc_elevation=np.zeros(ng),
    l_implemented=True, l_ascending_grid=True, grid_type=2, deltaz=np.full(ng, 40.0),
    zm_init=np.zeros(ng), zm_top=np.full(ng, zm[0, -1]),
    momentum_heights=np.asfortranarray(zm), thermodynamic_heights=np.asfortranarray(zt),
    err_info=ErrInfo(ngrdcol=ng))
invrs_dzm = np.asfortranarray(np.asarray(gr.invrs_dzm, np.float64))
w_zm2zt   = np.asfortranarray(np.asarray(gr.weights_zm2zt, np.float64))
print(f"grid zt match={np.abs(np.asarray(gr.zt)-zt).max():.1e}  invrs_dzm shape={invrs_dzm.shape}  w_zm2zt shape={w_zm2zt.shape}")

def cmp(name, fort, jax):
    fort = np.asarray(fort); jax = np.asarray(jax)
    dd = np.abs(fort - jax)
    print(f"  {name:14s} shape{tuple(fort.shape)}  max|d|={dd.max():.3e} @ {np.unravel_index(dd.argmax(), dd.shape)}  "
          f"rel={dd.max()/ (np.abs(jax).max() or 1):.2e}")

print("\n--- TA (turbulent advection, wpxp row) : SAME coef fed to both ---")
coef = np.asfortranarray(np.asarray(ta['coef_wp2rtp'], np.float64))   # JAX coef (zt)
lhs_ta_f = clubb_f2py.f2py_xpyp_term_ta_pdf_lhs(
    coef, F('rho_ds_zt'), F('rho_ds_zm'), F('invrs_rho_ds_zm'),
    0, np.asfortranarray(np.zeros((ng, nzm))), np.asfortranarray(np.zeros((ng, nzm))))
cmp('lhs_ta', lhs_ta_f, ta['lhs_ta'])

print("\n--- MA_zm (mean advection, wpxp row) ---")
lhs_ma_f = clubb_f2py.f2py_term_ma_zm_lhs(F('wm_zm'), invrs_dzm, w_zm2zt)
cmp('lhs_ma_zm', lhs_ma_f, ta['lhs_ma_zm'])

print("\n--- DIFF_zm (diffusion, wpxp row) ---")
Kw6 = np.asfortranarray(np.asarray(ta['Kw6'], np.float64))
lhs_diff_f = clubb_f2py.f2py_diffusion_zm_lhs(
    Kw6, np.asfortranarray(np.zeros((ng, nzm))), np.asfortranarray(np.full(ng, nu6)),
    F('invrs_rho_ds_zm'), F('rho_ds_zt'))
cmp('lhs_diff_zm', lhs_diff_f, ta['lhs_diff_zm'])
