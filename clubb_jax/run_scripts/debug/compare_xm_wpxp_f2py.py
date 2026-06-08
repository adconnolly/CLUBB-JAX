"""f2py vs JAX input-matched comparison for advance_xm_wpxp (Iter148).

DEFINITIVE localiser for the rico grid_type=2 divergence. Workflow:
  1. Capture rico's matched step-1 advance_xm_wpxp inputs + raw JAX output (eager):
       XMWP_CAP=1 JAX_DISABLE_JIT=1 python clubb_jax/run_scripts/run_scm.py rico -jax -max_iters 1
     (writes .tmp_claude/xmwp_in.npz and .tmp_claude/xmwp_out.npz; the capture hooks in
      advance_clubb_core_module.py are gated on $XMWP_CAP, no-op otherwise.)
  2. Run this script: it builds the Fortran grid+UDTs, calls the f2py .so with the SAME
     inputs, and diffs f2py vs the raw JAX advance_xm_wpxp output.

RESULT (Iter148): with l_implemented=False the f2py output matches the JAX output to the
EXACT rico full-run divergence (thlm 2.4e-6, um 2.5e-6, wpthlp 4.9e-8, upwp 3.1e-7, wprtp
machine-exact). So given identical inputs, advance_xm_wpxp != Fortran advance_xm_wpxp
on the stretched grid -> the bug is IN advance_xm_wpxp (assembly/solve), not upstream.

Requires: PYTHONPATH including clubb_release/clubb_python_api.
"""
import sys, numpy as np
from pathlib import Path
API = Path('/glade/work/adac/Claude/CLUBB-JAX/clubb_release/clubb_python_api')
sys.path.insert(0, str(API)); sys.path.insert(0, str(API / 'clubb_python')); sys.path.insert(0, str(API / 'tests'))
import clubb_f2py
from clubb_python import clubb_api
from clubb_python.derived_types.err_info import ErrInfo
sys.path.insert(0, '/glade/work/adac/Claude/CLUBB-JAX')
from clubb_jax.tests.test_f2py_advance_xm_wpxp import call_f2py_advance_xm_wpxp

d = np.load('/glade/work/adac/Claude/CLUBB-JAX/.tmp_claude/xmwp_in.npz', allow_pickle=True)
def F(name, shape=None):
    a = np.asarray(d[name], dtype=np.float64)
    return np.asfortranarray(a)
flags_list = d['_flags']; flg = {k: v for k, v in flags_list}

zt = np.asarray(d['gr_zt'], np.float64)   # (1, nzt)
zm = np.asarray(d['gr_zm'], np.float64)   # (1, nzm)
ng = zt.shape[0]; nzt = zt.shape[1]; nzm = zm.shape[1]

# --- Build the Fortran grid for rico (grid_type=2 stretched) ---
clubb_api.init_err_info(ng)
cflags = clubb_api.get_default_config_flags(); clubb_api.init_config_flags(cflags)
gr, _ = clubb_api.setup_grid(
    nzmax=nzm, ngrdcol=ng, sfc_elevation=np.zeros(ng),
    l_implemented=True, l_ascending_grid=True, grid_type=2,
    deltaz=np.full(ng, 40.0), zm_init=np.zeros(ng), zm_top=np.full(ng, zm[0, -1]),
    momentum_heights=np.asfortranarray(zm), thermodynamic_heights=np.asfortranarray(zt),
    err_info=ErrInfo(ngrdcol=ng))
print(f"grid: nzt={gr.nzt} nzm={gr.nzm}  zt match={np.abs(np.asarray(gr.zt)-zt).max():.2e}  zm match={np.abs(np.asarray(gr.zm)-zm).max():.2e}")

clubb_params = F('clubb_params')
nu, _, _ = clubb_api.calc_derrived_params(gr=gr, ngrdcol=ng, grid_type=2,
    deltaz=np.full(ng, 40.0), clubb_params=clubb_params, nu_vert_res_dep=None,
    l_prescribed_avg_deltaz=False)
pdf_ic = clubb_api.init_pdf_implicit(gr.nzm, ng, sclr_dim=1)
clubb_api.set_sclr_idx(0, 0, 0, 0, 0, 0)
err = ErrInfo(ngrdcol=ng)

z = lambda *s: np.asfortranarray(np.zeros(s))
a = {
    'nzm': nzm, 'nzt': nzt, 'ngrdcol': ng, 'sclr_dim': 1,
    'sclr_tol': np.asfortranarray([1e-8]), 'dt': 300.0,
    'sigma_sqd_w': F('sigma_sqd_w'), 'wm_zm': F('wm_zm'), 'wm_zt': F('wm_zt'), 'wp2': F('wp2'),
    'lscale_zm': F('Lscale_zm'), 'wp3_on_wp2': F('wp3_on_wp2'), 'wp3_on_wp2_zt': F('wp3_on_wp2_zt'),
    'kh_zt': F('Kh_zt'), 'kh_zm': F('Kh_zm'), 'stability_correction': F('stability_correction'),
    'invrs_tau_c6_zm': F('invrs_tau_C6_zm'), 'tau_max_zm': F('tau_max_zm'), 'skw_zm': F('Skw_zm'),
    'wp2rtp': F('wp2rtp'), 'rtpthvp': F('rtpthvp'), 'rtm_forcing': F('rtm_forcing'),
    'wprtp_forcing': F('wprtp_forcing'), 'rtm_ref': F('rtm_ref'), 'wp2thlp': F('wp2thlp'),
    'thlpthvp': F('thlpthvp'), 'thlm_forcing': F('thlm_forcing'), 'wpthlp_forcing': F('wpthlp_forcing'),
    'thlm_ref': F('thlm_ref'), 'rho_ds_zm': F('rho_ds_zm'), 'rho_ds_zt': F('rho_ds_zt'),
    'invrs_rho_ds_zm': F('invrs_rho_ds_zm'), 'invrs_rho_ds_zt': F('invrs_rho_ds_zt'),
    'thv_ds_zm': F('thv_ds_zm'), 'rtp2': F('rtp2'), 'thlp2': F('thlp2'),
    'w_1_zm': F('w_1_zm'), 'w_2_zm': F('w_2_zm'), 'varnce_w_1_zm': F('varnce_w_1_zm'),
    'varnce_w_2_zm': F('varnce_w_2_zm'), 'mixt_frac_zm': F('mixt_frac_zm'), 'l_implemented': False,
    'em': F('em'), 'wp2sclrp': z(ng, nzt, 1), 'sclrpthvp': z(ng, nzm, 1), 'sclrm_forcing': z(ng, nzt, 1),
    'sclrp2': np.asfortranarray(np.full((ng, nzm, 1), 1e-6)), 'cx_fnc_richardson': F('Cx_fnc_Richardson'),
    'um_forcing': F('um_forcing'), 'vm_forcing': F('vm_forcing'), 'ug': F('ug'), 'vg': F('vg'),
    'wpthvp': F('wpthvp'), 'fcor': F('fcor'), 'fcor_y': F('fcor_y'), 'um_ref': F('um_ref'),
    'vm_ref': F('vm_ref'), 'up2': F('up2'), 'vp2': F('vp2'), 'uprcp': F('uprcp'), 'vprcp': F('vprcp'),
    'rc_coef_zm': F('rc_coef_zm'), 'clubb_params': clubb_params, 'ts_nudge': float(np.asarray(d['ts_nudge']).flat[0]),
    'iipdf_type': int(flg['iiPDF_type']), 'penta_solve_method': int(flg['penta_solve_method']),
    'tridiag_solve_method': int(flg['tridiag_solve_method']), 'fill_holes_type': int(flg['fill_holes_type']),
    'wp3_on_wp2_zt2': None,
}
for fn in ('l_predict_upwp_vpwp','l_ho_nontrad_coriolis','l_ho_trad_coriolis','l_diffuse_rtm_and_thlm',
           'l_godunov_upwind_wpxp_ta','l_upwind_xm_ma','l_uv_nudge','l_tke_aniso','l_diag_lscale_from_tau',
           'l_use_c7_richardson','l_lmm_stepping','l_enable_relaxed_clipping','l_linearize_pbl_winds',
           'l_mono_flux_lim_thlm','l_mono_flux_lim_rtm','l_mono_flux_lim_um','l_mono_flux_lim_vm',
           'l_mono_flux_lim_spikefix'):
    key = {'l_diag_lscale_from_tau':'l_diag_Lscale_from_tau','l_use_c7_richardson':'l_use_C7_Richardson'}.get(fn, fn)
    a[fn] = bool(int(flg.get(key, 0)))
a['l_stability_correct_kh_n2_zm'] = bool(int(flg.get('l_stability_correct_Kh_N2_zm', 0)))
for k in ('wprtp_cl_num','wpthlp_cl_num','upwp_cl_num','vpwp_cl_num'): a[k] = 0
a['rtm'] = F('_rtm_pre11'); a['wprtp'] = F('_wprtp_pre11'); a['thlm'] = F('_thlm_pre11'); a['wpthlp'] = F('_wpthlp_pre11')
a['um'] = F('_um_pre37'); a['upwp'] = F('_upwp_pre37'); a['vm'] = F('_vm_pre37'); a['vpwp'] = F('_vpwp_pre37')
a['sclrm'] = z(ng, nzt, 1); a['wpsclrp'] = z(ng, nzm, 1)
a['um_pert'] = z(ng, nzt); a['vm_pert'] = z(ng, nzt); a['upwp_pert'] = z(ng, nzm); a['vpwp_pert'] = z(ng, nzm)

out = call_f2py_advance_xm_wpxp(gr, nu, pdf_ic, err, a)

# Compare f2py output to the RAW JAX advance_xm_wpxp output (pre-sponge/nudge)
jo_d = np.load('/glade/work/adac/Claude/CLUBB-JAX/.tmp_claude/xmwp_out.npz')
print("\nf2py(JAX inputs) vs RAW JAX advance_xm_wpxp output (pre-sponge/nudge):")
for nm in ('rtm','thlm','um','vm','wprtp','wpthlp','upwp','vpwp'):
    fo = np.asarray(out[nm])[0]
    jo = np.asarray(jo_d[nm])[0]
    n = min(fo.size, jo.size)
    dd = np.abs(fo[:n]-jo[:n]); mx = np.abs(jo[:n]).max()
    kbl = dd[:21]
    print(f"  {nm:7s} max|d|={dd.max():.3e}@k{int(np.argmax(dd))} rel={dd.max()/mx if mx else 0:.2e}  BL(k0-20)max={kbl.max():.2e}@k{int(np.argmax(kbl))}")
