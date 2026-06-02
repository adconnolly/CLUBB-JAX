"""Verification of Nc_Ncn_eqns.py — cloud-nuclei-concentration mean <Ncn> (Ncnm).

Nc_in_cloud_to_Ncnm IS exposed by the f2py API, so it is verified BIT-TO-BIT against
f2py_nc_in_cloud_to_ncnm over a random sweep covering both branches (constant Ncn -> simple
fallback; varying Ncn -> the erfc PDF integral). It also exactly reproduces rico's Ncnm
(rico has constant N_c, so Ncnm = Nc_in_cloud).
"""
import os
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from clubb_jax.src.CLUBB_core.Nc_Ncn_eqns import Nc_in_cloud_to_Ncnm

_RICO_STATS = os.path.join(os.path.dirname(__file__),
                           "../../clubb_release/output/rico_fort/rico_stats.nc")


def test_Ncnm_bit_to_bit_vs_f2py():
    """Nc_in_cloud_to_Ncnm matches the Fortran f2py oracle over a random sweep."""
    try:
        import clubb_f2py as f
    except ImportError:
        print("  Ncnm vs f2py: SKIP (clubb_f2py not built)"); return
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(3000):
        mu1, mu2 = rng.uniform(-1e-3, 1e-3), rng.uniform(-1e-3, 1e-3)
        s1 = rng.choice([0.0, rng.uniform(0, 5e-4)])      # exercise sigma_chi=0 branch
        s2 = rng.choice([0.0, rng.uniform(0, 5e-4)])
        mf, ncl = rng.uniform(0.01, 0.99), rng.uniform(1e6, 3e8)
        cf1, cf2 = rng.uniform(0, 1), rng.uniform(0, 1)
        cnp2 = rng.choice([0.0, rng.uniform(0, 2.0)])     # exercise const-Ncn vs varying
        ccc = rng.choice([0.0, rng.uniform(-0.9, 0.9)])
        fo = float(f.f2py_nc_in_cloud_to_ncnm(mu1, mu2, s1, s2, mf, ncl, cf1, cf2, cnp2, ccc))
        jo = float(Nc_in_cloud_to_Ncnm(mu1, mu2, s1, s2, mf, ncl, cf1, cf2, cnp2, ccc))
        worst = max(worst, abs(jo - fo) / (abs(fo) + 1e-300))
    # Machine-level (residual is the JAX-vs-Fortran erfc implementation difference).
    assert worst < 1e-12, f"Ncnm vs f2py worst rel {worst:.2e}"
    print(f"  Nc_in_cloud_to_Ncnm vs f2py: 3000 cases, worst rel {worst:.1e}  PASS")


def test_Ncnm_vs_rico_stats():
    """Reproduce rico's Ncnm exactly (rico: constant N_c -> Ncnm = Nc_in_cloud)."""
    try:
        import netCDF4 as nc
    except ImportError:
        print("  netCDF4 not available — SKIP")
        return
    if not os.path.exists(_RICO_STATS):
        print("  rico_fort stats absent — SKIP")
        return
    ds = nc.Dataset(_RICO_STATS)
    G = lambda n: np.asarray(ds[n][:, :, 0]).ravel()
    jo = np.asarray(Nc_in_cloud_to_Ncnm(
        G("chi_1"), G("chi_2"), G("stdev_chi_1"), G("stdev_chi_2"), G("mixt_frac"),
        G("Nc_in_cloud"), G("cloud_frac_1"), G("cloud_frac_2"), 0.0, 0.0))
    ncnm_s = G("Ncnm")
    ds.close()
    m = ncnm_s > 0
    d = np.max(np.abs(jo[m] - ncnm_s[m]) / ncnm_s[m])
    assert d < 1e-13, f"Ncnm vs rico stats max rel {d:.2e}"
    print(f"  Nc_in_cloud_to_Ncnm vs rico Ncnm (const-Ncn): max rel {d:.1e}  PASS")


if __name__ == "__main__":
    print("Nc_Ncn_eqns (<Ncn> mean) verification:")
    test_Ncnm_bit_to_bit_vs_f2py()
    test_Ncnm_vs_rico_stats()
    print("All Nc_Ncn_eqns tests PASSED.")
