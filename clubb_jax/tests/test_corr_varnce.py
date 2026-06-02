"""Verification of corr_varnce_module — the prescribed PDF-variable correlation arrays.

`set_corr_arrays_to_default` (corr_varnce_module.F90:282) builds the in-cloud/below-cloud
normal-space correlation arrays from the fixed 12×12 default tables. Not f2py-exposed, so this
verifies the build directly against the 6×6 block hand-extracted from the Fortran source tables
(corr_varnce_module.F90:147/163) for the standard KK PDF layout [chi, eta, w, Ncn, rr, Nr], and
confirms the rate-relevant correlations equal the values validated against the rico oracle.
"""
import numpy as np

from clubb_jax.src.CLUBB_core.corr_varnce_module import (
    set_corr_arrays_to_default, kk_prescribed_correlations, KK_PDF_TO_DEF,
    CORR_ARRAY_N_CLOUD_DEF, CORR_ARRAY_N_BELOW_DEF,
    II_CHI, II_ETA, II_W, II_NCN, II_RR, II_NR,
    init_pdf_hydromet_arrays, kk_hm_metadata)

# The 6×6 prescribed lower-triangular arrays, hand-extracted from the Fortran default tables.
# Row index = pdf var j, col index = pdf var i; entry [j,i] (j>i) = corr(j,i).
#                            chi    eta    w     Ncn    rr     Nr
_EXPECTED_CLOUD = np.array([[1.0,  0.0,  0.0,  0.0,  0.0,  0.0],   # chi
                            [-.6,  1.0,  0.0,  0.0,  0.0,  0.0],   # eta
                            [.09,  .027, 1.0,  0.0,  0.0,  0.0],   # w
                            [.09,  .027, .34,  1.0,  0.0,  0.0],   # Ncn
                            [.788, .114, .315, 0.0,  1.0,  0.0],   # rr
                            [.675, .115, .270, 0.0,  .821, 1.0]])  # Nr
_EXPECTED_BELOW = _EXPECTED_CLOUD.copy()
_EXPECTED_BELOW[II_ETA, II_CHI] = 0.3   # only difference: chi-eta -0.6 -> 0.3


def test_default_table_storage():
    """The column-major reshape reproduces Fortran storage: def[i,j] lower-triangular, with the
    known anchor entries in the right places."""
    d = CORR_ARRAY_N_CLOUD_DEF
    assert d.shape == (12, 12)
    np.testing.assert_array_equal(np.diag(d), np.ones(12))
    assert d[II_RR, II_CHI] == 0.788 and d[II_CHI, II_RR] == 0.0   # lower-triangular
    assert d[II_NR, II_CHI] == 0.675
    assert d[II_NR, II_RR] == 0.821
    assert d[II_NCN, II_CHI] == 0.09
    assert d[II_ETA, II_CHI] == -0.6
    assert CORR_ARRAY_N_BELOW_DEF[II_ETA, II_CHI] == 0.3
    print("  default 12×12 tables: column-major storage, anchors correct  PASS")


def test_set_corr_arrays_to_default():
    """The built 6×6 cloud/below prescribed arrays match the hand-extracted Fortran block."""
    cloud, below = set_corr_arrays_to_default(6, KK_PDF_TO_DEF)
    np.testing.assert_array_equal(cloud, _EXPECTED_CLOUD)
    np.testing.assert_array_equal(below, _EXPECTED_BELOW)
    # cloud and below differ ONLY in chi-eta (no rate-relevant entry).
    diff = np.abs(cloud - below)
    assert np.isclose(diff[II_ETA, II_CHI], 0.9)
    assert np.count_nonzero(diff) == 1   # exactly one entry differs
    print("  set_corr_arrays_to_default: 6×6 cloud/below == Fortran block; differ only in chi-eta  PASS")


def test_kk_prescribed_correlations():
    """The KK rate correlations derived from the prescribed array equal the rico-validated values."""
    c = kk_prescribed_correlations()
    assert c['corr_chi_rr'] == 0.788
    assert c['corr_chi_Nr'] == 0.675
    assert c['corr_rr_Nr'] == 0.821
    assert c['corr_chi_Ncn'] == 0.09
    print("  kk_prescribed_correlations: chi_rr=0.788, chi_Nr=0.675, rr_Nr=0.821, chi_Ncn=0.09  PASS")


def test_kk_hm_metadata():
    """init_pdf_hydromet_arrays produces rico's KK hydrometeor config: rr at array index 0,
    Nr at 1; names rrm/Nrm; rr a mixing ratio (Nr a number conc); neither frozen; the rico
    in-precip variance ratio 1.25 (slope=0, intrcpt=1.25); PDF indices iiPDF_rr=4, iiPDF_Nr=5,
    pdf_dim=6 (0-based, matching setup_clubb_pdf_params IIPDF_NCN+1)."""
    m = kk_hm_metadata()
    assert m.hydromet_dim == 2 and m.iirr == 0 and m.iiNr == 1
    assert m.hydromet_list == ["rrm", "Nrm"]
    assert list(m.l_mix_rat_hm) == [True, False]
    assert list(m.l_frozen_hm) == [False, False]
    np.testing.assert_allclose(m.hydromet_tol[0], 1.0e-10)
    np.testing.assert_allclose(m.hydromet_tol[1], 1.0e-10 / ((4.0 / 3.0) * np.pi * 1000.0 * (5.0e-3) ** 3))
    np.testing.assert_allclose(m.hmp2_ip_on_hmm2_ip, [1.25, 1.25])   # rico override
    assert m.iiPDF_chi == 0 and m.iiPDF_Ncn == 3
    assert m.iiPDF_rr == 4 and m.iiPDF_Nr == 5 and m.pdf_dim == 6
    # The PDF-variable indices must agree with the moment-assembly module.
    from clubb_jax.src.CLUBB_core.setup_clubb_pdf_params import IIPDF_NCN
    assert m.iiPDF_rr == IIPDF_NCN + 1 and m.iiPDF_Nr == IIPDF_NCN + 2
    # Default (non-override) ratio follows the linear law intrcpt + slope*max(dx,dy).
    d = init_pdf_hydromet_arrays(hydromet_dim=2, iirr=0, iiNr=1, host_dx=100.0, host_dy=50.0)
    np.testing.assert_allclose(d.hmp2_ip_on_hmm2_ip, [0.54 + 2.12e-5 * 100.0] * 2)
    print("  init_pdf_hydromet_arrays (KK): rrm/Nrm, tols, mix-rat/frozen flags, ratio 1.25, "
          "iiPDF_rr=4/Nr=5, pdf_dim=6  PASS")


if __name__ == "__main__":
    print("corr_varnce_module (prescribed PDF-variable correlation arrays) verification:")
    test_default_table_storage()
    test_set_corr_arrays_to_default()
    test_kk_prescribed_correlations()
    test_kk_hm_metadata()
    print("All corr_varnce_module tests PASSED.")
