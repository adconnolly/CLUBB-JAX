"""JAX port of corr_varnce_module.F90 — prescribed PDF-variable correlation arrays.

`set_corr_arrays_to_default` builds the in-cloud and below-cloud prescribed normal-space
correlation arrays (`corr_array_n_cloud`/`corr_array_n_below`) used as the hydrometeor-PDF
correlations when a case has no `corr_array.in` override (rico and the other KK cases). The
values come from the fixed `corr_array_n_{cloud,below}_def` 12×12 tables (oracle
corr_varnce_module.F90:147), indexed by `def_corr_idx` (the pdf-variable → default-table
index map). The arrays are lower-triangular: entry `[j,i]` (j>i) is corr(pdf var j, pdf var i).

For the KK PDF layout [chi, eta, w, Ncn, rr, Nr] the pdf indices coincide with the default-table
indices (identity map), so the prescribed array is the upper-left 6×6 block. The KK rate functions
consume corr(chi,rr)=0.788, corr(chi,Nr)=0.675, corr(rr,Nr)=0.821 (and corr(chi,Ncn)=0.09, unused
for rico since N_cn is constant). cloud and below differ ONLY in corr(chi,eta) (-0.6 vs 0.3), which
no rate consumes — hence the rc-based cloud/below selection is a no-op for the rate path.

NOT ported (SILHS-oriented, deferred): the `calc_corr_norm_and_cholesky_factor` adjustments
(ADG1 w-chi/w-eta zeroing, the Ncn below->cloud copy, the corr(hm,eta) estimate) and the Cholesky
factorization — they are needed for SILHS sampling, not the upscaled-KK rate means, and do not
touch the rate-relevant entries.
"""
import numpy as np

from clubb_jax.src.CLUBB_core.constants_clubb import (
    rho_lw, rho_ice, mvr_rain_max, mvr_ice_max, mvr_snow_max, mvr_graupel_max,
)

# Default-table indices (0-based) — order: chi, eta, w, Ncn, rr, Nr, ri, Ni, rs, Ns, rg, Ng.
II_CHI, II_ETA, II_W, II_NCN, II_RR, II_NR = 0, 1, 2, 3, 4, 5
_D_VAR_TOTAL = 12

# The two default correlation tables, written ROW-BY-ROW exactly as in the Fortran source
# (corr_varnce_module.F90:147/163). The Fortran `reshape` is column-major, so reshaping this
# row-major flat list with order='F' reproduces the Fortran STORAGE: _DEF[i,j] = Fortran def(i+1,j+1)
# (lower-triangular). cloud and below differ only in the chi-eta entry (-0.6 vs 0.3).
#                 chi    eta    w      Ncn    rr     Nr     ri     Ni     rs     Ns     rg     Ng
_ROWS_CLOUD = [
    [1.0,  -.6,   .09,   .09,   .788,  .675,  .240,  .222,  .240,  .222,  .240,  .222],   # chi
    [0.0,  1.0,   .027,  .027,  .114,  .115, -.029,  .093,  .022,  .013,  0.0,   0.0 ],   # eta
    [0.0,  0.0,   1.0,   .34,   .315,  .270,  .120,  .167,  0.0,   0.0,   0.0,   0.0 ],   # w
    [0.0,  0.0,   0.0,   1.0,   0.0,   0.0,   .464,  .320,  .168,  .232,  0.0,   0.0 ],   # Ncn
    [0.0,  0.0,   0.0,   0.0,   1.0,   .821,  0.0,   0.0,   .173,  .164,  .319,  .308],   # rr
    [0.0,  0.0,   0.0,   0.0,   0.0,   1.0,   .152,  .143,  0.0,   0.0,   .285,  .273],   # Nr
    [0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   1.0,   .758,  .585,  .571,  .379,  .363],   # ri
    [0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1.0,   .571,  .550,  .363,  .345],   # Ni
    [0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1.0,   .758,  .485,  .470],   # rs
    [0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1.0,   .470,  .450],   # Ns
    [0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1.0,   .758],   # rg
    [0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1.0 ],   # Ng
]

_flat_cloud = [v for row in _ROWS_CLOUD for v in row]
CORR_ARRAY_N_CLOUD_DEF = np.array(_flat_cloud, dtype=np.float64).reshape(
    (_D_VAR_TOTAL, _D_VAR_TOTAL), order='F')
# below = cloud with the single chi-eta entry changed (-0.6 -> 0.3).
_flat_below = list(_flat_cloud)
_flat_below[1] = 0.3                      # row chi, column eta (2nd element of the chi row)
CORR_ARRAY_N_BELOW_DEF = np.array(_flat_below, dtype=np.float64).reshape(
    (_D_VAR_TOTAL, _D_VAR_TOTAL), order='F')



def set_corr_arrays_to_default(pdf_dim, pdf_to_def):
    """Build the prescribed (lower-triangular) cloud/below correlation arrays for a PDF of size
    pdf_dim. `pdf_to_def[i]` is the default-table index (0-based) of pdf variable i. Oracle
    set_corr_arrays_to_default (corr_varnce_module.F90:282).

    Returns (corr_array_n_cloud, corr_array_n_below), each (pdf_dim, pdf_dim) with unit diagonal,
    zero upper triangle, and corr_array[j,i] (j>i) = corr(pdf var j, pdf var i)."""
    cloud = np.zeros((pdf_dim, pdf_dim), dtype=np.float64)
    below = np.zeros((pdf_dim, pdf_dim), dtype=np.float64)
    for i in range(pdf_dim):
        cloud[i, i] = 1.0
        below[i, i] = 1.0
    for i in range(pdf_dim - 1):
        for j in range(i + 1, pdf_dim):
            idx_i, idx_j = pdf_to_def[i], pdf_to_def[j]
            # def is lower-triangular (idx>=jdx nonzero); pick the populated orientation.
            a, b = (idx_i, idx_j) if idx_i > idx_j else (idx_j, idx_i)
            cloud[j, i] = CORR_ARRAY_N_CLOUD_DEF[a, b]
            below[j, i] = CORR_ARRAY_N_BELOW_DEF[a, b]
    return cloud, below


def assert_corr_symmetric(corr_array_n, tol=1.0e-6, eps=1.0e-10):
    """Validity check for a normal-space correlation matrix (corr_varnce_module.F90:assert_corr_symmetric).

    Returns True iff the (pdf_dim, pdf_dim) array is symmetric within ``tol`` (1e-6) AND has a unit diagonal
    within ``eps`` (constants_clubb ``eps`` = 1e-10). The Fortran sets ``err_info%err_code = clubb_fatal_error``
    and prints when the assertion fails (then ``return`` — it does not error-stop, and the err_code is not
    exposed by the f2py wrapper); the JAX path never error-stops, so this returns the boolean verdict instead.
    Pure-numpy concrete check (not in any gradient path).
    """
    c = np.asarray(corr_array_n, dtype=np.float64)
    symmetric = bool(np.all(np.abs(c - c.T) <= tol))
    unit_diag = bool(np.all(np.abs(np.diagonal(c) - 1.0) <= eps))
    return symmetric and unit_diag


def kk_prescribed_correlations():
    """The normal-space correlations the upscaled-KK rate functions consume, derived from the
    prescribed in-cloud array for the standard KK PDF layout [chi, eta, w, Ncn, rr, Nr].

    Returns dict with corr_chi_rr, corr_chi_Nr, corr_rr_Nr, corr_chi_Ncn. cloud == below for all
    of these (they differ only in chi-eta), so the rc-based cloud/below selection is a no-op."""
    # KK PDF layout [chi, eta, w, Ncn, rr, Nr]; def_corr_idx maps each PDF variable to its default-table
    # column (mirrors the Fortran set_corr_arrays_to_default <- def_corr_idx chain).
    kk_layout = HmMetadata(hydromet_dim=2, iiPDF_rr=II_RR, iiPDF_Nr=II_NR)
    pdf_to_def = tuple(def_corr_idx(i, kk_layout) for i in range(6))
    cloud, _below = set_corr_arrays_to_default(6, pdf_to_def)
    return {
        'corr_chi_rr': cloud[II_RR, II_CHI],
        'corr_chi_Nr': cloud[II_NR, II_CHI],
        'corr_rr_Nr': cloud[II_NR, II_RR],
        'corr_chi_Ncn': cloud[II_NCN, II_CHI],
        # eta/w rows — the additional normal-space correlations the second-moment covariance driver
        # (KK_upscaled_covar_driver) consumes. corr_w_* are 0 in the default KK array (and ADG1 zeros
        # corr_w_chi/corr_w_eta anyway). cloud == below for all of these.
        'corr_eta_rr': cloud[II_RR, II_ETA],
        'corr_eta_Nr': cloud[II_NR, II_ETA],
        'corr_eta_Ncn': cloud[II_NCN, II_ETA],
        'corr_w_rr': cloud[II_RR, II_W],
        'corr_w_Nr': cloud[II_NR, II_W],
        'corr_w_Ncn': cloud[II_NCN, II_W],
    }


# ---------------------------------------------------------------------------
# Hydrometeor metadata setup (init_pdf_hydromet_arrays_api, corr_varnce_module.F90:455).
# The data structure the microphysics + hydrometeor advance consume: per-hydrometeor names,
# tolerances, mixing-ratio/frozen flags, in-precip variance ratios, and the PDF-variable indices.
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field   # noqa: E402

# Hydrometeor tolerances (constants_clubb.F90:290-321). rr/Nr for KK; ice/snow/graupel for Morrison.
# N*_tol = (1/((4/3)π·ρ·mvr_max³))·r*_tol  (ρ_lw=1000 for rain, ρ_ice=917 for ice/snow/graupel).
_RR_TOL = _RI_TOL = _RS_TOL = _RG_TOL = 1.0e-10
_NR_TOL = _RR_TOL / ((4.0 / 3.0) * np.pi * rho_lw  * mvr_rain_max ** 3)    # ~1.9099e-7
_NI_TOL = _RI_TOL / ((4.0 / 3.0) * np.pi * rho_ice * mvr_ice_max ** 3)
_NS_TOL = _RS_TOL / ((4.0 / 3.0) * np.pi * rho_ice * mvr_snow_max ** 3)
_NG_TOL = _RG_TOL / ((4.0 / 3.0) * np.pi * rho_ice * mvr_graupel_max ** 3)

# Default in-precip variance-ratio linear law: ratio = intrcpt + slope * max(host_dx, host_dy).
# (corr_varnce_module.F90:51/65). rr and Nr share the same defaults.
_HMP2_IP_SLOPE_DEFAULT = 2.12e-5      # [1/m]
_HMP2_IP_INTRCPT_DEFAULT = 0.54       # [-]


@dataclass
class HmMetadata:
    """Pure-Python mirror of hm_metadata_type — the hydrometeor PDF metadata.

    Hydrometeor-array indices (iirr, iiNr, ...) are 0-based here (-1 = absent), as are the PDF-
    variable indices (iiPDF_chi=0, ... iiPDF_rr, iiPDF_Nr). The per-hydrometeor lists are indexed
    by the hydrometeor-array index."""
    hydromet_dim: int
    iirr: int = -1
    iiNr: int = -1
    iiri: int = -1
    iiNi: int = -1
    iirs: int = -1
    iiNs: int = -1
    iirg: int = -1
    iiNg: int = -1
    iiPDF_chi: int = II_CHI
    iiPDF_eta: int = II_ETA
    iiPDF_w: int = II_W
    iiPDF_Ncn: int = II_NCN
    iiPDF_rr: int = -1
    iiPDF_Nr: int = -1
    pdf_dim: int = 4
    Ncnp2_on_Ncnm2: float = 1.0
    hydromet_list: list = field(default_factory=list)
    hydromet_tol: np.ndarray = None
    l_mix_rat_hm: np.ndarray = None
    l_frozen_hm: np.ndarray = None
    hmp2_ip_on_hmm2_ip: np.ndarray = None


# Default correlation-table column for each PDF variable, in Fortran table order
# (chi, eta, w, Ncn, rr, Nr, ri, Ni, rs, Ns, rg, Ng); 0-based here (the Fortran is 1-based).
_DEF_CORR_TABLE = (
    ('iiPDF_chi', II_CHI), ('iiPDF_eta', II_ETA), ('iiPDF_w', II_W), ('iiPDF_Ncn', II_NCN),
    ('iiPDF_rr', II_RR), ('iiPDF_Nr', II_NR), ('iiPDF_ri', 6), ('iiPDF_Ni', 7),
    ('iiPDF_rs', 8), ('iiPDF_Ns', 9), ('iiPDF_rg', 10), ('iiPDF_Ng', 11),
)


def def_corr_idx(iiPDF_x, hm_metadata):
    """Map a PDF-variable index to its column in the default correlation arrays
    (corr_varnce_module.F90:def_corr_idx). Returns the 0-based default-table index, or -1 if iiPDF_x
    matches no PDF variable. The JAX `HmMetadata` carries only the warm-PDF indices (chi/eta/w/Ncn/rr/Nr);
    the frozen-species branches (ri/Ni/rs/Ns/rg/Ng) are kept for fidelity (absent → field defaults to -1)."""
    for field_name, def_idx in _DEF_CORR_TABLE:
        idx = getattr(hm_metadata, field_name, -1)
        if idx >= 0 and iiPDF_x == idx:
            return def_idx
    return -1


# PDF-variable name → HmMetadata field, in Fortran get_corr_var_index select-case order.
_CORR_VAR_NAMES = (
    ("chi", "iiPDF_chi"), ("eta", "iiPDF_eta"), ("w", "iiPDF_w"), ("Ncn", "iiPDF_Ncn"),
    ("rr", "iiPDF_rr"), ("Nr", "iiPDF_Nr"), ("ri", "iiPDF_ri"), ("Ni", "iiPDF_Ni"),
    ("rs", "iiPDF_rs"), ("Ns", "iiPDF_Ns"), ("rg", "iiPDF_rg"), ("Ng", "iiPDF_Ng"),
)


def get_corr_var_index(var_name, hm_metadata):
    """Return the iiPDF index of a PDF variable by its name (corr_varnce_module.F90:get_corr_var_index).

    Maps "chi"/"eta"/"w"/"Ncn"/"rr"/"Nr"/"ri"/"Ni"/"rs"/"Ns"/"rg"/"Ng" to the corresponding `HmMetadata`
    iiPDF index; returns -1 for an unknown name or an absent (frozen-species) field — matching the Fortran,
    which initializes `i = -1` and leaves it unset for unmatched names. Used by the correlation-array input
    reader to look up the two variates of each tabulated correlation by name."""
    for name, field_name in _CORR_VAR_NAMES:
        if var_name == name:
            return getattr(hm_metadata, field_name, -1)
    return -1


def print_corr_matrix(corr_array_n, stream=None):
    """Print a pdf_dim×pdf_dim normal-space correlation matrix to the console
    (corr_varnce_module.F90:print_corr_matrix). Each entry is formatted F5.2 in a 6-char field,
    one row per PDF variable — a debug aid (the Fortran writes to stdout; this defaults to stderr,
    matching the print_pdf_params convention)."""
    import sys
    out = sys.stderr if stream is None else stream
    c = np.asarray(corr_array_n)
    pdf_dim = c.shape[0]
    for n in range(pdf_dim):
        print("".join(f"{c[m, n]:5.2f} " for m in range(pdf_dim)).rstrip(), file=out)


def init_pdf_hydromet_arrays(hydromet_dim, iirr=-1, iiNr=-1, iiri=-1, iiNi=-1,
                             iirs=-1, iiNs=-1, iirg=-1, iiNg=-1,
                             host_dx=1.0e6, host_dy=1.0e6, Ncnp2_on_Ncnm2=1.0,
                             slope_overrides=None, intrcpt_overrides=None, l_hydromet_pdf=True):
    """Build the hydrometeor PDF metadata. Oracle init_pdf_hydromet_arrays_api
    (corr_varnce_module.F90:455). Indices are 0-based (-1 = absent).

    Per hydrometeor: name (hydromet_list), tolerance, l_mix_rat_hm (mixing ratio vs number),
    l_frozen_hm, and the in-precip variance ratio hmp2_ip_on_hmm2_ip = intrcpt + slope*max(dx,dy).
    `slope_overrides`/`intrcpt_overrides` are optional dicts keyed by 'rr'/'Nr'/... (rico passes
    slope=0, intrcpt=1.25 for rr and Nr -> ratio 1.25). Assigns the PDF-variable indices
    (chi,eta,w,Ncn then the hydrometeors in array order) and pdf_dim."""
    # Per-hydrometeor static properties: (name, tol, l_mix_rat, l_frozen). Ice/snow/graupel are
    # frozen; masses are mixing ratios, N* are number concentrations.
    _props = {'rr': ("rrm", _RR_TOL, True, False), 'Nr': ("Nrm", _NR_TOL, False, False),
              'ri': ("rim", _RI_TOL, True, True), 'Ni': ("Nim", _NI_TOL, False, True),
              'rs': ("rsm", _RS_TOL, True, True), 'Ns': ("Nsm", _NS_TOL, False, True),
              'rg': ("rgm", _RG_TOL, True, True), 'Ng': ("Ngm", _NG_TOL, False, True)}
    _idx = {'rr': iirr, 'Nr': iiNr, 'ri': iiri, 'Ni': iiNi,
            'rs': iirs, 'Ns': iiNs, 'rg': iirg, 'Ng': iiNg}

    hydromet_list = [""] * hydromet_dim
    hydromet_tol = np.zeros(hydromet_dim)
    l_mix_rat_hm = np.zeros(hydromet_dim, dtype=bool)
    l_frozen_hm = np.zeros(hydromet_dim, dtype=bool)
    hmp2_ip = np.zeros(hydromet_dim)
    md = max(host_dx, host_dy)
    slope_o = slope_overrides or {}
    intrcpt_o = intrcpt_overrides or {}

    for name, idx in _idx.items():
        if idx < 0:
            continue
        if name not in _props:
            raise NotImplementedError(f"hydrometeor '{name}' not yet supported")
        nm, tol, l_mix, l_froz = _props[name]
        hydromet_list[idx] = nm
        hydromet_tol[idx] = tol
        l_mix_rat_hm[idx] = l_mix
        l_frozen_hm[idx] = l_froz
        slope = slope_o.get(name, _HMP2_IP_SLOPE_DEFAULT)
        intrcpt = intrcpt_o.get(name, _HMP2_IP_INTRCPT_DEFAULT)
        hmp2_ip[idx] = intrcpt + slope * md

    md_meta = HmMetadata(
        hydromet_dim=hydromet_dim, iirr=iirr, iiNr=iiNr, iiri=iiri, iiNi=iiNi,
        iirs=iirs, iiNs=iiNs, iirg=iirg, iiNg=iiNg, Ncnp2_on_Ncnm2=Ncnp2_on_Ncnm2,
        hydromet_list=hydromet_list, hydromet_tol=hydromet_tol,
        l_mix_rat_hm=l_mix_rat_hm, l_frozen_hm=l_frozen_hm, hmp2_ip_on_hmm2_ip=hmp2_ip)

    # PDF-variable index assignment: chi, eta, w, Ncn already fixed; the precipitating hydrometeors
    # follow in array order (corr_varnce_module.F90:687) — ONLY for schemes that integrate the
    # hydrometeor PDF (KK upscaled). Morrison is a BULK scheme (grid-mean rates), so it does NOT add
    # hydrometeors to the PDF → pdf_dim stays 4 (chi/eta/w/Ncn) and iiPDF_rr/Nr stay absent.
    if l_hydromet_pdf:
        pdf_count = md_meta.iiPDF_Ncn
        for i in range(hydromet_dim):
            if i == iirr:
                pdf_count += 1; md_meta.iiPDF_rr = pdf_count
            if i == iiNr:
                pdf_count += 1; md_meta.iiPDF_Nr = pdf_count
        md_meta.pdf_dim = pdf_count + 1   # 0-based: pdf_dim = highest index + 1
    return md_meta


def morrison_hm_metadata(host_dx=1.0e6, host_dy=1.0e6):
    """Hydrometeor metadata for the Morrison 2-moment scheme: 8 fields rain/ice/snow/graupel ×
    (mass, number) at array indices rr=0, Nr=1, ri=2, Ni=3, rs=4, Ns=5, rg=6, Ng=7. Bulk scheme →
    no hydrometeor PDF (l_hydromet_pdf=False, pdf_dim=4)."""
    return init_pdf_hydromet_arrays(
        hydromet_dim=8, iirr=0, iiNr=1, iiri=2, iiNi=3, iirs=4, iiNs=5, iirg=6, iiNg=7,
        host_dx=host_dx, host_dy=host_dy, l_hydromet_pdf=False)


def kk_hm_metadata(host_dx=1.0e6, host_dy=1.0e6, Ncnp2_on_Ncnm2=0.0):
    """Hydrometeor metadata for the standard KK scheme (rr at array index 0, Nr at 1), with rico's
    in-precip variance ratio override (slope=0, intrcpt=1.25 -> ratio 1.25)."""
    return init_pdf_hydromet_arrays(
        hydromet_dim=2, iirr=0, iiNr=1, host_dx=host_dx, host_dy=host_dy,
        Ncnp2_on_Ncnm2=Ncnp2_on_Ncnm2,
        slope_overrides={'rr': 0.0, 'Nr': 0.0},
        intrcpt_overrides={'rr': 1.25, 'Nr': 1.25})
