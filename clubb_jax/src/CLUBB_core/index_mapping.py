"""JAX/Python port of index_mapping.F90.

Functions to map back and forth between the PDF arrays and the hydrometeor arrays.

The "iiPDF" indices index all PDF variates, including all hydrometeor variates
(chi, eta, w, Ncn, rr, Nr, ri, Ni, rs, Ns, rg, Ng — dimension pdf_dim). The "ii"
indices index hydrometeor arrays (rr, Nr, ri, Ni, rs, Ns, rg, Ng — dimension
hydromet_dim); the "ii" variates are a subset of the "iiPDF" variates.
Conversions between the two are done by pdf2hydromet_idx / hydromet2pdf_idx.

All indices are 0-based here (-1 = absent), as in the rest of the JAX port. The
JAX `HmMetadata` (corr_varnce_module.py) carries only the warm-PDF indices
(chi/eta/w/Ncn/rr/Nr); the frozen-species fields (ri/Ni/rs/Ns/rg/Ng) are absent
and resolve to -1 via getattr — matching the `def_corr_idx` convention. Matches
guard on `>= 0` so an absent (-1) metadata index never spuriously equals a -1 query.
"""
from clubb_jax.src.CLUBB_core.constants_clubb import (
    mvr_rain_max, mvr_ice_max, mvr_snow_max, mvr_graupel_max,
)

__all__ = ["pdf2hydromet_idx", "hydromet2pdf_idx",
           "rx2Nx_hm_idx", "Nx2rx_hm_idx", "mvr_hm_max"]


def _hm(hm_metadata, name):
    """Hydrometeor/PDF index `name` from the metadata, or -1 if the field is absent."""
    return getattr(hm_metadata, name, -1)


def pdf2hydromet_idx(pdf_idx, hm_metadata):
    """Return the hydromet-array index of the hydrometeor at PDF index `pdf_idx`
    (index_mapping.F90:pdf2hydromet_idx). Returns -1 (absent) if `pdf_idx` is not a
    hydrometeor PDF variate."""
    pairs = (("iiPDF_rr", "iirr"), ("iiPDF_Nr", "iiNr"),
             ("iiPDF_rs", "iirs"), ("iiPDF_Ns", "iiNs"),
             ("iiPDF_rg", "iirg"), ("iiPDF_Ng", "iiNg"),
             ("iiPDF_ri", "iiri"), ("iiPDF_Ni", "iiNi"))
    for pdf_name, hm_name in pairs:
        idx = _hm(hm_metadata, pdf_name)
        if idx >= 0 and pdf_idx == idx:
            return _hm(hm_metadata, hm_name)
    return -1


def hydromet2pdf_idx(hydromet_idx, hm_metadata):
    """Return the PDF-array index of the hydrometeor at hydromet-array index
    `hydromet_idx` (index_mapping.F90:hydromet2pdf_idx). Returns -1 if absent."""
    pairs = (("iirr", "iiPDF_rr"), ("iiNr", "iiPDF_Nr"),
             ("iiri", "iiPDF_ri"), ("iiNi", "iiPDF_Ni"),
             ("iirs", "iiPDF_rs"), ("iiNs", "iiPDF_Ns"),
             ("iirg", "iiPDF_rg"), ("iiNg", "iiPDF_Ng"))
    for hm_name, pdf_name in pairs:
        idx = _hm(hm_metadata, hm_name)
        if idx >= 0 and hydromet_idx == idx:
            return _hm(hm_metadata, pdf_name)
    return -1


def rx2Nx_hm_idx(rx_idx, hm_metadata):
    """Return the hydromet-array index of the concentration (Nx) for the species
    whose mixing-ratio index is `rx_idx` (index_mapping.F90:rx2Nx_hm_idx).
    Returns -1 if absent."""
    pairs = (("iirr", "iiNr"), ("iiri", "iiNi"),
             ("iirs", "iiNs"), ("iirg", "iiNg"))
    for r_name, n_name in pairs:
        idx = _hm(hm_metadata, r_name)
        if idx >= 0 and rx_idx == idx:
            return _hm(hm_metadata, n_name)
    return -1


def Nx2rx_hm_idx(Nx_idx, hm_metadata):
    """Return the hydromet-array index of the mixing ratio (rx) for the species
    whose concentration index is `Nx_idx` (index_mapping.F90:Nx2rx_hm_idx).
    Returns -1 if absent."""
    pairs = (("iiNr", "iirr"), ("iiNi", "iiri"),
             ("iiNs", "iirs"), ("iiNg", "iirg"))
    for n_name, r_name in pairs:
        idx = _hm(hm_metadata, n_name)
        if idx >= 0 and Nx_idx == idx:
            return _hm(hm_metadata, r_name)
    return -1


def mvr_hm_max(hydromet_idx, hm_metadata):
    """Return the maximum allowable mean volume radius [m] for the species at
    hydromet-array index `hydromet_idx` — whether mixing ratio or concentration
    (index_mapping.F90:mvr_hm_max). Returns 0.0 if absent."""
    species = ((("iirr", "iiNr"), mvr_rain_max),
               (("iiri", "iiNi"), mvr_ice_max),
               (("iirs", "iiNs"), mvr_snow_max),
               (("iirg", "iiNg"), mvr_graupel_max))
    for (r_name, n_name), mvr_max in species:
        r_idx = _hm(hm_metadata, r_name)
        n_idx = _hm(hm_metadata, n_name)
        if (r_idx >= 0 and hydromet_idx == r_idx) or (n_idx >= 0 and hydromet_idx == n_idx):
            return mvr_max
    return 0.0
