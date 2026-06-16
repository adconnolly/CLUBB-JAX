"""JAX/Python port of index_mapping.F90.

Functions to map back and forth between the PDF arrays and the hydrometeor
arrays.

JAX/Python adaptation: indices are 0-based, with -1 denoting an absent
hydrometeor/PDF variable. Matches guard on ``>= 0`` so an absent metadata index
never spuriously equals an absent query.
"""

from clubb_jax.src.CLUBB_core.clubb_constants import (
    mvr_graupel_max,
    mvr_ice_max,
    mvr_rain_max,
    mvr_snow_max,
)


def _hm(hm_metadata, name):
    return getattr(hm_metadata, name, -1)


def pdf2hydromet_idx(pdf_idx, hm_metadata):
    """Return the hydromet-array index corresponding to ``pdf_idx``."""
    pairs = (
        ("iiPDF_rr", "iirr"),
        ("iiPDF_Nr", "iiNr"),
        ("iiPDF_rs", "iirs"),
        ("iiPDF_Ns", "iiNs"),
        ("iiPDF_rg", "iirg"),
        ("iiPDF_Ng", "iiNg"),
        ("iiPDF_ri", "iiri"),
        ("iiPDF_Ni", "iiNi"),
    )
    for pdf_name, hm_name in pairs:
        idx = _hm(hm_metadata, pdf_name)
        if idx >= 0 and pdf_idx == idx:
            return _hm(hm_metadata, hm_name)
    return -1


def hydromet2pdf_idx(hydromet_idx, hm_metadata):
    """Return the PDF-array index corresponding to ``hydromet_idx``."""
    pairs = (
        ("iirr", "iiPDF_rr"),
        ("iiNr", "iiPDF_Nr"),
        ("iiri", "iiPDF_ri"),
        ("iiNi", "iiPDF_Ni"),
        ("iirs", "iiPDF_rs"),
        ("iiNs", "iiPDF_Ns"),
        ("iirg", "iiPDF_rg"),
        ("iiNg", "iiPDF_Ng"),
    )
    for hm_name, pdf_name in pairs:
        idx = _hm(hm_metadata, hm_name)
        if idx >= 0 and hydromet_idx == idx:
            return _hm(hm_metadata, pdf_name)
    return -1


def rx2Nx_hm_idx(rx_idx, hm_metadata):
    """Return the concentration hydromet index paired with ``rx_idx``."""
    pairs = (
        ("iirr", "iiNr"),
        ("iiri", "iiNi"),
        ("iirs", "iiNs"),
        ("iirg", "iiNg"),
    )
    for r_name, n_name in pairs:
        idx = _hm(hm_metadata, r_name)
        if idx >= 0 and rx_idx == idx:
            return _hm(hm_metadata, n_name)
    return -1


def Nx2rx_hm_idx(Nx_idx, hm_metadata):
    """Return the mixing-ratio hydromet index paired with ``Nx_idx``."""
    pairs = (
        ("iiNr", "iirr"),
        ("iiNi", "iiri"),
        ("iiNs", "iirs"),
        ("iiNg", "iirg"),
    )
    for n_name, r_name in pairs:
        idx = _hm(hm_metadata, n_name)
        if idx >= 0 and Nx_idx == idx:
            return _hm(hm_metadata, r_name)
    return -1


def mvr_hm_max(hydromet_idx, hm_metadata):
    """Return the maximum allowable mean volume radius for ``hydromet_idx``."""
    species = (
        (("iirr", "iiNr"), mvr_rain_max),
        (("iiri", "iiNi"), mvr_ice_max),
        (("iirs", "iiNs"), mvr_snow_max),
        (("iirg", "iiNg"), mvr_graupel_max),
    )
    for (r_name, n_name), mvr_max in species:
        r_idx = _hm(hm_metadata, r_name)
        n_idx = _hm(hm_metadata, n_name)
        if (r_idx >= 0 and hydromet_idx == r_idx) or (
            n_idx >= 0 and hydromet_idx == n_idx
        ):
            return mvr_max
    return 0.0


__all__ = [
    "pdf2hydromet_idx",
    "hydromet2pdf_idx",
    "rx2Nx_hm_idx",
    "Nx2rx_hm_idx",
    "mvr_hm_max",
]
