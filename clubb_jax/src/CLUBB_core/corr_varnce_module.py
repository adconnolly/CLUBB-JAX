"""JAX-side port of corr_varnce_module.F90.

This module owns the default normal-space correlation arrays and the
hydrometeor PDF metadata used by hydrometeor PDF setup.

JAX/Python adaptation: indices are 0-based, with -1 denoting an absent
hydrometeor/PDF variable. Fortran intent(out) arguments are returned.
"""

from dataclasses import dataclass, replace

import jax
import numpy as np

from clubb_jax.src.CLUBB_core.clubb_constants import (
    eps,
    Ng_tol,
    Ni_tol,
    Nr_tol,
    Ns_tol,
    rg_tol,
    ri_tol,
    rr_tol,
    rs_tol,
)


@dataclass(frozen=True)
class hmp2_ip_on_hmm2_ip_ratios_type:
    rr: float = 1.0
    Nr: float = 1.0
    ri: float = 1.0
    Ni: float = 1.0
    rs: float = 1.0
    Ns: float = 1.0
    rg: float = 1.0
    Ng: float = 1.0


@dataclass(frozen=True)
class hmp2_ip_on_hmm2_ip_slope_type:
    rr: float = 2.12e-5
    Nr: float = 2.12e-5
    ri: float = 2.12e-5
    Ni: float = 2.12e-5
    rs: float = 2.12e-5
    Ns: float = 2.12e-5
    rg: float = 2.12e-5
    Ng: float = 2.12e-5


@dataclass(frozen=True)
class hmp2_ip_on_hmm2_ip_intrcpt_type:
    rr: float = 0.54
    Nr: float = 0.54
    ri: float = 0.54
    Ni: float = 0.54
    rs: float = 0.54
    Ns: float = 0.54
    rg: float = 0.54
    Ng: float = 0.54


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class hm_metadata_type:
    iirr: int = -1
    iirs: int = -1
    iiri: int = -1
    iirg: int = -1
    iiNr: int = -1
    iiNs: int = -1
    iiNi: int = -1
    iiNg: int = -1
    l_frozen_hm: np.ndarray | None = None
    l_mix_rat_hm: np.ndarray | None = None
    hydromet_list: tuple[str, ...] = ()
    hydromet_tol: np.ndarray | None = None
    iiPDF_chi: int = -1
    iiPDF_eta: int = -1
    iiPDF_w: int = -1
    iiPDF_rr: int = -1
    iiPDF_rs: int = -1
    iiPDF_ri: int = -1
    iiPDF_rg: int = -1
    iiPDF_Nr: int = -1
    iiPDF_Ns: int = -1
    iiPDF_Ni: int = -1
    iiPDF_Ng: int = -1
    iiPDF_Ncn: int = -1
    hmp2_ip_on_hmm2_ip: np.ndarray | None = None
    Ncnp2_on_Ncnm2: float = 1.0

    def tree_flatten(self):
        children = (
            self.l_frozen_hm,
            self.l_mix_rat_hm,
            self.hydromet_tol,
            self.hmp2_ip_on_hmm2_ip,
        )
        aux_data = (
            self.iirr,
            self.iirs,
            self.iiri,
            self.iirg,
            self.iiNr,
            self.iiNs,
            self.iiNi,
            self.iiNg,
            self.hydromet_list,
            self.iiPDF_chi,
            self.iiPDF_eta,
            self.iiPDF_w,
            self.iiPDF_rr,
            self.iiPDF_rs,
            self.iiPDF_ri,
            self.iiPDF_rg,
            self.iiPDF_Nr,
            self.iiPDF_Ns,
            self.iiPDF_Ni,
            self.iiPDF_Ng,
            self.iiPDF_Ncn,
            self.Ncnp2_on_Ncnm2,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            iirr,
            iirs,
            iiri,
            iirg,
            iiNr,
            iiNs,
            iiNi,
            iiNg,
            hydromet_list,
            iiPDF_chi,
            iiPDF_eta,
            iiPDF_w,
            iiPDF_rr,
            iiPDF_rs,
            iiPDF_ri,
            iiPDF_rg,
            iiPDF_Nr,
            iiPDF_Ns,
            iiPDF_Ni,
            iiPDF_Ng,
            iiPDF_Ncn,
            Ncnp2_on_Ncnm2,
        ) = aux_data
        (
            l_frozen_hm,
            l_mix_rat_hm,
            hydromet_tol,
            hmp2_ip_on_hmm2_ip,
        ) = children
        return cls(
            iirr=iirr,
            iirs=iirs,
            iiri=iiri,
            iirg=iirg,
            iiNr=iiNr,
            iiNs=iiNs,
            iiNi=iiNi,
            iiNg=iiNg,
            l_frozen_hm=l_frozen_hm,
            l_mix_rat_hm=l_mix_rat_hm,
            hydromet_list=hydromet_list,
            hydromet_tol=hydromet_tol,
            iiPDF_chi=iiPDF_chi,
            iiPDF_eta=iiPDF_eta,
            iiPDF_w=iiPDF_w,
            iiPDF_rr=iiPDF_rr,
            iiPDF_rs=iiPDF_rs,
            iiPDF_ri=iiPDF_ri,
            iiPDF_rg=iiPDF_rg,
            iiPDF_Nr=iiPDF_Nr,
            iiPDF_Ns=iiPDF_Ns,
            iiPDF_Ni=iiPDF_Ni,
            iiPDF_Ng=iiPDF_Ng,
            iiPDF_Ncn=iiPDF_Ncn,
            hmp2_ip_on_hmm2_ip=hmp2_ip_on_hmm2_ip,
            Ncnp2_on_Ncnm2=Ncnp2_on_Ncnm2,
        )


d_var_total = 12

corr_array_n_cloud_def = np.array(
    [
        1.0, -0.6, 0.09, 0.09, 0.788, 0.675, 0.240, 0.222, 0.240, 0.222, 0.240, 0.222,
        0.0, 1.0, 0.027, 0.027, 0.114, 0.115, -0.029, 0.093, 0.022, 0.013, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.34, 0.315, 0.270, 0.120, 0.167, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.464, 0.320, 0.168, 0.232, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.821, 0.0, 0.0, 0.173, 0.164, 0.319, 0.308,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.152, 0.143, 0.0, 0.0, 0.285, 0.273,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.758, 0.585, 0.571, 0.379, 0.363,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.571, 0.550, 0.363, 0.345,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.758, 0.485, 0.470,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.470, 0.450,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.758,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ],
    dtype=np.float64,
).reshape((d_var_total, d_var_total), order="F")

corr_array_n_below_def = np.array(
    [
        1.0, 0.3, 0.09, 0.09, 0.788, 0.675, 0.240, 0.222, 0.240, 0.222, 0.240, 0.222,
        0.0, 1.0, 0.027, 0.027, 0.114, 0.115, -0.029, 0.093, 0.022, 0.013, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.34, 0.315, 0.270, 0.120, 0.167, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.464, 0.320, 0.168, 0.232, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.821, 0.0, 0.0, 0.173, 0.164, 0.319, 0.308,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.152, 0.143, 0.0, 0.0, 0.285, 0.273,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.758, 0.585, 0.571, 0.379, 0.363,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.571, 0.550, 0.363, 0.345,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.758, 0.485, 0.470,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.470, 0.450,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.758,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ],
    dtype=np.float64,
).reshape((d_var_total, d_var_total), order="F")


def def_corr_idx(iiPDF_x, hm_metadata):
    """Map an iiPDF index to the corresponding default correlation-array index."""
    ii_chi = 0
    ii_eta = 1
    ii_w = 2
    ii_Ncn = 3
    ii_rr = 4
    ii_Nr = 5
    ii_ri = 6
    ii_Ni = 7
    ii_rs = 8
    ii_Ns = 9
    ii_rg = 10
    ii_Ng = 11

    ii_def_corr = -1

    if iiPDF_x == hm_metadata.iiPDF_chi:
        ii_def_corr = ii_chi
    elif iiPDF_x == hm_metadata.iiPDF_eta:
        ii_def_corr = ii_eta
    elif iiPDF_x == hm_metadata.iiPDF_w:
        ii_def_corr = ii_w
    elif iiPDF_x == hm_metadata.iiPDF_Ncn:
        ii_def_corr = ii_Ncn
    elif iiPDF_x == hm_metadata.iiPDF_rr:
        ii_def_corr = ii_rr
    elif iiPDF_x == hm_metadata.iiPDF_Nr:
        ii_def_corr = ii_Nr
    elif iiPDF_x == hm_metadata.iiPDF_ri:
        ii_def_corr = ii_ri
    elif iiPDF_x == hm_metadata.iiPDF_Ni:
        ii_def_corr = ii_Ni
    elif iiPDF_x == hm_metadata.iiPDF_rs:
        ii_def_corr = ii_rs
    elif iiPDF_x == hm_metadata.iiPDF_Ns:
        ii_def_corr = ii_Ns
    elif iiPDF_x == hm_metadata.iiPDF_rg:
        ii_def_corr = ii_rg
    elif iiPDF_x == hm_metadata.iiPDF_Ng:
        ii_def_corr = ii_Ng

    return ii_def_corr


def set_corr_arrays_to_default(pdf_dim, hm_metadata):
    """Return the default lower-triangular cloud and below-cloud correlation arrays."""
    corr_array_n_cloud = np.zeros((pdf_dim, pdf_dim), dtype=np.float64)
    corr_array_n_below = np.zeros((pdf_dim, pdf_dim), dtype=np.float64)

    for i in range(pdf_dim):
        corr_array_n_cloud[i, i] = 1.0
        corr_array_n_below[i, i] = 1.0

    for i in range(pdf_dim - 1):
        for j in range(i + 1, pdf_dim):
            idx_i = def_corr_idx(i, hm_metadata)
            idx_j = def_corr_idx(j, hm_metadata)

            if idx_i > idx_j:
                corr_array_n_cloud[j, i] = corr_array_n_cloud_def[idx_i, idx_j]
                corr_array_n_below[j, i] = corr_array_n_below_def[idx_i, idx_j]
            else:
                corr_array_n_cloud[j, i] = corr_array_n_cloud_def[idx_j, idx_i]
                corr_array_n_below[j, i] = corr_array_n_below_def[idx_j, idx_i]

    return corr_array_n_cloud, corr_array_n_below


def get_corr_var_index(var_name, hm_metadata):
    """Return a PDF variable index by name."""
    i = -1

    var_name = var_name.strip()
    if var_name == "chi":
        i = hm_metadata.iiPDF_chi
    elif var_name == "eta":
        i = hm_metadata.iiPDF_eta
    elif var_name == "w":
        i = hm_metadata.iiPDF_w
    elif var_name == "Ncn":
        i = hm_metadata.iiPDF_Ncn
    elif var_name == "rr":
        i = hm_metadata.iiPDF_rr
    elif var_name == "Nr":
        i = hm_metadata.iiPDF_Nr
    elif var_name == "ri":
        i = hm_metadata.iiPDF_ri
    elif var_name == "Ni":
        i = hm_metadata.iiPDF_Ni
    elif var_name == "rs":
        i = hm_metadata.iiPDF_rs
    elif var_name == "Ns":
        i = hm_metadata.iiPDF_Ns
    elif var_name == "rg":
        i = hm_metadata.iiPDF_rg
    elif var_name == "Ng":
        i = hm_metadata.iiPDF_Ng

    return i


def init_pdf_hydromet_arrays_api(
    host_dx,
    host_dy,
    hydromet_dim,
    iirr,
    iiNr,
    iiri,
    iiNi,
    iirs,
    iiNs,
    iirg,
    iiNg,
    Ncnp2_on_Ncnm2,
    hmp2_ip_on_hmm2_ip_slope_in=None,
    hmp2_ip_on_hmm2_ip_intrcpt_in=None,
):
    """Initialize hydrometeor metadata and return ``hm_metadata, pdf_dim``."""
    hm_metadata = hm_metadata_type(
        iirr=iirr,
        iirs=iirs,
        iiri=iiri,
        iirg=iirg,
        iiNr=iiNr,
        iiNs=iiNs,
        iiNi=iiNi,
        iiNg=iiNg,
        Ncnp2_on_Ncnm2=Ncnp2_on_Ncnm2,
    )

    hmp2_ip_on_hmm2_ip_slope = (
        hmp2_ip_on_hmm2_ip_slope_in
        if hmp2_ip_on_hmm2_ip_slope_in is not None
        else hmp2_ip_on_hmm2_ip_slope_type()
    )
    hmp2_ip_on_hmm2_ip_intrcpt = (
        hmp2_ip_on_hmm2_ip_intrcpt_in
        if hmp2_ip_on_hmm2_ip_intrcpt_in is not None
        else hmp2_ip_on_hmm2_ip_intrcpt_type()
    )

    max_host_delta = max(host_dx, host_dy)
    hmp2_ip_on_hmm2_ip = np.zeros(hydromet_dim, dtype=np.float64)

    if iirr >= 0:
        hmp2_ip_on_hmm2_ip[iirr] = (
            hmp2_ip_on_hmm2_ip_intrcpt.rr
            + hmp2_ip_on_hmm2_ip_slope.rr * max_host_delta
        )
    if iirs >= 0:
        hmp2_ip_on_hmm2_ip[iirs] = (
            hmp2_ip_on_hmm2_ip_intrcpt.rs
            + hmp2_ip_on_hmm2_ip_slope.rs * max_host_delta
        )
    if iiri >= 0:
        hmp2_ip_on_hmm2_ip[iiri] = (
            hmp2_ip_on_hmm2_ip_intrcpt.ri
            + hmp2_ip_on_hmm2_ip_slope.ri * max_host_delta
        )
    if iirg >= 0:
        hmp2_ip_on_hmm2_ip[iirg] = (
            hmp2_ip_on_hmm2_ip_intrcpt.rg
            + hmp2_ip_on_hmm2_ip_slope.rg * max_host_delta
        )
    if iiNr >= 0:
        hmp2_ip_on_hmm2_ip[iiNr] = (
            hmp2_ip_on_hmm2_ip_intrcpt.Nr
            + hmp2_ip_on_hmm2_ip_slope.Nr * max_host_delta
        )
    if iiNs >= 0:
        hmp2_ip_on_hmm2_ip[iiNs] = (
            hmp2_ip_on_hmm2_ip_intrcpt.Ns
            + hmp2_ip_on_hmm2_ip_slope.Ns * max_host_delta
        )
    if iiNi >= 0:
        hmp2_ip_on_hmm2_ip[iiNi] = (
            hmp2_ip_on_hmm2_ip_intrcpt.Ni
            + hmp2_ip_on_hmm2_ip_slope.Ni * max_host_delta
        )
    if iiNg >= 0:
        hmp2_ip_on_hmm2_ip[iiNg] = (
            hmp2_ip_on_hmm2_ip_intrcpt.Ng
            + hmp2_ip_on_hmm2_ip_slope.Ng * max_host_delta
        )

    hydromet_list = [""] * hydromet_dim
    hydromet_tol = np.zeros(hydromet_dim, dtype=np.float64)
    l_mix_rat_hm = np.zeros(hydromet_dim, dtype=bool)
    l_frozen_hm = np.zeros(hydromet_dim, dtype=bool)

    if iirr >= 0:
        hydromet_list[iirr] = "rrm"
        l_mix_rat_hm[iirr] = True
        l_frozen_hm[iirr] = False
        hydromet_tol[iirr] = rr_tol
    if iiri >= 0:
        hydromet_list[iiri] = "rim"
        l_mix_rat_hm[iiri] = True
        l_frozen_hm[iiri] = True
        hydromet_tol[iiri] = ri_tol
    if iirs >= 0:
        hydromet_list[iirs] = "rsm"
        l_mix_rat_hm[iirs] = True
        l_frozen_hm[iirs] = True
        hydromet_tol[iirs] = rs_tol
    if iirg >= 0:
        hydromet_list[iirg] = "rgm"
        l_mix_rat_hm[iirg] = True
        l_frozen_hm[iirg] = True
        hydromet_tol[iirg] = rg_tol
    if iiNr >= 0:
        hydromet_list[iiNr] = "Nrm"
        l_frozen_hm[iiNr] = False
        l_mix_rat_hm[iiNr] = False
        hydromet_tol[iiNr] = Nr_tol
    if iiNi >= 0:
        hydromet_list[iiNi] = "Nim"
        l_mix_rat_hm[iiNi] = False
        l_frozen_hm[iiNi] = True
        hydromet_tol[iiNi] = Ni_tol
    if iiNs >= 0:
        hydromet_list[iiNs] = "Nsm"
        l_mix_rat_hm[iiNs] = False
        l_frozen_hm[iiNs] = True
        hydromet_tol[iiNs] = Ns_tol
    if iiNg >= 0:
        hydromet_list[iiNg] = "Ngm"
        l_mix_rat_hm[iiNg] = False
        l_frozen_hm[iiNg] = True
        hydromet_tol[iiNg] = Ng_tol

    hm_metadata = replace(
        hm_metadata,
        hmp2_ip_on_hmm2_ip=hmp2_ip_on_hmm2_ip,
        hydromet_list=tuple(hydromet_list),
        hydromet_tol=hydromet_tol,
        l_mix_rat_hm=l_mix_rat_hm,
        l_frozen_hm=l_frozen_hm,
        iiPDF_chi=0,
        iiPDF_eta=1,
        iiPDF_w=2,
        iiPDF_Ncn=3,
    )

    pdf_count = hm_metadata.iiPDF_Ncn

    if hydromet_dim > 0:
        for i in range(hydromet_dim):
            if i == iirr:
                pdf_count += 1
                hm_metadata = replace(hm_metadata, iiPDF_rr=pdf_count)
            if i == iiNr:
                pdf_count += 1
                hm_metadata = replace(hm_metadata, iiPDF_Nr=pdf_count)
            if i == iiri:
                pdf_count += 1
                hm_metadata = replace(hm_metadata, iiPDF_ri=pdf_count)
            if i == iiNi:
                pdf_count += 1
                hm_metadata = replace(hm_metadata, iiPDF_Ni=pdf_count)
            if i == iirs:
                pdf_count += 1
                hm_metadata = replace(hm_metadata, iiPDF_rs=pdf_count)
            if i == iiNs:
                pdf_count += 1
                hm_metadata = replace(hm_metadata, iiPDF_Ns=pdf_count)
            if i == iirg:
                pdf_count += 1
                hm_metadata = replace(hm_metadata, iiPDF_rg=pdf_count)
            if i == iiNg:
                pdf_count += 1
                hm_metadata = replace(hm_metadata, iiPDF_Ng=pdf_count)

    pdf_dim = pdf_count + 1

    return hm_metadata, pdf_dim


def setup_corr_varnce_array_api(
    pdf_dim,
    hm_metadata,
    l_fix_w_chi_eta_correlations,
    corr_array_n_cloud_default=None,
    corr_array_n_below_default=None,
):
    """Set up full symmetric cloud and below-cloud normal-space correlations."""
    del l_fix_w_chi_eta_correlations

    if corr_array_n_cloud_default is not None and corr_array_n_below_default is not None:
        corr_array_n_cloud = np.array(corr_array_n_cloud_default, dtype=np.float64)
        corr_array_n_below = np.array(corr_array_n_below_default, dtype=np.float64)
    else:
        corr_array_n_cloud, corr_array_n_below = set_corr_arrays_to_default(
            pdf_dim,
            hm_metadata,
        )

    corr_array_n_cloud = (
        np.tril(corr_array_n_cloud)
        + np.tril(corr_array_n_cloud, -1).T
    )
    corr_array_n_below = (
        np.tril(corr_array_n_below)
        + np.tril(corr_array_n_below, -1).T
    )

    return corr_array_n_cloud, corr_array_n_below


def assert_corr_symmetric(corr_array_n):
    """Return True when a normal-space correlation matrix is symmetric with unit diagonal."""
    tol = 1.0e-6
    corr_array_n = np.asarray(corr_array_n, dtype=np.float64)

    symmetric = np.all(np.abs(corr_array_n - corr_array_n.T) <= tol)
    unit_diagonal = np.all(np.abs(np.diagonal(corr_array_n) - 1.0) <= eps)

    return bool(symmetric and unit_diagonal)


def print_corr_matrix(pdf_dim, corr_array_n, stream=None):
    """Print a correlation matrix in the same orientation as the Fortran debug routine."""
    output = []

    for n in range(pdf_dim):
        row = []
        for m in range(pdf_dim):
            row.append(f"{corr_array_n[m, n]:5.2f}")
        output.append(" ".join(row))

    text = "\n".join(output)
    if stream is None:
        print(text)
    else:
        print(text, file=stream)
