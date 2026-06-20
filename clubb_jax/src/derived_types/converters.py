"""Explicit converters between JAX-side and API derived-type mirrors."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# The Fortran-backed `clubb_python` API derived types are only needed by the
# `*_to_api` direction and the `isinstance(x, Api*)` guards. The pure-JAX driver
# never produces those types, so make the import optional: when `clubb_python`
# (f2py) is unbuilt, bind the names to a sentinel class. `isinstance(jax_obj,
# _MissingApiType)` is correctly False, so every `*_from_api` normalizer still
# works; only the unused `*_to_api` constructors would raise if ever called.
try:
    from clubb_python.derived_types.err_info import ErrInfo as ApiErrInfo
    from clubb_python.derived_types.grid_class import Grid as ApiGrid
    from clubb_python.derived_types.nu_vert_res_dep import NuVertResDep as ApiNuVertResDep
    from clubb_python.derived_types.pdf_params import (
        implicit_coefs_terms as ApiImplicitCoefsTerms,
        pdf_parameter as ApiPdfParameter,
    )
    from clubb_python.derived_types.sclr_idx import SclrIdx as ApiSclrIdx
except ImportError:  # f2py / clubb_python not built — pure-JAX path only
    class _MissingApiType:
        """Sentinel for an unavailable clubb_python API type (no f2py build)."""
    ApiErrInfo = ApiGrid = ApiNuVertResDep = _MissingApiType
    ApiImplicitCoefsTerms = ApiPdfParameter = ApiSclrIdx = _MissingApiType

from clubb_jax.src.CLUBB_core.err_info import ErrInfo
from clubb_jax.src.CLUBB_core.grid_class import Grid
from clubb_jax.src.CLUBB_core.nu_vert_res_dep import NuVertResDep
from clubb_jax.src.CLUBB_core.pdf_params import implicit_coefs_terms, pdf_parameter
from clubb_jax.src.CLUBB_core.sclr_idx import SclrIdx


def _jnp_float(arr):
    return None if arr is None else jnp.asarray(arr, dtype=jnp.float64)


def _np_float_fortran(arr):
    return None if arr is None else np.asfortranarray(np.asarray(arr), dtype=np.float64)


def _np_int_fortran(arr):
    return None if arr is None else np.asfortranarray(np.asarray(arr), dtype=np.int32)


def grid_from_api(gr: Grid | ApiGrid) -> Grid:
    if isinstance(gr, Grid):
        return gr
    return Grid(
        nzm=int(gr.nzm),
        nzt=int(gr.nzt),
        ngrdcol=int(gr.ngrdcol),
        zm=jnp.asarray(gr.zm, dtype=jnp.float64),
        zt=jnp.asarray(gr.zt, dtype=jnp.float64),
        dzm=jnp.asarray(gr.dzm, dtype=jnp.float64),
        dzt=jnp.asarray(gr.dzt, dtype=jnp.float64),
        invrs_dzm=jnp.asarray(gr.invrs_dzm, dtype=jnp.float64),
        invrs_dzt=jnp.asarray(gr.invrs_dzt, dtype=jnp.float64),
        weights_zt2zm=jnp.asarray(gr.weights_zt2zm, dtype=jnp.float64),
        weights_zm2zt=jnp.asarray(gr.weights_zm2zt, dtype=jnp.float64),
        k_lb_zm=int(gr.k_lb_zm),
        k_ub_zm=int(gr.k_ub_zm),
        k_lb_zt=int(gr.k_lb_zt),
        k_ub_zt=int(gr.k_ub_zt),
        grid_dir_indx=int(gr.grid_dir_indx),
        grid_dir=float(gr.grid_dir),
    )


def grid_to_api(gr: Grid | ApiGrid) -> ApiGrid:
    if isinstance(gr, ApiGrid):
        return gr
    return ApiGrid(
        nzm=int(gr.nzm),
        nzt=int(gr.nzt),
        ngrdcol=int(gr.ngrdcol),
        zm=np.asfortranarray(np.asarray(gr.zm), dtype=np.float64),
        zt=np.asfortranarray(np.asarray(gr.zt), dtype=np.float64),
        dzm=np.asfortranarray(np.asarray(gr.dzm), dtype=np.float64),
        dzt=np.asfortranarray(np.asarray(gr.dzt), dtype=np.float64),
        invrs_dzm=np.asfortranarray(np.asarray(gr.invrs_dzm), dtype=np.float64),
        invrs_dzt=np.asfortranarray(np.asarray(gr.invrs_dzt), dtype=np.float64),
        weights_zt2zm=np.asfortranarray(np.asarray(gr.weights_zt2zm), dtype=np.float64),
        weights_zm2zt=np.asfortranarray(np.asarray(gr.weights_zm2zt), dtype=np.float64),
        k_lb_zm=int(gr.k_lb_zm),
        k_ub_zm=int(gr.k_ub_zm),
        k_lb_zt=int(gr.k_lb_zt),
        k_ub_zt=int(gr.k_ub_zt),
        grid_dir_indx=int(gr.grid_dir_indx),
        grid_dir=float(gr.grid_dir),
    )


def err_info_from_api(err_info: ErrInfo | ApiErrInfo) -> ErrInfo:
    if isinstance(err_info, ErrInfo):
        if err_info.err_code is None:
            return err_info.reset_code()
        if err_info.reason_code is None:
            return err_info._replace(
                reason_code=ErrInfo.initialized(int(err_info.ngrdcol)).reason_code,
            )
        return err_info
    return ErrInfo(
        ngrdcol=int(err_info.ngrdcol),
        chunk_idx=int(err_info.chunk_idx),
        mpi_rank=int(err_info.mpi_rank),
        lat=_jnp_float(err_info.lat),
        lon=_jnp_float(err_info.lon),
        err_code=(
            ErrInfo.initialized(int(err_info.ngrdcol)).err_code
            if err_info.err_code is None
            else jnp.asarray(err_info.err_code, dtype=jnp.int32)
        ),
        reason_code=ErrInfo.initialized(int(err_info.ngrdcol)).reason_code,
    )


def err_info_to_api(err_info: ErrInfo | ApiErrInfo) -> ApiErrInfo:
    if isinstance(err_info, ApiErrInfo):
        return err_info
    return ApiErrInfo(
        ngrdcol=int(err_info.ngrdcol),
        chunk_idx=int(err_info.chunk_idx),
        mpi_rank=int(err_info.mpi_rank),
        lat=_np_float_fortran(err_info.lat),
        lon=_np_float_fortran(err_info.lon),
        err_code=_np_int_fortran(err_info.err_code_or_default()),
    )


def nu_vert_res_dep_from_api(nu: NuVertResDep | ApiNuVertResDep) -> NuVertResDep:
    if isinstance(nu, NuVertResDep):
        return nu
    return NuVertResDep(
        nzm=int(nu.nzm),
        nu1=jnp.asarray(nu.nu1, dtype=jnp.float64),
        nu2=jnp.asarray(nu.nu2, dtype=jnp.float64),
        nu6=jnp.asarray(nu.nu6, dtype=jnp.float64),
        nu8=jnp.asarray(nu.nu8, dtype=jnp.float64),
        nu9=jnp.asarray(nu.nu9, dtype=jnp.float64),
        nu10=jnp.asarray(nu.nu10, dtype=jnp.float64),
        nu_hm=jnp.asarray(nu.nu_hm, dtype=jnp.float64),
    )


def nu_vert_res_dep_to_api(nu: NuVertResDep | ApiNuVertResDep) -> ApiNuVertResDep:
    if isinstance(nu, ApiNuVertResDep):
        return nu
    return ApiNuVertResDep(
        nzm=int(nu.nzm),
        nu1=np.asfortranarray(np.asarray(nu.nu1), dtype=np.float64),
        nu2=np.asfortranarray(np.asarray(nu.nu2), dtype=np.float64),
        nu6=np.asfortranarray(np.asarray(nu.nu6), dtype=np.float64),
        nu8=np.asfortranarray(np.asarray(nu.nu8), dtype=np.float64),
        nu9=np.asfortranarray(np.asarray(nu.nu9), dtype=np.float64),
        nu10=np.asfortranarray(np.asarray(nu.nu10), dtype=np.float64),
        nu_hm=np.asfortranarray(np.asarray(nu.nu_hm), dtype=np.float64),
    )


def implicit_coefs_terms_from_api(
    coefs: implicit_coefs_terms | ApiImplicitCoefsTerms,
) -> implicit_coefs_terms:
    if isinstance(coefs, implicit_coefs_terms):
        return coefs
    values = {
        name: _jnp_float(getattr(coefs, name))
        for name in implicit_coefs_terms._fields
        if name not in ("ngrdcol", "nz", "sclr_dim")
    }
    return implicit_coefs_terms(
        ngrdcol=int(coefs.ngrdcol),
        nz=int(coefs.nz),
        sclr_dim=int(coefs.sclr_dim),
        **values,
    )


def implicit_coefs_terms_to_api(
    coefs: implicit_coefs_terms | ApiImplicitCoefsTerms,
) -> ApiImplicitCoefsTerms:
    if isinstance(coefs, ApiImplicitCoefsTerms):
        return coefs
    values = {
        name: _np_float_fortran(getattr(coefs, name))
        for name in implicit_coefs_terms._fields
        if name not in ("ngrdcol", "nz", "sclr_dim")
    }
    return ApiImplicitCoefsTerms(
        ngrdcol=int(coefs.ngrdcol),
        nz=int(coefs.nz),
        sclr_dim=int(coefs.sclr_dim),
        **values,
    )


def sclr_idx_from_api(idx: SclrIdx | ApiSclrIdx) -> SclrIdx:
    if isinstance(idx, SclrIdx):
        return idx
    return SclrIdx(
        iisclr_rt=int(idx.iisclr_rt),
        iisclr_thl=int(idx.iisclr_thl),
        iisclr_CO2=int(idx.iisclr_CO2),
        iiedsclr_rt=int(idx.iiedsclr_rt),
        iiedsclr_thl=int(idx.iiedsclr_thl),
        iiedsclr_CO2=int(idx.iiedsclr_CO2),
    )


def sclr_idx_to_api(idx: SclrIdx | ApiSclrIdx) -> ApiSclrIdx:
    if isinstance(idx, ApiSclrIdx):
        return idx
    return ApiSclrIdx(
        iisclr_rt=int(idx.iisclr_rt),
        iisclr_thl=int(idx.iisclr_thl),
        iisclr_CO2=int(idx.iisclr_CO2),
        iiedsclr_rt=int(idx.iiedsclr_rt),
        iiedsclr_thl=int(idx.iiedsclr_thl),
        iiedsclr_CO2=int(idx.iiedsclr_CO2),
    )


def pdf_parameter_from_api(params: pdf_parameter | ApiPdfParameter) -> pdf_parameter:
    if isinstance(params, pdf_parameter):
        return params
    values = {
        name: _jnp_float(getattr(params, name))
        for name in pdf_parameter._fields
        if name not in ("ngrdcol", "nz")
    }
    return pdf_parameter(
        ngrdcol=int(params.ngrdcol),
        nz=int(params.nz),
        **values,
    )


def pdf_parameter_to_api(params: pdf_parameter | ApiPdfParameter) -> ApiPdfParameter:
    if isinstance(params, ApiPdfParameter):
        return params
    values = {
        name: _np_float_fortran(getattr(params, name))
        for name in pdf_parameter._fields
        if name not in ("ngrdcol", "nz")
    }
    return ApiPdfParameter(
        ngrdcol=int(params.ngrdcol),
        nz=int(params.nz),
        **values,
    )
