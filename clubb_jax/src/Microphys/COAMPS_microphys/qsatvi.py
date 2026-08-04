"""JAX port of COAMPS_microphys/qsatvi.F — saturation mixing ratios over water & ice.

Mirrors `subroutine qsatvi(t, p, qvs, qvi, lice, len)`. Converts the saturation vapour
pressures (esatv/esati) into saturation mixing ratios: qvs = 0.62197*es/(p-es). When
lice is false, qvi is returned as zeros. p is pressure [Pa].
"""

from __future__ import annotations

from clubb_jax.src.CLUBB_core.tracer_numpy import _xp
from clubb_jax.src.Microphys.COAMPS_microphys.esatv import esatv
from clubb_jax.src.Microphys.COAMPS_microphys.esati import esati

_EPS = 0.62197


def qsatvi(t, p, lice):
    """Return (qvs, qvi): saturation mixing ratios over water and ice [kg/kg]."""
    xp = _xp
    es_w = esatv(t)
    qvs = _EPS * es_w / (p - es_w)
    if lice:
        es_i = esati(t)
        qvi = _EPS * es_i / (p - es_i)
    else:
        qvi = xp.zeros_like(qvs)
    return qvs, qvi
