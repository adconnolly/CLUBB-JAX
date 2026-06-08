"""Shortwave radiative flux (Delta-Eddington) — pure Python port of rad_lwsw_module.F90.

Mirrors clubb_release/src/Radiation/rad_lwsw_module.F90, whose subroutine `sunray_sw` computes
the shortwave flux profile via the Delta-Eddington two-stream solution (Duynkerke). radiation.py
calls it from the simplified-radiation SW path (`use rad_lwsw_module`).
"""

import math

import numpy as np

from clubb_jax.src.CLUBB_core.constants_clubb import rho_lw, three_halves  # rad_lwsw_module.F90:397


def sunray_sw(ngrdcol: int, nzt: int,
              rcm: np.ndarray, rho: np.ndarray,
              xi_abs: float, dzt: np.ndarray,
              zm: np.ndarray, zt: np.ndarray,
              radius: float, A: float, gc: float,
              Fs0: float, omega: float, l_center: bool) -> np.ndarray:
    """Shortwave flux. Port of rad_lwsw_module.F90:sunray_sw (lines 343-755).

    Returns Frad_SW shape (ngrdcol, nzt+1) on momentum levels (bottom-up).
    """
    # Per-layer optical depth  tau(i,k) = three_halves * rcm * rho * dzt / radius / rho_lw
    tau = three_halves * rcm * rho * dzt / radius / rho_lw   # (ngrdcol, nzt)

    # Column total optical depth
    tauc = tau.sum(axis=1)   # (ngrdcol,)

    # Delta-Eddington transformation (Duynkerke eqn.18)
    ff = gc * gc
    gcde = gc / (1.0 + gc)
    omegade = (1.0 - ff) * omega / (1.0 - omega * ff)
    taude = (1.0 - omega * ff) * tau   # (ngrdcol, nzt)

    # Constants (scalar, same for all columns)
    x1 = 1.0 - omegade * gcde
    x2 = 1.0 - omegade
    rk = math.sqrt(3.0 * x2 * x1)
    xi_abs2 = xi_abs * xi_abs
    rk2 = rk * rk
    x3 = 4.0 * (1.0 - rk2 * xi_abs2)
    rp = math.sqrt(3.0 * x2 / x1)
    alpha = 3.0 * omegade * xi_abs2 * (1.0 + gcde * x2) / x3
    beta = 3.0 * omegade * xi_abs * (1.0 + 3.0 * gcde * xi_abs2 * x2) / x3

    rtt = 2.0 / 3.0
    xp23p = 1.0 + rtt * rp
    xm23p = 1.0 - rtt * rp
    ap23b = alpha + rtt * beta
    t1 = 1.0 - A - rtt * (1.0 + A) * rp
    t2 = 1.0 - A + rtt * (1.0 + A) * rp
    t3 = (1.0 - A) * alpha - rtt * (1.0 + A) * beta + A * xi_abs

    # Per-column: column total D-E optical depth, C1, C2
    taucde = (1.0 - omega * ff) * tauc   # (ngrdcol,)
    exmu0 = np.exp(-taucde / xi_abs)
    expk = np.exp(rk * taucde)
    exmk = 1.0 / expk

    c2 = (xp23p * t3 * exmu0 - t1 * ap23b * exmk) / (xp23p * t2 * expk - xm23p * t1 * exmk)
    c1 = (ap23b - c2 * xm23p) / xp23p   # both shape (ngrdcol,)

    # Flux computation on momentum levels: sequential taupath accumulation per column
    Frad_SW = np.zeros((ngrdcol, nzt + 1), dtype=np.float64)

    for i in range(ngrdcol):
        # Top momentum level (k = nzt+1, Python index nzt)
        taupath = 0.5 * taude[i, nzt - 1] if l_center else 0.0

        def _flux(tp):
            F_diff = (-4.0 / 3.0) * Fs0 * (
                rp * (c1[i] * math.exp(-rk * tp) - c2[i] * math.exp(rk * tp))
                - beta * math.exp(-tp / xi_abs)
            )
            F_dir = -Fs0 * xi_abs * math.exp(-tp / xi_abs)
            return F_diff + F_dir

        Frad_SW[i, nzt] = _flux(taupath)

        # Interior levels k = nzt-1 down to 1 (Python indices nzt-1 down to 1)
        for k_py in range(nzt - 1, 0, -1):
            if l_center:
                # lin_interpolate_two_points(zm[k], zt[k], zt[k-1], taude[k], taude[k-1])
                zm_k = zm[i, k_py]
                zt_k = zt[i, k_py]
                zt_km1 = zt[i, k_py - 1]
                td_k = taude[i, k_py]
                td_km1 = taude[i, k_py - 1]
                denom = zt_k - zt_km1
                if abs(denom) < 1e-300:
                    interp = 0.5 * (td_k + td_km1)
                else:
                    interp = td_km1 + (zm_k - zt_km1) * (td_k - td_km1) / denom
                taupath += interp
            else:
                taupath += taude[i, k_py - 1]
            Frad_SW[i, k_py] = _flux(taupath)

        # Bottom momentum level (k = 1, Python index 0)
        taupath += taude[i, 0]
        Frad_SW[i, 0] = _flux(taupath)

    return Frad_SW


__all__ = ["sunray_sw"]
