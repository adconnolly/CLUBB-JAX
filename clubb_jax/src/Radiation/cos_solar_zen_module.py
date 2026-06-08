"""Cosine of the solar zenith angle — pure Python port of cos_solar_zen_module.F90.

Mirrors clubb_release/src/Radiation/cos_solar_zen_module.F90, whose single function
`cos_solar_zen` returns cos(solar zenith angle) from the date/time and lat/lon, using the
Liou solar-declination coefficients (Clayson & Curry). The Gregorian-date helpers
The Gregorian-date helpers (`gregorian2julian_day`/`compute_current_date`/`leap_year`) live in
CLUBB_core/calendar.py (mirroring `use calendar`). radiation.py calls `cos_solar_zen` through the SW radiation path.
"""

import math

from clubb_jax.src.CLUBB_core.constants_clubb import sec_per_hr, radians_per_deg

# Liou coefficients for cos_solar_zen (cos_solar_zen_module.F90)
_CSZ_C0 =  0.006918
_CSZ_C1 = -0.399912
_CSZ_C2 = -0.006758
_CSZ_C3 = -0.002697
_CSZ_D1 =  0.070257
_CSZ_D2 =  0.000907
_CSZ_D3 =  0.000148


# gregorian2julian_day / leap_year / compute_current_date now live in their Fortran-home module
# CLUBB_core/calendar.py (mirror-refactor iter 115), imported below (mirrors cos_solar_zen_module.F90 `use calendar`).
from clubb_jax.src.CLUBB_core.calendar import (
    gregorian2julian_day, leap_year, compute_current_date)


def cos_solar_zen(day: int, month: int, year: int,
                  current_time: float,
                  lat_in_degrees: float, lon_in_degrees: float) -> float:
    """Cosine of solar zenith angle. Port of cos_solar_zen_module.F90."""
    present_day, present_month, present_year, present_time = \
        compute_current_date(day, month, year, current_time)

    jul_day = gregorian2julian_day(present_day, present_month, present_year)
    days_in_year = 366 if leap_year(present_year) else 365

    hour = present_time / sec_per_hr
    t = 2.0 * math.pi * (jul_day - 1) / days_in_year

    delta = (_CSZ_C0
             + _CSZ_C1 * math.cos(t)   + _CSZ_D1 * math.sin(t)
             + _CSZ_C2 * math.cos(2*t) + _CSZ_D2 * math.sin(2*t)
             + _CSZ_C3 * math.cos(3*t) + _CSZ_D3 * math.sin(3*t))

    h = int(hour)
    if 0 <= h <= 11:
        zln = 180.0 - hour * 15.0
    elif 12 <= h <= 23:
        zln = 540.0 - hour * 15.0
    else:
        raise ValueError(f"Hour={hour} > 24 in cos_solar_zen")

    longang = abs(lon_in_degrees - zln) * radians_per_deg
    latang = lat_in_degrees * radians_per_deg

    return (math.sin(latang) * math.sin(delta)
            + math.cos(latang) * math.cos(delta) * math.cos(longang))


__all__ = ["cos_solar_zen"]
