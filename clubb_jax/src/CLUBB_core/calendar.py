"""JAX (pure-Python) port of calendar.F90 — Gregorian/Julian date helpers.

Mirrors clubb_release/src/CLUBB_core/calendar.F90: the date arithmetic used by the radiation solar-zenith path —
`gregorian2julian_day` (day-of-year), `gregorian2julian_date`/`julian2gregorian_date` (the Fliegel & van Flandern
Julian-Day-Number round trip), `leap_year`, and `compute_current_date` (advance the start date by an elapsed number
of seconds, via the JDN round trip exactly as the Fortran does). cos_solar_zen_module.py imports these, mirroring
the Fortran `use calendar`.

Pure integer/float Python — deterministic, no array deps.
"""

from __future__ import annotations

import math

# Seconds per day (constants_clubb.F90:sec_per_day), used by compute_current_date.
_SEC_PER_DAY = 86400.0


def _itrunc_div(a: int, b: int) -> int:
    """Fortran integer division: truncate toward zero (Python `//` floors, which differs for mixed signs).

    The Fliegel & van Flandern algorithms below rely on truncation — e.g. `(month-14)/12` is negative for
    month<14, where Fortran gives 0/-1 by truncation but Python `//` would floor to -1/-2.
    """
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def leap_year(year: int) -> bool:
    """Port of calendar.F90:leap_year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def gregorian2julian_day(day: int, month: int, year: int) -> int:
    """Julian day number (day of year, 1-based). Port of calendar.F90:gregorian2julian_day."""
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leap_year(year):
        days_in_month[2] = 29
    return sum(days_in_month[:month]) + day


def gregorian2julian_date(day: int, month: int, year: int) -> int:
    """Julian Date (days since 1 January 4713 BC) from a Gregorian (day, month, year).
    Port of calendar.F90:gregorian2julian_date (Fliegel & van Flandern, CACM 11(10), 1968)."""
    i, j, k = year, month, day
    t = _itrunc_div(j - 14, 12)
    return (k - 32075
            + _itrunc_div(1461 * (i + 4800 + t), 4)
            + _itrunc_div(367 * (j - 2 - t * 12), 12)
            - _itrunc_div(3 * _itrunc_div(i + 4900 + t, 100), 4))


def julian2gregorian_date(julian_date: int):
    """Gregorian (day, month, year) from a Julian Date. Returns (day, month, year).
    Port of calendar.F90:julian2gregorian_date (Fliegel & van Flandern, CACM 11(10), 1968)."""
    l = julian_date + 68569
    n = _itrunc_div(4 * l, 146097)
    l = l - _itrunc_div(146097 * n + 3, 4)
    i = _itrunc_div(4000 * (l + 1), 1461001)
    l = l - _itrunc_div(1461 * i, 4) + 31
    j = _itrunc_div(80 * l, 2447)
    k = l - _itrunc_div(2447 * j, 80)
    l = _itrunc_div(j, 11)
    j = j + 2 - 12 * l
    i = 100 * (n - 49) + i + l
    return k, j, i   # day, month, year


def compute_current_date(day: int, month: int, year: int, current_time_s: float):
    """Advance start date by current_time_s seconds. Returns (day, month, year, time_in_day_s).
    Port of calendar.F90:compute_current_date_api — adds elapsed whole days to the starting Julian Date and
    converts back, exactly mirroring the Fortran's gregorian2julian_date / julian2gregorian_date round trip."""
    days_since_1jan4713bc = gregorian2julian_date(day, month, year)
    days_since_start = math.floor(current_time_s / _SEC_PER_DAY)
    days_since_1jan4713bc += days_since_start
    seconds_since_current_date = current_time_s - days_since_start * _SEC_PER_DAY
    current_day, current_month, current_year = julian2gregorian_date(days_since_1jan4713bc)
    return current_day, current_month, current_year, seconds_since_current_date
