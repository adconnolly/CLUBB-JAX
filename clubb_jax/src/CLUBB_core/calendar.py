"""Pure-Python port of `src/CLUBB_core/calendar.F90`.

Porting deviations:
- Fortran subroutines with output arguments return tuples in Python.
- The Fortran `month_names` constant is omitted because no JAX caller uses it.
- `_itrunc_div` emulates Fortran integer division, which truncates toward zero.
  Python `//` floors for negative intermediate values in the Fliegel and van
  Flandern formulas.
"""

from __future__ import annotations

import math

_SEC_PER_DAY = 86400.0


def _itrunc_div(a: int, b: int) -> int:
    """Fortran integer division: truncate toward zero."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def leap_year(year: int) -> bool:
    """Determines if the given year is a leap year.

    References:
      None
    """
    # ---- Begin Code ----
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def gregorian2julian_day(day: int, month: int, year: int) -> int:
    """Determine the Julian day (1-366) for a given Gregorian calendar date.

    Description:
      This subroutine determines the Julian day (1-366)
      for a given Gregorian calendar date(e.g. July 1, 2008).

    References:
      None
    """
    # Number of days per month (Jan..Dec) for a non leap year
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leap_year(year):
        days_in_month[2] = 29

    # ---- Begin Code ----

    # Add the days from the previous months
    return sum(days_in_month[:month]) + day


def gregorian2julian_date(day: int, month: int, year: int) -> int:
    """Compute the Julian Date from a Gregorian date.

    Description:
      Computes the Julian Date (gregorian2julian), or the number of days since
      1 January 4713 BC, given a Gregorian Calender date (day, month, year).

    Reference:
      Fliegel, H. F. and van Flandern, T. C.,
      Communications of the ACM, Vol. 11, No. 10 (October, 1968)
    """
    i, j, k = year, month, day
    t = _itrunc_div(j - 14, 12)
    return (k - 32075
            + _itrunc_div(1461 * (i + 4800 + t), 4)
            + _itrunc_div(367 * (j - 2 - t * 12), 12)
            - _itrunc_div(3 * _itrunc_div(i + 4900 + t, 100), 4))


def julian2gregorian_date(julian_date: int):
    """Compute the Gregorian Calendar date from a Julian date.

    Description:
      Computes the Gregorina Calendar date (day, month, year)
      given the Julian date (julian_date).

    Reference:
      Fliegel, H. F. and van Flandern, T. C.,
      Communications of the ACM, Vol. 11, No. 10 (October, 1968)
      http://portal.acm.org/citation.cfm?id=364097
    """
    # ---- Begin Code ----
    l = julian_date + 68569
    # Known magic number
    n = _itrunc_div(4 * l, 146097)
    # Known magic number
    l = l - _itrunc_div(146097 * n + 3, 4)
    # Known magic number
    i = _itrunc_div(4000 * (l + 1), 1461001)
    # Known magic number
    l = l - _itrunc_div(1461 * i, 4) + 31
    # Known magic number
    j = _itrunc_div(80 * l, 2447)
    # Known magic number
    k = l - _itrunc_div(2447 * j, 80)
    # Known magic number
    l = _itrunc_div(j, 11)
    # Known magic number
    j = j + 2 - 12 * l
    # Known magic number
    i = 100 * (n - 49) + i + l
    return k, j, i   # day, month, year


def compute_current_date(day: int, month: int, year: int, current_time_s: float):
    """Compute the current Gregorian date from a previous date and elapsed seconds.

    Description:
      Computes the current Gregorian date from a previous date and
      the seconds that have transpired since that date.

    References:
      None
    """
    # ---- Begin Code ----

    # Using Julian dates we are able to add the days that the model
    # has been running

    # Determine the Julian Date of the starting date,
    #    written in Gregorian (day, month, year) form
    days_since_1jan4713bc = gregorian2julian_date(day, month, year)

    # Determine the amount of days that have passed since start date
    days_since_start = math.floor(current_time_s / _SEC_PER_DAY)

    # Set days_since_1jan4713 to the present Julian date
    days_since_1jan4713bc += days_since_start

    # Set Present time to be seconds since the Julian date
    seconds_since_current_date = current_time_s - days_since_start * _SEC_PER_DAY
    current_day, current_month, current_year = julian2gregorian_date(days_since_1jan4713bc)
    return current_day, current_month, current_year, seconds_since_current_date
