"""
See all the bounding boxes (NOTE COMMENTS OF ERRORS) in this gist
    https://gist.github.com/graydon/11198540
"""

from collections import namedtuple
import calendar
from datetime import date

from typing import Optional, Tuple


Region = namedtuple('Region', ['name', 'lonmin', 'lonmax', 'latmin', 'latmax'])


def get_kenya() -> Region:
    """This pipeline is focused on drought prediction in Kenya.
    This function allows Kenya's bounding box to be easily accessed
    by all exporters.
    """
    return Region(name='kenya', lonmin=33.501, lonmax=42.283,
                  latmin=-5.202, latmax=6.002)

def get_ethiopia() -> Region:
    return Region(name='ethiopia', lonmin=32.9975838, lonmax=47.9823797,
                  latmin=3.397448, latmax=14.8940537)


def get_east_africa() -> Region:
    return Region(name='east_africa', lonmin=21, lonmax=51.8,
                  latmin=-11, latmax=23)


def minus_months(cur_year: int, cur_month: int, diff_months: int,
                 to_endmonth_datetime: bool = True) -> Tuple[int, int, Optional[date]]:
    """Given a year-month pair (e.g. 2018, 1), and a number of months subtracted
    from that (e.g. 2), return the new year-month pair (e.g. 2017, 11).

    Optionally, a date object representing the end of that month can be returned too
    """

    new_month = cur_month - diff_months
    if new_month < 1:
        new_month += 12
        new_year = cur_year - 1
    else:
        new_year = cur_year

    if to_endmonth_datetime:
        newdate: Optional[date] = date(new_year, new_month,
                                       calendar.monthrange(new_year, new_month)[-1])
    else:
        newdate = None
    return new_year, new_month, newdate


# dictionary lookup of regions
region_lookup = {
    "kenya": get_kenya(),
    "ethiopia": get_ethiopia(),
    "east_africa": get_east_africa(),
}
