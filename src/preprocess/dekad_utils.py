# Copyright (c) 2014, Vienna University of Technology (TU Wien), Department
# of Geodesy and Geoinformation (GEO).
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the Vienna University of Technology - Department of
#   Geodesy and Geoinformation nor the names of its contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author: Thomas Mistelbauer thomas.mistelbauer@geo.tuwien.ac.at
# Creation date: 2014-08-05

"""
This module provides functions for date manipulation on a dekadal basis.

A dekad is defined as days 1-10, 11-20 and 21-last day of a month.

Or in numbered dekads:

1: day 1-10
2: day 11-20
3: day 21-last
"""

import calendar
import pandas as pd
import math
from datetime import datetime


def dekad_index(begin, end=None):
    """Creates a pandas datetime index on a decadal basis.

    Parameters
    ----------
    begin : datetime
        Datetime index start date.
    end : datetime, optional
        Datetime index end date, set to current date if None.

    Returns
    -------
    dtindex : pandas.DatetimeIndex
        Dekadal datetime index.
    """

    if end is None:
        end = datetime.now()

    mon_begin = datetime(begin.year, begin.month, 1)
    mon_end = datetime(end.year, end.month, 1)

    daterange = pd.date_range(mon_begin, mon_end, freq="MS")

    dates = []

    for i, dat in enumerate(daterange):
        lday = calendar.monthrange(dat.year, dat.month)[1]
        if i == 0 and begin.day > 1:
            if begin.day < 11:
                if daterange.size == 1:
                    if end.day < 11:
                        dekads = [10]
                    elif end.day >= 11 and end.day < 21:
                        dekads = [10, 20]
                    else:
                        dekads = [10, 20, lday]
                else:
                    dekads = [10, 20, lday]
            elif begin.day >= 11 and begin.day < 21:
                if daterange.size == 1:
                    if end.day < 21:
                        dekads = [20]
                    else:
                        dekads = [20, lday]
                else:
                    dekads = [20, lday]
            else:
                dekads = [lday]
        elif i == (len(daterange) - 1) and end.day < 21:
            if end.day < 11:
                dekads = [10]
            else:
                dekads = [10, 20]
        else:
            dekads = [10, 20, lday]

        for j in dekads:
            dates.append(pd.datetime(dat.year, dat.month, j))

    dtindex = pd.DatetimeIndex(dates)

    return dtindex


def check_dekad(date):
    """Checks the dekad of a date and returns the dekad date.

    Parameters
    ----------
    date : datetime
        Date to check.

    Returns
    -------
    new_date : datetime
        Date of the dekad.
    """
    if date.day < 11:
        dekad = 10
    elif date.day > 10 and date.day < 21:
        dekad = 20
    else:
        dekad = calendar.monthrange(date.year, date.month)[1]

    new_date = datetime(date.year, date.month, dekad)

    return new_date


def dekad_startdate_from_date(dt_in):
    """
    dekadal startdate that a date falls in

    Parameters
    ----------
    run_dt: datetime.datetime

    Returns
    -------
    startdate: datetime.datetime
        startdate of dekad
    """
    if dt_in.day <= 10:
        startdate = datetime(dt_in.year, dt_in.month, 1, 0, 0, 0)
    if dt_in.day >= 11 and dt_in.day <= 20:
        startdate = datetime(dt_in.year, dt_in.month, 11, 0, 0, 0)
    if dt_in.day >= 21:
        startdate = datetime(dt_in.year, dt_in.month, 21, 0, 0, 0)
    return startdate


def check_dekad_enddate(dt):
    """
    Check if a date is a dekad enddate
    """
    return check_dekad(dt) == dt


def check_dekad_startdate(dt):
    """
    Check if a date is a dekad startdate
    """
    if dt.day in [1, 11, 21]:
        return True
    else:
        return False


def group_into_dekads(dates, use_dekad_startdate=False):
    """
    Group a list of dates into dekads.

    Parameters
    ----------
    dates: list of datetime.datetime
    use_dekad_startdates: boolean, optional
        If set the dekad reference dates will
        be the startdates of the dekad

    Returns
    -------
    groups: dict
        keys: dekad reference dates
        values: list of dates belonging to dekad
    """
    groups = {}
    for dt in dates:
        dekad_date = check_dekad(dt)
        if use_dekad_startdate:
            dekad_date = dekad_startdate_from_date(dekad_date)
        if dekad_date not in groups:
            groups[dekad_date] = []
        groups[dekad_date].append(dt)
    return groups


def dekad2day(year, month, dekad):
    """Gets the day of a dekad.

    Parameters
    ----------
    year : int
        Year of the date.
    month : int
        Month of the date.
    dekad : int
        Dekad of the date.

    Returns
    -------
    day : int
        Day value for the dekad.
    """

    if dekad == 1:
        day = 10
    elif dekad == 2:
        day = 20
    elif dekad == 3:
        day = calendar.monthrange(year, month)[1]

    return day


def runningdekad2date(year, rdekad):
    """Gets the date of the running dekad of a spacifc year.

    Parameters
    ----------
    year : int
        Year of the date.
    rdekad : int
        Running dekad of the date.

    Returns
    -------
    datetime.datetime
        Date value for the running dekad.
    """

    month = int(math.ceil(rdekad / 3.0))
    dekad = rdekad - month * 3 + 3
    day = dekad2day(year, month, dekad)

    return datetime(year, month, day)


def day2dekad(day):
    """Returns the dekad of a day.

    Parameters
    ----------
    day : int
        Day of the date.

    Returns
    -------
    dekad : int
        Number of the dekad in a month.
    """

    if day < 11:
        dekad = 1
    elif day > 10 and day < 21:
        dekad = 2
    else:
        dekad = 3

    return dekad


def get_dekad_period(dates):
    """Checks number of the dekad in the current year for dates given as list.

    Parameters
    ----------
    dates : list of datetime
        Dates to check.

    Returns
    -------
    period : list of int
        List of dekad periods.
    """

    period = []

    for dat in dates:

        d = check_dekad(dat)
        dekad = day2dekad(d.day)
        per = dekad + ((d.month - 1) * 3)
        period.append(per)

    return period
