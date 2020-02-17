#    Copyright (C) 2009 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
###############################################################################

from __future__ import division
import math
import datetime
import re

import numpy as N

from ..compat import crange, citems, cstr

# date format: YYYY-MM-DDTHH:MM:SS.mmmmmm
# date and time part are optional (check we have at least one!)
date_re = re.compile( r'''
^
# match date YYYY-MM-DD
([0-9]{4}-[0-9]{1,2}-[0-9]{1,2})?
[ ,A-Za-z]?
# match time HH:MM:SS
([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}(\.[0-9]+)?)?
$
''', re.VERBOSE )

# we store dates as intervals in sec from this date as a float
offsetdate = datetime.datetime(2009, 1, 1, 0, 0, 0, 0)
# this is the numpy version of this
offsetdate_np = N.datetime64('2009-01-01T00:00')

def isDateTime(datestr):
    """Check date/time string looks valid."""
    match = date_re.match(datestr)

    return match and (match.group(1) is not None or match.group(2) is not None)

def _isoDataStringToDate(datestr):
    """Convert ISO format date time to a datetime object."""
    match = date_re.match(datestr)
    val = None
    if match:
        try:
            dategrp, timegrp = match.group(1), match.group(2)
            if dategrp:
                # if there is a date part of the string
                dateval = [int(x) for x in dategrp.split('-')]
                if len(dateval) != 3:
                    raise ValueError("Invalid date '%s'" % dategrp)
            else:
                dateval = [2009, 1, 1]

            if timegrp:
                # if there is a time part of the string
                p = timegrp.split(':')
                if len(p) != 3:
                    raise ValueError("Invalid time '%s'" % timegrp)
                secfrac, sec = math.modf(float(p[2]))
                timeval = [ int(p[0]), int(p[1]), int(sec), int(secfrac*1e6) ]
            else:
                timeval = [0, 0, 0, 0]

            # either worked so return a datetime object
            if dategrp or timegrp:
                val = datetime.datetime( *(dateval+timeval) )

        except ValueError:
            # conversion failed, return nothing
            pass

    return val

def dateStringToDate(datestr):
    """Interpret a date string and return a Veusz-format date value."""
    dt = _isoDataStringToDate(datestr)
    if dt is None:
        # try local conversions (time, date and variations)
        # if ISO formats don't match
        for fmt in ('%X %x', '%x %X', '%x', '%X'):
            try:
                dt = datetime.datetime.strptime(datestr, fmt)
                break
            except (ValueError, TypeError):
                pass

    if dt is not None:
        delta = dt - offsetdate
        return (delta.days*24*60*60 + delta.seconds +
                delta.microseconds*1e-6)
    else:
        return N.nan

def floatUnixToVeusz(f):
    """Convert unix float to veusz float."""
    delta = datetime.datetime(1970,1,1) - offsetdate
    return f + delta.total_seconds()

def floatToDateTime(f):
    """Convert float to datetime."""
    days = int(f/24/60/60)
    frac, sec = math.modf(f - days*24*60*60)
    try:
        return datetime.timedelta(days,  sec,  frac*1e6) + offsetdate
    except OverflowError:
        return datetime.datetime(8000, 1, 1)

def dateFloatToString(f):
    """Convert date float to string."""
    if N.isfinite(f):
        return floatToDateTime(f).isoformat()
    else:
        return cstr(f)

def datetimeToTuple(dt):
    """Return tuple (year,month,day,hour,minute,second,microsecond) from
    datetime object."""
    return (dt.year, dt.month, dt.day, dt.hour, dt.minute,
            dt.second, dt.microsecond)

def datetimeToFloat(dt):
    """Convert datetime to float"""
    delta = dt - offsetdate
    # convert offset into a delta
    val = (delta.days*24*60*60 + (delta.seconds +
           delta.microseconds*1e-6))
    return val

def tupleToFloatTime(t):
    """Convert a tuple interval to a float style datetime"""
    dt = datetime.datetime(*t)
    return datetimeToFloat(dt)

def tupleToDateTime(t):
    """Convert a tuple to a datetime"""
    return datetime.datetime(*t)

def addTimeTupleToDateTime(dt,  tt):
    """Add a time tuple in the form (yr,mn,dy,h,m,s,us) to a datetime.
    Returns datetime
    """

    # add on most of the time intervals
    dt = dt + datetime.timedelta(days=tt[2], hours=tt[3], 
                                 minutes=tt[4], seconds=tt[5], 
                                 microseconds=tt[6])

    # add on years
    dt = dt.replace(year=dt.year + tt[0])
    
    # add on months - this could be much simpler
    if tt[1] > 0:
        for i in crange(tt[1]):
            # find interval between this month and next...
            m, y = dt.month + 1, dt.year
            if m == 13:
                m = 1
                y += 1          
            dt = dt.replace(year=y, month=m)
    elif tt[1] < 0:
        for i in crange(abs(tt[1])):
            # find interval between this month and next...
            m, y = dt.month - 1, dt.year
            if m == 0:
                m = 12
                y -= 1          
            dt = dt.replace(year=y, month=m)
        
    return dt

def roundDownToTimeTuple(dt,  tt):
    """Take a datetime, and round down using the (yr,mn,dy,h,m,s,ms) tuple.
    Returns a tuple."""

    #print "round down",  dt,  tt
    timein = list(datetimeToTuple(dt))
    i = 6
    while i >= 0 and tt[i] == 0:
        if i == 1 or i == 2: # month, day
            timein[i] = 1
        else:
            timein[i] = 0
        i -= 1
    # round to nearest interval
    if (i == 1 or i == 2): # month, day
        timein[i] = ((timein[i]-1) // tt[i])*tt[i] + 1
    else:
        timein[i] = (timein[i] // tt[i])*tt[i]
        
    #print "rounded",  timein
    return tuple(timein)

def dateStrToRegularExpression(instr):
    """Convert date-time string to regular expression.

    Converts format yyyy-mm-dd|T|hh:mm:ss to re for date
    """

    # first rename each special string to a unique string (this is a
    # unicode character which is in the private use area) then rename
    # back again to the regular expression. This avoids the regular
    # expression being remapped.
    maps = (
            ('YYYY', u'\ue001', r'(?P<YYYY>[0-9]{4})'),
            ('YY',   u'\ue002', r'(?P<YY>[0-9]{2})'),
            ('MM',   u'\ue003', r'(?P<MM>[0-9]{2})'),
            ('M',    u'\ue004', r'(?P<MM>[0-9]{1,2})'),
            ('DD',   u'\ue005', r'(?P<DD>[0-9]{2})'),
            ('D',    u'\ue006', r'(?P<DD>[0-9]{1,2})'),
            ('hh',   u'\ue007', r'(?P<hh>[0-9]{2})'),
            ('h',    u'\ue008', r'(?P<hh>[0-9]{1,2})'),
            ('mm',   u'\ue009', r'(?P<mm>[0-9]{2})'),
            ('m',    u'\ue00a', r'(?P<mm>[0-9]{1,2})'),
            ('ss',   u'\ue00b', r'(?P<ss>[0-9]{2}(\.[0-9]*)?)'),
            ('s',    u'\ue00c', r'(?P<ss>[0-9]{1,2}(\.[0-9]*)?)'),
        )

    out = []
    for p in instr.split('|'):
        # escape special characters (non alpha-num)
        p = re.escape(p)

        # replace strings with characters
        for search, char, repl in maps:
            p = p.replace(search, char, 1)
        # replace characters with re strings
        for search, char, repl in maps:
            p = p.replace(char, repl, 1)

        # save as an optional group
        out.append( '(?:%s)?' % p )

    # return final expression
    return r'^\s*%s\s*$' % (''.join(out))

def dateREMatchToDate(match):
    """Take match object for above regular expression,
    and convert to float date value."""

    if match is None:
        raise ValueError("match object is None")

    # remove None matches
    grps = {}
    for k, v in citems(match.groupdict()):
        if v is not None:
            grps[k] = v

    # bomb out if nothing matches
    if len(grps) == 0:
        raise ValueError("no groups matched")

    # get values of offset
    oyear = offsetdate.year
    omon = offsetdate.month
    oday = offsetdate.day
    ohour = offsetdate.hour
    omin = offsetdate.minute
    osec = offsetdate.second
    omicrosec = offsetdate.microsecond

    # now convert each element from the re
    if 'YYYY' in grps:
        oyear = int(grps['YYYY'])
    if 'YY' in grps:
        y = int(grps['YY'])
        if y >= 70:
            oyear = int('19' + grps['YY'])
        else:
            oyear = int('20' + grps['YY'])
    if 'MM' in grps:
        omon = int(grps['MM'])
    if 'DD' in grps:
        oday = int(grps['DD'])
    if 'hh' in grps:
        ohour = int(grps['hh'])
    if 'mm' in grps:
        omin = int(grps['mm'])
    if 'ss' in grps:
        s = float(grps['ss'])
        osec = int(s)
        omicrosec = int(1e6*(s-osec))

    # convert to python datetime object
    d = datetime.datetime(
        oyear, omon, oday, ohour, omin, osec, omicrosec)

    # return to veusz float time
    return datetimeToFloat(d)
