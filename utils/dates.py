import math
import datetime
import re

import numpy as N

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
offsetdate = datetime.datetime(2009, 01, 01, 0, 0, 0, 0)

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
                    raise ValueError, "Invalid date '%s'" % dategrp
            else:
                dateval = [2009, 01, 01]

            if timegrp:
                # if there is a time part of the string
                p = timegrp.split(':')
                if len(p) != 3:
                    raise ValueError, "Invalid time '%s'" % timegrp
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
            except ValueError:
                pass

    if dt is not None:
        delta = dt - offsetdate
        return (delta.days*24*60*60 + delta.seconds +
                delta.microseconds*1e-6)
    else:
        return N.nan

def floatToDateTime(f):
    """Convert float to datetime."""
    days = int(f/24/60/60)
    frac, sec = math.modf(f - days*24*60*60)
    return datetime.timedelta(days,  sec,  frac*1e6) + offsetdate
    
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
        for i in xrange(tt[1]):
            # find interval between this month and next...
            m, y = dt.month + 1, dt.year
            if m == 13:
                m = 1
                y += 1          
            dt = dt.replace(year=y, month=m)
    elif tt[1] < 0:
        for i in xrange(abs(tt[1])):
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
        timein[i] == ((timein[i]-1) // tt[i])*tt[i] + 1
    else:
        timein[i] = (timein[i] // tt[i])*tt[i]
        
    #print "rounded",  timein
    return tuple(timein)
    
    
