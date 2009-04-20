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
T?
# match time HH:MM:SS
([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}(\.[0-9]+)?)?
$
''', re.VERBOSE )

offsetdate = datetime.datetime(2009, 01, 01, 0, 0, 0, 0)

def isDateTime(datestr):
    """Check date/time string looks valid."""
    match = date_re.match(datestr)

    return match and (match.group(1) is not None or match.group(2) is not None)

def dateStringToDate(datestr):
    """Interpret a date string and return a Veusz-format date value."""
    val = N.nan
    match = date_re.match(datestr)
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

            # convert date to delta from offset
            dt = datetime.datetime( *(dateval+timeval) )
            delta = dt - offsetdate

            if not dategrp and not timegrp:
                # catch case when neither time nor date
                val = N.nan
            else:
                # convert offset into a delta
                val = (delta.days*24*60*60 + delta.seconds +
                       delta.microseconds*1e-6)

        except ValueError:
            # conversion failed, return nan
            pass

    return val
