# axisticks.py
# algorithm to work out what tick-marks to put on an axis

#    Copyright (C) 2003 Jeremy S. Sanders
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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

import numarray
import math

import utils

"""Algorithms for working with axis ticks.

These algorithms were designed by me (Jeremy Sanders), so there
may well be bugs. Please report them.

The idea is to try to achieve a set number of major and minor ticks
by looking though a list of allowable interval values (after taking
account of what power of 10 the coordinates are in).
"""

class AxisTicks:
    """Class to work out at what values axis major ticks should appear."""

    # the allowed values we allow ticks to increase by
    # first values are the major tick intervals, followed by a list
    # of allowed minors
    allowed_minorintervals_linear = { 1.:  (0.1, 0.2, 0.5),
                                      2.:  (0.2, 0.5, 1.),
                                      5.:  (0.5, 1., 2.5),
                                      2.5: (0.5,) }
    # just get the allowable majors
    allowed_intervals_linear = allowed_minorintervals_linear.keys()

    # the allowed values we can increase by in log space
    # by default we increase by 10^3
    # if the first value is chosen we can use the "special" log minor ticks
    allowed_intervals_log = (1., 3., 6., 9., 12., 15., 19.)

    # positions we're allowed to put minor intervals
    allowed_minorintervals_log = (1., 3., 6., 9., 12., 15., 19.)

    # how much we should allow axes to extend to zero or intervals
    max_extend_factor = 0.15

    def __init__( self, minval, maxval, noticks, nominorticks,
                  logaxis = False, prefermore = True,
                  extendbounds = True, extendzero = True ):
        """Initialise the class.

        minval and maxval are the range of the data to be plotted
        noticks number of major ticks to aim for
        logaxis: axis logarithmic?
        prefermore: prefer more ticks rather than fewer
        extendbounds: extend minval and maxval to nearest tick if okay
        extendzero: extend one end to zero if it is okay"""

        self.minval = minval
        self.maxval = maxval
        self.noticks = noticks
        self.nominorticks = nominorticks
        self.logaxis = logaxis
        self.prefermore = prefermore
        self.extendbounds = extendbounds
        self.extendzero = extendzero

    def _calcTickValues( self, minval, maxval, delta ):
        """Compute the tick values, given minval, maxval and delta."""

        startmult = int( math.ceil( minval / delta ) )
        stopmult = int( math.floor( maxval / delta ) )
        
        return numarray.arange(startmult, stopmult+1) * delta

    def _tickNums(self, minval, maxval, delta):
        """Calculate number of ticks between minval and maxval with delta."""

        startmult = int( math.ceil( minval / delta ) )
        stopmult = int( math.floor( maxval / delta ) )

        return (stopmult-startmult)+1

    def _calcNoTicks( self, interval, logdelta ):
        """Return the number of ticks with spacing interval*10^logdelta.

        Returns a tuple (noticks, minval, maxval).
        """

        # store these for modification (if we extend bounds)
        minval = self.minval
        maxval = self.maxval

        # calculate tick spacing and maximum extension factor
        delta = interval * (10**logdelta)
        maxextend = (maxval - minval) * AxisTicks.max_extend_factor

        # should we try to extend one of the bounds to zero?
        if self.extendzero:
            # extend to zero using heuristic
            if minval > 0. and minval <= maxextend:
                minval = 0.
            if maxval < 0. and math.fabs(maxval) <= maxextend:
                maxval = 0.

        # should we try to extend to nearest interval*10^logdelta?
        if self.extendbounds:
            # extend minval if possible
            if math.fabs( math.modf( minval / delta )[0] ) > 1e-8:
                d = minval - ( math.floor( minval / delta ) * delta )
                if d <= maxextend:
                    minval -= d

            # extend maxval if possible
            if math.fabs( math.modf( maxval / delta)[0] ) > 1e-8:
                d = ( (math.floor(maxval / delta)+1.) * delta) - maxval
                if d <= maxextend:
                    maxval += d

        # return (noticks, minbound, maxbound)
        return ( self._tickNums(minval, maxval, delta), minval, maxval )

    def _calcLinearMinorTickValues(self, minval, maxval, interval, logstep,
                                   allowedintervals):
        """Get the best values for minor ticks on a linear axis

        Algorithm tries to look for best match to nominorticks
        Pass routine major ticks from minval to maxval with steps of
        interval*(10**logstep)
        """

        # iterate over allowed minor intervals
        best = -1
        best_noticks = -1
        best_delta = 1000000
        mult = 10.**logstep

        # iterate over allowed minor intervals
        for minint in allowedintervals:

            noticks = self._tickNums(minval, maxval, minint*mult)
            d = abs( self.nominorticks - noticks )

            # if this is a better match to the number of ticks
            # we want, choose this
            if (d < best_delta ) or \
               (d == best_delta and
                (self.prefermore and noticks > best_noticks) or
                (not self.prefermore and noticks < best_noticks)):

                best = minint
                best_delta = d
                best_noticks = noticks

        # use best value to return tick values
        return self._calcTickValues(minval, maxval, best*mult)

    def _calcLogMinorTickValues( self, minval, maxval ):
        """Calculate minor tick values with a log scale."""

        # this is a scale going e.g. 1,2,3,...8,9,10,20,30...90,100,200...

        # round down to nearest power of 10 for each
        alpha = int( math.floor( numarray.log10(minval) ) )
        beta = int( math.floor( numarray.log10(maxval) ) )

        ticks = []
        # iterate over range in log space
        for i in range(alpha, beta+1):
            power = 10.**i
            # add ticks for values in correct range
            for j in range(2, 10):
                v = power*j
                # blah log conversions mean we have to use 'fuzzy logic'
                if ( math.fabs(v - minval)/v < 1e-6 or v > minval ) and \
                   ( math.fabs(v - maxval)/v < 1e-6 or v < maxval ) :
                    ticks.append(v)

        return numarray.array( ticks )
        
    def _axisScaler(self, allowed_intervals):
        """With minval and maxval find best tick positions."""

        # work out range and log range
        range = self.maxval - self.minval
        intlogrange = int( numarray.log10( range ) )

        # we step variable to move through log space to find best ticks
        logstep = intlogrange + 1

        # we iterate down in log spacing, until we have more than twice
        # the number of ticks requested.
        # Maybe a better algorithm is required
        selection = []

        largestno = 0
        while True:
            for interval in allowed_intervals:
                no, minval, maxval = self._calcNoTicks( interval, logstep )
                selection.append( (no, interval, logstep, minval, maxval ) )

                largestno = max(largestno, no)

            if largestno > self.noticks*2:
                break

            logstep -= 1

            # necessary as we don't want 10**x on axis if |x|<1
            # :-(
            if logstep < 0 and self.logaxis:
                break

        # we now try to find the best matching value
        minabsdelta = 1e99
        mindelta = 1e99
        bestsel = ()

        # find the best set of tick labels
        for s in selection:
            # difference between what we want and what we have
            delta = s[0] - self.noticks
            absdelta = abs(delta)

            # if it matches better choose this
            if absdelta < minabsdelta:
                minabsdelta = absdelta
                mindelta = delta
                bestsel = s

            # if we find two closest matching label sets, we
            # test whether we prefer too few to too many labels
            if absdelta == minabsdelta:
                if (self.prefermore and (delta > mindelta)) or \
                       (not self.prefermore and (delta < mindelta)):
                    minabsdelta = absdelta
                    mindelta = delta
                    bestsel = s

        # now we have the best, we work out the ticks and return
        interval  = bestsel[1]
        loginterval = bestsel[2]

        tickdelta = interval * 10.**loginterval
        minval = bestsel[3]
        maxval = bestsel[4]

        # calculate the positions of the ticks
        ticks = self._calcTickValues( minval, maxval, tickdelta )
        return (minval, maxval, ticks, interval, loginterval)

    def getTicks( self ):
        """Calculate and return the position of the major ticks.

        Returns a tuple (minval, maxval, ticks)"""

        if self.logaxis:
            # which intervals we'll accept for major ticks
            intervals = AxisTicks.allowed_intervals_log

            # transform range into log space
            self.minval = numarray.log10( self.minval )
            self.maxval = numarray.log10( self.maxval )

        else:
            # which linear intervals we'll allow
            intervals = AxisTicks.allowed_intervals_linear
            
        ticks = self._axisScaler( intervals )

        interval = ticks[3]
        loginterval = ticks[4]

        # work out the most appropriate minor tick intervals
        if not self.logaxis:
            # just plain minor ticks
            # try to achieve no of minors close to value requested
            
            minorticks = self._calcLinearMinorTickValues\
                         (ticks[0], ticks[1], interval, loginterval,
                          AxisTicks.allowed_minorintervals_linear[interval])
        else:
            if interval == 1.:
                # here we use 'conventional' minor log tick spacing
                # e.g. 0.9, 1, 2, .., 8, 9, 10, 20, 30 ...
                
                minorticks = self._calcLogMinorTickValues(10.**ticks[0],
                                                          10.**ticks[1])
            else:
                # if we increase by more than one power of 10 on the
                # axis, we can't do the above, so we do linear ticks
                # in log space
                # aim is to choose powers of 3 for majors and minors
                # to make it easy to read the axis. comments?
                
                minorticks = self._calcLinearMinorTickValues\
                             (ticks[0], ticks[1], interval, loginterval,
                              AxisTicks.allowed_minorintervals_log)
                minorticks = 10.**minorticks
                                                         
            # transform normal ticks back to real space
            ticks = ( 10.**ticks[0], 10.**ticks[1], 10.**ticks[2] )
            
        return ticks[:3] + (minorticks,)

