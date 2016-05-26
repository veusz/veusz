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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

from __future__ import division
import math
import numpy as N

from ..compat import crange
from .. import utils

"""Algorithms for working with axis ticks.

These algorithms were designed by me (Jeremy Sanders), so there
may well be bugs. Please report them.

The idea is to try to achieve a set number of major and minor ticks
by looking though a list of allowable interval values (after taking
account of what power of 10 the coordinates are in).
"""

class AxisTicksBase(object):
    """Base class of axis ticks classes."""

    def __init__( self, minval, maxval, numticks, numminorticks,
                  logaxis = False, prefermore = True,
                  extendmin = False, extendmax = False,
                  forceinterval = None ):
        """Initialise the class.

        minval and maxval are the range of the data to be plotted
        numticks number of major ticks to aim for
        logaxis: axis logarithmic?
        prefermore: prefer more ticks rather than fewer
        extendbounds: extend minval and maxval to nearest tick if okay
        forceinterval: force interval to one given (if allowed). interval
         is tuple as returned in self.interval after calling getTicks()
        """

        # clip to sensible range
        self.minval = max(min(minval, 1e100), -1e100)
        self.maxval = max(min(maxval, 1e100), -1e100)

        # tick parameters
        self.numticks = numticks
        self.numminorticks = numminorticks
        self.logaxis = logaxis
        self.prefermore = prefermore
        self.extendmin = extendmin
        self.extendmax = extendmax
        self.forceinterval = forceinterval

    def getTicks( self ):
        """Calculate and return the position of the major ticks.

        Results are returned as attributes of this object in
        interval, minval, maxval, tickvals, minorticks, autoformat
        """

class AxisTicks(AxisTicksBase):
    """Class to work out at what values axis major ticks should appear."""

    # the allowed values we allow ticks to increase by
    # first values are the major tick intervals, followed by a list
    # of allowed minors
    allowed_minorintervals_linear = { 1.:  (0.1, 0.2, 0.5),
                                      2.:  (0.2, 0.5, 1.),
                                      5.:  (0.5, 1., 2.5),
                                      2.5: (0.5,) }
    # just get the allowable majors
    allowed_intervals_linear = sorted(allowed_minorintervals_linear)

    # the allowed values we can increase by in log space
    # by default we increase by 10^3
    # if the first value is chosen we can use the "special" log minor ticks
    allowed_intervals_log = (1., 3., 6., 9., 12., 15., 19.)

    # positions we're allowed to put minor intervals
    allowed_minorintervals_log = (1., 3., 6., 9., 12., 15., 19.)

    # how much we should allow axes to extend to a tick
    max_extend_factor = 0.15

    def _calcTickValues( self, minval, maxval, delta ):
        """Compute the tick values, given minval, maxval and delta."""

        startmult = int( math.ceil( minval / delta ) )
        stopmult = int( math.floor( maxval / delta ) )

        return N.arange(startmult, stopmult+1) * delta

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

        # should we try to extend to nearest interval*10^logdelta?
        if self.extendmin:
            # extend minval if possible
            if math.fabs( math.modf( minval / delta )[0] ) > 1e-8:
                d = minval - ( math.floor( minval / delta ) * delta )
                if d <= maxextend:
                    minval -= d

        if self.extendmax:
            # extend maxval if possible
            if math.fabs( math.modf( maxval / delta)[0] ) > 1e-8:
                d = ( (math.floor(maxval / delta)+1.) * delta) - maxval
                if d <= maxextend:
                    maxval += d

        numticks = self._tickNums(minval, maxval, delta)
        return (numticks, minval, maxval)

    def _calcLinearMinorTickValues(self, minval, maxval, interval, logstep,
                                   allowedintervals):
        """Get the best values for minor ticks on a linear axis

        Algorithm tries to look for best match to nominorticks
        Pass routine major ticks from minval to maxval with steps of
        interval*(10**logstep)
        """

        # iterate over allowed minor intervals
        best = -1
        best_numticks = -1
        best_delta = 1000000
        mult = 10.**logstep

        # iterate over allowed minor intervals
        for minint in allowedintervals:

            numticks = self._tickNums(minval, maxval, minint*mult)
            d = abs( self.numminorticks - numticks )

            # if this is a better match to the number of ticks
            # we want, choose this
            if ((d < best_delta ) or
                (d == best_delta and
                (self.prefermore and numticks > best_numticks) or
                (not self.prefermore and numticks < best_numticks)) ):

                best = minint
                best_delta = d
                best_numticks = numticks

        # use best value to return tick values
        return self._calcTickValues(minval, maxval, best*mult)

    def _calcLogMinorTickValues( self, minval, maxval ):
        """Calculate minor tick values with a log scale."""

        # this is a scale going e.g. 1,2,3,...8,9,10,20,30...90,100,200...

        # round down to nearest power of 10 for each
        alpha = int( math.floor( N.log10(minval) ) )
        beta = int( math.floor( N.log10(maxval) ) )

        ticks = []
        # iterate over range in log space
        for i in crange(alpha, beta+1):
            power = 10.**i
            # add ticks for values in correct range
            for j in crange(2, 10):
                v = power*j
                # blah log conversions mean we have to use 'fuzzy logic'
                if ( math.fabs(v - minval)/v < 1e-6 or v > minval ) and \
                   ( math.fabs(v - maxval)/v < 1e-6 or v < maxval ) :
                    ticks.append(v)

        return N.array( ticks )

    def _selectBestTickFromSelection(self, selection):
        """Choose best tick from selection given."""
        # we now try to find the best matching value
        minabsdelta = 1e99
        mindelta = 1e99
        bestsel = ()

        # find the best set of tick labels
        for s in selection:
            # difference between what we want and what we have
            delta = s[0] - self.numticks
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

        return bestsel

    def _getBestTickSelection(self, allowed_intervals):
        """Go through allowed tick intervals and find one best matching
        requested parameters."""

        # work out range and log range
        therange = self.maxval - self.minval
        intlogrange = int( N.log10( therange ) )

        # we step variable to move through log space to find best ticks
        logstep = intlogrange + 1

        # we iterate down in log spacing, until we have more than twice
        # the number of ticks requested.
        # Maybe a better algorithm is required
        selection = []

        # keep track of largest number of ticks calculated
        largestno = 0

        while True:
            for interval in allowed_intervals:
                no, minval, maxval = self._calcNoTicks( interval, logstep )
                selection.append( (no, interval, logstep, minval, maxval ) )

                largestno = max(largestno, no)

            if largestno > self.numticks*2:
                break

            logstep -= 1

            # necessary as we don't want 10**x on axis if |x|<1
            # :-(
            if logstep < 0 and self.logaxis:
                break

        return selection

    def _tickSelector(self, allowed_intervals):
        """With minval and maxval find best tick positions."""

        if self.forceinterval is None:
            # get selection of closely matching ticks
            selection = self._getBestTickSelection(allowed_intervals)

            # now we have the best, we work out the ticks and return
            bestsel = self._selectBestTickFromSelection(selection)
            dummy, interval, loginterval, minval, maxval = bestsel
        else:
            # forced specific interval requested
            interval, loginterval = self.forceinterval
            no, minval, maxval = self._calcNoTicks(interval, loginterval)

        # calculate the positions of the ticks from parameters
        tickdelta = interval * 10.**loginterval
        ticks = self._calcTickValues( minval, maxval, tickdelta )

        return (minval, maxval, ticks, interval, loginterval)

    def getTicks(self):
        """Calculate and return the position of the major ticks.
        """

        if self.logaxis:
            # which intervals we'll accept for major ticks
            intervals = AxisTicks.allowed_intervals_log

            # transform range into log space
            self.minval = N.log10( self.minval )
            self.maxval = N.log10( self.maxval )
        else:
            # which linear intervals we'll allow
            intervals = AxisTicks.allowed_intervals_linear

        # avoid breakage if range is zero
        if abs(self.minval - self.maxval) < 1e-99:
            self.maxval = self.minval + 1.

        minval, maxval, tickvals, interval, loginterval = self._tickSelector(
            intervals )

        # work out the most appropriate minor tick intervals
        if not self.logaxis:
            # just plain minor ticks
            # try to achieve no of minors close to value requested

            minorticks = self._calcLinearMinorTickValues(
                minval, maxval, interval, loginterval,
                AxisTicks.allowed_minorintervals_linear[interval]
                )

        else:
            # log axis
            if interval == 1.:
                # calculate minor ticks
                # here we use 'conventional' minor log tick spacing
                # e.g. 0.9, 1, 2, .., 8, 9, 10, 20, 30 ...
                minorticks = self._calcLogMinorTickValues(
                    10.**minval, 10.**maxval)

                # Here we test whether more log major tick values are needed...
                # often we might only have one tick value, and so we add 2, then 5
                # this is a bit of a hack: better ideas please!!

                if len(tickvals) < 2:
                    # get lower power of 10
                    low10 = int( math.floor(minval) )

                    # could use numpy here
                    for i in (2., 5., 20., 50.):
                        n = low10 + math.log10(i)
                        if n >= minval and n <= maxval:
                            tickvals = N.concatenate( (tickvals, N.array([n]) ))

            else:
                # if we increase by more than one power of 10 on the
                # axis, we can't do the above, so we do linear ticks
                # in log space
                # aim is to choose powers of 3 for majors and minors
                # to make it easy to read the axis. comments?

                minorticks = self._calcLinearMinorTickValues(
                    minval, maxval, interval, loginterval,
                    AxisTicks.allowed_minorintervals_log)
                minorticks = 10.**minorticks

            # transform normal ticks back to real space
            minval = 10.**minval
            maxval = 10.**maxval
            tickvals = 10.**tickvals

        self.interval = (interval, loginterval)
        self.minorticks = minorticks
        self.minval = minval
        self.maxval = maxval
        self.tickvals = tickvals
        self.autoformat = '%Vg'

class DateTicks(AxisTicksBase):
    """For formatting dates. We want something that chooses appropriate
    intervals
    So we want to choose most apropriate interval depending on number of
    ticks requested
    """

    # possible intervals for a time/date axis
    # tuples of ((y, m, d, h, m, s, msec), autoformat)
    intervals = (
                 ((200, 0, 0, 0, 0, 0, 0), '%VDY'),
                 ((100, 0, 0, 0, 0, 0, 0), '%VDY'),
                 ((50, 0, 0, 0, 0, 0, 0), '%VDY'),
                 ((20, 0, 0, 0, 0, 0, 0), '%VDY'),
                 ((10, 0, 0, 0, 0, 0, 0), '%VDY'),
                 ((5, 0, 0, 0, 0, 0, 0), '%VDY'),
                 ((2, 0, 0, 0, 0, 0, 0), '%VDY'),
                 ((1, 0, 0, 0, 0, 0, 0), '%VDY'),
                 ((0, 6, 0, 0, 0, 0, 0), '%VDY-%VDm'),
                 ((0, 4, 0, 0, 0, 0, 0), '%VDY-%VDm'),
                 ((0, 3, 0, 0, 0, 0, 0), '%VDY-%VDm'),
                 ((0, 2, 0, 0, 0, 0, 0), '%VDY-%VDm'),
                 ((0, 1, 0, 0, 0, 0, 0), '%VDY-%VDm'),
                 ((0, 0, 28, 0, 0, 0, 0), '%VDY-%VDm-%VDd'),
                 ((0, 0, 14, 0, 0, 0, 0), '%VDY-%VDm-%VDd'),
                 ((0, 0, 7, 0, 0, 0, 0), '%VDY-%VDm-%VDd'),
                 ((0, 0, 2, 0, 0, 0, 0), '%VDY-%VDm-%VDd'),
                 ((0, 0, 1, 0, 0, 0, 0), '%VDY-%VDm-%VDd'),
                 ((0, 0, 0, 12, 0, 0, 0), '%VDY-%VDm-%VDd\\\\%VDH:%VDM'),
                 ((0, 0, 0, 6, 0, 0, 0), '%VDY-%VDm-%VDd\\\\%VDH:%VDM'),
                 ((0, 0, 0, 4, 0, 0, 0), '%VDY-%VDm-%VDd\\\\%VDH:%VDM'),
                 ((0, 0, 0, 3, 0, 0, 0), '%VDY-%VDm-%VDd\\\\%VDH:%VDM'),
                 ((0, 0, 0, 2, 0, 0, 0), '%VDH:%VDM'),
                 ((0, 0, 0, 1, 0, 0, 0), '%VDH:%VDM'),
                 ((0, 0, 0, 0, 30, 0, 0), '%VDH:%VDM'),
                 ((0, 0, 0, 0, 15, 0, 0), '%VDH:%VDM'),
                 ((0, 0, 0, 0, 10, 0, 0), '%VDH:%VDM'),
                 ((0, 0, 0, 0, 5, 0, 0), '%VDH:%VDM'),
                 ((0, 0, 0, 0, 2, 0, 0), '%VDH:%VDM'),
                 ((0, 0, 0, 0, 1, 0, 0), '%VDH:%VDM'),
                 ((0, 0, 0, 0, 0, 30, 0), '%VDH:%VDM:%VDS'),
                 ((0, 0, 0, 0, 0, 15, 0), '%VDH:%VDM:%VDS'),
                 ((0, 0, 0, 0, 0, 10, 0), '%VDH:%VDM:%VDS'),
                 ((0, 0, 0, 0, 0, 5, 0), '%VDH:%VDM:%VDS'),
                 ((0, 0, 0, 0, 0, 2, 0), '%VDH:%VDM:%VDS'),
                 ((0, 0, 0, 0, 0, 1, 0), '%VDH:%VDM:%VDS'),
                 ((0, 0, 0, 0, 0, 0, 500000), '%VDH:%VDM:%VDVS'),
                 ((0, 0, 0, 0, 0, 0, 200000), '%VDVS'),
                 ((0, 0, 0, 0, 0, 0, 100000), '%VDVS'),
                 ((0, 0, 0, 0, 0, 0, 50000), '%VDVS'),
                 ((0, 0, 0, 0, 0, 0, 10000), '%VDVS'),
                 )

    intervals_sec = N.array([(ms*1e-6+s+mi*60+hr*60*60+dy*24*60*60+
                              mn*(365/12.)*24*60*60+
                              yr*365*24*60*60)
                             for (yr, mn, dy, hr, mi, s, ms), fmt in intervals])

    def bestTickFinder(self, minval, maxval, numticks, extendmin, extendmax,
                       intervals, intervals_sec):
        """Try to find best choice of numticks ticks between minval and maxval
        intervals is an array similar to self.intervals
        intervals_sec is an array similar to self.intervals_sec

        Returns a tuple (minval, maxval, estimatedsize, ticks, textformat)"""

        delta = maxval - minval

        # iterate over different intervals and find one closest to what we want
        estimated = delta / intervals_sec

        tick1 = max(estimated.searchsorted(numticks)-1, 0)
        tick2 = min(tick1+1, len(estimated)-1)

        del1 = abs(estimated[tick1] - numticks)
        del2 = abs(estimated[tick2] - numticks)

        if del1 < del2:
            best = tick1
        else:
            best = tick2
        besttt, format = intervals[best]

        mindate = utils.floatToDateTime(minval)
        maxdate = utils.floatToDateTime(maxval)

        # round min and max to nearest
        minround = utils.tupleToDateTime(utils.roundDownToTimeTuple(mindate, besttt))
        maxround = utils.tupleToDateTime(utils.roundDownToTimeTuple(maxdate, besttt))

        if minround == mindate:
            mintick = minround
        else:
            # rounded down, so move on to next tick
            mintick = utils.addTimeTupleToDateTime(minround, besttt)
        maxtick = maxround

        # extend bounds if requested
        deltamin = utils.datetimeToFloat(mindate)-utils.datetimeToFloat(mintick)
        if extendmin and (deltamin != 0. and deltamin < delta*0.15):
            mindate = utils.addTimeTupleToDateTime(minround,
                                                   [-x for x in besttt])
            mintick = mindate
        deltamax = utils.datetimeToFloat(maxdate)-utils.datetimeToFloat(maxtick)
        if extendmax and (deltamax != 0. and deltamax < delta*0.15):
            maxdate = utils.addTimeTupleToDateTime(maxtick, besttt)
            maxtick = maxdate

        # make ticks
        ticks = []
        dt = mintick
        while dt <= maxtick:
            ticks.append( utils.datetimeToFloat(dt))
            dt = utils.addTimeTupleToDateTime(dt, besttt)

        return ( utils.datetimeToFloat(mindate),
                 utils.datetimeToFloat(maxdate),
                 intervals_sec[best],
                 N.array(ticks), format )

    def filterIntervals(self, estint):
        """Filter intervals and intervals_sec to be
        multiples of estint seconds."""
        intervals = []
        intervals_sec = []
        for i, inter in enumerate(self.intervals_sec):
            ratio = estint / inter
            if abs(ratio-int(ratio)) < ratio*.01:
                intervals.append(self.intervals[i])
                intervals_sec.append(inter)
        return intervals, N.array(intervals_sec)

    def getTicks(self):
        """Calculate and return the position of the major ticks.
        """

        # find minor ticks
        mindate, maxdate, est, ticks, format = self.bestTickFinder(
            self.minval, self.maxval, self.numticks,
            self.extendmin, self.extendmax,
            self.intervals, self.intervals_sec)

        # try to make minor ticks divide evenly into major ticks
        intervals, intervals_sec = self.filterIntervals(est)
        # get minor ticks
        ig, ig, ig, minorticks, ig = self.bestTickFinder(
            mindate, maxdate, self.numminorticks, False, False,
            intervals, intervals_sec)

        self.interval = (intervals, intervals_sec)
        self.minval = mindate
        self.maxval = maxdate
        self.minorticks = minorticks
        self.tickvals = ticks
        self.autoformat = format
