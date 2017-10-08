#    Copyright (C) 2013 Jeremy S. Sanders
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

'''An axis which can be broken in places.'''

from __future__ import division
import bisect

import numpy as N

from ..compat import crange, czip
from .. import qtall as qt4
from .. import setting
from .. import document
from .. import utils

from . import axis
from . import controlgraph

def _(text, disambiguation=None, context='BrokenAxis'):
    '''Translate text.'''
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class AxisBroken(axis.Axis):
    '''An axis widget which can have gaps in it.'''

    typename = 'axis-broken'
    description = 'Axis with breaks in it'

    def __init__(self, parent, name=None):
        """Initialise axis."""
        axis.Axis.__init__(self, parent, name=name)
        self.rangeswitch = None
        self.breakchangeset = -1

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        axis.Axis.addSettings(s)

        s.add( setting.FloatList(
                'breakPoints',
                [],
                descr = _('Pairs of values to start and stop breaks'),
                usertext = _('Break pairs'),
                ), 4 )
        s.add( setting.FloatList(
                'breakPosns',
                [],
                descr = _('Positions (fractions) along axis where to break'),
                usertext = _('Break positions'),
                formatting=True,
                ) )

    def switchBreak(self, num, posn, otherposition=None):
        """Switch to break given (or None to disable)."""
        self.rangeswitch = num
        if num is None:
            self.plottedrange = self.orig_plottedrange
        else:
            self.plottedrange = [self.breakvstarts[num], self.breakvstops[num]]
        self.updateAxisLocation(posn, otherposition=otherposition)

    def plotterToGraphCoords(self, bounds, vals):
        """Convert values in plotter coordinates to data values.  This
        needs to know about whether we've not switched between the
        breaks.

        Note that this implementation is very slow! Hopefully it won't
        be called often.
        """

        if self.rangeswitch is not None:
            return axis.Axis.plotterToGraphCoords(self, bounds, vals)

        # support single int/float values
        try:
            iter(vals)
            issingle = False
        except TypeError:
            vals = N.array([vals])
            issingle = True

        # scaled to be fractional coordinates in bounds
        if self.settings.direction == 'horizontal':
            svals = (vals - bounds[0]) / (bounds[2] - bounds[0])
        else:
            svals = (vals - bounds[3]) / (bounds[1] - bounds[3])

        # first work out which break region the values are in
        out = []
        for sval, val in czip(svals, vals):
            # find index for appropriated scaled starting value
            breaki = bisect.bisect_left(self.posstarts, sval) - 1

            if ( breaki >= 0 and breaki < self.breakvnum and
                 sval <= self.posstops[breaki] ):
                self.switchBreak(breaki, bounds)
                coord = axis.Axis.plotterToGraphCoords(
                    self, bounds, N.array([val]))[0]
            else:
                coord = N.nan
            out.append(coord)
        self.switchBreak(None, bounds)

        if issingle:
            return out[0]
        else:
            return N.array(out)

    def _graphToPlotter(self, vals):
        """Convert graph values to plotter coords.
        This could be slow if no range selected
        """

        if self.rangeswitch is not None:
            return axis.Axis._graphToPlotter(self, vals)

        out = []
        for val in vals:
            breaki = bisect.bisect_left(self.breakvstarts, val) - 1
            if breaki >= 0 and breaki < self.breakvnum:
                if val > self.breakvstops[breaki] and breaki < self.breakvnum-1:
                    # in gap, so use half-value
                    coord = 0.5*(self.posstops[breaki]+self.posstarts[breaki+1])

                    b = self.currentbounds
                    if self.settings.direction == 'horizontal':
                        coord = coord*(b[2] - b[0]) + b[0]
                    else:
                        coord = coord*(b[3] - b[1]) + b[1]

                else:
                    # lookup value
                    self.switchBreak(breaki, self.currentbounds)
                    coord = axis.Axis._graphToPlotter(self, N.array([val]))
            else:
                coord = N.nan
            out.append(coord)
        self.switchBreak(None, self.currentbounds)
        return N.array(out)

    def updateAxisLocation(self, bounds, otherposition=None):
        """Recalculate broken axis positions."""

        s = self.settings

        if self.document.changeset != self.breakchangeset:
            self.breakchangeset = self.document.changeset

            # actually start and stop values on axis
            num = len(s.breakPoints) // 2
            posns = list(s.breakPosns)
            posns.sort()

            # add on more break positions if not specified
            if len(posns) < num:
                start = 0.
                if len(posns) != 0:
                    start = posns[-1]
                posns = posns + list(
                    N.arange(1,num-len(posns)+1) *
                    ( (1.-start) / (num-len(posns)+1) + start ))

            # fractional difference between starts and stops
            breakgap = 0.05

            # collate fractional positions for starting and stopping
            starts = [0.]
            stops = []
            for pos in posns:
                stops.append( pos - breakgap/2. )
                starts.append( pos + breakgap/2. )
            stops.append(1.)

            # scale according to allowable range
            d = s.upperPosition - s.lowerPosition
            self.posstarts = N.array(starts)*d + s.lowerPosition
            self.posstops = N.array(stops)*d + s.lowerPosition

        # pass lower and upper ranges if a particular range is chosen
        if self.rangeswitch is None:
            lowerupper = None
        else:
            lowerupper = ( self.posstarts[self.rangeswitch],
                           self.posstops[self.rangeswitch] )

        axis.Axis.updateAxisLocation(self, bounds,
                                     otherposition=otherposition,
                                     lowerupperposition=lowerupper)

    def computePlottedRange(self):
        """Given range of data, recompute stops and start values of
        breaks."""

        axis.Axis.computePlottedRange(self)

        r = self.orig_plottedrange = self.plottedrange
        points = list(self.settings.breakPoints)
        points.sort()
        if r[1] < r[0]:
            points.reverse()

        # filter to range
        newpoints = []
        for i in crange(0, len(points)//2 * 2, 2):
            if points[i] >= min(r) and points[i+1] <= max(r):
                newpoints += [points[i], points[i+1]]

        self.breakvnum = num = len(newpoints)//2 + 1
        self.breakvlist = [self.plottedrange[0]] + newpoints + [
            self.plottedrange[1]]

        # axis values for starting and stopping
        self.breakvstarts = [ self.breakvlist[i*2] for i in crange(num) ]
        self.breakvstops = [ self.breakvlist[i*2+1] for i in crange(num) ]

        # compute ticks for each range
        self.minorticklist = []
        self.majorticklist = []
        for i in crange(self.breakvnum):
            self.plottedrange = [self.breakvstarts[i], self.breakvstops[i]]
            reverse = self.plottedrange[0] > self.plottedrange[1]
            if reverse:
                self.plottedrange.reverse()
            self.computeTicks(allowauto=False)
            if reverse:
                self.plottedrange.reverse()
            self.minorticklist.append(self.minortickscalc)
            self.majorticklist.append(self.majortickscalc)

        self.plottedrange = self.orig_plottedrange

    def _drawAutoMirrorTicks(self, posn, painter):
        """Mirror axis to opposite side of graph if there isn't an
        axis there already."""

        # swap axis to other side
        s = self.settings
        if s.otherPosition < 0.5:
            otheredge = 1.
        else:
            otheredge = 0.

        # temporarily change position of axis to other side for drawing
        self.updateAxisLocation(posn, otherposition=otheredge)
        if not s.Line.hide:
            self._drawAxisLine(painter)

        for i in crange(self.breakvnum):
            self.switchBreak(i, posn, otherposition=otheredge)

            # plot coordinates of ticks
            coordticks = self._graphToPlotter(self.majorticklist[i])
            coordminorticks = self._graphToPlotter(self.minorticklist[i])

            if not s.MinorTicks.hide:
                self._drawMinorTicks(painter, coordminorticks)
            if not s.MajorTicks.hide:
                self._drawMajorTicks(painter, coordticks)

        self.switchBreak(None, posn)

    def _drawAxisLine(self, painter):
        """Draw the line of the axis, indicating broken positions.
        We currently use a triangle to mark the broken position
        """

        # these are x and y, or y and x coordinates
        p1 = [self.posstarts[0]]
        p2 = [0.]

        # mirror shape using this setting
        markdirn = -1
        if self.coordReflected:
            markdirn = -markdirn

        # add shape for each break
        for start, stop in czip( self.posstarts[1:], self.posstops[:-1] ):
            p1 += [stop, (start+stop)*0.5, start]
            p2 += [0, markdirn*(start-stop)*0.5, 0]

        # end point
        p1.append(self.posstops[-1])
        p2.append(0.)

        # scale points by length of axis and add correct origin
        scale = self.coordParr2 - self.coordParr1
        p1 = N.array(p1) * scale + self.coordParr1
        p2 = N.array(p2) * scale + self.coordPerp

        if self.settings.direction == 'vertical':
            p1, p2 = p2, p1

        # convert to polygon and draw
        poly = qt4.QPolygonF()
        utils.addNumpyToPolygonF(poly, p1, p2)

        pen = self.settings.get('Line').makeQPen(painter)
        pen.setCapStyle(qt4.Qt.FlatCap)
        painter.setPen(pen)
        painter.drawPolyline(poly)

    def drawGrid(self, parentposn, phelper, outerbounds=None,
                 ontop=False):
        """Code to draw gridlines.

        This is separate from the main draw routine because the grid
        should be behind/infront the data points.
        """

        s = self.settings
        if ( s.hide or (s.MinorGridLines.hide and s.GridLines.hide) or
             s.GridLines.onTop != bool(ontop) ):
            return

        # draw grid on a different layer, depending on whether on top or not
        layer = (-2, -1)[bool(ontop)]
        painter = phelper.painter(self, parentposn, layer=layer)
        self.updateAxisLocation(parentposn)

        with painter:
            painter.save()
            painter.setClipRect( qt4.QRectF(
                    qt4.QPointF(parentposn[0], parentposn[1]),
                    qt4.QPointF(parentposn[2], parentposn[3]) ) )

            for i in crange(self.breakvnum):
                self.switchBreak(i, parentposn)
                if not s.MinorGridLines.hide:
                    coordminorticks = self._graphToPlotter(self.minorticklist[i])
                    self._drawGridLines('MinorGridLines', painter, coordminorticks,
                                        parentposn)
                if not s.GridLines.hide:
                    coordticks = self._graphToPlotter(self.majorticklist[i])
                    self._drawGridLines('GridLines', painter, coordticks,
                                        parentposn)

            self.switchBreak(None, parentposn)
            painter.restore()

    def _axisDraw(self, posn, parentposn, outerbounds, painter, phelper):
        """Main drawing routine of axis."""

        s = self.settings

        # multiplication factor if reflection on the axis is requested
        sign = 1
        if s.direction == 'vertical':
            sign *= -1
        if self.coordReflected:
            sign *= -1

        # keep track of distance from axis
        # text to output
        texttorender = []

        # plot the line along the axis
        if not s.Line.hide:
            self._drawAxisLine(painter)

        max_delta = 0
        for i in crange(self.breakvnum):
            self.switchBreak(i, posn)

            # plot coordinates of ticks
            coordticks = self._graphToPlotter(self.majorticklist[i])
            coordminorticks = self._graphToPlotter(self.minorticklist[i])

            self._delta_axis = 0

            # plot minor ticks
            if not s.MinorTicks.hide:
                self._drawMinorTicks(painter, coordminorticks)

            # plot major ticks
            if not s.MajorTicks.hide:
                self._drawMajorTicks(painter, coordticks)

            # plot tick labels
            suppresstext = self._suppressText(painter, parentposn, outerbounds)
            if not s.TickLabels.hide and not suppresstext:
                self._drawTickLabels(phelper, painter, coordticks, sign,
                                     outerbounds, self.majorticklist[i],
                                     texttorender)

            # this is the maximum delta of any of the breaks
            max_delta = max(max_delta, self._delta_axis)

        self.switchBreak(None, posn)
        self._delta_axis = max_delta

        # draw an axis label
        if not s.Label.hide and not suppresstext:
            self._drawAxisLabel(painter, sign, outerbounds, texttorender)

        self._drawTextWithoutOverlap(painter, texttorender)

        # make control item for axis
        phelper.setControlGraph(self, [ controlgraph.ControlAxisLine(
                    self, self.settings.direction, self.coordParr1,
                    self.coordParr2, self.coordPerp, posn) ])

# allow the factory to instantiate the widget
document.thefactory.register( AxisBroken )
