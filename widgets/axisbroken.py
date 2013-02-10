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

import bisect

import numpy as N

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.document as document

import axis
import controlgraph

def _(text, disambiguation=None, context='BrokenAxis'):
    '''Translate text.'''
    return unicode( 
        qt4.QCoreApplication.translate(context, text, disambiguation))

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
                ) )
        s.add( setting.FloatList(
                'breakPosns',
                [],
                descr = _('Positions (fractions) along axis where to break'),
                usertext = _('Break positions'),
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
        breaks."""

        if self.rangeswitch is not None:
            return axis.Axis.plotterToGraphCoords(self, bounds, vals)

        # first work out which break region the values are in
        out = []
        for val in vals:
            coord = N.nan
            breaki = bisect.bisect_left(self.breakvstarts, val) - 1
            print self.breakvstarts, val
            print breaki
            if ( breaki >= 0 and breaki < self.breakvnum and
                 val <= self.breakvstops[breaki] ):
                self.switchBreak(breaki, bounds)
                coord = axis.Axis.plotterToGraphCoords(self, bounds, val)[0]
            out.append(coord)
        self.switchBreak(None, bounds)

        return N.array(out)

    def updateAxisLocation(self, bounds, otherposition=None):
        """Recalculate broken axis positions."""

        s = self.settings

        if self.document.changeset != self.breakchangeset:
            self.breakchangeset = self.document.changeset

            # actually start and stop values on axis
            points = s.breakPoints
            num = len(points) / 2
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

        self.orig_plottedrange = self.plottedrange
        points = self.settings.breakPoints
        self.breakvnum = num = len(points)/2 + 1
        self.breakvlist = [self.plottedrange[0]] + points[:len(points)/2*2] + [
            self.plottedrange[1]]

        # axis values for starting and stopping
        self.breakvstarts = [ self.breakvlist[i*2] for i in xrange(num) ]
        self.breakvstops = [ self.breakvlist[i*2+1] for i in xrange(num) ]

        # compute ticks for each range
        self.minorticklist = []
        self.majorticklist = []
        for i in xrange(self.breakvnum):
            self.plottedrange = [self.breakvstarts[i], self.breakvstops[i]]
            self.computeTicks(allowauto=False)
            self.minorticklist.append(self.minortickscalc)
            self.majorticklist.append(self.majortickscalc)

        self.plottedrange = self.orig_plottedrange

    def _autoMirrorDraw(self, posn, painter):
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

        for i in xrange(self.breakvnum):
            self.switchBreak(i, posn, otherposition=otheredge)

            # plot coordinates of ticks
            coordticks = self._graphToPlotter(self.majorticklist[i])
            coordminorticks = self._graphToPlotter(self.minorticklist[i])

            if not s.MinorTicks.hide:
                self._drawMinorTicks(painter, coordminorticks)
            if not s.MajorTicks.hide:
                self._drawMajorTicks(painter, coordticks)

        self.switchBreak(None, posn)

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

            for i in xrange(self.breakvnum):
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
        for i in xrange(self.breakvnum):
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

        # mirror axis at other side of plot
        if s.autoMirror and self._shouldAutoMirror():
            self._autoMirrorDraw(posn, painter)

        self._drawTextWithoutOverlap(painter, texttorender)

        # make control item for axis
        phelper.setControlGraph(self, [ controlgraph.ControlAxisLine(
                    self, self.settings.direction, self.coordParr1,
                    self.coordParr2, self.coordPerp, posn) ])

# allow the factory to instantiate an image
document.thefactory.register( AxisBroken )
