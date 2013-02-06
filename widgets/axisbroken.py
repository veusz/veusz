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

    def switchBreak(self, num, posn):
        """Switch to break given (or None to disable)."""
        self.rangeswitch = num
        if num is None:
            self.plottedrange = self.orig_plottedrange
        else:
            self.plottedrange = [self.breakvstarts[num], self.breakvstops[num]]
        self.updateAxisLocation(posn)

    def updateAxisLocation(self, bounds, otherposition=None):
        """Recalculate broken axis positions."""

        s = self.settings

        bounds = list(bounds)
        if self.rangeswitch is None:
            pass
        else:
            if s.direction == 'horizontal':
                d = bounds[2]-bounds[0]
                n0 = bounds[0] + d*self.posstarts[self.rangeswitch]
                n2 = bounds[0] + d*self.posstops[self.rangeswitch]
                bounds[0] = n0
                bounds[2] = n2
            else:
                d = bounds[1]-bounds[3]
                n1 = bounds[3] + d*self.posstops[self.rangeswitch]
                n3 = bounds[3] + d*self.posstarts[self.rangeswitch]
                bounds[1] = n1
                bounds[3] = n3

        axis.Axis.updateAxisLocation(self, bounds, otherposition=otherposition)

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

        # get fractional positions for starting and stopping
        self.posstarts = starts = [0.]
        self.posstops = stops = []

        for pos in posns:
            stops.append( pos - breakgap/2. )
            starts.append( pos + breakgap/2. )

        stops.append(1.)

    def computePlottedRange(self):

        axis.Axis.computePlottedRange(self)

        self.orig_plottedrange = self.plottedrange
        points = self.settings.breakPoints
        self.breakvnum = num = len(points)/2 + 1
        self.breakvlist = [self.plottedrange[0]] + points[:len(points)/2*2] + [
            self.plottedrange[1]]

        # axis values for starting and stopping
        self.breakvstarts = [ self.breakvlist[i*2] for i in xrange(num) ]
        self.breakvstops = [ self.breakvlist[i*2+1] for i in xrange(num) ]

    def draw(self, parentposn, phelper, outerbounds=None):
        """Plot the axis on the painter.
        """

        posn = self.computeBounds(parentposn, phelper)
        self.updateAxisLocation(posn)

        # exit if axis is hidden
        if self.settings.hide:
            return

        self.computePlottedRange()
        painter = phelper.painter(self, posn)
        with painter:
            for i in xrange(self.breakvnum):
                self.switchBreak(i, posn)
                self.computeTicks(allowauto=False)
                self._axisDraw(posn, parentposn, outerbounds, painter, phelper)

            self.switchBreak(None, posn)

        # make control item for axis
        phelper.setControlGraph(self, [ controlgraph.ControlAxisLine(
                    self, self.settings.direction, self.coordParr1,
                    self.coordParr2, self.coordPerp, posn) ])

# allow the factory to instantiate an image
document.thefactory.register( AxisBroken )
