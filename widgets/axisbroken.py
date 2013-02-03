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

def _(text, disambiguation=None, context='BrokenAxis'):
    '''Translate text.'''
    return unicode( 
        qt4.QCoreApplication.translate(context, text, disambiguation))

class AxisBroken(axis.Axis):

    typename = 'axis-broken'
    description = 'Axis with breaks in it'

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

    def updateAxisLocation(self, bounds, otherposition=None):
        """Recalculate broken axis positions."""

        axis.Axis.updateAxisLocation(self, bounds, otherposition=otherposition)

        s = self.settings

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

        # save these
        self.orig_coordPerp = self.coordPerp
        self.orig_coordParr1 = self.coordParr1
        self.orig_coordParr2 = self.coordParr2

    def computePlottedRange(self):

        axis.Axis.computePlottedRange(self)

        self.orig_plottedrange = self.plottedrange
        points = self.settings.breakPoints
        self.breakvnum = num = len(points)/2 + 1
        self.breakvlist = [self.plottedrange[0]] + points[:len(points)/2*2] + [
            self.plottedrange[1]]
        print self.breakvlist, num

        # axis values for starting and stopping
        self.breakvstarts = [ self.breakvlist[i*2] for i in xrange(num) ]
        self.breakvstops = [ self.breakvlist[i*2+1] for i in xrange(num) ]

    def switchBreak(self, num):
        """Switch to break given (or None to disable."""

        print "switch", num

        if num is None:
            self.plottedrange = self.orig_plottedrange
            self.coordPerp = self.orig_coordPerp
            self.coordParr1 = self.orig_coordParr1
            self.coordParr2 = self.orig_coordParr2
        else:
            self.plottedrange = [self.breakvstarts[num], self.breakvstops[num]]
            d = self.orig_coordParr2 - self.orig_coordParr1
            self.coordPerp = self.orig_coordPerp
            self.coordParr1 = self.orig_coordParr1 + d*self.posstarts[num]
            self.coordParr2 = self.orig_coordParr1 + d*self.posstops[num]

    def draw(self, parentposn, phelper, outerbounds=None):
        """Plot the axis on the painter.
        """

        posn = self.computeBounds(parentposn, phelper)
        self.updateAxisLocation(posn)



# allow the factory to instantiate an image
document.thefactory.register( AxisBroken )
