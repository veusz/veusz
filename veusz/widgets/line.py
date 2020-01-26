#    Copyright (C) 2008 Jeremy S. Sanders
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

"""Plotting a line with arrowheads or labels."""

from __future__ import division
import math
import itertools
import numpy as N

from ..compat import czip
from .. import qtall as qt
from .. import setting
from .. import document
from .. import utils

from . import controlgraph
from . import plotters

def _(text, disambiguation=None, context='Line'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class Line(plotters.FreePlotter):
    """A line on the plot/graph."""
    typename='line'
    description=_('Line or arrow')
    allowusercreation = True

    @staticmethod
    def showOrHideSetn(v):
        """Whether to show or hide length/angle or position."""
        a = ('length', 'angle')
        b = ('xPos2', 'yPos2')
        return (a,b) if v=='length-angle' else (b,a)

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.FreePlotter.addSettings(s)

        s.add( setting.ChoiceSwitch(
            'mode',
            ('length-angle', 'point-to-point'),
            'length-angle',
            descr=_('Provide line position and length,angle or '
                    'first and second points'),
            usertext=_('Mode'),
            formatting=False,
            showfn=klass.showOrHideSetn), 0)

        s.add( setting.DatasetExtended(
            'length',
            [0.2],
            descr=_('List of fractional lengths, dataset or expression'),
            usertext=_('Lengths'),
            formatting=False), 4 )
        s.add( setting.DatasetExtended(
            'angle',
            [0.],
            descr=_('List of angle of lines, dataset or expression '
                    '(degrees)'),
            usertext=_('Angles'),
            formatting=False), 5 )
        s.add( setting.DatasetExtended(
            'xPos2',
            [1.],
            descr=_('List of fractional X coordinates, dataset or '
                    'expression for point 2'),
            usertext=_('X positions 2'),
            formatting=False), 6 )
        s.add( setting.DatasetExtended(
            'yPos2',
            [1.],
            descr=_('List of fractional Y coordinates, dataset or '
                    'expression for point 2'),
            usertext=_('Y positions 2'),
            formatting=False), 7 )

        s.add( setting.Bool('clip', False,
                            descr=_('Clip line to its container'),
                            usertext=_('Clip'),
                            formatting=True), 0 )

        s.add( setting.Line('Line',
                            descr = _('Line style'),
                            usertext = _('Line')),
               pixmap = 'settings_plotline' )
        s.add( setting.ArrowFill('Fill',
                                 descr = _('Arrow fill settings'),
                                 usertext = _('Arrow fill')),
               pixmap = 'settings_plotmarkerfill' )

        s.add( setting.DistancePt('arrowSize', '5pt',
                                  descr = _('Size of arrow to plot'),
                                  usertext=_('Arrow size'), formatting=True), 0)
        s.add( setting.Arrow('arrowright', 'none',
                             descr = _('Arrow to plot on right side'),
                             usertext=_('Arrow right'), formatting=True), 0)
        s.add( setting.Arrow('arrowleft', 'none',
                             descr = _('Arrow to plot on left side'),
                             usertext=_('Arrow left'), formatting=True), 0)

    def _computeLinesLengthAngle(self, posn, lengthscaling):
        """Return set of lines to plot for length-angle."""

        s = self.settings
        d = self.document

        # translate coordinates from axes or relative values
        xpos, ypos = self._getPlotterCoords(posn)
        # get lengths and angles of lines
        length = s.get('length').getFloatArray(d)
        angle = s.get('angle').getFloatArray(d)
        if xpos is None or ypos is None or length is None or angle is None:
            return None
        length *= lengthscaling

        maxlen = max( len(xpos), len(ypos), len(length), len(angle) )
        if maxlen > 1:
            if len(xpos) == 1: xpos = itertools.cycle(xpos)
            if len(ypos) == 1: ypos = itertools.cycle(ypos)
            if len(length) == 1: length = itertools.cycle(length)
            if len(angle) == 1: angle = itertools.cycle(angle)

        out = []
        for v in czip(xpos, ypos, length, angle):
            # skip lines which have nans
            if N.all( N.isfinite(v) ):
                out.append(v)
        return out

    def _computeLinesPointToPoint(self, posn):
        """Return set of lines for point to point."""

        # translate coordinates from axes or relative values
        xpos, ypos = self._getPlotterCoords(posn)
        xpos2, ypos2 = self._getPlotterCoords(posn, xsetting='xPos2', ysetting='yPos2')
        if xpos is None or ypos is None or xpos2 is None or ypos2 is None:
            return None

        maxlen = max( len(xpos), len(ypos), len(xpos2), len(ypos2) )
        if maxlen > 1:
            if len(xpos) == 1: xpos = itertools.cycle(xpos)
            if len(ypos) == 1: ypos = itertools.cycle(ypos)
            if len(xpos2) == 1: xpos2 = itertools.cycle(xpos2)
            if len(ypos2) == 1: ypos2 = itertools.cycle(ypos2)

        out = []
        for v in czip(xpos, ypos, xpos2, ypos2):
            # skip nans again
            if N.all( N.isfinite(v) ):
                length = math.sqrt( (v[0]-v[2])**2 + (v[1]-v[3])**2 )
                angle = math.atan2( v[3]-v[1], v[2]-v[0] ) / math.pi * 180.
                out.append( (v[0], v[1], length, angle) )
        return out

    def draw(self, posn, phelper, outerbounds = None):
        """Plot the key on a plotter."""

        s = self.settings
        d = self.document
        if s.hide:
            return

        # if a dataset is used, we can't use control items
        isnotdataset = ( not s.get('xPos').isDataset(d) and 
                         not s.get('yPos').isDataset(d) )

        if s.mode == 'length-angle':
            isnotdataset = ( isnotdataset and
                             not s.get('length').isDataset(d) and
                             not s.get('angle').isDataset(d) )
        else:
            isnotdataset = ( isnotdataset and
                             not s.get('xPos2').isDataset(d) and
                             not s.get('yPos2').isDataset(d) )

        # now do the drawing
        clip = None
        if s.clip:
            clip = qt.QRectF(
                qt.QPointF(posn[0], posn[1]),
                qt.QPointF(posn[2], posn[3]))
        painter = phelper.painter(self, posn, clip=clip)
        with painter:
            # adjustable positions for the lines
            arrowsize = s.get('arrowSize').convert(painter)

            # drawing settings for line
            if not s.Line.hide:
                painter.setPen( s.get('Line').makeQPen(painter) )
            else:
                painter.setPen( qt.QPen(qt.Qt.NoPen) )

            # settings for fill
            if not s.Fill.hide:
                painter.setBrush( s.get('Fill').makeQBrush(painter) )
            else:
                painter.setBrush( qt.QBrush() )

            # iterate over positions
            scaling = posn[2]-posn[0]
            if s.mode == 'length-angle':
                lines = self._computeLinesLengthAngle(posn, scaling)
            else:
                lines = self._computeLinesPointToPoint(posn)

            if lines is None:
                return

            controlgraphitems = []
            for index, (x, y, l, a) in enumerate(lines):

                utils.plotLineArrow(
                    painter, x, y, l, a,
                    arrowsize=arrowsize,
                    arrowleft=s.arrowleft,
                    arrowright=s.arrowright)

                if isnotdataset:
                    cgi = controlgraph.ControlLine(
                        self, phelper, x, y,
                        x + l*math.cos(a/180.*math.pi),
                        y + l*math.sin(a/180.*math.pi))
                    cgi.index = index
                    cgi.widgetposn = posn
                    controlgraphitems.append(cgi)

        phelper.setControlGraph(self, controlgraphitems)

    def updateControlItem(self, cgi, pt1, pt2):
        """If control items are moved, update line."""
        s = self.settings

        # calculate new position coordinate for item
        xpos, ypos = self._getGraphCoords(
            cgi.widgetposn, [pt1[0], pt1[0]+1], [pt1[1], pt1[1]+1])
        if xpos is None or ypos is None:
            return

        x = s.get('xPos').getFloatArray(self.document)
        y = s.get('yPos').getFloatArray(self.document)
        if x is None or y is None:
            return
        x, y = list(x), list(y)

        idx = min(cgi.index, len(x)-1)
        if not N.allclose(x[idx], xpos[0]):
            x[idx] = utils.round2delt(xpos[0], xpos[1])
        idx = min(cgi.index, len(y)-1)
        if not N.allclose(y[idx], ypos[0]):
            y[idx] = utils.round2delt(ypos[0], ypos[1])
        operations = [
            document.OperationSettingSet(s.get('xPos'), x),
            document.OperationSettingSet(s.get('yPos'), y),
            ]

        if s.mode == 'length-angle':
            # convert 2nd point to length, angle
            def la(ptx, pty):
                length = ( math.sqrt( (ptx-pt1[0])**2 + (pty-pt1[1])**2 ) /
                           (cgi.widgetposn[2]-cgi.widgetposn[0]) )
                angle = ( (math.atan2( pty-pt1[1], ptx-pt1[0] )
                           * 180. / math.pi) % 360. )
                return length, angle
            length, angle = la(pt2[0], pt2[1])
            # calculate length angle for neighbouring point, to get delta
            ldelt, adelt = la(pt2[0]+1, pt2[1]+1)

            # update values
            l = s.get('length').getFloatArray(self.document)
            a = s.get('angle').getFloatArray(self.document)
            if l is None or a is None:
                return
            l, a = list(l), list(a)

            idx = min(cgi.index, len(l)-1)
            if abs(l[idx]-length) > 1e-8:
                l[idx] = utils.round2delt(length, ldelt)
            idx = min(cgi.index, len(a)-1)
            if abs(a[idx]-angle) > 1e-8:
                a[idx] = utils.round2delt(angle, adelt)

            operations += [
                document.OperationSettingSet(s.get('length'), l),
                document.OperationSettingSet(s.get('angle'), a),
            ]
        else:
            xpos2, ypos2 = self._getGraphCoords(
                cgi.widgetposn, [pt2[0], pt2[0]+1], [pt2[1], pt2[1]+1])
            if xpos2 is not None and ypos2 is not None:
                x2 = s.get('xPos2').getFloatArray(self.document)
                y2 = s.get('yPos2').getFloatArray(self.document)
                if x2 is None or y2 is None:
                    return
                x2, y2 = list(x2), list(y2)

                idx = min(cgi.index, len(x2)-1)
                if not N.allclose(x2[idx], xpos2[0]):
                    x2[idx] = utils.round2delt(xpos2[0], xpos2[1])
                idx = min(cgi.index, len(y2)-1)
                if not N.allclose(y2[idx], ypos2[0]):
                    y2[idx] = utils.round2delt(ypos2[0], ypos2[1])

                operations += [
                    document.OperationSettingSet(s.get('xPos2'), x2),
                    document.OperationSettingSet(s.get('yPos2'), y2)
                ]

        self.document.applyOperation(
            document.OperationMultiple(operations, descr=_('adjust lines')) )

document.thefactory.register( Line )
