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

import itertools
import math
import numpy as N

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.document as document
import veusz.utils as utils

import controlgraph
import plotters

class Line(plotters.FreePlotter):
    """A line on the plot/graph."""
    typename='line'
    description='Line or arrow'
    allowusercreation = True
    
    def __init__(self, parent, name=None):
        """Construct plotter."""
        plotters.FreePlotter.__init__(self, parent, name=name)
        if type(self) == Line:
            self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.FreePlotter.addSettings(s)

        s.add( setting.DatasetOrFloatList('length', [0.2],
                                          descr='List of fractional '
                                          'lengths or dataset',
                                          usertext='Lengths',
                                          formatting=False), 3 )
        s.add( setting.DatasetOrFloatList('angle', [0.],
                                          descr='Angle of lines or '
                                          'dataset',
                                          usertext='Angles',
                                          formatting=False), 4 )

        s.add( setting.Line('Line',
                            descr = 'Line style',
                            usertext = 'Line'),
               pixmap = 'settings_plotline' )
        s.add( setting.ArrowFill('Fill',
                                 descr = 'Arrow fill settings',
                                 usertext = 'Arrow fill'),
               pixmap = 'settings_plotmarkerfill' )

        s.add( setting.DistancePt('arrowSize', '5pt',
                                  descr = 'Size of arrow to plot',
                                  usertext='Arrow size', formatting=True), 0)
        s.add( setting.Arrow('arrowright', 'none',
                             descr = 'Arrow to plot on right side',
                             usertext='Arrow right', formatting=True), 0)
        s.add( setting.Arrow('arrowleft', 'none',
                             descr = 'Arrow to plot on left side',
                             usertext='Arrow left', formatting=True), 0)


    def draw(self, posn, phelper, outerbounds = None):
        """Plot the key on a plotter."""

        s = self.settings
        d = self.document
        if s.hide:
            return

        # get lengths and angles of lines
        length = s.get('length').getFloatArray(d)
        angle = s.get('angle').getFloatArray(d)
        if length is None or angle is None:
            return

        # translate coordinates from axes or relative values
        xpos, ypos = self._getPlotterCoords(posn)
        if xpos is None or ypos is None:
            # we can't calculate coordinates
            return

        # if a dataset is used, we can't use control items
        isnotdataset = ( not s.get('xPos').isDataset(d) and 
                         not s.get('yPos').isDataset(d) and
                         not s.get('length').isDataset(d) and
                         not s.get('angle').isDataset(d) )

        # adjustable positions for the lines
        controlgraphitems = []
        arrowsize = s.get('arrowSize').convert(phelper)

        # now do the drawing
        painter = phelper.painter(self, posn)

        # drawing settings for line
        if not s.Line.hide:
            painter.setPen( s.get('Line').makeQPen(phelper) )
        else:
            painter.setPen( qt4.QPen(qt4.Qt.NoPen) )

        # settings for fill
        if not s.Fill.hide:
            painter.setBrush( s.get('Fill').makeQBrush() )
        else:
            painter.setBrush( qt4.QBrush() )

        # iterate over positions
        index = 0
        dx, dy = posn[2]-posn[0], posn[3]-posn[1]
        for x, y, l, a in itertools.izip(xpos, ypos,
                                         itertools.cycle(length),
                                         itertools.cycle(angle)):

            if N.isfinite(x) and N.isfinite(y) and N.isfinite(a):
                utils.plotLineArrow(painter, x, y, l*dx, a,
                                    arrowsize=arrowsize,
                                    arrowleft=s.arrowleft,
                                    arrowright=s.arrowright)

            if isnotdataset:
                cgi = controlgraph.ControlLine(
                    self, x, y,
                    x + l*dx*math.cos(a/180.*math.pi),
                    y + l*dx*math.sin(a/180.*math.pi))
                cgi.index = index
                cgi.widgetposn = posn
                index += 1
                controlgraphitems.append(cgi)

        phelper.setControlGraph(self, controlgraphitems)

    def updateControlItem(self, cgi, pt1, pt2):
        """If control items are moved, update line."""
        s = self.settings

        # calculate new position coordinate for item
        xpos, ypos = self._getGraphCoords(cgi.widgetposn,
                                          pt1[0], pt1[1])
        if xpos is None or ypos is None:
            return

        length = ( math.sqrt( (pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2 ) /
                   (cgi.widgetposn[2]-cgi.widgetposn[0]) )
        angle = ( (math.atan2( pt2[1]-pt1[1], pt2[0]-pt1[0] )
                   * 180. / math.pi) % 360. )

        x, y = list(s.xPos), list(s.yPos)
        l, a = list(s.length), list(s.angle)
        x[cgi.index] = xpos
        y[cgi.index] = ypos
        l[min(cgi.index, len(l)-1)] = length
        a[min(cgi.index, len(a)-1)] = angle

        operations = (
            document.OperationSettingSet(s.get('xPos'), x),
            document.OperationSettingSet(s.get('yPos'), y),
            document.OperationSettingSet(s.get('length'), l),
            document.OperationSettingSet(s.get('angle'), a),
            )
        self.document.applyOperation(
            document.OperationMultiple(operations, descr='adjust shape') )

document.thefactory.register( Line )
