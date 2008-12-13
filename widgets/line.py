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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
###############################################################################

# $Id$

"""Plotting a line with arrowheads or labels."""

import itertools
import math
import numpy as N

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.document as document
import veusz.utils as utils

import widget
import page
import graph
import controlgraph

class Line(widget.Widget):
    """A line on the plot/graph."""
    allowedparenttypes = [graph.Graph, page.Page]
    typename='line'
    description='Line or arrow'

    class ArrowFillBrush(setting.Brush):
        def __init__(self, name, **args):
            setting.Brush.__init__(self, name, **args)

            self.get('color').newDefault( setting.Reference(
                '../Line/color') )
    
    def __init__(self, parent, name=None):
        widget.Widget.__init__(self, parent, name=name)
        s = self.settings
        s.add( setting.DatasetOrFloatList('xPos', 0.5,
                                          descr='List of fractional X '
                                          'coordinates or dataset',
                                          usertext='X positions',
                                          formatting=False) )
        s.add( setting.DatasetOrFloatList('yPos', 0.5,
                                          descr='List of fractional Y '
                                          'coordinates or dataset',
                                          usertext='Y positions',
                                          formatting=False) )
        s.add( setting.DatasetOrFloatList('length', 0.2,
                                          descr='List of fractional '
                                          'lengths or dataset',
                                          usertext='Lengths',
                                          formatting=False) )
        s.add( setting.DatasetOrFloatList('angle', 0.,
                                          descr='Angle of lines or '
                                          'dataset',
                                          usertext='Angles',
                                          formatting=False) )

        s.add( setting.Line('Line',
                            descr = 'Line style',
                            usertext = 'Line'),
               pixmap = 'plotline' )
        s.add( Line.ArrowFillBrush('Fill',
                                   descr = 'Arrow fill settings',
                                   usertext = 'Arrow fill'),
               pixmap = 'plotmarkerfill' )

        s.add( setting.Choice('positioning',
                              ['axes', 'relative'], 'relative',
                              descr='Use axes or fractional '
                              'position to place label',
                              usertext='Position mode',
                              formatting=False) )
        s.add( setting.Axis('xAxis', 'x', 'horizontal',
                            descr = 'Name of X-axis to use',
                            usertext='X axis') )
        s.add( setting.Axis('yAxis', 'y', 'vertical',
                            descr = 'Name of Y-axis to use',
                            usertext='Y axis') )

        s.add( setting.Distance('arrowSize', '5pt',
                                descr = 'Size of arrow to plot',
                                usertext='Arrow size', formatting=True), 0)
        s.add( setting.Arrow('arrowright', 'none',
                             descr = 'Arrow to plot on right side',
                             usertext='Arrow right', formatting=True), 0)
        s.add( setting.Arrow('arrowleft', 'none',
                             descr = 'Arrow to plot on left side',
                             usertext='Arrow left', formatting=True), 0)


    def draw(self, posn, painter, outerbounds = None):
        """Plot the key on a plotter."""

        s = self.settings
        d = self.document
        if s.hide:
            return

        # get positions of shapes
        xpos = s.get('xPos').getFloatArray(d)
        ypos = s.get('yPos').getFloatArray(d)
        length = s.get('length').getFloatArray(d)
        angle = s.get('angle').getFloatArray(d)

        if (xpos is None or ypos is None or length is None or angle is None):
            return

        # translate coordinates from axes or relative values
        if s.positioning == 'axes':
            if hasattr(self.parent, 'getAxes'):
                axes = self.parent.getAxes( (s.xAxis, s.yAxis) )
            else:
                return
            if None in axes:
                return
            xpos = axes[0].graphToPlotterCoords(posn, xpos)
            ypos = axes[1].graphToPlotterCoords(posn, ypos)
        else:
            xpos = posn[0] + (posn[2]-posn[0])*xpos
            ypos = posn[3] - (posn[3]-posn[1])*ypos

        # if a dataset is used, we can't use control items
        isnotdataset = ( not s.get('xPos').isDataset(d) and 
                         not s.get('yPos').isDataset(d) and
                         not s.get('length').isDataset(d) and
                         not s.get('angle').isDataset(d) )
        self.controlgraphitems = []

        arrowsize = s.get('arrowSize').convert(painter)

        painter.beginPaintingWidget(self, posn)
        painter.save()

        # drawing settings for line
        if not s.Line.hide:
            painter.setPen( s.get('Line').makeQPen(painter) )
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
                self.controlgraphitems.append(cgi)

        painter.restore()
        painter.endPaintingWidget()

    def updateControlItem(self, cgi, pt1, pt2):
        """If control items are moved, update line."""
        s = self.settings

        # calculate new position coordinate for item
        if s.positioning == 'axes':
            if hasattr(self.parent, 'getAxes'):
                axes = self.parent.getAxes( (s.xAxis, s.yAxis) )
            else:
                return
            if None in axes:
                return
            
            xpos = axes[0].plotterToGraphCoords(cgi.widgetposn,
                                                N.array(pt1[0]))
            ypos = axes[1].plotterToGraphCoords(cgi.widgetposn,
                                                N.array(pt1[1]))
        else:
            xpos = ((pt1[0] - cgi.widgetposn[0]) /
                    (cgi.widgetposn[2]-cgi.widgetposn[0]))
            ypos = ((pt1[1] - cgi.widgetposn[3]) /
                    (cgi.widgetposn[1]-cgi.widgetposn[3]))

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
