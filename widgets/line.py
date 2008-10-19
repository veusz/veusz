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

        self.lastposn = posn

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
        del self.controlgraphitems[:]

        painter.beginPaintingWidget(self, posn)
        painter.save()

        # drawing settings for line
        if not s.Line.hide:
            painter.setPen( s.get('Line').makeQPen(painter) )
        else:
            painter.setPen( qt4.QPen(qt4.Qt.NoPen) )

        # iterate over positions
        index = 0
        dx, dy = posn[2]-posn[0], posn[3]-posn[1]
        for x, y, l, a in itertools.izip(xpos, ypos,
                                         itertools.cycle(length),
                                         itertools.cycle(angle)):
            painter.save()
            painter.translate(x, y)
            painter.rotate(a)

            painter.drawLine( qt4.QPointF(0., 0.),
                              qt4.QPointF(l*dx, 0.) )
            painter.restore()

        painter.restore()

document.thefactory.register( Line )
