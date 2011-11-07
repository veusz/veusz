#    plotters.py
#    plotting classes

#    Copyright (C) 2004 Jeremy S. Sanders
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

"""A generic plotter widget which is inherited by function and point."""

import veusz.qtall as qt4
import numpy as N

import veusz.setting as setting

import widget
import graph
import page

class GenericPlotter(widget.Widget):
    """Generic plotter."""

    typename='genericplotter'
    allowedparenttypes=[graph.Graph]
    isplotter = True

    def __init__(self, parent, name=None):
        """Initialise object, setting axes."""
        widget.Widget.__init__(self, parent, name=name)

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        s.add( setting.Str('key', '',
                           descr = 'Description of the plotted data to appear in key',
                           usertext='Key text') )
        s.add( setting.Axis('xAxis', 'x', 'horizontal',
                            descr = 'Name of X-axis to use',
                            usertext='X axis') )
        s.add( setting.Axis('yAxis', 'y', 'vertical',
                            descr = 'Name of Y-axis to use',
                            usertext='Y axis') )

    def getAxesNames(self):
        """Returns names of axes used."""
        s = self.settings
        return (s.xAxis, s.yAxis)

    def lookupAxis(self, axisname):
        """Find widget associated with axisname."""
        w = self.parent
        while w:
            for c in w.children:
                if c.name == axisname and hasattr(c, 'isaxis'):
                    return c
            w = w.parent
        return None

    def providesAxesDependency(self):
        """Returns information on the following axes.
        format is ( ('x', 'sx'), ('y', 'sy') )
        where key is the axis and value is a provided bound
        """
        return ()

    def requiresAxesDependency(self):
        """Requires information about the axis given before providing
        information.
        Format (('sx': 'x'), ('sy': 'y'))
        """
        return ()
    
    def updateAxisRange(self, axis, depname, range):
        """Update range variable for axis with dependency name given."""
        pass

    def getNumberKeys(self):
        """Return number of key entries."""
        if self.settings.key:
            return 1
        else:
            return 0

    def getKeyText(self, number):
        """Get key entry."""
        return self.settings.key

    def drawKeySymbol(self, number, painter, x, y, width, height):
        """Draw the plot symbol and/or line at (x,y) in a box width*height.

        This is used to plot a key
        """
        pass

    def clipAxesBounds(self, axes, bounds):
        """Returns clipping rectange for start and stop values of axis."""

        # update cached coordinates of axes
        axes[0].plotterToDataCoords(bounds, N.array([]))
        axes[1].plotterToDataCoords(bounds, N.array([]))

        # get range
        x1, x2 = axes[0].coordParr1, axes[0].coordParr2
        if x1 > x2:
            x1, x2 = x2, x1
        y1, y2 = axes[1].coordParr2, axes[1].coordParr1
        if y1 > y2:
            y1, y2 = y2, y1

        # actually clip the data
        cliprect = qt4.QRectF(qt4.QPointF(x1, y1), qt4.QPointF(x2, y2))
        return cliprect

    def getAxisLabels(self, direction):
        """Get labels for datapoints and coordinates, or None if none.
        direction is 'horizontal' or 'vertical'

        return (labels, coordinates)
        """
        return (None, None)

class FreePlotter(widget.Widget):
    """A plotter which can be plotted on the page or in a graph."""

    allowedparenttypes = [graph.Graph, page.Page]
    def __init__(self, parent, name=None):
        """Initialise object, setting axes."""
        widget.Widget.__init__(self, parent, name=name)

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        s.add( setting.DatasetOrFloatList('xPos', [0.5],
                                          descr='List of fractional X '
                                          'coordinates or dataset',
                                          usertext='X positions',
                                          formatting=False) )
        s.add( setting.DatasetOrFloatList('yPos', [0.5],
                                          descr='List of fractional Y '
                                          'coordinates or dataset',
                                          usertext='Y positions',
                                          formatting=False) )
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

    def _getPlotterCoords(self, posn):
        """Calculate coordinates from relative or axis positioning."""

        s = self.settings
        xpos = s.get('xPos').getFloatArray(self.document)
        ypos = s.get('yPos').getFloatArray(self.document)
        if xpos is None or ypos is None:
            return None, None
        if s.positioning == 'axes':

            if hasattr(self.parent, 'getAxes'):
                axes = self.parent.getAxes( (s.xAxis,
                                             s.yAxis) )
            else:
                return None, None
            if None in axes:
                return None, None

            xpos = axes[0].dataToPlotterCoords(posn, xpos)
            ypos = axes[1].dataToPlotterCoords(posn, ypos)
        else:
            xpos = posn[0] + (posn[2]-posn[0])*xpos
            ypos = posn[3] - (posn[3]-posn[1])*ypos
        return xpos, ypos

    def _getGraphCoords(self, posn, xplt, yplt):
        """Calculate graph coodinates given plot coordinates xplt, yplt."""

        s = self.settings
        if s.positioning == 'axes':
            if hasattr(self.parent, 'getAxes'):
                axes = self.parent.getAxes( (s.xAxis, s.yAxis) )
            else:
                return None, None
            if None in axes:
                return None, None
            
            xpos = axes[0].plotterToDataCoords(posn, N.array(xplt))
            ypos = axes[1].plotterToDataCoords(posn, N.array(yplt))
        else:
            xpos = (xplt - posn[0]) / (posn[2]-posn[0])
            ypos = (yplt - posn[3]) / (posn[1]-posn[3])
        return xpos, ypos
