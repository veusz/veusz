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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
###############################################################################

# $Id$

"""A generic plotter widget which is inherited by function and point."""

import veusz.qtall as qt4
import itertools
import numpy as N

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import widget
import graph

class GenericPlotter(widget.Widget):
    """Generic plotter."""

    typename='genericplotter'
    allowedparenttypes=[graph.Graph]
    isplotter = True

    def __init__(self, parent, name=None):
        """Initialise object, setting axes."""
        widget.Widget.__init__(self, parent, name=name)

        s = self.settings
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
    
    def updateAxisRange(self, depname, range):
        """Update range variable for axis with dependency name given."""
        pass

    def drawKeySymbol(self, painter, x, y, width, height):
        """Draw the plot symbol and/or line at (x,y) in a box width*height.

        This is used to plot a key
        """
        pass

    def clipAxesBounds(self, painter, axes, bounds):
        """Clip painter to start and stop values of axis."""

        # update cached coordinates of axes
        axes[0].plotterToGraphCoords(bounds, N.array([]))
        axes[1].plotterToGraphCoords(bounds, N.array([]))

        # get range
        x1 = axes[0].coordParr1
        x2 = axes[0].coordParr2
        y1 = axes[1].coordParr1
        y2 = axes[1].coordParr2

        # actually clip the data
        painter.setClipRect( qt4.QRectF(x1, y2, x2-x1, y1-y2) )
