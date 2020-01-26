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

from __future__ import division
from .. import qtall as qt
import numpy as N

from .. import setting

from . import widget

def _(text, disambiguation=None, context='Plotters'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class GenericPlotter(widget.Widget):
    """Generic plotter."""

    typename='genericplotter'
    isplotter = True

    @classmethod
    def allowedParentTypes(klass):
        from . import graph
        return (graph.Graph,)

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        s.add( setting.Str('key', '',
                           descr = _('Description of the plotted data to appear in key'),
                           usertext=_('Key text')) )
        s.add( setting.Axis('xAxis', 'x', 'horizontal',
                            descr = _('Name of X-axis to use'),
                            usertext=_('X axis')) )
        s.add( setting.Axis('yAxis', 'y', 'vertical',
                            descr = _('Name of Y-axis to use'),
                            usertext=_('Y axis')) )

    def autoColor(self, painter, dataindex=0):
        """Automatic color for plotting."""
        return painter.docColorAuto(
            painter.helper.autoColorIndex((self, dataindex)))

    def getAxesNames(self):
        """Returns names of axes used."""
        s = self.settings
        return (s.xAxis, s.yAxis)

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

        # get range
        x1, x2 = axes[0].coordParr1, axes[0].coordParr2
        if x1 > x2:
            x1, x2 = x2, x1
        y1, y2 = axes[1].coordParr2, axes[1].coordParr1
        if y1 > y2:
            y1, y2 = y2, y1

        # actually clip the data
        cliprect = qt.QRectF(qt.QPointF(x1, y1), qt.QPointF(x2, y2))
        return cliprect

    def getAxisLabels(self, direction):
        """Get labels for datapoints and coordinates, or None if none.
        direction is 'horizontal' or 'vertical'

        return (labels, coordinates)
        """
        return (None, None)

    def fetchAxes(self):
        """Returns the axes for this widget"""

        axes = self.parent.getAxes( (self.settings.xAxis,
                                     self.settings.yAxis) )

        # fail if we don't have good axes
        if ( axes[0] is None or axes[1] is None or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return None

        return axes

    def lookupAxis(self, axisname):
        """Find widget associated with axisname."""
        w = self.parent
        while w:
            for c in w.children:
                if c.name == axisname and c.isaxis:
                    return c
            w = w.parent
        return None

    def affectsAxisRange(self):
        """Returns information on the following axes.
        format is ( ('x', 'sx'), ('y', 'sy') )
        where key is the axis and value is a provided bound
        """
        return ()

    def requiresAxisRange(self):
        """Requires information about the axis given before providing
        information.
        Format (('sx', 'x'), ('sy', 'y'))
        """
        return ()

    def getRange(self, axis, depname, therange):
        """Update range variable for axis with dependency name given."""
        pass

    def draw(self, parentposn, painthelper, outerbounds = None):
        """Draw for generic plotters."""

        posn = self.computeBounds(parentposn, painthelper)

        # exit if hidden or function blank
        if self.settings.hide:
            return

        # get axes widgets
        axes = self.fetchAxes()
        if not axes:
            return

        # clip data within bounds of plotter
        cliprect = self.clipAxesBounds(axes, posn)
        painter = painthelper.painter(self, posn, clip=cliprect)
        with painter:
            self.dataDraw(painter, axes, posn, cliprect)

        for c in self.children:
            c.draw(posn, painthelper, outerbounds)

        return posn

    def dataDraw(self, painter, axes, posn, cliprect):
        """Actually plot the data."""
        pass

class FreePlotter(widget.Widget):
    """A plotter which can be plotted on the page or in a graph."""

    def __init__(self, parent, name=None):
        """Initialise object, setting axes."""
        widget.Widget.__init__(self, parent, name=name)

    @classmethod
    def allowedParentTypes(klass):
        from . import page, graph
        return (graph.Graph, page.Page)

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        s.add( setting.DatasetExtended(
                'xPos', [0.5],
                descr=_('List of fractional X coordinates or dataset'),
                usertext=_('X positions'),
                formatting=False) )
        s.add( setting.DatasetExtended(
                'yPos', [0.5],
                descr=_('List of fractional Y coordinates or dataset'),
                usertext=_('Y positions'),
                formatting=False) )
        s.add( setting.Choice('positioning',
                              ['axes', 'relative'], 'relative',
                              descr=_('Use axes or fractional '
                                      'position to place label'),
                              usertext=_('Position mode'),
                              formatting=False) )
        s.add( setting.Axis('xAxis', 'x', 'horizontal',
                            descr = _('Name of X-axis to use'),
                            usertext=_('X axis')) )
        s.add( setting.Axis('yAxis', 'y', 'vertical',
                            descr = _('Name of Y-axis to use'),
                            usertext=_('Y axis')) )

    def _getPlotterCoords(self, posn, xsetting='xPos', ysetting='yPos'):
        """Calculate coordinates from relative or axis positioning.

        xsetting and ysetting are the settings to get data from
        """

        s = self.settings
        xpos = s.get(xsetting).getFloatArray(self.document)
        ypos = s.get(ysetting).getFloatArray(self.document)
        if xpos is None or ypos is None:
            return None, None
        if s.positioning == 'axes':

            if hasattr(self.parent, 'getAxes'):
                axes = self.parent.getAxes( (s.xAxis,
                                             s.yAxis) )
            else:
                return None, None
            if axes[0] is None or axes[1] is None:
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
        xplt = N.array(xplt)
        yplt = N.array(yplt)
        if s.positioning == 'axes':
            if hasattr(self.parent, 'getAxes'):
                axes = self.parent.getAxes( (s.xAxis, s.yAxis) )
            else:
                return None, None
            if axes[0] is None or axes[1] is None:
                return None, None

            xpos = axes[0].plotterToDataCoords(posn, xplt)
            ypos = axes[1].plotterToDataCoords(posn, yplt)
        else:
            xpos = (xplt - posn[0]) / (posn[2]-posn[0])
            ypos = (yplt - posn[3]) / (posn[1]-posn[3])
        return xpos, ypos
