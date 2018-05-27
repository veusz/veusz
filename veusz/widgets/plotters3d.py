#    Copyright (C) 2014 Jeremy S. Sanders
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

"""Generic 3D plotting widet."""

from __future__ import division, print_function
from .. import qtall as qt
import numpy as N

from .. import setting
from . import widget
from ..helpers import threed

def _(text, disambiguation=None, context='Plotters3D'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class GenericPlotter3D(widget.Widget):
    """Generic plotter."""

    typename = 'genericplotter3d'
    isplotter = True

    @classmethod
    def allowedParentTypes(klass):
        from . import graph3d
        return (graph3d.Graph3D,)

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        # s.add( setting.Str(
        #     'key', '',
        #     descr = _('Description of the plotted data to appear in key'),
        #     usertext=_('Key text')) )
        s.add( setting.Axis(
            'xAxis', 'x', 'x',
            descr = _('Name of X-axis to use'),
            usertext=_('X axis')) )
        s.add( setting.Axis(
            'yAxis', 'y', 'y',
            descr = _('Name of Y-axis to use'),
            usertext=_('Y axis')) )
        s.add( setting.Axis(
            'zAxis', 'z', 'z',
            descr = _('Name of Z-axis to use'),
            usertext=_('Z axis')) )

    def autoColor(self, painter, dataindex=0):
        """Automatic color for plotting."""
        return painter.docColorAuto(
            painter.helper.autoColorIndex((self, dataindex)))

    def getAxesNames(self):
        """Returns names of axes used."""
        s = self.settings
        return (s.xAxis, s.yAxis, s.zAxis)

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

    def getAxisLabels(self, direction):
        """Get labels for datapoints and coordinates, or None if none.
        direction is 'horizontal' or 'vertical'

        return (labels, coordinates)
        """
        return (None, None, None)

    def fetchAxes(self):
        """Returns the axes for this widget"""

        axes = self.parent.getAxes(
            (self.settings.xAxis, self.settings.yAxis, self.settings.zAxis))

        # fail if we don't have good axes
        if ( axes[0] is None or axes[0].settings.direction != 'x' or
             axes[1] is None or axes[1].settings.direction != 'y' or
             axes[2] is None or axes[2].settings.direction != 'z' ):
            return None

        return axes

    def fetchAxis(self, var):
        """Return a particular axis given x, y or z"""
        if var == 'x':
            setn = self.settings.xAxis
        elif var == 'y':
            setn = self.settings.yAxis
        elif var == 'z':
            setn = self.settings.zAxis
        return self.parent.getAxes((setn,))[0]

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

    def simplifyObjectList(self, objlist):
        """Simplify list of objects, if possible."""
        if len(objlist) == 0:
            return None
        elif len(objlist) == 1:
            return objlist[0]
        else:
            cont = threed.ObjectContainer()
            for o in objlist:
                cont.addObject(o)
            return cont

    def makeClipContainer(self, axes):
        """Make an object container to clip data to axes."""
        return threed.ClipContainer(
            threed.Vec3(axes[0].settings.lowerPosition,
                        axes[1].settings.lowerPosition,
                        axes[2].settings.lowerPosition),
            threed.Vec3(axes[0].settings.upperPosition,
                        axes[1].settings.upperPosition,
                        axes[2].settings.upperPosition))

    def drawToObject(self, painter, painthelper):
        # exit if hidden or function blank
        if self.settings.hide:
            return

        # get axes widgets
        axes = self.fetchAxes()
        if not axes:
            return
        else:
            return self.dataDrawToObject(painter, axes)

    def dataDrawToObject(self, painter, axes):
        """Actually plot the data."""
        pass
