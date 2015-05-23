#    Copyright (C) 2015 Jeremy S. Sanders
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

"""3D point plotting widget."""

from __future__ import division, print_function
import itertools
import numpy as N

from ..compat import czip, crange
from .. import qtall as qt4
from .. import setting
from .. import document
from .. import utils
from ..helpers import threed

from . import plotters3d

def _(text, disambiguation=None, context='Point3D'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class Point3D(plotters3d.GenericPlotter3D):
    """Plotting points in 3D."""

    typename='point3d'
    description=_('Plot 3D points')

    @classmethod
    def addSettings(klass, s):
        plotters3d.GenericPlotter3D.addSettings(s)

        s.add( setting.DatasetExtended(
                'zData', 'z',
                descr=_('Z values, given by dataset, expression or list of values'),
                usertext=_('Z data')), 0 )
        s.add( setting.DatasetExtended(
                'yData', 'y',
                descr=_('Y values, given by dataset, expression or list of values'),
                usertext=_('Y data')), 0 )
        s.add( setting.DatasetExtended(
                'xData', 'x',
                descr=_('X values, given by dataset, expression or list of values'),
                usertext=_('X data')), 0 )
        s.add( setting.DatasetOrStr(
            'labels', '',
            descr=_('Dataset or string to label points'),
            usertext=_('Labels')), 5 )
        s.add( setting.DatasetExtended(
            'scalePoints', '',
            descr = _('Scale size of markers given by dataset, expression'
                      ' or list of values'),
            usertext=_('Scale markers')), 6 )

        s.add( setting.Float(
            'markerSize', 10,
            minval=0, maxval=1000,
            descr=_('Size of markers (relative to plot)'),
            usertext=_('Size'),
            formatting=True), 0 )
        s.add( setting.Marker(
            'marker', 'circle',
            descr = _('Type of marker to plot'),
            usertext=_('Marker'), formatting=True), 0 )
        s.add( setting.MarkerColor('Color') )

        s.add( setting.Line3D(
            'MarkerLine',
            descr = _('Line around marker settings'),
            usertext = _('Marker border')),
               pixmap = 'settings_plotmarkerline' )
        s.add( setting.Surface3D(
            'MarkerFill',
            descr = _('Marker fill settings'),
            usertext=_('Marker fill')),
              pixmap='settings_plotmarkerfill' )

    def affectsAxisRange(self):
        """Which axes this widget affects."""
        s = self.settings
        return ((s.xAxis, 'sx'), (s.yAxis, 'sy'), (s.zAxis, 'sz'))

    def requiresAxisRange(self):
        """Which axes this widget depends on."""
        return []

    def getRange(self, axis, depname, axrange):
        """Update axis range from data."""

        dataname = {'sx': 'xData', 'sy': 'yData', 'sz': 'zData'}[depname]
        dsetn = self.settings.get(dataname)
        data = dsetn.getData(self.document)

        if axis.settings.log:
            def updateRange(v):
                with N.errstate(invalid='ignore'):
                    chopzero = v[(v>0) & N.isfinite(v)]
                if len(chopzero) > 0:
                    axrange[0] = min(axrange[0], chopzero.min())
                    axrange[1] = max(axrange[1], chopzero.max())
        else:
            def updateRange(v):
                fvals = v[N.isfinite(v)]
                if len(fvals) > 0:
                    axrange[0] = min(axrange[0], fvals.min())
                    axrange[1] = max(axrange[1], fvals.max())

        if data:
            data.rangeVisit(updateRange)

    def dataDrawToObject(self, axes):

        s = self.settings

        axes = self.fetchAxes()
        if axes is None:
            return

        doc = self.document
        xv = s.get('xData').getData(doc)
        yv = s.get('yData').getData(doc)
        zv = s.get('zData').getData(doc)
        if not xv or not yv or not zv:
            return

        scalepoints = s.get('scalePoints').getData(doc)

        xcoord = threed.ValVector(axes[0].dataToLogicalCoords(xv.data))
        ycoord = threed.ValVector(axes[1].dataToLogicalCoords(yv.data))
        zcoord = threed.ValVector(axes[2].dataToLogicalCoords(zv.data))

        outobj = []

        s = self.settings

        pointpath, filled = utils.getPointPainterPath(
            s.marker, s.markerSize, s.MarkerLine.width)

        markerlineprop = markerfillprop = None
        if not s.MarkerLine.hide:
            markerlineprop = s.MarkerLine.makeLineProp()
        if filled and not s.MarkerFill.hide:
            markerfillprop = s.MarkerFill.makeSurfaceProp()

        ptobj = threed.Points(xcoord, ycoord, zcoord, pointpath,
                              markerlineprop, markerfillprop)
        outobj.append(ptobj)

        return self.simplifyObjectList(outobj)

document.thefactory.register(Point3D)
