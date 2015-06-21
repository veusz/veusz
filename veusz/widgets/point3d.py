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
import numpy as N

from .. import qtall as qt4
from .. import setting
from .. import document
from .. import utils

from . import plotters3d

try:
    from ..helpers import threed
except ImportError:
    threed = None

def _(text, disambiguation=None, context='Point3D'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class PlotLine3D(setting.Line3D):
    """Plot line."""
    def __init__(self, name, **args):
        setting.Line3D.__init__(self, name, **args)
        self.get('hide').newDefault(True)

class Point3D(plotters3d.GenericPlotter3D):
    """Plotting points in 3D."""

    typename='point3d'
    description=_('3D points')

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
        s.add( setting.DataColor('Color') )

        s.add( PlotLine3D(
            'PlotLine',
            descr = _('Plot line settings'),
            usertext = _('Plot line')),
               pixmap = 'settings_plotline' )
        s.add( setting.Surface3DWColorMap(
            'MarkerFill',
            descr = _('Marker fill settings'),
            usertext=_('Marker fill')),
              pixmap='settings_plotmarkerfill' )
        s.add( setting.Line3D(
            'MarkerLine',
            descr = _('Line around marker settings'),
            usertext = _('Marker border')),
               pixmap = 'settings_plotmarkerline' )
        s.add( setting.Line3D(
            'Error',
            descr = _('Error bar settings'),
            usertext = _('Error bar')),
               pixmap = 'settings_ploterrorline' )

    def affectsAxisRange(self):
        """Which axes this widget affects."""
        s = self.settings
        return ((s.xAxis, 'sx'), (s.yAxis, 'sy'), (s.zAxis, 'sz'))

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

    def dataDrawPointsLine(self, cont, coord):
        """Draw point and plot line."""

        s = self.settings
        doc = self.document
        scalepts = s.get('scalePoints').getData(doc)

        pointpath, filled = utils.getPointPainterPath(
            s.marker, s.markerSize, s.MarkerLine.width)

        markerlineprop = markerfillprop = None
        if not s.MarkerLine.hide:
            markerlineprop = s.MarkerLine.makeLineProp()
        if filled and not s.MarkerFill.hide:
            markerfillprop = s.MarkerFill.makeSurfaceProp()
            cvals = s.Color.get('data').getData(doc)
            if cvals is not None:
                colorvals = utils.applyScaling(
                    cvals.data, s.Color.scaling, s.Color.min, s.Color.max)
                cmap = self.document.getColormap(
                    s.MarkerFill.colorMap, s.MarkerFill.colorMapInvert)
                color2d = colorvals.reshape(1, len(colorvals))
                colorimg = utils.applyColorMap(
                    cmap, 'linear', color2d, 0., 1., s.MarkerFill.transparency)
                markerfillprop.setRGBs(colorimg)

        if markerlineprop or markerfillprop:
            ptobj = threed.Points(coord[0], coord[1], coord[2], pointpath,
                                  markerlineprop, markerfillprop)
            if scalepts:
                ptobj.setSizes(threed.ValVector(scalepts.data))
            cont.addObject(ptobj)

        if not s.PlotLine.hide:
            lineobj = threed.PolyLine(s.PlotLine.makeLineProp())
            lineobj.addPoints(xcoord, ycoord, zcoord)
            cont.addObject(lineobj)

    def dataDrawErrorBars(self, cont, axes, coord, datasets):
        """Draw error bars for points."""

        # TODO: Different error styles

        err = self.settings.Error
        if err.hide:
            return

        prop = err.makeLineProp()

        for i in range(3):
            ds = datasets[i]
            if not ds.hasErrors():
                continue

            neg = pos = None
            if ds.nerr is not None:
                neg = ds.data+ds.nerr
            elif ds.serr is not None:
                neg = ds.data-ds.serr

            if ds.perr is not None:
                pos = ds.data+ds.perr
            elif ds.serr is not None:
                pos = ds.data+ds.serr

            coordend = list(coord)
            if neg is not None:
                coordend[i] = threed.ValVector(axes[i].dataToLogicalCoords(neg))
                line = threed.LineSegments(coord[0], coord[1], coord[2],
                                           coordend[0], coordend[1], coordend[2],
                                           prop)
                cont.addObject(line)
            if pos is not None:
                coordend[i] = threed.ValVector(axes[i].dataToLogicalCoords(pos))
                line = threed.LineSegments(coord[0], coord[1], coord[2],
                                           coordend[0], coordend[1], coordend[2],
                                           prop)
                cont.addObject(line)

    def dataDrawToObject(self, axes):
        """Do drawing of axis."""

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

        coord = [
            threed.ValVector(axes[0].dataToLogicalCoords(xv.data)),
            threed.ValVector(axes[1].dataToLogicalCoords(yv.data)),
            threed.ValVector(axes[2].dataToLogicalCoords(zv.data))
        ]

        clipcont = self.makeClipContainer(axes)
        self.dataDrawPointsLine(clipcont, coord)
        self.dataDrawErrorBars(clipcont, axes, coord, [xv, yv, zv])

        return clipcont

document.thefactory.register(Point3D)
