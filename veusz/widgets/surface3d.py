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

"""3D surface plotting widget."""

from __future__ import division, print_function
import numpy as N

from .. import qtall as qt
from .. import setting
from .. import document
from .. import utils
from ..helpers import threed

from . import plotters3d

def _(text, disambiguation=None, context='Surface3D'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class Surface3D(plotters3d.GenericPlotter3D):
    """Plotting surface in 3D."""

    typename = 'surface3d'
    description = _('3D surface')
    allowusercreation=True

    # list of modes allowed
    modes = (
        'x(y,z)', 'x(z,y)',
        'y(x,z)', 'y(z,x)',
        'z(x,y)', 'z(y,x)',
    )
    # above, in numeric form
    mode_idxs = {
        'x(y,z)': (0,1,2), 'x(z,y)': (0,2,1),
        'y(x,z)': (1,0,2), 'y(z,x)': (1,2,0),
        'z(x,y)': (2,0,1), 'z(y,x)': (2,1,0),
    }

    @classmethod
    def addSettings(klass, s):
        plotters3d.GenericPlotter3D.addSettings(s)

        s.add(setting.Choice(
            'mode', klass.modes,
            'z(x,y)',
            descr=_('Axes of plot surface'),
            usertext=_('Mode')),
              0)
        s.add(setting.DatasetExtended(
            'data', '',
            dimensions=2,
            descr=_('Dataset to plot'),
            usertext=_('Dataset')),
              1)
        s.add(setting.DataColor(
            'DataColor', dimensions=2),
              2)

        s.add(setting.Bool(
            'highres', False,
            descr=_('High resolution surface (accurate bin centres)'),
            usertext=_('High res.'),
            formatting=True)
          )

        s.add(setting.LineGrid3D(
            'Line',
            descr = _('Grid line settings'),
            usertext = _('Grid line')),
               pixmap = 'settings_gridline' )
        s.add(setting.Surface3DWColorMap(
            'Surface',
            descr=_('Surface fill settings'),
            usertext=_('Surface')),
              pixmap='settings_bgfill')

    def affectsAxisRange(self):
        """Which axes this widget affects."""
        s = self.settings
        return ((s.xAxis, 'sx'), (s.yAxis, 'sy'), (s.zAxis, 'sz'))

    def getRange(self, axis, depname, axrange):
        """Update axis range from data."""

        s = self.settings
        # get real dataset
        data = s.get('data').getData(self.document)
        if data is None or data.dimensions != 2:
            return

        # convert axis dependency into an index (0,1,2) which
        # specifies whether to get value range, or 2d range
        axidx = {'sx': 0, 'sy': 1, 'sz': 2}[depname]
        idx = self.mode_idxs[s.mode].index(axidx)

        rng = None
        if idx == 0:
            # get range of values in 2D data
            data = N.ravel(data.data)
            findata = data[N.isfinite(data)]
            if len(findata) > 0:
                rng = findata.min(), findata.max()
        elif idx == 1:
            # range of data from 1st coordinate
            rng = data.getDataRanges()[0]
        elif idx == 2:
            # range of data from 2nd coordinate
            rng = data.getDataRanges()[1]
        if rng:
            axrange[0] = min(axrange[0], rng[0])
            axrange[1] = max(axrange[1], rng[1])

    def drawSurface(self, painter, container, axes, dataset):
        """Add the surface to the container."""

        s = self.settings
        if s.Surface.hide and s.Line.hide:
            return
        surfprop = s.Surface.makeSurfaceProp(painter)
        lineprop = s.Line.makeLineProp(painter)
        highres = s.highres

        # axes to plot coordinates on
        idxs = self.mode_idxs[self.settings.mode]

        # convert to logical coordinates
        data = axes[idxs[0]].dataToLogicalCoords(
            N.ravel(N.transpose(dataset.data)))
        e = dataset.getPixelEdges()
        edges1 = axes[idxs[1]].dataToLogicalCoords(e[0])
        edges2 = axes[idxs[2]].dataToLogicalCoords(e[1])

        # compute colors, if set
        colordata = s.DataColor.get('points').getData(self.document)
        if surfprop is not None and colordata is not None:
            cmap = self.document.evaluate.getColormap(
                s.Surface.colorMap, s.Surface.colorMapInvert)
            cdata = N.transpose(colordata.data)
            cdata = cdata.reshape((1, cdata.size))
            colorimg = utils.applyColorMap(
                cmap, s.DataColor.scaling,
                cdata,
                s.DataColor.min, s.DataColor.max,
                s.Surface.transparency)
            surfprop.setRGBs(colorimg)

        mesh = threed.DataMesh(
            threed.ValVector(edges1),
            threed.ValVector(edges2),
            threed.ValVector(data),
            idxs[0], idxs[1], idxs[2],
            highres,
            lineprop, surfprop,
            s.Line.hidehorz, s.Line.hidevert)
        container.addObject(mesh)

    def dataDrawToObject(self, painter, axes):
        """Do actual drawing of function."""

        s = self.settings
        mode = s.mode

        axes = self.fetchAxes()
        if axes is None:
            return

        s = self.settings
        data = s.get('data').getData(self.document)
        if s.hide or data is None or data.dimensions != 2:
            return

        clipcontainer = self.makeClipContainer(axes)
        self.drawSurface(painter, clipcontainer, axes, data)

        clipcontainer.assignWidgetId(id(self))
        return clipcontainer

document.thefactory.register(Surface3D)
