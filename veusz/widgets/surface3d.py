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

from .. import qtall as qt4
from .. import setting
from .. import document
from .. import utils
try:
    from ..helpers import threed
except ImportError:
    threed = None

from . import plotters3d

def _(text, disambiguation=None, context='Surface3D'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class Surface3D(plotters3d.GenericPlotter3D):
    """Plotting surface in 3D."""

    typename = 'surface3d'
    description = _('3D surface')

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
            descr=_('Directions to plot surface'),
            usertext=_('Mode')),
              0)
        s.add(setting.DatasetExtended(
            'data', '',
            dimensions=2,
            descr=_('Dataset to plot'),
            usertext=_('Dataset')),
              1)
        s.add(setting.Bool(
            'highres', False,
            descr=_('High resolution surface (accurate bin centres)'),
            usertext=_('High res.'),
            formatting=True)
          )

        s.add(setting.Line3D(
            'Line',
            descr = _('Line settings'),
            usertext = _('Grid line')),
               pixmap = 'settings_plotline' )
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

    def drawSurface(self, container, axes, dataset):
        """Add the surface to the container."""

        s = self.settings
        if s.Surface.hide and s.Line.hide:
            return
        surfprop = s.Surface.makeSurfaceProp()
        lineprop = s.Line.makeLineProp()
        highres = s.highres

        idxs = self.mode_idxs[self.settings.mode]

        # convert to logical coordinates
        data = axes[idxs[0]].dataToLogicalCoords(dataset.data)
        e = dataset.getPixelEdges()
        edges1 = axes[idxs[1]].dataToLogicalCoords(e[0])
        edges2 = axes[idxs[2]].dataToLogicalCoords(e[1])

        mesh = threed.DataMesh(
            threed.ValVector(edges1),
            threed.ValVector(edges2),
            threed.ValVector(N.ravel(data)),
            idxs[0], idxs[1], idxs[2],
            highres,
            lineprop, surfprop)
        container.addObject(mesh)
        return




        # make 2d arrays of coordinates for each direction
        e1start = edges1[:-1]
        e1end = edges1[1:]
        e2start = edges2[:-1]
        e2end = edges2[1:]

        def maketris(inpts, triplets):
            # swap round coordinates to match axes
            pts = []
            pt = [None,None,None]
            for p in inpts:
                for i, j in enumerate(idxs):
                    pt[j] = p[i]
                pts.append(threed.Vec3(*pt))

            for i1, i2, i3 in triplets:
                container.addObject(threed.Triangle(pts[i1], pts[i2], pts[i3], surfprop))

        def makelines(inpts, ptidxs):
            pts = []
            pt = [None,None,None]
            for p in inpts:
                for i, j in enumerate(idxs):
                    pt[j] = p[i]
                pts.append(threed.Vec3(*pt))

            for i1, i2 in ptidxs:
                l = threed.PolyLine(lineprop)
                l.addPoint(pts[i1])
                l.addPoint(pts[i2])
                container.addObject(l)

        size1, size2 = len(edges1)-1, len(edges2)-1
        for i1 in xrange(size1):
            for i2 in xrange(size2):
                # get coordinates of centre and diagonal neighbouring points
                coord0 = data[i2, i1]
                c = []

                # -1,-1 -1,0 -1,1   0,-1 0,0 0,1   1,-1 1,0 1,1
                for d1 in -1, 0, 1:
                    clip1 = max(min(i1+d1, size1-1), 0)
                    for d2 in -1, 0, 1:
                        clip2 = max(min(i2+d2, size2-1), 0)
                        c.append(data[clip2, clip1])

                corners = [
                    (0.25*(c[0]+c[3]+c[4]+c[1]), e1start[i1], e2start[i2]),
                    (0.5*(c[4]+c[3]), 0.5*(e1start[i1]+e1end[i1]), e2start[i2]),
                    (0.25*(c[3]+c[6]+c[7]+c[4]), e1end[i1], e2start[i2]),
                    (0.5*(c[4]+c[7]), e1end[i1], 0.5*(e2start[i2]+e2end[i2])), 
                    (0.25*(c[4]+c[7]+c[8]+c[5]), e1end[i1], e2end[i2]),
                    (0.5*(c[4]+c[5]), 0.5*(e1start[i1]+e1end[i1]), e2end[i2]),
                    (0.25*(c[1]+c[4]+c[5]+c[2]), e1start[i1], e2end[i2]),
                    (0.5*(c[4]+c[1]), e1start[i1], 0.5*(e2start[i2]+e2end[i2])),
                    (c[4], 0.5*(e1start[i1]+e1end[i1]), 0.5*(e2start[i2]+e2end[i2]))
                    ]

                if highres:
                    maketris(corners, ((8,0,1),(8,1,2),(8,2,3),(8,3,4),(8,4,5),
                                       (8,5,6),(8,6,7),(8,7,0)))
                    makelines(corners, ((0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0)))
                else:
                    maketris(corners, ((0,2,4), (0,6,4)))
                    makelines(corners, ((0,2), (0,6), (4,2), (4,6)))

    def dataDrawToObject(self, axes):
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
        self.drawSurface(clipcontainer, axes, data)

        return clipcontainer

document.thefactory.register(Surface3D)
