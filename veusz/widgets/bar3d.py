#    Copyright (C) 2018 Jeremy S. Sanders
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

"""3D bar plotting widget."""

from __future__ import division, print_function
import numpy as N

from .. import qtall as qt
from .. import setting
from .. import document
from .. import utils
from ..helpers import threed

from . import plotters3d
from .volume3d import autoedges

def _(text, disambiguation=None, context='Bar3D'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

def autoedges(data, fillfactor):
    # make sure boxes do not overlap
    fillfactor = min(0.999, fillfactor)

    unique, idxs = N.unique(data, return_inverse=True)
    if len(unique) < 2:
        return None, None
    deltas = unique[1:]-unique[:-1]

    minvals = N.hstack((
        [unique[0]-0.5*fillfactor*deltas[0]],
        unique[1:]-0.5*fillfactor*deltas))
    maxvals = N.hstack((
        unique[:-1]+0.5*fillfactor*deltas,
        [unique[-1]+0.5*fillfactor*deltas[-1]]))

    edges = N.column_stack((minvals,maxvals))

    return edges, idxs

class BarFill(setting.Settings):
    '''Fill colors and settings.'''

    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.ColorSet(
            'color', [['auto', False]],
            descr = _('Fill color'),
            usertext=_('Color')) )
        self.add( setting.FloatSlider(
            'reflectivity', 50.,
            minval=0., maxval=100., tick=20., scale=1., step=1.,
            descr=_('Reflectivity percentage'),
            usertext=_('Reflectivity')) )
        self.add( setting.FloatSlider(
            'transparency', 0.,
            minval=0., maxval=100., tick=20., scale=1., step=1.,
            descr=_('Transparency percentage'),
            usertext=_('Transparency')) )
        self.add(setting.Colormap(
            'colorMap',
            'grey',
            descr = _('Set of colors to plot data with'),
            usertext=_('Colormap'),
            formatting=True),
            0)
        self.add(setting.Bool(
            'colorMapInvert', False,
            descr = _('Invert color map'),
            usertext=_('Invert colormap'),
            formatting=True),
            1)
        self.add( setting.Bool(
            'hide', False,
            descr = _('Hide the fill'),
            usertext=_('Hide')) )

class Bar3D(plotters3d.GenericPlotter3D):
    """Plotting bars in 3D."""

    typename='bar3d'
    description=_('3D bars')
    allowusercreation=True

    # list of modes allowed
    modes = (
        'x(y,z)', 'x(z,y)',
        'y(x,z)', 'y(z,x)',
        'z(x,y)', 'z(y,x)',
    )

    # for a mode, find which of the three positions is x,y,z (0,1,2)
    mode_to_axis = {
        'x(y,z)': (0,1,2), 'x(z,y)': (0,2,1),
        'y(x,z)': (1,0,2), 'y(z,x)': (1,0,2),
        'z(x,y)': (1,2,0), 'z(y,x)': (2,1,0),
    }

    def __init__(self, parent, name=None):
        """Initialise plotter with axes."""
        plotters3d.GenericPlotter3D.__init__(self, parent, name=name)
        self._lastchangeset = -1
        self._validplot = False
        self._lengths = None

    @classmethod
    def addSettings(klass, s):
        plotters3d.GenericPlotter3D.addSettings(s)

        s.add(setting.Choice(
            'mode', klass.modes,
            'y(x,z)',
            descr=_('Axes for height and position'),
            usertext=_('Mode')),
              0)

        s.add( setting.Datasets(
            'lengths', ['lengths'],
            descr=_('Lengths (1D or 2D datasets)'),
            usertext=_('Lengths'),
            dimensions='all'), 1 )
        s.add( setting.DatasetExtended(
            'pos1', '',
            descr=_('Position 1 (if lengths not 2D datasets)'),
            usertext=_('Position 1')), 2 )
        s.add( setting.DatasetExtended(
            'pos2', '',
            descr=_('Position 2 (if lengths not 2D datasets)'),
            usertext=_('Position 2')), 3 )
        s.add(setting.DataColor(
            'DataColor', dimensions='all'),
              4)

        s.add(setting.Float(
            'fillfactor', 0.8,
            minval=0, maxval=1,
            descr=_('Filling factor (0-1)'),
            usertext=_('Fill factor'),
            formatting=True),
            5)

        s.add(BarFill(
            'Fill',
            descr = _('Fill settings'),
            usertext = _('Fill')),
              pixmap = 'settings_bgfill' )

        s.add(setting.Line3D(
            'Line',
            descr = _('Line settings'),
            usertext = _('Box line')),
               pixmap = 'settings_plotline' )

    def affectsAxisRange(self):
        """Which axes this widget affects."""
        s = self.settings
        return ((s.xAxis, 'sx'), (s.yAxis, 'sy'), (s.zAxis, 'sz'))

    def update(self):
        """Update plot from current datasets.
        Returns whether a valid plot
        """

        if self.document.changeset == self._lastchangeset:
            return self._validplot
        self._lastchangeset = self.document.changeset
        self._validplot = False

        # get datasets
        dsets = self.settings.get('lengths').getData(self.document)
        if not dsets:
            return False

        # check shapes are consistent
        shape = dsets[0].data.shape
        if any((d.data.shape != shape for d in dsets[1:])):
            return False
        # check dimensions ok
        if len(shape) not in (1,2):
            return False

        if not self.updatePositions(dsets):
            return False
        if not self.updateLengths(dsets):
            return False
        if not self.updateColors(dsets):
            return False

        self._validplot = True
        return True

    def updatePositions(self, dsets):
        """Update bar positions for datasets."""

        s = self.settings
        fillfactor = min(0.999, s.fillfactor)

        shape = dsets[0].data.shape
        if len(shape) == 2:
            # ignore positions arrays if data are twod
            edges = dsets[0].getPixelEdges()

            # lookup edges for each value in 2D array
            idxs = N.indices(dsets[0].data.shape)
            self._edge1_lo = N.ravel(edges[0][idxs[1]])
            self._edge1_hi = N.ravel(edges[0][idxs[1]+1])
            self._edge2_lo = N.ravel(edges[1][idxs[0]])
            self._edge2_hi = N.ravel(edges[1][idxs[0]+1])

            # implement fill factor by shifting values (fails for log?)
            delta1 = (self._edge1_hi-self._edge1_lo)*(1-fillfactor)/2
            self._edge1_lo += delta1
            self._edge1_hi -= delta1
            delta2 = (self._edge2_hi-self._edge2_lo)*(1-fillfactor)/2
            self._edge2_lo += delta2
            self._edge2_hi -= delta2

        else:
            # get position datasets
            pos1 = s.get('pos1').getData(self.document)
            pos2 = s.get('pos2').getData(self.document)
            if pos1 is None or pos2 is None:
                return False
            if pos1.data.shape != shape or pos2.data.shape != shape:
                return False

            edges1, idxs1 = autoedges(pos1, fillfactor)
            edges2, idxs2 = autoedges(pos2, fillfactor)
            if edges1 is None or edges2 is None:
                return False
            self._edge1_lo = edges1[:,0][idxs1]
            self._edge1_hi = edges1[:,1][idxs1]
            self._edge2_lo = edges2[:,0][idxs2]
            self._edge2_hi = edges2[:,1][idxs2]

        return True

    def updateLengths(self, dsets):
        """Update lengths in bars for datasets given.
        """

        # current extreme bar sizes
        shape = dsets[0].data.shape
        minbound = N.zeros(shape)
        maxbound = N.zeros(shape)
        # output bar stop and stops
        barmins = []
        barmaxs = []

        for ds in dsets:
            data = ds.data
            neg = data<0

            # work out ends of bars for this dataset
            barmin = N.where(neg, minbound, maxbound)
            barmax = barmin+data
            barmins.append(N.ravel(barmin))
            barmaxs.append(N.ravel(barmax))

            # extend total bounds where finite
            finite = N.isfinite(data)
            finneg = neg & finite
            minbound[finneg] = barmax[finneg]
            finpos = (~neg) & finite
            maxbound[finpos] = barmax[finpos]

        self._minbound = minbound
        self._maxbound = maxbound
        self._barmins = barmins
        self._barmaxs = barmaxs
        return True

    def updateColors(self, dsets):
        """Update colors (if set)."""

        s = self.settings
        col = s.DataColor.get('points').getData(self.document)
        print('col', col)
        if col is None:
            self._colorimg = None
            return True
        if col.data.shape != dsets[0].data.shape:
            return False

        cdata = col.data
        cdata = cdata.reshape((1, cdata.size))
        cmap = self.document.evaluate.getColormap(
            s.Fill.colorMap, s.Fill.colorMapInvert)
        self._colorimg = utils.applyColorMap(
            cmap, s.DataColor.scaling,
            cdata,
            s.DataColor.min, s.DataColor.max,
            s.Fill.transparency)
        return True

    def getRange(self, axis, depname, axrange):
        """Update axis range from data."""

        if not self.update():
            return

        axidx = {'sx': 0, 'sy': 1, 'sz': 2}[depname]
        #idx = self.mode_idxs[self.settings.mode].index(axidx)
        idx = self.mode_to_axis[self.settings.mode][axidx]

        if idx == 0:
            # length
            v = N.concatenate((self._minbound, self._maxbound))
        elif idx == 1:
            # axis 1
            v = N.concatenate((self._edge1_lo, self._edge1_hi))
        else:
            # axis 2
            v = N.concatenate((self._edge2_lo, self._edge2_hi))

        if axis.settings.log:
            with N.errstate(invalid='ignore'):
                chopzero = v[(v>0) & N.isfinite(v)]
            if len(chopzero) > 0:
                axrange[0] = min(axrange[0], chopzero.min())
                axrange[1] = max(axrange[1], chopzero.max())
        else:
            fvals = v[N.isfinite(v)]
            if len(fvals) > 0:
                axrange[0] = min(axrange[0], fvals.min())
                axrange[1] = max(axrange[1], fvals.max())

    def dataDrawToObject(self, painter, axes):
        """Do drawing of axis."""

        s = self.settings

        axes = self.fetchAxes()
        if axes is None:
            return

        if not self.update():
            return

        surfprop = threed.SurfaceProp(refl=s.Fill.reflectivity*0.01)
        if self._colorimg is not None:
            surfprop.setRGBs(self._colorimg)

        lineprop = s.Line.makeLineProp(painter)

        modeaxes = self.mode_to_axis[self.settings.mode]
        clipcont = self.makeClipContainer(axes)
        for bmin, bmax in zip(self._barmins, self._barmaxs):

            # these are the height, pos1 and pos2 axis coords
            edges = (
                (bmin, bmax),
                (self._edge1_lo, self._edge1_hi),
                (self._edge2_lo, self._edge2_hi),
            )

            # get coordinates for each axis
            xmin = edges[modeaxes[0]][0]
            xmax = edges[modeaxes[0]][1]
            ymin = edges[modeaxes[1]][0]
            ymax = edges[modeaxes[1]][1]
            zmin = edges[modeaxes[2]][0]
            zmax = edges[modeaxes[2]][1]

            # convert from axis coordinates to fractional coords
            xminl = threed.ValVector(axes[0].dataToLogicalCoords(xmin))
            yminl = threed.ValVector(axes[1].dataToLogicalCoords(ymin))
            zminl = threed.ValVector(axes[2].dataToLogicalCoords(zmin))
            xmaxl = threed.ValVector(axes[0].dataToLogicalCoords(xmax))
            ymaxl = threed.ValVector(axes[1].dataToLogicalCoords(ymax))
            zmaxl = threed.ValVector(axes[2].dataToLogicalCoords(zmax))

            # make the cubes
            cuboids = threed.MultiCuboid(
                xminl, xmaxl, yminl, ymaxl, zminl, zmaxl, lineprop, surfprop)
            clipcont.addObject(cuboids)

        clipcont.assignWidgetId(id(self))
        return clipcont

document.thefactory.register(Bar3D)
