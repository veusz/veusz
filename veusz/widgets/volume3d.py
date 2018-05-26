#    Copyright (C) 2017 Jeremy S. Sanders
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

"""3D volume plotting widget."""

from __future__ import division, print_function
import numpy as N

from .. import qtall as qt
from .. import setting
from .. import document
from .. import utils
from ..helpers import threed

from . import plotters3d

def _(text, disambiguation=None, context='Volume3D'):
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

class DataColor(setting.DataColor):
    def __init__(self, *args, **argsv):
        setting.DataColor.__init__(self, *args, **argsv)
        self.get('points').newDefault('v')

class Volume3D(plotters3d.GenericPlotter3D):
    """Plotting points in 3D."""

    typename='volume3d'
    description=_('3D volume')
    allowusercreation=True

    @classmethod
    def addSettings(klass, s):
        plotters3d.GenericPlotter3D.addSettings(s)

        s.add( setting.DatasetExtended(
            'transData', '',
            descr=_('Transparency dataset, optional, 0-1'),
            usertext=_('Transparency')), 0 )
        s.add( setting.DatasetExtended(
            'zData', 'z',
            descr=_('Z dataset'),
            usertext=_('Z data')), 0 )
        s.add( setting.DatasetExtended(
            'yData', 'y',
            descr=_('Y dataset'),
            usertext=_('Y data')), 0 )
        s.add( setting.DatasetExtended(
            'xData', 'x',
            descr=_('X dataset'),
            usertext=_('X data')), 0 )
        s.add(DataColor(
            'DataColor', dimensions=1),
              0)

        s.add(setting.Colormap(
            'colorMap',
            'grey',
            descr = _('Set of colors to plot data with'),
            usertext=_('Colormap'),
            formatting=True),
            0)
        s.add(setting.Bool(
            'colorInvert', False,
            descr = _('Invert color map'),
            usertext=_('Invert colormap'),
            formatting=True),
            1)
        s.add(setting.Int(
            'transparency', 50,
            descr = _('Transparency percentage'),
            usertext = _('Transparency'),
            minval = 0,
            maxval = 100,
            formatting=True),
            2)
        s.add(setting.Int(
            'reflectivity', 0,
            minval=0, maxval=100,
            descr=_('Reflectivity percentage'),
            usertext=_('Reflectivity'),
            formatting=True),
            3)
        s.add(setting.Float(
            'fillfactor', 1,
            minval=0, maxval=1,
            descr=_('Filling factor (0-1)'),
            usertext=_('Fill factor'),
            formatting=True),
            4)

        s.add(setting.Line3D(
            'Line',
            descr = _('Line settings'),
            usertext = _('Box line')),
               pixmap = 'settings_plotline' )

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

    def _getdata(self, axes):
        s = self.settings
        doc = self.document
        xv = s.get('xData').getData(doc)
        yv = s.get('yData').getData(doc)
        zv = s.get('zData').getData(doc)
        vv = s.DataColor.get('points').getData(doc)
        if not xv or not yv or not zv or not vv:
            return
        trans = s.get('transData').getData(doc)

        # numpy arrays
        xv = xv.data
        yv = yv.data
        zv = zv.data
        vv = vv.data
        trans = None if trans is None else N.clip(trans.data, 0, 1)

        # trim to length
        minlen = min(len(xv),len(yv),len(zv),len(vv))
        if trans is not None:
            minlen = min(minlen, len(trans))

        xv = axes[0].transformToAxis(xv[:minlen])
        yv = axes[1].transformToAxis(yv[:minlen])
        zv = axes[2].transformToAxis(zv[:minlen])
        vv = vv[:minlen]
        trans = None if trans is None else trans[:minlen]

        # get bits with valid coordinates
        valid = N.isfinite(xv) & N.isfinite(yv) & N.isfinite(zv)
        if not N.all(valid):
            xv = xv[valid]
            yv = yv[valid]
            zv = zv[valid]
            vv = vv[valid]
            trans = None if trans is None else trans[valid]

        # get edges of boxes in graph coordinates
        ff = self.settings.fillfactor
        xedges, xidxs = autoedges(xv, ff)
        yedges, yidxs = autoedges(yv, ff)
        zedges, zidxs = autoedges(zv, ff)
        if xedges is None or yedges is None or zedges is None:
            return

        # select finite values
        valid = N.isfinite(vv)
        if trans is not None:
            valid &= N.isfinite(trans)
        if not N.all(valid):
            xidxs = xidxs[valid]
            yidxs = yidxs[valid]
            zidxs = zidxs[valid]
            vv = vv[valid]
            trans = None if trans is None else trans[valid]

        # transform back edges from axis coordinates
        xedges = axes[0].transformFromAxis(xedges)
        yedges = axes[1].transformFromAxis(yedges)
        zedges = axes[2].transformFromAxis(zedges)

        return utils.Struct(
            xedges=xedges, xidxs=xidxs, yedges=yedges, yidxs=yidxs,
            zedges=zedges, zidxs=zidxs,
            vals=vv, trans=trans)

    def dataDrawToObject(self, painter, axes):
        """Do drawing of axis."""

        s = self.settings

        axes = self.fetchAxes()
        if axes is None:
            return

        data = self._getdata(axes)
        if data is None:
            return

        cmap = self.document.evaluate.getColormap(
            s.colorMap, s.colorInvert)
        cdata = data.vals.reshape((1, data.vals.size))
        if data.trans is not None:
            transimg = data.trans.reshape((1, data.vals.size))
        else:
            transimg = None

        colorimg = utils.applyColorMap(
            cmap, s.DataColor.scaling,
            cdata,
            s.DataColor.min, s.DataColor.max,
            s.transparency, transimg=transimg)

        surfprop = threed.SurfaceProp(refl=s.reflectivity*0.01)
        surfprop.setRGBs(colorimg)

        lineprop = s.Line.makeLineProp(painter)

        # convert from axis coordinates
        xedges = axes[0].dataToLogicalCoords(data.xedges)
        yedges = axes[1].dataToLogicalCoords(data.yedges)
        zedges = axes[2].dataToLogicalCoords(data.zedges)

        # get minimum and maximum of cubes
        xmin = threed.ValVector(xedges[data.xidxs,0])
        xmax = threed.ValVector(xedges[data.xidxs,1])
        ymin = threed.ValVector(yedges[data.yidxs,0])
        ymax = threed.ValVector(yedges[data.yidxs,1])
        zmin = threed.ValVector(zedges[data.zidxs,0])
        zmax = threed.ValVector(zedges[data.zidxs,1])

        clipcont = self.makeClipContainer(axes)
        cuboids = threed.MultiCuboid(
            xmin, xmax, ymin, ymax, zmin, zmax, lineprop, surfprop)
        clipcont.addObject(cuboids)

        clipcont.assignWidgetId(id(self))
        return clipcont

document.thefactory.register(Volume3D)
