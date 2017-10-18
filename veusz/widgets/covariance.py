#    Copyright (C) 2016 Jeremy S. Sanders
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

"""Widget for plotting data covariance."""

from __future__ import division
import numpy as N

from .. import qtall as qt4
from ..compat import czip
from . import plotters
from .. import document
from .. import setting
from .. import utils

def _(text, disambiguation=None, context='Covariance'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class CovarianceLine(setting.Line):
    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)

        self.add( setting.Int(
            'steps',
            25,
            minval=4,
            descr=_('Number of line steps to draw'),
            usertext=_('Steps') ))

        self.get('color').newDefault('auto')

class Covariance(plotters.GenericPlotter):
    """Plot covariance matrix for points as shapes."""

    typename = 'covariance'
    allowusercreation=True
    description=_('Plot covariance ellipses')

    def __init__(self, parent, **args):
        """Initialise plotter."""

        plotters.GenericPlotter.__init__(self, parent, **args)

        self._elpts = []
        self._changeset = -1

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.GenericPlotter.addSettings(s)

        s.add( setting.DatasetExtended(
            'covyy', '',
            descr=_('Covariance matrix entry (Y,Y) [computed from data if blank]'),
            usertext=_('Cov(Y,Y)')), 0 )
        s.add( setting.DatasetExtended(
            'covxy', '',
            descr=_('Covariance matrix entry (X,Y) [computed from data if blank]'),
            usertext=_('Cov(X,Y)')), 0 )
        s.add( setting.DatasetExtended(
            'covyx', '',
            descr=_('Covariance matrix entry (Y,X) [computed from data if blank]'),
            usertext=_('Cov(Y,X)')), 0 )
        s.add( setting.DatasetExtended(
            'covxx', '',
            descr=_('Covariance matrix entry (X,X) [computed from data if blank]'),
            usertext=_('Cov(X,X)')), 0 )

        s.add( setting.DatasetExtended(
            'yData', 'y',
            descr=_('Y values, given by dataset, expression or list of values'),
            usertext=_('Y data')), 0 )
        s.add( setting.DatasetExtended(
            'xData', 'x',
            descr=_('X values, given by dataset, expression or list of values'),
            usertext=_('X data')), 0 )

        s.add( CovarianceLine(
            'Line',
            descr = _('Line'),
            usertext = _('Ellipse line')),
               pixmap = 'settings_plotline' )

        s.add( setting.PlotterFill(
            'Fill',
            descr = _('Fill'),
            usertext = _('Ellipse fill')),
               pixmap = 'settings_plotfillbelow' )

    def _computeCovFromData(self, data):
        """Compute a single covariance matrix given data."""

        minlen = min(len(data['xData']), len(data['yData']))
        xd = data['xData'][:minlen]
        yd = data['yData'][:minlen]
        finite = N.isfinite(xd) & N.isfinite(yd)
        xd = xd[finite]
        yd = yd[finite]

        if len(xd) < 2:
            cov = N.array([[0,0],[0,0]])
        else:
            cov = N.cov(xd, y=yd)

        data['xData'] = N.array([N.mean(xd)])
        data['yData'] = N.array([N.mean(yd)])
        data['covxx'] = N.array([cov[0,0]])
        data['covxy'] = N.array([cov[1,0]])
        data['covyx'] = N.array([cov[0,1]])
        data['covyy'] = N.array([cov[1,1]])

    def _computeEllipses(self):
        """Calculate points for ellipses."""

        s = self.settings
        d = self.document

        # cache existing value if dataset unchanged
        if self._changeset == d.changeset:
            return

        self._elpts = []

        minlen = 1e99
        data = {}
        anynone = False
        for attr in 'xData', 'yData', 'covxx', 'covxy', 'covyx', 'covyy':
            dataset = s.get(attr).getData(d)
            # needs to be defined
            if dataset is not None:
                ds = dataset.data
                minlen = min(minlen, len(ds))
                data[attr] = ds
            else:
                data[attr] = None
                anynone = True

        # if covariance matrix not provided, compute from xy data
        if ( data['xData'] is not None and data['yData'] is not None and
             s.get('covxx').isEmpty() and s.get('covxy').isEmpty() and
             s.get('covyx').isEmpty() and s.get('covyy').isEmpty() ):
            self._computeCovFromData(data)
            minlen=1
        elif anynone:
            # invalid
            return

        # chop to minimum length
        for attr in data:
            if minlen != len(data[attr]):
                data[attr] = data[attr][:minlen]

        # remove invalid values
        valid = N.all([N.isfinite(ds) for ds in data.values()], axis=0)
        for attr in data:
            data[attr] = data[attr][valid]

        # construct covariance matrices
        cov = N.column_stack((
            data['covxx'], data['covyx'], data['covxy'], data['covyy']
            )).reshape(minlen, 2, 2)

        # compute eigenvalues and vectors from covariance matrices
        try:
            eigvals, eigvecs = N.linalg.eig(cov)
        except N.linalg.LinAlgError:
            return

        # multiply vectors be sqrt eigenvalues (error is sqrt)
        sqrtvals = N.sqrt(eigvals)
        eigcomb = eigvecs * sqrtvals[:,:,None]

        # generate points in ellipse
        numsteps = s.Line.steps
        x = N.linspace(0, N.pi*2, numsteps, endpoint=False)
        f1, f2 = N.cos(x), N.sin(x)

        combv = f1*eigcomb[:,0,:,None] + f2*eigcomb[:,1,:,None]
        xpts = data['xData'][:,None] - combv[:,0,:]
        ypts = data['yData'][:,None] + combv[:,1,:]

        # funny covariance matrix does this
        if N.any(N.iscomplex(xpts)) or N.any(N.iscomplex(ypts)):
            return

        # now we have the points
        self._elpts = [xpts, ypts]

    def affectsAxisRange(self):
        """This widget provides range information about these axes."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def getRange(self, axis, depname, axrange):
        """Return range of data."""

        self._computeEllipses()
        if not self._elpts:
            return

        vals = self._elpts[{'sx': 0, 'sy': 1}[depname]]
        if axis.settings.log:
            vals = N.clip(vals, 1e-99, 1e99)
        if len(vals) > 0:
            axrange[0] = min(axrange[0], vals.min())
            axrange[1] = max(axrange[1], vals.max())

    def dataDraw(self, painter, axes, posn, cliprect):
        """Draw dataset."""

        s = self.settings

        self._computeEllipses()
        if not self._elpts:
            return

        if axes[0].settings.log:
            self._elpts[0] = N.clip(self._elpts[0], 1e-99, 1e99)
        if axes[1].settings.log:
            self._elpts[1] = N.clip(self._elpts[1], 1e-99, 1e99)

        ptsx = axes[0].dataToPlotterCoords(posn, self._elpts[0])
        ptsy = axes[1].dataToPlotterCoords(posn, self._elpts[1])

        pen = s.Line.makeQPenWHide(painter)
        pw = pen.widthF()*2
        x1, y1, x2, y2 = posn
        lineclip = qt4.QRectF(
            qt4.QPointF(x1-pw, y1-pw), qt4.QPointF(x2+pw, y2+pw))

        for xvals, yvals in czip(ptsx, ptsy):
            path = qt4.QPainterPath()
            poly = qt4.QPolygonF()
            utils.addNumpyToPolygonF(poly, xvals, yvals)
            clippedpoly = qt4.QPolygonF()
            utils.polygonClip(poly, lineclip, clippedpoly)
            path.addPolygon(clippedpoly)
            path.closeSubpath()
            utils.brushExtFillPath(painter, s.Fill, path, stroke=pen)

document.thefactory.register(Covariance)
