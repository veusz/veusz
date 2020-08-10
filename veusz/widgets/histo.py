#    Copyright (C) 2020 Jeremy S. Sanders
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

"""For histogramming data."""

from __future__ import division
import numpy as N

from ..compat import czip
from .. import qtall as qt
from .. import datasets
from .. import document
from .. import setting
from .. import utils
from ..helpers import qtloops

from .plotters import GenericPlotter
from .point import ErrorBarDraw

def _(text, disambiguation=None, context='Histo'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

scaling_fwd = {
    'linear': lambda x: N.array(x),
    'sqrt': lambda x: N.sqrt(x),
    'log': lambda x: N.log(x),
    'arcsinh': lambda x: N.arcsinh(x),
    'exp': lambda x: N.exp(x),
    'sqr': lambda x: x**2,
    'sinh': lambda x: N.sinh(x),
}

scaling_bkd = {
    'linear': lambda x: N.array(x),
    'sqrt': lambda x: x**2,
    'log': lambda x: N.exp(x),
    'arcsinh': lambda x: N.sinh(x),
    'exp': lambda x: N.log(x),
    'sqr': lambda x: N.sqrt(x),
    'sinh': lambda x: N.arcsinh(x),
}

def _calcerrs(data, errmode):
    if errmode == 'sqrt':
        err = N.sqrt(data)
        return err, -err
    elif errmode == 'gehrels':
        perr = 1 + N.sqrt(data + 0.75)
        nerr = -N.sqrt(N.clip(data - 0.25, 0, None))
        return perr, nerr
    elif errmode == 'none':
        return None, None
    else:
        raise RuntimeError('Unknown error mode')

def doBinning(data, weights=None,
              scaling='linear', minval='Auto', maxval='Auto',
              mode='fixed', numbins=10, manualbins=None,
              calcmode='counts', errormode='gehrels',
):

    if weights:
        minlen = min(len(weights), len(data))
        weights = weights.data[:minlen]
        data = data[:minlen]

    # scale data according to function
    sfwd = scaling_fwd[scaling]
    sbkd = scaling_bkd[scaling]
    sdata = sfwd(data)

    if minval=='Auto' or minval is None:
        minval = N.nanmin(sdata)
    else:
        minval = sfwd(minval)

    if maxval=='Auto' or maxval is None:
        maxval = N.nanmax(sdata)
    else:
        maxval = sfwd(maxval)

    if not N.isfinite(minval) or not N.isfinite(maxval) or maxval<=minval:
        # problem with scaling or values
        return

    # non finite bins should be included in fractions, etc.
    sdata[~N.isfinite(sdata)] = N.inf

    if mode == 'fixed':
        bins = numbins
    elif mode == 'manual':
        bins = N.unique(manualbins)
        if len(bins)<2:
            return
    else:
        bins = mode

    hist, edges = N.histogram(
        sdata,
        weights=weights,
        range=(minval, maxval),
        bins=bins,
        density=calcmode=='density',
    )

    # scale edges back after transformation
    sedges = sbkd(edges)

    if calcmode in ('counts', 'fraction') and weights is None:
        perr, nerr = _calcerrs(hist, errormode)
    else:
        perr = nerr = None

    if calcmode == 'fraction':
        invcts = 1/len(data)
        hist = hist * invcts
        if perr is not None:
            perr *= invcts
        if nerr is not None:
            nerr *= invcts

    return hist, perr, nerr, sedges

class FillBrush1(setting.BrushExtended):
    def __init__(self, name, **args):
        setting.BrushExtended.__init__(self, name, **args)
        self.get('color').newDefault( setting.Reference('../color') )
        self.add( setting.Choice(
            'mode',
            ('under', 'over', 'tozero'),
            'tozero',
            descr=_('Mode'),
            usertext=_('Mode')), 0)
        self.add( setting.Bool(
            'hideerror', False,
            descr = _('Hide the filled region inside the error bars'),
            usertext=_('Hide error fill')) )

class FillBrush2(FillBrush1):
    def __init__(self, name, **args):
        FillBrush1.__init__(self, name, **args)
        self.get('hide').newDefault(True)
        self.get('mode').newDefault('over')

class PlotLine(setting.Line):
    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)
        self.get('color').newDefault( setting.Reference('../color') )

class PostLine(setting.Line):
    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)
        self.get('hide').newDefault(True)
        self.get('color').newDefault( setting.Reference('../color') )

class MarkerFillBrush(setting.Brush):
    def __init__(self, name, **args):
        setting.Brush.__init__(self, name, **args)
        self.get('color').newDefault( setting.Reference('../color') )

class Histo(GenericPlotter):
    """Widget for histogramming data."""

    typename='histo'
    allowusercreation=True
    description=_('Histogram of a dataset')

    def __init__(self, *args, **argsv):
        GenericPlotter.__init__(self, *args, **argsv)
        self.changeset = -1
        self.hist = self.edges = self.perrs = self.nerrs = None

    @staticmethod
    def _showmode(mode):
        # which bins to show depending on mode
        if mode == 'fixed':
            return ('numbins',), ('manual',)
        elif mode == 'manual':
            return ('manual',), ('numbins',),
        else:
            # named mode
            return (), ('manual', 'numbins')

    @classmethod
    def addSettings(klass, s):
        GenericPlotter.addSettings(s)

        s.add( setting.DatasetExtended(
            'data', '',
            descr=_('Dataset'),
            usertext=_('Data')), 1 )
        s.add( setting.DatasetExtended(
            'weights', '',
            descr=_('Optional weight applied to counts of data'),
            usertext=_('Weights')), 2 )

        s.add( setting.FloatOrAuto(
            'minval', 'Auto',
            descr=_('Minimum of range'),
            usertext=_('Minimum')), 3 )
        s.add( setting.FloatOrAuto(
            'maxval', 'Auto',
            descr=_('Maximum of range'),
            usertext=_('Maximum')), 4 )
        s.add( setting.ChoiceSwitch(
            'scaling',
            ('linear', 'log', 'sqrt', 'arcsinh', 'exp', 'sqr', 'sinh'),
            'linear',
            descr=_('Data scaling before binning'),
            usertext=_('Scaling')), 5 )

        s.add( setting.Choice(
            'calcmode',
            ('counts', 'fraction', 'density'),
            'counts',
            descr=_('Calculate when binning'),
            usertext=_('Calculate')), 6 )

        s.add( setting.ChoiceSwitch(
            'mode',
            ('fixed', 'manual', 'auto',
             'fd', 'doane', 'scott', 'stone', 'rice',
             'sturges', 'sqrt'),
            'fixed',
            showfn=klass._showmode,
            descr=_('Binning mode'),
            usertext=_('Mode')), 7 )

        s.add( setting.Int(
            'numbins', 10,
            minval=1, maxval=100000,
            descr=_('Number of bins'),
            usertext=_('Number')), 8)
        s.add( setting.FloatList(
            'manual',
            [],
            descr=_('Manual binning edges'),
            usertext=_('Manual')), 9)

        s.add( setting.Choice(
            'errormode',
            ('none', 'sqrt', 'gehrels'),
            'gehrels',
            descr=_('Error estimation'),
            usertext=_('Uncertainty')), 10 )

        s.add( setting.Choice(
            'direction',
            ('horizontal', 'vertical'),
            'vertical',
            descr=_('Bars direction'),
            usertext=_('Direction')), 11 )

        s.add( setting.Color(
            'color',
            'auto',
            descr = _('Master color'),
            usertext = _('Color'),
            formatting=True), 0 )
        s.add( setting.Choice(
            'style',
            ('step', 'join'),
            'step',
            descr = _('Drawing style'),
            usertext = _('Style'),
            formatting=True), 1 )
        s.add( setting.Marker(
            'marker',
            'none',
            descr = _('Type of marker to plot'),
            usertext=_('Marker'), formatting=True), 2 )
        s.add( setting.DistancePt(
            'markerSize',
            '3pt',
            descr = _('Size of marker to plot'),
            usertext=_('Marker size'), formatting=True), 3 )
        s.add( setting.ErrorStyle(
            'errorStyle',
            'none',
            descr=_('Style of error bars to plot'),
            usertext=_('Error style'), formatting=True), 4 )

        s.add( PlotLine(
            'Line',
            descr = _('Plot line'),
            usertext = _('Plot line')),
               pixmap = 'settings_plotline' )
        s.add( FillBrush1(
            'Fill1',
            descr = _('Fill under'),
            usertext = _('Fill under')),
               pixmap = 'settings_plotfillbelow' )
        s.add( FillBrush2(
            'Fill2',
            descr = _('Fill over'),
            usertext = _('Fill over')),
               pixmap = 'settings_plotfillabove' )
        s.add( PostLine(
            'PostLine',
            descr = _('Post line'),
            usertext = _('Post line')),
               pixmap = 'settings_postline' )

        s.add( setting.MarkerLine(
            'MarkerLine',
            descr = _('Line around marker'),
            usertext = _('Marker border')),
               pixmap = 'settings_plotmarkerline' )
        s.add( MarkerFillBrush(
            'MarkerFill',
            descr = _('Marker fill'),
            usertext = _('Marker fill')),
               pixmap = 'settings_plotmarkerfill' )

        s.add( setting.ErrorBarLine(
            'ErrorBarLine',
            descr = _('Error bar line'),
            usertext = _('Error bar line')),
               pixmap = 'settings_ploterrorline' )
        s.ErrorBarLine.get('color').newDefault( setting.Reference('../color') )

    def computeHisto(self):
        if self.document.changeset < self.changeset:
            return
        self.changeset = self.document.changeset

        s = self.settings
        self.hist = self.edges = self.perrs = self.nerrs = None

        dsetn = self.settings.get('data')
        ds = dsetn.getData(self.document)
        if ds is None or dsetn.isEmpty():
            return
        data = ds.data
        if len(data) == 0:
            return

        # weighting for points
        weights = self.settings.get('weights').getData(self.document)
        if not weights:
            weights = None

        retn = doBinning(
            data, weights=weights,
            scaling=s.scaling, minval=s.minval, maxval=s.maxval,
            mode=s.mode, numbins=s.numbins,
            manualbins=s.manual,
            calcmode=s.calcmode, errormode=s.errormode)
        if retn is None:
            return

        self.hist, self.perr, self.nerr, self.edges = retn

    def affectsAxisRange(self):
        """Which axes are affected by this plotter?"""
        s = self.settings
        if s.direction == 'vertical':
            return ( (s.xAxis, 'bins'), (s.yAxis, 'cts') )
        else:
            return ( (s.xAxis, 'cts'), (s.yAxis, 'bins') )

    def getRange(self, axis, depname, axrange):
        """Update axis range based on data."""
        self.computeHisto()
        if self.hist is None:
            return

        if depname == 'bins':
            # we're looking at the bin positions
            dlo = dhi = self.edges

        else:
            # these are the bin heights
            dhi = dlo = self.hist
            if self.perr is not None:
                dhi = dhi + self.perr
            if self.nerr is not None:
                dlo = dlo + self.nerr

        minv = maxv = None
        if axis.settings.log:
            sel = dlo>0
            if N.any(sel):
                minv = dlo[sel].min()
            sel = dhi>0
            if N.any(sel):
                maxv = dhi[sel].max()
        else:
            if len(dlo) > 0:
                minv = dlo.min()
            if len(dhi) > 0:
                maxv = dhi.max()

        if minv is not None:
            axrange[0] = min(axrange[0], minv)
        if maxv is not None:
            axrange[1] = max(axrange[1], maxv)

    def dataDraw(self, painter, axes, posn, cliprect):
        """Draw the histogram."""

        s = self.settings
        self.computeHisto()
        if self.hist is None:
            return

        vert = s.direction == 'vertical'
        if vert:
            binaxis, ctaxis = axes
        else:
            ctaxis, binaxis = axes
        ctslog = ctaxis.settings.log

        # calculate bin midpoint according to scaling
        scaling = s.scaling
        sedges = scaling_fwd[scaling](self.edges)
        midpts = scaling_bkd[scaling]( 0.5*(sedges[1:]+sedges[:-1]) )
        midplot = binaxis.dataToPlotterCoords(posn, midpts)

        linepoly = qt.QPolygonF()
        ctsplot = ctaxis.dataToPlotterCoords(posn, self.hist)
        style = s.style
        if style == 'step':
            # stepped line
            edgeplot = binaxis.dataToPlotterCoords(posn, self.edges)
            if vert:
                qtloops.addNumpyToPolygonF(
                    linepoly, edgeplot[:-1], ctsplot, edgeplot[1:], ctsplot)
            else:
                qtloops.addNumpyToPolygonF(
                    linepoly, ctsplot, edgeplot[:-1], ctsplot, edgeplot[1:])
        else:
            # non-stepped line
            if vert:
                qtloops.addNumpyToPolygonF(linepoly, midplot, ctsplot)
            else:
                qtloops.addNumpyToPolygonF(linepoly, ctsplot, midplot)

        # coordinates of zero value
        if ctslog:
            zeropos = ctaxis.dataToPlotterCoords(posn, N.array([1e-99]))[0]
        else:
            zeropos = ctaxis.dataToPlotterCoords(posn, N.array([0]))[0]

        # fill above or below the curve
        for fill in s.Fill1, s.Fill2:
            if not fill.hide:
                firstpt = linepoly.first()
                lastpt = linepoly.last()
                fillpoly = qt.QPolygonF(linepoly)
                if vert:
                    if fill.mode == 'under':
                        yfill = posn[3]
                    elif fill.mode == 'over':
                        yfill = posn[1]
                    elif fill.mode == 'tozero':
                        yfill = zeropos
                    fillpoly.append(qt.QPointF(lastpt.x(), yfill))
                    fillpoly.append(qt.QPointF(firstpt.x(), yfill))
                else:
                    if fill.mode == 'under':
                        xfill = posn[0]
                    elif fill.mode == 'over':
                        xfill = posn[2]
                    elif fill.mode == 'tozero':
                        xfill = zeropos
                    fillpoly.append(qt.QPointF(xfill, lastpt.y()))
                    fillpoly.append(qt.QPointF(xfill, firstpt.y()))
                clippoly = qt.QPolygonF()
                qtloops.polygonClip(fillpoly, cliprect, clippoly)
                path = qt.QPainterPath()
                path.addPolygon(clippoly)
                utils.brushExtFillPath(painter, fill, path)

        # the post line which goes down to zero from the data line
        if not s.PostLine.hide:
            painter.setPen(s.PostLine.makeQPen(painter))
            pos = edgeplot if style=='step' else midplot
            zeros = N.full(pos.shape, zeropos)
            if vert:
                qtloops.plotLinesToPainter(
                    painter, pos, ctsplot, pos, zeros,
                    cliprect)
            else:
                qtloops.plotLinesToPainter(
                    painter, ctsplot, pos, zeros, pos,
                    cliprect)

        # the plot line (stepped or not)
        if not s.Line.hide:
            painter.setPen(s.Line.makeQPen(painter))
            qtloops.plotClippedPolyline(painter, cliprect, linepoly)

        # points and error bars
        if vert:
            xplt, yplt = midplot, ctsplot
        else:
            yplt, xplt = ctsplot, midplot

        # any error bars
        markersize = s.get('markerSize').convert(painter)
        if s.errorStyle != 'none' and self.perr is not None:
            minv = ctaxis.dataToPlotterCoords(posn, self.hist+self.nerr)
            maxv = ctaxis.dataToPlotterCoords(posn, self.hist+self.perr)
            if vert:
                ymin, ymax = minv, maxv
                xmin = xmax = None
            else:
                xmin, xmax = minv, maxv
                ymin = ymax = None

            ebp = ErrorBarDraw(
                s.errorStyle, s.ErrorBarLine, s.Fill2, s.Fill1, markersize)
            ebp.plot(painter, xmin, xmax, ymin, ymax, xplt, yplt, cliprect)

        # plot data points
        if not s.MarkerLine.hide or not s.MarkerFill.hide:
            if not s.MarkerFill.hide:
                painter.setBrush(s.MarkerFill.makeQBrush(painter))
            else:
                painter.setBrush(qt.QBrush())

            if not s.MarkerLine.hide:
                painter.setPen(s.MarkerLine.makeQPen(painter))
            else:
                painter.setPen(qt.QPen(qt.Qt.NoPen))

            utils.plotMarkers(
                painter, xplt, yplt, s.marker, markersize,
                clip=cliprect)

document.thefactory.register(Histo)
