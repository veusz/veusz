#    Copyright (C) 2008 Jeremy S. Sanders
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

"""For plotting xy points."""

from __future__ import division
import numpy as N

from ..compat import czip
from .. import qtall as qt
from .. import datasets
from .. import document
from .. import setting
from .. import utils

from . import pickable
from .plotters import GenericPlotter

from ..helpers import qtloops

def _(text, disambiguation=None, context='XY'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class ErrorBarDraw:
    """For plotting error bars."""

    def __init__(self, style, linestyle, fillabove, fillbelow, markersize):
        self.style = style
        self.linestyle = linestyle
        self.fillabove = fillabove
        self.fillbelow = fillbelow
        self.markersize = markersize

    def plot(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        pen = self.linestyle.makeQPenWHide(painter)
        pen.setCapStyle(qt.Qt.FlatCap)

        painter.setPen(pen)
        for function in self.error_functions[self.style]:
            function(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip)

    def errorsBar(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Draw bar style error lines."""
        # vertical error bars
        if ymin is not None and ymax is not None and not self.linestyle.hideVert:
            qtloops.plotLinesToPainter(painter, xplt, ymin, xplt, ymax, clip)

        # horizontal error bars
        if xmin is not None and xmax is not None and not self.linestyle.hideHorz:
            qtloops.plotLinesToPainter(painter, xmin, yplt, xmax, yplt, clip)

    def errorsBarHi(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Draw bar style error lines (top half only)."""
        if ymin is not None and ymax is not None and not self.linestyle.hideVert:
            qtloops.plotLinesToPainter(painter, xplt, yplt, xplt, ymax, clip)
        if xmin is not None and xmax is not None and not self.linestyle.hideHorz:
            qtloops.plotLinesToPainter(painter, xplt, yplt, xmax, yplt, clip)

    def errorsBarLo(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Draw bar style error lines (bottom half only)."""
        if ymin is not None and ymax is not None and not self.linestyle.hideVert:
            qtloops.plotLinesToPainter(painter, xplt, yplt, xplt, ymin, clip)
        if xmin is not None and xmax is not None and not self.linestyle.hideHorz:
            qtloops.plotLinesToPainter(painter, xplt, yplt, xmin, yplt, clip)

    def errorsEnds(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Draw perpendiclar ends on error bars."""
        size = self.markersize * self.linestyle.endsize

        if ymin is not None and ymax is not None and not self.linestyle.hideVert:
            qtloops.plotLinesToPainter(
                painter, xplt-size, ymin, xplt+size, ymin, clip)
            qtloops.plotLinesToPainter(
                painter, xplt-size, ymax, xplt+size, ymax, clip)

        if xmin is not None and xmax is not None and not self.linestyle.hideHorz:
            qtloops.plotLinesToPainter(
                painter, xmin, yplt-size, xmin, yplt+size, clip)
            qtloops.plotLinesToPainter(
                painter, xmax, yplt-size, xmax, yplt+size, clip)

    def errorsEndsHi(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Draw perpendiclar ends on error bars (top half only)."""
        size = self.markersize * self.linestyle.endsize
        if ymin is not None and ymax is not None and not self.linestyle.hideVert:
            qtloops.plotLinesToPainter(
                painter, xplt-size, ymax, xplt+size, ymax, clip)
        if xmin is not None and xmax is not None and not self.linestyle.hideHorz:
            qtloops.plotLinesToPainter(
                painter, xmax, yplt-size, xmax, yplt+size, clip)

    def errorsEndsLo(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Draw perpendiclar ends on error bars (bottom half only)."""
        size = self.markersize * self.linestyle.endsize
        if ymin is not None and ymax is not None and not self.linestyle.hideVert:
            qtloops.plotLinesToPainter(
                painter, xplt-size, ymin, xplt+size, ymin, clip)
        if xmin is not None and xmax is not None and not self.linestyle.hideHorz:
            qtloops.plotLinesToPainter(
                painter, xmin, yplt-size, xmin, yplt+size, clip)

    def errorsBox(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Draw box around error region."""
        if utils.allNotNone(xmin, xmax, ymin, ymax):
            painter.setBrush(qt.QBrush())
            qtloops.plotBoxesToPainter(painter, xmin, ymin, xmax, ymax, clip)

    def errorsBoxFilled(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Draw box filled region inside error bars."""
        if utils.anyNone(xmin, xmax, ymin, ymax):
            return

        # filled region below
        if not self.fillbelow.hideerror:
            path = qt.QPainterPath()
            qtloops.addNumpyPolygonToPath(
                path, clip, xmin, ymin, xmin, yplt, xmax, yplt, xmax, ymin)
            utils.brushExtFillPath(painter, self.fillbelow, path, ignorehide=True)

        # filled region above
        if not self.fillabove.hideerror:
            path = qt.QPainterPath()
            qtloops.addNumpyPolygonToPath(
                path, clip, xmin, yplt, xmax, yplt, xmax, ymax, xmin, ymax)
            utils.brushExtFillPath(painter, self.fillabove, path, ignorehide=True)

    def errorsDiamond(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Draw diamond around error region."""
        if utils.anyNone(xmin, xmax, ymin, ymax):
            return

        # expand clip by pen width (urgh)
        pw = painter.pen().widthF()*2
        clip = qt.QRectF(
            qt.QPointF(clip.left()-pw,clip.top()-pw),
            qt.QPointF(clip.right()+pw,clip.bottom()+pw))

        path = qt.QPainterPath()
        qtloops.addNumpyPolygonToPath(
            path, clip, xmin, yplt, xplt, ymax, xmax, yplt, xplt, ymin)
        painter.setBrush( qt.QBrush() )
        painter.drawPath(path)

    def errorsDiamondFilled(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Draw diamond filled region inside error bars."""
        if utils.anyNone(xmin, xmax, ymin, ymax):
            return

        if not self.fillbelow.hideerror:
            path = qt.QPainterPath()
            qtloops.addNumpyPolygonToPath(
                path, clip, xmin, yplt, xplt, ymin, xmax, yplt)
            utils.brushExtFillPath(painter, self.fillbelow, path, ignorehide=True)

        if not self.fillabove.hideerror:
            path = qt.QPainterPath()
            qtloops.addNumpyPolygonToPath(
                path, clip, xmin, yplt, xplt, ymax, xmax, yplt)
            utils.brushExtFillPath(painter, self.fillabove, path, ignorehide=True)

    def errorsCurve(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Draw curve around error region."""
        if utils.anyNone(xmin, xmax, ymin, ymax):
            return

        # non-filling brush
        painter.setBrush( qt.QBrush() )

        for xp, yp, xmn, ymn, xmx, ymx in czip(xplt, yplt, xmin, ymin, xmax, ymax):
            p = qt.QPainterPath()
            p.moveTo(xp + (xmx-xp), yp)
            p.arcTo(qt.QRectF(
                xp - (xmx-xp), yp - (yp-ymx), (xmx-xp)*2, (yp-ymx)*2), 0., 90.)
            p.arcTo(qt.QRectF(
                xp - (xp-xmn), yp - (yp-ymx), (xp-xmn)*2, (yp-ymx)*2), 90., 90.)
            p.arcTo(qt.QRectF(
                xp - (xp-xmn), yp - (ymn-yp), (xp-xmn)*2, (ymn-yp)*2), 180., 90.)
            p.arcTo(qt.QRectF(
                xp - (xmx-xp), yp - (ymn-yp), (xmx-xp)*2, (ymn-yp)*2), 270., 90.)
            painter.drawPath(p)

    def errorsCurveFilled(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Fill area around error region."""

        if utils.anyNone(xmin, xmax, ymin, ymax):
            return

        for xp, yp, xmn, ymn, xmx, ymx in czip(xplt, yplt, xmin, ymin, xmax, ymax):

            if not self.fillabove.hideerror:
                p = qt.QPainterPath()
                p.moveTo(xp + (xmx-xp), yp)
                p.arcTo(qt.QRectF(
                    xp - (xmx-xp), yp - (yp-ymx), (xmx-xp)*2, (yp-ymx)*2), 0., 90.)
                p.arcTo(qt.QRectF(
                    xp - (xp-xmn), yp - (yp-ymx), (xp-xmn)*2, (yp-ymx)*2), 90., 90.)
                utils.brushExtFillPath(painter, self.fillabove, p, ignorehide=True)

            if not self.fillbelow.hideerror:
                p = qt.QPainterPath()
                p.moveTo(xp + (xp-xmn), yp)
                p.arcTo(qt.QRectF(
                    xp - (xp-xmn), yp - (ymn-yp), (xp-xmn)*2, (ymn-yp)*2), 180., 90.)
                p.arcTo(qt.QRectF(
                    xp - (xmx-xp), yp - (ymn-yp), (xmx-xp)*2, (ymn-yp)*2), 270., 90.)
                utils.brushExtFillPath(painter, self.fillbelow, p, ignorehide=True)

    def errorsFilled(self, painter, xmin, xmax, ymin, ymax, xplt, yplt, clip):
        """Draw filled region as error region."""

        ptsabove = qt.QPolygonF()
        ptsbelow = qt.QPolygonF()

        hidevert = True  # keep track of what's shown
        hidehorz = True
        if ( 'vert' in self.style and
             (ymin is not None and ymax is not None) and
             not self.linestyle.hideVert ):
            hidevert = False
            # lines above/below points
            if self.style[-2:] != 'hi':
                qtloops.addNumpyToPolygonF(ptsbelow, xplt, ymin)
            if self.style[-2:] != 'lo':
                qtloops.addNumpyToPolygonF(ptsabove, xplt, ymax)

        elif ( 'horz' in self.style and
               (xmin is not None and xmax is not None) and
               not self.linestyle.hideHorz ):
            hidehorz = False
            # lines left/right points
            if self.style[-2:] != 'hi':
                qtloops.addNumpyToPolygonF(ptsbelow, xmin, yplt)
            if self.style[-2:] != 'lo':
                qtloops.addNumpyToPolygonF(ptsabove, xmax, yplt)

        # draw filled regions above/left and below/right
        if 'fill' in self.style and not (hidehorz and hidevert):
            # construct points for error bar regions
            retnpts = qt.QPolygonF()
            qtloops.addNumpyToPolygonF(retnpts, xplt[::-1], yplt[::-1])

            # polygons consist of lines joining the points and continuing
            # back along the plot line (retnpts)
            if not self.fillbelow.hideerror and ptsbelow:
                utils.brushExtFillPolygon(
                    painter, self.fillbelow, clip, ptsbelow+retnpts, ignorehide=True)
            if not self.fillabove.hideerror and ptsabove:
                utils.brushExtFillPolygon(
                    painter, self.fillabove, clip, ptsabove+retnpts, ignorehide=True)

        # draw optional line (on top of fill)
        if ptsabove:
            qtloops.plotClippedPolyline(painter, clip, ptsabove)
        if ptsbelow:
            qtloops.plotClippedPolyline(painter, clip, ptsbelow)

    # map error bar names to lists of functions (above)
    error_functions = {
        'none': (),
        'bar': (errorsBar,),
        'bardiamond': (errorsBar, errorsDiamond,),
        'barcurve': (errorsBar, errorsCurve,),
        'barbox': (errorsBar, errorsBox,),
        'barends': (errorsBar, errorsEnds,),
        'box':  (errorsBox,),
        'boxfill': (errorsBoxFilled, errorsBox,),
        'diamond':  (errorsDiamond,),
        'diamondfill':  (errorsDiamond, errorsDiamondFilled),
        'curve': (errorsCurve,),
        'curvefill': (errorsCurveFilled, errorsCurve,),
        'fillhorz': (errorsFilled,),
        'fillvert': (errorsFilled,),
        'linehorz': (errorsFilled,),
        'linevert': (errorsFilled,),
        'linehorzbar': (errorsBar, errorsFilled),
        'linevertbar': (errorsBar, errorsFilled),
        'barhi': (errorsBarHi,),
        'barlo': (errorsBarLo,),
        'barendshi': (errorsBarHi, errorsEndsHi,),
        'barendslo': (errorsBarLo, errorsEndsLo,),
        'linehorzlo': (errorsFilled,),
        'linehorzhi': (errorsFilled,),
        'linevertlo': (errorsFilled,),
        'lineverthi': (errorsFilled,),
    }

def fillPtsToEdge(painter, pts, posn, cliprect, fillstyle):
    """Fill points depending on fill mode."""
    ft = fillstyle.fillto
    if ft == 'top':
        x1, x2 = pts[0].x(), pts[-1].x()
        y1 = y2 = posn[1]
    elif ft == 'bottom':
        x1, x2 = pts[0].x(), pts[-1].x()
        y1 = y2 = posn[3]
    elif ft == 'left':
        y1, y2 = pts[0].y(), pts[-1].y()
        x1 = x2 = posn[0]
    elif ft == 'right':
        y1, y2 = pts[0].y(), pts[-1].y()
        x1 = x2 = posn[2]
    else:
        raise RuntimeError('Invalid fillto mode')

    polypts = qt.QPolygonF([qt.QPointF(x1, y1)])
    polypts += pts
    polypts.append(qt.QPointF(x2, y2))

    utils.brushExtFillPolygon(painter, fillstyle, cliprect, polypts)

class MarkerFillBrush(setting.Brush):
    def __init__(self, name, **args):
        setting.Brush.__init__(self, name, **args)

        self.get('color').newDefault( setting.Reference('../color') )

        self.add( setting.Colormap(
            'colorMap', 'grey',
            descr = _('If color markers dataset is given, use this colormap '
                      'instead of the fill color'),
            usertext=_('Color map'),
            formatting=True) )
        self.add( setting.Bool(
            'colorMapInvert', False,
            descr = _('Invert color map'),
            usertext = _('Invert map'),
            formatting=True) )

class PointPlotter(GenericPlotter):
    """A class for plotting points and their errors."""

    typename='xy'
    allowusercreation=True
    description=_('Plot points with lines and errorbars')

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        GenericPlotter.addSettings(s)

        # non-formatting
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

        # formatting
        s.add( setting.Int(
            'errorthin', 1,
            minval=1,
            descr=_('Thin number of error bars plotted by this factor'),
            usertext=_('Thin errors'),
            formatting=True), 0 )
        s.add( setting.Int(
            'thinfactor', 1,
            minval=1,
            descr=_('Thin number of markers plotted'
                    ' for each datapoint by this factor'),
            usertext=_('Thin markers'),
            formatting=True), 0 )
        s.add( setting.Color(
            'color',
            'auto',
            descr = _('Master color'),
            usertext = _('Color'),
            formatting=True), 0 )
        s.add( setting.DistancePt(
            'markerSize',
            '3pt',
            descr = _('Size of marker to plot'),
            usertext=_('Marker size'), formatting=True), 0 )
        s.add( setting.Marker(
            'marker',
            'circle',
            descr = _('Type of marker to plot'),
            usertext=_('Marker'), formatting=True), 0 )
        s.add( setting.DataColor('Color') )

        s.add( setting.ErrorStyle(
            'errorStyle',
            'bar',
            descr=_('Style of error bars to plot'),
            usertext=_('Error style'), formatting=True) )

        s.add( setting.XYPlotLine(
            'PlotLine',
            descr = _('Plot line'),
            usertext = _('Plot line')),
               pixmap = 'settings_plotline' )

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

        s.add( setting.PointFill(
            'FillBelow',
            descr = _('Fill mode 1'),
            usertext = _('Fill 1')),
               pixmap = 'settings_plotfillbelow' )
        s.FillBelow.get('fillto').newDefault('bottom')
        s.add( setting.PointFill(
            'FillAbove',
            descr = _('Fill 2'),
            usertext = _('Fill 2')),
               pixmap = 'settings_plotfillabove' )
        s.add( setting.PointLabel(
            'Label',
            descr = _('Label settings'),
            usertext=_('Label')),
               pixmap = 'settings_axislabel' )

    @property
    def userdescription(self):
        """User-friendly description."""

        s = self.settings
        return "x='%s', y='%s', marker='%s'" % (
            s.xData, s.yData, s.marker)

    def _plotErrors(self, posn, painter, xplotter, yplotter,
                    axes, xdata, ydata, cliprect):
        """Plot error bars (horizontal and vertical).
        """

        s = self.settings
        style = s.errorStyle
        if style == 'none':
            return

        # optional thinning of error bars plotted
        thin = s.errorthin

        # default is no error bars
        xmin = xmax = ymin = ymax = None

        # draw horizontal error bars
        if xdata.hasErrors():
            xmin, xmax = xdata.getPointRanges()
            if thin>1:
                xmin, xmax = xmin[::thin], xmax[::thin]

            # convert xmin and xmax to graph coordinates
            xmin = axes[0].dataToPlotterCoords(posn, xmin)
            xmax = axes[0].dataToPlotterCoords(posn, xmax)

        # draw vertical error bars
        if ydata.hasErrors():
            ymin, ymax = ydata.getPointRanges()
            if thin>1:
                ymin, ymax = ymin[::thin], ymax[::thin]

            # convert ymin and ymax to graph coordinates
            ymin = axes[1].dataToPlotterCoords(posn, ymin)
            ymax = axes[1].dataToPlotterCoords(posn, ymax)

        # no error bars - break out of processing below
        if ymin is None and ymax is None and xmin is None and xmax is None:
            return

        if thin>1:
            xplotter, yplotter = xplotter[::thin], yplotter[::thin]

        markersize = s.get('markerSize').convert(painter)
        ebp = ErrorBarDraw(
            s.errorStyle, s.ErrorBarLine, s.FillAbove, s.FillBelow, markersize)
        ebp.plot(painter, xmin, xmax, ymin, ymax, xplotter, yplotter, cliprect)

    def affectsAxisRange(self):
        """This widget provides range information about these axes."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def getRange(self, axis, depname, axrange):
        """Compute the effect of data on the axis range."""
        dataname = {'sx': 'xData', 'sy': 'yData'}[depname]
        dsetn = self.settings.get(dataname)
        data = dsetn.getData(self.document)

        if data:
            data.updateRangeAuto(axrange, axis.settings.log)
        elif dsetn.isEmpty():
            # no valid dataset.
            # check if there a valid dataset for the other axis.
            # if there is, treat this as a row number
            dataname = {'sy': 'xData', 'sx': 'yData'}[depname]
            data = self.settings.get(dataname).getData(self.document)
            if data:
                length = data.data.shape[0]
                axrange[0] = min(axrange[0], 1)
                axrange[1] = max(axrange[1], length)

    def _getLinePoints( self, xvals, yvals, posn, xdata, ydata ):
        """Get the points corresponding to the line connecting the points."""

        pts = qt.QPolygonF()

        s = self.settings
        steps = s.PlotLine.steps

        # simple continuous line
        if steps == 'off':
            utils.addNumpyToPolygonF(pts, xvals, yvals)

        # stepped line, with points on left
        elif steps[:4] == 'left':
            x1 = xvals[:-1]
            x2 = xvals[1:]
            y1 = yvals[:-1]
            y2 = yvals[1:]
            utils.addNumpyToPolygonF(pts, x1, y1, x2, y1, x2, y2)

        # stepped line, with points on right
        elif steps[:5] == 'right':
            x1 = xvals[:-1]
            x2 = xvals[1:]
            y1 = yvals[:-1]
            y2 = yvals[1:]
            utils.addNumpyToPolygonF(pts, x1, y1, x1, y2, x2, y2)

        # stepped line, with points in centre
        # this is complex as we can't use the mean of the plotter coords,
        #  as the axis could be log
        elif steps[:6] == 'centre':
            axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

            if xdata.hasErrors():
                # Special case if error bars on x points:
                # here we use the error bars to define the steps
                xmin, xmax = xdata.getPointRanges()

                # this is duplicated from drawing error bars: bad
                # convert xmin and xmax to graph coordinates
                xmin = axes[0].dataToPlotterCoords(posn, xmin)
                xmax = axes[0].dataToPlotterCoords(posn, xmax)
                utils.addNumpyToPolygonF(pts, xmin, yvals, xmax, yvals)

            else:
                # we put the bin edges half way between the points
                # we assume this is the correct thing to do even in log space
                x1 = xvals[:-1]
                x2 = xvals[1:]
                y1 = yvals[:-1]
                y2 = yvals[1:]
                xc = 0.5*(x1+x2)
                utils.addNumpyToPolygonF(pts, x1, y1, xc, y1, xc, y2)

                if len(xvals) > 0:
                    pts.append( qt.QPointF(xvals[-1], yvals[-1]) )

        elif steps[:7] == 'vcentre':
            axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

            if ydata.hasErrors():
                # Special case if error bars on y points:
                # here we use the error bars to define the steps
                ymin, ymax = ydata.getPointRanges()

                # this is duplicated from drawing error bars: bad
                # convert ymin and ymax to graph coordinates
                ymin = axes[1].dataToPlotterCoords(posn, ymin)
                ymax = axes[1].dataToPlotterCoords(posn, ymax)
                utils.addNumpyToPolygonF(pts, xvals, ymin, xvals, ymax)

            else:
                # we put the bin edges half way between the points
                # we assume this is the correct thing to do even in log space
                y1 = yvals[:-1]
                y2 = yvals[1:]
                x1 = xvals[:-1]
                x2 = xvals[1:]
                yc = 0.5*(y1+y2)
                utils.addNumpyToPolygonF(pts, x1, y1, x1, yc, x2, yc)

                if len(yvals) > 0:
                    pts.append( qt.QPointF(xvals[-1], yvals[-1]) )

        else:
            assert False

        return pts

    def _getBezierLine(self, poly, cliprect):
        """Try to draw a bezier line connecting the points."""

        # clip to a larger box to help the lines get right angle
        bigclip = qt.QRectF(
            cliprect.left()-cliprect.width()*0.5,
            cliprect.top()-cliprect.height()*0.5,
            cliprect.width()*2, cliprect.height()*2)

        # clip poly to the rectangle and return the parts
        polys = qtloops.clipPolyline(bigclip, poly)

        # add each part as a bezier
        path = qt.QPainterPath()
        for lpoly in polys:
            if len(lpoly) >= 2:
                npts = qtloops.bezier_fit_cubic_multi(lpoly, 0.1, len(lpoly)+1)
                qtloops.addCubicsToPainterPath(path, npts);
        return path

    def _drawBezierLine( self, painter, xvals, yvals, posn,
                         xdata, ydata, cliprect ):
        """Handle bezier lines and fills."""

        pts = self._getLinePoints(xvals, yvals, posn, xdata, ydata)
        if len(pts) < 2:
            return
        path = self._getBezierLine(pts, cliprect)
        s = self.settings

        # do filling
        for fillstyle in s.FillBelow, s.FillAbove:
            if not fillstyle.hide:
                x1, y1, x2, y2 = {
                    'top': (pts[0].x(), posn[1], pts[-1].x(), posn[1]),
                    'bottom': (pts[0].x(), posn[3], pts[-1].x(), posn[3]),
                    'left': (posn[0], pts[0].y(), posn[0], pts[-1].y()),
                    'right': (posn[2], pts[0].y(), posn[2], pts[-1].y())
                }[fillstyle.fillto]

                temppath = qt.QPainterPath(path)
                temppath.lineTo(x2, y2)
                temppath.lineTo(x1, y1)
                utils.brushExtFillPath(painter, fillstyle, temppath)

        if not s.PlotLine.hide:
            painter.strokePath(path, s.PlotLine.makeQPen(painter))

    def _drawPlotLine( self, painter, xvals, yvals, posn, xdata, ydata,
                       cliprect ):
        """Draw the line connecting the points."""

        pts = self._getLinePoints(xvals, yvals, posn, xdata, ydata)
        if len(pts) < 2:
            return
        s = self.settings

        # do filling
        for fillstyle in s.FillBelow, s.FillAbove:
            if not fillstyle.hide:
                fillPtsToEdge(painter, pts, posn, cliprect, fillstyle)

        # draw line between points
        if not s.PlotLine.hide:
            painter.setPen( s.PlotLine.makeQPen(painter) )
            utils.plotClippedPolyline(painter, cliprect, pts)

    def drawKeySymbol(self, number, painter, x, y, width, height):
        """Draw the plot symbol and/or line."""

        s = self.settings

        # datasets from document
        xv = s.get('xData').getData(self.document)
        yv = s.get('yData').getData(self.document)

        # whether data has errors
        hasxerrs = xv and xv.hasErrors()
        hasyerrs = yv and yv.hasErrors()

        # convert horizontal errors to vertical ones
        errstyle = s.errorStyle
        if errstyle in ('linehorz', 'fillhorz', 'likehorzbar'):
            errstyle = errstyle.replace('horz', 'vert')
            hasxerrs, hasyerrs = hasyerrs, hasxerrs

        # make some fake error bar data to plot
        yp = y + height/2
        xpts = N.array([x-width, x+width/2, x+2*width])
        ypts = N.array([yp, yp, yp])

        # start drawing
        painter.save()
        cliprect = qt.QRectF(qt.QPointF(x,y), qt.QPointF(x+width,y+height))
        painter.setClipRect(cliprect)

        # draw fill setting
        if not s.FillBelow.hide:
            path = qt.QPainterPath()
            path.addRect(qt.QRectF(
                qt.QPointF(x, yp), qt.QPointF(x+width, yp+height*0.45)))
            utils.brushExtFillPath(painter, s.FillBelow, path)
        if not s.FillAbove.hide:
            path = qt.QPainterPath()
            path.addRect(qt.QRectF(
                qt.QPointF(x, yp), qt.QPointF(x+width, yp-height*0.45)))
            utils.brushExtFillPath(painter, s.FillAbove, path)

        # make points for error bars (if any)
        errorsize = height*0.4
        if xv and hasxerrs:
            xneg = N.array([x-width, x+width/2-errorsize, x+2*width])
            xpos = N.array([x-width, x+width/2+errorsize, x+2*width])
        else:
            xneg = xpos = xpts
        if yv and hasyerrs:
            yneg = N.array([yp-errorsize, yp-errorsize, yp-errorsize])
            ypos = N.array([yp+errorsize, yp+errorsize, yp+errorsize])
        else:
            yneg = ypos = ypts

        # plot error bar
        markersize = s.get('markerSize').convert(painter)
        ebp = ErrorBarDraw(
            errstyle, s.ErrorBarLine, s.FillAbove, s.FillBelow, markersize)
        ebp.plot(painter, xneg, xpos, yneg, ypos, xpts, ypts, cliprect)

        # draw line
        if not s.PlotLine.hide:
            painter.setPen( s.PlotLine.makeQPen(painter) )
            painter.drawLine( qt.QPointF(x, yp), qt.QPointF(x+width, yp) )

        # draw marker
        if not s.MarkerLine.hide or not s.MarkerFill.hide:
            if not s.MarkerFill.hide:
                painter.setBrush( s.MarkerFill.makeQBrush(painter) )

            if not s.MarkerLine.hide:
                painter.setPen( s.MarkerLine.makeQPen(painter) )
            else:
                painter.setPen( qt.QPen( qt.Qt.NoPen ) )

            utils.plotMarker(painter, x+width/2, yp, s.marker, markersize)

        painter.restore()

    def drawLabels(self, painter, xplotter, yplotter,
                   textvals, markersize):
        """Draw labels for the points."""

        s = self.settings
        lab = s.get('Label')

        # work out offset an alignment
        deltax = markersize*1.5*{'left':-1, 'centre':0, 'right':1}[lab.posnHorz]
        deltay = markersize*1.5*{'top':-1, 'centre':0, 'bottom':1}[lab.posnVert]
        alignhorz = {'left':1, 'centre':0, 'right':-1}[lab.posnHorz]
        alignvert = {'top':-1, 'centre':0, 'bottom':1}[lab.posnVert]

        # make font and len
        textpen = lab.makeQPen(painter)
        painter.setPen(textpen)
        font = lab.makeQFont(painter)
        angle = lab.angle

        # iterate over each point and plot each label
        for x, y, t in czip(xplotter+deltax, yplotter+deltay,
                            textvals):
            utils.Renderer(
                painter, font, x, y, t,
                alignhorz, alignvert, angle,
                doc=self.document).render()

    def getAxisLabels(self, direction):
        """Get labels for axis if using a label axis."""

        s = self.settings
        doc = self.document
        text = s.get('labels').getData(doc, checknull=True)
        xv = s.get('xData').getData(doc)
        yv = s.get('yData').getData(doc)

        # handle missing dataset
        if yv and not xv and s.get('xData').isEmpty():
            length = yv.data.shape[0]
            xv = datasets.DatasetRange(length, (1,length))
        elif xv and not yv and s.get('yData').isEmpty():
            length = xv.data.shape[0]
            yv = datasets.DatasetRange(length, (1,length))

        if text is None or xv is None or yv is None:
            return (None, None)
        if direction == 'horizontal':
            return (text, xv.data)
        else:
            return (text, yv.data)

    def _pickable(self, bounds):
        axes = self.fetchAxes()

        if axes is None:
            map_fn = None
        else:
            map_fn = lambda x, y: (
                axes[0].dataToPlotterCoords(bounds, x),
                axes[1].dataToPlotterCoords(bounds, y) )

        return pickable.DiscretePickable(self, 'xData', 'yData', map_fn)

    def pickPoint(self, x0, y0, bounds, distance = 'radial'):
        return self._pickable(bounds).pickPoint(x0, y0, bounds, distance)

    def pickIndex(self, oldindex, direction, bounds):
        return self._pickable(bounds).pickIndex(oldindex, direction, bounds)

    def getColorbarParameters(self):
        """Return parameters for colorbar."""
        s = self.settings
        c = s.Color
        return (c.min, c.max, c.scaling, s.MarkerFill.colorMap, 0,
                s.MarkerFill.colorMapInvert)

    def dataDraw(self, painter, axes, posn, cliprect):
        """Plot the data on a plotter."""

        # get data
        s = self.settings
        doc = self.document
        xv = s.get('xData').getData(doc)
        yv = s.get('yData').getData(doc)
        text = s.get('labels').getData(doc, checknull=True)
        scalepoints = s.get('scalePoints').getData(doc)
        colorpoints = s.Color.get('points').getData(doc)

        # if a missing dataset, make a fake dataset for the second one
        # based on a row number
        if xv and not yv and s.get('yData').isEmpty():
            # use index for y data
            length = xv.data.shape[0]
            yv = datasets.DatasetRange(length, (1,length))
        elif yv and not xv and s.get('xData').isEmpty():
            # use index for x data
            length = yv.data.shape[0]
            xv = datasets.DatasetRange(length, (1,length))
        if not xv or not yv:
            # no valid dataset, so exit
            return

        # if text entered, then multiply up to get same number of values
        # as datapoints
        if text:
            length = min( len(xv.data), len(yv.data) )
            text = text*(length // len(text)) + text[:length % len(text)]

        # loop over chopped up values
        for xvals, yvals, tvals, ptvals, cvals in (
            datasets.generateValidDatasetParts(
                [xv, yv, text, scalepoints, colorpoints])):

            #print "Calculating coordinates"
            # calc plotter coords of x and y points
            xplotter = axes[0].dataToPlotterCoords(posn, xvals.data)
            yplotter = axes[1].dataToPlotterCoords(posn, yvals.data)

            # points are plotted offset in shift-points modes
            if s.PlotLine.steps != 'off':
                xpltpoint = N.array(xplotter)
                if s.PlotLine.steps == 'right-shift-points':
                    xpltpoint[1:] = 0.5*(xplotter[:-1] + xplotter[1:])
                elif s.PlotLine.steps == 'left-shift-points':
                    xpltpoint[:-1] = 0.5*(xplotter[:-1] + xplotter[1:])
            else:
                xpltpoint = xplotter
            ypltpoint = yplotter

            # plot filled error bars
            if s.errorStyle in ('fillvert', 'fillhorz'):
                # filled region errors are painted first
                self._plotErrors(
                    posn, painter, xpltpoint, ypltpoint,
                    axes, xvals, yvals, cliprect)

            #print "Painting plot line"
            # plot data line (and/or filling above or below)
            if not s.PlotLine.hide or not s.FillAbove.hide or not s.FillBelow.hide:
                if s.PlotLine.bezierJoin:
                    self._drawBezierLine(
                        painter, xplotter, yplotter, posn,
                        xvals, yvals, cliprect )
                else:
                    self._drawPlotLine(
                        painter, xplotter, yplotter, posn,
                        xvals, yvals, cliprect )

            #print "Painting error bars"
            # plot normal errors bars
            if s.errorStyle not in ('fillvert', 'fillhorz'):
                # normally the error bar is painted after the line
                self._plotErrors(posn, painter, xpltpoint, ypltpoint,
                                 axes, xvals, yvals, cliprect)

            # plot the points (we do this last so they are on top)
            markersize = s.get('markerSize').convert(painter)
            if not s.MarkerLine.hide or not s.MarkerFill.hide:

                #print "Painting marker fill"
                if not s.MarkerFill.hide:
                    # filling for markers
                    painter.setBrush( s.MarkerFill.makeQBrush(painter) )
                else:
                    # no-filling brush
                    painter.setBrush( qt.QBrush() )

                #print "Painting marker lines"
                if not s.MarkerLine.hide:
                    # edges of markers
                    painter.setPen( s.MarkerLine.makeQPen(painter) )
                else:
                    # invisible pen
                    painter.setPen( qt.QPen(qt.Qt.NoPen) )

                # thin datapoints as required
                if s.thinfactor <= 1:
                    xplt, yplt = xpltpoint, ypltpoint
                else:
                    xplt, yplt = (
                        xpltpoint[::s.thinfactor],
                        ypltpoint[::s.thinfactor])

                # whether to scale markers
                scaling = colorvals = cmap = None
                if ptvals:
                    scaling = ptvals.data
                    if s.thinfactor > 1:
                        scaling = scaling[::s.thinfactor]

                # color point individually
                cmapname = s.MarkerFill.colorMap
                if cvals and not s.MarkerFill.hide and cmapname != 'none':
                    colorvals = utils.applyScaling(
                        cvals.data, s.Color.scaling,
                        s.Color.min, s.Color.max)
                    if s.thinfactor > 1:
                        colorvals = colorvals[::s.thinfactor]
                    cmap = self.document.evaluate.getColormap(
                        cmapname, s.MarkerFill.colorMapInvert)

                # actually plot datapoints
                utils.plotMarkers(
                    painter, xplt, yplt, s.marker, markersize,
                    scaling=scaling, clip=cliprect,
                    cmap=cmap, colorvals=colorvals,
                    scaleline=s.MarkerLine.scaleLine)

            # finally plot any labels
            if tvals and not s.Label.hide:
                self.drawLabels(
                    painter, xpltpoint, ypltpoint,
                    tvals, markersize)

# allow the factory to instantiate an x,y plotter
document.thefactory.register( PointPlotter )
