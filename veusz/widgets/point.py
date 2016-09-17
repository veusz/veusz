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

"""For plotting xy points.

Future expansion:

Plots set of datasets
 Sets: defined by
   expression: data_*
   tags
   list of datasets in a dataset
   gui chooser
 Single:
   real dataset
   data expression
   filters


Current dataset formats:
expr, expr, expr
expr, ...
(expr, ...)
datasetname
`dataset name`

future formats:
: y*
:y1:y2:y3


"""

from __future__ import division
import numpy as N

from ..compat import czip, crange
from .. import qtall as qt4
from .. import document
from .. import setting
from .. import utils

from . import pickable
from .plotters import GenericPlotter

try:
    from ..helpers import qtloops
    hasqtloops = True
except ImportError:
    hasqtloops = False

def _(text, disambiguation=None, context='XY'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

# functions for plotting error bars
# different styles are made up of combinations of these functions
# each function takes the same arguments
def _errorBarsBar(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                  s, painter, clip):
    """Draw bar style error lines."""
    # vertical error bars
    if ymin is not None and ymax is not None and not s.ErrorBarLine.hideVert:
        utils.plotLinesToPainter(painter, xplotter, ymin, xplotter, ymax, clip)

    # horizontal error bars
    if xmin is not None and xmax is not None and not s.ErrorBarLine.hideHorz:
        utils.plotLinesToPainter(painter, xmin, yplotter, xmax, yplotter, clip)

def _errorBarsEnds(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                   s, painter, clip):
    """Draw perpendiclar ends on error bars."""
    size = ( s.get('markerSize').convert(painter) *
             s.ErrorBarLine.endsize )

    if ymin is not None and ymax is not None and not s.ErrorBarLine.hideVert:
        utils.plotLinesToPainter(
            painter, xplotter-size, ymin,
            xplotter+size, ymin, clip)
        utils.plotLinesToPainter(
            painter, xplotter-size, ymax,
            xplotter+size, ymax, clip)

    if xmin is not None and xmax is not None and not s.ErrorBarLine.hideHorz:
        utils.plotLinesToPainter(
            painter, xmin, yplotter-size,
            xmin, yplotter+size, clip)
        utils.plotLinesToPainter(
            painter, xmax, yplotter-size,
            xmax, yplotter+size, clip)

def _errorBarsBox(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                  s, painter, clip):
    """Draw box around error region."""
    if utils.allNotNone(xmin, xmax, ymin, ymax):
        painter.setBrush( qt4.QBrush() )
        utils.plotBoxesToPainter(painter, xmin, ymin, xmax, ymax, clip)

def _errorBarsBoxFilled(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                        s, painter, clip):
    """Draw box filled region inside error bars."""
    if utils.allNotNone(xmin, xmax, ymin, ymax):
        # filled region below
        if not s.FillBelow.hideerror:
            path = qt4.QPainterPath()
            utils.addNumpyPolygonToPath(
                path, clip,
                xmin, ymin, xmin, yplotter,
                xmax, yplotter, xmax, ymin)
            utils.brushExtFillPath(painter, s.FillBelow, path, ignorehide=True)

        # filled region above
        if not s.FillAbove.hideerror:
            path = qt4.QPainterPath()
            utils.addNumpyPolygonToPath(
                path, clip,
                xmin, yplotter, xmax, yplotter,
                xmax, ymax, xmin, ymax)
            utils.brushExtFillPath(painter, s.FillAbove, path, ignorehide=True)

def _errorBarsDiamond(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                      s, painter, clip):
    """Draw diamond around error region."""
    if utils.allNotNone(xmin, xmax, ymin, ymax):
        # expand clip by pen width (urgh)
        pw = painter.pen().widthF()*2
        clip = qt4.QRectF(
            qt4.QPointF(clip.left()-pw,clip.top()-pw),
            qt4.QPointF(clip.right()+pw,clip.bottom()+pw))

        path = qt4.QPainterPath()
        utils.addNumpyPolygonToPath(
            path, clip,
            xmin, yplotter, xplotter, ymax,
            xmax, yplotter, xplotter, ymin)
        painter.setBrush( qt4.QBrush() )
        painter.drawPath(path)

def _errorBarsDiamondFilled(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                            s, painter, clip):
    """Draw diamond filled region inside error bars."""
    if utils.allNotNone(xmin, xmax, ymin, ymax):
        if not s.FillBelow.hideerror:
            path = qt4.QPainterPath()
            utils.addNumpyPolygonToPath(
                path, clip,
                xmin, yplotter, xplotter, ymin,
                xmax, yplotter)
            utils.brushExtFillPath(painter, s.FillBelow, path, ignorehide=True)

        if not s.FillAbove.hideerror:
            path = qt4.QPainterPath()
            utils.addNumpyPolygonToPath(
                path, clip,
                xmin, yplotter, xplotter, ymax,
                xmax, yplotter)
            utils.brushExtFillPath(painter, s.FillAbove, path, ignorehide=True)

def _errorBarsCurve(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                    s, painter, clip):
    """Draw curve around error region."""
    if utils.allNotNone(xmin, xmax, ymin, ymax):
        # non-filling brush
        painter.setBrush( qt4.QBrush() )

        for xp, yp, xmn, ymn, xmx, ymx in czip(
            xplotter, yplotter, xmin, ymin, xmax, ymax):

            p = qt4.QPainterPath()
            p.moveTo(xp + (xmx-xp), yp)
            p.arcTo(qt4.QRectF(
                xp - (xmx-xp), yp - (yp-ymx),
                (xmx-xp)*2, (yp-ymx)*2), 0., 90.)
            p.arcTo(qt4.QRectF(
                xp - (xp-xmn), yp - (yp-ymx),
                (xp-xmn)*2, (yp-ymx)*2), 90., 90.)
            p.arcTo(qt4.QRectF(
                xp - (xp-xmn), yp - (ymn-yp),
                (xp-xmn)*2, (ymn-yp)*2), 180., 90.)
            p.arcTo(qt4.QRectF(
                xp - (xmx-xp), yp - (ymn-yp),
                (xmx-xp)*2, (ymn-yp)*2), 270., 90.)
            painter.drawPath(p)

def _errorBarsCurveFilled(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                          s, painter, clip):
    """Fill area around error region."""

    if utils.allNotNone(xmin, xmax, ymin, ymax):
        for xp, yp, xmn, ymn, xmx, ymx in czip(
            xplotter, yplotter, xmin, ymin, xmax, ymax):

            if not s.FillAbove.hideerror:
                p = qt4.QPainterPath()
                p.moveTo(xp + (xmx-xp), yp)
                p.arcTo(qt4.QRectF(
                    xp - (xmx-xp), yp - (yp-ymx),
                    (xmx-xp)*2, (yp-ymx)*2), 0., 90.)
                p.arcTo(qt4.QRectF(
                    xp - (xp-xmn), yp - (yp-ymx),
                    (xp-xmn)*2, (yp-ymx)*2), 90., 90.)
                utils.brushExtFillPath(painter, s.FillAbove, p, ignorehide=True)

            if not s.FillBelow.hideerror:
                p = qt4.QPainterPath()
                p.moveTo(xp + (xp-xmn), yp)
                p.arcTo(qt4.QRectF(
                    xp - (xp-xmn), yp - (ymn-yp),
                    (xp-xmn)*2, (ymn-yp)*2), 180., 90.)
                p.arcTo(qt4.QRectF(
                    xp - (xmx-xp), yp - (ymn-yp),
                    (xmx-xp)*2, (ymn-yp)*2), 270., 90.)
                utils.brushExtFillPath(painter, s.FillBelow, p, ignorehide=True)

def _errorBarsFilled(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                     s, painter, clip):
    """Draw filled region as error region."""

    ptsabove = qt4.QPolygonF()
    ptsbelow = qt4.QPolygonF()

    hidevert = True  # keep track of what's shown
    hidehorz = True
    if ( 'vert' in style and
         (ymin is not None and ymax is not None) and
         not s.ErrorBarLine.hideVert ):
        hidevert = False
        # lines above/below points
        utils.addNumpyToPolygonF(ptsbelow, xplotter, ymin)
        utils.addNumpyToPolygonF(ptsabove, xplotter, ymax)

    elif ( 'horz' in style and
           (xmin is not None and xmax is not None) and
           not s.ErrorBarLine.hideHorz ):
        hidehorz = False
        # lines left/right points
        utils.addNumpyToPolygonF(ptsbelow, xmin, yplotter)
        utils.addNumpyToPolygonF(ptsabove, xmax, yplotter)

    # draw filled regions above/left and below/right
    if 'fill' in style and not (hidehorz and hidevert):
        # construct points for error bar regions
        retnpts = qt4.QPolygonF()
        utils.addNumpyToPolygonF(retnpts, xplotter[::-1], yplotter[::-1])

        # polygons consist of lines joining the points and continuing
        # back along the plot line (retnpts)
        if not s.FillBelow.hideerror:
            utils.brushExtFillPolygon(
                painter, s.FillBelow, clip,
                ptsbelow+retnpts, ignorehide=True)
        if not s.FillAbove.hideerror:
            utils.brushExtFillPolygon(
                painter, s.FillAbove, clip,
                ptsabove+retnpts, ignorehide=True)

    # draw optional line (on top of fill)
    utils.plotClippedPolyline(painter, clip, ptsabove)
    utils.plotClippedPolyline(painter, clip, ptsbelow)

# map error bar names to lists of functions (above)
_errorBarFunctionMap = {
    'none': (),
    'bar': (_errorBarsBar,),
    'bardiamond': (_errorBarsBar, _errorBarsDiamond,),
    'barcurve': (_errorBarsBar, _errorBarsCurve,),
    'barbox': (_errorBarsBar, _errorBarsBox,),
    'barends': (_errorBarsBar, _errorBarsEnds,),
    'box':  (_errorBarsBox,),
    'boxfill': (_errorBarsBoxFilled, _errorBarsBox,),
    'diamond':  (_errorBarsDiamond,),
    'diamondfill':  (_errorBarsDiamond, _errorBarsDiamondFilled),
    'curve': (_errorBarsCurve,),
    'curvefill': (_errorBarsCurveFilled, _errorBarsCurve,),
    'fillhorz': (_errorBarsFilled,),
    'fillvert': (_errorBarsFilled,),
    'linehorz': (_errorBarsFilled,),
    'linevert': (_errorBarsFilled,),
    'linehorzbar': (_errorBarsBar, _errorBarsFilled),
    'linevertbar': (_errorBarsBar, _errorBarsFilled),
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

    polypts = qt4.QPolygonF([qt4.QPointF(x1, y1)])
    polypts += pts
    polypts.append(qt4.QPointF(x2, y2))

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

class DatasetPartPlot:
    """Encapsulate drawing of dataset as a class."""

    def __init__(self, painter, axes, posn, cliprect, document, settings, datasets):
        self.painter = painter
        self.axes = axes
        self.posn = posn
        self.cliprect = cliprect
        self.document = document
        s = self.settings = settings
        self.xdata = datasets[0]
        self.ydata = datasets[1]
        self.labeldata = datasets[2]
        self.scaledata = datasets[3]
        self.colordata = datasets[4]

        xplotter = self.xplotter = axes[0].dataToPlotterCoords(
            posn, self.xdata.data)
        yplotter = self.yplotter = axes[1].dataToPlotterCoords(
            posn, self.ydata.data)

        self.markersize = s.get('markerSize').convert(painter)

        # xpltpoint and ypltpoint (position of points) are offset in
        # shift-points modes
        if s.PlotLine.steps != 'off':
            self.xpltpoint = N.array(xplotter)
            if s.PlotLine.steps == 'right-shift-points':
                self.xpltpoint[1:] = 0.5*(xplotter[:-1] + xplotter[1:])
            elif s.PlotLine.steps == 'left-shift-points':
                self.xpltpoint[:-1] = 0.5*(xplotter[:-1] + xplotter[1:])
        else:
            self.xpltpoint = xplotter
        self.ypltpoint = yplotter

    def plot(self):
        """Do all the steps of plotting."""

        s = self.settings

        # plot filled error bars
        if s.errorStyle in ('fillvert', 'fillhorz'):
            # filled region errors are painted first
            self.plotErrors()

        #print "Painting plot line"
        # plot data line (and/or filling above or below)
        if not s.PlotLine.hide or not s.FillAbove.hide or not s.FillBelow.hide:
            if s.PlotLine.bezierJoin and hasqtloops:
                self.plotBezierLine()
            else:
                self.plotPlotLine()

        #print "Painting error bars"
        # plot normal errors bars
        if s.errorStyle not in ('fillvert', 'fillhorz'):
            # normally the error bar is painted after the line
            self.plotErrors()

        if not s.MarkerLine.hide or not s.MarkerFill.hide:
            self.plotMarkers()

        # finally plot any labels
        if self.labeldata and not s.Label.hide:
            self.plotLabels()

    def plotErrors(self):
        """Plot error bars (horizontal and vertical)."""

        s = self.settings
        style = s.errorStyle
        if style == 'none':
            return

        # optional thinning of error bars plotted
        thin = s.errorthin

        # default is no error bars
        xmin = xmax = ymin = ymax = None

        # draw horizontal error bars
        if self.xdata.hasErrors():
            xmin, xmax = self.xdata.getPointRanges()
            if thin>1:
                xmin, xmax = xmin[::thin], xmax[::thin]

            # convert xmin and xmax to graph coordinates
            xmin = self.axes[0].dataToPlotterCoords(self.posn, xmin)
            xmax = self.axes[0].dataToPlotterCoords(self.posn, xmax)

        # draw vertical error bars
        if self.ydata.hasErrors():
            ymin, ymax = self.ydata.getPointRanges()
            if thin>1:
                ymin, ymax = ymin[::thin], ymax[::thin]

            # convert ymin and ymax to graph coordinates
            ymin = self.axes[1].dataToPlotterCoords(self.posn, ymin)
            ymax = self.axes[1].dataToPlotterCoords(self.posn, ymax)

        # no error bars - break out of processing below
        if ymin is None and ymax is None and xmin is None and xmax is None:
            return

        xplotter, yplotter = self.xplotter, self.yplotter
        if thin>1:
            xplotter, yplotter = xplotter[::thin], yplotter[::thin]

        # iterate to call the error bars functions required to draw style
        pen = s.ErrorBarLine.makeQPenWHide(self.painter)
        pen.setCapStyle(qt4.Qt.FlatCap)

        self.painter.setPen(pen)
        for function in _errorBarFunctionMap[style]:
            function(
                style, xmin, xmax, ymin, ymax,
                xplotter, yplotter, s, self.painter, self.cliprect)

    def plotMarkers(self):
        """Plot the markers (done last so on top)."""

        s = self.settings

        #print "Painting marker fill"
        if not s.MarkerFill.hide:
            self.painter.setBrush( s.MarkerFill.makeQBrush() )
        else:
            self.painter.setBrush( qt4.QBrush() )

        #print "Painting marker lines"
        if not s.MarkerLine.hide:
            self.painter.setPen( s.MarkerLine.makeQPen(self.painter) )
        else:
            self.painter.setPen( qt4.QPen(qt4.Qt.NoPen) )

        # thin datapoints as required
        if s.thinfactor <= 1:
            xplt, yplt = self.xpltpoint, self.ypltpoint
        else:
            xplt, yplt = (
                self.xpltpoint[::s.thinfactor],
                self.ypltpoint[::s.thinfactor])

        # optional attributes
        scaling = colorvals = cmap = None

        # whether to scale markers
        if self.scaledata:
            scaling = self.scaledata.data
            if s.thinfactor > 1:
                scaling = scaling[::s.thinfactor]

        # whether to color point individually
        if self.colordata and not s.MarkerFill.hide:
            colorvals = utils.applyScaling(
                self.colordata.data, s.Color.scaling,
                s.Color.min, s.Color.max)
            if s.thinfactor > 1:
                colorvals = colorvals[::s.thinfactor]
            cmap = self.document.getColormap(
                s.MarkerFill.colorMap, s.MarkerFill.colorMapInvert)

        # actually plot datapoints
        utils.plotMarkers(
            self.painter, xplt, yplt, s.marker,
            self.markersize, scaling=scaling, clip=self.cliprect,
            cmap=cmap, colorvals=colorvals,
            scaleline=s.MarkerLine.scaleLine)

    def plotPlotLine(self):
        """Draw the line connecting the points."""

        pts = self.getLinePoints()
        if len(pts) < 2:
            return
        s = self.settings

        # do filling
        for fillstyle in s.FillBelow, s.FillAbove:
            if not fillstyle.hide:
                fillPtsToEdge(
                    self.painter, pts, self.posn, self.cliprect, fillstyle)

        # draw line between points
        if not s.PlotLine.hide:
            self.painter.setPen(s.PlotLine.makeQPen(self.painter))
            utils.plotClippedPolyline(self.painter, self.cliprect, pts)

    def getLinePoints(self):
        """Return points of line as a QPolygonF."""

        pts = qt4.QPolygonF()

        s = self.settings
        steps = s.PlotLine.steps

        # simple continuous line
        if steps == 'off':
            utils.addNumpyToPolygonF(pts, self.xplotter, self.yplotter)

        # stepped line, with points on left
        elif steps[:4] == 'left':
            x1 = self.xplotter[:-1]
            x2 = self.xplotter[1:]
            y1 = self.yplotter[:-1]
            y2 = self.yplotter[1:]
            utils.addNumpyToPolygonF(pts, x1, y1, x2, y1, x2, y2)

        # stepped line, with points on right
        elif steps[:5] == 'right':
            x1 = self.xplotter[:-1]
            x2 = self.xplotter[1:]
            y1 = self.yplotter[:-1]
            y2 = self.yplotter[1:]
            utils.addNumpyToPolygonF(pts, x1, y1, x1, y2, x2, y2)

        # stepped line, with points in centre
        # this is complex as we can't use the mean of the plotter coords,
        #  as the axis could be log
        elif steps[:6] == 'centre':
            if self.xdata.hasErrors():
                # Special case if error bars on x points:
                # here we use the error bars to define the steps
                xmin, xmax = self.xdata.getPointRanges()

                # this is duplicated from drawing error bars: bad
                # convert xmin and xmax to graph coordinates
                xmin = self.axes[0].dataToPlotterCoords(self.posn, xmin)
                xmax = self.axes[0].dataToPlotterCoords(self.posn, xmax)
                utils.addNumpyToPolygonF(
                    pts, xmin, self.yplotter, xmax, self.yplotter)

            else:
                # we put the bin edges half way between the points
                # we assume this is the correct thing to do even in log space
                x1 = self.xplotter[:-1]
                x2 = self.xplotter[1:]
                y1 = self.yplotter[:-1]
                y2 = self.yplotter[1:]
                xc = 0.5*(x1+x2)
                utils.addNumpyToPolygonF(pts, x1, y1, xc, y1, xc, y2)

                if len(self.xplotter) > 0:
                    pts.append(
                        qt4.QPointF(self.xplotter[-1], self.yplotter[-1]))

        elif steps[:7] == 'vcentre':
            if self.ydata.hasErrors():
                # Special case if error bars on y points:
                # here we use the error bars to define the steps
                ymin, ymax = self.ydata.getPointRanges()

                # this is duplicated from drawing error bars: bad
                # convert ymin and ymax to graph coordinates
                ymin = self.axes[1].dataToPlotterCoords(self.posn, ymin)
                ymax = self.axes[1].dataToPlotterCoords(self.posn, ymax)
                utils.addNumpyToPolygonF(
                    pts, self.xplotter, ymin, self.xplotter, ymax)

            else:
                # we put the bin edges half way between the points
                # we assume this is the correct thing to do even in log space
                y1 = self.yplotter[:-1]
                y2 = self.yplotter[1:]
                x1 = self.xplotter[:-1]
                x2 = self.xplotter[1:]
                yc = 0.5*(y1+y2)
                utils.addNumpyToPolygonF(pts, x1, y1, x1, yc, x2, yc)

                if len(self.yplotter) > 0:
                    pts.append(
                        qt4.QPointF(self.xplotter[-1], self.yplotter[-1]) )

        else:
            raise RuntimeError('Invalid step mode')

        return pts

    def getBezierLine(self, poly):
        """Try to draw a bezier line connecting the points."""

        # clip to a larger box to help the lines get right angle
        bigclip = qt4.QRectF(
            self.cliprect.left()-self.cliprect.width()*0.5,
            self.cliprect.top()-self.cliprect.height()*0.5,
            self.cliprect.width()*2, self.cliprect.height()*2)

        # clip poly to the rectangle and return the parts
        polys = qtloops.clipPolyline(bigclip, poly)

        # add each part as a bezier
        path = qt4.QPainterPath()
        for lpoly in polys:
            if len(lpoly) >= 2:
                npts = qtloops.bezier_fit_cubic_multi(lpoly, 0.1, len(lpoly)+1)
                qtloops.addCubicsToPainterPath(path, npts);
        return path

    def plotBezierLine(self):
        """Handle bezier lines and fills."""

        pts = self.getLinePoints()
        if len(pts) < 2:
            return
        path = self.getBezierLine(pts)
        s = self.settings

        # do filling
        for fillstyle in s.FillBelow, s.FillAbove:
            if not fillstyle.hide:
                x1, y1, x2, y2 = {
                    'top': (
                        pts[0].x(), self.posn[1], pts[-1].x(), self.posn[1]),
                    'bottom': (
                        pts[0].x(), self.posn[3], pts[-1].x(), self.posn[3]),
                    'left': (
                        self.posn[0], pts[0].y(), self.posn[0], pts[-1].y()),
                    'right': (
                        self.posn[2], pts[0].y(), self.posn[2], pts[-1].y())
                }[fillstyle.fillto]

                temppath = qt4.QPainterPath(path)
                temppath.lineTo(x2, y2)
                temppath.lineTo(x1, y1)
                utils.brushExtFillPath(self.painter, fillstyle, temppath)

        if not s.PlotLine.hide:
            self.painter.strokePath(path, s.PlotLine.makeQPen(self.painter))

    def plotLabels(self):
        """Plot labels for the points."""

        s = self.settings
        lab = s.get('Label')

        # work out offset an alignment
        deltax = self.markersize*1.5*{
            'left':-1, 'centre':0, 'right':1}[lab.posnHorz]
        deltay = self.markersize*1.5*{
            'top':-1, 'centre':0, 'bottom':1}[lab.posnVert]
        alignhorz = {'left':1, 'centre':0, 'right':-1}[lab.posnHorz]
        alignvert = {'top':-1, 'centre':0, 'bottom':1}[lab.posnVert]

        # make font and len
        textpen = lab.makeQPen()
        self.painter.setPen(textpen)
        font = lab.makeQFont(self.painter)
        angle = lab.angle

        # iterate over each point and plot each label
        for x, y, t in czip(
                self.xplotter+deltax, self.yplotter+deltay,
                self.labeldata):

            utils.Renderer(
                self.painter, font, x, y, t,
                alignhorz, alignvert, angle,
                doc=self.document).render()

def cycledatasets(*datasets):
    """Cycle datasets, repeating if necessary."""
    maxlen = max([len(d) for d in datasets])
    for i in crange(maxlen):
        out = []
        for d in datasets:
            if len(d) == 0:
                out.append(None)
            else:
                out.append(d[i % len(d)])
        yield tuple(out)

class PointPlotter(GenericPlotter):
    """A class for plotting points and their errors."""

    typename='xy'
    allowusercreation=True
    description=_('Plot points with lines and errorbars')

    def __init__(self, parent, name=None):
        """Initialise XY plotter plotting (xdata, ydata).

        xdata and ydata are strings specifying the data in the document"""

        GenericPlotter.__init__(self, parent, name=name)
        if type(self) == PointPlotter:
            self.readDefaults()

        # datasets are cached for speed
        self._cachechangeset = -1
        self._cache_ds = None

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        GenericPlotter.addSettings(s)

        # non-formatting
        s.add( setting.DatasetExtendedMulti(
            'yData', 'y',
            descr=_('Y values, given by dataset, expression or list of values'),
            usertext=_('Y data')), 0 )
        s.add( setting.DatasetExtendedMulti(
            'xData', 'x',
            descr=_('X values, given by dataset, expression or list of values'),
            usertext=_('X data')), 0 )
        s.add( setting.DatasetOrStr(
            'labels', '',
            descr=_('Dataset or string to label points'),
            usertext=_('Labels')), 4 )
        s.add( setting.DatasetExtendedMulti(
            'scalePoints', '',
            descr = _('Scale size of markers given by dataset, expression'
                      ' or list of values'),
            usertext=_('Scale markers')), 5 )
        s.add( setting.MarkerColor('Color'), 6 )

        s.add( setting.Transform(
            'transform', '',
            descr=_('Transforms plotted data'),
            usertext=_('Transform')), 7)

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
            'black',
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

        s.add( setting.ErrorStyle(
            'errorStyle',
            'bar',
            descr=_('Style of error bars to plot'),
            usertext=_('Error style'), formatting=True) )

        s.add( setting.XYPlotLine(
            'PlotLine',
            descr = _('Plot line settings'),
            usertext = _('Plot line')),
               pixmap = 'settings_plotline' )

        s.add( setting.MarkerLine(
            'MarkerLine',
            descr = _('Line around the marker settings'),
            usertext = _('Marker border')),
               pixmap = 'settings_plotmarkerline' )
        s.add( MarkerFillBrush(
            'MarkerFill',
            descr = _('Marker fill settings'),
            usertext = _('Marker fill')),
               pixmap = 'settings_plotmarkerfill' )

        s.add( setting.ErrorBarLine(
            'ErrorBarLine',
            descr = _('Error bar line settings'),
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
            descr = _('Fill mode 2'),
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

    def _updateDatasetCache(self):
        """Check whether datasets needs updating due to doc change.

        This applies transforms, etc, so the calculations don't need
        repeating.
        """

        if self._cachechangeset == self.document.changeset:
            return
        self._cachechangeset = self.document.changeset

        s = self.settings
        doc = self.document
        xds = s.get('xData').getDatasets(doc)
        yds = s.get('yData').getDatasets(doc)
        text = s.get('labels').getData(doc, checknull=True)
        text = [] if text is None else [text]
        scaleds = s.get('scalePoints').getDatasets(doc)
        colords = s.Color.get('points').getData(doc)
        colords = [] if colords is None else [colords]

        # automatic indexing of dataset is the other is blank
        if xds and not yds and s.get('yData').isEmpty():
            yds = []
            for ds in xds:
                length = ds.data.shape[0]
                yds.append(document.DatasetRange(length, (1,length)))
        elif yds and not xds and s.get('xData').isEmpty():
            xds = []
            for ds in yds:
                length = ds.data.shape[0]
                xds.append(document.DatasetRange(length, (1,length)))

        # cache contains tuples of datasets
        self._cache_ds = []
        if xds and yds:
            for xd, yd, td, sd, cd in cycledatasets(xds, yds, text, scaleds, colords):
                # handle repeating labels
                if td is not None:
                    mlen = min(len(xd), len(yd))
                    td = td*(mlen // len(td)) + td[:mlen % len(td)]
                self._cache_ds.append((xd,yd,td,sd,cd))

        # transform datasets into new datasets
        if s.transform:
            transds = []
            for ds in self._cache_ds:
                retn = self.document.transform.evalExpr(s.transform, ds)
                transds.append(ds if retn is None else retn)
            self._cache_ds = transds

    def affectsAxisRange(self):
        """This widget provides range information about these axes."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def getRange(self, axis, depname, axrange):
        """Compute the effect of data on the axis range."""

        islog = axis.settings.log
        def updateRange(v):
            """Update range based on data."""
            filt = N.isfinite(v)
            if islog:
                filt &= v>0
            fvals = v[filt]
            if len(fvals) > 0:
                axrange[0] = min(axrange[0], fvals.min())
                axrange[1] = max(axrange[1], fvals.max())

        # this is the index into the cached returned datasets
        idx = {'sx': 0, 'sy': 1}[depname]

        # update range using each entry in cache
        self._updateDatasetCache()
        for datasets in self._cache_ds:
            datasets[idx].rangeVisit(updateRange)

    def getNumberKeys(self):
        """Return number of key entries."""
        self._updateDatasetCache()

        if self.settings.key:
            return len(self._cache_ds)
        else:
            return 0

    def drawKeySymbol(self, number, painter, x, y, width, height):
        """Draw the plot symbol and/or line."""

        self._updateDatasetCache()
        xdata, ydata, labeldata, scaledata, colordata = self._cache_ds[number]

        painter.save()
        cliprect = qt4.QRectF(qt4.QPointF(x,y), qt4.QPointF(x+width,y+height))
        painter.setClipRect(cliprect)

        # draw sample error bar
        s = self.settings
        size = s.get('markerSize').convert(painter)
        style = s.errorStyle

        # make some fake error bar data to plot
        yp = y + height/2
        xpts = N.array([x-width, x+width/2, x+2*width])
        ypts = N.array([yp, yp, yp])

        # size of error bars in key
        errorsize = height*0.4

        # make points for error bars (if any)
        if xdata and xdata.hasErrors():
            xneg = N.array([x-width, x+width/2-errorsize, x+2*width])
            xpos = N.array([x-width, x+width/2+errorsize, x+2*width])
        else:
            xneg = xpos = xpts
        if ydata and ydata.hasErrors():
            yneg = N.array([yp-errorsize, yp-errorsize, yp-errorsize])
            ypos = N.array([yp+errorsize, yp+errorsize, yp+errorsize])
        else:
            yneg = ypos = ypts

        # plot error bar
        painter.setPen( s.ErrorBarLine.makeQPenWHide(painter) )
        for function in _errorBarFunctionMap[style]:
            function(style, xneg, xpos, yneg, ypos, xpts, ypts, s,
                     painter, cliprect)

        # draw line
        if not s.PlotLine.hide:
            painter.setPen( s.PlotLine.makeQPen(painter) )
            painter.drawLine( qt4.QPointF(x, yp), qt4.QPointF(x+width, yp) )

        # draw marker
        if not s.MarkerLine.hide or not s.MarkerFill.hide:
            if not s.MarkerFill.hide:
                painter.setBrush( s.MarkerFill.makeQBrush() )

            if not s.MarkerLine.hide:
                painter.setPen( s.MarkerLine.makeQPen(painter) )
            else:
                painter.setPen( qt4.QPen( qt4.Qt.NoPen ) )

            utils.plotMarker(painter, x+width/2, yp, s.marker, size)

        painter.restore()

    def getAxisLabels(self, direction):
        """Get labels for axis if using a label axis."""

        self._updateDatasetCache()

        # use x or y dataset as appropriate
        dsidx = 0 if direction=='horizontal' else 1

        # build up text and positions for each dataset
        text = []
        posns = []
        for datasets in self._cache_ds:
            if datasets[2]: # label
                posns.append(datasets[dsidx].data)
                # merge text
                text += datasets[2]

        # nothing to plot
        if not text or not posns:
            return (None, None)

        # merge positions into single list
        posns = N.hstack(posns)

        return (text, posns)

    def _pickable(self, bounds):
        axes = self.fetchAxes()

        # FIXME: for multiple datasets

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
        return (
            c.min, c.max, c.scaling, s.MarkerFill.colorMap, 0,
            s.MarkerFill.colorMapInvert)

    def dataDraw(self, painter, axes, posn, cliprect):
        """Plot the data on a plotter."""

        self._updateDatasetCache()

        for datasets in self._cache_ds:
            for dsparts in document.generateValidDatasetParts(datasets):
                dd = DatasetPartPlot(
                    painter,axes, posn, cliprect,
                    self.document, self.settings, dsparts)
                dd.plot()

# allow the factory to instantiate an x,y plotter
document.thefactory.register(PointPlotter)
