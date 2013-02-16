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

import veusz.qtall as qt4
import itertools
import numpy as N

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import pickable
from plotters import GenericPlotter

try:
    import veusz.helpers.qtloops as qtloops
    hasqtloops = True
except ImportError:
    hasqtloops = False

def _(text, disambiguation=None, context='XY'):
    """Translate text."""
    return unicode(
        qt4.QCoreApplication.translate(context, text, disambiguation))

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
        utils.plotLinesToPainter(painter, xplotter-size, ymin,
                                 xplotter+size, ymin, clip)
        utils.plotLinesToPainter(painter, xplotter-size, ymax,
                                 xplotter+size, ymax, clip)

    if xmin is not None and xmax is not None and not s.ErrorBarLine.hideHorz:
        utils.plotLinesToPainter(painter, xmin, yplotter-size,
                                 xmin, yplotter+size, clip)
        utils.plotLinesToPainter(painter, xmax, yplotter-size,
                                 xmax, yplotter+size, clip)

def _errorBarsBox(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                  s, painter, clip):
    """Draw box around error region."""
    if None not in (xmin, xmax, ymin, ymax):
        painter.setBrush( qt4.QBrush() )
        utils.plotBoxesToPainter(painter, xmin, ymin, xmax, ymax, clip)

def _errorBarsBoxFilled(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                        s, painter, clip):
    """Draw box filled region inside error bars."""
    if None not in (xmin, xmax, ymin, ymax):
        # filled region below
        if not s.FillBelow.hideerror:
            path = qt4.QPainterPath()
            utils.addNumpyPolygonToPath(path, clip,
                                        xmin, ymin, xmin, yplotter,
                                        xmax, yplotter, xmax, ymin)
            utils.brushExtFillPath(painter, s.FillBelow, path, ignorehide=True)

        # filled region above
        if not s.FillAbove.hideerror:
            path = qt4.QPainterPath()
            utils.addNumpyPolygonToPath(path, clip,
                                        xmin, yplotter, xmax, yplotter,
                                        xmax, ymax, xmin, ymax)
            utils.brushExtFillPath(painter, s.FillAbove, path, ignorehide=True)

def _errorBarsDiamond(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                      s, painter, clip):
    """Draw diamond around error region."""
    if None not in (xmin, xmax, ymin, ymax):

        # expand clip by pen width (urgh)
        pw = painter.pen().widthF()*2
        clip = qt4.QRectF(qt4.QPointF(clip.left()-pw,clip.top()-pw),
                          qt4.QPointF(clip.right()+pw,clip.bottom()+pw))

        path = qt4.QPainterPath()
        utils.addNumpyPolygonToPath(path, clip,
                                    xmin, yplotter, xplotter, ymax,
                                    xmax, yplotter, xplotter, ymin)
        painter.setBrush( qt4.QBrush() )
        painter.drawPath(path)

def _errorBarsDiamondFilled(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                            s, painter, clip):
    """Draw diamond filled region inside error bars."""
    if None not in (xmin, xmax, ymin, ymax):
        if not s.FillBelow.hideerror:
            path = qt4.QPainterPath()
            utils.addNumpyPolygonToPath(path, clip,
                                        xmin, yplotter, xplotter, ymin,
                                        xmax, yplotter)
            utils.brushExtFillPath(painter, s.FillBelow, path, ignorehide=True)

        if not s.FillAbove.hideerror:
            path = qt4.QPainterPath()
            utils.addNumpyPolygonToPath(path, clip,
                                        xmin, yplotter, xplotter, ymax,
                                        xmax, yplotter)
            utils.brushExtFillPath(painter, s.FillAbove, path, ignorehide=True)

def _errorBarsCurve(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                    s, painter, clip):
    """Draw curve around error region."""
    if None not in (xmin, xmax, ymin, ymax):
        # non-filling brush
        painter.setBrush( qt4.QBrush() )

        for xp, yp, xmn, ymn, xmx, ymx in itertools.izip(
            xplotter, yplotter, xmin, ymin, xmax, ymax):

            # break up curve into four arcs (for asym error bars)
            # qt geometry means we have to calculate lots
            # the big numbers are in 1/16 degrees
            painter.drawArc(qt4.QRectF(xp - (xmx-xp), yp - (yp-ymx),
                                       (xmx-xp)*2, (yp-ymx)*2),
                            0, 1440)
            painter.drawArc(qt4.QRectF(xp - (xp-xmn), yp - (yp-ymx),
                                       (xp-xmn)*2, (yp-ymx)*2),
                            1440, 1440)
            painter.drawArc(qt4.QRectF(xp - (xp-xmn), yp - (ymn-yp),
                                       (xp-xmn)*2, (ymn-yp)*2),
                            2880, 1440)
            painter.drawArc(qt4.QRectF(xp - (xmx-xp), yp - (ymn-yp),
                                       (xmx-xp)*2, (ymn-yp)*2),
                            4320, 1440)

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
            utils.brushExtFillPolygon(painter, s.FillBelow, clip,
                                      ptsbelow+retnpts, ignorehide=True)
        if not s.FillAbove.hideerror:
            utils.brushExtFillPolygon(painter, s.FillAbove, clip,
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
    'fillhorz': (_errorBarsFilled,),
    'fillvert': (_errorBarsFilled,),
    'linehorz': (_errorBarsFilled,),
    'linevert': (_errorBarsFilled,),
    'linehorzbar': (_errorBarsBar, _errorBarsFilled),
    'linevertbar': (_errorBarsBar, _errorBarsFilled),
    }

class MarkerFillBrush(setting.Brush):
    def __init__(self, name, **args):
        setting.Brush.__init__(self, name, **args)

        self.get('color').newDefault( setting.Reference(
            '../PlotLine/color') )

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

class ColorSettings(setting.Settings):
    """Settings for a coloring points using data values."""

    def __init__(self, name):
        setting.Settings.__init__(self, name, setnsmode='groupedsetting')
        self.add( setting.DatasetOrFloatList(
                'points', '',
                descr = _('Use color value (0-1) in dataset to paint points'),
                usertext=_('Color markers')), 7 )
        self.add( setting.Float(
                'min', 0.,
                descr = _('Minimum value of color dataset'),
                usertext = _('Min val') ))
        self.add( setting.Float(
                'max', 1.,
                descr = _('Maximum value of color dataset'),
                usertext = _('Max val') ))
        self.add( setting.Choice(
                'scaling',
                ['linear', 'sqrt', 'log', 'squared'],
                'linear',
                descr = _('Scaling to transform numbers to color'),
                usertext=_('Scaling')))

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

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        GenericPlotter.addSettings(s)

        s.add( setting.Int('thinfactor', 1,
                           minval=1,
                           descr=_('Thin number of markers plotted'
                                   ' for each datapoint by this factor'),
                           usertext=_('Thin markers'),
                           formatting=True), 0 )
        s.add( setting.DistancePt('markerSize',
                                  '3pt',
                                  descr = _('Size of marker to plot'),
                                  usertext=_('Marker size'), formatting=True), 0 )
        s.add( setting.Marker('marker',
                              'circle',
                              descr = _('Type of marker to plot'),
                              usertext=_('Marker'), formatting=True), 0 )
        s.add( setting.DatasetOrStr('labels', '',
                                    descr=_('Dataset or string to label points'),
                                    usertext=_('Labels'), datatype='text'), 5 )
        s.add( setting.DatasetOrFloatList(
                'scalePoints', '',
                descr = _('Scale size of plotted markers by this dataset or'
                          ' list of values'),
                usertext=_('Scale markers')), 6 )

        s.add( ColorSettings('Color') )

        s.add( setting.DatasetOrFloatList(
                'yData', 'y',
                descr=_('Dataset containing y data or list of values'),
                usertext=_('Y data')), 0 )
        s.add( setting.DatasetOrFloatList(
                'xData', 'x',
                descr=_('Dataset containing x data or list of values'),
                usertext=_('X data')), 0 )
        s.add( setting.ErrorStyle('errorStyle',
                                  'bar',
                                  descr=_('Style of error bars to plot'),
                                  usertext=_('Error style'), formatting=True) )

        s.add( setting.XYPlotLine('PlotLine',
                                  descr = _('Plot line settings'),
                                  usertext = _('Plot line')),
               pixmap = 'settings_plotline' )
        s.add( setting.Line('MarkerLine',
                            descr = _('Line around the marker settings'),
                            usertext = _('Marker border')),
               pixmap = 'settings_plotmarkerline' )
        s.add( MarkerFillBrush('MarkerFill',
                               descr = _('Marker fill settings'),
                               usertext = _('Marker fill')),
               pixmap = 'settings_plotmarkerfill' )
        s.add( setting.ErrorBarLine('ErrorBarLine',
                                    descr = _('Error bar line settings'),
                                    usertext = _('Error bar line')),
               pixmap = 'settings_ploterrorline' )
        s.add( setting.PointFill('FillBelow',
                                 descr = _('Fill below plot line'),
                                 usertext = _('Fill below')),
               pixmap = 'settings_plotfillbelow' )
        s.add( setting.PointFill('FillAbove',
                                 descr = _('Fill above plot line'),
                                 usertext = _('Fill above')),
               pixmap = 'settings_plotfillabove' )
        s.add( setting.PointLabel('Label',
                                  descr = _('Label settings'),
                                  usertext=_('Label')),
               pixmap = 'settings_axislabel' )

    def _getUserDescription(self):
        """User-friendly description."""

        s = self.settings
        return "x='%s', y='%s', marker='%s'" % (s.xData, s.yData,
                                                s.marker)
    userdescription = property(_getUserDescription)

    def _plotErrors(self, posn, painter, xplotter, yplotter,
                    axes, xdata, ydata, cliprect):
        """Plot error bars (horizontal and vertical).
        """

        s = self.settings
        style = s.errorStyle
        if style == 'none':
            return

        # default is no error bars
        xmin = xmax = ymin = ymax = None

        # draw horizontal error bars
        if xdata.hasErrors():
            xmin, xmax = xdata.getPointRanges()

            # convert xmin and xmax to graph coordinates
            xmin = axes[0].dataToPlotterCoords(posn, xmin)
            xmax = axes[0].dataToPlotterCoords(posn, xmax)

        # draw vertical error bars
        if ydata.hasErrors():
            ymin, ymax = ydata.getPointRanges()

            # convert ymin and ymax to graph coordinates
            ymin = axes[1].dataToPlotterCoords(posn, ymin)
            ymax = axes[1].dataToPlotterCoords(posn, ymax)

        # no error bars - break out of processing below
        if ymin is None and ymax is None and xmin is None and xmax is None:
            return

        # iterate to call the error bars functions required to draw style
        pen = s.ErrorBarLine.makeQPenWHide(painter)
        pen.setCapStyle(qt4.Qt.FlatCap)

        painter.setPen(pen)
        for function in _errorBarFunctionMap[style]:
            function(style, xmin, xmax, ymin, ymax,
                     xplotter, yplotter, s, painter, cliprect)

    def providesAxesDependency(self):
        """This widget provides range information about these axes."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def updateAxisRange(self, axis, depname, axrange):
        """Compute the effect of data on the axis range."""
        dataname = {'sx': 'xData', 'sy': 'yData'}[depname]
        dsetn = self.settings.get(dataname)
        data = dsetn.getData(self.document)

        if data:
            drange = data.getRange()
            if drange:
                axrange[0] = min(axrange[0], drange[0])
                axrange[1] = max(axrange[1], drange[1])
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

        pts = qt4.QPolygonF()

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
                    pts.append( qt4.QPointF(xvals[-1], yvals[-1]) )

        else:
            assert False

        return pts

    def _getBezierLine(self, poly):
        """Try to draw a bezier line connecting the points."""

        npts = qtloops.bezier_fit_cubic_multi(poly, 0.1, len(poly)+1)
        i = 0
        path = qt4.QPainterPath()
        lastpt = qt4.QPointF(-999999,-999999)
        while i < len(npts):
            if lastpt != npts[i]:
                path.moveTo(npts[i])
            path.cubicTo(npts[i+1], npts[i+2], npts[i+3])
            lastpt = npts[i+3]
            i += 4
        return path

    def _drawBezierLine( self, painter, xvals, yvals, posn,
                         xdata, ydata):
        """Handle bezier lines and fills."""

        pts = self._getLinePoints(xvals, yvals, posn, xdata, ydata)
        if len(pts) < 2:
            return
        path = self._getBezierLine(pts)
        s = self.settings

        if not s.FillBelow.hide:
            temppath = qt4.QPainterPath(path)
            temppath.lineTo(pts[-1].x(), posn[3])
            temppath.lineTo(pts[0].x(), posn[3])
            utils.brushExtFillPath(painter, s.FillBelow, temppath)

        if not s.FillAbove.hide:
            temppath = qt4.QPainterPath(path)
            temppath.lineTo(pts[-1].x(), posn[1])
            temppath.lineTo(pts[0].x(), posn[1])
            utils.brushExtFillPath(painter, s.FillAbove, temppath)

        if not s.PlotLine.hide:
            painter.strokePath(path, s.PlotLine.makeQPen(painter))

    def _drawPlotLine( self, painter, xvals, yvals, posn, xdata, ydata,
                       cliprect ):
        """Draw the line connecting the points."""

        pts = self._getLinePoints(xvals, yvals, posn, xdata, ydata)
        if len(pts) < 2:
            return
        s = self.settings

        if not s.FillBelow.hide:
            # construct polygon to draw filled region
            polypts = qt4.QPolygonF([qt4.QPointF(pts[0].x(), posn[3])])
            polypts += pts
            polypts.append(qt4.QPointF(pts[len(pts)-1].x(), posn[3]))

            utils.brushExtFillPolygon(painter, s.FillBelow, cliprect, polypts)

        if not s.FillAbove.hide:
            polypts = qt4.QPolygonF([qt4.QPointF(pts[0].x(), posn[1])])
            polypts += pts
            polypts.append(qt4.QPointF(pts[len(pts)-1].x(), posn[1]))

            utils.brushExtFillPolygon(painter, s.FillAbove, cliprect, polypts)

        # draw line between points
        if not s.PlotLine.hide:
            painter.setPen( s.PlotLine.makeQPen(painter) )
            utils.plotClippedPolyline(painter, cliprect, pts)

    def drawKeySymbol(self, number, painter, x, y, width, height):
        """Draw the plot symbol and/or line."""
        painter.save()
        cliprect = qt4.QRectF(qt4.QPointF(x,y), qt4.QPointF(x+width,y+height))
        painter.setClipRect(cliprect)

        # draw sample error bar
        s = self.settings
        size = s.get('markerSize').convert(painter)
        style = s.errorStyle

        # make some fake error bar data to plot
        xv = s.get('xData').getData(self.document)
        yv = s.get('yData').getData(self.document)
        yp = y + height/2
        xpts = N.array([x-width, x+width/2, x+2*width])
        ypts = N.array([yp, yp, yp])

        # size of error bars in key
        errorsize = height*0.4

        # make points for error bars (if any)
        if xv and xv.hasErrors():
            xneg = N.array([x-width, x+width/2-errorsize, x+2*width])
            xpos = N.array([x-width, x+width/2+errorsize, x+2*width])
        else:
            xneg = xpos = xpts
        if yv and yv.hasErrors():
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
        textpen = lab.makeQPen()
        painter.setPen(textpen)
        font = lab.makeQFont(painter)
        angle = lab.angle

        # iterate over each point and plot each label
        for x, y, t in itertools.izip(xplotter+deltax, yplotter+deltay,
                                      textvals):
            utils.Renderer( painter, font, x, y, t,
                            alignhorz, alignvert, angle ).render()

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
            xv = document.DatasetRange(length, (1,length))
        elif xv and not yv and s.get('yData').isEmpty():
            length = xv.data.shape[0]
            yv = document.DatasetRange(length, (1,length))

        if None in (text, xv, yv):
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
            map_fn = lambda x, y: ( axes[0].dataToPlotterCoords(bounds, x),
                                    axes[1].dataToPlotterCoords(bounds, y) )

        return pickable.DiscretePickable(self, 'xData', 'yData', map_fn)

    def pickPoint(self, x0, y0, bounds, distance = 'radial'):
        return self._pickable(bounds).pickPoint(x0, y0, bounds, distance)

    def pickIndex(self, oldindex, direction, bounds):
        return self._pickable(bounds).pickIndex(oldindex, direction, bounds)

    def makeColorbarImage(self, direction='horz'):
        """Make a QImage colorbar for the current plot."""

        s = self.settings
        c = s.Color
        cmap = self.document.getColormap(
            s.MarkerFill.colorMap, s.MarkerFill.colorMapInvert)

        return utils.makeColorbarImage(
            c.min, c.max, c.scaling, cmap, 0,
            direction=direction)

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
            yv = document.DatasetRange(length, (1,length))
        elif yv and not xv and s.get('xData').isEmpty():
            # use index for x data
            length = yv.data.shape[0]
            xv = document.DatasetRange(length, (1,length))
        if not xv or not yv:
            # no valid dataset, so exit
            return

        # if text entered, then multiply up to get same number of values
        # as datapoints
        if text:
            length = min( len(xv.data), len(yv.data) )
            text = text*(length / len(text)) + text[:length % len(text)]

        # loop over chopped up values
        for xvals, yvals, tvals, ptvals, cvals in (
            document.generateValidDatasetParts(
                xv, yv, text, scalepoints, colorpoints)):

            #print "Calculating coordinates"
            # calc plotter coords of x and y points
            xplotter = axes[0].dataToPlotterCoords(posn, xvals.data)
            yplotter = axes[1].dataToPlotterCoords(posn, yvals.data)

            #print "Painting plot line"
            # plot data line (and/or filling above or below)
            if not s.PlotLine.hide or not s.FillAbove.hide or not s.FillBelow.hide:
                if s.PlotLine.bezierJoin and hasqtloops:
                    self._drawBezierLine( painter, xplotter, yplotter, posn,
                                          xvals, yvals )
                else:
                    self._drawPlotLine( painter, xplotter, yplotter, posn,
                                        xvals, yvals, cliprect )

            # shift points if in certain step modes
            if s.PlotLine.steps != 'off':
                steps = s.PlotLine.steps
                if s.PlotLine.steps == 'right-shift-points':
                    xplotter[1:] = 0.5*(xplotter[:-1] + xplotter[1:])
                elif s.PlotLine.steps == 'left-shift-points':
                    xplotter[:-1] = 0.5*(xplotter[:-1] + xplotter[1:])

            #print "Painting error bars"
            # plot errors bars
            self._plotErrors(posn, painter, xplotter, yplotter,
                             axes, xvals, yvals, cliprect)

            # plot the points (we do this last so they are on top)
            markersize = s.get('markerSize').convert(painter)
            if not s.MarkerLine.hide or not s.MarkerFill.hide:

                #print "Painting marker fill"
                if not s.MarkerFill.hide:
                    # filling for markers
                    painter.setBrush( s.MarkerFill.makeQBrush() )
                else:
                    # no-filling brush
                    painter.setBrush( qt4.QBrush() )

                #print "Painting marker lines"
                if not s.MarkerLine.hide:
                    # edges of markers
                    painter.setPen( s.MarkerLine.makeQPen(painter) )
                else:
                    # invisible pen
                    painter.setPen( qt4.QPen(qt4.Qt.NoPen) )

                # thin datapoints as required
                if s.thinfactor <= 1:
                    xplt, yplt = xplotter, yplotter
                else:
                    xplt, yplt = (xplotter[::s.thinfactor],
                                  yplotter[::s.thinfactor])

                # whether to scale markers
                scaling = colorvals = cmap = None
                if ptvals:
                    scaling = ptvals.data
                    if s.thinfactor > 1:
                        scaling = scaling[::s.thinfactor]

                # color point individually
                if cvals:
                    colorvals = utils.applyScaling(
                        cvals.data, s.Color.scaling,
                        s.Color.min, s.Color.max)
                    if s.thinfactor > 1:
                        colorvals = colorvals[::s.thinfactor]
                    cmap = self.document.getColormap(
                        s.MarkerFill.colorMap, s.MarkerFill.colorMapInvert)

                # actually plot datapoints
                utils.plotMarkers(painter, xplt, yplt, s.marker, markersize,
                                  scaling=scaling, clip=cliprect,
                                  cmap=cmap, colorvals=colorvals)

            # finally plot any labels
            if tvals and not s.Label.hide:
                self.drawLabels(painter, xplotter, yplotter,
                                tvals, markersize)

# allow the factory to instantiate an x,y plotter
document.thefactory.register( PointPlotter )
