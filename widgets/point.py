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

# $Id$

"""For plotting xy points."""

import veusz.qtall as qt4
import itertools
import numpy as N

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

from plotters import GenericPlotter

# functions for plotting error bars
# different styles are made up of combinations of these functions
# each function takes the same arguments
def _errorBarsBar(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                  s, painter):
    """Draw bar style error lines."""
    # list of output lines
    pts = []

    # vertical error bars
    if ymin is not None and ymax is not None and not s.ErrorBarLine.hideVert:
        for x, y1, y2 in itertools.izip(xplotter, ymin, ymax):
            pts.append( qt4.QLineF(x, y1, x, y2) )

    # horizontal error bars
    if xmin is not None and xmax is not None and not s.ErrorBarLine.hideHorz:
        for x1, x2, y in itertools.izip(xmin, xmax, yplotter):
            pts.append( qt4.QLineF(x1, y, x2, y) )
    if pts:
        painter.drawLines(pts)

def _errorBarsEnds(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                   s, painter):
    """Draw perpendiclar ends on error bars."""
    size = s.get('markerSize').convert(painter)
    lines = []
    if ymin is not None and ymax is not None and not s.ErrorBarLine.hideVert:
        for x, y1, y2 in itertools.izip(xplotter, ymin, ymax):
            lines.append( qt4.QLineF(x-size, y1, x+size, y1) )
            lines.append( qt4.QLineF(x-size, y2, x+size, y2) )

    if xmin is not None and xmax is not None and not s.ErrorBarLine.hideHorz:
        for x1, x2, y in itertools.izip(xmin, xmax, yplotter):
            lines.append( qt4.QLineF(x1, y-size, x1, y+size) )
            lines.append( qt4.QLineF(x2, y-size, x2, y+size) )
    if lines:
        painter.drawLines(lines)

def _errorBarsBox(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                  s, painter):
    """Draw box around error region."""
    if None not in (xmin, xmax, ymin, ymax):
        painter.setBrush( qt4.QBrush() )

        for xmn, ymn, xmx, ymx in itertools.izip(xmin, ymin, xmax, ymax):
            painter.drawPolygon( qt4.QPointF(xmn, ymn), qt4.QPointF(xmx, ymn),
                                 qt4.QPointF(xmx, ymx), qt4.QPointF(xmn, ymx) )

def _errorBarsDiamond(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                      s, painter):
    """Draw diamond around error region."""
    if None not in (xmin, xmax, ymin, ymax):
        painter.setBrush( qt4.QBrush() )

        for xp, yp, xmn, ymn, xmx, ymx in itertools.izip(
            xplotter, yplotter, xmin, ymin, xmax, ymax):

            painter.drawPolygon( qt4.QPointF(xmn, yp), qt4.QPointF(xp, ymx),
                                 qt4.QPointF(xmx, yp), qt4.QPointF(xp, ymn) )

def _errorBarsCurve(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                    s, painter):
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
                                       (xmx-xp)*2+1, (yp-ymx)*2+1),
                            0, 1440)
            painter.drawArc(qt4.QRectF(xp - (xp-xmn), yp - (yp-ymx),
                                       (xp-xmn)*2+1, (yp-ymx)*2+1),
                            1440, 1440)
            painter.drawArc(qt4.QRectF(xp - (xp-xmn), yp - (ymn-yp),
                                       (xp-xmn)*2+1, (ymn-yp)*2+1),
                            2880, 1440)
            painter.drawArc(qt4.QRectF(xp - (xmx-xp), yp - (ymn-yp),
                                       (xmx-xp)*2+1, (ymn-yp)*2+1),
                            4320, 1440)

def _errorBarsFilled(style, xmin, xmax, ymin, ymax, xplotter, yplotter,
                     s, painter):
    """Draw filled region as error region."""

    ptsabove = qt4.QPolygonF()
    ptsbelow = qt4.QPolygonF()

    hidevert = True  # keep track of what's shown
    hidehorz = True
    if ( (style == 'fillvert' or style == 'linevert') and
         (ymin is not None and ymax is not None) and
         not s.ErrorBarLine.hideVert ):
        hidevert = False
        # lines above/below points
        for x, y in itertools.izip(xplotter, ymin):
            ptsbelow.append(qt4.QPointF(x, y))
        for x, y in itertools.izip(xplotter, ymax):
            ptsabove.append(qt4.QPointF(x, y))

    elif ( (style == 'fillhorz' or style == 'linehorz') and
           (xmin is not None and xmax is not None) and
           not s.ErrorBarLine.hideHorz ):
        hidehorz = False
        # lines left/right points
        for x, y in itertools.izip(xmin, yplotter):
            ptsbelow.append(qt4.QPointF(x, y))
        for x, y in itertools.izip(xmax, yplotter):
            ptsabove.append(qt4.QPointF(x, y))

    # draw filled regions above/left and below/right
    if ( (style == 'fillvert' or style == 'fillhorz') and
         not (hidehorz and hidevert) ):
        # construct points for error bar regions
        retnpts = qt4.QPolygonF()
        for x, y in itertools.izip(xplotter[::-1], yplotter[::-1]):
            retnpts.append(qt4.QPointF(x, y))

        # polygons consist of lines joining the points and continuing
        # back along the plot line (retnpts)
        painter.save()
        painter.setPen( qt4.Qt.NoPen )
        if not s.FillBelow.hideerror:
            painter.setBrush( s.FillBelow.makeQBrush() )
            painter.drawPolygon( ptsbelow + retnpts )
        if not s.FillAbove.hideerror:
            painter.setBrush( s.FillAbove.makeQBrush() )
            painter.drawPolygon( ptsabove + retnpts )
        painter.restore()

    # draw optional line (on top of fill)
    painter.drawPolyline( ptsabove )
    painter.drawPolyline( ptsbelow )

# map error bar names to lists of functions (above)
_errorBarFunctionMap = {
    'none': (),
    'bar': (_errorBarsBar,),
    'bardiamond': (_errorBarsBar, _errorBarsDiamond,),
    'barcurve': (_errorBarsBar, _errorBarsCurve,),
    'barbox': (_errorBarsBar, _errorBarsBox,),
    'barends': (_errorBarsBar, _errorBarsEnds,),
    'box':  (_errorBarsBox,),
    'diamond':  (_errorBarsDiamond,),
    'curve': (_errorBarsCurve,),
    'fillhorz': (_errorBarsFilled,),
    'fillvert': (_errorBarsFilled,),
    'linehorz': (_errorBarsFilled,),
    'linevert': (_errorBarsFilled,),
    }


class PointPlotter(GenericPlotter):
    """A class for plotting points and their errors."""

    typename='xy'
    allowusercreation=True
    description='Plot points with lines and errorbars'

    class MarkerFillBrush(setting.Brush):
        def __init__(self, name, **args):
            setting.Brush.__init__(self, name, **args)

            self.get('color').newDefault( setting.Reference(
                '../PlotLine/color') )
    
    def __init__(self, parent, name=None):
        """Initialise XY plotter plotting (xdata, ydata).
        
        xdata and ydata are strings specifying the data in the document"""
        
        GenericPlotter.__init__(self, parent, name=name)
        s = self.settings

        s.add( setting.Int('thinfactor', 1,
                           minval=1,
                           descr='Thin number of markers plotted'
                           ' for each datapoint by this factor',
                           usertext='Thin markers',
                           formatting=True), 0 )
        s.add( setting.Distance('markerSize', '3pt',
                                descr = 'Size of marker to plot',
                                usertext='Marker size', formatting=True), 0 )
        s.add( setting.Marker('marker', 'circle',
                              descr = 'Type of marker to plot',
                              usertext='Marker', formatting=True), 0 )

        s.add( setting.DatasetOrFloatList('yData', 'y',
                                          descr = 'Dataset containing y data or list of values',
                                          usertext='Y data'), 0 )
        s.add( setting.DatasetOrFloatList('xData', 'x',
                                          descr = 'Dataset containing x data or list of values',
                                          usertext='X data'), 0 )
        s.add( setting.ErrorStyle('errorStyle',
                                  'bar',
                                  descr='Style of error bars to plot',
                                  usertext='Error style', formatting=True) )

        s.add( setting.DatasetOrStr('labels', '',
                                    descr='Dataset or string to label points',
                                    usertext='Labels', datatype='text') )

        s.add( setting.XYPlotLine('PlotLine',
                                  descr = 'Plot line settings',
                                  usertext = 'Plot line'),
               pixmap = 'plotline' )
        s.add( setting.Line('MarkerLine',
                            descr = 'Line around the marker settings',
                            usertext = 'Marker border'),
               pixmap = 'plotmarkerline' )
        s.add( PointPlotter.MarkerFillBrush('MarkerFill',
                                            descr = 'Marker fill settings',
                                            usertext = 'Marker fill'),
               pixmap = 'plotmarkerfill' )
        s.add( setting.ErrorBarLine('ErrorBarLine',
                                    descr = 'Error bar line settings',
                                    usertext = 'Error bar line'),
               pixmap = 'ploterrorline' )
        s.add( setting.PointFill('FillBelow',
                                 descr = 'Fill below plot line',
                                 usertext = 'Fill below'),
               pixmap = 'plotfillbelow' )
        s.add( setting.PointFill('FillAbove',
                                 descr = 'Fill above plot line',
                                 usertext = 'Fill above'),
               pixmap = 'plotfillabove' )
        s.add( setting.PointLabel('Label',
                                  descr = 'Label settings',
                                  usertext='Label'),
               pixmap = 'axislabel' )

        if type(self) == PointPlotter:
            self.readDefaults()

    def _getUserDescription(self):
        """User-friendly description."""

        s = self.settings
        return "x='%s', y='%s', marker='%s'" % (s.xData, s.yData,
                                                s.marker)
    userdescription = property(_getUserDescription)

    def _plotErrors(self, posn, painter, xplotter, yplotter,
                    axes, xdata, ydata):
        """Plot error bars (horizontal and vertical).
        """

        s = self.settings
        style = s.errorStyle
        if style == 'none':
            return

        # distances for clipping - we make them larger than the
        # real width, to help get gradients and so on correct
        xwc = abs(posn[2]-posn[0])*4
        ywc = abs(posn[3]-posn[1])*4
        # default is no error bars
        xmin = xmax = ymin = ymax = None

        # draw horizontal error bars
        if xdata.hasErrors():
            xmin, xmax = xdata.getPointRanges()

            # convert xmin and xmax to graph coordinates
            xmin = axes[0].dataToPlotterCoords(posn, xmin)
            xmax = axes[0].dataToPlotterCoords(posn, xmax)

            # clip... (avoids problems with INFs, etc)
            xmin = N.clip(xmin, posn[0]-xwc, posn[2]+xwc)
            xmax = N.clip(xmax, posn[0]-xwc, posn[2]+xwc)

        # draw vertical error bars
        if ydata.hasErrors():
            ymin, ymax = ydata.getPointRanges()

            # convert ymin and ymax to graph coordinates
            ymin = axes[1].dataToPlotterCoords(posn, ymin)
            ymax = axes[1].dataToPlotterCoords(posn, ymax)

            # clip...
            ymin = N.clip(ymin, posn[1]-ywc, posn[3]+ywc)
            ymax = N.clip(ymax, posn[1]-ywc, posn[3]+ywc)

        # no error bars - break out of processing below
        if ymin is None and ymax is None and xmin is None and xmax is None:
            return
        
        # iterate to call the error bars functions required to draw style
        painter.setPen( s.ErrorBarLine.makeQPenWHide(painter) )
        for function in _errorBarFunctionMap[style]:
            function(style, xmin, xmax, ymin, ymax,
                     xplotter, yplotter, s, painter)
            
    def providesAxesDependency(self):
        """This widget provides range information about these axes."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def updateAxisRange(self, axis, depname, axrange):
        dataname = {'sx': 'xData', 'sy': 'yData'}[depname]
        data = self.settings.get(dataname).getData(self.document)
        if data:
            drange = data.getRange()
            if drange:
                axrange[0] = min(axrange[0], drange[0])
                axrange[1] = max(axrange[1], drange[1])

    def _getLinePoints( self, xvals, yvals, posn, xdata, ydata ):
        """Get the points corresponding to the line connecting the points."""

        pts = qt4.QPolygonF()

        s = self.settings
        steps = s.PlotLine.steps

        # simple continuous line
        if steps == 'off':
            for xpt, ypt in itertools.izip(xvals, yvals):
                pts.append(qt4.QPointF(xpt, ypt))

        # stepped line, with points on left
        elif steps == 'left':
            for x1, x2, y1, y2 in itertools.izip(xvals[:-1], xvals[1:],
                                                 yvals[:-1], yvals[1:]):
                pts.append(qt4.QPointF(x1, y1))
                pts.append(qt4.QPointF(x2, y1))
                pts.append(qt4.QPointF(x2, y2))

        # stepped line, with points on right
        elif steps == 'right':
            for x1, x2, y1, y2 in itertools.izip(xvals[:-1], xvals[1:],
                                                 yvals[:-1], yvals[1:]):
                pts.append(qt4.QPointF(x1, y1))
                pts.append(qt4.QPointF(x1, y2))
                pts.append(qt4.QPointF(x2, y2))
            
        # stepped line, with points in centre
        # this is complex as we can't use the mean of the plotter coords,
        #  as the axis could be log
        elif steps == 'centre':
            axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

            if xdata.hasErrors():
                # Special case if error bars on x points:
                # here we use the error bars to define the steps
                xmin, xmax = xdata.getPointRanges()

                # this is duplicated from drawing error bars: bad
                # convert xmin and xmax to graph coordinates
                xmin = axes[0].dataToPlotterCoords(posn, xmin)
                xmax = axes[0].dataToPlotterCoords(posn, xmax)
                xmin = N.clip(xmin, -32767, 32767)
                xmax = N.clip(xmax, -32767, 32767)

                for xmn, xmx, y in itertools.izip(xmin, xmax, yvals):
                    pts.append( qt4.QPointF(xmn, y) )
                    pts.append( qt4.QPointF(xmx, y) )

            else:
                # we put the bin edges half way between the points
                # we assume this is the correct thing to do even in log space
                for x1, x2, y1, y2 in itertools.izip(xvals[:-1], xvals[1:],
                                                     yvals[:-1], yvals[1:]):
                    xc = 0.5*(x1+x2)
                    pts.append(qt4.QPointF(x1, y1))
                    pts.append(qt4.QPointF(xc, y1))
                    pts.append(qt4.QPointF(xc, y2))

                if len(xvals) > 0:
                    pts.append( qt4.QPointF(xvals[-1], yvals[-1]) )

        else:
            assert False

        return pts

    def _drawPlotLine( self, painter, xvals, yvals, posn, xdata, ydata ):
        """Draw the line connecting the points."""

        s = self.settings
        pts = self._getLinePoints(xvals, yvals, posn, xdata, ydata)

        if len(pts) >= 2:
            if not s.FillBelow.hide:
                # empty pen (line gets drawn below)
                painter.setPen( qt4.QPen( qt4.Qt.NoPen ) )
                painter.setBrush( s.FillBelow.makeQBrush() )

                newpts = qt4.QPolygonF(pts)
                newpts.insert(0, qt4.QPointF(pts[0].x(), posn[3]))
                newpts.append(qt4.QPointF(pts[len(pts)-1].x(), posn[3]))
                painter.drawPolygon(newpts)

            if not s.FillAbove.hide:
                # empty pen (line gets drawn below)
                painter.setPen( qt4.QPen( qt4.Qt.NoPen ) )
                painter.setBrush( s.FillAbove.makeQBrush() )

                newpts = qt4.QPolygonF(pts)
                newpts.insert(0, qt4.QPointF(pts[0].x(), posn[1]))
                newpts.append(qt4.QPointF(pts[len(pts)-1].x(), posn[1]))
                painter.drawPolygon(newpts)

            # draw line between points
            if not s.PlotLine.hide:
                painter.setPen( s.PlotLine.makeQPen(painter) )
                painter.drawPolyline(pts)

    def drawKeySymbol(self, painter, x, y, width, height):
        """Draw the plot symbol and/or line."""

        s = self.settings
        yp = y + height/2

        # draw line
        if not s.PlotLine.hide:
            painter.setPen( s.PlotLine.makeQPen(painter) )
            painter.drawLine( qt4.QPointF(x, yp), qt4.QPointF(x+width, yp) )

        # draw marker
        if not s.MarkerLine.hide or not s.MarkerFill.hide:
            size = s.get('markerSize').convert(painter)

            if not s.MarkerFill.hide:
                painter.setBrush( s.MarkerFill.makeQBrush() )

            if not s.MarkerLine.hide:
                painter.setPen( s.MarkerLine.makeQPen(painter) )
            else:
                painter.setPen( qt4.QPen( qt4.Qt.NoPen ) )
                
            utils.plotMarker(painter, x+width/2, yp, s.marker, size)

    def generateValidDatasetParts(self, *datasets):
        """Generator to return array of valid parts of datasets."""

        # find NaNs and INFs in input dataset
        invalid = datasets[0].invalidDataPoints()
        minlen = invalid.shape[0]
        for ds in datasets[1:]:
            try:
                nextinvalid = ds.invalidDataPoints()
                minlen = min(nextinvalid.shape[0], minlen)
                invalid = N.logical_or(invalid[:minlen], nextinvalid[:minlen])
            except AttributeError:
                # if not a dataset
                pass
        
        # get indexes of invalid pounts
        indexes = invalid.nonzero()[0].tolist()

        # no bad points: optimisation
        if not indexes:
            yield datasets
            return

        # add on shortest length of datasets
        indexes.append( minlen )
    
        lastindex = 0
        for index in indexes:
            if index != lastindex:
                retn = []
                for ds in datasets:
                    if ds is not None:
                        retn.append( ds[lastindex:index] )
                    else:
                        retn.append( None )
                yield retn
            lastindex = index+1

    def drawLabels(self, painter, xplotter, yplotter, textvals, markersize):
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

    def draw(self, parentposn, painter, outerbounds=None):
        """Plot the data on a plotter."""

        posn = GenericPlotter.draw(self, parentposn, painter,
                                   outerbounds=outerbounds)
        x1, y1, x2, y2 = posn

        s = self.settings

        # exit if hidden
        if s.hide:
            return

        # get data
        doc = self.document
        xv = s.get('xData').getData(doc)
        yv = s.get('yData').getData(doc)
        text = s.get('labels').getData(doc, checknull=True)

        if not xv or not yv:
            return

        # if text entered, then multiply up to get same number of values
        # as datapoints
        if text:
            length = min( len(xv.data), len(yv.data) )
            text = text*(length / len(text)) + text[:length % len(text)]

        # get axes widgets
        axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

        # return if there's no proper axes
        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return

        # clip data within bounds of plotter
        painter.beginPaintingWidget(self, posn)
        painter.save()
        self.clipAxesBounds(painter, axes, posn)

        # loop over chopped up values
        for xvals, yvals, tvals in self.generateValidDatasetParts(xv, yv, text):

            #print "Calculating coordinates"
            # calc plotter coords of x and y points
            xplotter = axes[0].dataToPlotterCoords(posn, xvals.data)
            yplotter = axes[1].dataToPlotterCoords(posn, yvals.data)

            # need to remove silly points as these stuff up output
            xplotter = N.clip(xplotter, -32767, 32767)
            yplotter = N.clip(yplotter, -32767, 32767)

            #print "Painting error bars"
            # plot errors bars
            self._plotErrors(posn, painter, xplotter, yplotter,
                             axes, xvals, yvals)

            #print "Painting plot line"
            # plot data line (and/or filling above or below)
            if not s.PlotLine.hide or not s.FillAbove.hide or not s.FillBelow.hide:
                self._drawPlotLine( painter, xplotter, yplotter, posn,
                                    xvals, yvals )

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
                    xplt, yplt = xplotter[::s.thinfactor], yplotter[::s.thinfactor]

                # actually plot datapoints
                utils.plotMarkers(painter, xplt, yplt, s.marker,
                                  markersize)
                    
            # finally plot any labels
            if tvals and not s.Label.hide:
                self.drawLabels(painter, xplotter, yplotter, tvals, markersize)

        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate an x,y plotter
document.thefactory.register( PointPlotter )
