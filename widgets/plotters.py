#    plotters.py
#    plotting classes

#    Copyright (C) 2004 Jeremy S. Sanders
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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
###############################################################################

# $Id$

import veusz.qtall as qt4
import itertools
import numpy as N

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import widget
import graph

class GenericPlotter(widget.Widget):
    """Generic plotter."""

    typename='genericplotter'
    allowedparenttypes=[graph.Graph]

    def __init__(self, parent, name=None):
        """Initialise object, setting axes."""
        widget.Widget.__init__(self, parent, name=name)

        s = self.settings
        s.add( setting.Str('key', '',
                           descr = 'Description of the plotted data to appear in key',
                           usertext='Key text') )
        s.add( setting.Axis('xAxis', 'x', 'horizontal',
                            descr = 'Name of X-axis to use',
                            usertext='X axis') )
        s.add( setting.Axis('yAxis', 'y', 'vertical',
                            descr = 'Name of Y-axis to use',
                            usertext='Y axis') )

    def getAxesNames(self):
        """Returns names of axes used."""
        s = self.settings
        return [s.xAxis, s.yAxis]

    def drawKeySymbol(self, painter, x, y, width, height):
        """Draw the plot symbol and/or line at (x,y) in a box width*height.

        This is used to plot a key
        """
        pass

    def clipAxesBounds(self, painter, axes, bounds):
        """Clip painter to start and stop values of axis."""

        # update cached coordinates of axes
        axes[0].plotterToGraphCoords(bounds, N.array([]))
        axes[1].plotterToGraphCoords(bounds, N.array([]))

        # get range
        x1 = axes[0].coordParr1
        x2 = axes[0].coordParr2
        y1 = axes[1].coordParr1
        y2 = axes[1].coordParr2

        # actually clip the data
        painter.setClipRect( qt4.QRectF(x1, y2, x2-x1, y1-y2) )

########################################################################
        
class FunctionPlotter(GenericPlotter):
    """Function plotting class."""

    typename='function'
    allowusercreation=True
    description='Plot a function'
    
    def __init__(self, parent, name=None):
        """Initialise plotter with axes."""

        GenericPlotter.__init__(self, parent, name=name)

        s = self.settings
        s.add( setting.Int('steps', 50,
                           descr = 'Number of steps to evaluate the function'
                           ' over', usertext='Steps', formatting=True), 0 )
        s.add( setting.Choice('variable', ['x', 'y'], 'x',
                              descr='Variable the function is a function of',
                              usertext='Variable'),
               0 )
        s.add( setting.Str('function', 'x',
                           descr='Function expression',
                           usertext='Function'), 0 )

        s.add(setting.FloatOrAuto('min', 'Auto',
                                  descr='Minimum value at which to plot function',
                                  usertext='Min'))
        
        s.add(setting.FloatOrAuto('max', 'Auto',
                                  descr='Maximum value at which to plot function',
                                  usertext='Max'))

        s.add( setting.Line('Line',
                            descr = 'Function line settings',
                            usertext = 'Plot line'),
               pixmap = 'plotline' )

        s.add( setting.PlotterFill('FillBelow',
                                   descr = 'Fill below function',
                                   usertext = 'Fill below'),
               pixmap = 'plotfillbelow' )
        
        s.add( setting.PlotterFill('FillAbove',
                                   descr = 'Fill above function',
                                   usertext = 'Fill above'),
               pixmap = 'plotfillabove' )

        if type(self) == FunctionPlotter:
            self.readDefaults()

        self.cachedfunc = None
        self.cachedvar = None
        
    def _getUserDescription(self):
        """User-friendly description."""
        return "%(variable)s = %(function)s" % self.settings
    userdescription = property(_getUserDescription)

    def _plotLine(self, painter, xpts, ypts, bounds):
        """ Plot the points in xpts, ypts."""
        x1, y1, x2, y2 = bounds

        maxdeltax = (x2-x1)*3/4
        maxdeltay = (y2-y1)*3/4

        # idea is to collect points until we go out of the bounds
        # or reach the end, then plot them
        pts = qt4.QPolygonF()
        lastx = lasty = -65536
        for x, y in itertools.izip(xpts, ypts):

            # ignore point if it outside sensible bounds
            if x < -32767 or y < -32767 or x > 32767 or y > 32767:
                if len(pts) >= 2:
                    painter.drawPolyline(pts)
                    pts.clear()
            else:
                # if the jump wasn't too large, add the point to the points
                if abs(x-lastx) < maxdeltax and abs(y-lasty) < maxdeltay:
                    pts.append( qt4.QPointF(x, y) )
                else:
                    # draw what we have until now, and start a new line
                    if len(pts) >= 2:
                        painter.drawPolyline(pts)
                    pts.clear()
                    pts.append( qt4.QPointF(x, y) )

            lastx = x
            lasty = y

        # draw remaining points
        if len(pts) >= 2:
            painter.drawPolyline(pts)

    def _fillRegion(self, painter, pxpts, pypts, bounds, belowleft):
        """Fill the region above/below or left/right of the points.

        belowleft fills below if the variable is 'x', or left if 'y'
        otherwise it fills above/right."""

        # find starting and ending points for the filled region
        x1, y1, x2, y2 = bounds
        s = self.settings
        
        pts = qt4.QPolygonF()
        if self.settings.variable == 'x':
            if belowleft:
                pts.append(qt4.QPointF(pxpts[0], y2))
                endpt = qt4.QPointF(pxpts[-1], y2)
            else:
                pts.append(qt4.QPointF(pxpts[0], y1))
                endpt = qt4.QPointF(pxpts[-1], y1)
        else:
            if belowleft:
                pts.append(qt4.QPointF(x1, pypts[0]))
                endpt = qt4.QPointF(x1, pypts[-1])
            else:
                pts.append(qt4.QPointF(x2, pypts[0]))
                endpt = qt4.QPointF(x2, pypts[-1])

        # add the points between (clipped to the bounds*2 - helps edges)
        xw = abs(x2-x1)
        xclip = N.clip(pxpts, x1-xw-1, x2+xw+1)
        yw = abs(y2-y1)
        yclip = N.clip(pypts, y1-yw-1, y2+yw+1)
        for x, y in itertools.izip(xclip, yclip):
            pts.append( qt4.QPointF(x, y) )

        # stick on the ending point
        pts.append(endpt)

        # actually do the filling
        painter.drawPolygon(pts)

    def drawKeySymbol(self, painter, x, y, width, height):
        """Draw the plot symbol and/or line."""

        s = self.settings
        yp = y + height/2

        # draw line
        if not s.Line.hide:
            painter.setBrush( qt4.QBrush() )
            painter.setPen( s.Line.makeQPen(painter) )
            painter.drawLine( qt4.QPointF(x, yp), qt4.QPointF(x+width, yp) )

    def _calcFunctionPoints(self, axes, posn):
        """Calculate the pixels to plot for the function
        returns (pxpts, pypts)."""

        s = self.settings
        x1, y1, x2, y2 = posn

        # check function doesn't contain dangerous code
        if self.cachedfunc != s.function or self.cachedvar != s.variable:
            checked = utils.checkCode(s.function)
            if checked is not None:
                return None, None
            self.cachedfunc = s.function
            self.cachedvar = s.variable

            try:
                # compile code
                self.cachedcomp = compile(self.cachedfunc, '<string>', 'eval')
            except:
                # return nothing
                return None, None

        env = utils.veusz_eval_context.copy()
        if s.variable == 'x':
            # x function
            if not(s.min == 'Auto') and s.min > axes[0].getPlottedRange()[0]:
                x_min = N.array([s.min])
                x1 = axes[0].graphToPlotterCoords(posn, x_min)[0]
            if not(s.max == 'Auto') and s.max < axes[0].getPlottedRange()[1]:
                x_max = N.array([s.max])
                x2 = axes[0].graphToPlotterCoords(posn, x_max)[0]
                
            delta = (x2 - x1) / float(s.steps)
            pxpts = N.arange(x1, x2+delta, delta)
            x = axes[0].plotterToGraphCoords(posn, pxpts)
            env['x'] = x
            try:
                y = eval(self.cachedcomp, env)
            except:
                pypts = None
            else:
                pypts = axes[1].graphToPlotterCoords(posn, y+x*0.)

        else:
            # y function
            if not(s.min == 'Auto') and s.min > axes[1].getPlottedRange()[0]:
                y_min = N.array([s.min])
                y2 = axes[1].graphToPlotterCoords(posn, y_min)[0]
            if not(s.max == 'Auto') and s.max < axes[1].getPlottedRange()[1]:
                y_max = N.array([s.max])
                y1 = axes[1].graphToPlotterCoords(posn, y_max)[0]
            
            delta = (y2 - y1) / float(s.steps)
            pypts = N.arange(y1, y2+delta, delta)
            y = axes[1].plotterToGraphCoords(posn, pypts)
            env['y'] = y
            try:
                x = eval(self.cachedcomp, env)
            except:
                pxpts = None
            else:
                pxpts = axes[0].graphToPlotterCoords(posn, x+y*0.)

        return pxpts, pypts

    def draw(self, parentposn, painter, outerbounds = None):
        """Draw the function."""

        posn = GenericPlotter.draw(self, parentposn, painter,
                                   outerbounds = outerbounds)
        x1, y1, x2, y2 = posn
        s = self.settings

        # exit if hidden
        if s.hide:
            return

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

        # get the points to plot by evaluating the function
        pxpts, pypts = self._calcFunctionPoints(axes, posn)

        # draw the function line
        if pxpts is None or pypts is None:
            # not sure how to deal with errors here
            painter.setPen( qt4.QColor('red') )
            f = qt4.QFont()
            f.setPointSize(20)
            painter.setFont(f)
            painter.drawText( qt4.QRectF(x1, y1, x2-x1, y2-y1),
                              qt4.Qt.AlignCenter,
                              "Cannot evaluate '%s'" % s.function )
        else:
            if not s.FillBelow.hide:
                painter.setBrush( s.FillBelow.makeQBrush() )
                painter.setPen( qt4.QPen(qt4.Qt.NoPen) )
                self._fillRegion(painter, pxpts, pypts, posn, True)

            if not s.FillAbove.hide:
                painter.setBrush( s.FillAbove.makeQBrush() )
                painter.setPen( qt4.QPen(qt4.Qt.NoPen) )
                self._fillRegion(painter, pxpts, pypts, posn, False)

            if not s.Line.hide:
                painter.setBrush( qt4.QBrush() )
                painter.setPen( s.Line.makeQPen(painter) )
                self._plotLine(painter, pxpts, pypts, posn)

        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate an function plotter
document.thefactory.register( FunctionPlotter )

###############################################################################

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
        s.add( setting.Choice('errorStyle',
                              ['bar', 'box', 'diamond', 'curve',
                               'barbox', 'bardiamond', 'barcurve'], 'bar',
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
        s.add( setting.PlotterFill('FillBelow',
                                   descr = 'Fill below plot line',
                                   usertext = 'Fill below'),
               pixmap = 'plotfillbelow' )
        s.add( setting.PlotterFill('FillAbove',
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
        """Plot error bars (horizontal and vertical)."""

        s = self.settings

        # distances for clipping - we make them larger than the
        # real width, to help get gradients and so on correct
        xwc = abs(posn[2]-posn[0])*4
        ywc = abs(posn[3]-posn[1])*4

        # draw horizontal error bars
        if xdata.hasErrors():
            xmin, xmax = xdata.getPointRanges()
                    
            # convert xmin and xmax to graph coordinates
            xmin = axes[0].graphToPlotterCoords(posn, xmin)
            xmax = axes[0].graphToPlotterCoords(posn, xmax)

            # clip... (avoids problems with INFs, etc)
            xmin = N.clip(xmin, posn[0]-xwc, posn[2]+xwc)
            xmax = N.clip(xmax, posn[0]-xwc, posn[2]+xwc)

            # draw lines between each of the points
        else:
            xmin = xmax = None

        # draw vertical error bars
        if ydata.hasErrors():
            ymin, ymax = ydata.getPointRanges()

            # convert ymin and ymax to graph coordinates
            ymin = axes[1].graphToPlotterCoords(posn, ymin)
            ymax = axes[1].graphToPlotterCoords(posn, ymax)

            # clip...
            ymin = N.clip(ymin, posn[1]-ywc, posn[3]+ywc)
            ymax = N.clip(ymax, posn[1]-ywc, posn[3]+ywc)

            # draw lines between each of the points
        else:
            ymin = ymax = None

        # draw normal error bars
        style = s.errorStyle
        if style in {'bar':True, 'bardiamond':True,
                     'barcurve':True, 'barbox': True}:
            # list of output lines
            pts = []

            # vertical error bars
            if ymin is not None and ymax is not None and not s.ErrorBarLine.hideVert :
                for x1, y1, x2, y2 in itertools.izip(xplotter, ymin, xplotter,
                                                     ymax):
                    pts.append(qt4.QPointF(x1, y1))
                    pts.append(qt4.QPointF(x2, y2))

            # horizontal error bars
            if xmin is not None and xmax is not None and not s.ErrorBarLine.hideHorz:
                for x1, y1, x2, y2 in itertools.izip(xmin, yplotter, xmax,
                                                     yplotter):
                    pts.append(qt4.QPointF(x1, y1))
                    pts.append(qt4.QPointF(x2, y2))

            if len(pts) != 0:
                painter.drawLines(pts)

        # special error bars (only works with proper x and y errors)
        if ( ymin is not None and ymax is not None and xmin is not None and
             xmax is not None ):

            # draw boxes
            if style in {'box':True, 'barbox':True}:

                # non-filling brush
                painter.setBrush( qt4.QBrush() )

                for xmn, ymn, xmx, ymx in (
                    itertools.izip(xmin, ymin, xmax, ymax)):

                    painter.drawPolygon( qt4.QPointF(xmn, ymn),
                                         qt4.QPointF(xmx, ymn),
                                         qt4.QPointF(xmx, ymx),
                                         qt4.QPointF(xmn, ymx) )

            # draw diamonds
            elif style in {'diamond':True, 'bardiamond':True}:

                # non-filling brush
                painter.setBrush( qt4.QBrush() )

                for xp, yp, xmn, ymn, xmx, ymx in itertools.izip(
                    xplotter, yplotter, xmin, ymin, xmax, ymax):

                    painter.drawPolygon( qt4.QPointF(xmn, yp),
                                         qt4.QPointF(xp, ymx),
                                         qt4.QPointF(xmx, yp),
                                         qt4.QPointF(xp, ymn) )

            # draw curved errors
            elif style in {'curve':True, 'barcurve': True}:

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

    def _autoAxis(self, dataname, bounds):
        """Determine range of data."""

        data = self.settings.get(dataname).getData(self.document)
        if data:
            range = data.getRange()
            if range:
                bounds[0] = min( bounds[0], range[0] )
                bounds[1] = max( bounds[1], range[1] )

    def autoAxis(self, name, bounds):
        """Automatically determine the ranges of variable on the axes."""

        s = self.settings
        if name == s.xAxis:
            self._autoAxis( 'xData', bounds )
        elif name == s.yAxis:
            self._autoAxis( 'yData', bounds )

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
                xmin = axes[0].graphToPlotterCoords(posn, xmin)
                xmax = axes[0].graphToPlotterCoords(posn, xmax)
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
            xplotter = axes[0].graphToPlotterCoords(posn, xvals.data)
            yplotter = axes[1].graphToPlotterCoords(posn, yvals.data)

            # need to remove silly points as these stuff up output
            xplotter = N.clip(xplotter, -32767, 32767)
            yplotter = N.clip(yplotter, -32767, 32767)

            #print "Painting error bars"
            # plot errors bars
            if not s.ErrorBarLine.hide:
                painter.setPen( s.ErrorBarLine.makeQPen(painter) )
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

                utils.plotMarkers(painter, xplotter, yplotter, s.marker,
                                  markersize)

            # finally plot any labels
            if tvals and not s.Label.hide:
                self.drawLabels(painter, xplotter, yplotter, tvals, markersize)

        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate an x,y plotter
document.thefactory.register( PointPlotter )

###############################################################################

class TextLabel(GenericPlotter):

    """Add a text label to a graph."""

    typename = 'label'
    description = "Text label"
    allowedparenttypes = [graph.Graph]
    allowusercreation = True

    def __init__(self, parent, name=None):
        GenericPlotter.__init__(self, parent, name=name)
        s = self.settings

        # text labels don't need key symbols
        s.remove('key')

        s.add( setting.DatasetOrStr('label', '',
                                    descr='Text to show or text dataset',
                                    usertext='Label', datatype='text'), 0 )
        s.add( setting.DatasetOrFloatList('xPos', 0.5,
                                          descr='List of X coordinates or dataset',
                                          usertext='X position',
                                          formatting=False), 1 )
        s.add( setting.DatasetOrFloatList('yPos', 0.5,
                                          descr='List of Y coordinates or dataset',
                                          usertext='Y position',
                                          formatting=False), 2 )

        s.add( setting.Choice('positioning',
                              ['axes', 'relative'], 'relative',
                              descr='Use axes or fractional position to '
                              'place label',
                              usertext='Position mode',
                              formatting=False), 6)

        s.add( setting.Choice('alignHorz',
                              ['left', 'centre', 'right'], 'left',
                              descr="Horizontal alignment of label",
                              usertext='Horz alignment',
                              formatting=True), 7)
        s.add( setting.Choice('alignVert',
                              ['top', 'centre', 'bottom'], 'bottom',
                              descr='Vertical alignment of label',
                              usertext='Vert alignment',
                              formatting=True), 8)

        s.add( setting.Float('angle', 0.,
                             descr='Angle of the label in degrees',
                             usertext='Angle',
                             formatting=True), 9 )

        s.add( setting.Text('Text',
                            descr = 'Text settings',
                            usertext='Text'),
               pixmap = 'axislabel' )

        if type(self) == TextLabel:
            self.readDefaults()

    # convert text to alignments used by Renderer
    cnvtalignhorz = { 'left': -1, 'centre': 0, 'right': 1 }
    cnvtalignvert = { 'top': 1, 'centre': 0, 'bottom': -1 }

    def draw(self, parentposn, painter, outerbounds = None):
        """Draw the text label."""

        posn = GenericPlotter.draw(self, parentposn, painter,
                                   outerbounds=outerbounds)

        s = self.settings
        d = self.document

        # exit if hidden
        if s.hide or s.Text.hide:
            return

        text = s.get('label').getData(d)
        pointsX = s.get('xPos').getFloatArray(d)
        pointsY = s.get('yPos').getFloatArray(d)

        if s.positioning == 'axes':
            # translate xPos and yPos to plotter coordinates

            axes = self.parent.getAxes( (s.xAxis, s.yAxis) )
            if None in axes:
                return
            xp = axes[0].graphToPlotterCoords(posn, pointsX)
            yp = axes[1].graphToPlotterCoords(posn, pointsY)
        else:
            # work out fractions inside pos
            xp = posn[0] + (posn[2]-posn[0])*pointsX
            yp = posn[3] + (posn[1]-posn[3])*pointsY

        painter.beginPaintingWidget(self, parentposn)
        painter.save()
        textpen = s.get('Text').makeQPen()
        painter.setPen(textpen)
        font = s.get('Text').makeQFont(painter)

        for x, y, t in itertools.izip(xp, yp, itertools.cycle(text)):
            utils.Renderer( painter, font, x, y, t,
                            TextLabel.cnvtalignhorz[s.alignHorz],
                            TextLabel.cnvtalignvert[s.alignVert],
                            s.angle ).render()
        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate a text label
document.thefactory.register( TextLabel )
