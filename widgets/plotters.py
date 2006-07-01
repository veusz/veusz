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
import numarray as N

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import widget
import graph

def _trim(x, x1, x2):
    """Truncate x in range x1->x2."""
    if x < x1: return x1
    if x > x2: return x2
    return x

class GenericPlotter(widget.Widget):
    """Generic plotter."""

    typename='genericplotter'
    allowedparenttypes=[graph.Graph]

    def __init__(self, parent, name=None):
        """Initialise object, setting axes."""
        widget.Widget.__init__(self, parent, name=name)

        s = self.settings
        s.add( setting.Str('key', '',
                           descr = 'Description of the plotted data') )
        s.add( setting.Axis('xAxis', 'x', 'horizontal',
                            descr = 'Name of X-axis to use') )
        s.add( setting.Axis('yAxis', 'y', 'vertical',
                            descr = 'Name of Y-axis to use') )

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
        painter.setClipRect( qt4.QRect(x1, y2, x2-x1, y1-y2) )

########################################################################
        
class FunctionPlotter(GenericPlotter):
    """Function plotting class."""

    typename='function'
    allowusercreation=True
    description='Plot a function'
    
    def __init__(self, parent, name=None):
        """Initialise plotter with axes."""

        GenericPlotter.__init__(self, parent, name=name)

        # define environment to evaluate functions
        self.fnenviron = globals()
        exec 'from numarray import *' in self.fnenviron

        s = self.settings
        s.add( setting.Int('steps', 50,
                           descr = 'Number of steps to evaluate the function'
                           ' over'), 0 )
        s.add( setting.Choice('variable', ['x', 'y'], 'x',
                              descr='Variable the function is a function of'),
               0 )
        s.add( setting.Str('function', 'x',
                           descr='Function expression'), 0 )

        s.add(setting.FloatOrAuto('min', 'Auto',
                             descr='Minimum value at which to plot function'))
        
        s.add(setting.FloatOrAuto('max', 'Auto',
                                  descr='Maximum value at which to plot function'))

        s.add( setting.Line('Line',
                            descr = 'Function line settings'),
               pixmap = 'plotline' )

        s.add( setting.PlotterFill('FillBelow',
                                   descr = 'Fill below function'),
               pixmap = 'plotfillbelow' )
        
        s.add( setting.PlotterFill('FillAbove',
                                   descr = 'Fill above function'),
               pixmap = 'plotfillabove' )

        if type(self) == FunctionPlotter:
            self.readDefaults()
        
    def _getUserDescription(self):
        """User-friendly description."""
        return "%s = %s" % ( self.settings.variable,
                             self.settings.function )
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
        xw = x2-x1
        xclip = N.clip(pxpts, x1-xw, x2+xw)
        yw = y2-y1
        yclip = N.clip(pypts, y1-yw, y2+yw)
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
            painter.drawLine(x, yp, x+width, yp)

    def initEnviron(self):
        """Initialise function evaluation environment each time."""
        return self.fnenviron.copy()

    def draw(self, parentposn, painter, outerbounds = None):
        """Draw the function."""

        posn = GenericPlotter.draw(self, parentposn, painter,
                                   outerbounds = outerbounds)
        x1, y1, x2, y2 = posn
        s = self.settings

        # get axes widgets
        axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

        # return if there's no proper axes
        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return
            
        env = self.initEnviron()
        if s.variable == 'x':
            if not(s.min == 'Auto') and s.min > axes[0].getPlottedRange()[0]:
                x_min = N.array([s.min])
                x1 = axes[0].graphToPlotterCoords(posn, x_min)[0]
            if not(s.max == 'Auto') and s.max < axes[0].getPlottedRange()[1]:
                x_max = N.array([s.max])
                x2 = axes[0].graphToPlotterCoords(posn, x_max)[0]
                
            # x function
            delta = (x2 - x1) / float(s.steps)
            pxpts = N.arange(x1, x2+delta, delta)
            x = axes[0].plotterToGraphCoords(posn, pxpts)
            env['x'] = x
            try:
                y = eval( s.function + ' + 0*x', env )
                bad = False
            except:
                bad = True
            else:
                pypts = axes[1].graphToPlotterCoords(posn, y)

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
            env['y'] = axes[1].plotterToGraphCoords(posn, pypts)
            try:
                x = eval( s.function + ' + 0*y', env )
                bad = False
            except:
                bad = True
            else:
                pxpts = axes[0].graphToPlotterCoords(posn, x)

        # clip data within bounds of plotter
        painter.beginPaintingWidget(self, posn)
        painter.save()
        self.clipAxesBounds(painter, axes, posn)

        # draw the function line
        if bad:
            # not sure how to deal with errors here
            painter.setPen( qt4.QColor('red') )
            f = qt4.QFont()
            f.setPointSize(20)
            painter.setFont(f)
            painter.drawText( qt4.QRect(x1, y1, x2-x1, y2-y1),
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
    
    def __init__(self, parent, name=None):
        """Initialise XY plotter plotting (xdata, ydata).

        xdata and ydata are strings specifying the data in the document"""
        
        GenericPlotter.__init__(self, parent, name=name)
        s = self.settings

        s.add( setting.Distance('markerSize', '3pt',
                                descr = 'Size of marker to plot'), 0 )
        s.add( setting.Marker('marker', 'circle',
                              descr = 'Type of marker to plot'), 0 )
        s.add( setting.Dataset('yData', 'y',
                               descr = 'Variable containing y data'), 0 )
        s.add( setting.Dataset('xData', 'x',
                               descr = 'Variable containing x data'), 0 )
        s.add( setting.Choice('errorStyle',
                              ['bar', 'box', 'diamond', 'curve',
                               'barbox', 'bardiamond', 'barcurve'], 'bar',
                              descr='Style of error bars to plot') )

        s.add( setting.XYPlotLine('PlotLine',
                                  descr = 'Plot line settings'),
               pixmap = 'plotline' )
        s.add( setting.Line('MarkerLine',
                            descr = 'Line around the marker settings'),
               pixmap = 'plotmarkerline' )
        s.add( setting.Brush('MarkerFill',
                             descr = 'Marker fill settings'),
               pixmap = 'plotmarkerfill' )
        s.add( setting.ErrorBarLine('ErrorBarLine',
                                    descr = 'Error bar line settings'),
               pixmap = 'ploterrorline' )
        s.add( setting.PlotterFill('FillBelow',
                                   descr = 'Fill below plot line'),
               pixmap = 'plotfillbelow' )
        s.add( setting.PlotterFill('FillAbove',
                                   descr = 'Fill above plot line'),
               pixmap = 'plotfillabove' )

        if type(self) == PointPlotter:
            self.readDefaults()

    def _getUserDescription(self):
        """User-friendly description."""

        s = self.settings
        return "x='%s', y='%s', marker='%s'" % (s.xData, s.yData,
                                                s.marker)
    userdescription = property(_getUserDescription)

    def _plotErrors(self, posn, painter, xplotter, yplotter,
                    axes):
        """Plot error bars (horizontal and vertical)."""

        # get the data
        s = self.settings
        xdata = self.document.getData(s.xData)

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
        # get data
        ydata = self.document.getData(s.yData)
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
            if ymin != None and ymax != None and not s.ErrorBarLine.hideVert :
                for x1, y1, x2, y2 in itertools.izip(xplotter, ymin, xplotter,
                                                     ymax):
                    pts.append(qt4.QPointF(x1, y1))
                    pts.append(qt4.QPointF(x2, y2))

            # horizontal error bars
            if xmin != None and xmax != None and not s.ErrorBarLine.hideHorz:
                for x1, y1, x2, y2 in itertools.izip(xmin, yplotter, xmax,
                                                     yplotter):
                    pts.append(qt4.QPointF(x1, y1))
                    pts.append(qt4.QPointF(x2, y2))

            if len(pts) != 0:
                painter.drawLines(pts)

        # special error bars (only works with proper x and y errors)
        if ( ymin != None and ymax != None and xmin != None and
             xmax != None ):

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
        if self.document.hasData(dataname):
            range = self.document.getData(dataname).getRange()
            bounds[0] = min( bounds[0], range[0] )
            bounds[1] = max( bounds[1], range[1] )

    def autoAxis(self, name, bounds):
        """Automatically determine the ranges of variable on the axes."""

        s = self.settings
        if name == s.xAxis:
            self._autoAxis( s.xData, bounds )
        elif name == s.yAxis:
            self._autoAxis( s.yData, bounds )

    def _drawPlotLine( self, painter, xvals, yvals, posn ):
        """Draw the line connecting the points."""

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
                pts.append(qt4.QPointF(x2, y2))
                pts.append(qt4.QPointF(x2, y2))
            
        # stepped line, with points in centre
        # this is complex as we can't use the mean of the plotter coords,
        #  as the axis could be log
        elif steps == 'centre':
            xv = self.document.getData(s.xData)
            axes = self.parent.getAxes( (s.xAxis, s.yAxis) )
            xcen = axes[0].graphToPlotterCoords(posn,
                                                0.5*(xv.data[:-1]+xv.data[1:]))

            for x1, x2, xc, y1, y2 in itertools.izip(xvals[:-1], xvals[1:],
                                                     xcen,
                                                     yvals[:-1], yvals[1:]):
                pts.append(qt4.QPointF(x1, y1))
                pts.append(qt4.QPointF(xc, y1))
                pts.append(qt4.QPointF(xc, y2))
                pts.append(qt4.QPointF(x2, y2))

        else:
            assert False

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
            painter.drawLine(x, yp, x+width, yp)

        # draw marker
        if not s.MarkerLine.hide or not s.MarkerFill.hide:
            size = int( s.get('markerSize').convert(painter) )

            if not s.MarkerFill.hide:
                painter.setBrush( s.MarkerFill.makeQBrush() )

            if not s.MarkerLine.hide:
                painter.setPen( s.MarkerLine.makeQPen(painter) )
            else:
                painter.setPen( qt4.QPen( qt4.Qt.NoPen ) )
                
            utils.plotMarker(painter, x+width/2, yp, s.marker, size)

    def draw(self, parentposn, painter, outerbounds=None):
        """Plot the data on a plotter."""

        posn = GenericPlotter.draw(self, parentposn, painter,
                                   outerbounds=outerbounds)
        x1, y1, x2, y2 = posn

        # skip if there's no data
        d = self.document
        s = self.settings
        if not d.hasData(s.xData) or not d.hasData(s.yData):
            return
        
        # get axes widgets
        axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

        # return if there's no proper axes
        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return

        xvals = d.getData(s.xData)
        yvals = d.getData(s.yData)

        # no points to plot
        if xvals.empty() or yvals.empty():
            return

        # clip data within bounds of plotter
        painter.beginPaintingWidget(self, posn)
        painter.save()
        self.clipAxesBounds(painter, axes, posn)

        # calc plotter coords of x and y points
        xplotter = axes[0].graphToPlotterCoords(posn, xvals.data)
        yplotter = axes[1].graphToPlotterCoords(posn, yvals.data)

        # plot data line (and/or filling above or below)
        if not s.PlotLine.hide or not s.FillAbove.hide or not s.FillBelow.hide:
            self._drawPlotLine( painter, xplotter, yplotter, posn )

        # plot errors bars
        if not s.ErrorBarLine.hide:
            painter.setPen( s.ErrorBarLine.makeQPen(painter) )
            self._plotErrors(posn, painter, xplotter, yplotter,
                             axes)

        # plot the points (we do this last so they are on top)
        if not s.MarkerLine.hide or not s.MarkerFill.hide:
            size = int( s.get('markerSize').convert(painter) )

            if not s.MarkerFill.hide:
                # filling for markers
                painter.setBrush( s.MarkerFill.makeQBrush() )
            else:
                # no-filling brush
                painter.setBrush( qt4.QBrush() )

            if not s.MarkerLine.hide:
                # edges of markers
                painter.setPen( s.MarkerLine.makeQPen(painter) )
            else:
                # invisible pen
                painter.setPen( qt4.QPen(qt4.Qt.NoPen) )
                
            utils.plotMarkers(painter, xplotter, yplotter, s.marker,
                              size)

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

        s.add( setting.Str('label', '',
                           descr='Text to show'), 0 )
        s.add( setting.Float('xPos', 0.5,
                             descr='x coordinate of the text'), 1 )
        s.add( setting.Float('yPos', 0.5,
                             descr='y coordinate of the text'), 2 )
        s.add( setting.Choice('positioning',
                              ['axes', 'relative'], 'relative',
                              descr='Use axes or fractional position to '
                              'place label'), 3)

        s.add( setting.Choice('alignHorz',
                              ['left', 'centre', 'right'], 'left',
                              descr="Horizontal alignment of label"), 4)
        s.add( setting.Choice('alignVert',
                              ['top', 'centre', 'bottom'], 'bottom',
                              descr='Vertical alignment of label'), 5)

        s.add( setting.Float('angle', 0.,
                             descr='Angle of the label in degrees'), 6 )

        s.add( setting.Text('Text',
                            descr = 'Text settings'),
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
        if s.positioning == 'axes':
            # translate xPos and yPos to plotter coordinates

            axes = self.parent.getAxes( (s.xAxis, s.yAxis) )
            if None in axes:
                return
            xp = axes[0].graphToPlotterCoords(posn, N.array( [s.xPos] ))[0]
            yp = axes[1].graphToPlotterCoords(posn, N.array( [s.yPos] ))[0]
        else:
            # work out fractions inside pos
            xp = posn[0] + (posn[2]-posn[0])*s.xPos
            yp = posn[3] + (posn[1]-posn[3])*s.yPos

        if not s.Text.hide:
            painter.beginPaintingWidget(self, parentposn)
            painter.save()
            textpen = s.get('Text').makeQPen()
            painter.setPen(textpen)
            font = s.get('Text').makeQFont(painter)

            utils.Renderer( painter, font, xp, yp,
                            s.label,
                            TextLabel.cnvtalignhorz[s.alignHorz],
                            TextLabel.cnvtalignvert[s.alignVert],
                            s.angle ).render()
            painter.restore()
            painter.endPaintingWidget()

# allow the factory to instantiate a text label
document.thefactory.register( TextLabel )
