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

import qt
import numarray as N

import widget
import widgetfactory
import graph
import setting

import utils

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
        s.add( setting.Str('xAxis', 'x',
                           descr = 'Name of X-axis to use') )
        s.add( setting.Str('yAxis', 'y',
                           descr = 'Name of Y-axis to use') )

    def getAxes(self):
        """Get the axes widgets to plot against."""

        xaxis = None
        yaxis = None

        x = self.settings.xAxis
        y = self.settings.yAxis

        # recursively go back up the tree to find axes 
        parent = self.parent
        while parent != None and (xaxis == None or yaxis == None):
            for i in parent.children:
                if i.name == x and xaxis == None:
                    xaxis = i
                if i.name == y and yaxis == None:
                    yaxis = i
            parent = parent.parent

        return (xaxis, yaxis)

    def drawKeySymbol(self, painter, x, y, width, height):
        """Draw the plot symbol and/or line at (x,y) in a box width*height.

        This is used to plot a key
        """
        pass

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

        s.add( setting.Line('Line',
                            descr = 'Function line settings') )

        s.readDefaults()
        
    def _getUserDescription(self):
        """User-friendly description."""
        return "%s = %s" % ( self.settings.variable,
                             self.settings.function )
    userdescription = property(_getUserDescription)

    def _plotLine(self, painter, xpts, ypts, bounds):
        """ Plot the points in xpts, ypts."""
        x1, y1, x2, y2 = bounds

        # idea is to collect points until we go out of the bounds
        # or reach the end, then plot them
        pts = []
        for x, y in zip(xpts, ypts):

            if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                pts.append(x)
                pts.append(y)
            else:
                if len(pts) >= 4:
                    painter.drawPolyline( qt.QPointArray(pts) )
                    pts = []

        if len(pts) >= 4:
            painter.drawPolyline( qt.QPointArray(pts) )

    def drawKeySymbol(self, painter, x, y, width, height):
        """Draw the plot symbol and/or line."""

        s = self.settings
        yp = y + height/2

        # draw line
        if not s.Line.hide:
            painter.setBrush( qt.QBrush() )
            painter.setPen( s.Line.makeQPen(painter) )
            painter.drawLine(x, yp, x+width, yp)

    def getKeySymbolWidth(self, height):
        """Get preferred width of key symbol of height."""

        if not self.settings.Line.hide:
            return 3*height
        else:
            return height

    def initEnviron(self):
        """Initialise function evaluation environment each time."""
        return self.fnenviron.copy()

    def draw(self, parentposn, painter):
        """Draw the function."""

        posn = GenericPlotter.draw(self, parentposn, painter)
        x1, y1, x2, y2 = posn

        # get axes widgets
        axes = self.getAxes()

        # return if there's no proper axes
        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return

        s = self.settings
        env = self.initEnviron()
        if s.variable == 'x':
            # x function
            delta = (x2 - x1) / s.steps
            pxpts = N.arange(x1, x2+delta, delta).astype(N.Int32)
            env['x'] = axes[0].plotterToGraphCoords(posn, pxpts)
            try:
                y = eval( s.function + ' + 0*x', env )
                bad = False
            except:
                bad = True
            else:
                pypts = axes[1].graphToPlotterCoords(posn, y)

        else:
            # y function
            delta = (y2 - y1) / s.steps
            pypts = N.arange(y1, y2+delta, delta).astype(N.Int32)
            env['y'] = axes[1].plotterToGraphCoords(posn, pypts)
            try:
                x = eval( s.function + ' + 0*y', env )
                bad = False
            except:
                bad = True
            else:
                pxpts = axes[0].graphToPlotterCoords(posn, x)

        painter.save()

        # draw the function line
        if not s.Line.hide and not bad:
            painter.setBrush( qt.QBrush() )
            painter.setPen( s.Line.makeQPen(painter) )
            self._plotLine(painter, pxpts, pypts, posn)

        if bad:
            # not sure how to deal with errors here
            painter.setPen( qt.QColor('red') )
            f = qt.QFont()
            f.setPointSize(20)
            painter.setFont(f)
            painter.drawText( qt.QRect(x1, y1, x2-x1, y2-y1),
                              qt.Qt.AlignCenter,
                              "Cannot evaluate '%s'" % s.function )

        painter.restore()

##     def _fillYFn(self, painter, xpts, ypts, bounds, leftfill):
##         """ Take the xpts and ypts, and fill above or below the line."""
##         if len(xpts) == 0:
##             return

##         x1, y1, x2, y2 = bounds

##         if leftfill:
##             pts = [x1, y1]
##         else:
##             pts = [x2, y1]

##         for x,y in zip(xpts, ypts):
##             pts.append( _trim(x, x1, x2) )
##             pts.append(y)

##         if leftfill:
##             pts.append(x2)
##         else:
##             pts.append(x1)
##         pts.append(y2)

##         painter.drawPolygon( qt.QPointArray(pts) )

##     def _fillXFn(self, painter, xpts, ypts, bounds, belowfill):
##         """ Take the xpts and ypts, and fill to left or right of the line."""
##         if len(ypts) == 0:
##             return

##         x1, y1, x2, y2 = bounds

##         if belowfill:
##             pts = [x1, y2]
##         else:
##             pts = [x1, y1]

##         for x,y in zip(xpts, ypts):
##             pts.append(x)
##             pts.append( _trim(y, y1, y2) )

##         pts.append( x2 )
##         if belowfill:
##             pts.append( y2 )
##         else:
##             pts.append( y1 )

##         painter.drawPolygon( qt.QPointArray(pts) )

##     def draw(self, parentposn, painter):
##         """Plot the function."""

##         posn = GenericPlotter.draw(self, parentposn, painter)

##         # the algorithm is to work out the fn for each pixel on the plot
##         # need to convert pixels -> graph coord -> calc fn -> pixels

##         x1, y1, x2, y2 = posn

##         ax1 = self.getAxisVar( self.axes[0] )
##         ax2 = self.getAxisVar( self.axes[1] )

##         if self.xfunc:
##             xplotter = numarray.arange(x1, x2+1, self.iter)
##             self.fnenviron['x'] = ax1.plotterToGraphCoords(posn, xplotter)
##             # HACK for constants
##             y = eval( self.function + " + (0*x)", self.fnenviron )
##             yplotter = ax2.graphToPlotterCoords(posn, y)
##         else:
##             yplotter = numarray.arange(y1, y2+1, self.iter)
##             self.fnenviron['y'] = ax2.plotterToGraphCoords(posn, yplotter)
##             # HACK for constants
##             x = eval( self.function + " + (0*y)", self.fnenviron )
##             xplotter = ax1.graphToPlotterCoords(posn, x)

##         # here we go through the generated points, and plot those that
##         # are in the plot (we can clip fairly easily).
##         # each time there is a section we can plot, we plot it
        
##         painter.save()
##         painter.setPen( qt.QPen( qt.QColor(), 0, qt.Qt.NoPen ) )

##         painter.setBrush( qt.QBrush(qt.QColor("darkcyan"),
##                                     qt.Qt.Dense6Pattern) )
##         self._fillXFn(painter, xplotter, yplotter, posn, 1)
        
##         painter.setBrush( qt.QBrush() )
##         painter.setPen( self.Line.makeQPen(painter) )
##         self._plotLine(painter, xplotter, yplotter, posn)

##         painter.restore()

# allow the factory to instantiate an function plotter
widgetfactory.thefactory.register( FunctionPlotter )

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
        s.add( setting.Distance('markerSize', '3pt'), 0 )
        s.add( setting.Choice('marker', utils.MarkerCodes, 'circle'), 0 )
        s.add( setting.Str('yData', 'y',
                           descr = 'Variable containing y data'), 0 )
        s.add( setting.Str('xData', 'x',
                           descr = 'Variable containing x data'), 0 )
        s.readDefaults()

        s.add( setting.XYPlotLine('PlotLine',
                                  descr = 'Plot line settings') )
        s.add( setting.Line('MarkerLine',
                            descr = 'Line around the marker settings') )
        s.add( setting.Brush('MarkerFill',
                             descr = 'Marker fill settings') )
        s.add( setting.Line('ErrorBarLine',
                            descr = 'Error bar line settings') )

    def _getUserDescription(self):
        """User-friendly description."""

        s = self.settings
        return "x='%s', y='%s', marker='%s'" % (s.xData, s.yData,
                                                s.marker)
    userdescription = property(_getUserDescription)

    def _plotErrors(self, posn, painter, xplotter, yplotter,
                    axes):
        """Plot error bars (horizontal and vertical)."""

        # list of output lines
        pts = []

        # get the data
        xdata = self.document.getData(self.settings.xData)

        # draw horizontal error bars
        if xdata.hasErrors():
            xmin, xmax = xdata.getPointRanges()
                    
            # convert xmin and xmax to graph coordinates
            xmin = axes[0].graphToPlotterCoords(posn, xmin)
            xmax = axes[0].graphToPlotterCoords(posn, xmax)

            # clip... (avoids problems with INFs, etc)
            xmin = N.clip(xmin, posn[0]-1, posn[2]+1)
            xmax = N.clip(xmax, posn[0]-1, posn[2]+1)

            # draw lines between each of the points
            for i in zip(xmin, yplotter, xmax, yplotter):
                pts += i

        # draw vertical error bars
        # get data
        ydata = self.document.getData(self.settings.yData)
        if ydata.hasErrors():
            ymin, ymax = ydata.getPointRanges()

            # convert ymin and ymax to graph coordinates
            ymin = axes[1].graphToPlotterCoords(posn, ymin)
            ymax = axes[1].graphToPlotterCoords(posn, ymax)

            # clip...
            ymin = N.clip(ymin, posn[1]-1, posn[3]+1)
            ymax = N.clip(ymax, posn[1]-1, posn[3]+1)

            # draw lines between each of the points
            for i in zip(xplotter, ymin, xplotter, ymax):
                pts += i

        if len(pts) != 0:
            painter.drawLineSegments( qt.QPointArray(pts) )

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

        pts = []

        s = self.settings
        steps = s.PlotLine.steps

        # simple continuous line
        if steps == 'off':
            for xpt, ypt in zip(xvals, yvals):
                pts.append(xpt)
                pts.append(ypt)

        # stepped line, with points on left
        elif steps == 'left':
            for x1, x2, y1, y2 in zip(xvals[:-1], xvals[1:],
                                      yvals[:-1], yvals[1:]):
                pts += [x1, y1, x2, y1, x2, y2]

        # stepped line, with points on right
        elif steps == 'right':
            for x1, x2, y1, y2 in zip(xvals[:-1], xvals[1:],
                                      yvals[:-1], yvals[1:]):
                pts += [x1, y1, x1, y2, x2, y2]
            
        # stepped line, with points in centre
        # this is complex as we can't use the mean of the plotter coords,
        #  as the axis could be log
        elif steps == 'centre':
            xv = self.document.getData(s.xData)
            axes = self.getAxes()
            xcen = axes[0].graphToPlotterCoords(posn,
                                                0.5*(xv.data[:-1]+xv.data[1:]))

            for x1, x2, xc, y1, y2 in zip(xvals[:-1], xvals[1:], xcen,
                                          yvals[:-1], yvals[1:]):
                pts += [x1, y1, xc, y1, xc, y2, x2, y2]

        else:
            assert 0

        painter.drawPolyline( qt.QPointArray(pts) )

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
            size = int( utils.cnvtDist(s.markerSize, painter) )

            if not s.MarkerFill.hide:
                painter.setBrush( s.MarkerFill.makeQBrush() )

            if not s.MarkerLine.hide:
                painter.setPen( s.MarkerLine.makeQPen(painter) )
            else:
                painter.setPen( qt.QPen( qt.Qt.NoPen ) )
                
            utils.plotMarker(painter, x+width/2, yp, s.marker, size)

    def getKeySymbolWidth(self, height):
        """Get preferred width of key symbol of height."""

        if not self.settings.PlotLine.hide:
            return 3*height
        else:
            return height

    def draw(self, parentposn, painter):
        """Plot the data on a plotter."""

        posn = GenericPlotter.draw(self, parentposn, painter)
        x1, y1, x2, y2 = posn

        # skip if there's no data
        d = self.document
        s = self.settings
        if not d.hasData(s.xData) or not d.hasData(s.yData):
            return
        
        # get axes widgets
        axes = self.getAxes()

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
        painter.save()
        painter.setClipRect( qt.QRect(x1, y1, x2-x1, y2-y1) )

        # calc plotter coords of x and y points
        xplotter = axes[0].graphToPlotterCoords(posn, xvals.data)
        yplotter = axes[1].graphToPlotterCoords(posn, yvals.data)

        # plot data line
        if not s.PlotLine.hide:
            painter.setPen( s.PlotLine.makeQPen(painter) )
            self._drawPlotLine( painter, xplotter, yplotter, posn )

        # plot errors bars
        if not s.ErrorBarLine.hide:
            painter.setPen( s.ErrorBarLine.makeQPen(painter) )
            self._plotErrors(posn, painter, xplotter, yplotter,
                             axes)

        # plot the points (we do this last so they are on top)
        if not s.MarkerLine.hide or not s.MarkerFill.hide:
            size = int( utils.cnvtDist(s.markerSize, painter) )

            if not s.MarkerFill.hide:
                painter.setBrush( s.MarkerFill.makeQBrush() )

            if not s.MarkerLine.hide:
                painter.setPen( s.MarkerLine.makeQPen(painter) )
            else:
                painter.setPen( qt.QPen( qt.Qt.NoPen ) )
                
            utils.plotMarkers(painter, xplotter, yplotter, s.marker,
                              size)

        painter.restore()

# allow the factory to instantiate an x,y plotter
widgetfactory.thefactory.register( PointPlotter )

###############################################################################
