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
import numarray

import widget
import widgetfactory
import axis

import utils

def _trim(x, x1, x2):
    """Truncate x in range x1->x2."""
    if x < x1: return x1
    if x > x2: return x2
    return x

class GenericPlotter(widget.Widget):
    """Generic plotter."""

    typename='genericplotter'

    def __init__(self, parent, name=None, axis1=None, axis2=None):
        """Initialise object, setting axes."""
        widget.Widget.__init__(self, parent, name=name)

        self.axes = [None]*2
        self.setAxes(axis1, axis2)

    def _autoGetAxis(self, dirn):
        """Automatically get the first axis of the given direction from the parent."""
        # iterate over siblings
        for c in self.parent.getChildren():
            try:
                if c._getAxisDirection() == dirn:
                    return c.getName()
            except AttributeError:
                pass

        # controversial??
        # get here if there's no axis..., so we automatically make one!
        # FIXME: we don't check whether there's an exisiting child called x
        # or y
        
        if dirn == 0:
            n = 'x'
        else:
            n = 'y'
        a = axis.Axis(self.parent, name=n)
        a.direction = dirn
        return n

    def setAxes(self, axis1='notset', axis2='notset'):
        """Set axes. If specified as None, the values aren't changed."""

        # automatically look up an appropriate axis in the parent
        if axis1 == None:
            axis1 = self._autoGetAxis(0)
        if axis2 == None:
            axis2 = self._autoGetAxis(1)

        # If the axis isn't none, set it
        if axis1 != 'notset':
            self.axes[0] = axis1
        if axis2 != 'notset':
            self.axes[1] = axis2

    def getAxisVar(self, axisname):
        """Get the actual axis variable corresponding to the axisname."""
        return self.parent.getChild( axisname )

    def getAxes(self):
        """Get the axis names as a tuple."""
        return self.axes

########################################################################
        
class FunctionPlotter(GenericPlotter):
    """Function plotting class."""

    typename='function'
    
    def __init__(self, parent, function=None, name=None, axis1=None,
                 axis2=None):
        """Initialise plotter with axes."""

        GenericPlotter.__init__(self, parent,
                                axis1=axis1, axis2=axis2, name=name)

        # define environment to evaluate functions
        self.fnenviron = globals()
        exec 'from numarray import *' in self.fnenviron

        # function to be plotted
        self.addPref('function', 'string', 'x')
        # is this a fn of x rather than y?
        self.addPref('xfunc', 'int', 1)
        # how often to evaluate the function
        self.addPref('iter', 'int', 1)
        self.readPrefs()

        if function != None:
            self.function = function
        
        self.Line = utils.PreferencesPlotLine('FunctionPlotterLine')
        self.Fill1 = utils.PreferencesPlotFill('FunctionFill1')
        self.Fill2 = utils.PreferencesPlotFill('FunctionFill2')

        self.addSubPref('PlotLine', self.Line)
        self.addSubPref('Fill1', self.Fill1)
        self.addSubPref('Fill2', self.Fill2)

        utils.nextAutos()

    def getUserDescription(self):
        """User-friendly description."""
        if self.xfunc:
            t = 'y = '
        else:
            t = 'x = '

        return t+self.function

    def _fillYFn(self, painter, xpts, ypts, bounds, leftfill):
        """ Take the xpts and ypts, and fill above or below the line."""
        if len(xpts) == 0:
            return

        x1, y1, x2, y2 = bounds

        if leftfill:
            pts = [x1, y1]
        else:
            pts = [x2, y1]

        for x,y in zip(xpts, ypts):
            pts.append( _trim(x, x1, x2) )
            pts.append(y)

        if leftfill:
            pts.append(x2)
        else:
            pts.append(x1)
        pts.append(y2)

        painter.drawPolygon( qt.QPointArray(pts) )

    def _fillXFn(self, painter, xpts, ypts, bounds, belowfill):
        """ Take the xpts and ypts, and fill to left or right of the line."""
        if len(ypts) == 0:
            return

        x1, y1, x2, y2 = bounds

        if belowfill:
            pts = [x1, y2]
        else:
            pts = [x1, y1]

        for x,y in zip(xpts, ypts):
            pts.append(x)
            pts.append( _trim(y, y1, y2) )

        pts.append( x2 )
        if belowfill:
            pts.append( y2 )
        else:
            pts.append( y1 )

        painter.drawPolygon( qt.QPointArray(pts) )

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

    def draw(self, parentposn, painter):
        """Plot the function."""

        posn = GenericPlotter.draw(self, parentposn, painter)

        # the algorithm is to work out the fn for each pixel on the plot
        # need to convert pixels -> graph coord -> calc fn -> pixels

        x1, y1, x2, y2 = posn

        ax1 = self.getAxisVar( self.axes[0] )
        ax2 = self.getAxisVar( self.axes[1] )

        if self.xfunc:
            xplotter = numarray.arange(x1, x2+1, self.iter)
            self.fnenviron['x'] = ax1.plotterToGraphCoords(posn, xplotter)
            # HACK for constants
            y = eval( self.function + " + (0*x)", self.fnenviron )
            yplotter = ax2.graphToPlotterCoords(posn, y)
        else:
            yplotter = numarray.arange(y1, y2+1, self.iter)
            self.fnenviron['y'] = ax2.plotterToGraphCoords(posn, yplotter)
            # HACK for constants
            x = eval( self.function + " + (0*y)", self.fnenviron )
            xplotter = ax1.graphToPlotterCoords(posn, x)

        # here we go through the generated points, and plot those that
        # are in the plot (we can clip fairly easily).
        # each time there is a section we can plot, we plot it
        
        painter.save()
        painter.setPen( qt.QPen( qt.QColor(), 0, qt.Qt.NoPen ) )

        painter.setBrush( qt.QBrush(qt.QColor("darkcyan"),
                                    qt.Qt.Dense6Pattern) )
        self._fillXFn(painter, xplotter, yplotter, posn, 1)
        
        painter.setBrush( qt.QBrush() )
        painter.setPen( self.Line.makeQPen(painter) )
        self._plotLine(painter, xplotter, yplotter, posn)

        painter.restore()

# allow the factory to instantiate an function plotter
widgetfactory.thefactory.register( FunctionPlotter )

###############################################################################

def _getRangeCoords(val, sym, neg, pos):
    """Find the range of value for drawing errors."""
    minvals = val.copy()
    maxvals = val.copy()

    if sym != None:
        minvals -= sym
        maxvals += sym

    if neg != None:
        minvals += neg

    if pos != None:
        maxvals += pos

    return (minvals, maxvals)
        
class PointPlotter(GenericPlotter):
    """A class for plotting points and their errors."""

    typename='xy'
    
    def __init__(self, parent, xdata, ydata, name=None,
                 axis1=None, axis2=None):
        """Initialise XY plotter plotting (xdata, ydata).

        xdata and ydata are strings specifying the data in the document
        axis1 and axis2 are strings specifying the axis in the parent"""
        
        GenericPlotter.__init__(self, parent, axis1=axis1, axis2=axis2,
                                name=name)
        # FIXME: Add prefs here
        self.addPref('marker', 'string', 'O')
        self.addPref('markerSize', 'int', 5 )
        self.readPrefs()

        self.PlotLine = utils.PreferencesPlotLine( 'XYPlotLine' )
        self.MarkerLine = utils.PreferencesPlotLine( 'XYMarkerLine' )
        self.ErrorBarLine = utils.PreferencesPlotLine( 'XYErrorBarLine' )
        self.MarkerFill = utils.PreferencesPlotFill( 'XYMarkerFill' )

        self.addSubPref('PlotLine', self.PlotLine)
        self.addSubPref('MarkerLine', self.MarkerLine)
        self.addSubPref('ErrorBarLine', self.ErrorBarLine)
        self.addSubPref('MarkerFill', self.MarkerFill)

        utils.nextAutos()

        self.xdata = xdata
        self.ydata = ydata

    def setData(self, xdata, ydata):
        """Set the variables to be plotted.
        
        xdata and ydata are strings specifying the data in the document"""

        self.xdata = xdata
        self.ydata = ydata

    def _plotErrors(self, posn, painter, xplotter, yplotter):
        """Plot error bars (horizontal and vertical)."""

        # list of output lines
        pts = []

        # get the axes
        ax1 = self.getAxisVar( self.axes[0] )
        ax2 = self.getAxisVar( self.axes[1] )

        # get the data
        xval, xsym, xneg, xpos = self.getDocument().getDataAll(self.xdata)

        # draw horizontal error bars
        if xsym != None or xpos != None or xneg != None:
            xmin, xmax = _getRangeCoords( xval, xsym, xneg, xpos )
                    
            # convert xmin and xmax to graph coordinates
            xmin = ax1.graphToPlotterCoords(posn, xmin)
            xmax = ax1.graphToPlotterCoords(posn, xmax)

            # draw lines between each of the points
            for i in xrange( len(xmin) ):
                pts += [ xmin[i], yplotter[i], xmax[i], yplotter[i] ]

        # draw vertical error bars
        # get data
        yval, ysym, yneg, ypos = self.getDocument().getDataAll(self.ydata)

        if ysym != None or yneg != None or ypos != None:
            ymin, ymax = _getRangeCoords( yval, ysym, yneg, ypos )

            # convert ymin and ymax to graph coordinates
            ymin = ax2.graphToPlotterCoords(posn, ymin)
            ymax = ax2.graphToPlotterCoords(posn, ymax)

            # draw lines between each of the points
            for i in xrange( len(ymin) ):
                pts += [ xplotter[i], ymin[i], xplotter[i], ymax[i] ]

        # finally draw the lines
        if len(pts) != 0:
            painter.drawLineSegments( qt.QPointArray(pts) )
            
    def _autoAxis(self, dataname):
        """Determine range of data."""
        vals = self.getDocument().getDataAll(dataname)

        minvals, maxvals = _getRangeCoords( *vals )

        # find the range of the values
        minval = numarray.minimum.reduce(minvals)
        maxval = numarray.maximum.reduce(maxvals)

        return (minval, maxval)

    def autoAxis(self, name):
        """Automatically determine the ranges of variable on the axes."""
        
        axes = self.getAxes()
        if name == axes[0]:
            return self._autoAxis( self.xdata )
        elif name == axes[1]:
            return self._autoAxis( self.ydata )
        else:
            return None

    def _drawPlotLine( self, painter, xvals, yvals ):
        """Draw the line connecting the points."""

        pts = []
        for xpt, ypt in zip(xvals, yvals):
            pts.append(xpt)
            pts.append(ypt)

        painter.drawPolyline( qt.QPointArray(pts) )

    def draw(self, parentposn, painter):
        """Plot the data on a plotter."""

        posn = GenericPlotter.draw(self, parentposn, painter)
        x1, y1, x2, y2 = posn

        # clip data within bounds of plotter
        painter.save()
        painter.setClipRect( qt.QRect(x1, y1, x2-x1, y2-y1) )

        xvals = self.getDocument().getData(self.xdata)
        yvals = self.getDocument().getData(self.ydata)

        # no points to plot
        if xvals == None or yvals == None or \
               len(xvals) == 0 or len(yvals) == 0:
            return

        # get the axes
        ax1 = self.getAxisVar( self.axes[0] )
        ax2 = self.getAxisVar( self.axes[1] )

        # calc plotter coords of x and y points
        xplotter = ax1.graphToPlotterCoords(posn, xvals)
        yplotter = ax2.graphToPlotterCoords(posn, yvals)

        # plot data line
        if self.PlotLine.notHidden():
            painter.setPen( self.PlotLine.makeQPen(painter) )
            self._drawPlotLine( painter, xplotter, yplotter )

        # plot errors bars
        if self.ErrorBarLine.notHidden():
            painter.setPen( self.ErrorBarLine.makeQPen(painter) )
            self._plotErrors(posn, painter, xplotter, yplotter)

        # plot the points (we do this last so they are on top)
        if self.MarkerLine.notHidden():
            size = int( utils.getPixelsPerPoint(painter) * self.markerSize )

            if self.MarkerFill.notHidden():
                painter.setBrush( self.MarkerFill.makeQBrush() )
                
            painter.setPen( self.MarkerLine.makeQPen(painter) )
            utils.plotMarkers(painter, xplotter, yplotter, self.marker,
                              size)

        painter.restore()

# allow the factory to instantiate an x,y plotter
widgetfactory.thefactory.register( PointPlotter )

###############################################################################
