#    Copyright (C) 2005 Jeremy S. Sanders
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

"""Contour plotting from 2d datasets.

Contour plotting requires that the veusz_helpers package is installed,
as a C routine (taken from matplotlib) is used to trace the contours.
"""

import itertools
import sys

import veusz.qtall as qt
import numarray as N

import veusz.setting as setting
import veusz.document as document

import plotters

class Contour(plotters.GenericPlotter):
    """A class which plots contours on a graph with a specified
    coordinate system."""

    typename='contour'
    allowusercreation=True
    description='Plot a 2d dataset as contours'

    def __init__(self, parent, name=None):
        """Initialise plotter with axes."""

        plotters.GenericPlotter.__init__(self, parent, name=name)

        # try to import contour helpers here
        try:
            from veusz.helpers._na_cntr import Cntr
        except ImportError:
            Cntr = None
            print >>sys.stderr,('WARNING: Veusz cannot import contour module\n'
                                'Please run python setup.py build\n'
                                'Contour support is disabled')
            
        self.Cntr = Cntr

        s = self.settings
        s.add( setting.Dataset('data', '',
                               dimensions = 2,
                               descr = 'Dataset to plot' ),
               0 )
        s.add( setting.FloatOrAuto('min', 'Auto',
                                   descr = 'Minimum value of contour scale'),
               1 )
        s.add( setting.FloatOrAuto('max', 'Auto',
                                   descr = 'Maximum value of contour scale'),
               2 )
        s.add( setting.Int('numLevels', 5,
                           minval = 1,
                           descr = 'Number of contour levels to plot'),
               3 )
        s.add( setting.Choice('scaling',
                              ['linear', 'sqrt', 'log', 'squared', 'manual'],
                              'linear',
                              descr = 'Scaling between contour levels'),
               4 )
        s.add( setting.FloatList('manualLevels',
                                 [],
                                 descr = 'Levels to use for manual scaling'),
               5 )
        s.add( setting.FloatList('levelsOut',
                                 [],
                                 descr = 'Levels used in the plot'),
               6, readonly=True )

        s.add( setting.LineSet('lines',
                               [('solid', '1pt', 'black', False)],
                               descr = 'Line styles to plot the contours '
                               'using'),
               7 )

        s.add( setting.FillSet('fills', [],
                               descr = 'Fill styles to plot between contours'),
               8 )

        # keep track of settings so we recalculate when necessary
        self.lastdataset = None
        self.contsettings = None

        # cached traced contours
        self._cachedcontours = None
        self._cachedpolygons = None

    def _calculateLevels(self):
        """Calculate contour levels from data and settings.

        Returns levels as 1d numarray
        """

        # get dataset
        s = self.settings
        d = self.document

        if s.data not in d.data:
            # this dataset doesn't exist
            minval = 0.
            maxval = 1.
        else:
            # scan data
            data = d.data[s.data]
            minval = data.data.min()
            maxval = data.data.max()

        # override if not auto
        if s.min != 'Auto':
            minval = s.min
        if s.max != 'Auto':
            maxval = s.max

        numlevels = s.numLevels
        scaling = s.scaling

        if numlevels == 1 and scaling != 'manual':
            # calculations below assume numlevels > 1
            levels = N.array([minval,])
        else:
            # trap out silly cases
            if minval == maxval:
                minval = 0.
                maxval = 1.
                
            # calculate levels for each scaling
            if scaling == 'linear':
                delta = (maxval - minval) / (numlevels-1)
                levels = minval + N.arange(numlevels)*delta
            elif scaling == 'sqrt':
                delta = N.sqrt(maxval - minval) / (numlevels-1)
                levels = minval + (N.arange(numlevels)*delta)**2
            elif scaling == 'log':
                delta = N.log(maxval - minval) / (numlevels-1)
                levels = minval + N.exp(N.arange(numlevels)*delta)
            elif scaling == 'squared':
                delta = (maxval - minval)**2 / (numlevels-1)
                levels = minval + N.sqrt(N.arange(numlevels)*delta)
            else:
                # manual
                levels = N.array(s.manualLevels)

        # for the user later
        s.levelsOut = list(levels)

        return levels

    def autoAxis(self, name, bounds):
        """Automatically determine the ranges of variable on the axes."""

        # this is copied from Image, probably should combine
        s = self.settings
        d = self.document

        # return if no data
        if s.data not in d.data:
            return

        # return if the dataset isn't two dimensional
        data = d.data[s.data]
        if data.dimensions != 2:
            return

        xrange = data.xrange
        yrange = data.yrange

        if name == s.xAxis:
            bounds[0] = min( bounds[0], xrange[0] )
            bounds[1] = max( bounds[1], xrange[1] )
        elif name == s.yAxis:
            bounds[0] = min( bounds[0], yrange[0] )
            bounds[1] = max( bounds[1], yrange[1] )

    def draw(self, parentposn, painter, outerbounds = None):
        """Draw the contours."""

        posn = plotters.GenericPlotter.draw(self, parentposn, painter,
                                            outerbounds = outerbounds)
        x1, y1, x2, y2 = posn
        s = self.settings
        d = self.document
        
        # get axes widgets
        axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

        # return if there's no proper axes
        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' or
             s.data not in d.data ):
            return

        # return if the dataset isn't two dimensional
        data = d.data[s.data]
        if data.dimensions != 2:
            return

        # delete cached polygons if no filling
        if len(s.fills) == 0:
            self._cachedpolygons = None

        # recalculate contours if image has changed
        # we also recalculate if the user has switched on fills
        contsettings = ( s.min, s.max, s.numLevels, s.scaling,
                         tuple(s.manualLevels) )

        if (data != self.lastdataset or contsettings != self.contsettings or
            (self._cachedpolygons == None and len(s.fills) != 0)):
            self.updateContours()
            self.lastdataset = data
            self.contsettings = contsettings

        # plot the precalculated contours
        painter.beginPaintingWidget(self, posn)
        self.plotContourFills(painter, posn, axes)
        self.plotContours(painter, posn, axes)
        painter.endPaintingWidget()

    def updateContours(self):
        """Update calculated contours."""

        s = self.settings
        d = self.document

        levels = self._calculateLevels()

        # find coordinates of image coordinate bounds
        data = d.data[s.data]
        rangex, rangey = data.getDataRanges()
        xw, yw = data.data.shape

        # calculate location of pixels...
        xpts = (0.5+N.arange(xw))*(rangex[1]-rangex[0])/xw + rangex[0]
        ypts = (0.5+N.arange(xw))*(rangey[1]-rangey[0])/yw + rangey[0]

        # convert 1D arrays into 2D
        y, x = N.indices( (yw, xw) )
        xpts = xpts[x]
        ypts = ypts[y]
        del x, y

        # iterate over the levels and trace the contours
        self._cachedcontours = None
        self._cachedpolygons = None

        if self.Cntr != None:
            c = self.Cntr(xpts, ypts, data.data)

            # trace the contour levels
            if len(s.lines) != 0:
                self._cachedcontours = []
                for level in levels:
                    linelist = c.trace(level)
                    self._cachedcontours.append(linelist)

            # trace the polygons between the contours
            if len(s.fills) != 0 and len(levels) > 1:
                self._cachedpolygons = []
                for level1, level2 in itertools.izip(levels[:-1], levels[1:]):
                    linelist = c.trace(level1, level2)
                    self._cachedpolygons.append(linelist)

    def plotContours(self, painter, posn, axes):
        """Plot the traced contours on the painter."""

        s = self.settings
        x1, y1, x2, y2 = posn

        # no lines cached as no line styles
        if self._cachedcontours == None:
            return

        # ensure plotting of contours does not go outside the area
        painter.save()
        painter.setClipRect( qt.QRect(x1, y1, x2-x1, y2-y1) )

        # iterate over each level, and list of lines
        for num, linelist in enumerate(self._cachedcontours):

            # move to the next line style
            painter.setPen(s.get('lines').makePen(painter, num))
                
            # iterate over each complete line of the contour
            for curve in linelist:
                # convert coordinates from graph to plotter
                xplt = axes[0].graphToPlotterCoords(posn, curve[0])
                yplt = axes[1].graphToPlotterCoords(posn, curve[1])

                # there should be a nice itertools way of doing this
                pts = []
                for x, y in itertools.izip(xplt, yplt):
                    pts.append(x)
                    pts.append(y)

                # actually draw the curve to the plotter
                painter.drawPolyline( qt.QPointArray(pts) )

        # remove clip region
        painter.restore()

    def plotContourFills(self, painter, posn, axes):
        """Plot the traced contours on the painter."""

        s = self.settings
        x1, y1, x2, y2 = posn

        # don't draw if there are no cached polygons
        if self._cachedpolygons == None:
            return

        # ensure plotting of contours does not go outside the area
        painter.save()
        painter.setClipRect( qt.QRect(x1, y1, x2-x1, y2-y1) )
        painter.setPen(qt.QPen(qt.Qt.NoPen))

        # iterate over each level, and list of lines
        for num, polylist in enumerate(self._cachedpolygons):

            # move to the next line style
            painter.setBrush(s.get('fills').makeBrush(num))
                
            # iterate over each complete line of the contour
            for poly in polylist:
                # convert coordinates from graph to plotter
                xplt = axes[0].graphToPlotterCoords(posn, poly[0])
                yplt = axes[1].graphToPlotterCoords(posn, poly[1])

                # there should be a nice itertools way of doing this
                pts = []
                for x, y in itertools.izip(xplt, yplt):
                    pts.append(x)
                    pts.append(y)

                # actually draw the curve to the plotter
                painter.drawPolygon( qt.QPointArray(pts) )

        # remove clip region
        painter.restore()

# allow the factory to instantiate a contour
document.thefactory.register( Contour )
