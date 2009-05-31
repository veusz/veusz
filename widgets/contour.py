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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
###############################################################################

# $Id$

"""Contour plotting from 2d datasets.

Contour plotting requires that the veusz_helpers package is installed,
as a C routine (taken from matplotlib) is used to trace the contours.
"""

import itertools
import sys

import veusz.qtall as qt4
import numpy as N

import veusz.setting as setting
import veusz.document as document
import veusz.utils as utils

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
            from veusz.helpers._nc_cntr import Cntr
        except ImportError:
            Cntr = None
            print >>sys.stderr,('WARNING: Veusz cannot import contour module\n'
                                'Please run python setup.py build\n'
                                'Contour support is disabled')
            
        self.Cntr = Cntr

        s = self.settings
        s.add( setting.Dataset('data', '',
                               dimensions = 2,
                               descr = 'Dataset to plot',
                               usertext='Dataset'),
               0 )
        s.add( setting.FloatOrAuto('min', 'Auto',
                                   descr = 'Minimum value of contour scale',
                                   usertext='Min. value'),
               1 )
        s.add( setting.FloatOrAuto('max', 'Auto',
                                   descr = 'Maximum value of contour scale',
                                   usertext='Max. value'),
               2 )
        s.add( setting.Int('numLevels', 5,
                           minval = 1,
                           descr = 'Number of contour levels to plot',
                           usertext='Number levels'),
               3 )
        s.add( setting.Choice('scaling',
                              ['linear', 'sqrt', 'log', 'squared', 'manual'],
                              'linear',
                              descr = 'Scaling between contour levels',
                              usertext='Scaling'),
               4 )
        s.add( setting.FloatList('manualLevels',
                                 [],
                                 descr = 'Levels to use for manual scaling',
                                 usertext='Manual levels'),
               5 )
        s.add( setting.FloatList('levelsOut',
                                 [],
                                 descr = 'Levels used in the plot',
                                 usertext='Output levels'),
               6, readonly=True )

        s.add( setting.LineSet('lines',
                               [('solid', '1pt', 'black', False)],
                               descr = 'Line styles to plot the contours '
                               'using', usertext='Line styles',
                               formatting=True),
               7)

        s.add( setting.FillSet('fills', [],
                               descr = 'Fill styles to plot between contours',
                               usertext='Fill styles',
                               formatting=True),
               8 )

        s.add( setting.ContourLabel('ContourLabels',
                                    descr = 'Contour label settings',
                                    usertext = 'Contour labels'),
               pixmap = 'axisticklabels' )

         # keep track of settings so we recalculate when necessary
        self.lastdataset = None
        self.contsettings = None

        # cached traced contours
        self._cachedcontours = None
        self._cachedpolygons = None

    def _getUserDescription(self):
        """User friendly description."""
        s = self.settings
        out = []
        if s.data:
            out.append( s.data )
        if s.scaling == 'manual':
            out.append('manual levels (%s)' %  (', '.join([str(i) for i in s.manualLevels])))
        else:
            out.append('%(numLevels)i %(scaling)s levels (%(min)s to %(max)s)' % s)
        return ', '.join(out)
    userdescription = property(_getUserDescription)

    def _calculateLevels(self):
        """Calculate contour levels from data and settings.

        Returns levels as 1d numpy
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
        # we do this to convert array to list of floats
        s.levelsOut = [float(i) for i in levels]

        return levels

    def providesAxesDependency(self):
        """Range information provided by widget."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def updateAxisRange(self, axis, depname, axrange):
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

        if depname == 'sx':
            dxrange = data.xrange
            axrange[0] = min( axrange[0], dxrange[0] )
            axrange[1] = max( axrange[1], dxrange[1] )
        elif depname == 'sy':
            dyrange = data.yrange
            axrange[0] = min( axrange[0], dyrange[0] )
            axrange[1] = max( axrange[1], dyrange[1] )

    def draw(self, parentposn, painter, outerbounds = None):
        """Draw the contours."""

        posn = plotters.GenericPlotter.draw(self, parentposn, painter,
                                            outerbounds = outerbounds)
        s = self.settings
        d = self.document

        # do not paint if hidden
        if s.hide:
            return
        
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
            (self._cachedpolygons is None and len(s.fills) != 0)):
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
        yw, xw = data.data.shape

        # arrays containing coordinates of pixels in x and y
        xpts = N.fromfunction(lambda y,x:
                              (x+0.5)*((rangex[1]-rangex[0])/xw) + rangex[0],
                              (yw, xw))
        ypts = N.fromfunction(lambda y,x:
                              (y+0.5)*((rangey[1]-rangey[0])/yw) + rangey[0],
                              (yw, xw))

        # iterate over the levels and trace the contours
        self._cachedcontours = None
        self._cachedpolygons = None

        if self.Cntr is not None:
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

    def plotContourLabel(self, painter, number, xplt, yplt):
        s = self.settings
        cl = s.get('ContourLabels')

        painter.save()

        # get text and font
        text = utils.formatNumber(number * cl.scale,
                                  s.ContourLabels.format)
        font = cl.makeQFont(painter)
        descent = qt4.QFontMetrics(font).descent()

        # work out where text lies
        half = len(xplt)/2
        hx, hy = xplt[half], yplt[half]
        r = utils.Renderer(painter, font, hx, hy, text, alignhorz=0,
                           alignvert=0, angle=0)
        bounds = r.getBounds()

        # heuristics of when to plot label
        # we try to only plot label if underlying line is long enough
        height = bounds[3]-bounds[1]
        showtext = ( height*1.5 < (yplt.max() - yplt.min()) or
                     (height*4 < (xplt.max() - xplt.min())) )

        if showtext:
            # clip region containing text
            oldclip = painter.clipRegion()
            cr = oldclip - qt4.QRegion( bounds[0]-descent, bounds[1]-descent,
                                        bounds[2]-bounds[0]+descent*2,
                                        bounds[3]-bounds[1]+descent*2 )
            painter.setClipRegion(cr)

        # draw lines
        pts = qt4.QPolygonF()
        for x, y in itertools.izip(xplt, yplt):
            pts.append( qt4.QPointF(x, y) )
        painter.drawPolyline(pts)

        # actually plot the label
        if showtext:
            painter.setClipRegion(oldclip)
            painter.setPen( cl.makeQPen() )
            r.render()

        painter.restore()

    def plotContours(self, painter, posn, axes):
        """Plot the traced contours on the painter."""

        s = self.settings

        # no lines cached as no line styles
        if self._cachedcontours is None:
            return

        # ensure plotting of contours does not go outside the area
        painter.save()
        self.clipAxesBounds(painter, axes, posn)

        showlabels = not s.ContourLabels.hide

        # iterate over each level, and list of lines
        for num, linelist in enumerate(self._cachedcontours):

            # move to the next line style
            painter.setPen(s.get('lines').makePen(painter, num))
                
            # iterate over each complete line of the contour
            for curve in linelist:
                # convert coordinates from graph to plotter
                xplt = axes[0].dataToPlotterCoords(posn, curve[:,0])
                yplt = axes[1].dataToPlotterCoords(posn, curve[:,1])
                    
                # there should be a nice itertools way of doing this
                pts = qt4.QPolygonF()
                for x, y in itertools.izip(xplt, yplt):
                    pts.append( qt4.QPointF(x, y) )

                if showlabels:
                    self.plotContourLabel(painter, s.levelsOut[num], xplt, yplt)
                else:
                    # actually draw the curve to the plotter
                    painter.drawPolyline(pts)

        # remove clip region
        painter.restore()

    def plotContourFills(self, painter, posn, axes):
        """Plot the traced contours on the painter."""

        s = self.settings

        # don't draw if there are no cached polygons
        if self._cachedpolygons is None:
            return

        # ensure plotting of contours does not go outside the area
        painter.save()
        self.clipAxesBounds(painter, axes, posn)
        painter.setPen(qt4.QPen(qt4.Qt.NoPen))

        # iterate over each level, and list of lines
        for num, polylist in enumerate(self._cachedpolygons):

            # move to the next line style
            painter.setBrush(s.get('fills').makeBrush(num))
                
            # iterate over each complete line of the contour
            for poly in polylist:
                # convert coordinates from graph to plotter
                xplt = axes[0].dataToPlotterCoords(posn, poly[:,0])
                yplt = axes[1].dataToPlotterCoords(posn, poly[:,1])

                # there should be a nice itertools way of doing this
                pts = qt4.QPolygonF()
                for x, y in itertools.izip(xplt, yplt):
                    pts.append( qt4.QPointF(x, y) )

                # actually draw the curve to the plotter
                painter.drawPolygon(pts)

        # remove clip region
        painter.restore()

# allow the factory to instantiate a contour
document.thefactory.register( Contour )
