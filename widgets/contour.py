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

from itertools import izip
import sys

import veusz.qtall as qt4
import numpy as N

import veusz.setting as setting
import veusz.document as document
import veusz.utils as utils

import plotters

def finitePoly(poly):
    """Remove non-finite coordinates from numpy arrays of coordinates."""
    out = []
    for line in poly:
        finite = N.isfinite(line)
        validrows = N.logical_and(finite[:,0], finite[:,1])
        out.append( line[validrows] )
    return out

class ContourFills(setting.Settings):
    """Settings for contour fills."""
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.FillSet(
                'fills', [],
                descr = 'Fill styles to plot between contours',
                usertext='Fill styles',
                formatting=True) )
        self.add( setting.Bool('hide', False,
                               descr = 'Hide fills',
                               usertext = 'Hide',
                               formatting = True) )
        
class ContourLines(setting.Settings):
    """Settings for contour lines."""
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.LineSet(
                'lines',
                [('solid', '1pt', 'black', False)],
                descr = 'Line styles to plot the contours '
                'using', usertext='Line styles',
                formatting=True) )
        self.add( setting.Bool('hide', False,
                               descr = 'Hide lines',
                               usertext = 'Hide',
                               formatting = True) )

class SubContourLines(setting.Settings):
    """Sub-dividing contour line settings."""
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.LineSet(
                'lines',
                [('dot1', '1pt', 'black', False)],
                descr = 'Line styles used for sub-contours',
                usertext='Line styles',
                formatting=True) )
        self.add( setting.Int('numLevels', 5,
                              minval=2,
                              descr='Number of sub-levels to plot between '
                              'each contour',
                              usertext='Levels') )
        self.add( setting.Bool('hide', True,
                               descr='Hide lines',
                               usertext='Hide',
                               formatting=True) )

class ContourLabel(setting.Text):
    """For tick labels on axes."""

    def __init__(self, name, **args):
        setting.Text.__init__(self, name, **args)
        self.add( setting.Str( 'format', '%.3Vg',
                               descr = 'Format of the tick labels',
                               usertext='Format') )
        self.add( setting.Float('scale', 1.,
                                descr='A scale factor to apply to the values '
                                'of the tick labels',
                                usertext='Scale') )

        self.get('hide').newDefault(True)

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
        Cntr = None
        try:
            from veusz.helpers._nc_cntr import Cntr
        except ImportError:
            print >>sys.stderr,('WARNING: Veusz cannot import contour module\n'
                                'Please run python setup.py build\n'
                                'Contour support is disabled')
            
        self.Cntr = Cntr
        # keep track of settings so we recalculate when necessary
        self.lastdataset = None
        self.contsettings = None

        # cached traced contours
        self._cachedcontours = None
        self._cachedpolygons = None
        self._cachedsubcontours = None

        if type(self) == Contour:
            self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.GenericPlotter.addSettings(s)

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

        s.add( setting.Bool('keyLevels', False, descr='Show levels in key',
                            usertext='Levels in key'),
               6 )

        s.add( setting.FloatList('levelsOut',
                                 [],
                                 descr = 'Levels used in the plot',
                                 usertext='Output levels'),
               7, readonly=True )

        s.add( ContourLabel('ContourLabels',
                            descr = 'Contour label settings',
                            usertext = 'Contour labels'),
               pixmap = 'settings_axisticklabels' )

        s.add( ContourLines('Lines',
                            descr='Contour lines',
                            usertext='Contour lines'),
               pixmap = 'settings_contourline' )

        s.add( ContourFills('Fills',
                            descr='Fill within contours',
                            usertext='Contour fills'),
               pixmap = 'settings_contourfill' )

        s.add( SubContourLines('SubLines',
                               descr='Sub-contour lines',
                               usertext='Sub-contour lines'),
               pixmap = 'settings_subcontourline' )

        s.add( setting.SettingBackwardCompat('lines', 'Lines/lines', None) )
        s.add( setting.SettingBackwardCompat('fills', 'Fills/fills', None) )

        s.remove('key')

    def _getUserDescription(self):
        """User friendly description."""
        s = self.settings
        out = []
        if s.data:
            out.append( s.data )
        if s.scaling == 'manual':
            out.append('manual levels (%s)' %  (
                    ', '.join([str(i) for i in s.manualLevels])))
        else:
            out.append('%(numLevels)i %(scaling)s levels (%(min)s to %(max)s)' % s)
        return ', '.join(out)
    userdescription = property(_getUserDescription)

    def calculateLevels(self):
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
                delta = N.log(maxval/minval) / (numlevels-1)
                levels = N.exp(N.arange(numlevels)*delta)*minval
            elif scaling == 'squared':
                delta = (maxval - minval)**2 / (numlevels-1)
                levels = minval + N.sqrt(N.arange(numlevels)*delta)
            else:
                # manual
                levels = N.array(s.manualLevels)

        # for the user later
        # we do this to convert array to list of floats
        s.levelsOut = [float(i) for i in levels]

        return minval, maxval, levels

    def calculateSubLevels(self, minval, maxval, levels):
        """Calculate sublevels between contours."""
        s = self.settings
        num = s.SubLines.numLevels
        if s.SubLines.hide or len(s.SubLines.lines) == 0:
            return N.array([])

        # indices where contour levels should be placed
        numcont = (len(levels)-1) * num
        indices = N.arange(numcont)
        indices = indices[indices % num != 0]

        scaling = s.scaling
        if scaling == 'linear':
            delta = (maxval-minval) / numcont
            slev = indices*delta + minval
        elif scaling == 'log':
            delta = N.log( maxval/minval ) / numcont
            slev = N.exp(indices*delta) * minval
        elif scaling == 'sqrt':
            delta = N.sqrt( maxval-minval ) / numcont
            slev = minval + (indices*delta)**2
        elif scaling == 'squared':
            delta = (maxval-minval)**2 / numcont
            slev = minval + N.sqrt(indices*delta)
        elif scaling == 'manual':
            drange = N.arange(1, num)
            out = [[]]
            for conmin, conmax in izip(levels[:-1], levels[1:]):
                delta = (conmax-conmin) / num
                out.append( conmin+drange*delta )
            slev = N.hstack(out)

        return slev

    def providesAxesDependency(self):
        """Range information provided by widget."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def updateAxisRange(self, axis, depname, axrange):
        """Automatically determine the ranges of variable on the axes."""

        # this is copied from Image, probably should combine
        s = self.settings
        d = self.document

        # return if no data or if the dataset isn't two dimensional
        data = d.data.get(s.data, None)
        if data is None or data.dimensions != 2:
            return

        if depname == 'sx':
            dxrange = data.xrange
            axrange[0] = min( axrange[0], dxrange[0] )
            axrange[1] = max( axrange[1], dxrange[1] )
        elif depname == 'sy':
            dyrange = data.yrange
            axrange[0] = min( axrange[0], dyrange[0] )
            axrange[1] = max( axrange[1], dyrange[1] )

    def getNumberKeys(self):
        """How many keys to show."""
        self.checkContoursUpToDate()
        if self.settings.keyLevels:
            return len( self.settings.levelsOut )
        else:
            return 0

    def getKeyText(self, number):
        """Get key entry."""
        s = self.settings
        if s.keyLevels:
            cl = s.get('ContourLabels')
            return utils.formatNumber( s.levelsOut[number] * cl.scale,
                                       cl.format )
        else:
            return ''

    def drawKeySymbol(self, number, painter, x, y, width, height):
        """Draw key for contour level."""
        painter.setPen(
            self.settings.Lines.get('lines').makePen(painter, number))
        painter.drawLine(x, y+height/2, x+width, y+height/2)

    def checkContoursUpToDate(self):
        """Update contours if necessary.
        Returns True if okay to plot contours, False if error
        """

        s = self.settings
        d = self.document

        # return if no data or if the dataset isn't two dimensional
        data = d.data.get(s.data, None)
        if data is None or data.dimensions != 2:
            self.contsettings = self.lastdataset = None
            s.levelsOut = []
            return False

        contsettings = ( s.min, s.max, s.numLevels, s.scaling,
                         len(s.Fills.fills) == 0 or s.Fills.hide,
                         len(s.SubLines.lines) == 0 or s.SubLines.hide,
                         tuple(s.manualLevels) )

        if data is not self.lastdataset or contsettings != self.contsettings:
            self.updateContours()
            self.lastdataset = data
            self.contsettings = contsettings

        return True

    def draw(self, parentposn, painter, outerbounds = None):
        """Draw the contours."""

        posn = plotters.GenericPlotter.draw(self, parentposn, painter,
                                            outerbounds = outerbounds)
        s = self.settings

        # do not paint if hidden
        if s.hide:
            return
        
        # get axes widgets
        axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

        # return if there's no proper axes
        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return

        # update contours if necessary
        if not self.checkContoursUpToDate():
            return

        # plot the precalculated contours
        painter.beginPaintingWidget(self, posn)
        painter.save()
        self.clipAxesBounds(painter, axes, posn)

        self.plotContourFills(painter, posn, axes)
        self.plotContours(painter, posn, axes)
        self.plotSubContours(painter, posn, axes)

        painter.restore()
        painter.endPaintingWidget()

    def updateContours(self):
        """Update calculated contours."""

        s = self.settings
        d = self.document

        minval, maxval, levels = self.calculateLevels()
        sublevels = self.calculateSubLevels(minval, maxval, levels)

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
        self._cachedsubcontours = None

        if self.Cntr is not None:
            c = self.Cntr(xpts, ypts, data.data)

            # trace the contour levels
            if len(s.Lines.lines) != 0:
                self._cachedcontours = []
                for level in levels:
                    linelist = c.trace(level)
                    self._cachedcontours.append( finitePoly(linelist) )

            # trace the polygons between the contours
            if len(s.Fills.fills) != 0 and len(levels) > 1 and not s.Fills.hide:
                self._cachedpolygons = []
                for level1, level2 in izip(levels[:-1], levels[1:]):
                    linelist = c.trace(level1, level2)
                    self._cachedpolygons.append( finitePoly(linelist) )

            # trace sub-levels
            if len(sublevels) > 0:
                self._cachedsubcontours = []
                for level in sublevels:
                    linelist = c.trace(level)
                    self._cachedsubcontours.append( finitePoly(linelist) )

    def plotContourLabel(self, painter, number, xplt, yplt, showline):
        s = self.settings
        cl = s.get('ContourLabels')

        painter.save()

        # get text and font
        text = utils.formatNumber(number * cl.scale, cl.format)
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
        if showline:
            pts = qt4.QPolygonF()
            for x, y in izip(xplt, yplt):
                pts.append( qt4.QPointF(x, y) )
            painter.drawPolyline(pts)

        # actually plot the label
        if showtext:
            painter.setClipRegion(oldclip)
            painter.setPen( cl.makeQPen() )
            r.render()

        painter.restore()

    def _plotContours(self, painter, posn, axes, linestyles, contours,
                      showlabels, hidelines):
        """Plot a set of contours.
        """

        s = self.settings

        # no lines cached as no line styles
        if contours is None:
            return

        # iterate over each level, and list of lines
        for num, linelist in enumerate(contours):

            # move to the next line style
            painter.setPen(linestyles.makePen(painter, num))
                
            # iterate over each complete line of the contour
            for curve in linelist:
                # convert coordinates from graph to plotter
                xplt = axes[0].dataToPlotterCoords(posn, curve[:,0])
                yplt = axes[1].dataToPlotterCoords(posn, curve[:,1])
                    
                # there should be a nice itertools way of doing this
                pts = qt4.QPolygonF()
                for x, y in izip(xplt, yplt):
                    pts.append( qt4.QPointF(x, y) )

                if showlabels:
                    self.plotContourLabel(painter, s.levelsOut[num], xplt, yplt,
                                          not hidelines)
                else:
                    # actually draw the curve to the plotter
                    if not hidelines:
                        painter.drawPolyline(pts)

    def plotContours(self, painter, posn, axes):
        """Plot the traced contours on the painter."""
        s = self.settings
        self._plotContours(painter, posn, axes, s.Lines.get('lines'),
                           self._cachedcontours,
                           not s.ContourLabels.hide, s.Lines.hide)

    def plotSubContours(self, painter, posn, axes):
        """Plot sub contours on painter."""
        s = self.settings
        self._plotContours(painter, posn, axes, s.SubLines.get('lines'),
                           self._cachedsubcontours,
                           False, s.SubLines.hide)

    def plotContourFills(self, painter, posn, axes):
        """Plot the traced contours on the painter."""

        s = self.settings

        # don't draw if there are no cached polygons
        if self._cachedpolygons is None or s.Fills.hide:
            return

        # ensure plotting of contours does not go outside the area
        painter.setPen(qt4.QPen(qt4.Qt.NoPen))

        # iterate over each level, and list of lines
        for num, polylist in enumerate(self._cachedpolygons):

            # move to the next line style
            painter.setBrush(s.Fills.get('fills').makeBrush(num))
                
            # iterate over each complete line of the contour
            for poly in polylist:
                # convert coordinates from graph to plotter
                xplt = axes[0].dataToPlotterCoords(posn, poly[:,0])
                yplt = axes[1].dataToPlotterCoords(posn, poly[:,1])

                pts = qt4.QPolygonF()
                for x, y in izip(xplt, yplt):
                    pts.append( qt4.QPointF(x, y) )
                painter.drawPolygon(pts)

# allow the factory to instantiate a contour
document.thefactory.register( Contour )
