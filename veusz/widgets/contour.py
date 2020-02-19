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

"""Contour plotting from 2d datasets.

Contour plotting requires that the veusz_helpers package is installed,
as a C routine (taken from matplotlib) is used to trace the contours.
"""

from __future__ import division, print_function
import sys
import math

from ..compat import czip, crange
from .. import qtall as qt
import numpy as N

from .. import setting
from .. import document
from .. import utils

from . import plotters

try:
    from ..helpers._nc_cntr import Cntr
    from ..helpers.qtloops import LineLabeller
except ImportError:
    Cntr = None
    LineLabeller = object   # allow class definition below

def _(text, disambiguation=None, context='Contour'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

def finitePoly(poly):
    """Remove non-finite coordinates from numpy arrays of coordinates."""
    out = []
    for line in poly:
        finite = N.isfinite(line)
        validrows = N.logical_and(finite[:,0], finite[:,1])
        out.append( line[validrows] )
    return out

class ContourLineLabeller(LineLabeller):
    def __init__(self, clip, rot, painter, font, doc):
        LineLabeller.__init__(self, clip, rot)
        self.clippath = qt.QPainterPath()
        self.clippath.addRect(clip)
        self.labels = []
        self.painter = painter
        self.font = font
        self.document = doc

    def drawAt(self, idx, rect):
        """Called to draw the label with the index given."""
        text = self.labels[idx]
        if not text:
            return

        angle = rect.angle*180/math.pi
        if angle < -90 or angle > 90:
            angle += 180

        rend = utils.Renderer(
            self.painter, self.font,
            rect.cx, rect.cy, text,
            alignhorz=0, alignvert=0,
            angle=angle,
            doc=self.document)

        rend.render()
        if rect.xw > 0:
            p = qt.QPainterPath()
            p.addPolygon(rect.makePolygon())
            self.clippath -= p

class ContourFills(setting.Settings):
    """Settings for contour fills."""
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.FillSet(
            'fills', [],
            descr = _('Fill styles to plot between contours'),
            usertext=_('Fill styles'),
            formatting=True) )
        self.add( setting.Bool(
            'hide', False,
            descr = _('Hide fills'),
            usertext = _('Hide'),
            formatting = True) )

class ContourLines(setting.Settings):
    """Settings for contour lines."""
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.LineSet(
            'lines',
            [('solid', '1pt', 'black', False)],
            descr = _('Line styles to plot the contours '
                      'using'), usertext=_('Line styles'),
            formatting=True) )
        self.add( setting.Bool(
            'hide', False,
            descr = _('Hide lines'),
            usertext = _('Hide'),
            formatting = True) )

class SubContourLines(setting.Settings):
    """Sub-dividing contour line settings."""
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.LineSet(
            'lines',
            [('dot1', '1pt', 'black', False)],
            descr = _('Line styles used for sub-contours'),
            usertext=_('Line styles'),
            formatting=True) )
        self.add( setting.Int(
            'numLevels', 5,
            minval=2,
            descr=_('Number of sub-levels to plot between '
                    'each contour'),
            usertext='Levels') )
        self.add( setting.Bool(
            'hide', True,
            descr=_('Hide lines'),
            usertext=_('Hide'),
            formatting=True) )

class ContourLabel(setting.Text):
    """For tick labels on axes."""

    def __init__(self, name, **args):
        setting.Text.__init__(self, name, **args)
        self.add( setting.Str(
            'format', '%.3Vg',
            descr = _('Format of the tick labels'),
            usertext=_('Format')) )
        self.add( setting.Float(
            'scale', 1.,
            descr=_('A scale factor to apply to the values '
                    'of the tick labels'),
            usertext=_('Scale')) )
        self.add( setting.Bool(
            'rotate',
            True,
            descr=_('Rotate labels to follow lines'),
            usertext=_('Rotate')) )

        self.get('hide').newDefault(True)

class Contour(plotters.GenericPlotter):
    """A class which plots contours on a graph with a specified
    coordinate system."""

    typename='contour'
    allowusercreation=True
    description=_('Plot a 2d dataset as contours')

    def __init__(self, parent, name=None):
        """Initialise plotter with axes."""

        plotters.GenericPlotter.__init__(self, parent, name=name)

        if Cntr is None:
            print(('WARNING: Veusz cannot import contour module\n'
                   'Please run python setup.py build\n'
                   'Contour support is disabled'), file=sys.stderr)

        # keep track of settings so we recalculate when necessary
        self.contsettings = None

        # cached traced contours
        self._cachedcontours = None
        self._cachedpolygons = None
        self._cachedsubcontours = None

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.GenericPlotter.addSettings(s)

        s.add( setting.DatasetExtended(
            'data', '',
            dimensions = 2,
            descr = _('Dataset to plot'),
            usertext=_('Dataset')),
               0 )
        s.add( setting.FloatOrAuto(
            'min', 'Auto',
            descr = _('Minimum value of contour scale'),
            usertext=_('Min. value')),
               1 )
        s.add( setting.FloatOrAuto(
            'max', 'Auto',
            descr = _('Maximum value of contour scale'),
            usertext=_('Max. value')),
               2 )
        s.add( setting.Int(
            'numLevels', 5,
            minval = 1,
            descr = _('Number of contour levels to plot'),
            usertext=_('Number levels')),
               3 )
        s.add( setting.Choice(
            'scaling',
            ['linear', 'sqrt', 'log', 'squared', 'manual'],
            'linear',
            descr = _('Scaling between contour levels'),
            usertext=_('Scaling')),
               4 )
        s.add( setting.FloatList(
            'manualLevels',
            [],
            descr = _('Levels to use for manual scaling'),
            usertext=_('Manual levels')),
               5 )

        s.add( setting.Bool(
            'keyLevels',
            False,
            descr=_('Show levels in key'),
            usertext=_('Levels in key')),
               6 )

        s.add( setting.FloatList(
            'levelsOut',
            [],
            descr = _('Levels used in the plot'),
            usertext=_('Output levels')),
               7, readonly=True )

        s.add( ContourLabel(
            'ContourLabels',
            descr = _('Contour label settings'),
            usertext = _('Contour labels')),
               pixmap = 'settings_axisticklabels' )

        s.add( ContourLines(
            'Lines',
            descr=_('Contour lines'),
            usertext=_('Contour lines')),
               pixmap = 'settings_contourline' )

        s.add( ContourFills(
            'Fills',
            descr=_('Fill within contours'),
            usertext=_('Contour fills')),
               pixmap = 'settings_contourfill' )

        s.add( SubContourLines(
            'SubLines',
            descr=_('Sub-contour lines'),
            usertext=_('Sub-contour lines')),
               pixmap = 'settings_subcontourline' )

        s.add( setting.SettingBackwardCompat('lines', 'Lines/lines', None) )
        s.add( setting.SettingBackwardCompat('fills', 'Fills/fills', None) )

        s.remove('key')

    @property
    def userdescription(self):
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

    def calculateLevels(self):
        """Calculate contour levels from data and settings.

        Returns levels as 1d numpy
        """

        # get dataset
        s = self.settings
        d = self.document

        minval, maxval = 0., 1.
        # scan data
        data = s.get('data').getData(d)
        if data is None or data.dimensions != 2 or data.data.size == 0:
            return

        minval, maxval = N.nanmin(data.data), N.nanmax(data.data)
        if not N.isfinite(minval):
            minval = 0.
        if not N.isfinite(maxval):
            maxval = 1.

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
                if minval == 0.:
                    minval = 1.
                if minval == maxval:
                    maxval = minval + 1
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
        if s.SubLines.hide or len(s.SubLines.lines) == 0 or len(levels) <= 1:
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
            for conmin, conmax in czip(levels[:-1], levels[1:]):
                delta = (conmax-conmin) / num
                out.append( conmin+drange*delta )
            slev = N.hstack(out)

        return slev

    def affectsAxisRange(self):
        """Range information provided by widget."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def getRange(self, axis, depname, axrange):
        """Automatically determine the ranges of variable on the axes."""

        # this is copied from Image, probably should combine
        s = self.settings
        d = self.document

        # return if no data or if the dataset isn't two dimensional
        data = s.get('data').getData(d)
        if data is None or data.dimensions != 2 or data.data.size == 0:
            return

        xr, yr = data.getDataRanges()
        if depname == 'sx':
            axrange[0] = min( axrange[0], xr[0] )
            axrange[1] = max( axrange[1], xr[1] )
        elif depname == 'sy':
            axrange[0] = min( axrange[0], yr[0] )
            axrange[1] = max( axrange[1], yr[1] )

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
            return utils.formatNumber(
                s.levelsOut[number] * cl.scale,
                cl.format,
                locale=self.document.locale )
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
        data = s.get('data').getData(d)
        if data is None or data.dimensions != 2 or data.data.size == 0:
            self.contsettings = None
            s.levelsOut = []
            return False

        hashval = hash(bytes(data.data))
        contsettings = (
            s.min, s.max, s.numLevels, s.scaling,
            s.SubLines.numLevels,
            len(s.Fills.fills) == 0 or s.Fills.hide,
            len(s.SubLines.lines) == 0 or s.SubLines.hide,
            tuple(s.manualLevels),
            hashval
        )

        if contsettings != self.contsettings:
            self.updateContours()
            self.contsettings = contsettings

        return True

    def dataDraw(self, painter, axes, posn, cliprect):
        """Draw the contours."""

        # update contours if necessary
        if not self.checkContoursUpToDate():
            return

        self.plotContourFills(painter, posn, axes, cliprect)
        self.plotContours(painter, posn, axes, cliprect)
        self.plotSubContours(painter, posn, axes, cliprect)

    def updateContours(self):
        """Update calculated contours."""

        s = self.settings
        d = self.document

        minval, maxval, levels = self.calculateLevels()
        sublevels = self.calculateSubLevels(minval, maxval, levels)

        # find coordinates of image coordinate bounds
        data = s.get('data').getData(d)
        if data is None or data.dimensions != 2 or data.data.size == 0:
            return

        rangex, rangey = data.getDataRanges()
        yw, xw = data.data.shape
        xc, yc = data.getPixelCentres()
        xpts = N.reshape( N.tile(xc, yw), (yw, xw) )
        ypts = N.tile(yc[:, N.newaxis], xw)

        # only keep finite data points
        mask = N.logical_not(N.isfinite(data.data))

        # iterate over the levels and trace the contours
        self._cachedcontours = None
        self._cachedpolygons = None
        self._cachedsubcontours = None

        if Cntr is not None:
            c = Cntr(xpts, ypts, data.data, mask)

            # trace the contour levels
            if len(s.Lines.lines) != 0:
                self._cachedcontours = []
                for level in levels:
                    linelist = c.trace(level)
                    self._cachedcontours.append( finitePoly(linelist) )

            # trace the polygons between the contours
            if len(s.Fills.fills) != 0 and len(levels) > 1 and not s.Fills.hide:
                self._cachedpolygons = []
                for level1, level2 in czip(levels[:-1], levels[1:]):
                    linelist = c.trace(level1, level2)
                    self._cachedpolygons.append( finitePoly(linelist) )

            # trace sub-levels
            if len(sublevels) > 0:
                self._cachedsubcontours = []
                for level in sublevels:
                    linelist = c.trace(level)
                    self._cachedsubcontours.append( finitePoly(linelist) )

    def _plotContours(self, painter, posn, axes, linestyles,
                      contours, showlabels, hidelines, clip):
        """Plot a set of contours.
        """

        s = self.settings

        # no lines cached as no line styles
        if contours is None:
            return

        cl = s.get('ContourLabels')
        font = cl.makeQFont(painter)
        labelpen = cl.makeQPen(painter)
        descent = qt.QFontMetricsF(font).descent()

        # linelabeller does clipping and labelling of contours
        linelabeller = ContourLineLabeller(
            clip, cl.rotate, painter, font, self.document)
        levels = []

        # iterate over each level, and list of lines
        for num, linelist in enumerate(contours):

            if showlabels and num<len(s.levelsOut):
                number = s.levelsOut[num]
                text = utils.formatNumber(
                    number * cl.scale, cl.format,
                    locale=self.document.locale)
                rend = utils.Renderer(
                    painter, font, 0, 0, text, alignhorz=0,
                    alignvert=0, angle=0, doc=self.document)
                textdims = qt.QSizeF(*rend.getDimensions())
                textdims += qt.QSizeF(descent*2, descent*2)
            else:
                textdims = qt.QSizeF(0, 0)

            # iterate over each complete line of the contour
            for curve in linelist:
                # convert coordinates from graph to plotter
                xplt = axes[0].dataToPlotterCoords(posn, curve[:,0])
                yplt = axes[1].dataToPlotterCoords(posn, curve[:,1])

                pts = qt.QPolygonF()
                utils.addNumpyToPolygonF(pts, xplt, yplt)
                linelabeller.addLine(pts, textdims)

                if showlabels:
                    linelabeller.labels.append(text)
                else:
                    linelabeller.labels.append(None)
                levels.append(num)

        painter.save()
        painter.setPen(labelpen)
        linelabeller.process()
        painter.setClipPath(linelabeller.clippath)

        for i in crange(linelabeller.getNumPolySets()):
            polyset = linelabeller.getPolySet(i)
            painter.setPen(linestyles.makePen(painter, levels[i]))
            for poly in polyset:
                painter.drawPolyline(poly)

        painter.restore()

    def plotContours(self, painter, posn, axes, clip):
        """Plot the traced contours on the painter."""
        s = self.settings
        self._plotContours(painter, posn, axes, s.Lines.get('lines'),
                           self._cachedcontours,
                           not s.ContourLabels.hide, s.Lines.hide, clip)

    def plotSubContours(self, painter, posn, axes, clip):
        """Plot sub contours on painter."""
        s = self.settings
        self._plotContours(painter, posn, axes, s.SubLines.get('lines'),
                           self._cachedsubcontours,
                           False, s.SubLines.hide, clip)

    def plotContourFills(self, painter, posn, axes, clip):
        """Plot the traced contours on the painter."""

        s = self.settings

        # don't draw if there are no cached polygons
        if self._cachedpolygons is None or s.Fills.hide:
            return

        # iterate over each level, and list of lines
        for num, polylist in enumerate(self._cachedpolygons):

            # iterate over each complete line of the contour
            path = qt.QPainterPath()
            for poly in polylist:
                # convert coordinates from graph to plotter
                xplt = axes[0].dataToPlotterCoords(posn, poly[:,0])
                yplt = axes[1].dataToPlotterCoords(posn, poly[:,1])

                pts = qt.QPolygonF()
                utils.addNumpyToPolygonF(pts, xplt, yplt)

                clippedpoly = qt.QPolygonF()
                utils.polygonClip(pts, clip, clippedpoly)
                path.addPolygon(clippedpoly)

            # fill polygons
            brush = s.Fills.get('fills').returnBrushExtended(num)
            utils.brushExtFillPath(painter, brush, path)

# allow the factory to instantiate a contour
document.thefactory.register( Contour )
