#    Copyright (C) 2009 Jeremy S. Sanders
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

"""For plotting bar graphs."""

from __future__ import division
import numpy as N

from ..compat import crange, czip
from .. import qtall as qt4
from .. import document
from .. import setting
from .. import utils

from .plotters import GenericPlotter

def _(text, disambiguation=None, context='BarPlotter'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class BarFill(setting.Settings):
    '''Filling of bars.'''
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.FillSet('fills', [('solid', 'auto', False)],
                                  descr = _('Fill styles for dataset bars'),
                                  usertext=_('Fill styles')) )

class BarLine(setting.Settings):
    '''Edges of bars.'''
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.LineSet('lines',
                                  [('solid', '0.5pt', 'black', False)],
                                  descr = _('Line styles for dataset bars'), 
                                  usertext=_('Line styles')) )

def extend1DArray(array, length, missing=0.):
    """Return array with length given (original if appropriate.
    Values are extended with value given."""

    if len(array) == length:
        return array
    retn = N.resize(array, length)
    retn[len(array):] = missing
    return retn

class BarPlotter(GenericPlotter):
    """Plot bar charts."""

    typename='bar'
    allowusercreation=True
    description=_('Plot bar charts')

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        GenericPlotter.addSettings(s)

        # get rid of default key setting
        s.remove('key')

        s.add( setting.Strings('keys', ('',),
                               descr=_('Key text for each dataset'),
                               usertext=_('Key text')), 0)

        s.add( setting.DatasetOrStr('labels', '',
                                    descr=_('Dataset or string to label bars'),
                                    usertext=_('Labels')), 5 )

        s.add( setting.Choice('mode', ('grouped', 'stacked', 'stacked-area'),
                              'grouped',
                              descr=_('Show datasets grouped '
                                      'together or as a single bar'),
                              usertext=_('Mode')), 0)
        s.add( setting.Choice('direction',
                              ('horizontal', 'vertical'), 'vertical', 
                              descr = _('Horizontal or vertical bar chart'),
                              usertext=_('Direction')), 0 )
        s.add( setting.DatasetExtended('posn', '',
                                       descr = _('Position of bars, dataset '
                                                 ' or expression (optional)'),
                                       usertext=_('Positions')), 0 )
        s.add( setting.Datasets('lengths', ('y',),
                                descr = _('Datasets containing lengths of bars'),
                                usertext=_('Lengths')), 0 )

        s.add( setting.Float('barfill', 0.75,
                             minval = 0., maxval = 1.,
                             descr = _('Filling fraction of bars'
                                       ' (between 0 and 1)'),
                             usertext=_('Bar fill'),
                             formatting=True) )
        s.add( setting.Float('groupfill', 0.9,
                             minval = 0., maxval = 1.,
                             descr = _('Filling fraction of groups of bars'
                                       ' (between 0 and 1)'),
                             usertext=_('Group fill'),
                             formatting=True) )

        s.add( setting.Choice('errorstyle', ('none', 'bar', 'barends'),
                              'bar',
                              descr=_('Error bar style to show'),
                              usertext=_('Error style'),
                              formatting=True) )

        s.add(BarFill('BarFill', descr=_('Bar fill'), usertext=_('Fill')),
              pixmap = 'settings_bgfill')
        s.add(BarLine('BarLine', descr=_('Bar line'), usertext=_('Line')),
              pixmap = 'settings_border')

        s.add( setting.ErrorBarLine('ErrorBarLine',
                                    descr = _('Error bar line settings'),
                                    usertext = _('Error bar line')),
               pixmap = 'settings_ploterrorline' )

    @property
    def userdescription(self):
        """User-friendly description."""

        s = self.settings
        return _("lengths='%s', position='%s'") % (', '.join(s.lengths), 
                                                   s.posn)

    def affectsAxisRange(self):
        """This widget provides range information about these axes."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def getAxisLabels(self, direction):
        """Get labels for bar for appropriate axis."""
        s = self.settings
        if s.direction != direction:
            # if horizontal bars, want labels on vertical axis and vice versa
            doc = self.document

            labels = s.get('labels').getData(doc, checknull=True)
            positions = s.get('posn').getData(doc)
            if positions is None:
                lengths = s.get('lengths').getData(doc)
                if not lengths:
                    return (None, None)
                p = N.arange( max([len(d.data) for d in lengths]) )+1.
            else:
                p = positions.data
            
            return (labels, p)

        else:
            return (None, None)

    def singleBarDataRange(self, datasets):
        """For single bars where multiple datasets are added,
        compute maximum range."""
        minv, maxv = 0., 0.
        for data in czip(*[ds.data for ds in datasets]):
            totpos = sum( [d for d in data if d > 0] )
            totneg = sum( [d for d in data if d < 0] )
            
            minv = min(minv, totneg)
            maxv = max(maxv, totpos)
        return minv,  maxv

    def getRange(self, axis, depname, axrange):
        """Update axis range from data."""
        s = self.settings
        if ((s.direction == 'horizontal' and depname == 'sx') or
            (s.direction == 'vertical' and depname == 'sy')):
            # update from lengths
            data = s.get('lengths').getData(self.document)
            if s.mode == 'grouped':
                # update range from individual datasets
                for d in data:
                    drange = d.getRange()
                    if drange is not None:
                        axrange[0] = min(axrange[0], drange[0])
                        axrange[1] = max(axrange[1], drange[1])
            else:
                # update range from sum of datasets
                minv, maxv = self.singleBarDataRange(data)
                axrange[0] = min(axrange[0], minv)
                axrange[1] = max(axrange[1], maxv)
        else:
            if s.posn:
                # use given positions
                data = s.get('posn').getData(self.document)
                if data:
                    drange = data.getRange()
                    if drange is not None:
                        axrange[0] = min(axrange[0], drange[0])
                        axrange[1] = max(axrange[1], drange[1])
            else:
                # count bars
                data = s.get('lengths').getData(self.document)
                if data:
                    maxlen = max([len(d) for d in data])
                    axrange[0] = min(1-0.5, axrange[0])
                    axrange[1] = max(maxlen+0.5,  axrange[1])

    def findBarPositions(self, lengths, positions, axes, posn):
        """Work out centres of bar / bar groups and maximum width."""

        ishorz = self.settings.direction == 'horizontal'

        if positions is None:
            p = N.arange( max([len(d.data) for d in lengths]) )+1.
        else:
            p = positions.data

        # work out positions of bars
        # get vertical axis if horz, and vice-versa
        axis = axes[ishorz]
        posns = axis.dataToPlotterCoords(posn, p)
        if len(posns) <= 1:
            if ishorz:
                maxwidth = posn[2]-posn[0]
            else:
                maxwidth = posn[3]-posn[1]
        else:
            maxwidth = N.nanmin(N.abs(posns[1:]-posns[:-1]))

        return posns,  maxwidth

    def calculateErrorBars(self, dataset, vals):
        """Get values for error bars."""
        minval = None
        maxval = None
        if 'serr' in dataset:
            s = N.nan_to_num(dataset['serr'])
            minval = vals - s
            maxval = vals + s
        else:
            if 'nerr' in dataset:
                minval = vals + N.nan_to_num(dataset['nerr'])
            if 'perr' in dataset:
                maxval = vals + N.nan_to_num(dataset['perr'])
        return minval, maxval

    def drawErrorBars(self, painter, posns, barwidth,
                      yvals, dataset, axes, widgetposn):
        """Draw (optional) error bars on bars."""
        s = self.settings
        if s.errorstyle == 'none':
            return

        minval, maxval = self.calculateErrorBars(dataset, yvals)
        if minval is None and maxval is None:
            return

        # handle one sided errors
        if minval is None:
            minval = yvals
        if maxval is None:
            maxval = yvals

        # convert errors to coordinates
        ishorz = s.direction == 'horizontal'
        mincoord = axes[not ishorz].dataToPlotterCoords(widgetposn, minval)
        mincoord = N.clip(mincoord, -32767, 32767)
        maxcoord = axes[not ishorz].dataToPlotterCoords(widgetposn, maxval)
        maxcoord = N.clip(maxcoord, -32767, 32767)

        # draw error bars
        ebl = self.settings.ErrorBarLine
        painter.setPen( ebl.makeQPenWHide(painter) )
        w = barwidth*0.25*ebl.endsize
        if ishorz and not ebl.hideHorz:
            utils.plotLinesToPainter(painter, mincoord, posns,
                                     maxcoord, posns)
            if s.errorstyle == 'barends':
                utils.plotLinesToPainter(painter, mincoord, posns-w,
                                         mincoord, posns+w)
                utils.plotLinesToPainter(painter, maxcoord, posns-w,
                                         maxcoord, posns+w)
        elif not ishorz and not ebl.hideVert:
            utils.plotLinesToPainter(painter, posns, mincoord,
                                     posns, maxcoord)
            if s.errorstyle == 'barends':
                utils.plotLinesToPainter(painter, posns-w, mincoord,
                                         posns+w, mincoord)
                utils.plotLinesToPainter(painter, posns-w, maxcoord,
                                         posns+w, maxcoord)

    def plotBars(self, painter, s, dsnum, clip, corners):
        """Plot a set of boxes."""
        # get style
        brush = s.BarFill.get('fills').returnBrushExtended(dsnum)
        pen = s.BarLine.get('lines').makePen(painter, dsnum)
        lw = pen.widthF() * 2

        # make clip box bigger to avoid lines showing
        extclip = qt4.QRectF(qt4.QPointF(clip.left()-lw, clip.top()-lw),
                             qt4.QPointF(clip.right()+lw, clip.bottom()+lw))

        # plot bars
        path = qt4.QPainterPath()
        utils.addNumpyPolygonToPath(
            path, extclip, corners[0], corners[1], corners[2], corners[1],
            corners[2], corners[3], corners[0], corners[3])
        utils.brushExtFillPath(
            painter, brush, path, stroke=pen, dataindex=dsnum)

    def barDrawGroup(self, painter, posns, maxwidth, dsvals,
                     axes, widgetposn, clip):
        """Draw groups of bars."""

        s = self.settings

        # calculate bar and group widths
        numgroups = len(dsvals)
        groupwidth = maxwidth
        usablewidth = groupwidth * s.groupfill
        bardelta = usablewidth / numgroups
        barwidth = bardelta * s.barfill

        ishorz = s.direction == 'horizontal'

        # bar extends from these coordinates
        zeropt = axes[not ishorz].dataToPlotterCoords(widgetposn,
                                                      N.array([0.]))

        for dsnum, dataset in enumerate(dsvals):

            # convert bar length to plotter coords
            lengthcoord = axes[not ishorz].dataToPlotterCoords(
                widgetposn, dataset['data'])
 
            # these are the coordinates perpendicular to the bar
            posns1 = posns + (-usablewidth*0.5 + bardelta*dsnum +
                              (bardelta-barwidth)*0.5)
            posns2 = posns1 + barwidth

            if ishorz:
                p = (zeropt + N.zeros(posns1.shape), posns1,
                     lengthcoord, posns2)
            else:
                p = (posns1, zeropt + N.zeros(posns2.shape),
                     posns2, lengthcoord)

            self.plotBars(painter, s, dsnum, clip, p)

            # draw error bars
            self.drawErrorBars(painter, posns2-barwidth*0.5, barwidth,
                               dataset['data'], dataset,
                               axes, widgetposn)

    def calcStackedPoints(self, dsvals, axis, widgetposn):
        """Calculate stacked dataset coordinates for plotting."""

        # keep track of last most negative or most positive values in bars
        poslen = len(dsvals[0]['data'])
        lastneg = N.zeros(poslen)
        lastpos = N.zeros(poslen)

        # returned stacked values and coordinates
        stackedvals = []
        stackedcoords = []

        for dsnum, data in enumerate(dsvals):
            # add on value to last value in correct direction
            data = data['data']
            new = N.where(data < 0., lastneg+data, lastpos+data)

            # work out maximum extents for next time
            lastneg = N.min( N.vstack((lastneg, new)), axis=0 )
            lastpos = N.max( N.vstack((lastpos, new)), axis=0 )

            # convert values to plotter coordinates
            newplt = axis.dataToPlotterCoords(widgetposn, new)

            stackedvals.append(new)
            stackedcoords.append(newplt)

        return stackedvals, stackedcoords

    def barDrawStacked(self, painter, posns, maxwidth, dsvals,
                       axes, widgetposn, clip):
        """Draw each dataset in a single bar."""

        s = self.settings

        # get positions of groups of bars
        barwidth = maxwidth * s.barfill

        # get axis which values are plotted along
        ishorz = s.direction == 'horizontal'
        vaxis = axes[not ishorz]

        # compute stacked coordinates
        stackedvals, stackedcoords = self.calcStackedPoints(
            dsvals, vaxis, widgetposn)
        # coordinates of origin
        zerocoords = vaxis.dataToPlotterCoords(widgetposn, N.zeros(posns.shape))

        # positions of bar perpendicular to bar direction
        posns1 = posns - barwidth*0.5
        posns2 = posns1 + barwidth

        # draw bars (reverse order, so edges are plotted correctly)
        for dsnum, coords in czip( crange(len(stackedcoords)-1, -1, -1),
                                   stackedcoords[::-1]):
            # we iterate over each of these coordinates
            if ishorz:
                p = (zerocoords, posns1, coords, posns2)
            else:
                p = (posns1, zerocoords, posns2, coords)
            self.plotBars(painter, s, dsnum, clip, p)

        # draw error bars
        for barval, dsval in czip(stackedvals, dsvals):
            self.drawErrorBars(painter, posns, barwidth,
                               barval, dsval,
                               axes, widgetposn)

    def areaDrawStacked(self, painter, posns, maxwidth, dsvals,
                        axes, widgetposn, clip):
        """Draw a stacked area plot"""

        s = self.settings

        # get axis which values are plotted along
        ishorz = s.direction == 'horizontal'
        vaxis = axes[not ishorz]

        # compute stacked coordinates
        stackedvals, stackedcoords = self.calcStackedPoints(
            dsvals, vaxis, widgetposn)
        # coordinates of origin
        zerocoords = vaxis.dataToPlotterCoords(widgetposn, N.zeros(posns.shape))

        # bail out if problem
        if len(zerocoords) == 0 or len(posns) == 0:
            return

        # draw areas (reverse order, so edges are plotted correctly)
        for dsnum, coords in czip( crange(len(stackedcoords)-1, -1, -1),
                                   stackedcoords[::-1]):

            # add points at end to make polygon
            p1 = N.hstack( [ [zerocoords[0]], coords, [zerocoords[-1]] ] )
            p2 = N.hstack( [ [posns[0]], posns, [posns[-1]] ] )

            # construct polygon on path, clipped
            poly = qt4.QPolygonF()
            if ishorz:
                utils.addNumpyToPolygonF(poly, p1, p2)
            else:
                utils.addNumpyToPolygonF(poly, p2, p1)
            clippoly = qt4.QPolygonF()
            utils.polygonClip(poly, clip, clippoly)
            path = qt4.QPainterPath()
            path.addPolygon(clippoly)
            path.closeSubpath()

            # actually draw polygon
            brush = s.BarFill.get('fills').returnBrushExtended(dsnum)
            utils.brushExtFillPath(painter, brush, path, dataindex=dsnum)

            # now draw lines
            poly = qt4.QPolygonF()
            if ishorz:
                utils.addNumpyToPolygonF(poly, coords, posns)
            else:
                utils.addNumpyToPolygonF(poly, posns, coords)

            pen = s.BarLine.get('lines').makePen(painter, dsnum)
            painter.setPen(pen)
            utils.plotClippedPolyline(painter, clip, poly)

        # draw error bars
        barwidth = maxwidth * s.barfill
        for barval, dsval in czip(stackedvals, dsvals):
            self.drawErrorBars(painter, posns, barwidth,
                               barval, dsval,
                               axes, widgetposn)

    def getNumberKeys(self):
        """Return maximum number of keys."""
        lengths = self.settings.get('lengths').getData(self.document)
        if not lengths:
            return 0
        return min( len([k for k in self.settings.keys if k]), len(lengths) )

    def setupAutoColor(self, painter):
        """Initialise correct number of colors."""
        lengths = self.settings.get('lengths').getData(self.document)
        for i in crange(len(lengths)):
            self.autoColor(painter, dataindex=i)

    def getKeyText(self, number):
        """Get key entry."""
        return [k for k in self.settings.keys if k][number]

    def drawKeySymbol(self, number, painter, x, y, width, height):
        """Draw a fill rectangle for key entry."""

        self.plotBars(painter, self.settings, number,
                      qt4.QRectF(0,0,32767,32767),
                      ([x], [y+height*0.1],
                       [x+width], [y+height*0.8]))

    def dataDraw(self, painter, axes, widgetposn, clip):
        """Plot the data on a plotter."""
        s = self.settings

        # get data
        doc = self.document
        positions = s.get('posn').getData(doc)
        lengths = s.get('lengths').getData(doc)
        if not lengths:
            return

        # where the bars are to be placed horizontally
        barposns, maxwidth = self.findBarPositions(lengths, positions,
                                                   axes, widgetposn)
        
        # only use finite positions
        origposnlen = len(barposns)
        validposn = N.isfinite(barposns)
        barposns = barposns[validposn]

        # this is a bit rubbish - we take the datasets and
        # make sure they have the same lengths as posns and remove NaNs
        # Datasets are stored as dicts
        dsvals = []
        for dataset in lengths:
            vals = {}
            for key in ('data', 'serr', 'nerr', 'perr'):
                v = getattr(dataset, key)
                if v is not None:
                    vals[key] = extend1DArray(N.nan_to_num(v),
                                              origposnlen)[validposn]
            dsvals.append(vals)

        # actually do the drawing
        fn = {'stacked': self.barDrawStacked,
              'stacked-area': self.areaDrawStacked,
              'grouped': self.barDrawGroup}[s.mode]
        fn(painter, barposns, maxwidth, dsvals, axes, widgetposn, clip)

# allow the factory to instantiate a bar plotter
document.thefactory.register( BarPlotter )
