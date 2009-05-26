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

# $Id$

"""For plotting bar graphs."""

from itertools import izip
import numpy as N

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

from plotters import GenericPlotter

class BarFill(setting.Settings):
    '''Filling of bars.'''
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.FillSet('fills', [('solid', 'grey', False)],
                                  descr = 'Fill styles for dataset bars',
                                  usertext='Fill styles') )

class BarLine(setting.Settings):
    '''Edges of bars.'''
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.LineSet('lines',
                                  [('solid', '0.5pt', 'black', False)],
                                  descr = 'Line styles for dataset bars', 
                                  usertext='Line styles') )

class BarPlotter(GenericPlotter):
    """Plot bar charts."""

    typename='bar'
    allowusercreation=True
    description='Plot bar charts'

    def __init__(self, parent, name=None):
        """Initialise bar chart."""
        
        GenericPlotter.__init__(self, parent, name=name)
        s = self.settings

        # get rid of default key setting
        s.remove('key')

        s.add( setting.Strings('keys', (''),
                               descr='Key text for each dataset',
                               usertext='Key text'), 0)

        s.add( setting.Choice('mode', ('grouped', 'stacked'), 
                              'grouped', 
                              descr='Show datasets grouped '
                              'together or as a single bar', 
                              usertext='Mode'), 0)
        s.add( setting.Choice('direction', 
                              ('horizontal', 'vertical'), 'vertical', 
                              descr = 'Horizontal or vertical bar chart', 
                              usertext='Direction'), 0 )
        s.add( setting.Dataset('posn', '', 
                               descr = 'Dataset containing position of bars'
                               ' (optional)',
                               usertext='Positions'), 0 )
        s.add( setting.Datasets('lengths', ('y',),
                                descr = 'Datasets containing lengths of bars',
                                usertext='Lengths'), 0 )

        s.add( setting.Float('barfill', 0.75,
                             minval = 0., maxval = 1.,
                             descr = 'Filling fraction of bars'
                             ' (between 0 and 1)',
                             usertext='Bar fill',
                             formatting=True) )
        s.add( setting.Float('groupfill', 0.9,
                             minval = 0., maxval = 1.,
                             descr = 'Filling fraction of groups of bars'
                             ' (between 0 and 1)',
                             usertext='Group fill',
                             formatting=True) )

        s.add( setting.Choice('errorstyle', ('none', 'bar', 'barends'), 
                              'bar', 
                              descr='Error bar style to show', 
                              usertext='Error style',
                              formatting=True) )

        s.add(BarFill('BarFill', descr='Bar fill', usertext='Fill'),
              pixmap='bgfill')
        s.add(BarLine('BarLine', descr='Bar line', usertext='Line'),
              pixmap='border')

        s.add( setting.ErrorBarLine('ErrorBarLine',
                                    descr = 'Error bar line settings',
                                    usertext = 'Error bar line'),
               pixmap = 'ploterrorline' )

        if type(self) == BarPlotter:
            self.readDefaults()

    def _getUserDescription(self):
        """User-friendly description."""

        s = self.settings
        return "lengths='%s', position='%s'" % (', '.join(s.lengths), 
                                                s.posn)
    userdescription = property(_getUserDescription)

    def providesAxesDependency(self):
        """This widget provides range information about these axes."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def singleBarDataRange(self, datasets):
        """For single bars where multiple datasets are added,
        compute maximum range."""
        minv, maxv = 0., 0.
        for data in izip(*[ds.data for ds in datasets]):
            totpos = sum( [d for d in data if d > 0] )
            totneg = sum( [d for d in data if d < 0] )
            
            minv = min(minv, totneg)
            maxv = max(maxv, totpos)
        return minv,  maxv

    def updateAxisRange(self, axis, depname, axrange):
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
        length = len(vals)
        if dataset.serr is not None:
            minval = vals - dataset.serr[:length]
            maxval = vals + dataset.serr[:length]
        else:
            if dataset.nerr is not None:
                minval = vals + dataset.nerr[:length]
            if dataset.perr is not None:
                maxval = vals + dataset.perr[:length]
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
        maxcoord = axes[not ishorz].dataToPlotterCoords(widgetposn, maxval)

        # draw error bars
        painter.setPen( self.settings.ErrorBarLine.makeQPenWHide(painter) )
        pts = []
        w = barwidth*0.25
        if ishorz:
            for x1, x2, y in izip(mincoord, maxcoord, posns):
                pts.append( qt4.QPointF(x1, y) )
                pts.append( qt4.QPointF(x2, y) )
            if s.errorstyle == 'barends':
                for x1, x2, y in izip(mincoord, maxcoord, posns):
                    pts.append( qt4.QPointF(x1, y-w) )
                    pts.append( qt4.QPointF(x1, y+w) )
                    pts.append( qt4.QPointF(x2, y-w) )
                    pts.append( qt4.QPointF(x2, y+w) )
        else:
            for y1, y2, x in izip(mincoord, maxcoord, posns):
                pts.append( qt4.QPointF(x, y1) )
                pts.append( qt4.QPointF(x, y2) )
            if s.errorstyle == 'barends':
                for y1, y2, x in izip(mincoord, maxcoord, posns):
                    pts.append( qt4.QPointF(x-w, y1) )
                    pts.append( qt4.QPointF(x+w, y1) )
                    pts.append( qt4.QPointF(x-w, y2) )
                    pts.append( qt4.QPointF(x+w, y2) )
        painter.drawLines(pts)

    def barDrawGroup(self, painter, lengths, positions, axes, widgetposn):
        """Draw groups of bars."""

        s = self.settings
        numgroups = len(lengths)

        # get positions of groups of bars
        posns, maxwidth = self.findBarPositions(lengths, positions,
                                                axes, widgetposn)

        # calculate bar and group widths
        groupwidth = maxwidth
        usablewidth = groupwidth * s.groupfill
        bardelta = usablewidth / float(numgroups)
        barwidth = bardelta * s.barfill

        ishorz = s.direction == 'horizontal'

        # trim datasets to minimum lengths
        datasets = [l.data for l in lengths]
        minlen = min([len(d) for d in datasets] + [len(posns)])
        datasets = [d[:minlen] for d in datasets]
        posns = posns[:minlen]

        # bar extends from these coordinates
        zeropts = axes[not ishorz].dataToPlotterCoords(widgetposn,
                                                       N.zeros(minlen))

        for dsnum, dataset in enumerate(datasets):
            # set correct attributes for datasets
            painter.setBrush( s.BarFill.get('fills').makeBrush(dsnum) )
            painter.setPen( s.BarLine.get('lines').makePen(painter, dsnum) )
            
            # convert bar length to plotter coords
            lengthcoord = axes[not ishorz].dataToPlotterCoords(
                widgetposn, dataset)
 
            # these are the coordinates perpendicular to the bar
            posns1 = posns + (-usablewidth*0.5 + bardelta*dsnum +
                              (bardelta-barwidth)*0.5)
            posns2 = posns1 + barwidth

            if ishorz:
                coords = (zeropts, lengthcoord, posns1, posns2)
            else:
                coords = (posns1, posns2, zeropts, lengthcoord)

            # iterate over coordinates to plot bars
            # column_stack is actually much slower than izip!
            # hopefully we won't get many bars to draw, however
            for x1, x2, y1, y2 in N.nan_to_num(N.column_stack(coords)):
                painter.drawRect( qt4.QRectF(qt4.QPointF(x1, y1),
                                             qt4.QPointF(x2, y2) ) )

            # draw error bars
            self.drawErrorBars(painter, posns2-barwidth*0.5, barwidth,
                               dataset, lengths[dsnum],
                               axes, widgetposn)

    def barDrawStacked(self, painter, lengths, positions, axes, widgetposn):
        """Draw each dataset in a single bar."""

        s = self.settings

        # get positions of groups of bars
        posns,  maxwidth = self.findBarPositions(lengths, positions,
                                                 axes, widgetposn)
        barwidth = maxwidth * s.barfill

        ishorz = s.direction == 'horizontal'

        # trim data to minimum length
        datasets = [l.data for l in lengths]
        minlen = min([len(d) for d in datasets] + [len(posns)])
        datasets = [d[:minlen] for d in datasets]
        posns = posns[:minlen]

        # keep track of last most negative or most positive values in bars
        lastneg = N.zeros(minlen)
        lastpos = N.zeros(minlen)

        # keep track of bars for error bars
        barvals = []
        for dsnum, data in enumerate(datasets):
            # set correct attributes for datasets
            painter.setBrush( s.BarFill.get('fills').makeBrush(dsnum) )
            painter.setPen( s.BarLine.get('lines').makePen(painter, dsnum) )
            
            # add on value to last value in correct direction
            last = N.where(data < 0., lastneg, lastpos)
            new = N.where(data < 0., lastneg+data, lastpos+data)

            # work out maximum extents for next time
            lastneg = N.nanmin( N.vstack((lastneg, new)), axis=0 )
            lastpos = N.nanmax( N.vstack((lastpos, new)), axis=0 )

            # convert values to plotter coordinates
            lastplt = axes[not ishorz].dataToPlotterCoords(
                widgetposn, last)
            newplt = axes[not ishorz].dataToPlotterCoords(
                widgetposn, new)

            # positions of bar perpendicular to bar direction
            posns1 = posns - barwidth*0.5
            posns2 = posns1 + barwidth 

            # we iterate over each of these coordinates
            if ishorz:
                coords = (lastplt, newplt, posns1, posns2)
            else:
                coords = (posns1, posns2, lastplt, newplt)

            # draw bars
            for x1, x2, y1, y2 in N.nan_to_num(N.column_stack(coords)):
                painter.drawRect( qt4.QRectF(qt4.QPointF(x1, y1),
                                             qt4.QPointF(x2, y2)) )
            barvals.append(new)

        for dsnum, data in enumerate(datasets):
            # draw error bars
            self.drawErrorBars(painter, posns, barwidth,
                               barvals[dsnum], lengths[dsnum],
                               axes, widgetposn)

    def getNumberKeys(self):
        """Return maximum number of keys."""
        lengths = self.settings.get('lengths').getData(self.document)
        if not lengths:
            return 0
        return min( len([k for k in self.settings.keys if k]), len(lengths) )

    def getKeyText(self, number):
        """Get key entry."""
        return [k for k in self.settings.keys if k][number]

    def drawKeySymbol(self, number, painter, x, y, width, height):
        """Draw a fill rectangle for key entry."""

        s = self.settings
        painter.setBrush( s.BarFill.get('fills').makeBrush(number) )
        painter.setPen( s.BarLine.get('lines').makePen(painter, number) )
        painter.drawRect( qt4.QRectF(qt4.QPointF(x, y+height*0.1),
                                     qt4.QPointF(x+width, y+height*0.8)) )


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
        positions = s.get('posn').getData(doc)
        lengths = s.get('lengths').getData(doc)
        if not lengths:
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

        # actually do the drawing
        if s.mode == 'stacked':
            self.barDrawStacked(painter, lengths, positions, axes, posn)
        else:
            self.barDrawGroup(painter, lengths, positions, axes, posn)

        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate a bar plotter
document.thefactory.register( BarPlotter )
