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
        self.add( setting.FillSet('fills', [('solid', 'black', False)],
                                  descr = 'Fill styles for dataset bars',
                                  usertext='Fill styles') )

class BarLine(setting.Settings):
    '''Edges of bars.'''
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.LineSet('lines',
                                  [('solid', '1pt', 'black', False)],
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

        s.add(BarFill('BarFill', descr='Bar fill', usertext='Fill'),
              pixmap='bgfill')
        s.add(BarLine('BarLine', descr='Bar line', usertext='Line'),
              pixmap='border')

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
                if s.mode == 'grouping':
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

    def barDrawStacked(self, lengths, positions, axes, posn):
        """Draw each dataset in a single bar."""

        if positions is None:
            x = posn[2]

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
            maxwidth = N.nanmin(posns[1:]-posns[:-1])

        return posns,  maxwidth

    def barDrawGroup(self, painter, lengths, positions, axes, widgetposn):
        """Draw groups of bars."""

        s = self.settings
        numgroups = len(lengths)

        # get positions of groups of bars
        posns,  maxwidth = self.findBarPositions(lengths, positions,
                                                 axes, widgetposn)
        bardelta = maxwidth / float(numgroups)
        barwidth = bardelta*0.75

        ishorz = s.direction == 'horizontal'
        # bar extends from this coordinate
        zeropt = axes[not ishorz].dataToPlotterCoords(widgetposn, N.array([0.]))
    
        for dsnum, dataset in enumerate(lengths):
            # set correct attributes for datasets
            painter.setBrush( s.BarFill.get('fills').makeBrush(dsnum) )
            painter.setPen( s.BarLine.get('lines').makePen(painter, dsnum) )
            
            lengthcoord = axes[not ishorz].dataToPlotterCoords(
                widgetposn, N.array(dataset.data))
 
            for length,  posn in izip(lengthcoord, posns):
                if N.isfinite(length) and N.isfinite(posn):
                    # work out positions of bar perpendicular to bar length
                    p1 = posn - maxwidth*0.5 + bardelta*dsnum + (bardelta-
                                                                 barwidth)*0.5
                    p2 = p1 + barwidth
                    # draw bar from zero point
                    if ishorz:
                        painter.drawRect( qt4.QRectF(qt4.QPointF(zeropt, p1),
                                                     qt4.QPointF(length, p2) ) )
                    else:
                        painter.drawRect( qt4.QRectF(qt4.QPointF(p1, zeropt),
                                                     qt4.QPointF(p2, length) ) )

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

        if s.mode == 'stacked':
            self.barDrawStacked(painter, lengths, positions, axes, posn)
        else:
            self.barDrawGroup(painter, lengths, positions, axes, posn)

        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate a bar plotter
document.thefactory.register( BarPlotter )
