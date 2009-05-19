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

class BarPlotter(GenericPlotter):
    """Plot bar charts."""

    typename='bar'
    allowusercreation=True
    description='Plot bar charts'

    def __init__(self, parent, name=None):
        """Initialise bar chart."""
        
        GenericPlotter.__init__(self, parent, name=name)
        s = self.settings

        s.add( setting.Choice('mode', ('grouping', 'singlebar'), 
                              'grouping', 
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
            # use positions
            if s.posn:
                data = s.get('posn').getData(self.document)
                if data:
                    drange = data.getRange()
                    axrange[0] = min(axrange[0], drange[0])
                    axrange[1] = max(axrange[1], drange[1])

    def barDrawSingle(self, lengths, positions, axes, posn):
        """Draw each dataset as a single bar."""

        if positions is None:
            x = posn[2]

    def findBarPositions(self, lengths, positions, axes, posn):
        """Work out centres of bar / bar groups and maximum width."""

        ishorz = self.settings.direction == 'horizontal'
        if positions is not None:
            # work out positions of bars
            # get vertical axis if horz, and vice-versa
            axis = axes[ishorz]
            posns = axis.dataToPlotterCoords(posn, positions.data)
            if len(posns) == 1:
                if ishorz:
                    maxwidth = posn[2]-posn[0]
                else:
                    maxwidth = posn[3]-posn[1]
            else:
                maxwidth = N.nanmin(posns[1:]-posns[:-1])
        else:
            # equally space bars
            if ishorz:
                minv, maxv = posn[0], posn[2]
            else:
                minv, maxv = posn[1], posn[3]
            # get number of bars
            numbars = max([len(x.data) for x in lengths])
            posns = N.arange(1, numbars) * ((1./numbars) * (maxv-minv)) + minv
            maxwidth = (maxv-minv)*1./numbars

        return posns,  maxwidth

    def barDrawGroup(self, lengths, positions, axes, posn):
        """Draw groups of bars."""

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

        if s.mode == 'single':
            self.barDrawSingle(lengths, positions, axes, posn)
        else:
            self.barDrawGroup(lengths, positions, axes, posn)

        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate a bar plotter
document.thefactory.register( BarPlotter )
