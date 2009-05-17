#    Copyright (C) 2008 Jeremy S. Sanders
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

import veusz.qtall as qt4
import itertools
import numpy as N

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
                              descr = 'Direction of bars', 
                              usertext='Direction'), 0 )
        s.add( setting.Dataset('posn', '', 
                                descr = 'Dataset containing position of bars',
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
        #xv = s.get('xData').getData(doc)
        #yv = s.get('yData').getData(doc)
        #text = s.get('labels').getData(doc, checknull=True)

        #if not xv or not yv:
        #    return

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

        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate a bar plotter
document.thefactory.register( BarPlotter )
