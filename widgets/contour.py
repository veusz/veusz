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

"""Contour plotting from 2d datasets."""

import qt
import numarray as N

import setting
import widgetfactory
import plotters

# try to get _na_contour from somewhere
# options are veusz_helpers or matplotlib
try:
    import veusz_helpers._na_contour as _contour
except ImportError:
    try:
        import matplotlib._na_contour as _contour
    except ImportError:
        _contour = None

class Contour(plotters.GenericPlotter):
    """A class which plots contours on a graph with a specified
    coordinate system."""

    typename='contour'
    allowusercreation=True
    description='Plot a 2d dataset as contours'

    def __init__(self, parent, name=None):
        """Initialise plotter with axes."""

        plotters.GenericPlotter.__init__(self, parent, name=name)

        s = self.settings
        s.add( setting.Dataset('data', '', self.document,
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
                                 [0.],
                                 descr = 'Levels to use for manual scaling'),
               5 )

        self.lastdataset = None
        self.schangeset = -1

    def _calculateLevels(self):
        """Calculate contour levels."""

        # get dataset
        s = self.settings
        d = self.document

        if s.data not in d.data:
            minval = 0.
            maxval = 1.
        else:
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
            levels = minval
        else:
            # trap out silly cases
            if minval == maxval:
                minval = 0.
                maxval = 1.
        
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
                levels = s.manualLevels

# allow the factory to instantiate a contour
widgetfactory.thefactory.register( Contour )
