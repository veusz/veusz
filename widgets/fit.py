# fit.py
# fitting plotter

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

import plotters
import widgetfactory
import setting
import numarray
import numarray.ieeespecial
import utils

class Fit(plotters.FunctionPlotter):
    """A plotter to fit a function to data."""

    typename='fit'
    allowusercreation=True
    description='Fit a function to data'

    def __init__(self, parent, name=None):
        plotters.FunctionPlotter.__init__(self, parent, name=name)

        s = self.settings
        s.add( setting.FloatDict('values',
                                 {'a': 0.0, 'b': 1.0},
                                 descr = 'Variables and fit values'), 1 )
        s.add( setting.Str('xData', 'x',
                           descr = 'X positions of data to fit'), 2 )
        s.add( setting.Str('yData', 'y',
                           descr = 'Y positions of data to fit'), 3 )

        f = s.get('function')
        f.newDefault('a + b*x')
        f.descr = 'Function to fit'

        # FIXME need way not to read defaults if not "topmost" class
        s.readDefaults()

        self.addAction( 'fit', self.actionFit,
                        descr='Fit function' )

    def _autoAxis(self, dataname):
        """Determine range of data."""
        if self.document.hasData(dataname):
            return self.getDocument().getData(dataname).getRange()
        else:
            return None

    def autoAxis(self, name):
        """Automatically determine the ranges of variable on the axes."""
        s = self.settings
        if name == s.xAxis:
            return self._autoAxis( s.xData )
        elif name == s.yAxis:
            return self._autoAxis( s.yData )
        else:
            return None

    def initEnviron(self):
        """Copy data into environment."""
        env = self.fnenviron.copy()
        for name, val in self.settings.values.iteritems():
            env[name] = val
        return env

    def actionFit(self):
        """Fit the data."""

        s = self.settings
        names = s.values.keys()
        names.sort()
        params = numarray.array( [s.values[i] for i in names] )

        # FIXME: loads of error handling!!
        d = self.document
        retn = utils.fitLM(self.evalfunc, params,
                           d.getData(s.xData).data,
                           d.getData(s.yData).data,
                           1. / d.getData(s.yData).serr)

        vals = {}
        for i, v in zip(names, retn):
            vals[i] = v
        s.values = vals

    def evalfunc(self, params, xvals):

        # make an environment
        env = self.initEnviron()
        s = self.settings
        env[s.variable] = xvals

        # set values for real function
        names = s.values.keys()
        names.sort()
        for name, val in zip(names, params):
            env[name] = val

        try:
            return eval(s.function, env)
        except:
            return numarray.ieeespecial.nan

# allow the factory to instantiate an x,y plotter
widgetfactory.thefactory.register( Fit )
