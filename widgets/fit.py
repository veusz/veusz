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

import numarray as N
import numarray.ieeespecial as NIE

import plotters
import widgetfactory
import setting
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

    def _autoAxis(self, dataname, bounds):
        """Determine range of data."""
        if self.document.hasData(dataname):
            range = self.document.getData(dataname).getRange()
            bounds[0] = min( bounds[0], range[0] )
            bounds[1] = max( bounds[1], range[1] )

    def autoAxis(self, name, bounds):
        """Automatically determine the ranges of variable on the axes."""

        s = self.settings
        if name == s.xAxis:
            self._autoAxis( s.xData, bounds )
        elif name == s.yAxis:
            self._autoAxis( s.yData, bounds )

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
        params = N.array( [s.values[i] for i in names] )

        # FIXME: loads of error handling!!
        d = self.document

        xvals = d.getData(s.xData).data
        ydata = d.getData(s.yData)
        yvals = ydata.data
        yserr = ydata.serr

        if yserr == None:
            if ydata.perr != None and ydata.nerr != None:
                print "Warning: Symmeterising positive and negative errors"
                yserr = N.sqrt( ydata.perr**2 + ydata.nerr**2 )
            else:
                print "Warning: No errors on y values. Assuming 5% errors."
                yserr = yvals*0.05
        
        retn = utils.fitLM(self.evalfunc, params,
                           xvals,
                           yvals, yserr)

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
            return NIE.nan

# allow the factory to instantiate an x,y plotter
widgetfactory.thefactory.register( Fit )
