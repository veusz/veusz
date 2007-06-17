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

import re
import sys

import numpy as N

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import plotters
import widget

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
                                 descr = 'Variables and fit values',
                                 usertext='Parameters'), 1 )
        s.add( setting.Dataset('xData', 'x',
                               descr = 'Variable containing x data to fit',
                               usertext='X dataset'), 2 )
        s.add( setting.Dataset('yData', 'y',
                               descr = 'Variable containing y data to fit',
                               usertext='Y dataset'), 3 )
        s.add( setting.Bool('fitRange', False,
                            descr = 'Fit only the data between the '
                            'minimum and maximum of the axis for '
                            'the function variable',
                            usertext='Fit only range'),
               4 )
        s.add( setting.Str('outExpr', '',
                           descr = 'Output best fitting expression',
                           usertext='Output expression'),
               5, readonly=True )
        s.add( setting.Float('chi2', -1,
                             descr = 'Output chi^2 from fitting',
                             usertext='Fit &chi;<sup>2</sup>'),
               6, readonly=True )
        s.add( setting.Int('dof', -1,
                           descr = 'Output degrees of freedom from fitting',
                           usertext='Fit d.o.f.'),
               7, readonly=True )
        s.add( setting.Float('redchi2', -1,
                             descr = 'Output reduced-chi-squared from fitting',
                             usertext='Fit reduced &chi;<sup>2</sup>'),
               8, readonly=True )

        f = s.get('function')
        f.newDefault('a + b*x')
        f.descr = 'Function to fit'

        # modify description
        s.get('min').usertext='Min. fit range'
        s.get('max').usertext='Max. fit range'

        if type(self) == Fit:
            self.readDefaults()

        self.addAction( widget.Action('fit', self.actionFit,
                                      descr = 'Fit function',
                                      usertext = 'Fit function') )

    def _autoAxis(self, dataname, bounds):
        """Determine range of data."""
        if self.document.hasData(dataname):
            range = self.document.getData(dataname).getRange()
            if range:
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

        # populate the input parameters
        names = s.values.keys()
        names.sort()
        params = N.array( [s.values[i] for i in names] )

        # FIXME: loads of error handling!!
        d = self.document

        # choose dataset depending on fit variable
        if s.variable == 'x':
            xvals = d.getData(s.xData).data
            ydata = d.getData(s.yData)
            yvals = ydata.data
            yserr = ydata.serr
        else:
            xvals = d.getData(s.yData).data
            ydata = d.getData(s.xData)
            yvals = ydata.data
            yserr = ydata.serr

        # if there are no errors on data
        if yserr is None:
            if ydata.perr is not None and ydata.nerr is not None:
                print "Warning: Symmeterising positive and negative errors"
                yserr = N.sqrt( 0.5*(ydata.perr**2 + ydata.nerr**2) )
            else:
                print "Warning: No errors on y values. Assuming 5% errors."
                yserr = yvals*0.05
                yserr[yserr < 1e-8] = 1e-8
        
        # if the fitRange parameter is on, we chop out data outside the
        # range of the axis
        if s.fitRange:
            # get ranges for axes
            if s.variable == 'x':
                range = self.parent.getAxes((s.xAxis,))[0].getPlottedRange()
                mask = N.logical_and(xvals >= range[0], xvals <= range[1])
            else:
                range = self.parent.getAxes((s.yAxis,))[0].getPlottedRange()
                mask = N.logical_and(yvals >= range[0], yvals <= range[1])
            xvals = xvals[mask]
            yvals = yvals[mask]
            yserr = yserr[mask]
            print "Fitting %s from %g to %g" % (s.variable,
                                                range[0], range[1])

        # various error checks
        if len(xvals) == 0:
            sys.stderr.write('No data values. Not fitting.\n')
            return
        if len(xvals) != len(yvals) or len(xvals) != len(yserr):
            sys.stderr.write('Fit data not equal in length. Not fitting.\n')
            return
        if len(params) > len(xvals):
            sys.stderr.write('No degrees of freedom for fit. Not fitting\n')
            return

        # actually do the fit
        retn, chi2, dof = utils.fitLM(self.evalfunc, params,
                                      xvals,
                                      yvals, yserr)

        # list of operations do we can undo the changes
        operations = []
                                      
        # populate the return parameters
        vals = {}
        for i, v in zip(names, retn):
            vals[i] = float(v)
        operations.append( document.OperationSettingSet(s.get('values'), vals) )

        # populate the read-only fit quality params
        operations.append( document.OperationSettingSet(s.get('chi2'), float(chi2)) )
        operations.append( document.OperationSettingSet(s.get('dof'), int(dof)) )
        if dof <= 0:
            print 'No degrees of freedom in fit.\n'
            redchi2 = -1.
        else:
            redchi2 = float(chi2/dof)
        operations.append( document.OperationSettingSet(s.get('redchi2'), redchi2) )

        # expression for fit
        expr = self.generateOutputExpr(vals)
        operations.append( document.OperationSettingSet(s.get('outExpr'), expr) )

        # actually change all the settings
        d.applyOperation( document.OperationMultiple(operations, descr='fit') )
    
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
            return N.nan

    def generateOutputExpr(self, vals):
        """Try to generate text form of output expression.
        
        vals is a dict of variable: value pairs
        returns the expression
        """

        paramvals = vals.copy()
        s = self.settings

        # also substitute in data name for variable
        if s.variable == 'x':
            paramvals['x'] = s.xData
        else:
            paramvals['y'] = s.yData

        # split expression up into parts of text and nums, separated
        # by non-text/nums
        parts = re.split('([^A-Za-z0-9.])', s.function)

        # replace part by things in paramvals, if they exist
        for i, p in enumerate(parts):
            if p in paramvals:
                parts[i] = str( paramvals[p] )

        return ''.join(parts)

# allow the factory to instantiate an x,y plotter
document.thefactory.register( Fit )
