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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
###############################################################################

# $Id$

import re
import sys

import numpy as N

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

from function import FunctionPlotter
import widget

try:
    import minuit
except ImportError:
    minuit = None

def minuitFit(evalfunc, params, names, values, xvals, yvals, yserr):
    """Do fitting with minuit (if installed)."""

    def chi2(params):
        """generate a lambda function to impedance-match between PyMinuit's
        use of multiple parameters versus our use of a single numpy vector."""
        c = ((evalfunc(params, xvals) - yvals)**2 / yserr**2).sum()
        if chi2.runningFit:
            chi2.iters += 1
            p = [chi2.iters, c] + params.tolist()
            str = ("%5i " + "%8g " * (len(params)+1)) % tuple(p)
            print str

        return c

    namestr = ', '.join(names)
    fnstr = 'lambda %s: chi2(N.array([%s]))' % (namestr, namestr)

    # this is safe because the only user-controlled variable is len(names)
    fn = eval(fnstr, {'chi2' : chi2, 'N' : N})

    print 'Fitting via Minuit:'
    m = minuit.Minuit(fn, fix_x=True, **values)

    # run the fit
    chi2.runningFit = True
    chi2.iters = 0
    m.migrad()

    # do some error analysis
    have_symerr, have_err = False, False
    try:
        chi2.runningFit = False
        m.hesse()
        have_symerr = True
        m.minos()
        have_err = True
    except minuit.MinuitError, e:
        print e
        if str(e).startswith('Discovered a new minimum'):
            # the initial fit really failed
            raise

    # print the results
    retchi2 = m.fval
    dof = len(yvals) - len(params)
    redchi2 = retchi2 / dof

    if have_err:
        print 'Fit results:\n', "\n".join([
            u"    %s = %g \u00b1 %g (+%g / %g)"
                % (n, m.values[n], m.errors[n], m.merrors[(n, 1.0)], m.merrors[(n, -1.0)]) for n in names])
    elif have_symerr:
        print 'Fit results:\n', "\n".join([
            u"    %s = %g \u00b1 %g" % (n, m.values[n], m.errors[n]) for n in names])
        print 'MINOS error estimate not available.'
    else:
        print 'Fit results:\n', "\n".join(['    %s = %g' % (n, m.values[n]) for n in names])
        print 'No error analysis available: fit quality uncertain'

    print "chi^2 = %g, dof = %i, reduced-chi^2 = %g" % (retchi2, dof, redchi2)

    vals = m.values
    return vals, retchi2, dof

class Fit(FunctionPlotter):
    """A plotter to fit a function to data."""

    typename='fit'
    allowusercreation=True
    description='Fit a function to data'

    def __init__(self, parent, name=None):
        FunctionPlotter.__init__(self, parent, name=name)

        if type(self) == Fit:
            self.readDefaults()

        self.addAction( widget.Action('fit', self.actionFit,
                                      descr = 'Fit function',
                                      usertext = 'Fit function') )

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        FunctionPlotter.addSettings(s)

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

    def providesAxesDependency(self):
        """This widget provides range information about these axes."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def updateAxisRange(self, axis, depname, axrange):
        """Update range with range of data."""
        dataname = {'sx': 'xData', 'sy': 'yData'}[depname]
        data = self.settings.get(dataname).getData(self.document)
        if data:
            drange = data.getRange()
            if drange:
                axrange[0] = min(axrange[0], drange[0])
                axrange[1] = max(axrange[1], drange[1])

    def initEnviron(self):
        """Copy data into environment."""
        env = self.document.eval_context.copy()
        env.update( self.settings.values )
        return env

    def actionFit(self):
        """Fit the data."""

        s = self.settings

        # update function for fitting
        try:
            self.checker.check(s.function, s.variable)
        except RuntimeError, e:
            self.logEvalError(e)
            return

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
                drange = self.parent.getAxes((s.xAxis,))[0].getPlottedRange()
                mask = N.logical_and(xvals >= drange[0], xvals <= drange[1])
            else:
                drange = self.parent.getAxes((s.yAxis,))[0].getPlottedRange()
                mask = N.logical_and(yvals >= drange[0], yvals <= drange[1])
            xvals, yvals, yserr = xvals[mask], yvals[mask], yserr[mask]
            print "Fitting %s from %g to %g" % (s.variable,
                                                drange[0], drange[1])

        # minimum set for fitting
        if s.min != 'Auto':
            if s.variable == 'x':
                mask = xvals >= s.min
            else:
                mask = yvals >= s.min
            xvals, yvals, yserr = xvals[mask], yvals[mask], yserr[mask]

        # maximum set for fitting
        if s.max != 'Auto':
            if s.variable == 'x':
                mask = xvals <= s.max
            else:
                mask = yvals <= s.max
            xvals, yvals, yserr = xvals[mask], yvals[mask], yserr[mask]

        if s.min != 'Auto' or s.max != 'Auto':
            print "Fitting %s between %s and %s" % (s.variable, s.min, s.max)

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

        # actually do the fit, either via Minuit or our own LM fitter
        chi2 = 1
        dof = 1

        if minuit is not None:
            vals, chi2, dof = minuitFit(self.evalfunc, params, names, s.values, xvals, yvals, yserr)
        else:
            print 'Minuit not available, falling back to simple L-M fitting:'
            retn, chi2, dof = utils.fitLM(self.evalfunc, params,
                                          xvals,
                                          yvals, yserr)
            vals = {}
            for i, v in zip(names, retn):
                vals[i] = float(v)

        # list of operations do we can undo the changes
        operations = []
                                      
        # populate the return parameters
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
            return eval(self.checker.compiled, env) + xvals*0.
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
