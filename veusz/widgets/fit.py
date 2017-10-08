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

from __future__ import division, absolute_import, print_function
import re
import sys

import numpy as N

from ..compat import czip, cstr
from .. import document
from .. import setting
from .. import utils
from .. import qtall as qt4

from .function import FunctionPlotter
from . import widget

# try importing iminuit first, then minuit, then None
try:
    import iminuit as minuit
except ImportError:
    try:
        import minuit
    except ImportError:
        minuit = None

def _(text, disambiguation=None, context='Fit'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

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
            print(str)

        return c

    namestr = ', '.join(names)
    fnstr = 'lambda %s: chi2(N.array([%s]))' % (namestr, namestr)

    # this is safe because the only user-controlled variable is len(names)
    fn = eval(fnstr, {'chi2' : chi2, 'N' : N})

    print(_('Fitting via Minuit:'))
    m = minuit.Minuit(fn, **values)

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
    except minuit.MinuitError as e:
        print(e)
        if str(e).startswith('Discovered a new minimum'):
            # the initial fit really failed
            raise

    # print the results
    retchi2 = m.fval
    dof = len(yvals) - len(params)
    redchi2 = retchi2 / dof

    if have_err:
        print(_('Fit results:\n') + "\n".join([
                    u"    %s = %g \u00b1 %g (+%g / %g)"
                    % (n, m.values[n], m.errors[n], m.merrors[(n, 1.0)],
                       m.merrors[(n, -1.0)]) for n in names]))
    elif have_symerr:
        print(_('Fit results:\n') + "\n".join([
                    u"    %s = %g \u00b1 %g" % (n, m.values[n], m.errors[n])
                    for n in names]))
        print(_('MINOS error estimate not available.'))
    else:
        print(_('Fit results:\n') + "\n".join([
                    '    %s = %g' % (n, m.values[n]) for n in names]))
        print(_('No error analysis available: fit quality uncertain'))

    print("chi^2 = %g, dof = %i, reduced-chi^2 = %g" % (retchi2, dof, redchi2))

    vals = m.values
    return vals, retchi2, dof

class Fit(FunctionPlotter):
    """A plotter to fit a function to data."""

    typename='fit'
    allowusercreation=True
    description=_('Fit a function to data')

    def __init__(self, parent, name=None):
        FunctionPlotter.__init__(self, parent, name=name)

        self.addAction( widget.Action('fit', self.actionFit,
                                      descr = _('Fit function'),
                                      usertext = _('Fit function')) )

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        FunctionPlotter.addSettings(s)

        s.add( setting.FloatDict(
                'values',
                {'a': 0.0, 'b': 1.0},
                descr = _('Variables and fit values'),
                usertext=_('Parameters')), 1 )
        s.add( setting.DatasetExtended(
                'xData', 'x',
                descr = _('X data to fit (dataset name, list of values '
                          'or expression)'),
                usertext=_('X data')), 2 )
        s.add( setting.DatasetExtended(
                'yData', 'y',
                descr = _('Y data to fit (dataset name, list of values '
                          'or expression)'),
                usertext=_('Y data')), 3 )
        s.add( setting.Bool(
                'fitRange', False,
                descr = _('Fit only the data between the '
                          'minimum and maximum of the axis for '
                          'the function variable'),
                usertext=_('Fit only range')),
               4 )
        s.add( setting.WidgetChoice(
                'outLabel', '',
                descr=_('Write best fit parameters to this text label '
                        'after fitting'),
                widgettypes=('label',),
                usertext=_('Output label')),
               5 )
        s.add( setting.Str('outExpr', '',
                           descr = _('Output best fitting expression'),
                           usertext=_('Output expression')),
               6, readonly=True )
        s.add( setting.Float('chi2', -1,
                             descr = 'Output chi^2 from fitting',
                             usertext=_('Fit &chi;<sup>2</sup>')),
               7, readonly=True )
        s.add( setting.Int('dof', -1,
                           descr = _('Output degrees of freedom from fitting'),
                           usertext=_('Fit d.o.f.')),
               8, readonly=True )
        s.add( setting.Float('redchi2', -1,
                             descr = _('Output reduced-chi-squared from fitting'),
                             usertext=_('Fit reduced &chi;<sup>2</sup>')),
               9, readonly=True )

        f = s.get('function')
        f.newDefault('a + b*x')
        f.descr = _('Function to fit')

        # modify description
        s.get('min').usertext=_('Min. fit range')
        s.get('max').usertext=_('Max. fit range')

    def affectsAxisRange(self):
        """This widget provides range information about these axes."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def getRange(self, axis, depname, axrange):
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
        env = self.document.evaluate.context.copy()
        env.update( self.settings.values )
        return env

    def updateOutputLabel(self, ops, vals, chi2, dof):
        """Use best fit parameters to update text label."""
        s = self.settings
        labelwidget = s.get('outLabel').findWidget()

        if labelwidget is not None:
            # build up a set of X=Y values
            loc = self.document.locale
            txt = []
            for l, v in sorted(vals.items()):
                val = utils.formatNumber(v, '%.4Vg', locale=loc)
                txt.append( '%s = %s' % (l, val) )
            # add chi2 output
            txt.append( r'\chi^{2}_{\nu} = %s/%i = %s' % (
                    utils.formatNumber(chi2, '%.4Vg', locale=loc),
                    dof,
                    utils.formatNumber(chi2/dof, '%.4Vg', locale=loc) ))

            # update label with text
            text = r'\\'.join(txt)
            ops.append( document.OperationSettingSet(
                    labelwidget.settings.get('label') , text ) )

    def actionFit(self):
        """Fit the data."""

        s = self.settings

        # check and get compiled for of function
        compiled = self.document.evaluate.compileCheckedExpression(s.function)
        if compiled is None:
            return

        # populate the input parameters
        paramnames = sorted(s.values)
        params = N.array( [s.values[p] for p in paramnames] )

        # FIXME: loads of error handling!!
        d = self.document

        # choose dataset depending on fit variable
        if s.variable == 'x':
            xvals = s.get('xData').getData(d).data
            ydata = s.get('yData').getData(d)
        else:
            xvals = s.get('yData').getData(d).data
            ydata = s.get('xData').getData(d)
        yvals = ydata.data
        yserr = ydata.serr

        # if there are no errors on data
        if yserr is None:
            if ydata.perr is not None and ydata.nerr is not None:
                print("Warning: Symmeterising positive and negative errors")
                yserr = N.sqrt( 0.5*(ydata.perr**2 + ydata.nerr**2) )
            else:
                print("Warning: No errors on y values. Assuming 5% errors.")
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
            print("Fitting %s from %g to %g" % (s.variable,
                                                drange[0], drange[1]))

        evalenv = self.initEnviron()
        def evalfunc(params, xvals):
            # update environment with variable and parameters
            evalenv[self.settings.variable] = xvals
            evalenv.update( czip(paramnames, params) )

            try:
                return eval(compiled, evalenv) + xvals*0.
            except Exception as e:
                self.document.log(cstr(e))
                return N.nan

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
            print("Fitting %s between %s and %s" % (s.variable, s.min, s.max))

        # various error checks
        if len(xvals) != len(yvals) or len(xvals) != len(yserr):
            sys.stderr.write(_('Fit data not equal in length. Not fitting.\n'))
            return
        if len(params) > len(xvals):
            sys.stderr.write(_('No degrees of freedom for fit. Not fitting\n'))
            return

        # actually do the fit, either via Minuit or our own LM fitter
        chi2 = 1
        dof = 1

        # only consider finite values
        finite = N.isfinite(xvals) & N.isfinite(yvals) & N.isfinite(yserr)
        xvals = xvals[finite]
        yvals = yvals[finite]
        yserr = yserr[finite]

        # check length after excluding non-finite values
        if len(xvals) == 0:
            sys.stderr.write(_('No data values. Not fitting.\n'))
            return

        if minuit is not None:
            vals, chi2, dof = minuitFit(evalfunc, params, paramnames, s.values,
                                        xvals, yvals, yserr)
        else:
            print(_('Minuit not available, falling back to simple L-M fitting:'))
            retn, chi2, dof = utils.fitLM(evalfunc, params,
                                          xvals,
                                          yvals, yserr)
            vals = {}
            for i, v in czip(paramnames, retn):
                vals[i] = float(v)

        # list of operations do we can undo the changes
        operations = []
                                      
        # populate the return parameters
        operations.append( document.OperationSettingSet(s.get('values'), vals) )

        # populate the read-only fit quality params
        operations.append( document.OperationSettingSet(s.get('chi2'), float(chi2)) )
        operations.append( document.OperationSettingSet(s.get('dof'), int(dof)) )
        if dof <= 0:
            print(_('No degrees of freedom in fit.\n'))
            redchi2 = -1.
        else:
            redchi2 = float(chi2/dof)
        operations.append( document.OperationSettingSet(s.get('redchi2'), redchi2) )

        # expression for fit
        expr = self.generateOutputExpr(vals)
        operations.append( document.OperationSettingSet(s.get('outExpr'), expr) )

        self.updateOutputLabel(operations, vals, chi2, dof)

        # actually change all the settings
        d.applyOperation(
            document.OperationMultiple(operations, descr=_('fit')) )
    
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
