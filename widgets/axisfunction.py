#    Copyright (C) 2013 Jeremy S. Sanders
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
##############################################################################

'''An axis based on a function of another axis.'''

import numpy as N

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.document as document
import veusz.utils as utils

import axis

def _(text, disambiguation=None, context='FunctionAxis'):
    '''Translate text.'''
    return unicode( 
        qt4.QCoreApplication.translate(context, text, disambiguation))

class AxisError(RuntimeError):
    pass

def solveFunction(function, vals):
    '''Solve a function for a list of values (vals), if we don't know
    where the solution lies. function is a function to call.

    This tries a range of possible input values, and uses binary
    search to refine the solution.
    '''

    xvals = N.array(
        ( -1e90, -1e70, -1e50, -1e40, -1e30, -1e20,
           -1e10, -1e8, -1e6, -1e5, -1e4,
           -1e3, -1e2, -1e1, -1e0,
           -1e-1, -1e-2, -1e-3, -1e-4,
           -1e-6, -1e-8, -1e-10, -1e-12, -1e-14,
           -1e-18, -1e-22, -1e-26, -1e-30, -1e-34,
           -1e-40, -1e-50, -1e-70, -1e-90,
           0,
           1e-90, 1e-70, 1e-50, 1e-40,
           1e-34, 1e-30, 1e-26, 1e-22, 1e-18,
           1e-14, 1e-12, 1e-10, 1e-8, 1e-6,
           1e-4, 1e-3, 1e-2, 1e-1,
           1e0, 1e1, 1e2, 1e3,
           1e4, 1e5, 1e6, 1e8, 1e10,
           1e20, 1e30, 1e40, 1e50, 1e70, 1e90 ))

    # yvalue in correct shape
    yvals = function(xvals) + N.zeros(len(xvals))

    # remove any regions where the function goes wrong
    f = N.isfinite(yvals)
    xfilt = xvals[f]
    yfilt = yvals[f]

    if len(yfilt) < 2:
        raise AxisError, _('Solutions to equation cannot be found')

    # check for monotonicity
    delta = N.sign(yfilt[1:] - yfilt[:-1])
    pos, neg = N.all(delta >= 0), N.all(delta <= 0)
    if not (pos or neg):
        raise AxisError, _('Not a monotonic function')
    if pos and neg:
        raise AxisError, _('Constant function')

    # easier if the values are increasing only
    if neg:
        yfilt = yfilt[::-1]
        xfilt = xfilt[::-1]

    # do binary search for each input value
    out = []
    for thisval in vals:
        # renorm to zero
        ydelta = yfilt - thisval

        # solution is between this and the next
        idx = N.searchsorted(ydelta, 0.)
        if idx == 0:
            if ydelta[0] == 0.:
                # work around value being at start of array
                idx = 1
            else:
                raise AxisError, _('No solution found')
        elif idx == len(ydelta):
            raise AxisError, _('No solution found')

        x1, x2 = xfilt[idx-1], xfilt[idx]
        y1, y2 = ydelta[idx-1], ydelta[idx]

        # binary search
        tol = abs(1e-6 * thisval)
        for i in xrange(30):
            # print x1, y1, "->", x2, y2

            if abs(y1) <= tol and abs(y1) < abs(y2):
                x2, y2 = x1, y1
                break   # found solution
            if abs(y2) <= tol:
                x1, y1 = x2, y2
                break   # found solution

            if y1 == y2 or ((y1<0) and (y2<0)) or ((y1>0) and (y2>0)):
                raise AxisError, _('No solution found')

            ### This is a bit faster, but bisection is simpler
            # xv = N.linspace(x1, x2, num=100)
            # yv = function(xv) + xv*0. - thisval

            # idx = N.searchsorted(yv, 0.)
            # if idx == 0:
            #     idx = 1

            # x1 = xv[idx-1]
            # y1 = yv[idx-1]
            # x2 = xv[idx]
            # y2 = yv[idx]

            x3 = 0.5*(x1+x2)
            y3 = function(x3) - thisval
            if not N.isfinite(y3):
                raise AxisError, _('Non-finite value encountered')

            if y3 < 0:
                x1 = x3
                y1 = y3
            else:
                x2 = x3
                y2 = y3

        out.append(0.5*(x1+x2))

    return out

class AxisFunction(axis.Axis):
    '''An axis using an function of another axis.'''

    typename = 'axis-function'
    description = 'An axis based on a function of the values of another axis'

    def __init__(self, *args, **argsv):
        axis.Axis.__init__(self, *args, **argsv)

        self.cachedfunctxt = None
        self.cachedcompiled = None
        self.cachedfuncobj = None
        self.funcchangeset = -1

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        axis.Axis.addSettings(s)

        s.add( setting.Str('function', 't',
                           descr=_('Monotonic function (use t as variable)'),
                           usertext=_('Function')), 1 )
        s.add( setting.Axis('otheraxis', '', 'both',
                            descr =
                            _('Axis for which this axis is based on'),
                            usertext=_('Other axis')), 2 )

        s.get('min').hidden = True
        s.get('max').hidden = True
        s.get('match').hidden = True
        s.get('autoRange').hidden = True
        s.get('autoRange').val = 'exact'

    def logError(self, ex):
        '''Write error message to document log for exception ex.'''
        self.document.log(
            _("Error in axis function: '%s'") % unicode(ex))

    def getFunction(self):
        '''Check whether function needs to be compiled.'''

        if self.funcchangeset == self.document.changeset:
            return self.cachedfuncobj
        self.funcchangeset = self.document.changeset

        functxt = self.settings.function.strip()
        if functxt != self.cachedfunctxt:
            self.cachedfunctxt = functxt
            self.cachedcompiled = None

            # check function obeys safety rules
            checked = utils.checkCode(functxt)
            if checked is not None:
                try:
                    msg = checked[0][0]
                except Exception, e:
                    msg = e
                logError(msg)
            else:
                # compile result
                try:
                    self.cachedcompiled = compile(functxt, '<string>', 'eval')
                except Exception, e:
                    self.logError(e)

        if self.cachedcompiled is None:
            self.cachedfuncobj = lambda t: N.nan
        else:
            # a python function for doing the evaluation and handling
            # errors
            env = self.document.eval_context.copy()
            def function(t):
                env['t'] = t
                try:
                    return eval(self.cachedcompiled, env)
                except Exception, e:
                    self.logError(e)
                    return N.nan
            self.cachedfuncobj = function

        return self.cachedfuncobj

    def getOtherAxis(self):
        '''Get the widget for the other axis.'''
        other = self.lookupAxis(self.settings.otheraxis)
        if other is self:
            return None
        return other

    def setAutoRange(self, autorange):
        axis.Axis.setAutoRange(self, autorange)

        # convert values on this axis to the other axis by solving
        # the equation
        other = self.getOtherAxis()
        if other is not None:
            try:
                inverse = solveFunction(self.getFunction(), autorange)
            except Exception, e:
                self.logError(e)
                return
            other.setAutoRange(inverse)

    def getRange(self, axis, depname, axrange):
        '''Update range variable for axis with dependency name given.'''
        print "uAR", axis, depname, axrange

    def computePlottedRange(self, force=False):
        '''Use other axis to compute range.'''

        if self.docchangeset == self.document.changeset and not force:
            return

        therange = None

        other = self.lookupAxis(self.settings.otheraxis)
        if other is not None:
            # compute our range from the other axis

            #other.computePlottedRange()
            print other.autorange, other.plottedrange

            try:
                fn = self.getFunction()
                therange = fn(N.array(other.plottedrange)) * N.ones(2)
            except Exception, e:
                self.logError(e)

        axis.Axis.computePlottedRange(self, force=force,
                                      overriderange=therange)

# allow the factory to instantiate the widget
document.thefactory.register( AxisFunction )
