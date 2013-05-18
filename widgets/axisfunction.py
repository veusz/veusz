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

class FunctionError(AxisError):
    pass

def solveFunction(function, vals, mint=None, maxt=None):
    '''Solve a function for a list of values (vals), if we don't know
    where the solution lies. function is a function to call.

    This tries a range of possible input values, and uses binary
    search to refine the solution.

    mint and maxt are the bounds to use when solving
    '''

    xvals = N.array(
        ( -1e90, -1e70, -1e50, -1e40, -1e30, -1e20,
           -1e10, -1e8, -1e6, -1e5, -1e4,
           -1e3, -1e2, -1e1, -4e0, -2e0, -1e0,
           -1e-1, -1e-2, -1e-3, -1e-4,
           -1e-6, -1e-8, -1e-10, -1e-12, -1e-14,
           -1e-18, -1e-22, -1e-26, -1e-30, -1e-34,
           -1e-40, -1e-50, -1e-70, -1e-90,
           0,
           1e-90, 1e-70, 1e-50, 1e-40,
           1e-34, 1e-30, 1e-26, 1e-22, 1e-18,
           1e-14, 1e-12, 1e-10, 1e-8, 1e-6,
           1e-4, 1e-3, 1e-2, 1e-1,
           1e0, 2e0, 4e0, 1e1, 1e2, 1e3,
           1e4, 1e5, 1e6, 1e8, 1e10,
           1e20, 1e30, 1e40, 1e50, 1e70, 1e90 ))

    if mint is not None:
        xvals = N.hstack(( mint, xvals[xvals > mint] ))
    if maxt is not None:
        xvals = N.hstack(( xvals[xvals < maxt], maxt ))

    # yvalue in correct shape
    yvals = function(xvals) + N.zeros(len(xvals))

    anynan = N.any( N.isnan(yvals) )
    if anynan:
        raise FunctionError, _('Invalid regions in function '
                               '(try setting minimum or maximum t)')

    # remove any infinite regions
    f = N.isfinite(yvals)
    xfilt = xvals[f]
    yfilt = yvals[f]

    if len(yfilt) < 2:
        raise FunctionError, _('Solutions to equation cannot be found')

    # check for monotonicity
    delta = yfilt[1:] - yfilt[:-1]
    pos, neg = N.all(delta >= 0), N.all(delta <= 0)
    if not (pos or neg):
        raise FunctionError, _('Not a monotonic function '
                               '(try setting minimum or maximum t)')
    if pos and neg:
        raise FunctionError, _('Constant function')

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

    isaxisfunction = True

    def __init__(self, *args, **argsv):
        axis.Axis.__init__(self, *args, **argsv)

        self.cachedfunctxt = None
        self.cachedcompiled = None
        self.cachedfuncobj = None
        self.cachedbounds = None
        self.funcchangeset = -1
        self.boundschangeset = -1

        if type(self) == AxisFunction:
            self.readDefaults()

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
        s.add( setting.FloatOrAuto('mint', 'Auto',
                                   descr=_('Minimum value of t or Auto'),
                                   usertext=('Min t')), 3 )
        s.add( setting.FloatOrAuto('maxt', 'Auto',
                                   descr=_('Maximum value of t or Auto'),
                                   usertext=('Max t')), 4 )


        s.get('min').hidden = True
        s.get('max').hidden = True
        s.get('match').hidden = True
        s.get('autoRange').hidden = True
        s.get('autoRange').val = 'exact'

    def logError(self, ex):
        '''Write error message to document log for exception ex.'''
        self.document.log(
            _("Error in axis-function (%s): '%s'") % (
                self.settings.function, unicode(ex)))

    def getMinMaxT(self):
        '''Get minimum and maximum t.'''
        mint = self.settings.mint
        if mint == 'Auto':
            mint = None
        maxt = self.settings.maxt
        if maxt == 'Auto':
            maxt = None
        return mint, maxt

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
                self.logError(msg)
            else:
                # compile result
                try:
                    self.cachedcompiled = compile(functxt, '<string>', 'eval')
                except Exception, e:
                    self.logError(e)

        if self.cachedcompiled is None:
            self.cachedfuncobj = None
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
                    return N.nan + t
            self.cachedfuncobj = function

            mint, maxt = self.getMinMaxT()
            try:
                solveFunction(function, [0.], mint=mint, maxt=maxt)
            except FunctionError, e:
                self.logError(e)
                self.cachedfuncobj = None
            except AxisError, e:
                pass

        return self.cachedfuncobj

    def invertFunctionVals(self, vals):
        '''Convert values which are a function of fn and compute t.'''
        fn = self.getFunction()
        if fn is None:
            return None
        mint, maxt = self.getMinMaxT()
        try:
            return solveFunction(fn, vals, mint=mint, maxt=maxt)
        except Exception, e:
            self.logError(e)
            return None

    def lookupAxis(self, axisname):
        '''Find widget associated with axisname.'''
        w = self.parent
        while w:
            for c in w.children:
                if ( c.name == axisname and c.isaxis and
                     c is not self ):
                    return c
            w = w.parent
        return None

    def getOtherAxis(self):
        '''Get the widget for the other axis.'''
        other = self.lookupAxis(self.settings.otheraxis)
        if other is self:
            return None
        return other

    def computePlottedRange(self, force=False):
        '''Use other axis to compute range.'''

        if self.docchangeset == self.document.changeset and not force:
            return

        therange = None

        other = self.getOtherAxis()
        fn = self.getFunction()
        if other is not None and fn is not None:
            # compute our range from the other axis
            other.computePlottedRange()
            try:
                therange = fn(N.array(other.plottedrange)) * N.ones(2)
            except Exception, e:
                self.logError(e)
            if not N.all( N.isfinite(therange) ):
                therange = None

        axis.Axis.computePlottedRange(self, force=force,
                                      overriderange=therange)

    def updateAxisLocation(self, bounds, otherposition=None,
                           lowerupperposition=None):
        '''Calculate conversion from pixels to axis values.'''

        if ( self.boundschangeset == self.document.changeset and
             bounds == self.cachedbounds ):
            # don't recalculate unless document updated or bounds changes
            return
        self.cachedbounds = list(bounds)
        self.boundschangeset = self.document.changeset

        axis.Axis.updateAxisLocation(self, bounds, otherposition=otherposition,
                                     lowerupperposition=lowerupperposition)

        other = self.getOtherAxis()
        self.graphcoords = None
        if other is None:
            return

        if self.settings.direction == 'horizontal':
            p1, p2 = bounds[0], bounds[2]
        else:
            p1, p2 = bounds[1], bounds[3]
        w = p2-p1

        # To do the inverse calculation, we define a grid of pixel
        # values.  We need some sensitivity outside the axis range to
        # get angles of lines correct.

        pixcoords = N.hstack((
                N.linspace(p1-10.*w, p1-2.0*w, 5, endpoint=False),
                N.linspace(p1-2.0*w, p1-0.2*w, 25, endpoint=False),
                N.linspace(p1-0.2*w, p2+0.2*w, 250, endpoint=False),
                N.linspace(p2+0.2*w, p2+2.0*w, 25, endpoint=False),
                N.linspace(p2+2.0*w, p2+10.*w, 5, endpoint=False)
                ))

        # lookup what pixels are on other axis
        othercoordvals = other.plotterToGraphCoords(bounds, pixcoords)

        # chop to range
        mint, maxt = self.getMinMaxT()
        if mint is not None and mint > othercoordvals[0]:
            # only use coordinates bigger than mint, and add on mint
            # at beginning
            mintpix = other.graphToPlotterCoords(bounds, N.array([mint]))
            sel = othercoordvals > mint
            othercoordvals = N.hstack((mint, othercoordvals[sel]))
            pixcoords = N.hstack((mintpix, pixcoords[sel]))
        if maxt is not None and maxt < othercoordvals[-1]:
            maxtpix = other.graphToPlotterCoords(bounds, N.array([maxt]))
            sel = othercoordvals < maxt
            othercoordvals = N.hstack((othercoordvals[sel], maxt))
            pixcoords = N.hstack((pixcoords[sel], maxtpix))

        try:
            ourgraphcoords = self.getFunction()(othercoordvals)
        except Exception, e:
            return

        deltas = ourgraphcoords[1:] - ourgraphcoords[:-1]
        pos = N.all(deltas >= 0.)
        neg = N.all(deltas <= 0.)
        if (not pos and not neg) or (pos and neg):
            self.logError(_('Not a monotonic function'))
            return

        # Select only finite vals. We store _inv coords separately
        # as linear interpolation requires increasing values.
        f = N.isfinite(othercoordvals + ourgraphcoords)
        self.graphcoords = self.graphcoords_inv = ourgraphcoords[f]
        self.pixcoords = self.pixcoords_inv = pixcoords[f]

        if len(self.graphcoords) == 0:
            self.graphcoords = None
            return

        if self.graphcoords[0] > self.graphcoords[-1]:
            # order must be increasing (for forward conversion)
            self.graphcoords = self.graphcoords[::-1]
            self.pixcoords = self.pixcoords[::-1]

    def _linearInterpolWarning(self, vals, xcoords, ycoords):
        '''Linear interpolation, giving out of bounds warning.'''
        if any(vals < xcoords[0]) or any(vals > xcoords[-1]):
            self.document.log(
                _('Warning: values exceed bounds in axis-function'))

        return N.interp(vals, xcoords, ycoords)

    def _graphToPlotter(self, vals):
        '''Override normal axis graph->plotter coords to do lookup.'''
        if self.graphcoords is None:
            return axis.Axis._graphToPlotter(self, vals)
        else:
            return self._linearInterpolWarning(
                vals, self.graphcoords, self.pixcoords)

    def plotterToGraphCoords(self, bounds, vals):
        '''Override normal axis plotter->graph coords to do lookup.'''
        if self.graphcoords is None:
            return axis.Axis.plotterToGraphCoords(self, bounds, vals)
        else:
            self.updateAxisLocation(bounds)

            return self._linearInterpolWarning(
                vals, self.pixcoords_inv, self.graphcoords_inv)

# allow the factory to instantiate the widget
document.thefactory.register( AxisFunction )
