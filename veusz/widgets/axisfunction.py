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

from __future__ import division
import numpy as N

from ..compat import crange, cstr
from .. import qtall as qt
from .. import setting
from .. import document

from . import axis

def _(text, disambiguation=None, context='FunctionAxis'):
    '''Translate text.'''
    return qt.QCoreApplication.translate(context, text, disambiguation)

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
    try:
        yvals = function(xvals) + N.zeros(len(xvals))
    except Exception as e:
        raise FunctionError(_('Error evaluating function: %s') % cstr(e))

    anynan = N.any( N.isnan(yvals) )
    if anynan:
        raise FunctionError(_('Invalid regions in function '
                               '(try setting minimum or maximum t)'))

    # remove any infinite regions
    f = N.isfinite(yvals)
    xfilt = xvals[f]
    yfilt = yvals[f]

    if len(yfilt) < 2:
        raise FunctionError(_('Solutions to equation cannot be found'))

    # check for monotonicity
    delta = yfilt[1:] - yfilt[:-1]
    pos, neg = N.all(delta >= 0), N.all(delta <= 0)
    if not (pos or neg):
        raise FunctionError(_('Not a monotonic function '
                               '(try setting minimum or maximum t)'))
    if pos and neg:
        raise FunctionError(_('Constant function'))

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
                raise AxisError(_('No solution found'))
        elif idx == len(ydelta):
            raise AxisError(_('No solution found'))

        x1, x2 = xfilt[idx-1], xfilt[idx]
        y1, y2 = ydelta[idx-1], ydelta[idx]

        # binary search
        tol = abs(1e-6 * thisval)
        for i in crange(30):
            # print x1, y1, "->", x2, y2

            if abs(y1) <= tol and abs(y1) < abs(y2):
                x2, y2 = x1, y1
                break   # found solution
            if abs(y2) <= tol:
                x1, y1 = x2, y2
                break   # found solution

            if y1 == y2 or ((y1<0) and (y2<0)) or ((y1>0) and (y2>0)):
                raise AxisError(_('No solution found'))

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
                raise AxisError(_('Non-finite value encountered'))

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

        self.cachedfuncobj = None
        self.cachedbounds = None
        self.funcchangeset = -1
        self.boundschangeset = -1

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        axis.Axis.addSettings(s)

        s.add( setting.BoolSwitch(
                'linked', False,
                settingsfalse=('min', 'max'),
                settingstrue=('linkedaxis',),
                descr=_('Link axis to another axis'),
                usertext=_('Linked') ), 0 )

        s.add( setting.Str('function', 't',
                           descr=_('Monotonic function (use t as variable)'),
                           usertext=_('Function')), 1 )
        s.add( setting.Axis('linkedaxis', '', 'both',
                            descr =
                            _('Axis which this axis is based on'),
                            usertext=_('Linked axis')), 6 )
        s.add( setting.FloatOrAuto('mint', 'Auto',
                                   descr=_('Minimum value of t or Auto'),
                                   usertext=('Min t')), 7 )
        s.add( setting.FloatOrAuto('maxt', 'Auto',
                                   descr=_('Maximum value of t or Auto'),
                                   usertext=('Max t')), 8 )

        s.get('autoRange').hidden = True
        s.get('autoRange').newDefault('exact')

    @property
    def userdescription(self):
        """User friendly description."""
        s = self.settings
        return _("axis='%s', function='%s'") % (s.linkedaxis, s.function)

    def logError(self, ex):
        '''Write error message to document log for exception ex.'''
        self.document.log(
            _("Error in axis-function (%s): '%s'") % (
                self.settings.function, cstr(ex)))

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

        compiled = self.document.evaluate.compileCheckedExpression(
            self.settings.function.strip())

        if compiled is None:
            self.cachedfuncobj = None
        else:
            # a python function for doing the evaluation and handling
            # errors
            env = self.document.evaluate.context.copy()

            def function(t):
                env['t'] = t
                try:
                    return eval(compiled, env)
                except Exception as e:
                    self.logError(e)
                    return N.nan + t
            self.cachedfuncobj = function

            mint, maxt = self.getMinMaxT()
            try:
                solveFunction(function, [0.], mint=mint, maxt=maxt)
            except FunctionError as e:
                self.logError(e)
                self.cachedfuncobj = None
            except AxisError:
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
        except Exception as e:
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

    def isLinked(self):
        '''Is this axis linked to another?'''
        return self.settings.linked

    def getLinkedAxis(self):
        '''Get the widget for the linked axis.'''
        if not self.settings.linked:
            return None
        linked = self.lookupAxis(self.settings.linkedaxis)
        if linked is self:
            return None
        return linked

    def isLinear(self):
        return False

    def computePlottedRange(self, force=False):
        '''Use other axis to compute range.'''

        if self.docchangeset == self.document.changeset and not force:
            return

        therange = None

        linked = self.getLinkedAxis()
        fn = self.getFunction()
        if linked is not None and fn is not None:
            # compute our range from the linked axis
            linked.computePlottedRange()
            try:
                therange = fn(N.array(linked.plottedrange)) * N.ones(2)
            except Exception as e:
                self.logError(e)
            if not N.all( N.isfinite(therange) ):
                therange = None

        axis.Axis.computePlottedRange(self, force=force,
                                      overriderange=therange)

    def _orderCoordinates(self):
        '''Put coordinates in correct order for linear interpolation.'''

        if len(self.graphcoords) == 0:
            self.graphcoords = None
            return

        if self.graphcoords[0] > self.graphcoords[-1]:
            # order must be increasing (for forward conversion)
            self.graphcoords = self.graphcoords[::-1]
            self.pixcoords = self.pixcoords[::-1]
        if self.pixcoords_inv[0] > self.pixcoords_inv[-1]:
            # likewise increasing order for inverse
            self.pixcoords_inv = self.pixcoords_inv[::-1]
            self.graphcoords_inv = self.graphcoords_inv[::-1]

    def _updateLinkedAxis(self, bounds, fraccoords):
        '''Calculate coordinate conversion for linked axes.'''

        link = self.getLinkedAxis()
        if link is None:
            return

        # To do the inverse calculation, we define a grid of pixel
        # values.  We need some sensitivity outside the axis range to
        # get angles of lines correct. We start with fractional graph
        # coordinates to translate to the other axis coordinates.

        # coordinate values on the other axis
        try:
            linkwidth = link.coordParr2-link.coordParr1
            linkorigin = link.coordParr1
            linkbounds = link.currentbounds
        except AttributeError:
            # if hasn't been initialised
            return

        # lookup what pixels are on linked axis in values
        linkpixcoords = fraccoords*linkwidth + linkorigin
        linkgraphcoords = link.plotterToGraphCoords(linkbounds, linkpixcoords)

        # flip round if coordinates reversed
        if linkgraphcoords[0] > linkgraphcoords[-1]:
            linkgraphcoords = linkgraphcoords[::-1]
            linkpixcoords = linkpixcoords[::-1]
            fraccoords = fraccoords[::-1]

        # Chop to range. This is rather messy as there are several
        # sets of coordinates to extend and chop: graph coordinates,
        # pixel coordinates and fractional coordinates.
        mint, maxt = self.getMinMaxT()
        if mint is not None and mint > linkgraphcoords[0]:
            mintpix = link.graphToPlotterCoords(linkbounds, N.array([mint]))
            sel = linkgraphcoords > mint
            linkgraphcoords = N.hstack((mint, linkgraphcoords[sel]))
            linkpixcoords = N.hstack((mintpix, linkpixcoords[sel]))
            frac = (mintpix - linkorigin) / linkwidth
            fraccoords = N.hstack((frac, fraccoords[sel]))
        if maxt is not None and maxt < linkgraphcoords[-1]:
            maxtpix = link.graphToPlotterCoords(linkbounds, N.array([maxt]))
            sel = linkgraphcoords < maxt
            linkgraphcoords = N.hstack((linkgraphcoords[sel], maxt))
            linkpixcoords = N.hstack((linkpixcoords[sel], maxtpix))
            frac = (maxtpix - linkorigin) / linkwidth
            fraccoords = N.hstack((fraccoords[sel], frac))

        try:
            ourgraphcoords = self.getFunction()(linkgraphcoords)
        except:
            return

        deltas = ourgraphcoords[1:] - ourgraphcoords[:-1]
        pos = N.all(deltas >= 0.)
        neg = N.all(deltas <= 0.)
        if (not pos and not neg) or (pos and neg):
            self.logError(_('Not a monotonic function'))
            return

        # Select only finite vals. We store _inv coords separately
        # as linear interpolation requires increasing values.
        f = N.isfinite(linkgraphcoords + ourgraphcoords)
        self.graphcoords = self.graphcoords_inv = ourgraphcoords[f]

        # This is true if the axis is plotting on the same graph in
        # the same direction. If this is the case, use our coordinates
        # directly.
        if ( link.settings.direction == self.settings.direction
             and link.currentbounds == bounds ):
            self.pixcoords = self.pixcoords_inv = linkpixcoords[f]
        else:
            # convert fractions to our coordinates
            self.pixcoords = self.pixcoords_inv = (
                fraccoords[f]*(self.coordParr2-self.coordParr1)+self.coordParr1)

        # put output coordinates in correct order
        self._orderCoordinates()

    def _updateFreeAxis(self, bounds, fraccoords):
        '''Calculate coordinates for a free axis.'''

        self.computePlottedRange()
        trange = self.invertFunctionVals(N.array(self.plottedrange))
        if trange is None:
            return

        tvals = fraccoords*(trange[1]-trange[0]) + trange[0]

        if tvals[0] > tvals[-1]:
            # simplifies below if t is in order
            tvals = tvals[::-1]
            fraccoords = fraccoords[::-1]

        # limit t to the range if given
        mint, maxt = self.getMinMaxT()
        if mint is not None and mint > tvals[0]:
            sel = tvals > mint
            minfrac = (mint - trange[0]) / (trange[1]-trange[0])
            fraccoords = N.hstack( (minfrac, fraccoords[sel]) )
            tvals = N.hstack( (mint, tvals[sel]) )
        if maxt is not None and maxt < tvals[-1]:
            sel = tvals < maxt
            maxfrac = (maxt - trange[0]) / (trange[1]-trange[0])
            fraccoords = N.hstack( (fraccoords[sel], maxfrac) )
            tvals = N.hstack( (tvals[sel], maxt) )

        try:
            ourgraphcoords = self.getFunction()(tvals)
        except Exception:
            return

        deltas = ourgraphcoords[1:] - ourgraphcoords[:-1]
        pos = N.all(deltas >= 0.)
        neg = N.all(deltas <= 0.)
        if (not pos and not neg) or (pos and neg):
            self.logError(_('Not a monotonic function'))
            return

        # Select only finite vals. We store _inv coords separately
        # as linear interpolation requires increasing values.
        f = N.isfinite(ourgraphcoords)
        self.graphcoords = self.graphcoords_inv = ourgraphcoords[f]
        self.pixcoords = self.pixcoords_inv = (
            fraccoords[f]*(self.coordParr2-self.coordParr1) + self.coordParr1 )

        # put output coordinates in correct order
        self._orderCoordinates()

    def updateAxisLocation(self, bounds, otherposition=None,
                           lowerupperposition=None):
        '''Calculate conversion from pixels to axis values.'''

        axis.Axis.updateAxisLocation(self, bounds, otherposition=otherposition,
                                     lowerupperposition=lowerupperposition)

        if ( self.boundschangeset == self.document.changeset and
             bounds == self.cachedbounds ):
            # don't recalculate unless document updated or bounds changes
            return

        self.cachedbounds = list(bounds)
        self.boundschangeset = self.document.changeset

        self.graphcoords = None

        # fractional coordinate grid to evaluate functions
        fraccoords = N.hstack((
                N.linspace(-10., -2.0, 5, endpoint=False),
                N.linspace(-2.0, -0.2, 25, endpoint=False),
                N.linspace(-0.2, +1.2, 250, endpoint=False),
                N.linspace(+1.2, +3.0, 25, endpoint=False),
                N.linspace(+3.0, +11., 5, endpoint=False)
                ))

        if self.isLinked():
            self._updateLinkedAxis(bounds, fraccoords)
        else:
            self._updateFreeAxis(bounds, fraccoords)

    def _linearInterpolWarning(self, vals, xcoords, ycoords):
        '''Linear interpolation, giving out of bounds warning.'''
        if N.any(vals < xcoords[0]) or N.any(vals > xcoords[-1]):
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
