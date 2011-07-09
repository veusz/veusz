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

"""For plotting numerical functions."""

import veusz.qtall as qt4
import itertools
import numpy as N

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import pickable
from plotters import GenericPlotter

class FunctionChecker(object):
    """Help check function is valid."""
    def __init__(self):
        self.cachedfunc = None
        self.cachedvar = None
        self.compiled = None

    def check(self, fn, var):
        """check function doesn't contain dangerous code.
        fn:  function
        var: function is a variable of this
        
        raises a RuntimeError(msg) if a problem
        """
        fn = fn.strip()
        if self.cachedfunc != fn or self.cachedvar != var:
            checked = utils.checkCode(fn)
            if checked is not None:
                try:
                    msg = checked[0][0]
                except Exception:
                    msg = ''
                raise RuntimeError(msg)

            self.cachedfunc = fn
            self.cachedvar = var

            try:
                # compile code
                self.compiled = compile(fn, '<string>', 'eval')
            except Exception, e:
                raise RuntimeError(e)

class FunctionPlotter(GenericPlotter):
    """Function plotting class."""

    typename='function'
    allowusercreation=True
    description='Plot a function'
    
    def __init__(self, parent, name=None):
        """Initialise plotter."""

        GenericPlotter.__init__(self, parent, name=name)

        if type(self) == FunctionPlotter:
            self.readDefaults()

        self.checker = FunctionChecker()

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        GenericPlotter.addSettings(s)

        s.add( setting.Int('steps',
                           50,
                           minval = 3,
                           descr = 'Number of steps to evaluate the function'
                           ' over', usertext='Steps', formatting=True), 0 )
        s.add( setting.Choice('variable', ['x', 'y'], 'x',
                              descr='Variable the function is a function of',
                              usertext='Variable'),
               0 )
        s.add( setting.Str('function', 'x',
                           descr='Function expression',
                           usertext='Function'), 0 )

        s.add(setting.FloatOrAuto('min', 'Auto',
                                  descr='Minimum value at which to plot function',
                                  usertext='Min'))
        
        s.add(setting.FloatOrAuto('max', 'Auto',
                                  descr='Maximum value at which to plot function',
                                  usertext='Max'))

        s.add( setting.Line('Line',
                            descr = 'Function line settings',
                            usertext = 'Plot line'),
               pixmap = 'settings_plotline' )

        s.add( setting.PlotterFill('FillBelow',
                                   descr = 'Fill below function',
                                   usertext = 'Fill below'),
               pixmap = 'settings_plotfillbelow' )
        
        s.add( setting.PlotterFill('FillAbove',
                                   descr = 'Fill above function',
                                   usertext = 'Fill above'),
               pixmap = 'settings_plotfillabove' )

    @property
    def userdescription(self):
        """User-friendly description."""
        return "%(variable)s = %(function)s" % self.settings

    def logEvalError(self, ex):
        """Write error message to document log for exception ex."""
        self.document.log(
            "Error evaluating expression in function widget '%s': '%s'" % (
                self.name, unicode(ex)))

    def providesAxesDependency(self):
        s = self.settings
        if s.variable == 'x':
            return ((s.yAxis, 'both'),)
        else:
            return ((s.xAxis, 'both'),)

    def requiresAxesDependency(self):
        s = self.settings
        if s.variable == 'x':
            return (('both', s.xAxis),)
        else:
            return (('both', s.yAxis),)

    def updateAxisRange(self, axis, depname, axrange):
        """Adjust the range of the axis depending on the values plotted."""
        s = self.settings

        # ignore empty function
        if s.function.strip() == '':
            return

        # ignore if function isn't sensible
        try:
            self.checker.check(s.function, s.variable)
        except RuntimeError, e:
            self.logEvalError(e)
            return

        # find axis to find variable range over
        axis = self.lookupAxis( {'x': s.xAxis, 'y': s.yAxis}[s.variable] )
        if not axis:
            return

        # get range of that axis
        varaxrange = list(axis.getPlottedRange())
        if varaxrange[0] == varaxrange[1]:
            return

        # trim to range
        if s.min != 'Auto':
            varaxrange[0] = max(s.min, varaxrange[0])
        if s.max != 'Auto':
            varaxrange[1] = min(s.max, varaxrange[1])

        # work out function in steps
        try:
            if axis.settings.log:
                # log spaced steps 
                l1, l2 = N.log(varaxrange[1]), N.log(varaxrange[0])
                delta = (l2-l1)/20.
                points = N.exp(N.arange(l1, l2+delta, delta))
            else:
                # linear spaced steps
                delta = (varaxrange[1] - varaxrange[0])/20.
                points = N.arange(varaxrange[0], varaxrange[1]+delta, delta)
        except ZeroDivisionError:
            # delta is zero
            return

        env = self.initEnviron()
        env[s.variable] = points
        try:
            vals = eval(self.checker.compiled, env) + points*0.
        except:
            # something wrong in the evaluation
            return

        # get values which are finite: excluding nan and inf
        finitevals = vals[N.isfinite(vals)]

        # update the automatic range
        axrange[0] = min(N.min(finitevals), axrange[0])
        axrange[1] = max(N.max(finitevals), axrange[1])

    def _plotLine(self, painter, xpts, ypts, bounds, clip):
        """ Plot the points in xpts, ypts."""
        x1, y1, x2, y2 = bounds

        maxdeltax = (x2-x1)*3/4
        maxdeltay = (y2-y1)*3/4

        # idea is to collect points until we go out of the bounds
        # or reach the end, then plot them
        pts = qt4.QPolygonF()
        lastx = lasty = -65536
        for x, y in itertools.izip(xpts, ypts):

            # ignore point if it outside sensible bounds
            if x < -32767 or y < -32767 or x > 32767 or y > 32767:
                if len(pts) >= 2:
                    utils.plotClippedPolyline(painter, clip, pts)
                    pts.clear()
            else:
                # if the jump wasn't too large, add the point to the points
                if abs(x-lastx) < maxdeltax and abs(y-lasty) < maxdeltay:
                    pts.append( qt4.QPointF(x, y) )
                else:
                    # draw what we have until now, and start a new line
                    if len(pts) >= 2:
                        utils.plotClippedPolyline(painter, clip, pts)
                    pts.clear()
                    pts.append( qt4.QPointF(x, y) )

            lastx = x
            lasty = y

        # draw remaining points
        if len(pts) >= 2:
            utils.plotClippedPolyline(painter, clip, pts)

    def _fillRegion(self, painter, pxpts, pypts, bounds, belowleft, clip):
        """Fill the region above/below or left/right of the points.

        belowleft fills below if the variable is 'x', or left if 'y'
        otherwise it fills above/right."""

        # find starting and ending points for the filled region
        x1, y1, x2, y2 = bounds
        s = self.settings
        
        pts = qt4.QPolygonF()
        if self.settings.variable == 'x':
            if belowleft:
                pts.append(qt4.QPointF(pxpts[0], y2))
                endpt = qt4.QPointF(pxpts[-1], y2)
            else:
                pts.append(qt4.QPointF(pxpts[0], y1))
                endpt = qt4.QPointF(pxpts[-1], y1)
        else:
            if belowleft:
                pts.append(qt4.QPointF(x1, pypts[0]))
                endpt = qt4.QPointF(x1, pypts[-1])
            else:
                pts.append(qt4.QPointF(x2, pypts[0]))
                endpt = qt4.QPointF(x2, pypts[-1])

        # add the points between
        utils.addNumpyToPolygonF(pts, pxpts, pypts)

        # stick on the ending point
        pts.append(endpt)

        # actually do the filling
        utils.plotClippedPolygon(painter, clip, pts)

    def drawKeySymbol(self, number, painter, phelper, x, y, width, height):
        """Draw the plot symbol and/or line."""

        s = self.settings
        yp = y + height/2

        # draw line
        if not s.Line.hide:
            painter.setBrush( qt4.QBrush() )
            painter.setPen( s.Line.makeQPen(phelper) )
            painter.drawLine( qt4.QPointF(x, yp), qt4.QPointF(x+width, yp) )

    def initEnviron(self):
        """Set up function environment."""
        return self.document.eval_context.copy()

    def getIndependentPoints(self, axes, posn):
        """Calculate the real and screen points to plot for the independent axis"""

        s = self.settings

        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return None, None

        # get axes function is plotted along and on and
        # plot coordinates along axis function plotted along
        if s.variable == 'x':
            axis1, axis2 = axes[0], axes[1]
            minval, maxval = posn[0], posn[2]
        else:
            axis1, axis2 = axes[1], axes[0]
            minval, maxval = posn[1], posn[3]

        # get equally spaced coordinates along axis in plotter coords
        plotpts = N.arange(s.steps) * ((maxval-minval) / (s.steps-1)) + minval
        # convert to axis coordinates
        axispts = axis1.plotterToDataCoords(posn, plotpts)

        # trim according to min and max. have to convert back to plotter too.
        if s.min != 'Auto':
            axispts = axispts[ axispts >= s.min ]
            plotpts = axis1.dataToPlotterCoords(posn, axispts)
        if s.max != 'Auto':
            axispts = axispts[ axispts <= s.max ]
            plotpts = axis1.dataToPlotterCoords(posn, axispts)

        return axispts, plotpts

    def calcDependentPoints(self, axispts, axes, posn):
        """Calculate the real and screen points to plot for the dependent axis"""

        s = self.settings

        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return None, None

        if axispts is None:
            return None, None

        try:
            self.checker.check(s.function, s.variable)
        except RuntimeError, e:
            self.logEvalError(e)
            return None, None

        axis2 = axes[1] if s.variable == 'x' else axes[0]

        # evaluate function
        env = self.initEnviron()
        env[s.variable] = axispts
        try:
            results = eval(self.checker.compiled, env)
            resultpts = axis2.dataToPlotterCoords(
                posn, results + N.zeros(axispts.shape))
        except Exception, e:
            self.logEvalError(e)
            results = None
            resultpts = None

        return results, resultpts

    def calcFunctionPoints(self, axes, posn):
        ipts, pipts = self.getIndependentPoints(axes, posn)
        dpts, pdpts = self.calcDependentPoints(ipts, axes, posn)
        
        if self.settings.variable == 'x':
            return (ipts, dpts), (pipts, pdpts)
        else:
            return (dpts, ipts), (pdpts, pipts)

    def _pickable(self, posn):
        s = self.settings

        axisnames = [s.xAxis, s.yAxis]
        axes = self.parent.getAxes(axisnames)

        if s.variable == 'x':
            axisnames[1] = axisnames[1] + '(' + axisnames[0] + ')'
        else:
            axisnames[0] = axisnames[0] + '(' + axisnames[1] + ')'

        (xpts, ypts), (pxpts, pypts) = self.calcFunctionPoints(axes, posn)

        return pickable.GenericPickable(
                    self, axisnames, (xpts, ypts), (pxpts, pypts) )

    def pickPoint(self, x0, y0, bounds, distance='radial'):
        return self._pickable(bounds).pickPoint(x0, y0, bounds, distance)

    def pickIndex(self, oldindex, direction, bounds):
        return self._pickable(bounds).pickIndex(oldindex, direction, bounds)

    def draw(self, parentposn, painthelper, outerbounds = None):
        """Draw the function."""

        posn = GenericPlotter.draw(self, parentposn, painthelper,
                                   outerbounds = outerbounds)
        x1, y1, x2, y2 = posn
        s = self.settings

        # exit if hidden or function blank
        if s.hide or s.function.strip() == '':
            return

        # get axes widgets
        axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

        # return if there's no proper axes
        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return

        # clip data within bounds of plotter
        cliprect = self.clipAxesBounds(axes, posn)
        painter = painthelper.painter(self, self.parent, posn, clip=cliprect)

        # get the points to plot by evaluating the function
        (xpts, ypts), (pxpts, pypts) = self.calcFunctionPoints(axes, posn)

        # draw the function line
        if pxpts is None or pypts is None:
            # not sure how to deal with errors here
            painter.setPen( setting.settingdb.color('error') )
            f = qt4.QFont()
            f.setPointSize(20)
            painter.setFont(f)
            painter.drawText( qt4.QRectF(x1, y1, x2-x1, y2-y1),
                              qt4.Qt.AlignCenter,
                              "Cannot evaluate '%s'" % s.function )
        else:
            if not s.FillBelow.hide:
                painter.setBrush( s.FillBelow.makeQBrush() )
                painter.setPen( qt4.QPen(qt4.Qt.NoPen) )
                self._fillRegion(painter, pxpts, pypts, posn, True, cliprect)

            if not s.FillAbove.hide:
                painter.setBrush( s.FillAbove.makeQBrush() )
                painter.setPen( qt4.QPen(qt4.Qt.NoPen) )
                self._fillRegion(painter, pxpts, pypts, posn, False, cliprect)

            if not s.Line.hide:
                painter.setBrush( qt4.QBrush() )
                painter.setPen( s.Line.makeQPen(painthelper) )
                self._plotLine(painter, pxpts, pypts, posn, cliprect)

# allow the factory to instantiate an function plotter
document.thefactory.register( FunctionPlotter )

