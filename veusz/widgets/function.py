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

from __future__ import division
import numpy as N

from ..compat import czip, cstr
from .. import qtall as qt
from .. import document
from .. import setting
from .. import utils

from . import pickable
from .plotters import GenericPlotter

def _(text, disambiguation=None, context='Function'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class FunctionPlotter(GenericPlotter):
    """Function plotting class."""

    typename='function'
    allowusercreation=True
    description=_('Plot a function')

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        GenericPlotter.addSettings(s)

        s.add( setting.Int(
            'steps',
            50,
            minval = 3,
            descr = _('Number of steps to evaluate the function'
                      ' over'),
            usertext=_('Steps'), formatting=True), 0 )
        s.add( setting.Choice(
            'variable', ['x', 'y'], 'x',
            descr=_('Variable the function is a function of'),
            usertext=_('Variable')),
               0 )
        s.add( setting.Str(
            'function', 'x',
            descr=_('Function expression'),
            usertext=_('Function')), 0 )

        s.add(setting.FloatOrAuto(
            'min', 'Auto',
            descr=_('Minimum value at which to plot function'),
            usertext=_('Min')))

        s.add(setting.FloatOrAuto(
            'max', 'Auto',
            descr=_('Maximum value at which to plot function'),
            usertext=_('Max')))

        s.add( setting.Line(
            'Line',
            descr = _('Function line settings'),
            usertext = _('Plot line')),
               pixmap = 'settings_plotline' )
        s.Line.get('color').newDefault('auto')

        s.add( setting.PlotterFill(
            'FillBelow',
            descr = _('Fill below/left function'),
            usertext = _('Fill below')),
               pixmap = 'settings_plotfillbelow' )
        s.add( setting.PlotterFill(
            'FillAbove',
            descr = _('Fill mode above/right function'),
            usertext = _('Fill above')),
               pixmap = 'settings_plotfillabove' )

    @property
    def userdescription(self):
        """User-friendly description."""
        return "%(variable)s = %(function)s" % self.settings

    def logEvalError(self, ex):
        """Write error message to document log for exception ex."""
        self.document.log(
            "Error evaluating expression in function widget '%s': '%s'" % (
                self.name, cstr(ex)))

    def affectsAxisRange(self):
        s = self.settings
        if s.variable == 'x':
            return ((s.yAxis, 'both'),)
        else:
            return ((s.xAxis, 'both'),)

    def requiresAxisRange(self):
        s = self.settings
        if s.variable == 'x':
            return (('both', s.xAxis),)
        else:
            return (('both', s.yAxis),)

    def getRange(self, axis, depname, axrange):
        """Adjust the range of the axis depending on the values plotted."""
        s = self.settings

        # ignore empty function
        if s.function.strip() == '':
            return

        # ignore if function isn't sensible
        compiled = self.document.evaluate.compileCheckedExpression(s.function)
        if compiled is None:
            return

        # find axis to find variable range over
        varaxis = self.lookupAxis( {'x': s.xAxis, 'y': s.yAxis}[s.variable] )
        if not varaxis:
            return

        # get range of that axis
        varaxrange = list(varaxis.getPlottedRange())

        # trim to range
        if s.min != 'Auto':
            varaxrange[0] = max(s.min, varaxrange[0])
        if s.max != 'Auto':
            varaxrange[1] = min(s.max, varaxrange[1])

        if varaxrange[0] == varaxrange[1]:
            return

        # work out function in steps
        try:
            if varaxis.settings.log:
                # log spaced steps
                l1, l2 = N.log(varaxrange[1]), N.log(varaxrange[0])
                delta = (l2-l1)/20.
                points = N.exp(N.arange(l1, l2+delta, delta))
            else:
                # linear spaced steps
                delta = (varaxrange[1] - varaxrange[0])/20.
                points = N.arange(varaxrange[0], varaxrange[1]+delta, delta)
        except (ZeroDivisionError, ValueError) as e:
            # delta is zero
            return

        env = self.initEnviron()
        env[s.variable] = points
        try:
            vals = eval(compiled, env) + points*0.
        except:
            # something wrong in the evaluation
            return

        # get values which are finite: excluding nan and inf
        finitevals = vals[N.isfinite(vals)]

        if axis.settings.log:
            finitevals = finitevals[finitevals > 0]

        # update the automatic range
        if len(finitevals) > 0:
            axrange[0] = min(N.min(finitevals), axrange[0])
            axrange[1] = max(N.max(finitevals), axrange[1])

    def _plotLine(self, painter, xpts, ypts, bounds, clip):
        """ Plot the points in xpts, ypts."""
        x1, y1, x2, y2 = bounds

        maxdeltax = (x2-x1)*3/4
        maxdeltay = (y2-y1)*3/4

        # idea is to collect points until we go out of the bounds
        # or reach the end, then plot them
        pts = qt.QPolygonF()
        lastx = lasty = -65536
        for x, y in czip(xpts, ypts):

            # ignore point if it outside sensible bounds
            if x < -32767 or y < -32767 or x > 32767 or y > 32767:
                if len(pts) >= 2:
                    utils.plotClippedPolyline(painter, clip, pts)
                    pts.clear()
            else:
                # if the jump wasn't too large, add the point to the points
                if abs(x-lastx) < maxdeltax and abs(y-lasty) < maxdeltay:
                    pts.append( qt.QPointF(x, y) )
                else:
                    # draw what we have until now, and start a new line
                    if len(pts) >= 2:
                        utils.plotClippedPolyline(painter, clip, pts)
                    pts.clear()
                    pts.append( qt.QPointF(x, y) )

            lastx = x
            lasty = y

        # draw remaining points
        if len(pts) >= 2:
            utils.plotClippedPolyline(painter, clip, pts)

    def _fillRegion(self, painter, pxpts, pypts, bounds, belowleft, clip,
                    brush):
        """Fill the region above/below or left/right of the points.

        belowleft fills below if the variable is 'x', or left if 'y'
        otherwise it fills above/right."""

        # find starting and ending points for the filled region
        x1, y1, x2, y2 = bounds

        # trimming can lead to too few points
        if len(pxpts) < 2 or len(pypts) < 2:
            return

        pts = qt.QPolygonF()
        if self.settings.variable == 'x':
            if belowleft:
                pts.append(qt.QPointF(pxpts[0], y2))
                endpt = qt.QPointF(pxpts[-1], y2)
            else:
                pts.append(qt.QPointF(pxpts[0], y1))
                endpt = qt.QPointF(pxpts[-1], y1)
        else:
            if belowleft:
                pts.append(qt.QPointF(x1, pypts[0]))
                endpt = qt.QPointF(x1, pypts[-1])
            else:
                pts.append(qt.QPointF(x2, pypts[0]))
                endpt = qt.QPointF(x2, pypts[-1])

        # add the points between
        utils.addNumpyToPolygonF(pts, pxpts, pypts)

        # stick on the ending point
        pts.append(endpt)

        # draw the clipped polygon
        clipped = qt.QPolygonF()
        utils.polygonClip(pts, clip, clipped)
        path = qt.QPainterPath()
        path.addPolygon(clipped)
        utils.brushExtFillPath(painter, brush, path)

    def drawKeySymbol(self, number, painter, x, y, width, height):
        """Draw the plot symbol and/or line."""

        s = self.settings
        yp = y + height/2

        # draw line
        if not s.Line.hide:
            painter.setBrush( qt.QBrush() )
            painter.setPen( s.Line.makeQPen(painter) )
            painter.drawLine( qt.QPointF(x, yp), qt.QPointF(x+width, yp) )

    def initEnviron(self):
        """Set up function environment."""
        return self.document.evaluate.context.copy()

    def getIndependentPoints(self, axes, posn):
        """Calculate the real and screen points to plot for the independent axis"""

        s = self.settings

        if ( axes[0] is None or axes[1] is None or
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

        if ( axes[0] is None or axes[1] is None or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return None, None

        if axispts is None:
            return None, None

        compiled = self.document.evaluate.compileCheckedExpression(s.function)
        if not compiled:
            return None, None

        axis2 = axes[1] if s.variable == 'x' else axes[0]

        # evaluate function
        env = self.initEnviron()
        env[s.variable] = axispts
        try:
            results = eval(compiled, env) + N.zeros(axispts.shape)
            resultpts = axis2.dataToPlotterCoords(posn, results)
        except Exception as e:
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

    def dataDraw(self, painter, axes, posn, cliprect):
        """Draw the function."""

        s = self.settings

        # exit if hidden or function blank
        if s.function.strip() == '':
            return
        # get the points to plot by evaluating the function
        (xpts, ypts), (pxpts, pypts) = self.calcFunctionPoints(axes, posn)

        # draw the function line
        if ( pxpts is None or pypts is None or
             pxpts.ndim != 1 or pypts.ndim != 1 ):
            # not sure how to deal with errors here
            painter.setPen( setting.settingdb.color('error') )
            f = qt.QFont()
            f.setPointSize(20)
            painter.setFont(f)
            painter.drawText(
                cliprect,
                qt.Qt.AlignCenter,
                "Cannot evaluate '%s'" % s.function)
        else:
            if not s.FillBelow.hide:
                self._fillRegion(painter, pxpts, pypts, posn, True, cliprect,
                                 s.FillBelow)

            if not s.FillAbove.hide:
                self._fillRegion(painter, pxpts, pypts, posn, False, cliprect,
                                 s.FillAbove)

            if not s.Line.hide:
                painter.setBrush( qt.QBrush() )
                painter.setPen( s.Line.makeQPen(painter) )
                self._plotLine(painter, pxpts, pypts, posn, cliprect)

# allow the factory to instantiate an function plotter
document.thefactory.register( FunctionPlotter )
