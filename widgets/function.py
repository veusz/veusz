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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
###############################################################################

# $Id$

"""For plotting numerical functions."""

import veusz.qtall as qt4
import itertools
import numpy as N

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

from plotters import GenericPlotter

class FunctionPlotter(GenericPlotter):
    """Function plotting class."""

    typename='function'
    allowusercreation=True
    description='Plot a function'
    
    def __init__(self, parent, name=None):
        """Initialise plotter with axes."""

        GenericPlotter.__init__(self, parent, name=name)

        s = self.settings
        s.add( setting.Int('steps', 50,
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
               pixmap = 'plotline' )

        s.add( setting.PlotterFill('FillBelow',
                                   descr = 'Fill below function',
                                   usertext = 'Fill below'),
               pixmap = 'plotfillbelow' )
        
        s.add( setting.PlotterFill('FillAbove',
                                   descr = 'Fill above function',
                                   usertext = 'Fill above'),
               pixmap = 'plotfillabove' )

        if type(self) == FunctionPlotter:
            self.readDefaults()

        self.cachedfunc = None
        self.cachedvar = None
        self.cachedcomp = None
        
    def _getUserDescription(self):
        """User-friendly description."""
        return "%(variable)s = %(function)s" % self.settings
    userdescription = property(_getUserDescription)

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

        # ignore if function isn't sensible
        if not self._checkCachedFunction():
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
        if axis.settings.log:
            # log spaced steps 
            l1, l2 = N.log(varaxrange[1]), N.log(varaxrange[0])
            delta = (l2-l1)/20.
            points = N.exp(N.arange(l1, l2+delta, delta))
        else:
            # linear spaced steps
            delta = (varaxrange[1] - varaxrange[0])/20.
            points = N.arange(varaxrange[0], varaxrange[1]+delta, delta)

        env = self.initEnviron()
        env[s.variable] = points
        try:
            vals = eval(self.cachedcomp, env) + points*0.
        except:
            # something wrong in the evaluation
            return

        # get values which are finite: excluding nan and inf
        finitevals = vals[N.isfinite(vals)]

        # update the automatic range
        axrange[0] = min(N.min(finitevals), axrange[0])
        axrange[1] = max(N.max(finitevals), axrange[1])

    def _plotLine(self, painter, xpts, ypts, bounds):
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
                    painter.drawPolyline(pts)
                    pts.clear()
            else:
                # if the jump wasn't too large, add the point to the points
                if abs(x-lastx) < maxdeltax and abs(y-lasty) < maxdeltay:
                    pts.append( qt4.QPointF(x, y) )
                else:
                    # draw what we have until now, and start a new line
                    if len(pts) >= 2:
                        painter.drawPolyline(pts)
                    pts.clear()
                    pts.append( qt4.QPointF(x, y) )

            lastx = x
            lasty = y

        # draw remaining points
        if len(pts) >= 2:
            painter.drawPolyline(pts)

    def _fillRegion(self, painter, pxpts, pypts, bounds, belowleft):
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

        # add the points between (clipped to the bounds*2 - helps edges)
        xw = abs(x2-x1)
        xclip = N.clip(pxpts, x1-xw-1, x2+xw+1)
        yw = abs(y2-y1)
        yclip = N.clip(pypts, y1-yw-1, y2+yw+1)
        for x, y in itertools.izip(xclip, yclip):
            pts.append( qt4.QPointF(x, y) )

        # stick on the ending point
        pts.append(endpt)

        # actually do the filling
        painter.drawPolygon(pts)

    def drawKeySymbol(self, painter, x, y, width, height):
        """Draw the plot symbol and/or line."""

        s = self.settings
        yp = y + height/2

        # draw line
        if not s.Line.hide:
            painter.setBrush( qt4.QBrush() )
            painter.setPen( s.Line.makeQPen(painter) )
            painter.drawLine( qt4.QPointF(x, yp), qt4.QPointF(x+width, yp) )

    def initEnviron(self):
        """Set up function environment."""
        return utils.veusz_eval_context.copy()
       
    def _checkCachedFunction(self):
        """check function doesn't contain dangerous code."""
        s = self.settings
        fn = s.function.strip()
        if self.cachedfunc != fn or self.cachedvar != s.variable:
            checked = utils.checkCode(fn)
            if checked is not None:
                return None, None
            self.cachedfunc = fn
            self.cachedvar = s.variable

            try:
                # compile code
                self.cachedcomp = compile(fn, '<string>', 'eval')
            except:
                # return nothing
                return False
        return True
     
    def _calcFunctionPoints(self, axes, posn):
        """Calculate the pixels to plot for the function
        returns (pxpts, pypts)."""

        s = self.settings
        x1, y1, x2, y2 = posn

        if not self._checkCachedFunction():
            return

        env = self.initEnviron()
        if s.variable == 'x':
            # x function
            if s.min != 'Auto' and s.min > axes[0].getPlottedRange()[0]:
                x_min = N.array([s.min])
                x1 = axes[0].graphToPlotterCoords(posn, x_min)[0]
            if s.max != 'Auto' and s.max < axes[0].getPlottedRange()[1]:
                x_max = N.array([s.max])
                x2 = axes[0].graphToPlotterCoords(posn, x_max)[0]
                
            delta = (x2 - x1) / float(s.steps)
            pxpts = N.arange(x1, x2+delta, delta)
            x = axes[0].plotterToGraphCoords(posn, pxpts)
            env['x'] = x
            try:
                y = eval(self.cachedcomp, env)
            except:
                pypts = None
            else:
                pypts = axes[1].graphToPlotterCoords(posn, y+x*0.)

        else:
            # y function
            if not(s.min == 'Auto') and s.min > axes[1].getPlottedRange()[0]:
                y_min = N.array([s.min])
                y2 = axes[1].graphToPlotterCoords(posn, y_min)[0]
            if not(s.max == 'Auto') and s.max < axes[1].getPlottedRange()[1]:
                y_max = N.array([s.max])
                y1 = axes[1].graphToPlotterCoords(posn, y_max)[0]
            
            delta = (y2 - y1) / float(s.steps)
            pypts = N.arange(y1, y2+delta, delta)
            y = axes[1].plotterToGraphCoords(posn, pypts)
            env['y'] = y
            try:
                x = eval(self.cachedcomp, env)
            except:
                pxpts = None
            else:
                pxpts = axes[0].graphToPlotterCoords(posn, x+y*0.)

        return pxpts, pypts

    def draw(self, parentposn, painter, outerbounds = None):
        """Draw the function."""

        posn = GenericPlotter.draw(self, parentposn, painter,
                                   outerbounds = outerbounds)
        x1, y1, x2, y2 = posn
        s = self.settings

        # exit if hidden
        if s.hide:
            return

        # get axes widgets
        axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

        # return if there's no proper axes
        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return

        # clip data within bounds of plotter
        painter.beginPaintingWidget(self, posn)
        painter.save()
        self.clipAxesBounds(painter, axes, posn)

        # get the points to plot by evaluating the function
        pxpts, pypts = self._calcFunctionPoints(axes, posn)

        # draw the function line
        if pxpts is None or pypts is None:
            # not sure how to deal with errors here
            painter.setPen( qt4.QColor('red') )
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
                self._fillRegion(painter, pxpts, pypts, posn, True)

            if not s.FillAbove.hide:
                painter.setBrush( s.FillAbove.makeQBrush() )
                painter.setPen( qt4.QPen(qt4.Qt.NoPen) )
                self._fillRegion(painter, pxpts, pypts, posn, False)

            if not s.Line.hide:
                painter.setBrush( qt4.QBrush() )
                painter.setPen( s.Line.makeQPen(painter) )
                self._plotLine(painter, pxpts, pypts, posn)

        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate an function plotter
document.thefactory.register( FunctionPlotter )

