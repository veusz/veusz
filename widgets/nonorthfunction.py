#    Copyright (C) 2010 Jeremy S. Sanders
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

'''Non orthogonal function plotting.'''

import numpy as N

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import pickable
from nonorthgraph import NonOrthGraph, FillBrush
from widget import Widget
from function import FunctionChecker

def _(text, disambiguation=None, context='NonOrthFunction'):
    """Translate text."""
    return unicode( 
        qt4.QCoreApplication.translate(context, text, disambiguation))

class NonOrthFunction(Widget):
    '''Widget for plotting a function on a non-orthogonal plot.'''

    typename = 'nonorthfunc'
    allowusercreation = True
    description = _('Plot a function on graphs with non-orthogonal axes')

    allowedparenttypes = [NonOrthGraph]

    def __init__(self, parent, name=None):
        '''Initialise plotter.'''
        Widget.__init__(self, parent, name=name)
        if type(self) == NonOrthFunction:
            self.readDefaults()

        self.checker = FunctionChecker()

    @classmethod
    def addSettings(klass, s):
        '''Settings for widget.'''
        Widget.addSettings(s)

        s.add( setting.Str('function', 'a',
                           descr=_('Function expression'),
                           usertext=_('Function')) )
        s.add( setting.Choice('variable', ['a', 'b'], 'a',
                              descr=_('Variable the function is a function of'),
                              usertext=_('Variable')) )
        s.add(setting.FloatOrAuto('min', 'Auto',
                                  descr=_('Minimum value at which to plot function'),
                                  usertext=_('Min')))
        
        s.add(setting.FloatOrAuto('max', 'Auto',
                                  descr=_('Maximum value at which to plot function'),
                                  usertext=_('Max')))


        s.add( setting.Line('PlotLine',
                            descr = _('Plot line settings'),
                            usertext = _('Plot line')),
               pixmap = 'settings_plotline' )
        s.add( FillBrush('Fill1',
                         descr = _('Fill settings (1)'),
                         usertext = _('Area fill 1')),
               pixmap = 'settings_plotfillbelow' )
        s.add( FillBrush('Fill2',
                         descr = _('Fill settings (2)'),
                         usertext = _('Area fill 2')),
               pixmap = 'settings_plotfillbelow' )

        s.add( setting.Int('steps', 50,
                           descr = _('Number of steps to evaluate the function'
                                     ' over'),
                           usertext=_('Steps'), formatting=True), 0 )

    def initEnviron(self):
        '''Set up function environment.'''
        return self.document.eval_context.copy()
       
    def logEvalError(self, ex):
        '''Write error message to document log for exception ex.'''
        self.document.log(
            "Error evaluating expression in function widget '%s': '%s'" % (
                self.name, unicode(ex)))

    def getFunctionPoints(self):
        '''Get points for plotting function.
        Return (apts, bpts)
        '''
        # get range of variable in expression
        s = self.settings
        crange = self.parent.coordRanges()[ {'a': 0, 'b': 1}[s.variable] ]
        if s.min != 'Auto':
            crange[0] = s.min
        if s.max != 'Auto':
            crange[1] = s.max

        steps = max(2, s.steps)
        # input values for function
        invals = ( N.arange(steps)*(1./(steps-1))*(crange[1]-crange[0]) +
                   crange[0] )

        # do evaluation
        env = self.initEnviron()
        env[s.variable] = invals
        try:
            vals = eval(self.checker.compiled, env) + invals*0.
        except Exception, e:
            self.logEvalError(e)
            vals = invals = N.array([])

        # return points
        if s.variable == 'a':
            return invals, vals
        else:
            return vals, invals

    def updateDataRanges(self, inrange):
        '''Update ranges of data given function.'''

    def _pickable(self):
        apts, bpts = self.getFunctionPoints()
        px, py = self.parent.graphToPlotCoords(apts, bpts)

        if self.settings.variable == 'a':
            labels = ('a', 'b(a)')
        else:
            labels = ('a(b)', 'b')

        return pickable.GenericPickable( self, labels, (apts, bpts), (px, py) )

    def pickPoint(self, x0, y0, bounds, distance='radial'):
        return self._pickable().pickPoint(x0, y0, bounds, distance)

    def pickIndex(self, oldindex, direction, bounds):
        return self._pickable().pickIndex(oldindex, direction, bounds)

    def draw(self, parentposn, phelper, outerbounds=None):
        '''Plot the function on a plotter.'''

        posn = self.computeBounds(parentposn, phelper)
        s = self.settings

        # exit if hidden
        if s.hide:
            return

        # ignore if function isn't sensible
        try:
            self.checker.check(s.function, s.variable)
        except RuntimeError, e:
            self.logEvalError(e)
            return

        apts, bpts = self.getFunctionPoints()
        px, py = self.parent.graphToPlotCoords(apts, bpts)

        x1, y1, x2, y2 = posn
        cliprect = qt4.QRectF( qt4.QPointF(x1, y1), qt4.QPointF(x2, y2) )
        painter = phelper.painter(self, posn)
        with painter:
            self.parent.setClip(painter, posn)

            # plot line
            painter.setBrush(qt4.QBrush())
            painter.setPen( s.PlotLine.makeQPenWHide(painter) )
            for x, y in utils.validLinePoints(px, py):
                if not s.Fill1.hide:
                    self.parent.drawFillPts(painter, s.Fill1, cliprect, x, y)
                if not s.Fill2.hide:
                    self.parent.drawFillPts(painter, s.Fill2, cliprect, x, y)
                if not s.PlotLine.hide:
                    p = qt4.QPolygonF()
                    utils.addNumpyToPolygonF(p, x, y)
                    painter.setBrush(qt4.QBrush())
                    painter.setPen( s.PlotLine.makeQPen(painter) )
                    utils.plotClippedPolyline(painter, cliprect, p)

document.thefactory.register( NonOrthFunction )
