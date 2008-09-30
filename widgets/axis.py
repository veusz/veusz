#    Copyright (C) 2003-2007 Jeremy S. Sanders
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
##############################################################################

# $Id$

"""Widget to plot axes, and to handle conversion of coordinates to plot
positions."""

import veusz.qtall as qt4
import numpy as N

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import widget
import axisticks
import graph
import containers
import itertools

class Axis(widget.Widget):
    """Manages and draws an axis."""

    typename = 'axis'
    allowedparenttypes = [graph.Graph, containers.Grid]
    allowusercreation = True
    description = 'Axis to a plot or shared in a grid'
    isaxis = True

    def __init__(self, parent, name=None):
        """Initialise axis."""

        widget.Widget.__init__(self, parent, name=name)
        s = self.settings
        s.add( setting.Str('label', '',
                           descr='Axis label text',
                           usertext='Label') )
        s.add( setting.FloatOrAuto('min', 'Auto',
                                   descr='Minimum value of axis',
                                   usertext='Min') )
        s.add( setting.FloatOrAuto('max', 'Auto',
                                   descr='Maximum value of axis',
                                   usertext='Max') )
        s.add( setting.Bool('log', False,
                            descr = 'Whether axis is logarithmic',
                            usertext='Log') )
        s.add( setting.Bool('autoExtend', True,
                            descr = 'Extend axis to nearest major tick',
                            usertext='Auto extend') )
        s.add( setting.Bool('autoExtendZero', True,
                            descr = 'Extend axis to zero if close',
                            usertext='Zero extend') )
        s.add( setting.Bool('autoMirror', True,
                            descr = 'Place axis on opposite side of graph '
                            'if none',
                            usertext='Auto mirror') )
        s.add( setting.Bool('reflect', False,
                            descr = 'Place axis text and ticks on other side'
                            ' of axis',
                            usertext='Reflect') )
        s.add( setting.WidgetPath('match', '',
                                  descr =
                                  'Match the scale of this axis to the '
                                  'axis specified',
                                  usertext='Match',
                                  allowedwidgets = [Axis] ))

        s.add( setting.Choice('direction',
                              ['horizontal', 'vertical'],
                              'horizontal',
                              descr = 'Direction of axis',
                              usertext='Direction') )
        s.add( setting.Float('lowerPosition', 0.,
                             descr='Fractional position of lower end of '
                             'axis on graph',
                             usertext='Min position') )
        s.add( setting.Float('upperPosition', 1.,
                             descr='Fractional position of upper end of '
                             'axis on graph',
                             usertext='Max position') )
        s.add( setting.Float('otherPosition', 0.,
                             descr='Fractional position of axis '
                             'in its perpendicular direction',
                             usertext='Axis position') )

        s.add( setting.Line('Line',
                            descr = 'Axis line settings',
                            usertext = 'Axis line'),
               pixmap = 'axisline' )
        s.add( setting.AxisLabel('Label',
                                 descr = 'Axis label settings',
                                 usertext = 'Axis label'),
               pixmap = 'axislabel' )
        s.add( setting.TickLabel('TickLabels',
                                 descr = 'Tick label settings',
                                 usertext = 'Tick labels'),
               pixmap = 'axisticklabels' )
        s.add( setting.MajorTick('MajorTicks',
                                 descr = 'Major tick line settings',
                                 usertext = 'Major ticks'),
               pixmap = 'axismajorticks' )
        s.add( setting.MinorTick('MinorTicks',
                                 descr = 'Minor tick line settings',
                                 usertext = 'Minor ticks'),
               pixmap = 'axisminorticks' )
        s.add( setting.GridLine('GridLines',
                                descr = 'Grid line settings',
                                usertext = 'Grid lines'),
               pixmap = 'axisgridlines' )

        if type(self) == Axis:
            self.readDefaults()

        if self.name == 'y':
            s.direction = 'vertical'
        elif self.name == 'x':
            s.direction = 'horizontal'

        self.minorticks = None
        self.majorticks = None

        # automatic range 
        self.setAutoRange(None)

        # document updates change set variable when things need recalculating
        self.docchangeset = -1

    def _getUserDescription(self):
        """User friendly description."""
        s = self.settings
        return "range %s to %s%s" % ( str(s.min), str(s.max),
                                      ['',' (log)'][s.log])
    userdescription = property(_getUserDescription)

    def setAutoRange(self, autorange):
        """Set the automatic range of this axis (called from page helper)."""

        if autorange:
            self.autorange = ar = list(autorange)
            if self.settings.log:
                ar[0] = max(1e-99, ar[0])
        else:
            if self.settings.log:
                self.autorange = [1e-2, 1.]
            else:
                self.autorange = [0., 1.]
                
    def _computePlottedRange(self):
        """Convert the range requested into a plotted range."""

        s = self.settings
        self.plottedrange = [s.min, s.max]

        # match the scale of this axis to another
        matched = False
        if s.match != '':
            # locate widget we're matching
            # this is ensured to be an Axis
            widget = s.get('match').getWidget()

            # this looks valid + sanity checks
            if (widget is not None and widget != self and
                widget.settings.match == ''):
                # update if out of date
                if widget.docchangeset != self.document.changeset:
                    widget._computePlottedRange()
                # copy the range
                self.plottedrange = list(widget.plottedrange)
                matched = True

        # automatic lookup of minimum
        if s.min == 'Auto' and not matched:
            self.plottedrange[0] = self.autorange[0]

        if s.max == 'Auto' and not matched:
            self.plottedrange[1] = self.autorange[1]

        # yuck, but sometimes it's true
        # tweak range to make sure things don't blow up further down the
        # line
        if self.plottedrange[0] == self.plottedrange[1]:
               self.plottedrange[1] = ( self.plottedrange[0] +
                                        max(1., self.plottedrange[0]*0.1) )

        # handle axis values round the wrong way
        invertaxis = self.plottedrange[0] > self.plottedrange[1]
        if invertaxis:
            self.plottedrange.reverse()

        # make sure log axes don't blow up
        if s.log:
            if self.plottedrange[0] <= 0.:
                self.plottedrange[0] = 1e-99
            if self.plottedrange[1] <= 0.:
                self.plottedrange[1] = 1e-99

        # work out tick values and expand axes if necessary
        as = axisticks.AxisTicks( self.plottedrange[0], self.plottedrange[1],
                                  s.MajorTicks.number, s.MinorTicks.number,
                                  extendbounds = s.autoExtend,
                                  extendzero = s.autoExtendZero,
                                  logaxis = s.log )

        (self.plottedrange[0],self.plottedrange[1],
         self.majortickscalc, self.minortickscalc) =  as.getTicks()

        # override values if requested
        if len(s.MajorTicks.manualTicks) > 0:
            ticks = []
            for i in s.MajorTicks.manualTicks:
                if i >= self.plottedrange[0] and i <= self.plottedrange[1]:
                    ticks.append(i)
            self.majortickscalc = N.array(ticks)

        # invert bounds if axis was inverted
        if invertaxis:
            self.plottedrange.reverse()

        if self.majorticks is not None:
            self.majortickscalc = N.array(self.majorticks)

        if self.minorticks is not None:
            self.minortickscalc = N.array(self.minorticks)

        self.docchangeset = self.document.changeset
        
    def getPlottedRange(self):
        """Return the range plotted by the axes."""

        if self.docchangeset != self.document.changeset:
            self._computePlottedRange()
        return (self.plottedrange[0], self.plottedrange[1])

    def _updatePlotRange(self, bounds):
        """Calculate coordinates on plotter of axis."""

        s = self.settings
        x1, y1, x2, y2 = bounds
        dx = x2 - x1
        dy = y2 - y1
        p1, p2, pp = s.lowerPosition, s.upperPosition, s.otherPosition

        if s.direction == 'horizontal': # horizontal
            self.coordParr1 = x1 + dx*p1
            self.coordParr2 = x1 + dx*p2

            # other axis coordinates
            self.coordPerp  = y2 - dy*pp
            self.coordPerp1 = y2 - dy*p1
            self.coordPerp2 = y2 - dy*p2

        else: # vertical
            self.coordParr1 = y2 - dy*p1
            self.coordParr2 = y2 - dy*p2

            # other axis coordinates
            self.coordPerp  = x1 + dx*pp
            self.coordPerp1 = x1 + dx*p1
            self.coordPerp2 = x1 + dx*p2
     
    def graphToPlotterCoords(self, bounds, vals):
        """Convert graph coordinates to plotter coordinates on this axis.

        bounds specifies the plot bounds
        vals is numpy of coordinates
        Returns positions as a numpy
        """

        # if the doc was modified, recompute the range
        if self.docchangeset != self.document.changeset:
            self._computePlottedRange()

        self._updatePlotRange(bounds)

        return self._graphToPlotter(vals)

    def _graphToPlotter(self, vals):
        """Convert the coordinates assuming the machinery is in place."""
        
        # work out fractional posistions, then convert to pixels
        if self.settings.log:
            fracposns = self.logConvertToPlotter( vals )
        else:
            fracposns = self.linearConvertToPlotter( vals )

        return self.coordParr1 + fracposns*(self.coordParr2-self.coordParr1)
    
    def plotterToGraphCoords(self, bounds, vals):
        """Convert plotter coordinates on this axis to graph coordinates.
        
        bounds specifies the plot bounds
        vals is a numpy of coordinates
        returns a numpy of floats
        """

        # if the doc was modified, recompute the range
        if self.docchangeset != self.document.changeset:
            self._computePlottedRange()

        self._updatePlotRange( bounds )

        # work out fractional positions of the plotter coords
        frac = ( ( vals.astype('float64') - self.coordParr1 ) /
                 ( self.coordParr2 - self.coordParr1 ) )

        # scaling...
        if self.settings.log:
            return self.logConvertFromPlotter( frac )
        else:
            return self.linearConvertFromPlotter( frac )
        
    def linearConvertToPlotter(self, v):
        """Convert graph coordinates to fractional plotter units for linear scale.
        """
        return ( ( v - self.plottedrange[0] ) /
                 ( self.plottedrange[1]-self.plottedrange[0] ) )
    
    def linearConvertFromPlotter(self, v):
        """Convert from (fractional) plotter coords to graph coords.
        """
        return ( self.plottedrange[0] + v *
                 (self.plottedrange[1]-self.plottedrange[0] ) )
    
    def logConvertToPlotter(self, v):
        """Convert graph coordinates to fractional plotter units for log10 scale.
        """

        log1 = N.log(self.plottedrange[0])
        log2 = N.log(self.plottedrange[1])
        return ( N.log( N.clip(v, 1e-99, 1e99) ) - log1 )/( log2 - log1 )
    
    def logConvertFromPlotter(self, v):
        """Convert from fraction plotter coords to graph coords with log scale.
        """
        return ( self.plottedrange[0] *
                 ( self.plottedrange[1]/self.plottedrange[0] )**v )
    
    def againstWhichEdge(self):
        """Returns edge this axis is against, if any.

        Returns 0-3,None for (left, top, right, bottom, None)
        """

        s = self.settings
        op = abs(s.otherPosition)
        if op > 1e-3 and op < 0.999:
            return None
        else:
            if s.direction == 'vertical':
                if op <= 1e-3:
                    return 0
                else:
                    return 2
            else:
                if op <= 1e-3:
                    return 3
                else:
                    return 1
    
    def swapline(self, painter, a1, b1, a2, b2):
        """ Draw line, but swap x & y coordinates if vertical axis."""
        if self.settings.direction == 'horizontal':
            painter.drawLine(qt4.QPointF(a1, b1), qt4.QPointF(a2, b2))
        else:
            painter.drawLine(qt4.QPointF(b1, a1), qt4.QPointF(b2, a2))

    def _drawGridLines(self, painter, coordticks):
        """Draw grid lines on the plot."""
        
        painter.setPen( self.settings.get('GridLines').makeQPen(painter) )
        for t in coordticks:
            self.swapline( painter,
                           t, self.coordPerp1,
                           t, self.coordPerp2 )

    def _drawAxisLine(self, painter):
        """Draw the line of the axis."""

        painter.setPen( self.settings.get('Line').makeQPen(painter) )
        self.swapline( painter,
                       self.coordParr1, self.coordPerp,
                       self.coordParr2, self.coordPerp )        

    def _reflected(self):
        """Is the axis reflected?"""

        s = self.settings
        if s.otherPosition > 0.5:
            return not s.reflect
        else:
            return s.reflect

    def _drawMinorTicks(self, painter):
        """Draw minor ticks on plot."""

        s = self.settings
        mt = s.get('MinorTicks')
        painter.setPen( mt.makeQPen(painter) )
        delta = mt.getLength(painter)
        if len(self.minortickscalc):
            minorticks = self._graphToPlotter(self.minortickscalc)
        else:
            minorticks = []

        if s.direction == 'vertical':
            delta *= -1
        if self._reflected():
            delta *= -1
        for t in minorticks:
            self.swapline( painter,
                           t, self.coordPerp,
                           t, self.coordPerp - delta )

    def _drawMajorTicks(self, painter, tickcoords):
        """Draw major ticks on the plot."""

        s = self.settings
        painter.setPen( s.get('MajorTicks').makeQPen(painter) )
        startdelta = s.get('MajorTicks').getLength(painter)
        delta = startdelta

        if s.direction == 'vertical':
            delta *= -1
        if self._reflected():
            delta *= -1
        for t in tickcoords:
            self.swapline( painter,
                           t, self.coordPerp,
                           t, self.coordPerp - delta )

        # account for ticks if they are in the direction of the label
        if startdelta < 0:
            self._delta_axis += abs(delta)

    def _drawTickLabels(self, painter, coordticks, sign, texttorender):
        """Draw tick labels on the plot.

        texttorender is a list which contains text for the axis to render
        after checking for collisions
        """

        s = self.settings
        vertical = s.direction == 'vertical'
        font = s.get('TickLabels').makeQFont(painter)
        painter.setFont(font)
        tl_spacing = ( painter.fontMetrics().leading() +
                       painter.fontMetrics().descent() )

        # work out font alignment
        if s.TickLabels.rotate:
            if self._reflected():
                angle = 90
            else:
                angle = 270
        else:
            angle = 0
        
        if vertical:
            # limit tick labels to be directly below/besides axis
            bounds = { 'miny': min(self.coordParr1, self.coordParr2),
                       'maxy': max(self.coordParr1, self.coordParr2) }
            ax = 1
            ay = 0
        else:
            bounds = { 'minx': min(self.coordParr1, self.coordParr2),
                       'maxx': max(self.coordParr1, self.coordParr2) }
            ax = 0
            ay = 1

        if self._reflected():
            ax = -ax
            ay = -ay

        # plot numbers
        format = s.TickLabels.format
        maxdim = 0

        b = self.coordPerp + sign*(self._delta_axis+tl_spacing)
        tl = s.get('TickLabels')
        scale = tl.scale
        pen = tl.makeQPen()
        for a, num in itertools.izip(coordticks, self.majortickscalc):

            # x and y round other way if vertical
            if vertical:
                x, y = b, a
            else:
                x, y = a, b

            num = utils.formatNumber(num*scale, format)
            r = utils.Renderer(painter, font, x, y, num, alignhorz=ax,
                               alignvert=ay, angle=angle)
            r.ensureInBox(extraspace=True, **bounds)
            bnd = r.getBounds()
            texttorender.append( (r, pen) )

            if vertical:
                maxdim = max(maxdim, bnd[2] - bnd[0])
            else:
                maxdim = max(maxdim, bnd[3] - bnd[1])

        # keep track of where we are
        self._delta_axis += 2*tl_spacing + maxdim

    def _drawAxisLabel(self, painter, sign, outerbounds,
                       texttorender):
        """Draw an axis label on the plot.

        texttorender is a list which contains text for the axis to render
        after checking for collisions
        """

        s = self.settings
        sl = s.Label
        font = s.get('Label').makeQFont(painter)
        painter.setFont(font)
        al_spacing = ( painter.fontMetrics().leading() +
                       painter.fontMetrics().descent() )

        text = s.label
        # avoid adding blank text to plot
        if not text:
            return

        horz = s.direction == 'horizontal'
        if not horz:
            ax = 1
            ay = 0
        else:
            ax = 0
            ay = 1

        reflected = self._reflected()
        if reflected:
            ax = -ax
            ay = -ay

        # angle of text
        if ( (horz and not sl.rotate) or
             (not horz and sl.rotate) ):
            angle = 0
        else:
            if reflected:
                angle = 90
            else:
                angle = 270

        x = ( self.coordParr1 + self.coordParr2 ) / 2
        y = self.coordPerp + sign*(self._delta_axis+al_spacing)
        if not horz:
            x, y = y, x

        # make axis label flush with edge of plot if
        # it's appropriate
        if outerbounds is not None and sl.atEdge:
            if abs(s.otherPosition) < 1e-4 and not reflected:
                if horz:
                    y = outerbounds[3]
                    ay = -ay
                else:
                    x = outerbounds[0]
                    ax = -ax
            elif abs(s.otherPosition-1.) < 1e-4 and reflected:
                if horz:
                    y = outerbounds[1]
                    ay = -ay
                else:
                    x = outerbounds[2]
                    ax = -ax

        r = utils.Renderer(painter, font, x, y, text,
                           ax, ay, angle,
                           usefullheight = True)

        # make sure text is in plot rectangle
        if outerbounds is not None:
            r.ensureInBox( minx=outerbounds[0], maxx=outerbounds[2],
                           miny=outerbounds[1], maxy=outerbounds[3] )

        texttorender.insert(0, (r, s.get('Label').makeQPen()) )

    def _autoMirrorDraw(self, posn, painter, coordticks):
        """Mirror axis to opposite side of graph if there isn't
        an axis there already."""

        # This is a nasty hack: must think of a better way to do this
        s = self.settings
        countaxis = 0
        for c in self.parent.children:
            try:
                # don't allow descendents of axis to look like an axis
                # to this function (e.g. colorbar)
                if c.typename == 'axis' and s.direction == c.settings.direction:
                    countaxis += 1
            except AttributeError:
                # if it's not an axis we get here
                pass

        # another axis in the same direction, so we don't mirror it
        if countaxis > 1:
            return

        # swap axis to other side
        other = s.otherPosition
        if other < 0.5:
            next = 1.
        else:
            next = 0.

        # temporarily change position of axis
        # ** FIXME: this is a horrible kludge **
        s.get('otherPosition').setSilent(next)

        self._updatePlotRange(posn)
        if not s.Line.hide:
            self._drawAxisLine(painter)
        if not s.MinorTicks.hide:
            self._drawMinorTicks(painter)
        if not s.MajorTicks.hide:
            self._drawMajorTicks(painter, coordticks)

        # put axis back
        s.get('otherPosition').setSilent(other)

    def chooseName(self):
        """Axis are called x and y."""

        checks = {'x': False, 'y': False}

        # avoid choosing name which already exists
        for i in self.parent.children:
            if i.name in checks:
                checks[i.name] = True

        if not checks['x']:
            return 'x'
        if not checks['y']:
            return 'y'

        return widget.Widget.chooseName(self)

    def draw(self, parentposn, painter, suppresstext=False, outerbounds=None):
        """Plot the axis on the painter.

        if suppresstext is True, then we don't number or label the axis
        """

        s = self.settings

        # recompute if document modified
        if self.docchangeset != self.document.changeset:
            self._computePlottedRange()

        posn = widget.Widget.draw(self, parentposn, painter, outerbounds)
        self._updatePlotRange(posn)

        # get tick vals
        coordticks = self._graphToPlotter(self.majortickscalc)

        # exit if axis is hidden
        if s.hide:
            return

        # save the state of the painter for later
        painter.beginPaintingWidget(self, posn)
        painter.save()

        texttorender = []

        # multiplication factor if reflection on the axis is requested
        sign = 1
        if s.direction == 'vertical':
            sign *= -1
        if self._reflected():
            sign *= -1

        # plot gridlines
        if not s.GridLines.hide:
            self._drawGridLines(painter, coordticks)

        # plot the line along the axis
        if not s.Line.hide:
            self._drawAxisLine(painter)

        # plot minor ticks
        if not s.MinorTicks.hide:
            self._drawMinorTicks(painter)

        # keep track of distance from axis
        self._delta_axis = 0

        # plot major ticks
        if not s.MajorTicks.hide:
            self._drawMajorTicks(painter, coordticks)

        # plot tick labels
        if not s.TickLabels.hide and not suppresstext:
            self._drawTickLabels(painter, coordticks, sign,
                                 texttorender)

        # draw an axis label
        if not s.Label.hide and not suppresstext:
            self._drawAxisLabel(painter, sign, outerbounds,
                                texttorender)

        # mirror axis at other side of plot
        if s.autoMirror:
            self._autoMirrorDraw(posn, painter, coordticks)

        # all the text is drawn at the end so that
        # we can check it doesn't overlap
        drawntext = qt4.QPainterPath()
        for r, pen in texttorender:
            bounds = r.getBounds()
            rect = qt4.QRectF(bounds[0], bounds[1], bounds[2]-bounds[0]+1,
                              bounds[3]-bounds[1]+1)

            if not drawntext.intersects(rect):
                painter.setPen(pen)
                box = r.render()
                drawntext.addRect(rect)

        # restore the state of the painter
        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate an axis
document.thefactory.register( Axis )
