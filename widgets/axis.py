#    Copyright (C) 2003 Jeremy S. Sanders
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

"""Widget to plot axes, and to handle conversion of coordinates to plot
positions."""

from itertools import izip
import numpy as N

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import widget
import axisticks
import graph
import grid
import controlgraph

###############################################################################

class MajorTick(setting.Line):
    '''Major tick settings.'''

    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)
        self.add( setting.DistancePt( 'length',
                                      '6pt',
                                      descr = 'Length of major ticks',
                                      usertext='Length') )
        self.add( setting.Int( 'number',
                               6,
                               descr = 'Number of major ticks to aim for',
                               usertext='Number') )
        self.add( setting.FloatList('manualTicks',
                                    [],
                                    descr = 'List of tick values'
                                    ' overriding defaults',
                                    usertext='Manual ticks') )

    def getLength(self, painter):
        '''Return tick length in painter coordinates'''
        
        return self.get('length').convert(painter)
    
class MinorTick(setting.Line):
    '''Minor tick settings.'''

    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)
        self.add( setting.DistancePt( 'length',
                                      '3pt',
                                      descr = 'Length of minor ticks',
                                      usertext='Length') )
        self.add( setting.Int( 'number',
                               20,
                               descr = 'Number of minor ticks to aim for',
                               usertext='Number') )

    def getLength(self, painter):
        '''Return tick length in painter coordinates'''
        
        return self.get('length').convert(painter)
    
class GridLine(setting.Line):
    '''Grid line settings.'''

    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)

        self.get('color').newDefault( 'grey' )
        self.get('hide').newDefault( True )
        self.get('style').newDefault( 'dotted' )

class MinorGridLine(setting.Line):
    '''Minor tick grid line settings.'''

    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)

        self.get('color').newDefault( 'lightgrey' )
        self.get('hide').newDefault( True )
        self.get('style').newDefault( 'dotted' )

class AxisLabel(setting.Text):
    """For axis labels."""

    def __init__(self, name, **args):
        setting.Text.__init__(self, name, **args)
        self.add( setting.Bool( 'atEdge', False,
                                descr = 'Place axis label close to edge'
                                ' of graph',
                                usertext='At edge') )
        self.add( setting.Bool( 'rotate', False,
                                descr = 'Rotate the label by 90 degrees',
                                usertext='Rotate') )
        self.add( setting.DistancePt( 'offset',
                                      '0pt',
                                      descr = 'Additional offset of axis label'
                                      ' from axis tick labels',
                                      usertext='Label offset') )

class TickLabel(setting.Text):
    """For tick labels on axes."""

    formatchoices = ('Auto', '%Vg', '%Ve', '%VE',
                     '%g', '%e', '%.2f')
    descriptions = ('Automatic',
                    'General numerical format',
                    'Scientific notation',
                    'Engineering suffix notation',
                    'C-style general format',
                    'C-style scientific notation',
                    '2 decimal places always shown')

    def __init__(self, name, **args):
        setting.Text.__init__(self, name, **args)
        self.add( setting.Bool( 'rotate', False,
                                descr = 'Rotate the label by 90 degrees',
                                usertext='Rotate') )
        self.add( setting.ChoiceOrMore( 'format',
                                        TickLabel.formatchoices,
                                        'Auto',
                                        descr = 'Format of the tick labels',
                                        descriptions=TickLabel.descriptions,
                                        usertext='Format') )

        self.add( setting.Float('scale', 1.,
                                descr='A scale factor to apply to the values '
                                'of the tick labels',
                                usertext='Scale') )

        self.add( setting.DistancePt( 'offset',
                                      '0pt',
                                      descr = 'Additional offset of axis tick '
                                      'labels from axis',
                                      usertext='Tick offset') )

###############################################################################

class Axis(widget.Widget):
    """Manages and draws an axis."""

    typename = 'axis'
    allowedparenttypes = [graph.Graph, grid.Grid]
    allowusercreation = True
    description = 'Axis to a plot or shared in a grid'
    isaxis = True

    def __init__(self, parent, name=None):
        """Initialise axis."""

        widget.Widget.__init__(self, parent, name=name)
        s = self.settings

        if type(self) == Axis:
            self.readDefaults()

        if self.name == 'y' and s.direction != 'vertical':
            s.direction = 'vertical'
        elif self.name == 'x' and s.direction != 'horizontal':
            s.direction = 'horizontal'

        self.minorticks = None
        self.majorticks = None

        # automatic range 
        self.setAutoRange(None)

        # document updates change set variable when things need recalculating
        self.docchangeset = -1

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)
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
        s.add( setting.Choice('mode',
                              ('numeric', 'datetime', 'labels'), 
                              'numeric', 
                              descr = 'Type of ticks to show on on axis', 
                              usertext='Mode') )
                            
        s.add( setting.Bool('autoExtend', True,
                            descr = 'Extend axis to nearest major tick',
                            usertext='Auto extend',
                            formatting=True ) )
        s.add( setting.Bool('autoExtendZero', True,
                            descr = 'Extend axis to zero if close',
                            usertext='Zero extend',
                            formatting=True) )
        s.add( setting.Bool('autoMirror', True,
                            descr = 'Place axis on opposite side of graph '
                            'if none',
                            usertext='Auto mirror',
                            formatting=True) )
        s.add( setting.Bool('reflect', False,
                            descr = 'Place axis text and ticks on other side'
                            ' of axis',
                            usertext='Reflect',
                            formatting=True) )
        s.add( setting.Bool('outerticks', False,
                            descr = 'Place ticks on outside of graph',
                            usertext='Outer ticks',
                            formatting=True) )

        s.add( setting.Float('datascale', 1.,
                             descr='Scale data plotted by this factor',
                             usertext='Scale') )

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

        s.add( setting.WidgetPath('match', '',
                                  descr =
                                  'Match the scale of this axis to the '
                                  'axis specified',
                                  usertext='Match',
                                  allowedwidgets = [Axis] ))

        s.add( setting.Line('Line',
                            descr = 'Axis line settings',
                            usertext = 'Axis line'),
               pixmap='settings_axisline' )
        s.add( AxisLabel('Label',
                         descr = 'Axis label settings',
                         usertext = 'Axis label'),
               pixmap='settings_axislabel' )
        s.add( TickLabel('TickLabels',
                         descr = 'Tick label settings',
                         usertext = 'Tick labels'),
               pixmap='settings_axisticklabels' )
        s.add( MajorTick('MajorTicks',
                         descr = 'Major tick line settings',
                         usertext = 'Major ticks'),
               pixmap='settings_axismajorticks' )
        s.add( MinorTick('MinorTicks',
                         descr = 'Minor tick line settings',
                         usertext = 'Minor ticks'),
               pixmap='settings_axisminorticks' )
        s.add( GridLine('GridLines',
                        descr = 'Grid line settings',
                        usertext = 'Grid lines'),
               pixmap='settings_axisgridlines' )
        s.add( MinorGridLine('MinorGridLines',
                             descr = 'Minor grid line settings',
                             usertext = 'Grid lines for minor ticks'),
               pixmap='settings_axisminorgridlines' )

    def _getUserDescription(self):
        """User friendly description."""
        s = self.settings
        return "range %s to %s%s" % ( str(s.min), str(s.max),
                                      ['',' (log)'][s.log])
    userdescription = property(_getUserDescription)

    def setAutoRange(self, autorange):
        """Set the automatic range of this axis (called from page helper)."""

        if autorange:
            scale = self.settings.datascale
            self.autorange = ar = [x*scale for x in autorange]
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
            try:
                widget = s.get('match').getReferredWidget()
            except setting.InvalidType:
                widget = None

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
            if self.plottedrange[0] < 1e-99:
                self.plottedrange[0] = 1e-99
            if self.plottedrange[1] < 1e-99:
                self.plottedrange[1] = 1e-99
            if self.plottedrange[0] == self.plottedrange[1]:
                self.plottedrange[1] = self.plottedrange[0]*2

        # work out tick values and expand axes if necessary
        if s.mode in ('numeric', 'labels'):
            tickclass = axisticks.AxisTicks
        else:
            tickclass = axisticks.DateTicks
        
        axs = tickclass(self.plottedrange[0], self.plottedrange[1],
                        s.MajorTicks.number, s.MinorTicks.number,
                        extendbounds = s.autoExtend,
                        extendzero = s.autoExtendZero,
                        logaxis = s.log )

        (self.plottedrange[0],self.plottedrange[1],
         self.majortickscalc, self.minortickscalc, 
         self.autoformat) =  axs.getTicks()

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

    def _updatePlotRange(self, bounds, otherposition=None):
        """Calculate coordinates on plotter of axis."""

        s = self.settings
        x1, y1, x2, y2 = bounds
        dx = x2 - x1
        dy = y2 - y1
        p1, p2 = s.lowerPosition, s.upperPosition
        if otherposition is None:
            otherposition = s.otherPosition

        if s.direction == 'horizontal': # horizontal
            self.coordParr1 = x1 + dx*p1
            self.coordParr2 = x1 + dx*p2

            # other axis coordinates
            self.coordPerp  = y2 - dy*otherposition
            self.coordPerp1 = y2 - dy*p1
            self.coordPerp2 = y2 - dy*p2

        else: # vertical
            self.coordParr1 = y2 - dy*p1
            self.coordParr2 = y2 - dy*p2

            # other axis coordinates
            self.coordPerp  = x1 + dx*otherposition
            self.coordPerp1 = x1 + dx*p1
            self.coordPerp2 = x1 + dx*p2

        # is this axis reflected
        if otherposition > 0.5:
            self.coordReflected = not s.reflect
        else:
            self.coordReflected = s.reflect
     
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
            fracposns = self.logConvertToPlotter(vals)
        else:
            fracposns = self.linearConvertToPlotter(vals)

        return self.coordParr1 + fracposns*(self.coordParr2-self.coordParr1)

    def dataToPlotterCoords(self, posn, data):
        """Convert data values to plotter coordinates, scaling if necessary."""
        # if the doc was modified, recompute the range
        if self.docchangeset != self.document.changeset:
            self._computePlottedRange()

        self._updatePlotRange(posn)
        return self._graphToPlotter(data*self.settings.datascale)
    
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
        frac = ( (vals.astype(N.float64) - self.coordParr1) /
                 (self.coordParr2 - self.coordParr1) )

        # convert from fractional to graph
        if self.settings.log:
            return self.logConvertFromPlotter(frac)
        else:
            return self.linearConvertFromPlotter(frac)
        
    def plotterToDataCoords(self, bounds, vals):
        """Convert plotter coordinates to data, removing scaling."""
        try:
            scale = 1./self.settings.datascale
        except ZeroDivisionError:
            scale = 0.
        return scale * self.plotterToGraphCoords(bounds, vals)

    def linearConvertToPlotter(self, v):
        """Convert graph coordinates to fractional plotter units for linear scale.
        """
        return ( (v - self.plottedrange[0]) /
                 (self.plottedrange[1] - self.plottedrange[0]) )
    
    def linearConvertFromPlotter(self, v):
        """Convert from (fractional) plotter coords to graph coords.
        """
        return ( self.plottedrange[0] + v *
                 (self.plottedrange[1]-self.plottedrange[0]) )
    
    def logConvertToPlotter(self, v):
        """Convert graph coordinates to fractional plotter units for log10 scale.
        """

        log1 = N.log(self.plottedrange[0])
        log2 = N.log(self.plottedrange[1])
        return ( N.log( N.clip(v, 1e-99, 1e99) ) - log1 )/(log2 - log1)
    
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
        """Draw line, but swap x & y coordinates if vertical axis."""
        if self.settings.direction == 'horizontal':
            painter.drawLine(qt4.QPointF(a1, b1), qt4.QPointF(a2, b2))
        else:
            painter.drawLine(qt4.QPointF(b1, a1), qt4.QPointF(b2, a2))

    def swaplines(self, painter, a1, b1, a2, b2):
        """Multiline version of swapline where a1, b1, a2, b2 are arrays."""
        if self.settings.direction == 'horizontal':
            a = (a1, b1, a2, b2)
        else:
            a = (b1, a1, b2, a2)
        utils.plotLinesToPainter(painter, a[0], a[1], a[2], a[3])

    def _drawGridLines(self, subset, painter, coordticks):
        """Draw grid lines on the plot."""
        painter.setPen( self.settings.get(subset).makeQPen(painter) )
        self.swaplines(painter,
                       coordticks, coordticks*0.+self.coordPerp1,
                       coordticks, coordticks*0.+self.coordPerp2)

    def _drawAxisLine(self, painter):
        """Draw the line of the axis."""

        pen = self.settings.get('Line').makeQPen(painter)
        pen.setCapStyle(qt4.Qt.FlatCap)
        painter.setPen(pen)
        self.swapline( painter,
                       self.coordParr1, self.coordPerp,
                       self.coordParr2, self.coordPerp )        

    def _drawMinorTicks(self, painter, coordminorticks):
        """Draw minor ticks on plot."""

        s = self.settings
        mt = s.get('MinorTicks')
        pen = mt.makeQPen(painter)
        pen.setCapStyle(qt4.Qt.FlatCap)
        painter.setPen(pen)
        delta = mt.getLength(painter)

        if s.direction == 'vertical':
            delta *= -1
        if self.coordReflected:
            delta *= -1
        if s.outerticks:
            delta *= -1
        
        y = coordminorticks*0.+self.coordPerp
        self.swaplines( painter,
                        coordminorticks, y,
                        coordminorticks, y-delta )

    def _drawMajorTicks(self, painter, tickcoords):
        """Draw major ticks on the plot."""

        s = self.settings
        mt = s.get('MajorTicks')
        pen = mt.makeQPen(painter)
        pen.setCapStyle(qt4.Qt.FlatCap)
        painter.setPen(pen)
        startdelta = mt.getLength(painter)
        delta = startdelta

        if s.direction == 'vertical':
            delta *= -1
        if self.coordReflected:
            delta *= -1
        if s.outerticks:
            delta *= -1

        y = tickcoords*0.+self.coordPerp
        self.swaplines( painter,
                        tickcoords, y,
                        tickcoords, y-delta )

        # account for ticks if they are in the direction of the label
        if s.outerticks and not self.coordReflected:
            self._delta_axis += abs(delta)

    def generateLabelLabels(self, phelper):
        """Generate list of positions and labels from widgets using this
        axis."""
        try:
            plotters = phelper.axisplottermap[self]
        except (AttributeError, KeyError):
            return

        dir = self.settings.direction
        minval, maxval = self.plottedrange
        for plotter in plotters:
            # get label and label coordinates from plotter (if any)
            labels, coords = plotter.getAxisLabels(dir)
            if None not in (labels, coords):
                # convert coordinates to plotter coordinates
                pcoords = self._graphToPlotter(coords)
                for coord, pcoord, lab in izip(coords, pcoords, labels):
                    # return labels that are within the plotted range
                    # of coordinates
                    if N.isfinite(coord) and (minval <= coord <= maxval):
                        yield pcoord, lab

    def _drawTickLabels(self, phelper, painter, coordticks, sign, outerbounds,
                        texttorender):
        """Draw tick labels on the plot.

        texttorender is a list which contains text for the axis to render
        after checking for collisions
        """

        s = self.settings
        vertical = s.direction == 'vertical'
        font = s.get('TickLabels').makeQFont(painter)
        painter.setFont(font)
        fm = utils.FontMetrics(font, painter.device())
        tl_spacing = fm.leading() + fm.descent()

        # work out font alignment
        if s.TickLabels.rotate:
            if self.coordReflected:
                angle = 90
            else:
                angle = 270
        else:
            angle = 0
        
        if vertical:
            # limit tick labels to be directly below/besides axis
            bounds = { 'miny': min(self.coordParr1, self.coordParr2),
                       'maxy': max(self.coordParr1, self.coordParr2) }
            ax, ay = 1, 0
        else:
            bounds = { 'minx': min(self.coordParr1, self.coordParr2),
                       'maxx': max(self.coordParr1, self.coordParr2) }
            ax, ay = 0, 1

        if self.coordReflected:
            ax, ay = -ax, -ay

        # get information about text scales
        tl = s.get('TickLabels')
        scale = tl.scale
        pen = tl.makeQPen()

        # an extra offset if required
        self._delta_axis += tl.get('offset').convert(painter)

        def generateTickLabels():
            """Return plotter position of labels and label text."""
            # get format for labels
            format = s.TickLabels.format
            if format.lower() == 'auto':
                format = self.autoformat

            # generate positions and labels
            for posn, tickval in izip(coordticks, self.majortickscalc):
                text = utils.formatNumber(tickval*scale, format,
                                          locale=self.document.locale)
                yield posn, text

        # position of label perpendicular to axis
        perpposn = self.coordPerp + sign*(self._delta_axis+tl_spacing)

        # use generator function to get labels and positions
        if s.mode == 'labels':
            ticklabels = self.generateLabelLabels(phelper)
        else:
            ticklabels = generateTickLabels()

        # iterate over each label
        maxdim = 0
        for parlposn, text in ticklabels:

            # x and y round other way if vertical
            if vertical:
                x, y = perpposn, parlposn
            else:
                x, y = parlposn, perpposn

            r = utils.Renderer(painter, font, x, y, text, alignhorz=ax,
                               alignvert=ay, angle=angle)
            if outerbounds is not None:
                # make sure ticks are within plot
                if vertical:
                    r.ensureInBox(miny=outerbounds[1], maxy=outerbounds[3],
                                  extraspace=True)
                else:
                    r.ensureInBox(minx=outerbounds[0], maxx=outerbounds[2],
                                  extraspace=True)

            bnd = r.getBounds()
            texttorender.append( (r, pen) )

            # keep track of maximum extent of label perpendicular to axis
            if vertical:
                maxdim = max(maxdim, bnd[2] - bnd[0])
            else:
                maxdim = max(maxdim, bnd[3] - bnd[1])

        # keep track of where we are
        self._delta_axis += 2*tl_spacing + maxdim

    def _drawAxisLabel(self, painter, sign, outerbounds, texttorender):
        """Draw an axis label on the plot.

        texttorender is a list which contains text for the axis to render
        after checking for collisions
        """

        s = self.settings
        sl = s.Label
        label = s.get('Label')
        font = label.makeQFont(painter)
        painter.setFont(font)
        fm = utils.FontMetrics(font, painter.device())
        al_spacing = fm.leading() + fm.descent()

        # an extra offset if required
        self._delta_axis += label.get('offset').convert(painter)

        text = s.label
        # avoid adding blank text to plot
        if not text:
            return

        horz = s.direction == 'horizontal'
        if not horz:
            ax, ay = 1, 0
        else:
            ax, ay = 0, 1

        reflected = self.coordReflected
        if reflected:
            ax, ay = -ax, -ay

        # angle of text
        if ( (horz and not sl.rotate) or
             (not horz and sl.rotate) ):
            angle = 0
        else:
            if reflected:
                angle = 90
            else:
                angle = 270

        x = 0.5*(self.coordParr1 + self.coordParr2)
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

    def _autoMirrorDraw(self, posn, painter, coordticks, coordminorticks):
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
        if s.otherPosition < 0.5:
            otheredge = 1.
        else:
            otheredge = 0.

        # temporarily change position of axis to other side for drawing
        self._updatePlotRange(posn, otherposition=otheredge)
        if not s.Line.hide:
            self._drawAxisLine(painter)
        if not s.MinorTicks.hide:
            self._drawMinorTicks(painter, coordminorticks)
        if not s.MajorTicks.hide:
            self._drawMajorTicks(painter, coordticks)

    def chooseName(self):
        """Get default name for axis. Make x and y axes, then axisN."""

        try:
            widgets = set(self.parent.childnames)
        except AttributeError:
            widgets = set()
        for name in ('x', 'y'):
            if name not in widgets:
                return name
        return widget.Widget.chooseName(self)

    def _suppressText(self, painter, parentposn, outerbounds):
        """Whether to suppress drawing text on this axis because it
        is too close to the edge of its parent bounding box.

        If the edge of the plot is within textheight then suppress text
        """
        s = self.settings
        height = utils.FontMetrics( s.get('Label').makeQFont(painter),
                                    painter.device()).height()
        otherposition = s.otherPosition

        if s.direction == 'vertical':
            if ( ( otherposition < 0.01 and
                   abs(parentposn[0]-outerbounds[0]) < height) or
                 ( otherposition > 0.99 and
                   abs(parentposn[2]-outerbounds[2]) < height) ):
                return True
        else:
            if ( ( otherposition < 0.01 and
                   abs(parentposn[3]-outerbounds[3]) < height) or
                 ( otherposition > 0.99 and
                   abs(parentposn[1]-outerbounds[1]) < height) ):
                return True
        return False

    def draw(self, parentposn, phelper, outerbounds=None,
             useexistingpainter=None):
        """Plot the axis on the painter.

        useexistingpainter is a hack so that a colorbar can reuse the
        drawing code here. If set to a painter, it will use this rather
        than opening a new one.
        """

        s = self.settings

        # recompute if document modified
        if self.docchangeset != self.document.changeset:
            self._computePlottedRange()

        posn = widget.Widget.draw(self, parentposn, phelper, outerbounds)
        self._updatePlotRange(posn)

        # get ready to draw
        if useexistingpainter is not None:
            painter = useexistingpainter
        else:
            painter = phelper.painter(self, posn)

        # make control item for axis
        phelper.setControlGraph(self, [ controlgraph.ControlAxisLine(
                    self, s.direction, self.coordParr1,
                    self.coordParr2, self.coordPerp, posn) ])

        # get tick vals
        coordticks = self._graphToPlotter(self.majortickscalc)
        coordminorticks = self._graphToPlotter(self.minortickscalc)

        # exit if axis is hidden
        if s.hide:
            return

        texttorender = []

        # multiplication factor if reflection on the axis is requested
        sign = 1
        if s.direction == 'vertical':
            sign *= -1
        if self.coordReflected:
            sign *= -1

        # plot gridlines
        if not s.MinorGridLines.hide:
            self._drawGridLines('MinorGridLines', painter, coordminorticks)
        if not s.GridLines.hide:
            self._drawGridLines('GridLines', painter, coordticks)

        # plot the line along the axis
        if not s.Line.hide:
            self._drawAxisLine(painter)

        # plot minor ticks
        if not s.MinorTicks.hide:
            self._drawMinorTicks(painter, coordminorticks)

        # keep track of distance from axis
        self._delta_axis = 0

        # plot major ticks
        if not s.MajorTicks.hide:
            self._drawMajorTicks(painter, coordticks)

        # plot tick labels
        suppresstext = self._suppressText(painter, parentposn, outerbounds)
        if not s.TickLabels.hide and not suppresstext:
            self._drawTickLabels(phelper, painter, coordticks, sign,
                                 outerbounds, texttorender)

        # draw an axis label
        if not s.Label.hide and not suppresstext:
            self._drawAxisLabel(painter, sign, outerbounds, texttorender)

        # mirror axis at other side of plot
        if s.autoMirror:
            self._autoMirrorDraw(posn, painter, coordticks, coordminorticks)

        # all the text is drawn at the end so that
        # we can check it doesn't overlap
        drawntext = qt4.QPainterPath()
        for r, pen in texttorender:
            bounds = r.getBounds()
            rect = qt4.QRectF(bounds[0], bounds[1], bounds[2]-bounds[0],
                              bounds[3]-bounds[1])

            if not drawntext.intersects(rect):
                painter.setPen(pen)
                box = r.render()
                drawntext.addRect(rect)
                
    def updateControlItem(self, cgi):
        """Update axis position from control item."""

        s = self.settings
        p = cgi.maxposn
        if s.direction == 'horizontal':
            minfrac = abs((cgi.minpos - p[0]) / (p[2] - p[0]))
            maxfrac = abs((cgi.maxpos - p[0]) / (p[2] - p[0]))
            axisfrac = abs((cgi.axispos - p[3]) / (p[1] - p[3]))
        else:
            minfrac = abs((cgi.minpos - p[3]) / (p[1] - p[3]))
            maxfrac = abs((cgi.maxpos - p[3]) / (p[1] - p[3]))
            axisfrac = abs((cgi.axispos - p[0]) / (p[2] - p[0]))

        if minfrac > maxfrac:
            minfrac, maxfrac = maxfrac, minfrac

        operations = (
            document.OperationSettingSet(s.get('lowerPosition'), minfrac),
            document.OperationSettingSet(s.get('upperPosition'), maxfrac),
            document.OperationSettingSet(s.get('otherPosition'), axisfrac),
            )
        self.document.applyOperation(
            document.OperationMultiple(operations, descr='adjust axis'))

# allow the factory to instantiate an axis
document.thefactory.register( Axis )
