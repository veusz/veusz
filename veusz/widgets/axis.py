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

from __future__ import division
import numpy as N
import re

from ..compat import czip
from .. import qtall as qt4
from .. import document
from .. import setting
from .. import utils

from . import widget
from . import axisticks
from . import controlgraph

###############################################################################

def _(text, disambiguation=None, context='Axis'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class MajorTick(setting.Line):
    '''Major tick settings.'''

    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)
        self.add( setting.DistancePt( 'length',
                                      '6pt',
                                      descr = _('Length of major ticks'),
                                      usertext= _('Length') ) )
        self.add( setting.Int( 'number',
                               6,
                               descr = _('Number of major ticks to aim for'),
                               usertext= _('Number') ) )
        self.add( setting.FloatList('manualTicks',
                                    [],
                                    descr = _('List of tick values'
                                              ' overriding defaults'),
                                    usertext= _('Manual ticks') ) )

    def getLength(self, painter):
        '''Return tick length in painter coordinates'''

        return self.get('length').convert(painter)

class MinorTick(setting.Line):
    '''Minor tick settings.'''

    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)
        self.add( setting.DistancePt( 'length',
                                      '3pt',
                                      descr = _('Length of minor ticks'),
                                      usertext= _('Length')) )
        self.add( setting.Int( 'number',
                               20,
                               descr = _('Number of minor ticks to aim for'),
                               usertext= _('Number') ) )

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
        self.add( setting.Bool( 'onTop', False,
                                descr = _('Put grid lines on top of graph'),
                                usertext = _('On top') ) )

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
                                descr = _('Place axis label close to edge'
                                          ' of graph'),
                                usertext= _('At edge') ) )
        self.add( setting.RotateInterval(
                'rotate', '0',
                descr = 'Angle by which to rotate label by',
                usertext='Rotate') )
        self.add( setting.DistancePt( 'offset',
                                      '0pt',
                                      descr = _('Additional offset of axis label'
                                                ' from axis tick labels'),
                                      usertext= _('Label offset') ) )
        self.add( setting.Choice(
                'position',
                ('at-minimum', 'centre', 'at-maximum'),
                'centre',
                descr = _('Position of axis label'),
                usertext = _('Position') ) )

class TickLabel(setting.Text):
    """For tick labels on axes."""

    formatchoices = ('Auto', '%Vg', '%Ve', '%VE',
                     '%g', '%e', '%.2f')
    descriptions = ( _('Automatic'),
                     _('General numerical format'),
                     _('Scientific notation'),
                     _('Engineering suffix notation'),
                     _('C-style general format'),
                     _('C-style scientific notation'),
                     _('2 decimal places always shown') )

    def __init__(self, name, **args):
        setting.Text.__init__(self, name, **args)
        self.add( setting.RotateInterval(
                'rotate', '0',
                descr = _('Angle by which to rotate label by'),
                usertext= _('Rotate') ) )
        self.add( setting.ChoiceOrMore( 'format',
                                        TickLabel.formatchoices,
                                        'Auto',
                                        descr = _('Format of the tick labels'),
                                        descriptions=TickLabel.descriptions,
                                        usertext= _('Format') ) )

        self.add( setting.Float('scale', 1.,
                                descr=_('A scale factor to apply to the values '
                                        'of the tick labels'),
                                usertext=_('Scale') ) )

        self.add( setting.DistancePt( 'offset',
                                      '0pt',
                                      descr = _('Additional offset of axis tick '
                                                'labels from axis'),
                                      usertext= _('Tick offset') ) )

###############################################################################

class Axis(widget.Widget):
    """Manages and draws an axis."""

    typename = 'axis'
    allowusercreation = True
    description = 'Axis to a plot or shared in a grid'
    isaxis = True

    def __init__(self, parent, name=None):
        """Initialise axis."""

        widget.Widget.__init__(self, parent, name=name)
        s = self.settings

        if self.name == 'y' and s.direction != 'vertical':
            s.direction = 'vertical'
        elif self.name == 'x' and s.direction != 'horizontal':
            s.direction = 'horizontal'

        # automatic range
        self.setAutoRange(None)

        # document updates change set variable when things need recalculating
        self.docchangeset = -1
        self.currentbounds = [0,0,1,1]

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)
        s.add( setting.Str('label', '',
                           descr=_('Axis label text'),
                           usertext=_('Label')) )
        s.add( setting.AxisBound('min', 'Auto',
                                 descr=_('Minimum value of axis'),
                                 usertext=_('Min')) )
        s.add( setting.AxisBound('max', 'Auto',
                                 descr=_('Maximum value of axis'),
                                 usertext=_('Max')) )
        s.add( setting.Bool('log', False,
                            descr = _('Whether axis is logarithmic'),
                            usertext=_('Log')) )
        s.add( setting.ChoiceOrMore(
                'autoRange',
                ('exact', 'next-tick',
                 '+2%', '+5%', '+10%', '+15%',
                 '-2%', '-5%', '-10%', '-15%',
                 '20-80%', '<20%', '>80%',
                ),
                'next-tick',
                descr = _('If axis range not specified, use range of '
                          'data and this setting'),
                descriptions = (
                    _('Use exact data range'),
                    _('Round up to tick marks from data range'),
                    _('Expand 2% beyond data range'),
                    _('Expand 5% beyond data range'),
                    _('Expand 10% beyond data range'),
                    _('Expand 15% beyond data range'),
                    _('Shrink 2% inside data range'),
                    _('Shrink 5% inside data range'),
                    _('Shrink 10% inside data range'),
                    _('Shrink 15% inside data range'),
                    _('20 to 80% of the data range'),
                    _('Up to 20% of the data range'),
                    _('Above 80% of the data range'),
                ),
                formatting = True,
                usertext = _('Auto range') ) )
        s.add( setting.Choice('mode',
                              ('numeric', 'datetime', 'labels'),
                              'numeric',
                              descr = _('Type of ticks to show on on axis'),
                              usertext=_('Mode')) )

        s.add( setting.SettingBackwardCompat(
                'autoExtend', 'autoRange', True,
                translatefn = lambda x: ('exact', 'next-tick')[x],
                formatting=True ) )

        # this setting no longer used
        s.add( setting.Bool('autoExtendZero', True,
                            descr = _('Extend axis to zero if close (UNUSED)'),
                            usertext=_('Zero extend'),
                            hidden=True,
                            formatting=True) )

        s.add( setting.Bool('autoMirror', True,
                            descr = _('Place axis on opposite side of graph '
                                      'if none'),
                            usertext=_('Auto mirror'),
                            formatting=True) )
        s.add( setting.Bool('reflect', False,
                            descr = _('Place axis text and ticks on other side'
                                      ' of axis'),
                            usertext=_('Reflect'),
                            formatting=True) )
        s.add( setting.Bool('outerticks', False,
                            descr = _('Place ticks on outside of graph'),
                            usertext=_('Outer ticks'),
                            formatting=True) )

        s.add( setting.Float('datascale', 1.,
                             descr=_('Scale data plotted by this factor'),
                             usertext=_('Scale')) )

        s.add( setting.Choice('direction',
                              ['horizontal', 'vertical'],
                              'horizontal',
                              descr = _('Direction of axis'),
                              usertext=_('Direction')) )
        s.add( setting.Float('lowerPosition', 0.,
                             descr=_('Fractional position of lower end of '
                                     'axis on graph'),
                             usertext=_('Min position')) )
        s.add( setting.Float('upperPosition', 1.,
                             descr=_('Fractional position of upper end of '
                                     'axis on graph'),
                             usertext=_('Max position')) )
        s.add( setting.Float('otherPosition', 0.,
                             descr=_('Fractional position of axis '
                                     'in its perpendicular direction'),
                             usertext=_('Axis position')) )

        s.add( setting.WidgetPath('match', '',
                                  descr =
                                  _('Match the scale of this axis to the '
                                    'axis specified'),
                                  usertext=_('Match'),
                                  allowedwidgets = [Axis] ))

        s.add( setting.Line('Line',
                            descr = _('Axis line settings'),
                            usertext = _('Axis line')),
               pixmap='settings_axisline' )
        s.add( AxisLabel('Label',
                         descr = _('Axis label settings'),
                         usertext = _('Axis label')),
               pixmap='settings_axislabel' )
        s.add( TickLabel('TickLabels',
                         descr = _('Tick label settings'),
                         usertext = _('Tick labels')),
               pixmap='settings_axisticklabels' )
        s.add( MajorTick('MajorTicks',
                         descr = _('Major tick line settings'),
                         usertext = _('Major ticks')),
               pixmap='settings_axismajorticks' )
        s.add( MinorTick('MinorTicks',
                         descr = _('Minor tick line settings'),
                         usertext = _('Minor ticks')),
               pixmap='settings_axisminorticks' )
        s.add( GridLine('GridLines',
                        descr = _('Grid line settings'),
                        usertext = _('Grid lines')),
               pixmap='settings_axisgridlines' )
        s.add( MinorGridLine('MinorGridLines',
                             descr = _('Minor grid line settings'),
                             usertext = _('Grid lines for minor ticks')),
               pixmap='settings_axisminorgridlines' )

    @classmethod
    def allowedParentTypes(klass):
        from . import graph, grid
        return (graph.Graph, grid.Grid)

    @property
    def userdescription(self):
        """User friendly description."""
        s = self.settings
        return "range %s to %s%s" % (
            str(s.min), str(s.max),
            ['',' (log)'][self.plottedLog()])

    def isLinked(self):
        """Whether is an axis linked to another."""
        return False

    def getLinkedAxis(self):
        """Return axis linked to this one (or None)."""
        return None

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

    def usesAutoRange(self):
        """Return whether any of the bounds are automatically determined."""
        return self.settings.min == 'Auto' or self.settings.max == 'Auto'

    # +5% or -5%
    re_dr_plusminus = re.compile(r'^\s*([+-][0-9]+)\s*%\s*$')
    # 5-10%
    re_dr_percrange = re.compile(r'^\s*(-?[0-9]+)\s*-\s*([0-9]+)\s*%\s*$')
    # < 5%
    re_dr_lower = re.compile(r'^\s*<\s*([0-9]+)\s*%\s*$')
    # > 95%
    re_dr_upper = re.compile(r'^\s*>\s*([0-9]+)\s*%\s*$')

    def autoRangeToFracs(self, rng):
        """Convert auto range setting to fractions to expand range by.

        Returns tuple of expansion fractions for left and right
        """

        # +X% or -Y%
        m = self.re_dr_plusminus.match(rng)
        if m:
            v = float(m.group(1))*0.01
            return v, v

        # X-Y%
        m = self.re_dr_percrange.match(rng)
        if m:
            v1 = -float(m.group(1))*0.01
            v2 = -(1-float(m.group(2))*0.01)
            return v1, v2

        # <X%
        m = self.re_dr_lower.match(rng)
        if m:
            return 0, -(1-float(m.group(1))*0.01)

        # >Y%
        m = self.re_dr_upper.match(rng)
        if m:
            return -(float(m.group(1))*0.01), 0

        # error
        self.document.log(
            _("Invalid axis range '%s'") % rng)
        return 0, 0

    def computePlottedRange(self, force=False, overriderange=None):
        """Convert the range requested into a plotted range."""

        if self.docchangeset == self.document.changeset and not force:
            return

        s = self.settings
        if overriderange is None:
            self.plottedrange = [s.min, s.max]
        else:
            self.plottedrange = overriderange

        # match the scale of this axis to another
        matched = False
        if s.match != '':
            # locate widget we're matching
            # this is ensured to be an Axis
            try:
                widget = s.get('match').getReferredWidget()
            except utils.InvalidType:
                widget = None

            # this looks valid + sanity checks
            if (widget is not None and widget != self and
                widget.settings.match == ''):
                # update if out of date
                if widget.docchangeset != self.document.changeset:
                    widget.computePlottedRange()
                # copy the range
                self.plottedrange = list(widget.plottedrange)
                matched = True

        # automatic lookup of minimum
        if not matched and overriderange is None:
            if s.min == 'Auto':
                self.plottedrange[0] = self.autorange[0]
            if s.max == 'Auto':
                self.plottedrange[1] = self.autorange[1]

        # yuck, but sometimes it's true
        # tweak range to make sure things don't blow up further down the
        # line
        if ( abs(self.plottedrange[0] - self.plottedrange[1]) <
             ( abs(self.plottedrange[0]) + abs(self.plottedrange[1]) )*1e-12 ):
               self.plottedrange[1] = ( self.plottedrange[0] +
                                        max(1., self.plottedrange[0]*0.1) )

        # handle axis values round the wrong way
        invertaxis = self.plottedrange[0] > self.plottedrange[1]
        if invertaxis:
            self.plottedrange = self.plottedrange[::-1]

        # make sure log axes don't blow up
        if s.log:
            if self.plottedrange[0] < 1e-99:
                self.plottedrange[0] = 1e-99
            if self.plottedrange[1] < 1e-99:
                self.plottedrange[1] = 1e-99
            if self.plottedrange[0] == self.plottedrange[1]:
                self.plottedrange[1] = self.plottedrange[0]*2

        rng = s.autoRange
        if rng == 'exact':
            pass
        elif rng == 'next-tick':
            pass
        else:
            # get fractions to expand range by
            expandfrac1, expandfrac2 = self.autoRangeToFracs(rng)
            origrange = list(self.plottedrange)
            if s.log:
                # logarithmic
                logrng = abs( N.log(self.plottedrange[1]) -
                           N.log(self.plottedrange[0]) )
                if s.min == 'Auto':
                    self.plottedrange[0] /= N.exp(logrng * expandfrac1)
                if s.max == 'Auto':
                    self.plottedrange[1] *= N.exp(logrng * expandfrac2)
            else:
                # linear
                rng = self.plottedrange[1] - self.plottedrange[0]
                if s.min == 'Auto':
                    self.plottedrange[0] -= rng*expandfrac1
                if s.max == 'Auto':
                    self.plottedrange[1] += rng*expandfrac2

            # if order is wrong, then give error!
            if self.plottedrange[1] <= self.plottedrange[0]:
                self.document.log(_("Invalid axis range '%s'") % rng)
                self.plottedrange = origrange

        self.computeTicks()

        # invert bounds if axis was inverted
        if invertaxis:
            self.plottedrange = self.plottedrange[::-1]

        self.docchangeset = self.document.changeset

    def plottedLog(self):
        """Plotted in log?
        This is overridden if the mode is incorrect."""
        return (self.settings.log and
                self.settings.mode in ('numeric', 'labels'))

    def computeTicks(self, allowauto=True):
        """Update ticks given plotted range.
        if allowauto is False, then do not allow ticks to be
        updated
        """

        s = self.settings

        if s.mode in ('numeric', 'labels'):
            tickclass = axisticks.AxisTicks
        else:
            tickclass = axisticks.DateTicks

        nexttick = s.autoRange == 'next-tick'
        extendmin = nexttick and s.min == 'Auto' and allowauto
        extendmax = nexttick and s.max == 'Auto' and allowauto

        # create object to compute ticks
        axs = tickclass(self.plottedrange[0], self.plottedrange[1],
                        s.MajorTicks.number, s.MinorTicks.number,
                        extendmin = extendmin, extendmax = extendmax,
                        logaxis = self.plottedLog())

        axs.getTicks()
        self.plottedrange[0] = axs.minval
        self.plottedrange[1] = axs.maxval
        self.majortickscalc = axs.tickvals
        self.minortickscalc = axs.minorticks
        self.autoformat = axs.autoformat

        # override values if requested
        if len(s.MajorTicks.manualTicks) > 0:
            ticks = []
            for i in s.MajorTicks.manualTicks:
                if i >= self.plottedrange[0] and i <= self.plottedrange[1]:
                    ticks.append(i)
            self.majortickscalc = N.array(ticks)

    def getPlottedRange(self):
        """Return the range plotted by the axes."""
        self.computePlottedRange()
        return (self.plottedrange[0], self.plottedrange[1])

    def updateAxisLocation(self, bounds, otherposition=None,
                           lowerupperposition=None):
        """Recalculate coordinates on plotter of axis.

        otherposition: override otherPosition setting
        lowerupperposition: set to tuple (lower, upper) to override
         lowerPosition and upperPosition settings
        """

        s = self.settings

        if lowerupperposition is None:
            p1, p2 = s.lowerPosition, s.upperPosition
        else:
            p1, p2 = lowerupperposition
        if otherposition is None:
            otherposition = s.otherPosition

        x1, y1, x2, y2 = self.currentbounds = bounds
        dx = x2 - x1
        dy = y2 - y1

        if s.direction == 'horizontal': # horizontal
            self.coordParr1 = x1 + dx*p1
            self.coordParr2 = x1 + dx*p2

            # other axis coordinates
            self.coordPerp  = y2 - dy*otherposition
            self.coordPerp1 = y1
            self.coordPerp2 = y2

        else: # vertical
            self.coordParr1 = y2 - dy*p1
            self.coordParr2 = y2 - dy*p2

            # other axis coordinates
            self.coordPerp  = x1 + dx*otherposition
            self.coordPerp1 = x1
            self.coordPerp2 = x2

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
        self.updateAxisLocation(bounds)

        return self._graphToPlotter(vals)

    def _graphToPlotter(self, vals):
        """Convert the coordinates assuming the machinery is in place."""

        # work out fractional posistions, then convert to pixels
        if self.plottedLog():
            fracposns = self.logConvertToPlotter(vals)
        else:
            fracposns = self.linearConvertToPlotter(vals)

        return self.coordParr1 + fracposns*(self.coordParr2-self.coordParr1)

    def dataToPlotterCoords(self, posn, data):
        """Convert data values to plotter coordinates, scaling if necessary."""
        self.updateAxisLocation(posn)
        return self._graphToPlotter(data*self.settings.datascale)

    def plotterToGraphCoords(self, bounds, vals):
        """Convert plotter coordinates on this axis to graph coordinates.

        bounds specifies the plot bounds
        vals is a numpy of coordinates
        returns a numpy of floats
        """

        self.updateAxisLocation( bounds )

        # work out fractional positions of the plotter coords
        frac = ( (vals.astype(N.float64) - self.coordParr1) /
                 (self.coordParr2 - self.coordParr1) )

        # convert from fractional to graph
        if self.plottedLog():
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
        return (N.log(N.clip(v, 1e-99, 1e99)) - log1)/(log2 - log1)

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

    def _drawGridLines(self, subset, painter, coordticks, parentposn):
        """Draw grid lines on the plot."""
        painter.setPen( self.settings.get(subset).makeQPen(painter) )

        # drop points which overlap with graph box (if used)
        if self.parent.typename == 'graph':
            if not self.parent.settings.Border.hide:
                if self.settings.direction == 'horizontal':
                    ok = ( (N.abs(coordticks-parentposn[0]) > 1e-3) &
                           (N.abs(coordticks-parentposn[2]) > 1e-3) )
                else:
                    ok = ( (N.abs(coordticks-parentposn[1]) > 1e-3) &
                           (N.abs(coordticks-parentposn[3]) > 1e-3) )
                coordticks = coordticks[ok]

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
            if labels is not None and coords is not None:
                # convert coordinates to plotter coordinates
                pcoords = self._graphToPlotter(coords)
                for coord, pcoord, lab in czip(coords, pcoords, labels):
                    # return labels that are within the plotted range
                    # of coordinates
                    if N.isfinite(coord) and (minval <= coord <= maxval):
                        yield pcoord, lab

    def _drawTickLabels(self, phelper, painter, coordticks, sign, outerbounds,
                        tickvals, texttorender):
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
        angle = int(s.TickLabels.rotate)
        if not self.coordReflected and angle != 0:
            angle = 360-angle

        if vertical:
            # limit tick labels to be directly below/besides axis
            ax, ay = 1, 0
        else:
            ax, ay = 0, 1

        if self.coordReflected:
            ax, ay = -ax, -ay

        # get information about text scales
        tl = s.get('TickLabels')
        scale = tl.scale
        pen = tl.makeQPen(painter)

        # an extra offset if required
        self._delta_axis += tl.get('offset').convert(painter)

        def generateTickLabels():
            """Return plotter position of labels and label text."""
            # get format for labels
            format = s.TickLabels.format
            if format.lower() == 'auto':
                format = self.autoformat

            # generate positions and labels
            for posn, tickval in czip(coordticks, tickvals):
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

            r = utils.Renderer(
                painter, font, x, y, text, alignhorz=ax,
                alignvert=ay, angle=angle,
                doc=self.document)

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

        align1 = 1
        align2 = {'centre': 0,
                  'at-minimum': -1,
                  'at-maximum': 1}[sl.position]

        if horz:
            ax, ay = align2, align1
        else:
            ax, ay = align1, align2

        reflected = self.coordReflected
        if reflected:
            if horz:
                ay = -ay
            else:
                ax = -ax

        # angle of text (logic is slightly complex)
        angle = int(sl.rotate)
        if horz:
            if not reflected:
                angle = 360-angle
        else:
            angle = angle+270
            if reflected:
                angle = 360-angle
        angle = angle % 360

        if sl.position == 'centre':
            x = 0.5*(self.coordParr1 + self.coordParr2)
        elif sl.position == 'at-minimum':
            x = self.coordParr1
        else:
            x = self.coordParr2

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

        r = utils.Renderer(
            painter, font, x, y, text,
            ax, ay, angle,
            usefullheight=True,
            doc=self.document)

        # make sure text is in plot rectangle
        if outerbounds is not None:
            r.ensureInBox( minx=outerbounds[0], maxx=outerbounds[2],
                           miny=outerbounds[1], maxy=outerbounds[3] )

        texttorender.insert(0, (r, s.get('Label').makeQPen(painter)) )

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

        if outerbounds is None:
            return False

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

    def drawGrid(self, parentposn, phelper, outerbounds=None,
                 ontop=False):
        """Code to draw gridlines.

        This is separate from the main draw routine because the grid
        should be behind/infront the data points.
        """

        s = self.settings
        if ( s.hide or (s.MinorGridLines.hide and s.GridLines.hide) or
             s.GridLines.onTop != bool(ontop) ):
            return

        # draw grid on a different layer, depending on whether on top or not
        layer = (-2, -1)[bool(ontop)]
        painter = phelper.painter(self, parentposn, layer=layer)
        self.updateAxisLocation(parentposn)

        with painter:
            painter.save()
            painter.setClipRect( qt4.QRectF(
                    qt4.QPointF(parentposn[0], parentposn[1]),
                    qt4.QPointF(parentposn[2], parentposn[3]) ) )

            if not s.MinorGridLines.hide:
                coordminorticks = self._graphToPlotter(self.minortickscalc)
                self._drawGridLines('MinorGridLines', painter, coordminorticks,
                                    parentposn)
            if not s.GridLines.hide:
                coordticks = self._graphToPlotter(self.majortickscalc)
                self._drawGridLines('GridLines', painter, coordticks,
                                    parentposn)

            painter.restore()

    def _drawAutoMirrorTicks(self, posn, painter):
        s = self.settings
        coordticks = self._graphToPlotter(self.majortickscalc)
        coordminorticks = self._graphToPlotter(self.minortickscalc)

        if s.otherPosition < 0.5:
            otheredge = 1.
        else:
            otheredge = 0.

        # temporarily change position of axis to other side for drawing
        self.updateAxisLocation(posn, otherposition=otheredge)
        if not s.Line.hide:
            self._drawAxisLine(painter)
        if not s.MinorTicks.hide:
            self._drawMinorTicks(painter, coordminorticks)
        if not s.MajorTicks.hide:
            self._drawMajorTicks(painter, coordticks)

    def drawAutoMirror(self, parentposn, phelper, allaxes):
        """Draw mirrored ticks."""

        s = self.settings

        # if there's another axis in this direction, we don't mirror
        count = 0
        thisdir = s.direction
        for a in allaxes:
            if a.settings.direction == thisdir and not a.settings.hide:
                count += 1
        if count > 1 or not s.autoMirror or s.hide:
            return

        painter = phelper.painter(self, parentposn, layer=-1)
        self.updateAxisLocation(parentposn)
        with painter:
            painter.save()
            self._drawAutoMirrorTicks(parentposn, painter)
            painter.restore()

    def draw(self, parentposn, phelper, outerbounds=None):
        """Plot the axis on the painter.
        """

        self.updateAxisLocation(parentposn)

        # exit if axis is hidden
        if self.settings.hide:
            return

        self.computePlottedRange()
        painter = phelper.painter(self, parentposn)
        with painter:
            self._axisDraw(
                parentposn, parentposn, outerbounds, painter, phelper)

    def _drawTextWithoutOverlap(self, painter, texttorender):
        """Aall the text is drawn at the end so that we can check it
        doesn't overlap.

        texttorender is a list of (Renderer, QPen) tuples.
        """
        overlaps = utils.RectangleOverlapTester()

        for r, pen in texttorender:
            rect = r.getTightBounds()

            if not overlaps.willOverlap(rect):
                painter.setPen(pen)
                r.render()
                overlaps.addRect(rect)

            # debug
            # poly = rect.makePolygon()
            # painter.drawPolygon(poly)

    def _axisDraw(self, posn, parentposn, outerbounds, painter, phelper):
        """Internal drawing routine."""

        s = self.settings

        # make control item for axis
        phelper.setControlGraph(self, [ controlgraph.ControlAxisLine(
                    self, s.direction, self.coordParr1,
                    self.coordParr2, self.coordPerp, posn) ])

        # get tick vals
        coordticks = self._graphToPlotter(self.majortickscalc)
        coordminorticks = self._graphToPlotter(self.minortickscalc)

        texttorender = []

        # multiplication factor if reflection on the axis is requested
        sign = 1
        if s.direction == 'vertical':
            sign *= -1
        if self.coordReflected:
            sign *= -1

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
                                 outerbounds, self.majortickscalc, texttorender)

        # draw an axis label
        if not s.Label.hide and not suppresstext:
            self._drawAxisLabel(painter, sign, outerbounds, texttorender)

        self._drawTextWithoutOverlap(painter, texttorender)

    def updateControlItem(self, cgi):
        """Update axis position from control item."""

        s = self.settings
        p = cgi.maxposn

        if cgi.zoomed():
            # zoom axis scale
            # we convert a neighbouring pixel to see how we should
            # round the text
            c1, c2, c1delt, c2delt = self.plotterToGraphCoords(
                cgi.maxposn, N.array([cgi.minzoom, cgi.maxzoom,
                                      cgi.minzoom+1, cgi.maxzoom-1]))
            if c1 > c2:
                c1, c2 = c2, c1
                c1delt, c2delt = c2delt, c1delt

            round1 = utils.round2delt(c1, c1delt)
            round2 = utils.round2delt(c2, c2delt)

            ops = []
            if ( (s.min == 'Auto' or not N.allclose(c1, s.min, rtol=1e-8))
                 and N.isfinite(round1) ):
                ops.append( document.OperationSettingSet(
                        s.get('min'), round1) )
            if ( (s.max == 'Auto' or not N.allclose(c2, s.max, rtol=1e-8))
                 and N.isfinite(round2) ):
                ops.append( document.OperationSettingSet(
                        s.get('max'), round2) )

            self.document.applyOperation(
                document.OperationMultiple(ops, descr=_('zoom axis')))

        elif cgi.moved():
            # move axis
            # convert positions to fractions
            pt1, pt2, ppt1, ppt2 = ( (3, 1, 0, 2), (0, 2, 3, 1)
                                     ) [s.direction == 'horizontal']
            minfrac = abs((cgi.minpos - p[pt1]) / (p[pt2] - p[pt1]))
            maxfrac = abs((cgi.maxpos - p[pt1]) / (p[pt2] - p[pt1]))
            axisfrac = abs((cgi.axispos - p[ppt1]) / (p[ppt2] - p[ppt1]))

            # swap if wrong way around
            if minfrac > maxfrac:
                minfrac, maxfrac = maxfrac, minfrac

            # update doc
            ops = []
            if s.lowerPosition != minfrac:
                ops.append( document.OperationSettingSet(
                        s.get('lowerPosition'), round(minfrac, 3)) )
            if s.upperPosition != maxfrac:
                ops.append( document.OperationSettingSet(
                        s.get('upperPosition'), round(maxfrac, 3)) )
            if s.otherPosition != axisfrac:
                ops.append( document.OperationSettingSet(
                        s.get('otherPosition'), round(axisfrac, 3)) )
            self.document.applyOperation(
                document.OperationMultiple(ops, descr=_('adjust axis')))

# allow the factory to instantiate an axis
document.thefactory.register( Axis )
