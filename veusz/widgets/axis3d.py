#    Copyright (C) 2014 Jeremy S. Sanders
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
import itertools

from . import widget
from . import axisticks
from .axis import AxisLabel, TickLabel, AutoRange
from ..compat import czip
from .. import qtall as qt4
from .. import document
from .. import setting
from .. import utils

try:
    from ..helpers import threed
except ImportError:
    threed = None

def _(text, disambiguation=None, context='Axis3D'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

foo = []

class AxisTickText(threed.Text):
    """For drawing text at 3D locations."""
    def __init__(self, posns, textlist, font, params):
        threed.Text.__init__(
            self, threed.ValVector(posns), threed.ValVector(posns))
        self.textlist = textlist
        self.font = font
        self.params = params

    def draw(self, painter, pt1, pt2, index, scale, linescale):
        painter.save()
        painter.setPen(qt4.QPen())
        r = utils.Renderer(
            painter, self.font, pt1.x(), pt1.y(), self.textlist[index],
            *self.params)
        r.render()
        painter.restore()

class MajorTick(setting.Line3D):
    '''Major tick settings.'''

    def __init__(self, name, **args):
        setting.Line3D.__init__(self, name, **args)
        self.get('color').newDefault('grey')
        self.add(setting.Float(
            'length',
            20.,
            descr = _('Length of major ticks'),
            usertext= _('Length')))
        self.add(setting.Int(
            'number',
            6,
            descr = _('Number of major ticks to aim for'),
            usertext= _('Number')))
        self.add(setting.FloatList(
            'manualTicks',
            [],
            descr = _('List of tick values'
                      ' overriding defaults'),
            usertext= _('Manual ticks')))

class MinorTick(setting.Line3D):
    '''Minor tick settings.'''

    def __init__(self, name, **args):
        setting.Line3D.__init__(self, name, **args)
        self.get('color').newDefault('grey')
        self.add( setting.Float(
            'length',
            10,
            descr = _('Length of minor ticks'),
            usertext= _('Length')))
        self.add( setting.Int(
            'number',
            20,
            descr = _('Number of minor ticks to aim for'),
            usertext= _('Number')))

class GridLine(setting.Line3D):
    '''Grid line settings.'''

    def __init__(self, name, **args):
        setting.Line3D.__init__(self, name, **args)

        self.get('color').newDefault('grey')
        self.get('hide').newDefault(True)

class MinorGridLine(setting.Line3D):
    '''Minor tick grid line settings.'''

    def __init__(self, name, **args):
        setting.Line3D.__init__(self, name, **args)

        self.get('color').newDefault('lightgrey')
        self.get('hide').newDefault(True)


class Axis3D(widget.Widget):
    """Manages and draws an axis."""

    typename = 'axis3d'
    allowusercreation = True
    description = 'Axis on 3d graph'
    isaxis = True
    isaxis3d = True

    def __init__(self, parent, name=None):
        """Initialise axis."""

        widget.Widget.__init__(self, parent, name=name)
        s = self.settings

        if type(self) == Axis3D:
            self.readDefaults()

        for n in ('x', 'y', 'z'):
            if self.name == n and s.direction != n:
                s.direction = n

        # automatic range
        self.setAutoRange(None)

        # document updates change set variable when things need recalculating
        self.docchangeset = -1

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
        s.add( AutoRange('autoRange', 'next-tick') )
        s.add( setting.Choice('mode',
                              ('numeric', 'datetime', 'labels'),
                              'numeric',
                              descr = _('Type of ticks to show on on axis'),
                              usertext=_('Mode')) )

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
                              ['x', 'y', 'z'],
                              'x',
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
        s.add( setting.Float('otherPosition1', 0.,
                             descr=_('Fractional position of axis '
                                     'in its perpendicular direction 1'),
                             usertext=_('Axis position 1')) )
        s.add( setting.Float('otherPosition2', 0.,
                             descr=_('Fractional position of axis '
                                     'in its perpendicular direction 2'),
                             usertext=_('Axis position 2')) )

        s.add( setting.Line3D('Line',
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
    def allowedParentTypes(self):
        from . import graph3d
        return (graph3d.Graph3D,)

    @property
    def userdescription(self):
        """User friendly description."""
        s = self.settings
        return "range %s to %s%s" % ( str(s.min), str(s.max),
                                      ['',' (log)'][s.log])

    def isLinked(self):
        """Whether is an axis linked to another."""
        return False

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

    def computePlottedRange(self, force=False, overriderange=None):
        """Convert the range requested into a plotted range."""

        if self.docchangeset == self.document.changeset and not force:
            return

        s = self.settings
        if overriderange is None:
            self.plottedrange = [s.min, s.max]
        else:
            self.plottedrange = overriderange

        # automatic lookup of minimum
        if overriderange is None:
            if s.min == 'Auto':
                self.plottedrange[0] = self.autorange[0]
            if s.max == 'Auto':
                self.plottedrange[1] = self.autorange[1]

        # yuck, but sometimes it's true
        # tweak range to make sure things don't blow up further down the
        # line
        if ( abs(self.plottedrange[0] - self.plottedrange[1]) <
             ( abs(self.plottedrange[0]) + abs(self.plottedrange[1]) )*1e-8 ):
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

        r = s.autoRange
        if r == 'exact':
            pass
        elif r == 'next-tick':
            pass
        else:
            val = {'+2%': 0.02, '+5%': 0.05, '+10%': 0.1, '+15%': 0.15}[r]

            if s.log:
                # logarithmic
                logrng = abs( N.log(self.plottedrange[1]) -
                           N.log(self.plottedrange[0]) )
                if s.min == 'Auto':
                    self.plottedrange[0] /= N.exp(logrng * val)
                if s.max == 'Auto':
                    self.plottedrange[1] *= N.exp(logrng * val)
            else:
                # linear
                rng = self.plottedrange[1] - self.plottedrange[0]
                if s.min == 'Auto':
                    self.plottedrange[0] -= rng*val
                if s.max == 'Auto':
                    self.plottedrange[1] += rng*val

        self.computeTicks()

        # invert bounds if axis was inverted
        if invertaxis:
            self.plottedrange = self.plottedrange[::-1]

        self.docchangeset = self.document.changeset

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
                        logaxis = s.log )

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

    def dataToLogicalCoords(self, vals):
        """Compute coordinates on graph to logical graph coordinates (0..1)"""

        self.computePlottedRange()
        s = self.settings

        svals = vals * s.datascale
        if s.log:
            fracposns = self.logConvertToPlotter(svals)
        else:
            fracposns = self.linearConvertToPlotter(svals)

        return s.lowerPosition + fracposns*(s.upperPosition-s.lowerPosition)

    def linearConvertToPlotter(self, v):
        """Convert graph coordinates to 0..1 coordinates"""
        return ( (v - self.plottedrange[0]) /
                 (self.plottedrange[1] - self.plottedrange[0]) )

    def logConvertToPlotter(self, v):
        """Convert graph coordinates to 0..1 coordinates"""
        log1 = N.log(self.plottedrange[0])
        log2 = N.log(self.plottedrange[1])
        return (N.log(N.clip(v, 1e-99, 1e99)) - log1) / (log2 - log1)

    def getAutoMirrorCombs(self):
        """Get combinations of other position for auto mirroring."""
        s = self.settings
        op1 = s.otherPosition1
        op2 = s.otherPosition2
        if not s.autoMirror:
            return ((op1, op2),)
        if op1 == 0 or op1 == 1:
            op1list = [0, 1]
        else:
            op1list = [op1]
        if op2 == 0 or op2 == 1:
            op2list = [0, 1]
        else:
            op2list = [op2]
        return itertools.product(op1list, op2list)

    def addAxisLine(self, cont, dirn):
        """Build list of lines to draw axis line, mirroring if necessary."""

        s = self.settings
        if s.Line.hide:
            return
        lower, upper = s.lowerPosition, s.upperPosition

        outstart = []
        outend = []
        for op1, op2 in self.getAutoMirrorCombs():
            if dirn == 'x':
                outstart += [lower, op1, op2]
                outend += [upper, op1, op2]
            elif dirn == 'y':
                outstart += [op1, lower, op2]
                outend += [op1, upper, op2]
            else:
                outstart += [op1, op2, lower]
                outend += [op1, op2, upper]

        startpts = threed.ValVector(outstart)
        endpts = threed.ValVector(outend)
        lineprop = s.Line.makeLineProp()
        cont.addObject(threed.LineSegments(startpts, endpts, lineprop))

    def addAxisTicks(self, tickprops, tickvals, cont, dirn):
        """Add ticks for the vals and tick properties class given.
        cont: container to add ticks
        dirn: 'x', 'y', 'z' for axis
        """

        if tickprops.hide:
            return

        ticklen = tickprops.length * 1e-3
        tfracs = self.dataToLogicalCoords(tickvals)

        outstart = []
        outend = []
        for op1, op2 in self.getAutoMirrorCombs():
            op1pts = N.full_like(tfracs, op1)
            op2pts = N.full_like(tfracs, op2)
            op1pts2 = N.full_like(tfracs, op1+ticklen*(1 if op1 < 0.5 else -1))
            op2pts2 = N.full_like(tfracs, op2+ticklen*(1 if op2 < 0.5 else -1))

            if dirn == 'x':
                pts1 = (tfracs, op1pts, op2pts)
                pts2 = (tfracs, op1pts2, op2pts)
                pts3 = (tfracs, op1pts, op2pts2)
            elif dirn == 'y':
                pts1 = (op1pts, tfracs, op2pts)
                pts2 = (op1pts2, tfracs, op2pts)
                pts3 = (op1pts, tfracs, op2pts2)
            else:
                pts1 = (op1pts, op2pts, tfracs)
                pts2 = (op1pts2, op2pts, tfracs)
                pts3 = (op1pts, op2pts2, tfracs)

            outstart += [N.ravel(N.column_stack(pts1)), N.ravel(N.column_stack(pts1))]
            outend += [N.ravel(N.column_stack(pts2)), N.ravel(N.column_stack(pts3))]

        text = ['%g' % t for t in tickvals]
        att = AxisTickText(N.ravel(N.column_stack(pts1)), text,
                           qt4.QFont(), {})
        cont.addObject(att)
        foo.append(att)

        startpts = threed.ValVector(N.concatenate(outstart))
        endpts = threed.ValVector(N.concatenate(outend))
        lineprop = tickprops.makeLineProp()
        cont.addObject(threed.LineSegments(startpts, endpts, lineprop))

    def drawToObject(self):

        s = self.settings
        dirn = s.direction

        cont = threed.ObjectContainer()

        self.addAxisLine(cont, dirn)
        self.addAxisTicks(s.MajorTicks, self.majortickscalc, cont, dirn)
        self.addAxisTicks(s.MinorTicks, self.minortickscalc, cont, dirn)

        return cont

# allow the factory to instantiate an axis
document.thefactory.register(Axis3D)
