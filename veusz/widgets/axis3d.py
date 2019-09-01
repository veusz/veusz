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

"""Widget to plot 3D axes, and to handle conversion of coordinates to plot
positions."""

from __future__ import division, print_function, absolute_import
import math
import numpy as N
import itertools

from . import widget
from . import axisticks
from . import axis
from ..compat import czip
from .. import qtall as qt
from .. import document
from .. import setting
from .. import utils
from ..helpers import threed

def _(text, disambiguation=None, context='Axis3D'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class _AxisLabels(threed.AxisLabels):
    """For drawing tick labels.

    box1,2: points at corners of graph
    tickfracs: fractions along axis for tick labels
    ticklabels: list of labels to plot at fracs
    ticklabelsprop: font properties for ticks
    axislabel: label for axis
    axislabelposn: position of axis label (or -1 for no label)
    axislabelprop: font properties for axislabel
    """

    def __init__(self, box1, box2, tickfracs, ticklabels, ticklabelsprop,
                 axislabel, axislabelposn, axislabelprop):
        threed.AxisLabels.__init__(self, box1, box2, tickfracs, axislabelposn)
        self.ticklabels = ticklabels
        self.ticklabelsprop = ticklabelsprop
        self.axislabel = axislabel
        self.axislabelposn = axislabelposn
        self.axislabelprop = axislabelprop

    def drawLabel(self, painter, index, pt, ax1, ax2, axangle):
        """Called when scene wants to paint label."""

        painter.translate(pt.x(), pt.y())

        # angle of axis inclination
        angle = math.atan2(ax2.y()-ax1.y(), ax2.x()-ax1.x()) * (180/math.pi)

        # vector to axis from graph centre
        cvecy = math.sin(axangle*math.pi/180)
        cvecx = math.cos(axangle*math.pi/180)

        # character upright vector
        avecy = math.sin((angle-90)*math.pi/180)
        avecx = math.cos((angle-90)*math.pi/180)
        dot = cvecx*avecx+cvecy*avecy

        #print('axangle % 7.2f  angle % 7.2f  delta % 7.2f  dot % 7.2f' % (
        #    axangle, angle, axangle+angle, dot), self.ticklabels[index])

        # flip depending on relative direction of label and graph centre
        if dot < 0:
            angle = angle+180
            if angle > 180:
                angle = angle-360

        # flip if upside down
        if angle < -90 or angle > 90:
            angle = angle+180
            valign = 1
        else:
            valign = -1

        painter.rotate(angle)

        if index >= 0:
            self.drawTickLabel(painter, pt.x(), pt.y(), angle, index, valign)
        else:
            self.drawAxisLabel(painter, valign)

    def drawTickLabel(self, painter, x, y, angle, index, valign):
        """Draw a tick label."""

        font = self.ticklabelsprop.makeQFont(painter)
        painter.setFont(font)

        label = self.ticklabels[index]
        renderer = utils.Renderer(
            painter, font, 0, 0, label,
            alignhorz=0, alignvert=valign,
            usefullheight=True)

        # get text bounds
        rect = renderer.getTightBounds()
        rect.rotateAboutOrigin(angle * math.pi/180)
        rect.translate(x, y)

        # draw text if it doesn't overlap with existing text
        if not painter.textrects.willOverlap(rect):
            pen = self.ticklabelsprop.makeQPen(painter)
            painter.setPen(pen)
            renderer.render()
            painter.textrects.addRect(rect)

    def drawAxisLabel(self, painter, valign):
        """Draw label for axis."""

        # y increment from labels closer to axis
        deltay = 0
        if self.ticklabels:
            font = self.ticklabelsprop.makeQFont(painter)
            fm = utils.FontMetrics(font, painter.device())
            deltay = valign*fm.lineSpacing()

        # change alignment depending on label position
        if self.axislabelposn==0.:
            halign = -1
        elif self.axislabelposn==1.:
            halign = 1
        else:
            halign = 0

        # draw label
        font = self.axislabelprop.makeQFont(painter)
        painter.setFont(font)
        pen = self.axislabelprop.makeQPen(painter)
        painter.setPen(pen)

        renderer = utils.Renderer(
            painter, font, 0, deltay, self.axislabel,
            alignhorz=halign, alignvert=valign,
            usefullheight=True)
        renderer.render()

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

class AxisLabel(setting.Text):
    """Axis label."""

    def __init__(self, name, **args):
        setting.Text.__init__(self, name, **args)
        self.add( setting.Choice(
            'position',
            ('at-minimum', 'centre', 'at-maximum'),
            'centre',
            descr = _('Position of axis label'),
            usertext = _('Position') ) )

class TickLabel(axis.TickLabel):
    """3D axis tick label."""
    def __init__(self, name, **args):
        axis.TickLabel.__init__(self, name, **args)
        self.remove('rotate')
        self.remove('offset')

class Axis3D(widget.Widget):
    """Manages and draws an axis."""

    typename = 'axis3d'
    allowusercreation = True
    description = _('3D axis')
    isaxis = True
    isaxis3d = True

    def __init__(self, parent, name=None):
        """Initialise axis."""

        widget.Widget.__init__(self, parent, name=name)
        s = self.settings

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
        s.add( setting.Str(
            'label', '',
            descr=_('Axis label text'),
            usertext=_('Label')) )
        s.add( setting.AxisBound(
            'min', 'Auto',
            descr=_('Minimum value of axis'),
            usertext=_('Min')) )
        s.add( setting.AxisBound(
            'max', 'Auto',
            descr=_('Maximum value of axis'),
            usertext=_('Max')) )
        s.add( setting.Bool(
            'log', False,
            descr = _('Whether axis is logarithmic'),
            usertext=_('Log')) )
        s.add( axis.AutoRange(
            'autoRange', 'next-tick') )
        s.add( setting.Choice(
            'mode',
            ('numeric', 'datetime', 'labels'),
            'numeric',
            descr = _('Type of ticks to show on on axis'),
            usertext=_('Mode')) )

        s.add( setting.Bool(
            'autoMirror', True,
            descr = _('Place axis on opposite side of graph '
                      'if none'),
            usertext=_('Auto mirror'),
            formatting=True) )

        s.add( setting.Float(
            'datascale', 1.,
            descr=_('Scale data plotted by this factor'),
            usertext=_('Scale')) )

        s.add( setting.Choice(
            'direction',
            ['x', 'y', 'z'],
            'x',
            descr = _('Direction of axis'),
            usertext=_('Direction')) )
        s.add( setting.Float(
            'lowerPosition', 0.,
            descr=_('Fractional position of lower end of '
                    'axis on graph'),
            usertext=_('Min position')) )
        s.add( setting.Float(
            'upperPosition', 1.,
            descr=_('Fractional position of upper end of '
                    'axis on graph'),
            usertext=_('Max position')) )
        s.add( setting.Float(
            'otherPosition1', 0.,
            descr=_('Fractional position of axis '
                    'in its perpendicular direction 1'),
            usertext=_('Axis position 1')) )
        s.add( setting.Float(
            'otherPosition2', 0.,
            descr=_('Fractional position of axis '
                    'in its perpendicular direction 2'),
            usertext=_('Axis position 2')) )

        s.add( setting.Line3D(
            'Line',
            descr = _('Axis line settings'),
            usertext = _('Axis line')),
               pixmap='settings_axisline' )
        s.add( AxisLabel(
            'Label',
            descr = _('Axis label settings'),
            usertext = _('Axis label')),
               pixmap='settings_axislabel' )
        s.add( TickLabel(
            'TickLabels',
            descr = _('Tick label settings'),
            usertext = _('Tick labels')),
               pixmap='settings_axisticklabels' )
        s.add( MajorTick(
            'MajorTicks',
            descr = _('Major tick line settings'),
            usertext = _('Major ticks')),
               pixmap='settings_axismajorticks' )
        s.add( MinorTick(
            'MinorTicks',
            descr = _('Minor tick line settings'),
            usertext = _('Minor ticks')),
               pixmap='settings_axisminorticks' )
        s.add( GridLine(
            'GridLines',
            descr = _('Grid line settings'),
            usertext = _('Grid lines')),
               pixmap='settings_axisgridlines' )
        s.add( MinorGridLine(
            'MinorGridLines',
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

        s.get('autoRange').adjustPlottedRange(
            self.plottedrange, s.min=='Auto', s.max=='Auto', s.log, self.document)

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

    def dataToLogicalCoords(self, vals, scaling=True):
        """Compute coordinates on graph to logical graph coordinates (0..1)

        If scaling is True, apply scaling factor for data
        """

        self.computePlottedRange()
        s = self.settings

        svals = vals*s.datascale if scaling else vals
        if s.log:
            fracposns = self.logConvertToPlotter(svals)
        else:
            fracposns = self.linearConvertToPlotter(svals)

        lower, upper = s.lowerPosition, s.upperPosition
        return lower + fracposns*(upper-lower)

    def linearConvertToPlotter(self, v):
        """Convert graph coordinates to 0..1 coordinates"""
        return ( (v - self.plottedrange[0]) /
                 (self.plottedrange[1] - self.plottedrange[0]) )

    def logConvertToPlotter(self, v):
        """Convert graph coordinates to 0..1 coordinates"""
        log1 = N.log(self.plottedrange[0])
        log2 = N.log(self.plottedrange[1])
        return (N.log(N.clip(v, 1e-99, 1e99)) - log1) / (log2 - log1)

    def transformToAxis(self, v):
        """Return value to give equal-spaced values in transformed coordinates."""
        x = v*self.settings.datascale
        return N.log(x) if self.settings.log else x

    def transformFromAxis(self, v):
        """Convert transformed values back."""
        x = v/self.settings.datascale
        return N.exp(x) if self.settings.log else x

    def getAutoMirrorCombs(self):
        """Get combinations of other position for auto mirroring."""
        s = self.settings
        op1 = s.otherPosition1
        op2 = s.otherPosition2
        if not s.autoMirror:
            return ((op1, op2),)
        if op1 == 0 or op1 == 1:
            op1list = [1, 0]
        else:
            op1list = [op1]
        if op2 == 0 or op2 == 1:
            op2list = [1, 0]
        else:
            op2list = [op2]
        return itertools.product(op1list, op2list)

    def addAxisLine(self, painter, cont, dirn):
        """Build list of lines to draw axis line, mirroring if necessary.

        Returns list of start and end points of axis lines
        """

        s = self.settings
        lower, upper = s.lowerPosition, s.upperPosition

        outstart = []
        outend = []
        for op1, op2 in self.getAutoMirrorCombs():
            if dirn == 'x':
                outstart += [(lower, op1, op2)]
                outend += [(upper, op1, op2)]
            elif dirn == 'y':
                outstart += [(op1, lower, op2)]
                outend += [(op1, upper, op2)]
            else:
                outstart += [(op1, op2, lower)]
                outend += [(op1, op2, upper)]

        if not s.Line.hide:
            startpts = threed.ValVector(N.ravel(outstart))
            endpts = threed.ValVector(N.ravel(outend))
            lineprop = s.Line.makeLineProp(painter)
            cont.addObject(threed.LineSegments(startpts, endpts, lineprop))

        return list(zip(outstart, outend))

    def addLabels(self, cont, linecoords, ticklabelsprop, tickfracs, tickvals,
                  axislabel, axislabelprop):
        """Make tick labels for axis."""

        if not ticklabelsprop.hide:
            # make strings for labels
            fmt = ticklabelsprop.format
            if fmt.lower() == 'auto':
                fmt = self.autoformat
            scale = ticklabelsprop.scale
            ticklabels = [
                utils.formatNumber(v*scale, fmt, locale=self.document.locale)
                for v in tickvals ]
        else:
            return

        # disable drawing of label
        if axislabelprop.hide:
            axislabel = ""
            axislabelposn = -1
        else:
            axislabelposn = {
                'at-minimum': 0,
                'centre': 0.5,
                'at-maximum': 1}[axislabelprop.position]

        # this is to get the z ordering of the ends of axes correct
        # where two axes join together
        tf = N.array(tickfracs)
        tf[tf==0.] = 0.00001
        tf[tf==1.] = 0.99999
        atl = _AxisLabels(
            threed.Vec3(0,0,0), threed.Vec3(1,1,1),
            threed.ValVector(tf), ticklabels, ticklabelsprop,
            axislabel, axislabelposn, axislabelprop)

        for startpos, endpos in linecoords:
            atl.addAxisChoice(threed.Vec3(*startpos), threed.Vec3(*endpos))
        cont.addObject(atl)

    def addAxisTicks(self, painter, cont, dirn, linecoords, tickprops,
                     ticklabelsprop, tickvals):
        """Add ticks for the vals and tick properties class given.
        linecoords: coordinates of start and end points of lines
        labelprops: properties of label, or None
        cont: container to add ticks
        dirn: 'x', 'y', 'z' for axis
        """

        ticklen = tickprops.length * 1e-3
        tfracs = self.dataToLogicalCoords(tickvals, scaling=False)

        outstart = []
        outend = []
        for op1, op2 in self.getAutoMirrorCombs():
            # where to draw tick from
            op1pts = N.full_like(tfracs, op1)
            op2pts = N.full_like(tfracs, op2)
            # where to draw tick to
            op1pts2 = N.full_like(tfracs, op1+ticklen*(1 if op1 < 0.5 else -1))
            op2pts2 = N.full_like(tfracs, op2+ticklen*(1 if op2 < 0.5 else -1))

            # swap coordinates depending on axis direction
            if dirn == 'x':
                ptsonaxis = (tfracs, op1pts, op2pts)
                ptsoff1 = (tfracs, op1pts2, op2pts)
                ptsoff2 = (tfracs, op1pts, op2pts2)
            elif dirn == 'y':
                ptsonaxis = (op1pts, tfracs, op2pts)
                ptsoff1 = (op1pts2, tfracs, op2pts)
                ptsoff2 = (op1pts, tfracs, op2pts2)
            else:
                ptsonaxis = (op1pts, op2pts, tfracs)
                ptsoff1 = (op1pts2, op2pts, tfracs)
                ptsoff2 = (op1pts, op2pts2, tfracs)

            outstart += [N.ravel(N.column_stack(ptsonaxis)),
                         N.ravel(N.column_stack(ptsonaxis))]
            outend += [N.ravel(N.column_stack(ptsoff1)),
                       N.ravel(N.column_stack(ptsoff2))]

        # add labels for ticks and axis label
        if ticklabelsprop is not None:
            self.addLabels(
                cont, linecoords, ticklabelsprop, tfracs, tickvals,
                self.settings.label, self.settings.Label)

        # add ticks themselves
        if not tickprops.hide:
            startpts = threed.ValVector(N.concatenate(outstart))
            endpts = threed.ValVector(N.concatenate(outend))
            lineprop = tickprops.makeLineProp(painter)
            cont.addObject(threed.LineSegments(startpts, endpts, lineprop))

    def addGridLines(self, painter, cont, dirn, linecoords, gridprops, tickvals):
        """Add ticks for the vals and tick properties class given.
        linecoords: coordinates of start and end points of lines
        cont: container to add ticks
        dirn: 'x', 'y', 'z' for axis
        """

        if gridprops.hide:
            return

        tfracs = self.dataToLogicalCoords(tickvals, scaling=False)
        ones = N.ones(tfracs.shape)
        zeros = N.zeros(tfracs.shape)

        outstart = []
        outend = []

        # positions of grid lines for x axis
        pts1 = [
            (tfracs, zeros, zeros),
            (tfracs, zeros, zeros),
            (tfracs, ones, ones),
            (tfracs, ones, ones)
        ]
        pts2 = [
            (tfracs, zeros, ones),
            (tfracs, ones, zeros),
            (tfracs, zeros, ones),
            (tfracs, ones, zeros)
        ]
        # norm for each face, so we can draw back of cube only
        norms = [
            (0,  1,  0),
            (0,  0,  1),
            (0,  0, -1),
            (0, -1,  0)
        ]

        # swap coordinates for other axes
        if dirn == 'y':
            pts1  = [(c,a,b) for a,b,c in pts1]
            pts2  = [(c,a,b) for a,b,c in pts2]
            norms = [(c,a,b) for a,b,c in norms]
        elif dirn == 'z':
            pts1  = [(b,c,a) for a,b,c in pts1]
            pts2  = [(b,c,a) for a,b,c in pts2]
            norms = [(b,c,a) for a,b,c in norms]

        # add lines on each face
        lineprop = gridprops.makeLineProp(painter)
        for p1, p2, n in czip(pts1, pts2, norms):
            # container only shows face if norm points to observer
            face = threed.FacingContainer(threed.Vec3(*n))
            c1 = threed.ValVector(N.ravel(N.column_stack(p1)))
            c2 = threed.ValVector(N.ravel(N.column_stack(p2)))
            face.addObject(threed.LineSegments(c1, c2, lineprop))
            cont.addObject(face)

    def drawToObject(self, painter, painthelper):

        self.computePlottedRange()

        s = self.settings
        dirn = s.direction

        cont = threed.ObjectContainer()

        linecoords = self.addAxisLine(painter, cont, dirn)
        self.addAxisTicks(
            painter, cont, dirn, linecoords, s.MajorTicks, s.TickLabels,
            self.majortickscalc)
        self.addAxisTicks(
            painter, cont, dirn, linecoords, s.MinorTicks, None,
            self.minortickscalc)

        self.addGridLines(
            painter, cont, dirn, linecoords, s.GridLines,
            self.majortickscalc)
        self.addGridLines(
            painter, cont, dirn, linecoords, s.MinorGridLines,
            self.minortickscalc)

        cont.assignWidgetId(id(self))
        return cont

# allow the factory to instantiate an axis
document.thefactory.register(Axis3D)
