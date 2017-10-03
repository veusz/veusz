# -*- coding: utf-8 -*-

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

"""Polar plot widget."""

from __future__ import division
import numpy as N
import fractions
import math

from .nonorthgraph import NonOrthGraph
from .axisticks import AxisTicks
from . import axis

from ..compat import crange
from .. import qtall as qt4
from .. import document
from .. import setting
from .. import utils

def _(text, disambiguation=None, context='Polar'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

def radianToAlign(a):
    # fraction of circle
    f = a / (2*math.pi)

    # slow, but ok here as we don't expect values much outside range
    while f < 0:
        f += 1
    while f >= 1:
        f -= 1

    if f == 0:
        return (-1, 1)
    elif f < 0.25:
        return (-1, 1)
    elif f == 0.25:
        return (0, 1)
    elif f < 0.5:
        return (1, 1)
    elif f == 0.5:
        return (1, 0)
    elif f < 0.75:
        return (1, -1)
    elif f == 0.75:
        return (0, -1)
    else:
        return (-1, -1)

class OldLine(setting.Line):
    '''Polar tick settings.'''

    def __init__(self, name):
        setting.Line.__init__(self, name, setnsmode='hide')

        self.add( setting.SettingBackwardCompat(
            'number', '../RadiiLine/number', None))
        self.add( setting.SettingBackwardCompat(
            'hidespokes', '../SpokeLine/length', None))
        self.add( setting.SettingBackwardCompat(
            'hideannuli', '../SpokeAnnuli/length', None))

class SpokeLine(setting.Line):
    '''Spokes in polar plot.'''

    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)
        self.add( setting.Int(
            'number', 12,
            minval=1,
            descr = _('Number of spokes to use'),
            usertext=_('Number')) )
        self.get('color').newDefault('grey')

class RadiiLine(setting.Line):
    '''Radii in polar plot.'''

    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)
        self.add( setting.Int(
            'number', 6,
            minval=1,
            descr = _('Number of radial ticks to aim for'),
            usertext=_('Number')) )
        self.get('color').newDefault('grey')

class TickLabel(axis.TickLabel):
    """For tick label."""
    def __init__(self, *args, **argsv):
        axis.TickLabel.__init__(self, *args, **argsv)
        self.remove('offset')
        self.remove('rotate')
        self.remove('hide')
        self.add( setting.Bool(
            'hideradial', False,
            descr = _('Hide radial labels'),
            usertext=_('Hide radial') ) )
        self.add( setting.Bool(
            'hidetangential', False,
            descr = _('Hide spoke labels'),
            usertext=_('Hide spokes') ) )

class Polar(NonOrthGraph):
    '''Polar plotter.'''

    typename='polar'
    allowusercreation = True
    description = _('Polar graph')

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        NonOrthGraph.addSettings(s)

        s.add( setting.FloatOrAuto(
            'minradius', 'Auto',
            descr=_('Minimum value of radius'),
            usertext=_('Min radius')) )
        s.add( setting.FloatOrAuto(
            'maxradius', 'Auto',
            descr=_('Maximum value of radius'),
            usertext=_('Max radius')) )
        s.add( setting.Choice(
            'units',
            ('degrees', 'radians', 'fractions', 'percentages'),
            'degrees',
            descr = _('Angular units'),
            usertext=_('Units')) )
        s.add( setting.Choice(
            'direction',
            ('clockwise', 'anticlockwise'),
            'anticlockwise',
            descr = _('Angle direction'),
            usertext = _('Direction')) )
        s.add( setting.Choice(
            'position0',
            ('right', 'top', 'left', 'bottom'),
            'right',
            descr = _('Direction of 0 angle'),
            usertext = _(u'Position of 0°')) )
        s.add( setting.Bool(
            'log', False,
            descr = _('Logarithmic radial axis'),
            usertext = _('Log')) )

        s.add( TickLabel(
            'TickLabels',
            descr = _('Radial tick labels'),
            usertext=_('Radial tick labels')),
               pixmap='settings_axisticklabels' )

        s.add( OldLine('Tick') )

        s.add( SpokeLine(
            'SpokeLine',
            descr = _('Spoke line'),
            usertext=_('Spoke line')),
               pixmap='settings_axismajorticks' )

        s.add( RadiiLine(
            'RadiiLine',
            descr = _('Radii line'),
            usertext=_('Radii line')),
               pixmap='settings_contourline' )

        s.get('leftMargin').newDefault('1cm')
        s.get('rightMargin').newDefault('1cm')
        s.get('topMargin').newDefault('1cm')
        s.get('bottomMargin').newDefault('1cm')

    @property
    def userdescription(self):
        s = self.settings
        return _("'units=%s, direction=%s, log=%s") % (
            s.units, s.direction, str(s.log))

    def coordRanges(self):
        '''Get ranges of coordinates.'''

        angularrange = {
            'degrees': [0., 360.],
            'radians': [0., 2*math.pi],
            'fractions': [0., 1.],
            'percentages': [0., 100.],
        }[self.settings.units]

        return [
            [self._minradius, self._maxradius],
            angularrange
            ]

    def toPlotAngle(self, angles):
        """Convert one or more angles to angle on plot."""
        s = self.settings

        # unit conversion
        if s.units == 'degrees':
            angles = angles * (math.pi/180.)
        # change direction
        if self.settings.direction == 'anticlockwise':
            angles = -angles
        # add offset
        angles -= {'right': 0, 'top': 0.5*math.pi, 'left': math.pi,
                   'bottom': 1.5*math.pi}[self.settings.position0]
        return angles

    def toPlotRadius(self, radii):
        """Convert radii to a plot radii."""
        if self.settings.log:
            logmin = N.log(self._minradius)
            logmax = N.log(self._maxradius)
            r = ( N.log(N.clip(radii, 1e-99, 1e99)) - logmin ) / (
                logmax - logmin)
        else:
            r = (radii - self._minradius) / (
                self._maxradius - self._minradius)
        return N.where(r > 0., r, 0.)

    def graphToPlotCoords(self, coorda, coordb):
        '''Convert coordinates in r, theta to x, y.'''

        ca = self.toPlotRadius(coorda)
        cb = self.toPlotAngle(coordb)

        x = self._xc + ca * N.cos(cb) * self._xscale
        y = self._yc + ca * N.sin(cb) * self._yscale
        return x, y

    def drawFillPts(self, painter, extfill, cliprect,
                    ptsx, ptsy):
        '''Draw points for plotting a fill.'''
        pts = qt4.QPolygonF()
        utils.addNumpyToPolygonF(pts, ptsx, ptsy)

        filltype = extfill.filltype
        if filltype == 'center':
            pts.append( qt4.QPointF(self._xc, self._yc) )
            utils.brushExtFillPolygon(painter, extfill, cliprect, pts)
        elif filltype == 'outside':
            pp = qt4.QPainterPath()
            pp.moveTo(self._xc, self._yc)
            pp.arcTo(cliprect, 0, 360)
            pp.addPolygon(pts)
            utils.brushExtFillPath(painter, extfill, pp)
        elif filltype == 'polygon':
            utils.brushExtFillPolygon(painter, extfill, cliprect, pts)

    def drawGraph(self, painter, bounds, datarange, outerbounds=None):
        '''Plot graph area and axes.'''

        s = self.settings

        if datarange is None:
            datarange = [0., 1., 0., 1.]

        if s.maxradius == 'Auto':
            self._maxradius = datarange[1]
        else:
            self._maxradius = s.maxradius

        if s.minradius == 'Auto':
            if s.log:
                if datarange[0] > 0.:
                    self._minradius = datarange[0]
                else:
                    self._minradius = self._maxradius / 100.
            else:
                if datarange[0] >= 0:
                    self._minradius = 0.
                else:
                    self._minradius = datarange[0]
        else:
            self._minradius = s.minradius

        # stop negative values
        if s.log:
            self._minradius = N.clip(self._minradius, 1e-99, 1e99)
            self._maxradius = N.clip(self._maxradius, 1e-99, 1e99)
        if self._minradius == self._maxradius:
            self._maxradius = self._minradius + 1

        self._xscale = (bounds[2]-bounds[0])*0.5
        self._yscale = (bounds[3]-bounds[1])*0.5
        self._xc = 0.5*(bounds[0]+bounds[2])
        self._yc = 0.5*(bounds[3]+bounds[1])

        path = qt4.QPainterPath()
        path.addEllipse( qt4.QRectF( qt4.QPointF(bounds[0], bounds[1]),
                                     qt4.QPointF(bounds[2], bounds[3]) ) )
        utils.brushExtFillPath(painter, s.Background, path,
                               stroke=s.Border.makeQPenWHide(painter))

    def setClip(self, painter, bounds):
        '''Set clipping for graph.'''
        p = qt4.QPainterPath()
        p.addEllipse( qt4.QRectF( qt4.QPointF(bounds[0], bounds[1]),
                                  qt4.QPointF(bounds[2], bounds[3]) ) )
        painter.setClipPath(p)

    def drawAxes(self, painter, bounds, datarange, outerbounds=None):
        '''Plot axes.'''

        s = self.settings

        spokesL = s.SpokeLine
        radiiL = s.RadiiLine

        # handle reversed axes using min and max below
        r = [self._minradius, self._maxradius]
        atick = AxisTicks(
            min(r), max(r),
            radiiL.number, radiiL.number*4,
            extendmin=False, extendmax=False,
            logaxis=s.log)
        atick.getTicks()
        majtick = atick.tickvals

        # drop 0 at origin
        if self._minradius == 0. and not s.log:
            majtick = majtick[1:]

        # pen for radii circles and axis
        painter.setPen( radiiL.makeQPenWHide(painter) )
        painter.setBrush( qt4.QBrush() )

        # draw ticks as circles
        if not radiiL.hide:
            for tick in majtick:
                radius = self.toPlotRadius(tick)
                if radius > 0:
                    rect = qt4.QRectF(
                        qt4.QPointF(
                            self._xc - radius*self._xscale,
                            self._yc - radius*self._yscale ),
                        qt4.QPointF(
                            self._xc + radius*self._xscale,
                            self._yc + radius*self._yscale ) )
                    painter.drawEllipse(rect)

        # setup axes plot
        tl = s.TickLabels
        scale, fmt = tl.scale, tl.format
        if fmt == 'Auto':
            fmt = atick.autoformat
        painter.setPen( tl.makeQPen(painter) )
        font = tl.makeQFont(painter)

        # draw ticks
        if not s.TickLabels.hideradial:
            for tick in majtick:
                num = utils.formatNumber(
                    tick*scale, fmt,locale=self.document.locale)
                x = self.toPlotRadius(tick) * self._xscale + self._xc
                r = utils.Renderer(
                    painter, font, x, self._yc, num,
                    alignhorz=-1,
                    alignvert=-1, usefullheight=True,
                    doc=self.document)
                r.render()

        numspokes = spokesL.number
        if s.units == 'degrees':
            vals = N.linspace(0., 360., numspokes+1)[:-1]
            labels = [u'%g°' % x for x in vals]
        elif s.units == 'radians':
            labels = []
            for i in crange(numspokes):
                # use fraction module to work out labels
                # angle in radians (/pi)
                f = fractions.Fraction(2*i, numspokes)
                if f.numerator == 0:
                    txt = '0'
                else:
                    txt = u'%iπ/%i' % (f.numerator, f.denominator)
                    # remove superfluous 1* or /1
                    if txt[-2:] == '/1': txt = txt[:-2]
                    txt = txt.lstrip('1')
                labels.append(txt)
        elif s.units == 'fractions':
            labels = []
            for i in crange(numspokes):
                val = i/numspokes
                label = ('%.2f' % val).rstrip('0').rstrip('.')
                labels.append(label)
        elif s.units == 'percentages':
            labels = []
            for i in crange(numspokes):
                txt = '%.1f' % (100*i/numspokes)
                if txt[-2:] == '.0': txt = txt[:-2]
                labels.append(txt)
        else:
            raise RuntimeError()

        if s.direction == 'anticlockwise':
            labels = labels[0:1] + labels[1:][::-1]

        # these are the angles the spokes lie in
        # (by default 0 is to the right)
        angles = 2 * math.pi * N.arange(numspokes) / numspokes

        # rotate labels if zero not at right
        if s.position0 == 'top':
            angles -= math.pi/2
        elif s.position0 == 'left':
            angles += math.pi
        elif s.position0 == 'bottom':
            angles += math.pi/2

        # draw labels around plot
        if not s.TickLabels.hidetangential:
            for angle, label in zip(angles, labels):
                align = radianToAlign(angle)
                x = self._xc +  N.cos(angle) * self._xscale
                y = self._yc +  N.sin(angle) * self._yscale
                r = utils.Renderer(
                    painter, font, x, y, label,
                    alignhorz=align[0],
                    alignvert=align[1],
                    usefullheight=True)
                r.render()

        # draw spokes
        if not spokesL.hide:
            painter.setPen( spokesL.makeQPenWHide(painter) )
            painter.setBrush( qt4.QBrush() )
            angle = 2 * math.pi / numspokes
            lines = []
            for i in crange(numspokes):
                x = self._xc +  N.cos(angle*i) * self._xscale
                y = self._yc +  N.sin(angle*i) * self._yscale
                lines.append(
                    qt4.QLineF(
                        qt4.QPointF(self._xc, self._yc), qt4.QPointF(x, y)) )
            painter.drawLines(lines)

document.thefactory.register(Polar)
