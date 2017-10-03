# -*- coding: utf-8 -*-

#    Copyright (C) 2011 Jeremy S. Sanders
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

"""Ternary plot widget."""

from __future__ import division
import numpy as N
import math

from ..compat import czip
from .nonorthgraph import NonOrthGraph
from .axisticks import AxisTicks
from .axis import MajorTick, MinorTick, GridLine, MinorGridLine, AxisLabel, \
    TickLabel

from .. import qtall as qt4
from .. import document
from .. import setting
from .. import utils

def _(text, disambiguation=None, context='Ternary'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

def rotatePts(x, y, theta):
    '''Rotate points by theta degrees.'''
    s = math.sin(theta*math.pi/180.)
    c = math.cos(theta*math.pi/180.)
    return x*c-y*s, x*s+y*c

# translate coordinates a,b,c from user to plot
# user can select different coordinate systems
coord_lookup = {
    'bottom-left': (0, 1, 2),
    'bottom-right': (0, 2, 1),
    'left-bottom': (1, 0, 2),
    'left-right': (2, 0, 1),
    'right-bottom': (1, 2, 0),
    'right-left': (2, 1, 0)
}

# useful trigonometric identities
sin30 = 0.5
sin60 = cos30 = 0.86602540378
tan30 = 0.5773502691

class Ternary(NonOrthGraph):
    '''Ternary plotter.'''

    typename='ternary'
    allowusercreation = True
    description = _('Ternary graph')

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        NonOrthGraph.addSettings(s)

        s.add( setting.Choice('mode',
                              ('percentage', 'fraction'),
                              'percentage',
                              descr=_('Show percentages or fractions'),
                              usertext=_('Mode')) )

        s.add( setting.Choice('coords',
                              ('bottom-left', 'bottom-right',
                               'left-bottom', 'left-right',
                               'right-left', 'right-bottom'),
                              'bottom-left',
                              descr=_('Axes to use for plotting coordinates'),
                              usertext=_('Coord system')) )

        s.add( setting.Str('labelbottom', '',
                           descr=_('Bottom axis label text'),
                           usertext=_('Label bottom')) )
        s.add( setting.Str('labelleft', '',
                           descr=_('Left axis label text'),
                           usertext=_('Label left')) )
        s.add( setting.Str('labelright', '',
                           descr=_('Right axis label text'),
                           usertext=_('Label right')) )

        s.add( setting.Float('originleft', 0.,
                             descr=_('Fractional origin of left axis at its top'),
                             usertext=_('Left origin')) )
        s.add( setting.Float('originbottom', 0.,
                             descr=_('Fractional origin of bottom axis at its left'),
                             usertext=_('Bottom origin')) )
        s.add( setting.Float('fracsize', 1.,
                             descr=_('Fractional size of plot'),
                             usertext=_('Size')) )

        s.add( setting.Bool('reverse', False,
                            descr=_('Reverse axes'),
                            usertext=_('Reverse')) )

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

        s.get('leftMargin').newDefault('1cm')
        s.get('rightMargin').newDefault('1cm')
        s.get('topMargin').newDefault('1cm')
        s.get('bottomMargin').newDefault('1cm')

        s.MajorTicks.get('number').newDefault(10)
        s.MinorTicks.get('number').newDefault(50)
        s.GridLines.get('hide').newDefault(False)
        s.TickLabels.remove('rotate')

    @property
    def userdescription(self):
        return _("mode='%s'") % self.settings.mode

    def _maxVal(self):
        '''Get maximum value on axis.'''
        if self.settings.mode == 'percentage':
            return 100.
        else:
            return 1.

    def coordRanges(self):
        '''Get ranges of coordinates.'''

        mv = self._maxVal()

        # ranges for each coordinate
        ra = [self._orgbot*mv, (self._orgbot+self._size)*mv]
        rb = [self._orgleft*mv, (self._orgleft+self._size)*mv]
        rc = [self._orgright*mv, (self._orgright+self._size)*mv]
        ranges = [ra, rb, rc]

        lookup = coord_lookup[self.settings.coords]
        return ranges[lookup.index(0)], ranges[lookup.index(1)]

    def graphToPlotCoords(self, coorda, coordb):
        '''Convert coordinates in r, theta to x, y.'''

        s = self.settings

        # normalize coordinates
        maxval = self._maxVal()
        coordan = coorda / maxval
        coordbn = coordb / maxval

        # the three coordinates on the plot
        clist = [coordan, coordbn, 1.-coordan-coordbn]

        # select the right coordinates for a, b and c given the system
        # requested by the user
        # normalise by origins and plot size
        lookup = coord_lookup[s.coords]

        cbot = ( clist[ lookup[0] ] - self._orgbot ) / self._size
        cleft = ( clist[ lookup[1] ] - self._orgleft ) / self._size
        cright = ( clist[ lookup[2] ] - self._orgright ) / self._size

        # from Ingram, 1984, Area, 16, 175
        # remember that y goes in the opposite direction here
        if s.reverse:
            cleft, cright = cright, cleft
            x = (1-(0.5*cright + cbot))*self._width + self._box[0]
        else:
            x = (0.5*cright + cbot)*self._width + self._box[0]

        y = self._box[3] - cright * sin60 * self._width

        return x, y

    def drawFillPts(self, painter, brushext, cliprect, ptsx, ptsy):
        '''Draw points for plotting a fill.'''
        pts = qt4.QPolygonF()
        utils.addNumpyToPolygonF(pts, ptsx, ptsy)

        filltype = brushext.filltype
        # this is broken: FIXME
        if filltype == 'left':
            dyend = ptsy[-1]-self._box[1]
            pts.append( qt4.QPointF(ptsx[-1]-dyend*tan30, self._box[1]) )
            dystart = ptsy[0]-self._box[1]
            pts.append( qt4.QPointF(ptsx[0]-dystart*tan30, self._box[1]) )
        elif filltype == 'right':
            pts.append( qt4.QPointF(self._box[2], ptsy[-1]) )
            pts.append( qt4.QPointF(self._box[2], ptsy[0]) )
        elif filltype == 'bottom':
            dyend = self._box[3]-ptsy[-1]
            pts.append( qt4.QPointF(ptsx[-1]-dyend*tan30, self._box[3]) )
            dystart = self._box[3]-ptsy[0]
            pts.append( qt4.QPointF(ptsx[0]-dystart*tan30, self._box[3]) )
        elif filltype == 'polygon':
            pass
        else:
            pts = None

        if pts is not None:
            utils.brushExtFillPolygon(painter, brushext, cliprect, pts)

    def drawGraph(self, painter, bounds, datarange, outerbounds=None):
        '''Plot graph area and axes.'''

        s = self.settings

        xw, yw = bounds[2]-bounds[0], bounds[3]-bounds[1]

        d60 = 60./180.*math.pi
        ang = math.atan2(yw, xw/2.)
        if ang > d60:
            # taller than wider
            widthh = xw/2
            height = math.tan(d60) * widthh
        else:
            # wider than taller
            height = yw
            widthh = height / math.tan(d60)

        # box for equilateral triangle
        self._box = ( (bounds[2]+bounds[0])/2 - widthh,
                      (bounds[1]+bounds[3])/2 - height/2,
                      (bounds[2]+bounds[0])/2 + widthh,
                      (bounds[1]+bounds[3])/2 + height/2 )
        self._width = widthh*2
        self._height = height

        # triangle shaped polygon for graph
        self._tripoly = p = qt4.QPolygonF()
        p.append( qt4.QPointF(self._box[0], self._box[3]) )
        p.append( qt4.QPointF(self._box[0]+widthh, self._box[1]) )
        p.append( qt4.QPointF(self._box[2], self._box[3]) )

        path = qt4.QPainterPath()
        path.addPolygon(p)
        path.closeSubpath()
        utils.brushExtFillPath(painter, s.Background, path,
                               stroke=s.Border.makeQPenWHide(painter))

        # work out origins and size
        self._size = max(min(s.fracsize, 1.), 0.)

        # make sure we don't go past the ends of the allowed range
        # value of origin of left axis at top
        self._orgleft = min(s.originleft, 1.-self._size)
        # value of origin of bottom axis at left
        self._orgbot = min(s.originbottom, 1.-self._size)
        # origin of right axis at bottom
        self._orgright = 1. - self._orgleft - (self._orgbot + self._size)

    def _computeTickVals(self):
        """Compute tick values."""

        s = self.settings

        # this is a hack as we lose ends off the axis otherwise
        d = 1e-6

        # get ticks along left axis
        atickleft = AxisTicks(self._orgleft-d, self._orgleft+self._size+d,
                              s.MajorTicks.number, s.MinorTicks.number,
                              extendmin=False, extendmax=False)
        atickleft.getTicks()
        # use the interval from above to calculate ticks for right
        atickright = AxisTicks(self._orgright-d, self._orgright+self._size+d,
                               s.MajorTicks.number, s.MinorTicks.number,
                               extendmin=False, extendmax=False,
                               forceinterval = atickleft.interval)
        atickright.getTicks()
        # then calculate for bottom
        atickbot = AxisTicks(self._orgbot-d, self._orgbot+self._size+d,
                             s.MajorTicks.number, s.MinorTicks.number,
                             extendmin=False, extendmax=False,
                             forceinterval = atickleft.interval)
        atickbot.getTicks()

        return atickbot, atickleft, atickright

    def setClip(self, painter, bounds):
        '''Set clipping for graph.'''
        p = qt4.QPainterPath()
        p.addPolygon( self._tripoly )
        painter.setClipPath(p)

    def _getLabels(self, ticks, autoformat):
        """Return tick labels."""
        labels = []
        tl = self.settings.TickLabels
        format = tl.format
        scale = tl.scale
        if format.lower() == 'auto':
            format = autoformat
        for v in ticks:
            l = utils.formatNumber(v*scale, format, locale=self.document.locale)
            labels.append(l)
        if self.settings.reverse:
            labels = labels[::-1]
        return labels

    def _drawTickSet(self, painter, tickSetn, gridSetn,
                     tickbot, tickleft, tickright,
                     tickLabelSetn=None, labelSetn=None):
        '''Draw a set of ticks (major or minor).

        tickSetn: tick setting to get line details
        gridSetn: setting for grid line (if any)
        tickXXX: tick arrays for each axis
        tickLabelSetn: setting used to label ticks, or None if minor ticks
        labelSetn: setting for labels, if any
        '''

        # this is mostly a lot of annoying trigonometry
        # compute line ends for ticks and grid lines

        tl = tickSetn.get('length').convert(painter)
        mv = self._maxVal()

        reverse = bool(self.settings.reverse)
        revsign = [1, -1][reverse]

        # bottom ticks
        x1 = (tickbot - self._orgbot)/self._size*self._width + self._box[0]
        y1 = self._box[3] + N.zeros(x1.shape)
        x2 = x1 - revsign * tl * sin30
        y2 = y1 + tl * cos30
        tickbotline = (x1, y1, x2, y2)

        # bottom grid (removing lines at edge of plot)
        scaletick = 1 - (tickbot-self._orgbot)/self._size
        gx = x1 + scaletick*self._width*sin30
        gy = y1 - scaletick*self._width*cos30
        ne = (scaletick > 1e-6) & (scaletick < (1-1e-6))
        gridbotline = (x1[ne], y1[ne], gx[ne], gy[ne])

        # left ticks
        x1 = -(tickleft - self._orgleft)/self._size*self._width*sin30 + (
            self._box[0] + self._box[2])*0.5
        y1 = (tickleft - self._orgleft)/self._size*self._width*cos30 + self._box[1]
        if reverse:
            x2 = x1 - tl
            y2 = y1
        else:
            x2 = x1 - tl * sin30
            y2 = y1 - tl * cos30
        tickleftline = (x1, y1, x2, y2)

        # left grid
        scaletick = 1 - (tickleft-self._orgleft)/self._size
        gx = x1 + scaletick*self._width*sin30
        gy = self._box[3] + N.zeros(y1.shape)
        ne = (scaletick > 1e-6) & (scaletick < (1-1e-6))
        gridleftline = (x1[ne], y1[ne], gx[ne], gy[ne])

        # right ticks
        x1 = -(tickright - self._orgright)/self._size*self._width*sin30+self._box[2]
        y1 = -(tickright - self._orgright)/self._size*self._width*cos30+self._box[3]
        if reverse:
            x2 = x1 + tl * sin30
            y2 = y1 - tl * cos30
        else:
            x2 = x1 + tl
            y2 = y1
        tickrightline = (x1, y1, x2, y2)

        # right grid
        scaletick = 1 - (tickright-self._orgright)/self._size
        gx = x1 - scaletick*self._width
        gy = y1
        gridrightline = (x1[ne], y1[ne], gx[ne], gy[ne])

        if not gridSetn.hide:
            # draw the grid
            pen = gridSetn.makeQPen(painter)
            painter.setPen(pen)
            utils.plotLinesToPainter(painter, *gridbotline)
            utils.plotLinesToPainter(painter, *gridleftline)
            utils.plotLinesToPainter(painter, *gridrightline)

        # calculate deltas for ticks
        bdelta = ldelta = rdelta = 0

        if not tickSetn.hide:
            # draw ticks themselves
            pen = tickSetn.makeQPen(painter)
            pen.setCapStyle(qt4.Qt.FlatCap)
            painter.setPen(pen)
            utils.plotLinesToPainter(painter, *tickbotline)
            utils.plotLinesToPainter(painter, *tickleftline)
            utils.plotLinesToPainter(painter, *tickrightline)

            ldelta += tl*sin30
            bdelta += tl*cos30
            rdelta += tl

        if tickLabelSetn is not None and not tickLabelSetn.hide:
            # compute the labels for the ticks
            tleftlabels = self._getLabels(tickleft*mv, '%Vg')
            trightlabels = self._getLabels(tickright*mv, '%Vg')
            tbotlabels = self._getLabels(tickbot*mv, '%Vg')

            painter.setPen( tickLabelSetn.makeQPen(painter) )
            font = tickLabelSetn.makeQFont(painter)
            painter.setFont(font)

            fm = utils.FontMetrics(font, painter.device())
            sp = fm.leading() + fm.descent()
            off = tickLabelSetn.get('offset').convert(painter)

            # draw tick labels in each direction
            hlabbot = wlableft = wlabright = 0
            for l, x, y in czip(tbotlabels, tickbotline[2], tickbotline[3]+off):
                r = utils.Renderer(painter, font, x, y, l, 0, 1, 0, doc=self.document)
                bounds = r.render()
                hlabbot = max(hlabbot, bounds[3]-bounds[1])
            for l, x, y in czip(tleftlabels, tickleftline[2]-off-sp, tickleftline[3]):
                r = utils.Renderer(painter, font, x, y, l, 1, 0, 0, doc=self.document)
                bounds = r.render()
                wlableft = max(wlableft, bounds[2]-bounds[0])
            for l, x, y in czip(trightlabels,tickrightline[2]+off+sp, tickrightline[3]):
                r = utils.Renderer(painter, font, x, y, l, -1, 0, 0, doc=self.document)
                bounds = r.render()
                wlabright = max(wlabright, bounds[2]-bounds[0])

            bdelta += hlabbot+off+sp
            ldelta += wlableft+off+sp
            rdelta += wlabright+off+sp

        if labelSetn is not None and not labelSetn.hide:
            # draw label on edges (if requested)
            painter.setPen( labelSetn.makeQPen(painter) )
            font = labelSetn.makeQFont(painter)
            painter.setFont(font)

            fm = utils.FontMetrics(font, painter.device())
            sp = fm.leading() + fm.descent()
            off = labelSetn.get('offset').convert(painter)

            # alignment
            align = {'at-minimum': -1, 'centre': 0, 'at-maximum': 1}[
                labelSetn.position]
            aoffset = {'at-minimum': -self._width/2,
                       'centre': 0,
                       'at-maximum': self._width/2}[labelSetn.position]

            # bottom label
            r = utils.Renderer(
                painter, font, 0, 0,
                self.settings.labelbottom,
                align, 1,
                doc=self.document)
            painter.save()
            painter.translate(self._box[0]+self._width/2,
                              self._box[3] + bdelta + off)
            painter.translate(aoffset, 0)
            r.render()
            painter.restore()

            # left label - rotate frame before drawing so we can get
            # the bounds correct
            r = utils.Renderer(
                painter, font, 0, -sp,
                self.settings.labelleft,
                -align, -1,
                doc=self.document)

            painter.save()
            painter.translate(self._box[0]+self._width*0.25,
                              0.5*(self._box[1]+self._box[3]))
            painter.rotate(-60)
            painter.translate(-aoffset, -ldelta - off)
            r.render()
            painter.restore()

            # right label
            r = utils.Renderer(
                painter, font, 0, -sp,
                self.settings.labelright,
                -align, -1,
                doc=self.document)

            painter.save()
            painter.translate(self._box[0]+self._width*0.75 ,
                              0.5*(self._box[1]+self._box[3]) )
            painter.rotate(60)
            painter.translate(-aoffset, -ldelta - off)
            r.render()
            painter.restore()

    def drawAxes(self, painter, bounds, datarange, outerbounds=None):
        '''Draw plot axes.'''

        s = self.settings

        # compute tick values for later when plotting axes
        tbot, tleft, tright = self._computeTickVals()

        # draw the major ticks
        self._drawTickSet(painter, s.MajorTicks, s.GridLines,
                          tbot.tickvals, tleft.tickvals, tright.tickvals,
                          tickLabelSetn=s.TickLabels,
                          labelSetn=s.Label)

        # now draw the minor ones
        self._drawTickSet(painter, s.MinorTicks, s.MinorGridLines,
                          tbot.minorticks, tleft.minorticks, tright.minorticks)

document.thefactory.register(Ternary)
