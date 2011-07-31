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

import numpy as N
import math
from itertools import izip

from nonorthgraph import NonOrthGraph
from axisticks import AxisTicks
from axis import MajorTick, MinorTick, GridLine, MinorGridLine, AxisLabel, \
    TickLabel

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

def rotatePts(x, y, theta):
    '''Rotate points by theta degrees.'''
    s = math.sin(theta*math.pi/180.)
    c = math.cos(theta*math.pi/180.)
    return x*c-y*s, x*s+y*c

class Ternary(NonOrthGraph):
    '''Ternary plotter.'''

    typename='ternary'
    allowusercreation = True
    description = 'Ternary graph'

    def __init__(self, parent, name=None):
        '''Initialise ternary plot.'''
        NonOrthGraph.__init__(self, parent, name=name)
        if type(self) == NonOrthGraph:
            self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        NonOrthGraph.addSettings(s)

        s.add( setting.Choice('mode',
                              ('percentage', 'fraction'),
                              'percentage',
                              descr='Show percentages or fractions',
                              usertext='Mode') )
        s.add( setting.Str('labelleft', '',
                           descr='Left axis label text',
                           usertext='Label left') )
        s.add( setting.Str('labelright', '',
                           descr='Right axis label text',
                           usertext='Label right') )
        s.add( setting.Str('labelbottom', '',
                           descr='Bottom axis label text',
                           usertext='Label bottom') )
        s.add( setting.Choice('coords',
                              ('bottom-left', 'bottom-right',
                               'left-bottom', 'left-right',
                               'right-left', 'right-bottom'),
                              'bottom-left',
                              descr='Axes to use for plotting coordinates',
                              usertext='Coord system') )

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

        s.get('leftMargin').newDefault('1cm')
        s.get('rightMargin').newDefault('1cm')
        s.get('topMargin').newDefault('1cm')
        s.get('bottomMargin').newDefault('1cm')

        s.MajorTicks.get('number').newDefault(10)
        s.MinorTicks.get('number').newDefault(50)
        s.GridLines.get('hide').newDefault(False)
        s.TickLabels.remove('rotate')

    def _maxVal(self):
        '''Get maximum value on axis.'''
        if self.settings.mode == 'percentage':
            return 100.
        else:
            return 1.

    def coordRanges(self):
        '''Get ranges of coordinates.'''
        r = [0., self._maxVal()]
        return [r, r]

    def graphToPlotCoords(self, coorda, coordb):
        '''Convert coordinates in r, theta to x, y.'''

        s = self.settings

        # normalize coordinates
        maxval = self._maxVal()
        coordan = coorda / maxval
        coordbn = coordb / maxval

        # the three coordinates on the plot
        clist = (coordan, coordbn, 1.-coordan-coordbn)

        # select the right coordinates for a, b and c given the system
        # requested by the user
        lookup = {'bottom-left': (0, 1, 2),
                  'bottom-right': (0, 2, 1),
                  'left-bottom': (1, 0, 2),
                  'left-right': (2, 0, 1),
                  'right-bottom': (1, 2, 0),
                  'right-left': (2, 1, 0)}[s.coords]
        a = clist[ lookup[0] ]
        b = clist[ lookup[1] ]
        c = clist[ lookup[2] ]

        # from Ingram, 1984, Area, 16, 175
        # remember that y goes in the opposite direction here
        scale = self._box[2]-self._box[0]
        x = (0.5*a + c)*scale + self._box[0]
        y = self._box[3] - a*math.sin(60./180*math.pi)*scale

        return x, y

    def drawFillPts(self, painter, cliprect, ptsx, ptsy, filltype):
        '''Draw points for plotting a fill.'''
        pts = qt4.QPolygonF()
        utils.addNumpyToPolygonF(pts, ptsx, ptsy)

        # this is broken: FIXME
        if filltype == 'left':
            pts.append( qt4.QPointF(self._box[0], self._box[3]) )
            pts.append( qt4.QPointF(self._box[0]+self._width/2., self._box[1]) )
        elif filltype == 'right':
            pts.append( qt4.QPointF(self._box[2], self._box[3]) )
            pts.append( qt4.QPointF(self._box[2]-self._width/2., self._box[1]) )
        elif filltype == 'bottom':
            pts.append( qt4.QPointF(self._box[0], self._box[3]) )
            pts.append( qt4.QPointF(self._box[2], self._box[3]) )

        utils.plotClippedPolygon(painter, cliprect, pts)

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

        painter.setPen( s.Border.makeQPenWHide(painter) )
        painter.setBrush( s.Background.makeQBrushWHide() )
        painter.drawPolygon(p)

    def setClip(self, painter, bounds):
        '''Set clipping for graph.'''
        p = qt4.QPainterPath()
        p.addPolygon( self._tripoly )
        painter.setClipPath(p)

    def _drawTickSet(self, painter, tickSetn, gridSetn, tickvals,
                     ticklabels=None, labelSetn=None):
        '''Draw a set of ticks (major or minor).'''

        tl = tickSetn.get('length').convert(painter)

        # first set of ticks to draw and origin for coords
        x1 = x2 = (tickvals/self._maxVal()) * self._width
        y1 = 0*x1
        y2 = y1-tl
        ox, oy = self._box[0], self._box[3]

        # 2nd set
        x1p, y1p = rotatePts(x1, y1, -120)
        x2p, y2p = rotatePts(x2, y2, -120)
        oxp, oyp = self._box[0] + self._width*0.5, self._box[3] - self._height

        # 3rd set
        x1pp, y1pp = rotatePts(x1, y1, 120)
        x2pp, y2pp = rotatePts(x2, y2, 120)
        oxpp, oypp = self._box[0] + self._width, self._box[3]

        # draw grid if not hidden
        if not gridSetn.hide:
            painter.setPen( gridSetn.makeQPen(painter) )
            utils.plotLinesToPainter(painter, x1[1:-1]+ox, oy-y1[1:-1],
                                     x1p[1:-1][::-1]+oxp, oyp-y1p[1:-1][::-1])
            utils.plotLinesToPainter(painter, x1[1:-1]+ox, oy-y1[1:-1],
                                     x1pp[1:-1][::-1]+oxpp,
                                     oypp-y1pp[1:-1][::-1])
            utils.plotLinesToPainter(painter, x1p[1:-1]+oxp, oyp-y1p[1:-1],
                                     x1pp[1:-1][::-1]+oxpp,
                                     oypp-y1pp[1:-1][::-1])

        # draw ticks if not hidden
        if not tickSetn.hide:
            painter.setPen( tickSetn.makeQPen(painter) )
            utils.plotLinesToPainter(painter, ox+x1, oy-y1, ox+x2, oy-y2)
            utils.plotLinesToPainter(painter, oxp+x1p, oyp-y1p, oxp+x2p,
                                     oyp-y2p)
            utils.plotLinesToPainter(painter, oxpp+x1pp, oypp-y1pp,
                                     oxpp+x2pp, oypp-y2pp)

        # draw labels (if any)
        if ticklabels and not labelSetn.hide:
            painter.setPen( labelSetn.makeQPen() )
            font = labelSetn.makeQFont(painter)
            painter.setFont(font)

            fm = utils.FontMetrics(font, painter.device())
            sp = fm.leading() + fm.descent()
            off = labelSetn.get('offset').convert(painter)

            for l, x, y in izip(ticklabels, ox+x2, oy-y2+off):
                r = utils.Renderer(painter, font, x, y, l, 0, 1, 0)
                r.render()
            for l, x, y in izip(ticklabels, oxp-sp-off+x2p, oyp-y2p):
                r = utils.Renderer(painter, font, x, y, l, 1, 0, 0)
                r.render()
            for l, x, y in izip(ticklabels, oxpp+sp+off+x2pp, oypp-y2pp):
                r = utils.Renderer(painter, font, x, y, l, -1, 0, 0)
                r.render()

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
        return labels

    def drawAxes(self, painter, bounds, datarange, outerbounds=None):
        '''Draw plot axes.'''

        s = self.settings

        atick = AxisTicks(0, self._maxVal(), s.MajorTicks.number,
                          s.MinorTicks.number,
                          extendbounds=False,  extendzero=False)
        minval, maxval, majtick, mintick, tickformat = atick.getTicks()

        self._drawTickSet(painter, s.MinorTicks, s.MinorGridLines, mintick)
        self._drawTickSet(painter, s.MajorTicks, s.GridLines, majtick,
                          ticklabels=self._getLabels(majtick, tickformat),
                          labelSetn=s.TickLabels)


document.thefactory.register(Ternary)
