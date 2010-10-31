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

# $Id$

"""Non orthogonal point plotting."""

import numpy as N

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

from nonorthgraph import NonOrthGraph
from widget import Widget
from point import MarkerFillBrush

class NonOrthPoint(Widget):

    typename = 'NOxy'
    allowusercreation = True
    description = 'Plot points on graphs with non-orthogonal axes.'

    allowedparenttypes = [NonOrthGraph]

    @classmethod
    def addSettings(klass, s):
        '''Settings for widget.'''
        Widget.addSettings(s)

        s.add( setting.DistancePt('markerSize',
                                  '3pt',
                                  descr = 'Size of marker to plot',
                                  usertext='Marker size', formatting=True), 0 )
        s.add( setting.Marker('marker',
                              'circle',
                              descr = 'Type of marker to plot',
                              usertext='Marker', formatting=True), 0 )
        s.add( setting.XYPlotLine('PlotLine',
                                  descr = 'Plot line settings',
                                  usertext = 'Plot line'),
               pixmap = 'settings_plotline' )
        s.add( setting.Line('MarkerLine',
                            descr = 'Line around the marker settings',
                            usertext = 'Marker border'),
               pixmap = 'settings_plotmarkerline' )
        s.add( MarkerFillBrush('MarkerFill',
                               descr = 'Marker fill settings',
                               usertext = 'Marker fill'),
               pixmap = 'settings_plotmarkerfill' )
        s.add( setting.DatasetOrFloatList(
                'data2', 'y',
                descr='Dataset containing 1st dataset or list of values',
                usertext='Dataset 1'), 0 )
        s.add( setting.DatasetOrFloatList(
                'data1', 'x',
                descr='Dataset containing 2nd dataset or list of values',
                usertext='Dataset 2'), 0 )

    def updateDataRanges(self, inrange):
        '''Extend inrange to range of data.'''

        d1 = self.settings.get('data1').getData(self.document)
        if d1:
            inrange[0] = min( N.nanmin(d1.data), inrange[0] )
            inrange[1] = max( N.nanmax(d1.data), inrange[1] )
        d2 = self.settings.get('data2').getData(self.document)
        if d2:
            inrange[2] = min( N.nanmin(d2.data), inrange[2] )
            inrange[3] = max( N.nanmax(d2.data), inrange[3] )

    def plotMarkers(self, painter, plta, pltb):
        # draw marker
        s = self.settings
        if not s.MarkerLine.hide or not s.MarkerFill.hide:
            painter.setBrush( s.MarkerFill.makeQBrushWHide() )
            painter.setPen( s.MarkerLine.makeQPenWHide(painter) )
                
            size = s.get('markerSize').convert(painter)
            utils.plotMarkers(painter, plta, pltb, s.marker, size)

    def draw(self, parentposn, painter, outerbounds=None):
        '''Plot the data on a plotter.'''

        posn = Widget.draw(self, parentposn, painter,
                           outerbounds=outerbounds)
        x1, y1, x2, y2 = posn

        s = self.settings

        # exit if hidden
        if s.hide:
            return

        d1 = self.settings.get('data1').getData(self.document)
        d2 = self.settings.get('data2').getData(self.document)
        if not d1 or not d2:
            return
        d1d = d1.data
        d2d = d2.data
        minlen = min(d1d.shape[0], d2d.shape[0])
        d1d, d2d = d1d[:minlen], d2d[:minlen]

        painter.beginPaintingWidget(self, posn)
        painter.save()

        plta, pltb = self.parent.graphToPlotCoords(d1d, d2d)
        self.plotMarkers(painter, plta, pltb)

        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate plotter
document.thefactory.register( NonOrthPoint )
