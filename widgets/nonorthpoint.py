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

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import pickable

from nonorthgraph import NonOrthGraph, FillBrush
from widget import Widget
from point import MarkerFillBrush

class NonOrthPoint(Widget):
    '''Widget for plotting points in a non-orthogonal plot.'''

    typename = 'nonorthpoint'
    allowusercreation = True
    description = 'Plot points on a graph with non-orthogonal axes'

    allowedparenttypes = [NonOrthGraph]

    def __init__(self, parent, name=None):
        """Initialise plotter."""
        Widget.__init__(self, parent, name=name)
        if type(self) == NonOrthPoint:
            self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        '''Settings for widget.'''
        Widget.addSettings(s)

        s.add( setting.DatasetOrFloatList(
                'data1', 'x',
                descr='Dataset containing 1st dataset or list of values',
                usertext='Dataset 1') )
        s.add( setting.DatasetOrFloatList(
                'data2', 'y',
                descr='Dataset containing 2nd dataset or list of values',
                usertext='Dataset 2') )
        s.add( setting.DatasetOrFloatList(
                'scalePoints', '',
                descr = 'Scale size of plotted markers by this dataset or'
                ' list of values', usertext='Scale markers') )

        s.add( setting.DistancePt('markerSize',
                                  '3pt',
                                  descr = 'Size of marker to plot',
                                  usertext='Marker size', formatting=True), 0 )
        s.add( setting.Marker('marker',
                              'circle',
                              descr = 'Type of marker to plot',
                              usertext='Marker', formatting=True), 0 )
        s.add( setting.Line('PlotLine',
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
        s.add( FillBrush('Fill1',
                         descr = 'Fill settings (1)',
                         usertext = 'Area fill 1'),
               pixmap = 'settings_plotfillbelow' )
        s.add( FillBrush('Fill2',
                         descr = 'Fill settings (2)',
                         usertext = 'Area fill 2'),
               pixmap = 'settings_plotfillbelow' )

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

    def pickPoint(self, x0, y0, bounds, distance = 'radial'):
        p = pickable.DiscretePickable(self, 'data1', 'data2',
                lambda v1, v2, b: self.parent.graphToPlotCoords(v1, v2))
        return p.pickPoint(x0, y0, bounds, distance)

    def pickIndex(self, oldindex, direction, bounds):
        p = pickable.DiscretePickable(self, 'data1', 'data2',
                lambda v1, v2, b: self.parent.graphToPlotCoords(v1, v2))
        return p.pickIndex(oldindex, direction, bounds)

    def plotMarkers(self, painter, plta, pltb, scaling, clip):
        '''Draw markers in widget.'''
        s = self.settings
        if not s.MarkerLine.hide or not s.MarkerFill.hide:
            painter.setBrush( s.MarkerFill.makeQBrushWHide() )
            painter.setPen( s.MarkerLine.makeQPenWHide(painter) )
                
            size = s.get('markerSize').convert(painter)
            utils.plotMarkers(painter, plta, pltb, s.marker, size,
                              scaling=scaling, clip=clip)

    def draw(self, parentposn, painter, outerbounds=None):
        '''Plot the data on a plotter.'''

        posn = Widget.draw(self, parentposn, painter,
                           outerbounds=outerbounds)
        x1, y1, x2, y2 = posn
        cliprect = qt4.QRectF( qt4.QPointF(x1, y1), qt4.QPointF(x2, y2) )

        s = self.settings
        d = self.document

        # exit if hidden
        if s.hide:
            return

        d1 = s.get('data1').getData(d)
        d2 = s.get('data2').getData(d)
        dscale = s.get('scalePoints').getData(d)
        if not d1 or not d2:
            return

        painter.beginPaintingWidget(self, posn)
        painter.save()

        # split parts separated by NaNs
        for v1, v2, vs in document.generateValidDatasetParts(d1, d2, dscale):
            # convert data (chopping down length)
            v1d, v2d = v1.data, v2.data
            minlen = min(v1d.shape[0], v2d.shape[0])
            v1d, v2d = v1d[:minlen], v2d[:minlen]
            px, py = self.parent.graphToPlotCoords(v1d, v2d)

            # do fill1 (if any)
            if not s.Fill1.hide:
                painter.setBrush( s.Fill1.makeQBrush() )
                painter.setPen( qt4.QPen(qt4.Qt.NoPen) )
                self.parent.drawFillPts(painter, cliprect, px, py,
                                        s.Fill1.filltype)
            # do fill2
            if not s.Fill2.hide:
                painter.setBrush( s.Fill2.makeQBrush() )
                painter.setPen( qt4.QPen(qt4.Qt.NoPen) )
                self.parent.drawFillPts(painter, cliprect, px, py,
                                        s.Fill2.filltype)

            # plot line
            if not s.PlotLine.hide:
                painter.setBrush( qt4.QBrush() )
                painter.setPen(s.PlotLine.makeQPen(painter))
                pts = qt4.QPolygonF()
                utils.addNumpyToPolygonF(pts, px, py)
                utils.plotClippedPolyline(painter, cliprect, pts)

            # markers
            pscale = None
            if vs:
                pscale = vs.data
            self.plotMarkers(painter, px, py, pscale, cliprect)

        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate plotter
document.thefactory.register( NonOrthPoint )
