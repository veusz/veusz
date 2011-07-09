#    Copyright (C) 2009 Jeremy S. Sanders
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
###############################################################################

from itertools import izip

import veusz.document as document
import veusz.setting as setting
import veusz.qtall as qt4

import plotters

class Polygon(plotters.FreePlotter):
    """For plotting polygons."""

    typename = 'polygon'
    allowusercreeation = True
    description = 'Plot a polygon'

    def __init__(self, parent, name=None):
        """Initialise object, setting axes."""
        plotters.FreePlotter.__init__(self, parent, name=name)

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.FreePlotter.addSettings(s)

        s.add( setting.Line('Line',
                            descr = 'Line around polygon',
                            usertext = 'Line'),
               pixmap = 'settings_plotline' )
        s.add( setting.Brush('Fill',
                             descr = 'Fill within polygon',
                             usertext = 'Fill'),
               pixmap = 'settings_plotfillbelow' )

    def draw(self, posn, phelper, outerbounds=None):
        """Plot the data on a plotter."""

        s = self.settings
        d = self.document

        # exit if hidden
        if s.hide:
            return

        # get points in plotter coordinates
        xp, yp = self._getPlotterCoords(posn)
        if xp is None or yp is None:
            # we can't calculate coordinates
            return

        x1, y1, x2, y2 = posn
        cliprect = qt4.QRectF( qt4.QPointF(x1, y1), qt4.QPointF(x2, y2) )
        painter = phelper.painter(self, posn, clip=cliprect)

        painter.setPen( s.Line.makeQPenWHide(phelper) )
        painter.setBrush( s.Fill.makeQBrushWHide() )

        # this is a hack as we generate temporary fake datasets
        for xvals, yvals in document.generateValidDatasetParts(
            document.Dataset(xp), document.Dataset(yp)):

            # construct polygon
            poly = qt4.QPolygonF()
            xd, yd = xvals.data, yvals.data
            for x, y in izip(xd, yd):
                poly.append(qt4.QPointF(x, y))
            # draw it
            painter.drawPolygon(poly)

# allow the factory to instantiate this
document.thefactory.register( Polygon )
