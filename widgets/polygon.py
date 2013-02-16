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

import veusz.document as document
import veusz.setting as setting
import veusz.qtall as qt4
import veusz.utils as utils

import plotters

def _(text, disambiguation=None, context='Polygon'):
    """Translate text."""
    return unicode( 
        qt4.QCoreApplication.translate(context, text, disambiguation))

class Polygon(plotters.FreePlotter):
    """For plotting polygons."""

    typename = 'polygon'
    allowusercreeation = True
    description = _('Plot a polygon')

    def __init__(self, parent, name=None):
        """Initialise object, setting axes."""
        plotters.FreePlotter.__init__(self, parent, name=name)

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.FreePlotter.addSettings(s)

        s.add( setting.Line('Line',
                            descr = _('Line around polygon'),
                            usertext = _('Line')),
               pixmap = 'settings_plotline' )
        s.add( setting.BrushExtended('Fill',
                                     descr = _('Fill within polygon'),
                                     usertext = _('Fill')),
               pixmap = 'settings_plotfillbelow' )

    def draw(self, posn, phelper, outerbounds=None):
        """Plot the data on a plotter."""

        s = self.settings

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
        with painter:
            pen = s.Line.makeQPenWHide(painter)
            pw = pen.widthF()*2
            lineclip = qt4.QRectF( qt4.QPointF(x1-pw, y1-pw),
                                   qt4.QPointF(x2+pw, y2+pw) )

            # this is a hack as we generate temporary fake datasets
            path = qt4.QPainterPath()
            for xvals, yvals in document.generateValidDatasetParts(
                document.Dataset(xp), document.Dataset(yp)):

                poly = qt4.QPolygonF()
                utils.addNumpyToPolygonF(poly, xvals.data, yvals.data)
                clippedpoly = qt4.QPolygonF()
                utils.polygonClip(poly, lineclip, clippedpoly)
                path.addPolygon(clippedpoly)
                path.closeSubpath()

            utils.brushExtFillPath(painter, s.Fill, path, stroke=pen)

# allow the factory to instantiate this
document.thefactory.register( Polygon )
