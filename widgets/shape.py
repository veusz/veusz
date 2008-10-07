#    Copyright (C) 2008 Jeremy S. Sanders
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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
###############################################################################

# $Id$

"""For plotting shapes."""

import itertools

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.document as document

import widget
import page
import graph

class Shape(widget.Widget):
    """A shape on a page/graph."""

    allowedparenttypes = [graph.Graph, page.Page]

    def __init__(self, parent, name=None):
        widget.Widget.__init__(self, parent, name=name)
        s = self.settings

        s.add( setting.ShapeFill('Fill',
                                 descr = 'Shape fill',
                                 usertext='Fill'),
               pixmap = 'bgfill' )
        s.add( setting.Line('Border',
                            descr = 'Shape border',
                            usertext='Border'),
               pixmap = 'border' )

class BoxShape(Shape):
    """For drawing box-like shapes."""

    def __init__(self, parent, name=None):
        Shape.__init__(self, parent, name=name)

        s = self.settings

        s.add( setting.DatasetOrFloatList('xPos', 0.5,
                                          descr='List of fractional X coordinates or dataset',
                                          usertext='X position',
                                          formatting=False), 0)
        s.add( setting.DatasetOrFloatList('yPos', 0.5,
                                          descr='List of fractional Y coordinates or dataset',
                                          usertext='Y position',
                                          formatting=False), 1)
        s.add( setting.DatasetOrFloatList('width', 0.1,
                                          descr='List of fractional widths or dataset',
                                          usertext='Width',
                                          formatting=False), 2)
        s.add( setting.DatasetOrFloatList('height', 0.1,
                                          descr='List of fractional heights or dataset',
                                          usertext='Height',
                                          formatting=False), 3)

    def drawShape(self, painter, rect):
        pass

    def draw(self, posn, painter, outerbounds = None):
        """Plot the key on a plotter."""

        s = self.settings
        d = self.document
        if s.hide:
            return

        # get positions of shapes
        xpos = s.get('xPos').getFloatArray(d)
        ypos = s.get('yPos').getFloatArray(d)
        width = s.get('width').getFloatArray(d)
        height = s.get('height').getFloatArray(d)

        if xpos is None or ypos is None or width is None or height is None:
            return

        isnotdataset = ( not s.get('xPos').isDataset(d) and 
                         not s.get('yPos').isDataset(d) and
                         not s.get('width').isDataset(d) and
                         not s.get('height').isDataset(d) )
        del self.controlgraphitems[:]

        painter.beginPaintingWidget(self, posn)
        painter.save()

        painter.setPen( s.get('Border').makeQPen(painter) )
        painter.setBrush( s.get('Fill').makeQBrush() )

        # iterate over positions
        dx, dy = posn[2]-posn[0], posn[3]-posn[1]
        for x, y, w, h in itertools.izip(xpos, ypos, width, height):
            xp = posn[0] + dx*x
            yp = posn[3] - dy*y
            wp = dx*w
            hp = dy*h
            self.drawShape(painter, qt4.QRectF(xp-wp*0.5, yp-hp*0.5, wp, hp))

        painter.restore()
        painter.endPaintingWidget()

class Rectangle(BoxShape):
    typename = 'rect'
    description = 'Rectangle'
    allowusercreation = True

    def drawShape(self, painter, rect):
        painter.drawRect( qt4.QRectF(rect) )

class RoundRect(BoxShape):
    typename = 'roundrect'
    description = 'Rounded rectangle'
    allowusercreation = True

    def __init__(self, parent, name=None):
        BoxShape.__init__(self, parent, name=name)
        self.settings.add( setting.Int('rounding', 50,
                                       minval=0, maxval=100,
                                       descr='Rounding percentage',
                                       usertext='Rounding') )

    def drawShape(self, painter, rect):
        r = self.settings.rounding
        painter.drawRoundRect( qt4.QRectF(rect), r, r )

class Ellipse(BoxShape):
    typename = 'ellipse'
    description = 'Ellipse'
    allowusercreation = True

    def drawShape(self, painter, rect):
        painter.drawEllipse( qt4.QRectF(rect) )

document.thefactory.register( Ellipse )
document.thefactory.register( Rectangle )
document.thefactory.register( RoundRect )
