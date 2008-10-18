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
##############################################################################

# $Id$

"""
Classes for moving widgets around
"""

import math
import veusz.qtall as qt4

class _ShapeCorner(qt4.QGraphicsRectItem):
    """Representing the corners of the rectangle."""
    def __init__(self, parent, rotator=False):
        qt4.QGraphicsRectItem.__init__(self, parent)
        if rotator:
            self.setBrush(qt4.QBrush(qt4.Qt.blue))
            self.setRect(-3, -3, 6, 6)
        else:
            self.setBrush(qt4.QBrush(qt4.Qt.black))
            self.setRect(-5, -5, 10, 10)
        self.setPen(qt4.QPen(qt4.Qt.NoPen))
        self.setFlag(qt4.QGraphicsItem.ItemIsMovable)
        self.setZValue(2.)

    def mouseMoveEvent(self, event):
        """Notify parent on move."""
        qt4.QGraphicsRectItem.mouseMoveEvent(self, event)
        self.parentItem().updateFromCorner(self, event)

    def mouseReleaseEvent(self, event):
        """Notify parent on unclicking."""
        qt4.QGraphicsRectItem.mouseReleaseEvent(self, event)
        self.parentItem().doUpdate()

class ControlGraphResizableBox(qt4.QGraphicsRectItem):
    """Control a resizable box.
    Item resizes centred around a position
    """

    def __init__(self, widget, posn, dims, angle, allowrotate=False):
        """Initialise with widget and boxbounds shape.
        Rotation is allowed if allowrotate is set
        """

        qt4.QGraphicsRectItem.__init__(self, 0., 0., dims[0], dims[1])
        self.setPos(posn[0], posn[1])

        self.widget = widget
        self.posn = posn
        self.dims = dims
        self.angle = angle
        self.rotate(angle)

        self.setCursor(qt4.Qt.SizeAllCursor)
        self.setZValue(1.)
        self.setFlag(qt4.QGraphicsItem.ItemIsMovable)
        self.setPen( qt4.QPen(qt4.Qt.DotLine) )
        self.setBrush( qt4.QBrush() )

        # create child graphicsitem for each corner
        self.corners = [_ShapeCorner(self) for i in xrange(4)]
        self.corners[0].setCursor(qt4.Qt.SizeFDiagCursor)
        self.corners[1].setCursor(qt4.Qt.SizeBDiagCursor)
        self.corners[2].setCursor(qt4.Qt.SizeBDiagCursor)
        self.corners[3].setCursor(qt4.Qt.SizeFDiagCursor)

        self.rotator = None
        if allowrotate:
            self.rotator = _ShapeCorner(self, rotator=True)
            self.rotator.setCursor(qt4.Qt.CrossCursor)

        self.updateCorners()
        self.rotator.setPos( 0, -abs(self.dims[1]*0.5) )

    def updateFromCorner(self, corner, event):
        """Take position and update corners."""

        if corner in self.corners:
            # compute size from corner position
            self.dims[0] = corner.pos().x()*2
            self.dims[1] = corner.pos().y()*2
        elif corner == self.rotator:
            # work out angle relative to centre of widget
            delta = event.scenePos() - self.scenePos()
            angle = math.atan2( delta.y(), delta.x() )
            # change to degrees from correct direction
            self.angle = (angle*(180/math.pi) + 90.) % 360

            # apply rotation
            selfpt = self.pos()
            self.resetTransform()
            self.setPos(selfpt)
            self.rotate(self.angle)

        self.updateCorners()

    def updateCorners(self):
        """Update corners on size."""
        size = 5

        # update position and size
        self.setPos( self.posn[0], self.posn[1] )
        self.setRect( -self.dims[0]*0.5, -self.dims[1]*0.5,
                       self.dims[0], self.dims[1] )

        # update corners
        self.corners[0].setPos(-self.dims[0]*0.5, -self.dims[1]*0.5)
        self.corners[1].setPos( self.dims[0]*0.5, -self.dims[1]*0.5)
        self.corners[2].setPos(-self.dims[0]*0.5,  self.dims[1]*0.5)
        self.corners[3].setPos( self.dims[0]*0.5,  self.dims[1]*0.5)

        if self.rotator:
            # set rotator position (constant distance)
            self.rotator.setPos( 0, -abs(self.dims[1]*0.5) )

    def mouseReleaseEvent(self, event):
        """If the item has been moved, do and update."""
        qt4.QGraphicsRectItem.mouseReleaseEvent(self, event)
        self.doUpdate()

    def mouseMoveEvent(self, event):
        """Keep track of movement."""
        qt4.QGraphicsRectItem.mouseMoveEvent(self, event)
        self.posn = [self.pos().x(), self.pos().y()]

    def doUpdate(self):
        """Tell the user the graphicsitem has been moved or resized."""
        self.widget.updateControlItem(self)

class ControlGraphMovableBox(qt4.QGraphicsItem):
    """Item for user display for controlling widget.
    This is a dotted movable box with an optional "cross" where
    the real position of the widget is
    """

    def __init__(self, widget, boxbounds, crosspos=None):
        qt4.QGraphicsItem.__init__(self)
        self.widget = widget
        self.boxbounds = boxbounds
        self.setCursor(qt4.Qt.SizeAllCursor)
        self.setFlag(qt4.QGraphicsItem.ItemIsMovable)
        self.setZValue(1.)
        self.crosspos = crosspos

    def mousePressEvent(self, event):
        self.update()
        self.startpos = self.pos()
        qt4.QGraphicsItem.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        """If widget has moved, tell it."""
        self.update()
        if self.pos() != self.startpos:
            self.widget.updateControlItem(self)
        qt4.QGraphicsItem.mouseReleaseEvent(self, event)

    def paint(self, painter, option, widget):
        """Draw box and 'cross'."""
        if self.crosspos:
            painter.setPen( qt4.Qt.NoPen )
            painter.setBrush( qt4.Qt.black )
            painter.drawRect(self.crosspos[0]-4, self.crosspos[1]-4, 8, 8)

        painter.setPen( qt4.Qt.DotLine )
        painter.setBrush( qt4.QBrush() )
        bb = self.boxbounds
        painter.drawRect(bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1])

    def boundingRect(self):
        """Work out bounding rectangle"""
        bb = self.boxbounds
        br = qt4.QRectF(bb[0]-1, bb[1]-1, bb[2]-bb[0]+2, bb[3]-bb[1]+2)
        if self.crosspos:
            br |= qt4.QRectF(self.crosspos[0]-4, self.crosspos[1]-4, 8, 8)
        return br

