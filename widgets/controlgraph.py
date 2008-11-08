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
import itertools

import veusz.qtall as qt4
import veusz.document as document

##############################################################################

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
        self.setZValue(3.)

    def mouseMoveEvent(self, event):
        """Notify parent on move."""
        qt4.QGraphicsRectItem.mouseMoveEvent(self, event)
        self.parentItem().updateFromCorner(self, event)

    def mouseReleaseEvent(self, event):
        """Notify parent on unclicking."""
        qt4.QGraphicsRectItem.mouseReleaseEvent(self, event)
        self.parentItem().doUpdate()

##############################################################################

dottedlinepen = qt4.QPen(qt4.Qt.blue, 2, qt4.Qt.DotLine)

class _EdgeLine(qt4.QGraphicsLineItem):
    """Line used for edges of resizing box."""
    def __init__(self, parent, ismovable = True):
        qt4.QGraphicsLineItem.__init__(self, parent)
        self.setPen(dottedlinepen)
        self.setZValue(2.)
        if ismovable:
            self.setFlag(qt4.QGraphicsItem.ItemIsMovable)
            self.setCursor(qt4.Qt.SizeAllCursor)

    def mouseMoveEvent(self, event):
        """Notify parent on move."""
        qt4.QGraphicsLineItem.mouseMoveEvent(self, event)
        self.parentItem().updateFromLine(self, self.pos())

    def mouseReleaseEvent(self, event):
        """Notify parent on unclicking."""
        qt4.QGraphicsLineItem.mouseReleaseEvent(self, event)
        self.parentItem().doUpdate()

##############################################################################

class ControlGraphMarginBox(qt4.QGraphicsItem):
    """A box which can be moved or resized.

    Can automatically set margins or widget
    """

    # posn coords of each corner
    mapcornertoposn = ( (0, 1), (2, 1), (0, 3), (2, 3) )

    def __init__(self, widget, posn, maxposn, painter,
                 ismovable = True, isresizable = True):
        """Create control box item.

        widget: widget this is controllng
        posn: coordinates of box [x1, y1, x2, y2]
        maxposn: coordinates of biggest possibe box
        painter: painter to get scaling from
        ismovable: box can be moved
        isresizable: box can be resized
        """

        qt4.QGraphicsItem.__init__(self)
        self.setZValue(2.)

        # corners of box
        self.corners = [_ShapeCorner(self)
                        for i in xrange(4)]

        # lines connecting corners
        self.lines = [_EdgeLine(self, ismovable=ismovable)
                      for i in xrange(4)]

        # hide corners if box is not resizable
        if not isresizable:
            for c in self.corners:
                c.hide()

        self.origposn = self.posn = posn
        self.maxposn = maxposn
        self.widget = widget
        self.updateCornerPosns()

        # we need these later to convert back to original units
        self.page_size = painter.veusz_page_size
        self.scaling = painter.veusz_scaling
        self.pixperpt = painter.veusz_pixperpt

    def updateCornerPosns(self):
        """Update all corners from updated box."""

        p = self.posn
        # update cursors
        self.corners[0].setCursor(qt4.Qt.SizeFDiagCursor)
        self.corners[1].setCursor(qt4.Qt.SizeBDiagCursor)
        self.corners[2].setCursor(qt4.Qt.SizeBDiagCursor)
        self.corners[3].setCursor(qt4.Qt.SizeFDiagCursor)

        # trim box to maximum size
        p[0] = max(p[0], self.maxposn[0])
        p[1] = max(p[1], self.maxposn[1])
        p[2] = min(p[2], self.maxposn[2])
        p[3] = min(p[3], self.maxposn[3])

        # move corners
        for corner, (xindex, yindex) in itertools.izip(self.corners,
                                                       self.mapcornertoposn):
            corner.setPos( qt4.QPointF( p[xindex], p[yindex] ) )

        # move lines
        w, h = p[2]-p[0], p[3]-p[1]
        self.lines[0].setPos(p[0], p[1])
        self.lines[0].setLine(0, 0,  w,  0)
        self.lines[1].setPos(p[2], p[1])
        self.lines[1].setLine(0, 0,  0,  h)
        self.lines[2].setPos(p[2], p[3])
        self.lines[2].setLine(0, 0, -w,  0)
        self.lines[3].setPos(p[0], p[3])
        self.lines[3].setLine(0, 0,  0, -h)

    def updateFromLine(self, line, thispos):
        """Edge line of box was moved - update bounding box."""

        # need old coordinate to work out how far line has moved
        li = self.lines.index(line)
        ox = self.posn[ (0, 2, 2, 0)[li] ]
        oy = self.posn[ (1, 1, 3, 3)[li] ]

        # add on deltas to box coordinates
        dx, dy = thispos.x()-ox, thispos.y()-oy

        # make sure box can't be moved outside the allowed region
        if dx > 0:
            dx = min(dx, self.maxposn[2]-self.posn[2])
        else:
            dx = -min(abs(dx), abs(self.maxposn[0]-self.posn[0]))
        if dy > 0:
            dy = min(dy, self.maxposn[3]-self.posn[3])
        else:
            dy = -min(abs(dy), abs(self.maxposn[1]-self.posn[1]))

        # move the box
        self.posn[0] += dx
        self.posn[1] += dy
        self.posn[2] += dx
        self.posn[3] += dy

        # update corner coords and other line coordinates
        self.updateCornerPosns()

    def updateFromCorner(self, corner, event):
        """Move corner of box to new position."""
        index = self.corners.index(corner)
        self.posn[ self.mapcornertoposn[index][0] ] = corner.x()
        self.posn[ self.mapcornertoposn[index][1] ] = corner.y()

        # this is needed if the corners move past each other
        if self.posn[0] > self.posn[2]:
            # swap x
            self.posn[0], self.posn[2] = self.posn[2], self.posn[0]
            self.corners[0], self.corners[1] = self.corners[1], self.corners[0]
            self.corners[2], self.corners[3] = self.corners[3], self.corners[2]
        if self.posn[1] > self.posn[3]:
            # swap y
            self.posn[1], self.posn[3] = self.posn[3], self.posn[1]
            self.corners[0], self.corners[2] = self.corners[2], self.corners[0]
            self.corners[1], self.corners[3] = self.corners[3], self.corners[1]

        self.updateCornerPosns()
        
    def boundingRect(self):
        return qt4.QRectF(0, 0, 0, 0)

    def paint(self, painter, option, widget):
        """Intentionally empty painter."""

    def doUpdate(self):
        """Update widget margins."""
        self.widget.updateControlItem(self)

    def setWidgetMargins(self):
        """A helpful routine for setting widget margins after
        moving or resizing.

        This is called by the widget after receiving
        updateControlItem
        """
        s = self.widget.settings

        # get margins in pixels
        left = self.posn[0] - self.maxposn[0]
        right = self.maxposn[2] - self.posn[2]
        top = self.posn[1] - self.maxposn[1]
        bottom = self.maxposn[3] - self.posn[3]

        # set up fake painter containing veusz scalings
        fakepainter = qt4.QPainter()
        fakepainter.veusz_page_size = self.page_size
        fakepainter.veusz_scaling = self.scaling
        fakepainter.veusz_pixperpt = self.pixperpt

        # convert to physical units
        left = s.get('leftMargin').convertInverse(left, fakepainter)
        right = s.get('rightMargin').convertInverse(right, fakepainter)
        top = s.get('topMargin').convertInverse(top, fakepainter)
        bottom = s.get('bottomMargin').convertInverse(bottom, fakepainter)

        # modify widget margins
        operations = (
            document.OperationSettingSet(s.get('leftMargin'), left),
            document.OperationSettingSet(s.get('rightMargin'), right),
            document.OperationSettingSet(s.get('topMargin'), top),
            document.OperationSettingSet(s.get('bottomMargin'), bottom)
            )
        self.widget.document.applyOperation(
            document.OperationMultiple(operations, descr='resize margins'))

##############################################################################

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

        # initial setup
        self.setCursor(qt4.Qt.SizeAllCursor)
        self.setZValue(1.)
        self.setFlag(qt4.QGraphicsItem.ItemIsMovable)
        self.setPen(dottedlinepen)
        self.setBrush( qt4.QBrush() )

        # create child graphicsitem for each corner
        self.corners = [_ShapeCorner(self) for i in xrange(4)]
        self.corners[0].setCursor(qt4.Qt.SizeFDiagCursor)
        self.corners[1].setCursor(qt4.Qt.SizeBDiagCursor)
        self.corners[2].setCursor(qt4.Qt.SizeBDiagCursor)
        self.corners[3].setCursor(qt4.Qt.SizeFDiagCursor)

        # whether box is allowed to be rotated
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

##############################################################################

class ControlGraphMovableBox(ControlGraphMarginBox):
    """Item for user display for controlling widget.
    This is a dotted movable box with an optional "cross" where
    the real position of the widget is
    """

    def __init__(self, widget, posn, painter, crosspos=None):
        ControlGraphMarginBox.__init__(self, widget, posn,
                                       [-10000, -10000, 10000, 10000],
                                       painter, isresizable=False)
        self.cross = _ShapeCorner(self)
        self.crosspos = (crosspos[0] - posn[0],
                         crosspos[1] - posn[1])
        self.updateCornerPosns()

    def updateCornerPosns(self):
        ControlGraphMarginBox.updateCornerPosns(self)

        # this fails if called before self.cross is initialised!
        if hasattr(self, 'cross'):
            self.cross.setPos( self.crosspos[0] + self.posn[0],
                               self.crosspos[1] + self.posn[1] )

    def updateFromCorner(self, corner, event):
        if corner == self.cross:
            # if cross moves, move whole box
            cx, cy = self.cross.pos().x(), self.cross.pos().y()
            dx = cx - (self.crosspos[0] + self.posn[0])
            dy = cy - (self.crosspos[1] + self.posn[1])

            self.posn[0] += dx
            self.posn[1] += dy
            self.posn[2] += dx
            self.posn[3] += dy
            self.updateCornerPosns()
        else:
            ControlGraphMarginBox.updateFromCorner(self, corner, event)

##############################################################################

class ControlGraphLine(qt4.QGraphicsLineItem):
    """For controlling the position and ends of a line."""

    def __init__(self, widget, x1, y1, x2, y2):
        qt4.QGraphicsLineItem.__init__(self, x1, y1, x2, y2)
        self.widget = widget
        self.setCursor(qt4.Qt.SizeAllCursor)
        self.setFlag(qt4.QGraphicsItem.ItemIsMovable)
        self.setZValue(1.)
        self.setPen(dottedlinepen)
        self.pts = [_ShapeCorner(self, rotator=True),
                    _ShapeCorner(self, rotator=True)]
        self.pts[0].setPos(x1, y1)
        self.pts[1].setPos(x2, y2)
        self.pts[0].setCursor(qt4.Qt.CrossCursor)
        self.pts[1].setCursor(qt4.Qt.CrossCursor)

    def updateFromCorner(self, corner, event):
        """Take position and update ends of line."""
        self.setLine( self.pts[0].x(), self.pts[0].y(),
                      self.pts[1].x(), self.pts[1].y() )

    def mouseReleaseEvent(self, event):
        """If widget has moved, tell it."""
        qt4.QGraphicsItem.mouseReleaseEvent(self, event)
        self.doUpdate()

    def doUpdate(self):
        """Update caller with position and line positions."""

        pt1 = ( self.pts[0].x() + self.pos().x(),
                self.pts[0].y() + self.pos().y() )
        pt2 = ( self.pts[1].x() + self.pos().x(),
                self.pts[1].y() + self.pos().y() )

        self.widget.updateControlItem(self, pt1, pt2)
