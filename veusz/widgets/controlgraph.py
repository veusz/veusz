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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

"""
Classes for moving widgets around

Control items have a createGraphicsItem method which returns a graphics
item to control the object
"""

import math
import os.path
from types import SimpleNamespace

from .. import qtall as qt
from .. import document
from .. import setting
from .. import utils
from ..utils import DEG2RAD, RAD2DEG
from ..helpers import threed

def _(text, disambiguation=None, context='controlgraph'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

##############################################################################

class _ScaledShape:
    """Mixing class for shapes which can return scaled positions.

    The control items are plotted on a zoomed plot, so the raw
    unscaled coordinates need to be scaled by params.cgscale to be in
    the correct position.

    We could scale everything to achieve this, but the line widths and
    size of objects would be changed.

    """

    def setScaledPos(self, x, y):
        self.setPos(x*self.params.cgscale, y*self.params.cgscale)

    def scaledPos(self):
        return self.pos()/self.params.cgscale

    def setScaledLine(self, x1, y1, x2, y2):
        s = self.params.cgscale
        self.setLine(x1*s, y1*s, x2*s, y2*s)

    def setScaledLinePos(self, x1, y1, x2, y2):
        s = self.params.cgscale
        self.setPos(x1*s, y1*s)
        self.setLine(0,0,(x2-x1)*s,(y2-y1)*s)

    def setScaledRect(self, x, y, w, h):
        s = self.params.cgscale
        self.setRect(x*s, y*s, w*s, h*s)

    def scaledX(self):
        return self.x()/self.params.cgscale

    def scaledY(self):
        return self.y()/self.params.cgscale

    def scaledRect(self):
        r = self.rect()
        s = self.params.cgscale
        return qt.QRectF(r.left()/s, r.top()/s, r.width()/s, r.height()/s)

##############################################################################

class _ShapeCorner(qt.QGraphicsRectItem, _ScaledShape):
    """Representing the corners of the rectangle."""
    def __init__(self, parent, params, rotator=False):
        qt.QGraphicsRectItem.__init__(self, parent)
        self.params = params
        if rotator:
            self.setBrush( qt.QBrush(setting.settingdb.color('cntrlline')) )
            self.setRect(-3, -3, 6, 6)
        else:
            self.setBrush(qt.QBrush(setting.settingdb.color('cntrlcorner')) )
            self.setRect(-5, -5, 10, 10)
        self.setPen(qt.QPen(qt.Qt.PenStyle.NoPen))
        self.setFlag(qt.QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setZValue(3.)

    def mouseMoveEvent(self, event):
        """Notify parent on move."""
        qt.QGraphicsRectItem.mouseMoveEvent(self, event)
        self.parentItem().updateFromCorner(self, event)

    def mouseReleaseEvent(self, event):
        """Notify parent on unclicking."""
        qt.QGraphicsRectItem.mouseReleaseEvent(self, event)
        self.parentItem().updateWidget()

##############################################################################

def controlLinePen():
    """Get pen for lines around shapes."""
    return qt.QPen(setting.settingdb.color('cntrlline'), 2, qt.Qt.PenStyle.DotLine)

class _EdgeLine(qt.QGraphicsLineItem, _ScaledShape):
    """Line used for edges of resizing box."""
    def __init__(self, parent, params, ismovable=True):
        qt.QGraphicsLineItem.__init__(self, parent)
        self.setPen(controlLinePen())
        self.setZValue(2.)
        self.params = params
        if ismovable:
            self.setFlag(qt.QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            self.setCursor(qt.Qt.CursorShape.SizeAllCursor)

    def mouseMoveEvent(self, event):
        """Notify parent on move."""
        qt.QGraphicsLineItem.mouseMoveEvent(self, event)
        self.parentItem().updateFromLine(self, self.scaledPos())

    def mouseReleaseEvent(self, event):
        """Notify parent on unclicking."""
        qt.QGraphicsLineItem.mouseReleaseEvent(self, event)
        self.parentItem().updateWidget()

##############################################################################

class ControlMarginBox:
    def __init__(self, widget, posn, maxposn, painthelper,
                 ismovable = True, isresizable = True):
        """Create control box item.

        widget: widget this is controllng
        posn: coordinates of box [x1, y1, x2, y2]
        maxposn: coordinates of biggest possibe box
        painthelper: painterhelper to get scaling from
        ismovable: box can be moved
        isresizable: box can be resized
        """

        # save values
        self.posn = posn
        self.maxposn = maxposn
        self.widget = widget
        self.ismovable = ismovable
        self.isresizable = isresizable

        # we need these later to convert back to original units
        self.document = painthelper.document
        self.pagesize = painthelper.pagesize
        self.cgscale = painthelper.cgscale
        self.dpi = painthelper.dpi

    def createGraphicsItem(self, parent):
        return _GraphMarginBox(parent, self)

    def setWidgetMargins(self):
        """A helpful routine for setting widget margins after
        moving or resizing.

        This is called by the widget after receiving updateControlItem
        """
        s = self.widget.settings

        # get margins in pixels
        left = self.posn[0] - self.maxposn[0]
        right = self.maxposn[2] - self.posn[2]
        top = self.posn[1] - self.maxposn[1]
        bottom = self.maxposn[3] - self.posn[3]

        # set up fake painthelper containing veusz scalings
        helper = document.PaintHelper(
            self.document, self.pagesize,
            scaling=self.cgscale, dpi=self.dpi)

        # convert to physical units
        left = s.get('leftMargin').convertInverse(left, helper)
        right = s.get('rightMargin').convertInverse(right, helper)
        top = s.get('topMargin').convertInverse(top, helper)
        bottom = s.get('bottomMargin').convertInverse(bottom, helper)

        # modify widget margins
        operations = (
            document.OperationSettingSet(s.get('leftMargin'), left),
            document.OperationSettingSet(s.get('rightMargin'), right),
            document.OperationSettingSet(s.get('topMargin'), top),
            document.OperationSettingSet(s.get('bottomMargin'), bottom)
        )
        self.widget.document.applyOperation(
            document.OperationMultiple(operations, descr=_('resize margins')))

    def setPageSize(self):
        """Helper for setting document/page widget size.

        This is called by the widget after receiving updateControlItem
        """
        s = self.widget.settings

        # get margins in pixels
        width = self.posn[2] - self.posn[0]
        height = self.posn[3] - self.posn[1]

        # set up fake painter containing veusz scalings
        helper = document.PaintHelper(
            self.document, self.pagesize,
            scaling=self.cgscale, dpi=self.dpi)

        # convert to physical units
        width = s.get('width').convertInverse(width, helper)
        height = s.get('height').convertInverse(height, helper)

        # modify widget margins
        operations = (
            document.OperationSettingSet(s.get('width'), width),
            document.OperationSettingSet(s.get('height'), height),
        )
        self.widget.document.applyOperation(
            document.OperationMultiple(operations, descr=_('change page size')))

class _GraphMarginBox(qt.QGraphicsItem):
    """A box which can be moved or resized.

    Can automatically set margins or widget
    """

    # posn coords of each corner
    mapcornertoposn = ( (0, 1), (2, 1), (0, 3), (2, 3) )

    def __init__(self, parent, params):
        """Create control box item."""

        qt.QGraphicsItem.__init__(self, parent)
        self.params = params

        self.setZValue(2.)

        # create corners of box
        self.corners = [_ShapeCorner(self, params) for i in range(4)]

        # lines connecting corners
        self.lines = [
            _EdgeLine(self, params, ismovable=params.ismovable)
            for i in range(4)]

        # hide corners if box is not resizable
        if not params.isresizable:
            for c in self.corners:
                c.hide()

        self.updateCornerPosns()

    def updateCornerPosns(self):
        """Update all corners from updated box."""

        par = self.params
        pos = par.posn
        # update cursors
        self.corners[0].setCursor(qt.Qt.CursorShape.SizeFDiagCursor)
        self.corners[1].setCursor(qt.Qt.CursorShape.SizeBDiagCursor)
        self.corners[2].setCursor(qt.Qt.CursorShape.SizeBDiagCursor)
        self.corners[3].setCursor(qt.Qt.CursorShape.SizeFDiagCursor)

        # trim box to maximum size
        pos[0] = max(pos[0], par.maxposn[0])
        pos[1] = max(pos[1], par.maxposn[1])
        pos[2] = min(pos[2], par.maxposn[2])
        pos[3] = min(pos[3], par.maxposn[3])

        # move corners
        for corner, (xindex, yindex) in zip(
                self.corners, self.mapcornertoposn):
            corner.setScaledPos(pos[xindex], pos[yindex])

        # move lines
        w, h = pos[2]-pos[0], pos[3]-pos[1]
        self.lines[0].setScaledLinePos(pos[0], pos[1], pos[0]+w, pos[1])
        self.lines[1].setScaledLinePos(pos[2], pos[1], pos[2], pos[1]+h)
        self.lines[2].setScaledLinePos(pos[2], pos[3], pos[2]-w, pos[3])
        self.lines[3].setScaledLinePos(pos[0], pos[3], pos[0], pos[3]-h)

    def updateFromLine(self, line, thispos):
        """Edge line of box was moved - update bounding box."""

        par = self.params
        # need old coordinate to work out how far line has moved
        try:
            li = self.lines.index(line)
        except ValueError:
            return
        ox = par.posn[ (0, 2, 2, 0)[li] ]
        oy = par.posn[ (1, 1, 3, 3)[li] ]

        # add on deltas to box coordinates
        dx, dy = thispos.x()-ox, thispos.y()-oy

        # make sure box can't be moved outside the allowed region
        if dx > 0:
            dx = min(dx, par.maxposn[2]-par.posn[2])
        else:
            dx = -min(abs(dx), abs(par.maxposn[0]-par.posn[0]))
        if dy > 0:
            dy = min(dy, par.maxposn[3]-par.posn[3])
        else:
            dy = -min(abs(dy), abs(par.maxposn[1]-par.posn[1]))

        # move the box
        par.posn[0] += dx
        par.posn[1] += dy
        par.posn[2] += dx
        par.posn[3] += dy

        # update corner coords and other line coordinates
        self.updateCornerPosns()

    def updateFromCorner(self, corner, event):
        """Move corner of box to new position."""
        try:
            index = self.corners.index(corner)
        except ValueError:
            return

        pos = self.params.posn
        pos[ self.mapcornertoposn[index][0] ] = corner.scaledX()
        pos[ self.mapcornertoposn[index][1] ] = corner.scaledY()

        # this is needed if the corners move past each other
        if pos[0] > pos[2]:
            # swap x
            pos[0], pos[2] = pos[2], pos[0]
            self.corners[0], self.corners[1] = self.corners[1], self.corners[0]
            self.corners[2], self.corners[3] = self.corners[3], self.corners[2]
        if pos[1] > pos[3]:
            # swap y
            pos[1], pos[3] = pos[3], pos[1]
            self.corners[0], self.corners[2] = self.corners[2], self.corners[0]
            self.corners[1], self.corners[3] = self.corners[3], self.corners[1]

        self.updateCornerPosns()

    def boundingRect(self):
        return qt.QRectF(0, 0, 0, 0)

    def paint(self, painter, option, widget):
        pass

    def updateWidget(self):
        """Update widget margins."""
        self.params.widget.updateControlItem(self.params)

##############################################################################

class ControlResizableBox:
    """Control a resizable box.
    Item resizes centred around a position
    """

    def __init__(self, widget, phelper, posn, dims, angle, allowrotate=False):
        """Initialise with widget and boxbounds shape.
        Rotation is allowed if allowrotate is set
        """
        self.widget = widget
        self.posn = posn
        self.dims = dims
        self.angle = angle
        self.allowrotate = allowrotate
        self.cgscale = phelper.cgscale

    def createGraphicsItem(self, parent):
        return _GraphResizableBox(parent, self)

class _GraphResizableBox(qt.QGraphicsItem):
    """Control a resizable box.
    Item resizes centred around a position
    """

    def __init__(self, parent, params):
        """Initialise with widget and boxbounds shape.
        Rotation is allowed if allowrotate is set
        """

        qt.QGraphicsItem.__init__(self, parent)
        self.params = params

        # create child graphicsitem for each corner
        self.corners = [_ShapeCorner(self, params) for i in range(4)]
        self.corners[0].setCursor(qt.Qt.CursorShape.SizeFDiagCursor)
        self.corners[1].setCursor(qt.Qt.CursorShape.SizeBDiagCursor)
        self.corners[2].setCursor(qt.Qt.CursorShape.SizeBDiagCursor)
        self.corners[3].setCursor(qt.Qt.CursorShape.SizeFDiagCursor)
        for c in self.corners:
            c.setToolTip(_('Hold shift to resize symmetrically'))

        # lines connecting corners
        self.lines = [
            _EdgeLine(self, params, ismovable=True) for i in range(4)]

        # whether box is allowed to be rotated
        self.rotator = None
        if params.allowrotate:
            self.rotator = _ShapeCorner(self, params, rotator=True)
            self.rotator.setCursor(qt.Qt.CursorShape.CrossCursor)

        self.updateCorners()

    def updateFromCorner(self, corner, event):
        """Take position and update corners."""

        par = self.params

        x = corner.scaledX()-par.posn[0]
        y = corner.scaledY()-par.posn[1]

        if corner in self.corners:
            # rotate position back
            angle = -par.angle/180.*math.pi
            s, c = math.sin(angle), math.cos(angle)
            tx = x*c-y*s
            ty = x*s+y*c

            if event.modifiers() & qt.Qt.KeyboardModifier.ShiftModifier:
                # expand around centre
                par.dims[0] = abs(tx*2)
                par.dims[1] = abs(ty*2)
            else:
                # moved distances of corner point
                mdx = par.dims[0]*0.5 - abs(tx)
                mdy = par.dims[1]*0.5 - abs(ty)

                # The direction to move the centre depends on which corner
                # it is. This makes the other side of the box stay in the
                # same place.
                signx = 1 if tx<0 else -1
                signy = 1 if ty<0 else -1

                # compute how much to move box centre by
                dx = 0.5*signx*mdx
                dy = 0.5*signy*mdy
                rdx =  dx*c+dy*s  # rotate forwards again
                rdy = -dx*s+dy*c

                par.posn[0] += rdx
                par.posn[1] += rdy
                par.dims[0] -= mdx
                par.dims[1] -= mdy

        elif corner is self.rotator:
            # work out angle relative to centre of widget
            angle = math.atan2(y, x)
            # change to degrees from correct direction
            par.angle = round((angle*RAD2DEG + 90.) % 360, 2)

        self.updateCorners()

    def updateCorners(self):
        """Update corners on size."""
        par = self.params

        # update corners
        angle = par.angle*DEG2RAD
        s, c = math.sin(angle), math.cos(angle)

        for corn, (xd, yd) in zip(
                self.corners, ((-1, -1), (1, -1), (-1, 1), (1, 1))):
            dx, dy = xd*par.dims[0]*0.5, yd*par.dims[1]*0.5
            corn.setScaledPos(
                dx*c-dy*s + par.posn[0],
                dx*s+dy*c + par.posn[1])

        if self.rotator:
            # set rotator position (constant distance)
            dx, dy = 0, -par.dims[1]*0.5
            nx = dx*c-dy*s
            ny = dx*s+dy*c
            self.rotator.setScaledPos(nx+par.posn[0], ny+par.posn[1])

        self.linepos = []
        corn = self.corners
        for i, (ci1, ci2) in enumerate(((0, 1), (2, 0), (1, 3), (2, 3))):
            pos1 = corn[ci1].scaledX(), corn[ci1].scaledY()
            self.lines[i].setScaledLinePos(
                pos1[0], pos1[1],
                corn[ci2].scaledX(), corn[ci2].scaledY())
            self.linepos.append(pos1)

    def updateFromLine(self, line, thispos):
        """Edge line of box was moved - update bounding box."""

        # need old coordinate to work out how far line has moved
        oldpos = self.linepos[self.lines.index(line)]

        dx = line.scaledX() - oldpos[0]
        dy = line.scaledY() - oldpos[1]
        self.params.posn[0] += dx
        self.params.posn[1] += dy

        # update corner coords and other line coordinates
        self.updateCorners()

    def updateWidget(self):
        """Tell the user the graphicsitem has been moved or resized."""
        self.params.widget.updateControlItem(self.params)

    def boundingRect(self):
        """Intentionally zero bounding rect."""
        return qt.QRectF(0, 0, 0, 0)

    def paint(self, painter, option, widget):
        """Intentionally empty painter."""

##############################################################################

class ControlMovableBox(ControlMarginBox):
    """Item for user display for controlling widget.
    This is a dotted movable box with an optional "cross" where
    the real position of the widget is
    """

    def __init__(self, widget, posn, painthelper, crosspos=None):
        ControlMarginBox.__init__(
            self, widget, posn,
            [-10000, -10000, 10000, 10000],
            painthelper, isresizable=False
        )
        self.deltacrosspos = (
            crosspos[0] - self.posn[0],
            crosspos[1] - self.posn[1]
        )

    def createGraphicsItem(self, parent):
        return _GraphMovableBox(parent, self)

class _GraphMovableBox(_GraphMarginBox):
    def __init__(self, parent, params):
        _GraphMarginBox.__init__(self, parent, params)
        self.cross = _ShapeCorner(self, params)
        self.cross.setCursor(qt.Qt.CursorShape.SizeAllCursor)
        self.updateCornerPosns()

    def updateCornerPosns(self):
        _GraphMarginBox.updateCornerPosns(self)

        par = self.params
        if hasattr(self, 'cross'):
            # this fails if called before self.cross is initialised!
            self.cross.setScaledPos(
                par.deltacrosspos[0] + par.posn[0],
                par.deltacrosspos[1] + par.posn[1])

    def updateFromCorner(self, corner, event):
        if corner == self.cross:
            # if cross moves, move whole box
            par = self.params
            cx, cy = self.cross.scaledX(), self.cross.scaledY()
            dx = cx - (par.deltacrosspos[0] + par.posn[0])
            dy = cy - (par.deltacrosspos[1] + par.posn[1])

            par.posn[0] += dx
            par.posn[1] += dy
            par.posn[2] += dx
            par.posn[3] += dy
            self.updateCornerPosns()
        else:
            _GraphMarginBox.updateFromCorner(self, corner, event)

##############################################################################

class ControlLine:
    """For controlling the position and ends of a line."""
    def __init__(self, widget, phelper, x1, y1, x2, y2):
        self.widget = widget
        self.line = x1, y1, x2, y2
        self.cgscale = phelper.cgscale

    def createGraphicsItem(self, parent):
        return _GraphLine(parent, self)

class _GraphLine(qt.QGraphicsLineItem, _ScaledShape):
    """Represents the line as a graphics item."""
    def __init__(self, parent, params):
        qt.QGraphicsLineItem.__init__(self, parent)
        self.params = params
        l = self.params.line
        self.setScaledLine(l[0], l[1], l[2], l[3])
        self.setCursor(qt.Qt.CursorShape.SizeAllCursor)
        self.setFlag(qt.QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setPen(controlLinePen())
        self.setZValue(1.)

        self.p0 = _ShapeCorner(self, params, rotator=True)
        self.p0.setScaledPos(params.line[0], params.line[1])
        self.p0.setCursor(qt.Qt.CursorShape.CrossCursor)
        self.p1 = _ShapeCorner(self, params, rotator=True)
        self.p1.setScaledPos(params.line[2], params.line[3])
        self.p1.setCursor(qt.Qt.CursorShape.CrossCursor)

    def updateFromCorner(self, corner, event):
        """Take position and update ends of line."""
        c = (self.p0.scaledX(), self.p0.scaledY(),
             self.p1.scaledX(), self.p1.scaledY())
        self.setScaledLine(*c)

    def mouseReleaseEvent(self, event):
        """If widget has moved, tell it."""
        qt.QGraphicsItem.mouseReleaseEvent(self, event)
        self.updateWidget()

    def updateWidget(self):
        """Update caller with position and line positions."""

        x, y = self.scaledX(), self.scaledY()
        pt1 = self.p0.scaledX()+x, self.p0.scaledY()+y
        pt2 = self.p1.scaledX()+x, self.p1.scaledY()+y

        self.params.widget.updateControlItem(self.params, pt1, pt2)

#############################################################################

class _AxisGraphicsLineItem(qt.QGraphicsLineItem, _ScaledShape):
    def __init__(self, parent, params):
        qt.QGraphicsLineItem.__init__(self, parent)
        self.parent = parent
        self.params = params

        self.setPen(controlLinePen())
        self.setZValue(2.)
        self.setFlag(qt.QGraphicsItem.GraphicsItemFlag.ItemIsMovable)

    def mouseReleaseEvent(self, event):
        """Notify finished."""
        qt.QGraphicsLineItem.mouseReleaseEvent(self, event)
        self.parent.updateWidget()

    def mouseMoveEvent(self, event):
        """Move the axis."""
        qt.QGraphicsLineItem.mouseMoveEvent(self, event)
        self.parent.doLineUpdate()

class ControlAxisLine:
    """Controlling position of an axis."""

    def __init__(self, widget, painthelper, direction,
                 minpos, maxpos, axispos, maxposn):
        self.widget = widget
        self.direction = direction
        if minpos > maxpos:
            minpos, maxpos = maxpos, minpos
        self.minpos = self.minzoom = self.minorig = minpos
        self.maxpos = self.maxzoom = self.maxorig = maxpos
        self.axisorigpos = self.axispos = axispos
        self.maxposn = maxposn
        self.cgscale = painthelper.cgscale
        self.reset = None

    def done_reset(self):
        return self.reset is not None

    def zoomed(self):
        """Is this a zoom?"""
        return self.minzoom != self.minorig or self.maxzoom != self.maxorig

    def moved(self):
        """Has axis moved?"""
        return (
            self.minpos != self.minorig or self.maxpos != self.maxorig or
            self.axisorigpos != self.axispos
        )

    def createGraphicsItem(self, parent):
        return _GraphAxisLine(parent, self)

class _AxisRange(_ShapeCorner):
    """A control item which allows double click for a reset."""

    def mouseDoubleClickEvent(self, event):
        qt.QGraphicsRectItem.mouseDoubleClickEvent(self, event)
        self.parentItem().resetRange(self)

class _GraphAxisLine(qt.QGraphicsItem):

    # cursors to use
    curs = {
        True: qt.Qt.CursorShape.SizeVerCursor,
        False: qt.Qt.CursorShape.SizeHorCursor
    }
    curs_zoom = {
        True: qt.Qt.CursorShape.SplitVCursor,
        False: qt.Qt.CursorShape.SplitHCursor
    }

    def __init__(self, parent, params):
        """Line is about to be shown."""
        qt.QGraphicsItem.__init__(self, parent)
        self.params = params
        self.pts = [
            _ShapeCorner(self, params), _ShapeCorner(self, params),
            _AxisRange(self, params), _AxisRange(self, params)
        ]
        self.line = _AxisGraphicsLineItem(self, params)

        # set cursors and tooltips for items
        self.horz = (params.direction == 'horizontal')
        for p in self.pts[0:2]:
            p.setCursor(self.curs[not self.horz])
            p.setToolTip("Move axis ends")
        for p in self.pts[2:]:
            p.setCursor(self.curs_zoom[not self.horz])
            p.setToolTip("Change axis scale. Double click to reset end.")
        self.line.setCursor( self.curs[self.horz] )
        self.line.setToolTip("Move axis position")
        self.setZValue(2.)

        self.updatePos()

    def updatePos(self):
        """Set ends of line and line positions from stored values."""
        par = self.params
        scaling = par.cgscale
        mxp = par.maxposn

        def _clip(*args):
            """Clip positions to bounds of box given coords."""
            par.minpos = max(par.minpos, mxp[args[0]])
            par.maxpos = min(par.maxpos, mxp[args[1]])
            par.axispos = max(par.axispos, mxp[args[2]])
            par.axispos = min(par.axispos, mxp[args[3]])

        # distance zoom boxes offset from axis
        offset = 15/scaling

        if self.horz:
            _clip(0, 2, 1, 3)

            # set positions
            if par.zoomed():
                self.line.setScaledPos(par.minzoom, par.axispos)
                self.line.setScaledLine(0, 0, par.maxzoom-par.minzoom, 0)
            else:
                self.line.setScaledPos(par.minpos, par.axispos)
                self.line.setScaledLine(0, 0, par.maxpos-par.minpos, 0)
            self.pts[0].setScaledPos(par.minpos, par.axispos)
            self.pts[1].setScaledPos(par.maxpos, par.axispos)
            self.pts[2].setScaledPos(par.minzoom, par.axispos-offset)
            self.pts[3].setScaledPos(par.maxzoom, par.axispos-offset)
        else:
            _clip(1, 3, 0, 2)

            # set positions
            if par.zoomed():
                self.line.setScaledPos(par.axispos, par.minzoom)
                self.line.setScaledLine(0, 0, 0, par.maxzoom-par.minzoom)
            else:
                self.line.setScaledPos(par.axispos, par.minpos)
                self.line.setScaledLine(0, 0, 0, par.maxpos-par.minpos)
            self.pts[0].setScaledPos(par.axispos, par.minpos)
            self.pts[1].setScaledPos(par.axispos, par.maxpos)
            self.pts[2].setScaledPos(par.axispos+offset, par.minzoom)
            self.pts[3].setScaledPos(par.axispos+offset, par.maxzoom)

    def updateFromCorner(self, corner, event):
        """Ends of axis have moved, so update values."""

        par = self.params
        pt = (corner.scaledY(), corner.scaledX())[self.horz]
        # which end has moved?
        if corner is self.pts[0]:
            # horizonal or vertical axis?
            par.minpos = pt
        elif corner is self.pts[1]:
            par.maxpos = pt
        elif corner is self.pts[2]:
            par.minzoom = pt
        elif corner is self.pts[3]:
            par.maxzoom = pt

        # swap round end points if min > max
        if par.minpos > par.maxpos:
            par.minpos, par.maxpos = par.maxpos, par.minpos
            self.pts[0], self.pts[1] = self.pts[1], self.pts[0]

        self.updatePos()

    def doLineUpdate(self):
        """Line has moved, so update position."""
        if self.horz:
            self.params.axispos = self.line.scaledY()
        else:
            self.params.axispos = self.line.scaledX()
        self.updatePos()

    def updateWidget(self):
        """Tell widget to update."""
        self.params.widget.updateControlItem(self.params)

    def boundingRect(self):
        """Intentionally zero bounding rect."""
        return qt.QRectF(0, 0, 0, 0)

    def paint(self, painter, option, widget):
        """Intentionally empty painter."""

    def resetRange(self, corner):
        """User wants to reset range."""

        if corner is self.pts[2]:
            self.params.reset = 0
        else:
            self.params.reset = 1
        self.updateWidget()



############################################################

class _SceneEdgeLine(qt.QGraphicsLineItem):

    def __init__(self, parent, params, axis=None):
        qt.QGraphicsLineItem.__init__(self, parent)
        self.params = params
        pen = controlLinePen()
        if axis is not None:
            pen.setWidth(4)
            self.text = qt.QGraphicsSimpleTextItem(axis, self)
            self.text.setBrush(qt.QBrush(pen.color()))
            self.text.setFont(qt.QFont("sans", 16))
        else:
            self.text = None

        self.setPen(pen)
        self.setZValue(2.)

    def setScaledLine(self, x1, y1, x2, y2):
        s = self.params.cgscale
        self.setPos(x1*s, y1*s)
        self.setRotation(-qt.QLineF(x1, y1, x2, y2).angle())
        linelen = math.sqrt((x2-x1)**2+(y2-y1)**2)*s
        self.setLine(0, 0, linelen, 0)
        if self.text is not None:
            self.text.setPos(linelen, 0)

class _SvgRotItem(qt.QGraphicsSvgItem, _ScaledShape):
    """Draw an SVG item."""
    def __init__(self, filename, parent, mode):
        qt.QGraphicsSvgItem.__init__(
            self, os.path.join(utils.imagedir, filename), parent)
        self.setFlag(qt.QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setZValue(3.)
        self.mode = mode

    def mouseMoveEvent(self, event):
        """Notify parent on move."""
        qt.QGraphicsSvgItem.mouseMoveEvent(self, event)
        self.parentItem().updateMovedCntrl(self, event, self.mode)

    def mouseReleaseEvent(self, event):
        """Notify parent on unclicking."""
        qt.QGraphicsSvgItem.mouseReleaseEvent(self, event)
        self.parentItem().updateWidget()

def rotM_to_angles(rotM):
    """Convert a rotation matrix to its constituent Euler angles (deg)."""

    # Note: uses atan2 to get correct quadrants, and for theta_y, uses
    # larger denominator to avoid division by zero

    theta_z = math.atan2(rotM.get(1,0),rotM.get(0,0))
    theta_x = math.atan2(rotM.get(2,1),rotM.get(2,2))
    sz, cz = math.sin(theta_z), math.cos(theta_z)
    if abs(cz) > abs(sz):
        theta_y = math.atan2(-rotM.get(2,0),rotM.get(0,0)/cz)
    else:
        theta_y = math.atan2(-rotM.get(2,0),rotM.get(1,0)/sz)

    return (theta_x*RAD2DEG, theta_y*RAD2DEG, theta_z*RAD2DEG)

class _SceneRotationItem(qt.QGraphicsItem):
    """For controlling the rotation of a 3D scene."""

    def __init__(self, parent, params):

        qt.QGraphicsItem.__init__(self, parent)
        self.params = params
        self.cgscale = params.cgscale

        angles = params.angles

        # the current rotation matrix
        self.rotM = threed.rotate3M4(
            angles[0]*DEG2RAD, angles[1]*DEG2RAD, angles[2]*DEG2RAD)

        self.setZValue(2.)

        self.boxpts = []
        boxvecs = []
        rotvecs = []
        for dx, dy, dz in (
                (-0.5,-0.5,-0.5), (+0.5,-0.5,-0.5),
                (-0.5,+0.5,-0.5), (+0.5,+0.5,-0.5),
                (-0.5,-0.5,+0.5), (+0.5,-0.5,+0.5),
                (-0.5,+0.5,+0.5), (+0.5,+0.5,+0.5), ):

            invec = threed.Vec4(dx, dy, dz, 1)
            self.boxpts.append(threed.Vec4(dx, dy, dz, 1))

        # make rotation controls
        posn = params.posn

        s = self.params.cgscale
        bx, by = posn[0]*s, posn[1]*s

        cntrl = _SvgRotItem("veusz-arrow-nesw.svg", self, "xy")
        cntrl.setCursor(qt.Qt.CursorShape.SizeAllCursor)
        cntrl.setPos(bx, by)
        cntrl.setToolTip(_(
            "Click and drag to rotate in x and y (hold Ctrl for x and z)"))

        cntrl = _SvgRotItem("veusz-arrow-ns.svg", self, "y")
        cntrl.setCursor(qt.Qt.CursorShape.SizeVerCursor)
        cntrl.setPos(bx+20, by)
        cntrl.setToolTip(_(
            "Click and drag to rotate in y"))

        cntrl = _SvgRotItem("veusz-arrow-ew.svg", self, "x")
        cntrl.setCursor(qt.Qt.CursorShape.SizeHorCursor)
        cntrl.setPos(bx+40, by)
        cntrl.setToolTip(_(
            "Click and drag to rotate in x"))

        cntrl = _SvgRotItem("veusz-arrow-circ.svg", self, "z")
        cntrl.setCursor(qt.Qt.CursorShape.SizeBDiagCursor)
        cntrl.setPos(bx+68, by)
        cntrl.setToolTip(_(
            "Click and drag to rotate in z"))

        # lines for the box
        axes = {0:'x', 3:'y', 8:'z'}
        self.lines = []
        for i in range(12):
            line = _SceneEdgeLine(self, self.params, axis=axes.get(i))
            self.lines.append(line)

        self.updatePositions()

    def updatePositions(self):
        combM = self.params.camM * self.rotM

        points = []
        for v in self.boxpts:
            proj = threed.vec4to3(combM * v)
            sp = threed.projVecToScreen(self.params.screenM, proj)
            point = qt.QPointF(sp.get(0), sp.get(1))
            points.append(point)

        idxs = (
            (0,1), (1,3), (3,2), (0,2),
            (4,5), (5,7), (7,6), (6,4),
            (0,4), (1,5), (2,6), (3,7),
        )
        for i, (i0, i1) in enumerate(idxs):
            self.lines[i].setScaledLine(
                points[i0].x(), points[i0].y(),
                points[i1].x(), points[i1].y())

    def boundingRect(self):
        """Intentionally zero bounding rect."""
        return qt.QRectF(0, 0, 0, 0)

    def paint(self, painter, option, widget):
        """Intentionally empty painter."""

    def updateMovedCntrl(self, corner, event, mode):
        """Rotate given moving control."""

        event.accept()
        newpos = event.screenPos()
        oldpos = event.lastScreenPos()
        if newpos == oldpos:
            return

        delta = newpos-oldpos

        if mode == 'xy':
            if (int(event.modifiers()) & qt.Qt.KeyboardModifier.ControlModifier) == 0:
                # rotate in x,y axes on screen
                deltaM = threed.rotate3M4(
                    -delta.y()*DEG2RAD, -delta.x()*DEG2RAD, 0)
            else:
                # if control is pressed, rotate along z axis for y direction
                deltaM = threed.rotate3M4(
                    -delta.x()*DEG2RAD, 0, -delta.y()*DEG2RAD)
        elif mode == 'y':
            deltaM = threed.rotate3M4(-delta.y()*DEG2RAD, 0, 0)
        elif mode == 'x':
            deltaM = threed.rotate3M4(0, -delta.x()*DEG2RAD, 0)
        elif mode == 'z':
            deltaM = threed.rotate3M4(0, 0, -delta.x()*DEG2RAD)

        self.rotM = deltaM * self.rotM

        self.updatePositions()

    def updateWidget(self):
        """Update widget angles."""

        # get angles for rotation matrix
        tx, ty, tz = rotM_to_angles(self.rotM)

        # update in document
        s = self.params.scene.settings
        operations = (
            document.OperationSettingSet(s.get('xRotation'), round(tx,1)),
            document.OperationSettingSet(s.get('yRotation'), round(ty,1)),
            document.OperationSettingSet(s.get('zRotation'), round(tz,1)),
        )
        self.params.scene.document.applyOperation(
            document.OperationMultiple(operations, descr=_('rotate scene')))

class ControlSceneRotation:
    def __init__(self, posn, scene, camM, screenM, angles, painthelper):
        """Control rotation of scene
        """

        self.posn = posn
        self.scene = scene
        self.camM = camM
        self.screenM = screenM
        self.angles = angles
        self.cgscale = painthelper.cgscale

    def createGraphicsItem(self, parent):
        """Make the box and corner control graphs."""

        return _SceneRotationItem(parent, self)
