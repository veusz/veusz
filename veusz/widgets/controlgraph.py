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

from __future__ import division
import math

from ..compat import crange, czip
from .. import qtall as qt
from .. import document
from .. import setting

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
        self.setPen(qt.QPen(qt.Qt.NoPen))
        self.setFlag(qt.QGraphicsItem.ItemIsMovable)
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
    return qt.QPen(setting.settingdb.color('cntrlline'), 2, qt.Qt.DotLine)

class _EdgeLine(qt.QGraphicsLineItem, _ScaledShape):
    """Line used for edges of resizing box."""
    def __init__(self, parent, params, ismovable=True):
        qt.QGraphicsLineItem.__init__(self, parent)
        self.setPen(controlLinePen())
        self.setZValue(2.)
        self.params = params
        if ismovable:
            self.setFlag(qt.QGraphicsItem.ItemIsMovable)
            self.setCursor(qt.Qt.SizeAllCursor)

    def mouseMoveEvent(self, event):
        """Notify parent on move."""
        qt.QGraphicsLineItem.mouseMoveEvent(self, event)
        self.parentItem().updateFromLine(self, self.scaledPos())

    def mouseReleaseEvent(self, event):
        """Notify parent on unclicking."""
        qt.QGraphicsLineItem.mouseReleaseEvent(self, event)
        self.parentItem().updateWidget()

##############################################################################

class ControlMarginBox(object):
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
        self.corners = [_ShapeCorner(self, params) for i in crange(4)]

        # lines connecting corners
        self.lines = [
            _EdgeLine(self, params, ismovable=params.ismovable)
            for i in crange(4)]

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
        self.corners[0].setCursor(qt.Qt.SizeFDiagCursor)
        self.corners[1].setCursor(qt.Qt.SizeBDiagCursor)
        self.corners[2].setCursor(qt.Qt.SizeBDiagCursor)
        self.corners[3].setCursor(qt.Qt.SizeFDiagCursor)

        # trim box to maximum size
        pos[0] = max(pos[0], par.maxposn[0])
        pos[1] = max(pos[1], par.maxposn[1])
        pos[2] = min(pos[2], par.maxposn[2])
        pos[3] = min(pos[3], par.maxposn[3])

        # move corners
        for corner, (xindex, yindex) in czip(self.corners,
                                             self.mapcornertoposn):
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

class ControlResizableBox(object):
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
        self.corners = [_ShapeCorner(self, params) for i in crange(4)]
        self.corners[0].setCursor(qt.Qt.SizeFDiagCursor)
        self.corners[1].setCursor(qt.Qt.SizeBDiagCursor)
        self.corners[2].setCursor(qt.Qt.SizeBDiagCursor)
        self.corners[3].setCursor(qt.Qt.SizeFDiagCursor)
        for c in self.corners:
            c.setToolTip(_('Hold shift to resize symmetrically'))

        # lines connecting corners
        self.lines = [
            _EdgeLine(self, params, ismovable=True) for i in crange(4)]

        # whether box is allowed to be rotated
        self.rotator = None
        if params.allowrotate:
            self.rotator = _ShapeCorner(self, params, rotator=True)
            self.rotator.setCursor(qt.Qt.CrossCursor)

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

            if event.modifiers() & qt.Qt.ShiftModifier:
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
            par.angle = round((angle*(180/math.pi) + 90.) % 360, 2)

        self.updateCorners()

    def updateCorners(self):
        """Update corners on size."""
        par = self.params

        # update corners
        angle = par.angle/180.*math.pi
        s, c = math.sin(angle), math.cos(angle)

        for corn, (xd, yd) in czip(
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
        ControlMarginBox.__init__(self, widget, posn,
                                  [-10000, -10000, 10000, 10000],
                                  painthelper, isresizable=False)
        self.deltacrosspos = (crosspos[0] - self.posn[0],
                              crosspos[1] - self.posn[1])

    def createGraphicsItem(self, parent):
        return _GraphMovableBox(parent, self)

class _GraphMovableBox(_GraphMarginBox):
    def __init__(self, parent, params):
        _GraphMarginBox.__init__(self, parent, params)
        self.cross = _ShapeCorner(self, params)
        self.cross.setCursor(qt.Qt.SizeAllCursor)
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

class ControlLine(object):
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
        self.setCursor(qt.Qt.SizeAllCursor)
        self.setFlag(qt.QGraphicsItem.ItemIsMovable)
        self.setPen(controlLinePen())
        self.setZValue(1.)

        self.p0 = _ShapeCorner(self, params, rotator=True)
        self.p0.setScaledPos(params.line[0], params.line[1])
        self.p0.setCursor(qt.Qt.CrossCursor)
        self.p1 = _ShapeCorner(self, params, rotator=True)
        self.p1.setScaledPos(params.line[2], params.line[3])
        self.p1.setCursor(qt.Qt.CrossCursor)

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
        self.setFlag(qt.QGraphicsItem.ItemIsMovable)

    def mouseReleaseEvent(self, event):
        """Notify finished."""
        qt.QGraphicsLineItem.mouseReleaseEvent(self, event)
        self.parent.updateWidget()

    def mouseMoveEvent(self, event):
        """Move the axis."""
        qt.QGraphicsLineItem.mouseMoveEvent(self, event)
        self.parent.doLineUpdate()

class ControlAxisLine(object):
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

    def zoomed(self):
        """Is this a zoom?"""
        return self.minzoom != self.minorig or self.maxzoom != self.maxorig

    def moved(self):
        """Has axis moved?"""
        return ( self.minpos != self.minorig or self.maxpos != self.maxorig or
                 self.axisorigpos != self.axispos )

    def createGraphicsItem(self, parent):
        return _GraphAxisLine(parent, self)

class _GraphAxisLine(qt.QGraphicsItem):

    curs = {True: qt.Qt.SizeVerCursor,
            False: qt.Qt.SizeHorCursor}
    curs_zoom = {True: qt.Qt.SplitVCursor,
                 False: qt.Qt.SplitHCursor}

    def __init__(self, parent, params):
        """Line is about to be shown."""
        qt.QGraphicsItem.__init__(self, parent)
        self.params = params
        self.pts = [ _ShapeCorner(self, params), _ShapeCorner(self, params),
                     _ShapeCorner(self, params), _ShapeCorner(self, params) ]
        self.line = _AxisGraphicsLineItem(self, params)

        # set cursors and tooltips for items
        self.horz = (params.direction == 'horizontal')
        for p in self.pts[0:2]:
            p.setCursor(self.curs[not self.horz])
            p.setToolTip("Move axis ends")
        for p in self.pts[2:]:
            p.setCursor(self.curs_zoom[not self.horz])
            p.setToolTip("Change axis scale")
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
