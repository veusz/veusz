import abc
import enum

import numpy as np
from numpy.typing import NDArray

from ... import qtall as qt
from ...document.painthelper import RecordPainter


# Cohen-Sutherland line clipping algorithm


class Endpoint(enum.Flag):
    """codes used classify endpoints are combinations of these"""

    NONE = 0
    LEFT = 1
    RIGHT = 2
    TOP = 4
    BOTTOM = 8


def horzIntersection(yval: float, pt1: qt.QPointF, pt2: qt.QPointF) -> qt.QPointF:
    """compute intersection with horizontal line"""
    if pt1.y() == pt2.y():
        # line is vertical
        return qt.QPointF(pt1.x(), yval)
    else:
        return qt.QPointF(
            pt1.x() + (yval - pt1.y()) * (pt2.x() - pt1.x()) / (pt2.y() - pt1.y()),
            yval,
        )


def vertIntersection(xval: float, pt1: qt.QPointF, pt2: qt.QPointF) -> qt.QPointF:
    """compute intersection with vertical line"""
    if pt1.x() == pt2.x():
        # line is horizontal
        return qt.QPointF(xval, pt1.y())
    else:
        return qt.QPointF(
            xval,
            pt1.y() + (xval - pt1.x()) * (pt2.y() - pt1.y()) / (pt2.x() - pt1.x()),
        )


def smallDelta(pt1: qt.QPointF, pt2: qt.QPointF) -> bool:
    """is difference between points very small?"""
    if pt1.isNull() or pt2.isNull():
        return False
    return abs(pt1.x() - pt2.x()) < 0.01 and abs(pt1.y() - pt2.y()) < 0.01


class _Clipper:
    """private class for helping clip"""

    def __init__(self, cliprect: qt.QRectF) -> None:
        self.clip: qt.QRectF = cliprect

    def computeCode(self, pt: qt.QPointF) -> Endpoint:
        """compute the Cohen-Sutherland code"""
        code: Endpoint = Endpoint.NONE
        if pt.x() < self.clip.left():
            code |= Endpoint.LEFT
        elif pt.x() > self.clip.right():
            code |= Endpoint.RIGHT
        if pt.y() < self.clip.top():
            code |= Endpoint.TOP
        elif pt.y() > self.clip.bottom():
            code |= Endpoint.BOTTOM
        return code

    def fixPt(self, pt: qt.QPointF) -> None:
        """
        get consistent clipping on different platforms by making line
        edges meet clipping box if close
        """
        if abs(pt.x() - self.clip.left()) < 1e-4:
            pt.setX(self.clip.left())
        if abs(pt.x() - self.clip.right()) < 1e-4:
            pt.setX(self.clip.right())
        if abs(pt.y() - self.clip.top()) < 1e-4:
            pt.setY(self.clip.top())
        if abs(pt.y() - self.clip.bottom()) < 1e-4:
            pt.setY(self.clip.bottom())

    def clipLine(self, pt1: qt.QPointF, pt2: qt.QPointF) -> bool:
        """modifies points, returning true if okay to accept"""
        # fixup ends to meet np_clip box if close
        self.fixPt(pt1)
        self.fixPt(pt2)

        code1: Endpoint = self.computeCode(pt1)
        code2: Endpoint = self.computeCode(pt2)

        # repeat until points are at edge of box
        # bail out if we repeat too many times (avoid numerical issues)
        for count in range(16):
            if ~(code1 | code2):
                # no more clipping required - inside
                return True
            elif code1 & code2:
                # line should not be drawn - outside
                return False
            else:
                # compute intersection

                # which point to compute for?
                code: Endpoint = code1 or code2

                # get intersection new point and new code for line
                pt: qt.QPointF = qt.QPointF()
                if Endpoint.LEFT in code:
                    pt = vertIntersection(self.clip.left(), pt1, pt2)
                elif Endpoint.RIGHT in code:
                    pt = vertIntersection(self.clip.right(), pt1, pt2)
                elif Endpoint.TOP in code:
                    pt = horzIntersection(self.clip.top(), pt1, pt2)
                elif Endpoint.BOTTOM in code:
                    pt = horzIntersection(self.clip.bottom(), pt1, pt2)

                # update point as intersection
                if code == code1:
                    # changed first point
                    pt1 = pt
                    code1 = self.computeCode(pt1)
                else:
                    # changed second point
                    pt2 = pt
                    code2 = self.computeCode(pt2)
        return False


def clipLine(clip: qt.QRectF, pt1: qt.QPointF, pt2: qt.QPointF) -> bool:
    """
    clip a line made up of the points given, returning True
    if is in region or False if not
    """
    clipper = _Clipper(clip)
    return clipper.clipLine(pt1, pt2)


class _PolyClipper:
    """
    This class is used to separate out the clipping of polylines
    override emit_polyline to handle the clipped part of the original
    polyline
    """

    def __init__(self, clip: qt.QRectF) -> None:
        self._clipper = _Clipper(clip)

    # override this to draw the output polylines
    @abc.abstractmethod
    def emitPolyline(self, poly: list[qt.QPointF] | qt.QPolygonF) -> None: ...

    # do clipping on the polyline
    def clipPolyline(self, poly: list[qt.QPointF] | qt.QPolygonF) -> None:
        # exit if fewer than 2 points in polygon
        if len(poly) < 2:
            return

        # output goes here
        pout: list[qt.QPointF] = []

        lastpt: qt.QPointF = poly[0]
        for p2 in list(poly)[1:]:
            p1: qt.QPointF = lastpt

            plotline: bool = self._clipper.clipLine(p1, p2)
            if plotline:
                if not pout:
                    # add first line
                    pout.append(p1)
                    if not smallDelta(p1, p2):
                        pout.append(p2)
                else:
                    if p1 == pout[-1]:
                        if not smallDelta(p1, p2):
                            # extend polyline
                            pout.append(p2)
                    else:
                        # paint existing line
                        if len(pout) >= 2:
                            self.emitPolyline(pout)

                        # start new line
                        pout = [p1]
                        if not smallDelta(p1, p2):
                            pout.append(p2)
            else:
                # line isn't in region, so ignore results from clip function

                # paint existing line
                if len(pout) >= 2:
                    self.emitPolyline(pout)

                # cleanup
                pout.clear()

            lastpt = p2

        if len(pout) >= 2:
            self.emitPolyline(pout)


class PlotDrawCallback(_PolyClipper):
    def __init__(self, clip: qt.QRectF, painter: RecordPainter) -> None:
        super().__init__(clip)
        self._painter: RecordPainter = painter

    def emitPolyline(self, poly: list[qt.QPointF]) -> None:
        # BUG PYSIDE-3002: only the 1st point is passed further in PySide6
        polygon: qt.QPolygonF = qt.QPolygonF()
        for point in poly:
            polygon.append(point)
        self._painter.drawPolyline(polygon)


def plotClippedPolyline(
    painter: RecordPainter,
    clip: qt.QRectF,
    poly: list[qt.QPointF] | qt.QPolygonF,
    auto_expand: bool = True,
) -> None:
    """take polyline and paint to painter, clipping"""
    # if auto_expand, expand rectangle by line width
    if auto_expand:
        lw: float = painter.pen().widthF()
        clip.adjust(-lw, -lw, lw, lw)

    pcb: PlotDrawCallback = PlotDrawCallback(clip, painter)
    pcb.clipPolyline(poly)


class PolyAddCallback(_PolyClipper):
    """clip polyline and add polylines clipped to a vector"""

    def __init__(self, clip: qt.QRectF) -> None:
        super().__init__(clip)

        self.polys: list[list[qt.QPointF]] = []

    def emitPolyline(self, poly: list[qt.QPointF]) -> None:
        self.polys.append(poly)


def clipPolyline(clip: qt.QRectF, poly: list[qt.QPointF]) -> list[list[qt.QPointF]]:
    """
    clip polyline to rectangle
    return list of lines to plot
    """
    pcb: PolyAddCallback = PolyAddCallback(clip)
    pcb.clipPolyline(poly)
    return pcb.polys


class _LineLabClipper(_PolyClipper):
    def __init__(self, cliprect: qt.QRectF, polyvec: list[list[qt.QPointF]]) -> None:
        super().__init__(cliprect)
        self._polyvec: list[list[qt.QPointF]] = polyvec

    def emitPolyline(self, poly: list[qt.QPointF]) -> None:
        self._polyvec.append(poly)


def doPolygonsIntersect(a: qt.QPolygonF, b: qt.QPolygonF) -> bool:
    """
    Check whether polygons intersect
    credit: http://stackoverflow.com/questions/10962379/how-to-check-intersection-between-2-rotated-rectangles

    note: requires clockwise polygons
    """
    for poly in (a, b):
        prev_pt = qt.QPointF(poly.last())

        for curr_pt in poly:
            # normal to line segment
            norm_x: float = curr_pt.y() - prev_pt.y()
            norm_y: float = prev_pt.x() - curr_pt.x()

            min_a: float = np.inf
            max_a: float = -np.inf
            for pt in a:
                proj: float = norm_x * pt.x() + norm_y * pt.y()
                min_a = min(min_a, proj)
                max_a = max(max_a, proj)

            min_b: float = np.inf
            max_b: float = -np.inf
            for pt in b:
                proj: float = norm_x * pt.x() + norm_y * pt.y()
                min_b = min(min_b, proj)
                max_b = max(max_b, proj)

            if max_a < min_b or max_b < min_a:
                return False

            prev_pt = curr_pt

    return True


class RotatedRectangle:
    """class for describing a rectangle with a rotation angle"""

    # a lot of boilerplate so it can go in QVector
    def __init__(
        self,
        _cx: float = 0,
        _cy: float = 0,
        _xw: float = 0,
        _yw: float = 0,
        _angle: float = 0,
    ) -> None:
        self.cx = _cx
        self.cy = _cy
        self.xw = _xw
        self.yw = _xw
        self.angle = _angle

    def isValid(self) -> bool:
        return self.xw > 0 and self.yw > 0

    def rotate(self, dtheta) -> None:
        self.angle += dtheta

    def rotateAboutOrigin(self, dtheta: float) -> None:
        """rotate centre as well as increasing angle to rotate about origin"""
        c: float = np.cos(dtheta)
        s: float = np.sin(dtheta)
        tcx: float = self.cx
        tcy: float = self.cy

        self.cx = c * tcx - s * tcy
        self.cy = c * tcy + s * tcx

        self.angle += dtheta

    def translate(self, dx: float, dy: float) -> None:
        self.cx += dx
        self.cy += dy

    def makePolygon(self) -> qt.QPolygonF:
        # note: output polygon is clockwise
        c: float = np.cos(self.angle)
        s: float = np.sin(self.angle)
        xh: float = 0.5 * self.xw
        yh: float = 0.5 * self.yw

        poly: qt.QPolygonF = qt.QPolygonF()
        poly.append(qt.QPointF(-xh * c + yh * s + self.cx, -xh * s - yh * c + self.cy))
        poly.append(qt.QPointF(-xh * c - yh * s + self.cx, -xh * s + yh * c + self.cy))
        poly.append(qt.QPointF(xh * c - yh * s + self.cx, xh * s + yh * c + self.cy))
        poly.append(qt.QPointF(xh * c + yh * s + self.cx, xh * s - yh * c + self.cy))
        return poly


def resampleLinearImage(img: qt.QImage, xpts: NDArray, ypts: NDArray) -> qt.QImage:
    # reversed mode
    revx: bool = xpts[0] > xpts[-1]
    revy: bool = ypts[0] > ypts[-1]

    # get smallest spacing
    mindeltax: float = np.inf
    for i in range(xpts.size - 1):
        mindeltax = min(mindeltax, abs(xpts[i + 1] - xpts[i]))
    mindeltay: float = np.inf
    for i in range(ypts.size - 1):
        mindeltay = min(mindeltay, abs(ypts[i + 1] - ypts[i]))

    # get bounds
    minx: float = xpts[-1] if revx else xpts[0]
    maxx: float = xpts[0] if revx else xpts[-1]
    miny: float = ypts[-1] if revy else ypts[0]
    maxy: float = ypts[0] if revy else ypts[-1]

    # output size (trimmed to 1024)
    sizex: int = min(1024, int((maxx - minx) / (mindeltax * 0.25) + 0.01))
    sizey: int = min(1024, int((maxy - miny) / (mindeltay * 0.25) + 0.01))
    deltax: float = (maxx - minx) / sizex
    deltay: float = (maxy - miny) / sizey

    outimg = qt.QImage(sizex, sizey, img.format())

    # need to account for reverse direction, so count backwards
    xptsbase: int = xpts.size - 1 if revx else 0
    xptsdir: int = -1 if revx else 1
    yptsbase: int = ypts.size - 1 if revy else 0
    yptsdir: int = -1 if revy else 1

    iy: int = 0
    for oy in range(sizey):
        # do we move to the next pixel in y?
        while (
            miny + (oy + 0.5) * deltay > ypts[yptsbase + yptsdir * (iy + 1)]
            and iy < ypts.size - 2
        ):
            iy += 1

        iscanline = img.scanLine(iy)
        oscanline = outimg.scanLine(oy)

        ix: int = 0
        for ox in range(sizex):
            # do we move to the next pixel in x?
            while (
                minx + (ox + 0.5) * deltax > xpts[xptsbase + xptsdir * (ix + 1)]
                and ix < xpts.size - 2
            ):
                ix += 1

            # copy pixel
            oscanline[ox] = iscanline[ix]

    return outimg


class RectangleOverlapTester:
    def __init__(self) -> None:
        self._rectangles: list[RotatedRectangle] = []

    def willOverlap(self, rect: RotatedRectangle) -> bool:
        this_polygon = qt.QPolygonF(rect.makePolygon())

        for ir in self._rectangles:
            if doPolygonsIntersect(this_polygon, ir.makePolygon()):
                return True

        return False

    def addRect(self, rect: RotatedRectangle) -> None:
        self._rectangles.append(rect)

    def reset(self) -> None:
        self._rectangles = []

    # debug by drawing all the rectangles
    def debug(self, painter: qt.QPainter) -> None:
        for rect in self._rectangles:
            poly: qt.QPolygonF = rect.makePolygon()
            painter.drawPolygon(poly)


# these are the positions where labels might be placed
label_positions: list[float] = [0.5, 1 / 3.0, 2 / 3.0, 0.4, 0.6, 0.25, 0.75]


class LineLabeller:
    """for labelling of sets of contour lines"""

    def __init__(self, cliprect: qt.QRectF, rotatelabels: bool) -> None:
        self._cliprect: qt.QRectF = cliprect
        self._rotatelabels: bool = rotatelabels

        self._polys: list[list[qt.QPolygonF]] = []
        self._textsizes: list[qt.QSizeF] = []

    def drawAt(self, idx: int, r: RotatedRectangle) -> None:
        """override this to receive the label to draw"""
        pass

    def addLine(self, poly: list[qt.QPointF], textsize: qt.QSizeF) -> None:
        self._polys.append([])
        self._textsizes.append(textsize)
        clipper: _LineLabClipper = _LineLabClipper(self._cliprect, self._polys[-1])
        clipper.clipPolyline(poly)

    def process(self) -> None:
        rtest: RectangleOverlapTester = RectangleOverlapTester()

        # iterate over each set of polylines
        pv: list[qt.QPolygonF]
        size: qt.QSizeF
        for polyseti, (pv, size) in enumerate(zip(self._polys, self._textsizes)):
            for polyi in pv:
                for posi in label_positions:
                    r: RotatedRectangle = self.findLinePosition(polyi, posi, size)
                    if not r.isValid():
                        break

                    if not rtest.willOverlap(r):
                        self.drawAt(polyseti, r)
                        rtest.addRect(r)
                        break  # only add label once
                # positions
            # polylines in set of polylines
        # sets of polylines

    def getNumPolySets(self) -> int:
        return len(self._polys)

    def getPolySet(self, i: int) -> list[qt.QPolygonF]:
        if 0 <= i < len(self._polys):
            return self._polys[i]
        return []

    def findLinePosition(
        self,
        poly: qt.QPolygonF,
        frac: float,
        size: qt.QSizeF,
    ) -> RotatedRectangle:
        """returns RotatedRectangle with zero size if error"""
        totlength: float = 0
        for i in range(1, poly.size()):
            totlength += np.hypot(
                poly.at(i - 1).x() - poly.at(i).x(),
                poly.at(i - 1).y() - poly.at(i).y(),
            )

        # don't label lines which are too short
        if totlength / 2 < max(size.width(), size.height()):
            return RotatedRectangle()

        # now repeat and stop when we've reached the right length
        length: float = 0
        for i in range(1, poly.size()):
            seglength: float = np.hypot(
                poly.at(i - 1).x() - poly.at(i).x(),
                poly.at(i - 1).y() - poly.at(i).y(),
            )
            if length + seglength >= totlength * frac:
                # interpolate along edge
                fseg: float = (totlength * frac - length) / seglength
                xp: float = poly.at(i - 1).x() * (1 - fseg) + poly.at(i).x() * fseg
                yp: float = poly.at(i - 1).y() * (1 - fseg) + poly.at(i).y() * fseg

                angle: float = (
                    np.arctan2(
                        poly.at(i - 1).y() - poly.at(i).y(),
                        poly.at(i - 1).x() - poly.at(i).x(),
                    )
                    if self._rotatelabels
                    else 0.0
                )
                return RotatedRectangle(xp, yp, size.width(), size.height(), angle)

            length += seglength

        # shouldn't get here
        return RotatedRectangle()
