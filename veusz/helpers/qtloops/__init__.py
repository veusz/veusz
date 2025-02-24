import numpy as np
from PyQt5.QtGui import QPolygonF
from numpy.typing import NDArray
from packaging.version import Version

from .beziers_qtwrap import (
    bezier_fit_cubic_tight,
    bezier_fit_cubic_multi,
    bezier_fit_cubic_single,
)
from .polygonclip import plotClippedPolygon, polygonClip
from .polylineclip import (
    LineLabeller,
    RectangleOverlapTester,
    RotatedRectangle,
    clipLine,
    clipPolyline,
    plotClippedPolyline,
)
from ... import qtall as qt

__all__ = [
    "plotClippedPolyline",
    "plotClippedPolygon",
    "clipPolyline",
    "polygonClip",
    "scalePath",
    "addNumpyToPolygonF",
    "addNumpyPolygonToPath",
    "plotPathsToPainter",
    "plotLinesToPainter",
    "plotBoxesToPainter",
    "numpyToQImage",
    "applyImageTransparancy",
    "RotatedRectangle",
    "RectangleOverlapTester",
    "LineLabeller",
    "bezier_fit_cubic_tight",
    "bezier_fit_cubic_multi",
    "bezier_fit_cubic_single",
    "addCubicsToPainterPath",
]

# ruff: noqa: N802


def clipval(
    val: int | float | np.float64,
    minv: int | float | np.float64,
    maxv: int | float | np.float64,
) -> int | float | np.float64:
    if val < minv:
        return minv
    if val > maxv:
        return maxv
    return val


def scalePath(path: qt.QPainterPath, scale: float) -> qt.QPainterPath:
    count: int = path.elementCount()
    out: qt.QPainterPath = qt.QPainterPath()
    i: int = 0
    while i < count:
        el: qt.QPainterPath.Element = path.elementAt(i)
        if el.isMoveTo():
            out.moveTo(qt.QPointF(el) * scale)
        elif el.isLineTo():
            out.lineTo(qt.QPointF(el) * scale)
        elif el.isCurveTo():
            out.cubicTo(
                qt.QPointF(el) * scale,
                qt.QPointF(path.elementAt(i + 1)) * scale,
                qt.QPointF(path.elementAt(i + 2)) * scale,
            )
            i += 2
        i += 1
    return out


def addNumpyToPolygonF(
    poly: list[qt.QPointF],
    *d: NDArray[np.float64],
) -> list[qt.QPointF]:
    """add sets of points to a QPolygonF"""

    from .polylineclip import smallDelta

    # iterate over rows until none left
    numcols: int = len(d)
    lastpt = qt.QPointF()
    row: int = 0
    ifany: bool = True
    while ifany:
        ifany = False
        # the numcols-1 makes sure we don't get odd numbers of columns
        for col in range(0, numcols - 1, 2):
            # add point if point in two columns
            if row < d[col].size and row < d[col + 1].size:
                pt: qt.QPointF = qt.QPointF(d[col][row], d[col + 1][row])
                if not smallDelta(pt, lastpt):
                    poly.append(pt)
                    lastpt = pt
                ifany = True
        row += 1
    # exit loop if no more columns

    return poly


def addNumpyPolygonToPath(
    path: qt.QPainterPath,
    clip: qt.QRectF | None = None,
    *d: tuple[
        float | int | NDArray[np.float64],
        float | int | NDArray[np.float64],
        float | int | NDArray[np.float64],
        float | int | NDArray[np.float64],
    ],
) -> qt.QPainterPath:
    """add sets of polygon points to a path"""
    numcols: int = len(d)
    row: int = 0
    ifany: bool = True
    while ifany:
        ifany = False
        # output polygon
        poly: list[qt.QPointF] = []

        # the (numcols - 1) makes sure we don't get odd numbers of columns
        for col in range(0, numcols - 1, 2):
            # add point if point in two columns
            if row < len(d[col]) and row < len(d[col + 1]):
                pt: qt.QPointF = qt.QPointF(d[col][row], d[col + 1][row])
                poly.append(pt)
                ifany = True

        if ifany:
            if clip is not None:
                clipped_poly: qt.QPolygonF = qt.QPolygonF()
                polygonClip(poly, clip, clipped_poly)
                path.addPolygon(clipped_poly)
            else:
                path.addPolygon(poly)
            path.closeSubpath()
        row += 1
        # exit loop if no more columns

    return path


def plotPathsToPainter(
    painter: qt.QPainter,
    path: qt.QPainterPath,
    x: NDArray,
    y: NDArray,
    scaling: NDArray[float] | None = None,
    clip: qt.QRectF | None = None,
    colorimg: qt.QImage | None = None,
    scaleline: bool = False,
) -> None:
    """
    plot paths to painter
    x and y locations are given in x and y
    if scaling is not 0, is an array to scale the data points by
    if colorimg is not 0, is a Nx1 image containing color points for path fills
    clip is a clipping rectangle if set
    """

    from .polylineclip import smallDelta

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    clip_rectangle = qt.QRectF(qt.QPointF(-32767, -32767), qt.QPointF(32767, 32767))
    if clip is not None:
        clip_rectangle.setCoords(*clip.getCoords())
    pathbox: qt.QRectF = path.boundingRect()
    clip_rectangle.adjust(
        pathbox.left(),
        pathbox.top(),
        pathbox.bottom(),
        pathbox.right(),
    )

    # keep track of duplicate points
    lastpt = qt.QPointF(-1e6, -1e6)
    # keep original transformation for restoration after each iteration
    origtrans: qt.QTransform = qt.QTransform(painter.worldTransform())

    # number of iterations
    size: int = min(x.size, y.size)

    # if few color points, trim down number of paths
    if colorimg is not None:
        size = min(size, colorimg.width())
    # too few scaling points
    if scaling is not None:
        size = min(size, scaling.size)

    # draw each path
    pt: qt.QPointF
    for i in range(size):
        pt = qt.QPointF(float(x[i]), float(y[i]))
        if clip_rectangle.contains(pt) and not smallDelta(lastpt, pt):
            painter.translate(pt)

            if colorimg is not None:
                # get color from pixel and create a new brush
                b = qt.QBrush(qt.QColor.fromRgba(colorimg.pixel(i, 0)))
                painter.setBrush(b)

            if scaling is None:
                painter.drawPath(path)
            else:
                # scale point if requested
                s = scaling[i]
                if scaleline:
                    painter.scale(s, s)
                    painter.drawPath(path)
                else:
                    scaled: qt.QPainterPath = scalePath(path, s)
                    painter.drawPath(scaled)

            painter.setWorldTransform(origtrans)
            lastpt = pt


def plotLinesToPainter(
    painter: qt.QPainter,
    x1: NDArray[np.float64],
    y1: NDArray[np.float64],
    x2: NDArray[np.float64],
    y2: NDArray[np.float64],
    clip: qt.QRectF | None = None,
    autoexpand: bool = True,
) -> None:
    maxsize: int = min(x1.size, x2.size, y1.size, y2.size)

    # if autoexpand, expand rectangle by line width
    clip_copy: qt.QRectF = qt.QRectF()
    if clip is not None and autoexpand:
        lw: float = painter.pen().widthF()
        clip_copy.setCoords(*clip.getCoords())
        clip_copy.adjust(-lw, -lw, lw, lw)

    if maxsize != 0:
        for i in range(maxsize):
            pt1: qt.QPointF = qt.QPointF(float(x1[i]), float(y1[i]))
            pt2: qt.QPointF = qt.QPointF(float(x2[i]), float(y2[i]))
            if clip is not None:
                if clipLine(clip_copy, pt1, pt2):
                    painter.drawLine(pt1, pt2)
            else:
                painter.drawLine(pt1, pt2)


def plotBoxesToPainter(
    painter: qt.QPainter,
    x1: NDArray[np.float64],
    y1: NDArray[np.float64],
    x2: NDArray[np.float64],
    y2: NDArray[np.float64],
    clip: qt.QRectF | None = None,
    autoexpand: bool = True,
) -> None:
    """if autoexpand, expand rectangle by line width"""
    clipcopy = qt.QRectF()
    if clip is not None and autoexpand:
        lw: float = painter.pen().widthF()
        clipcopy.setCoords(*clip.getCoords())
        clipcopy.adjust(-lw, -lw, lw, lw)

    maxsize: int = min(x1.size, x2.size, y1.size, y2.size)

    for i in range(maxsize):
        pt1 = qt.QPointF(float(x1[i]), float(y1[i]))
        pt2 = qt.QPointF(float(x2[i]), float(y2[i]))
        rect = qt.QRectF(pt1, pt2)

        if clip is not None and clipcopy.intersects(rect):
            rect = clipcopy.intersected(rect)
        painter.drawRect(rect)


def numpyToQImage(
    imgdata: NDArray[np.float64],
    colors: NDArray[np.int64],
    _forcetrans: bool = False,
) -> qt.QImage:
    # make format use alpha transparency if required
    numcolors: int = colors.shape[0]
    if colors.shape[1] != 4:
        raise TypeError("4 columns required in colors array")
    if numcolors < 1:
        raise TypeError("at least 1 color required")
    numbands: int = numcolors - 1
    xw: int = imgdata.shape[1]
    yw: int = imgdata.shape[0]

    # if the first value in the color is -1 then switch to jumping mode
    jumps: bool = colors[0, 0] == -1

    # make image
    img: qt.QImage = qt.QImage(xw, yw, qt.QImage.Format.Format_ARGB32)

    # does the image use alpha values?
    hasalpha: bool = False

    # iterate over input pixels
    y: int
    for y in range(yw):
        # direction of images is different for qt and numpy image
        # scanline: memoryview = img.scanLine(yw - y - 1).cast("I")
        x: int
        for x in range(xw):
            val: np.float64 = imgdata[y, x]

            # output color
            b: int
            g: int
            r: int
            a: int

            if not np.isfinite(val):
                # transparent
                b = g = r = a = 0
            else:
                val = clipval(val, 0.0, 1.0)

                band: int
                if jumps:
                    # jumps between colours in discrete mode
                    # (ignores 1st color, which signals this mode)
                    band = clipval(
                        int(val * (numcolors - 1)) + 1,
                        1,
                        numcolors - 1,
                    )

                    b = int(colors[band, 0])
                    g = int(colors[band, 1])
                    r = int(colors[band, 2])
                    a = int(colors[band, 3])
                else:
                    # do linear interpolation between bands
                    # make sure between 0 and 1

                    band = clipval(int(val * numbands), 0, numbands - 1)
                    delta: float = val * numbands - band

                    # ensure we don't read beyond where we should
                    band2: int = min(band + 1, numbands)
                    delta1: float = 1.0 - delta

                    # we add 0.5 before truncating to round to nearest int
                    b = int(delta1 * colors[band, 0] + delta * colors[band2, 0] + 0.5)
                    g = int(delta1 * colors[band, 1] + delta * colors[band2, 1] + 0.5)
                    r = int(delta1 * colors[band, 2] + delta * colors[band2, 2] + 0.5)
                    a = int(delta1 * colors[band, 3] + delta * colors[band2, 3] + 0.5)

            if a != 255:
                hasalpha = True

            img.setPixelColor(x, y, qt.QColor(qt.qRgba(r, g, b, a)))

    if not hasalpha:
        # return image without transparency for speed / space improvements
        if Version(qt.qVersion()) >= Version("5.9.0"):
            # recent qt version
            # just change the format to the non-transparent version
            img.reinterpretAsFormat(qt.QImage.Format.Format_RGB32)
        else:
            # do slower conversion of data
            return img.convertToFormat(qt.QImage.Format.Format_RGB32)

    return img


def applyImageTransparancy(img: qt.QImage, data: NDArray[np.int64]) -> qt.QImage:
    xw: int = min(data.shape[1], img.width())
    yw: int = min(data.shape[0], img.height())

    for y in range(yw):
        # direction of images is different for qt and numpy image
        for x in range(xw):
            val: float = clipval(data[x, y], 0.0, 1.0)
            col: qt.QColor = img.pixelColor(yw - y - 1, x)

            # update pixel alpha component
            new_color = qt.qRgba(
                col.red(),
                col.green(),
                col.blue(),
                int(col.alpha() * val),
            )
            img.setPixelColor(yw - y - 1, x, new_color)
    return img


def plotNonlinearImageAsBoxes(
    painter: qt.QPainter,
    img: qt.QImage,
    xedges: NDArray[np.float64],
    yedges: NDArray[np.float64],
) -> None:
    width: int = img.width()
    height: int = img.height()

    # safety
    if xedges.shape[0] != width + 1 or yedges.shape[0] != height + 1:
        raise ValueError("Number of edges did not match image size")

    cliprect: qt.QRectF = painter.clipBoundingRect()
    clipped: bool = not cliprect.isEmpty()

    painter.save()

    for y in range(height):
        for x in range(width):
            x0: float = min(xedges[x], xedges[x + 1])
            x1: float = max(xedges[x], xedges[x + 1])
            y0: float = min(yedges[y], yedges[y + 1])
            y1: float = max(yedges[y], yedges[y + 1])

            r: qt.QRectF = qt.QRectF(x0, y0, x1 - x0, y1 - y0)
            if clipped:
                r &= cliprect

            if r.isValid():
                # note: axis coordinates are reversed wrt to image
                col: qt.QColor = qt.QColor(img.pixelColor(x, height - 1 - y))

                alpha: int = col.alpha()
                if alpha == 0:
                    # transparent
                    pass
                elif alpha == 255:
                    # opaque, so draw line to avoid antialiasing gaps round
                    # boxes
                    painter.setPen(qt.QPen(qt.QBrush(col), 0.0))
                    painter.setBrush(qt.QBrush(col))
                    painter.drawRect(r)
                else:
                    painter.fillRect(r, col)

    painter.restore()


def resampleNonlinearImage(
    img: qt.QImage,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    xedge: NDArray[np.float64],
    yedge: NDArray[np.float64],
):
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    xw: int = x1 - x0
    yw: int = y1 - y0

    outimg: qt.QImage = qt.QImage(xw, yw, img.format())

    iy: int = 0
    for oy in range(yw):
        while yedge[-2 - iy] <= oy + y0 + 0.5 and iy < yedge.shape[0] - 1:
            iy += 1

        # BUG: set right type
        oscanline: NDArray[np.uint32] = np.frombuffer(
            outimg.scanLine(oy).asarray(xw * 4), dtype=np.uint32
        )
        iscanline: NDArray[np.uint32] = np.frombuffer(
            img.scanLine(iy).asarray(xw * 4), dtype=np.uint32
        )

        ix: int = 0
        for ox in range(xw):
            while xedge[ix + 1] <= ox + x0 + 0.5 and ix < xedge.shape[0] - 1:
                ix += 1

            oscanline[ox] = iscanline[ix]

    return outimg


def plotImageAsRects(painter: qt.QPainter, bounds: qt.QRectF, img: qt.QImage) -> None:
    width: int = img.width()
    height: int = img.height()

    if width <= 0 or height <= 0:
        return

    dx: float = bounds.width() / width
    dy: float = bounds.height() / height
    x0: float = bounds.left()
    y0: float = bounds.top()

    cliprect: qt.QRectF = painter.clipBoundingRect()
    clipped: bool = not cliprect.isEmpty()

    painter.save()
    for y in range(height):
        for x in range(width):
            r: qt.QRectF = qt.QRectF(x0 + x * dx, y0 + y * dy, dx, dy)
            if clipped:
                r &= cliprect

            if r.isValid():
                col: qt.QColor = img.pixelColor(x, y)
                alpha: int = col.alpha()
                if alpha == 0:
                    # transparent
                    pass
                elif alpha == 255:
                    # opaque, so draw line to avoid antialiasing gaps round
                    # boxes
                    painter.setPen(qt.QPen(qt.QBrush(col), 0.0))
                    painter.setBrush(qt.QBrush(col))
                    painter.drawRect(r)
                else:
                    painter.fillRect(r, col)

    painter.restore()


def addCubicsToPainterPath(path: qt.QPainterPath, poly: qt.QPolygonF) -> None:
    lastpt: qt.QPointF | None = None
    for i in range(0, poly.size() - 3, 4):
        if lastpt != poly[i]:
            path.moveTo(poly[i])
        path.cubicTo(poly[i + 1], poly[i + 2], poly[i + 3])
        lastpt = poly[i + 3]
