import abc
import sys
from copy import deepcopy
from typing import Sequence, overload

from ... import qtall as qt


class PaintElement:
    @abc.abstractmethod
    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None: ...


# Drawing Elements
# these are defined for each type of painting
# the QPaintEngine does


class EllipseElement(PaintElement):
    """draw an ellipse (QRect and QRectF)"""

    def __init__(self, rect: qt.QRectF | qt.QRect) -> None:
        self._ellipse: qt.QRectF | qt.QRect = rect

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.drawEllipse(self._ellipse)


class ImageElement(PaintElement):
    """draw QImage"""

    def __init__(
        self,
        rect: qt.QRectF,
        image: qt.QImage,
        sr: qt.QRectF,
        flags: qt.Qt.ImageConversionFlag,
    ) -> None:
        self._image: qt.QImage = image
        self._rect: qt.QRectF = rect
        self._sr: qt.QRectF = sr
        self._flags: qt.Qt.ImageConversionFlag = flags

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.drawImage(self._rect, self._image, self._sr, self._flags)


class LineElement(PaintElement):
    """
    draw lines
    this is for painting QLine and QLineF
    """

    def __init__(self, *lines: qt.QLineF | qt.QLine) -> None:
        if lines and isinstance(lines[-1], int):
            lines = lines[:-1]
        self._lines: tuple[qt.QLineF | qt.QLine, ...] = deepcopy(lines)

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        if self._lines:
            # BUG: inconsistent types
            if qt.sip.SIP_VERSION:
                # PyQt
                painter.drawLines(*self._lines)
            else:
                # PySide
                painter.drawLines(self._lines)


class PathElement(PaintElement):
    """draw QPainterPath"""

    def __init__(self, path: qt.QPainterPath) -> None:
        self._path: qt.QPainterPath = path

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.drawPath(self._path)


class PixmapElement(PaintElement):
    """draw Pixmap"""

    def __init__(self, r: qt.QRectF, pm: qt.QPixmap, sr: qt.QRectF) -> None:
        self._r: qt.QRectF = r
        self._pm: qt.QPixmap = pm
        self._sr: qt.QRectF = sr

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.drawPixmap(self._r, self._pm, self._sr)


class PointElement(PaintElement):
    """draw points (QPoint and QPointF)"""

    def __init__(self, points: Sequence[qt.QPointF | qt.QPoint]) -> None:
        if points and isinstance(points[-1], int):
            points = points[:-1]
        self._pts: qt.QPolygonF = qt.QPolygonF(points)

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        if self._pts:
            painter.drawPoints(self._pts)


class PolygonElement(PaintElement):
    """for QPolygon and QPolygonF"""

    def __init__(
        self,
        points: Sequence[qt.QPointF | qt.QPoint],
        mode: qt.QPaintEngine.PolygonDrawMode,
    ) -> None:
        self._mode: qt.QPaintEngine.PolygonDrawMode = mode
        self._pts: qt.QPolygonF = qt.QPolygonF(points)

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        if not self._pts:
            return
        if self._mode == qt.QPaintEngine.PolygonDrawMode.OddEvenMode:
            painter.drawPolygon(self._pts, qt.Qt.FillRule.OddEvenFill)
        elif self._mode == qt.QPaintEngine.PolygonDrawMode.WindingMode:
            painter.drawPolygon(self._pts, qt.Qt.FillRule.WindingFill)
        elif self._mode == qt.QPaintEngine.PolygonDrawMode.ConvexMode:
            painter.drawConvexPolygon(self._pts)
        elif self._mode == qt.QPaintEngine.PolygonDrawMode.PolylineMode:
            painter.drawPolyline(self._pts)  # BUG PYSIDE-3002: only 1 point in PySide6


class RectElement(PaintElement):
    """for QRect and QRectF"""

    def __init__(self, *rects: qt.QRectF | qt.QRect) -> None:
        if rects and isinstance(rects[-1], int):
            rects = rects[:-1]
        self._rects: tuple[qt.QRectF | qt.QRect, ...] = deepcopy(rects)

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        if self._rects:
            # BUG: inconsistent types
            if qt.sip.SIP_VERSION:
                painter.drawRects(*self._rects)
            else:
                painter.drawRects(self._rects)


class TextElement(PaintElement):
    """draw Text"""

    def __init__(self, pt: qt.QPointF, txt: qt.QTextItem) -> None:
        self._pt: qt.QPointF = pt
        self._text: str = txt.text()

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.drawText(self._pt, self._text)


class TiledPixmapElement(PaintElement):
    def __init__(self, rect: qt.QRectF, pixmap: qt.QPixmap, pt: qt.QPointF) -> None:
        self._rect: qt.QRectF = rect
        self._pixmap: qt.QPixmap = pixmap
        self._pt: qt.QPointF = pt

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.drawTiledPixmap(self._rect, self._pixmap, self._pt)


# State paint elements
# these define and change the state of the painter


class BackgroundBrushElement(PaintElement):
    def __init__(self, brush: qt.QBrush) -> None:
        self._brush: qt.QBrush = brush

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.setBackground(self._brush)


class BackgroundModeElement(PaintElement):
    def __init__(self, mode: qt.Qt.BGMode) -> None:
        self._mode: qt.Qt.BGMode = mode

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.setBackgroundMode(self._mode)


class BrushElement(PaintElement):
    def __init__(self, brush: qt.QBrush) -> None:
        self._brush: qt.QBrush = brush

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.setBrush(self._brush)


class BrushOriginElement(PaintElement):
    def __init__(self, origin: qt.QPointF) -> None:
        self._origin: qt.QPointF = origin

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.setBrushOrigin(self._origin)


class ClipRegionElement(PaintElement):
    def __init__(self, op: qt.Qt.ClipOperation, region: qt.QRegion) -> None:
        self._op: qt.Qt.ClipOperation = op
        self._region: qt.QRegion = region

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.setClipRegion(self._region, self._op)


class ClipPathElement(PaintElement):
    def __init__(self, op: qt.Qt.ClipOperation, region: qt.QPainterPath) -> None:
        self._op: qt.Qt.ClipOperation = op
        self._region: qt.QPainterPath = region

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.setClipPath(self._region, self._op)


class CompositionElement(PaintElement):
    def __init__(self, mode: qt.QPainter.CompositionMode) -> None:
        self._mode: qt.QPainter.CompositionMode = mode

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.setCompositionMode(self._mode)


class FontElement(PaintElement):
    def __init__(self, font: qt.QFont, dpi: float) -> None:
        self._dpi: float = dpi
        self._font: qt.QFont = font

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        temp_font: qt.QFont = qt.QFont(self._font)
        if temp_font.pointSizeF() > 0.0:
            # scale font sizes in points using dpi ratio
            this_dpi: int = painter.device().logicalDpiY()
            scale: float = temp_font.pointSizeF() / this_dpi * self._dpi
            temp_font.setPointSizeF(scale)

        painter.setFont(temp_font)


class TransformElement(PaintElement):
    def __init__(self, t: qt.QTransform) -> None:
        self._t: qt.QTransform = t

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform):
        painter.setWorldTransform(orig_transform)
        painter.setWorldTransform(self._t, True)


class ClipEnabledElement(PaintElement):
    def __init__(self, enabled: bool) -> None:
        self._enabled: bool = enabled

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.setClipping(self._enabled)


class PenElement(PaintElement):
    def __init__(self, pen: qt.QPen) -> None:
        self._pen = qt.QPen(pen)

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.setPen(self._pen)


class HintsElement(PaintElement):
    def __init__(self, hints: qt.QPainter.RenderHint) -> None:
        self._hints = hints

    def paint(self, painter: qt.QPainter, orig_transform: qt.QTransform) -> None:
        painter.setRenderHints(self._hints)


class RecordPaintEngine(qt.QPaintEngine):
    def __init__(self) -> None:
        super().__init__(qt.QPaintEngine.PaintEngineFeature.AllFeatures)
        self._drawitemcount: int = 0
        self._pdev: RecordPaintDevice | None = None

    def type(self) -> qt.QPaintEngine.Type:
        return qt.QPaintEngine.Type.User

    # standard methods to be overridden in engine
    def begin(self, pdev: qt.QPaintDevice) -> bool:
        self._pdev = pdev
        # signal started ok
        return True

    def drawEllipse(self, rect: qt.QRectF | qt.QRect) -> None:
        self._pdev.addElement(EllipseElement(rect))
        self._drawitemcount += 1

    def drawImage(
        self,
        rectangle: qt.QRectF,
        image: qt.QImage,
        sr: qt.QRectF,
        flags: qt.Qt.ImageConversionFlag = qt.Qt.ImageConversionFlag.AutoColor,
    ) -> None:
        self._pdev.addElement(ImageElement(rectangle, image, sr, flags))
        self._drawitemcount += 1

    def drawLines(self, *lines: qt.QLineF | qt.QLine, **kwargs: int) -> None:
        if not lines:
            return
        self._pdev.addElement(LineElement(*lines))
        self._drawitemcount += len(lines)

    def drawPath(self, path: qt.QPainterPath) -> None:
        self._pdev.addElement(PathElement(path))
        self._drawitemcount += 1

    def drawPixmap(self, r: qt.QRectF, pm: qt.QPixmap, sr: qt.QRectF) -> None:
        self._pdev.addElement(PixmapElement(r, pm, sr))
        self._drawitemcount += 1

    def drawPoints(
        self,
        points: Sequence[qt.QPointF | qt.QPoint],
        *args: int,
        **kwargs: int,
    ) -> None:
        if not points:
            return
        self._pdev.addElement(PointElement(points))
        self._drawitemcount += len(points)

    # noinspection PyMethodOverriding
    @overload
    def drawPolygon(
        self,
        points: qt.QPointF | qt.QPoint | Sequence[qt.QPointF | qt.QPoint],
        mode: qt.QPaintEngine.PolygonDrawMode = qt.QPaintEngine.PolygonDrawMode.PolylineMode,
    ) -> None: ...
    @overload
    def drawPolygon(
        self,
        points: qt.QPointF | qt.QPoint | Sequence[qt.QPointF | qt.QPoint],
        pointCount: int,
        mode: qt.QPaintEngine.PolygonDrawMode = qt.QPaintEngine.PolygonDrawMode.PolylineMode,
    ) -> None: ...
    def drawPolygon(
        self,
        points: qt.QPointF | qt.QPoint | Sequence[qt.QPointF | qt.QPoint],
        *args: qt.QPaintEngine.PolygonDrawMode | int,
        **kwargs: qt.QPaintEngine.PolygonDrawMode | int,
    ) -> None:
        """The function is compatible with PyQt* and PySide*"""
        mode: qt.QPaintEngine.PolygonDrawMode | None = None
        if "mode" in kwargs:
            mode = kwargs["mode"]
        else:
            for arg in args:
                if isinstance(arg, qt.QPaintEngine.PolygonDrawMode):
                    mode = arg
                    break
        if not isinstance(points, Sequence):
            points = [points]
        points = [p for p in points if not p.isNull()]
        if not points:
            return
        self._pdev.addElement(PolygonElement(points, mode=mode))
        self._drawitemcount += 1

    def drawRects(self, *rects: qt.QRectF | qt.QRect, rectsCount: int = ...) -> None:
        if not rects:
            return
        self._pdev.addElement(RectElement(*rects))
        self._drawitemcount += len(rects)

    def drawTextItem(self, p: qt.QPointF, text_item: qt.QTextItem) -> None:
        self._pdev.addElement(TextElement(p, text_item))
        self._drawitemcount += len(text_item.text())

    def drawTiledPixmap(
        self,
        rect: qt.QRectF,
        pixmap: qt.QPixmap,
        p: qt.QPointF,
    ) -> None:
        self._pdev.addElement(TiledPixmapElement(rect, pixmap, p))
        self._drawitemcount += 1

    def end(self) -> bool:
        # signal finished ok
        return True

    def updateState(self, state: qt.QPaintEngineState) -> None:
        # we add a new element for each change of state
        # these are replayed later
        flags: qt.QPaintEngine.DirtyFlag = state.state()
        if flags & qt.QPaintEngine.DirtyFlag.DirtyPen:
            self._pdev.addElement(PenElement(state.pen()))
        if flags & qt.QPaintEngine.DirtyFlag.DirtyBrush:
            self._pdev.addElement(BrushElement(state.brush()))
        if flags & qt.QPaintEngine.DirtyFlag.DirtyBrushOrigin:
            self._pdev.addElement(BrushOriginElement(state.brushOrigin()))
        if flags & qt.QPaintEngine.DirtyFlag.DirtyFont:
            self._pdev.addElement(FontElement(state.font(), self._pdev.dpiy))
        if flags & qt.QPaintEngine.DirtyFlag.DirtyBackground:
            self._pdev.addElement(BackgroundBrushElement(state.backgroundBrush()))
        if flags & qt.QPaintEngine.DirtyFlag.DirtyBackgroundMode:
            self._pdev.addElement(BackgroundModeElement(state.backgroundMode()))
        if flags & qt.QPaintEngine.DirtyFlag.DirtyTransform:
            self._pdev.addElement(TransformElement(state.transform()))
        if flags & qt.QPaintEngine.DirtyFlag.DirtyClipRegion:
            self._pdev.addElement(
                ClipRegionElement(state.clipOperation(), state.clipRegion())
            )
        if flags & qt.QPaintEngine.DirtyFlag.DirtyClipPath:
            self._pdev.addElement(
                ClipPathElement(state.clipOperation(), state.clipPath())
            )
        if flags & qt.QPaintEngine.DirtyFlag.DirtyHints:
            self._pdev.addElement(HintsElement(state.renderHints()))
        if flags & qt.QPaintEngine.DirtyFlag.DirtyCompositionMode:
            self._pdev.addElement(CompositionElement(state.compositionMode()))
        if flags & qt.QPaintEngine.DirtyFlag.DirtyClipEnabled:
            self._pdev.addElement(ClipEnabledElement(state.isClipEnabled()))

    def drawItemCount(self) -> int:
        """return an estimate of number of items drawn"""
        return self._drawitemcount


INCH_MM: float = 25.4


class RecordPaintDevice(qt.QPaintDevice):
    def __init__(self, width: int, height: int, dpix: float, dpiy: float) -> None:
        super().__init__()
        self._engine: RecordPaintEngine = RecordPaintEngine()
        self._width: int = width
        self._height: int = height
        self.dpix: float = dpix
        self.dpiy: float = dpiy
        self._elements: list[PaintElement] = []

    def paintEngine(self) -> RecordPaintEngine:
        return self._engine

    def play(self, painter: qt.QPainter) -> None:
        """play back all"""
        orig_transform: qt.QTransform = painter.worldTransform()
        el: LineElement
        for el in self._elements:
            el.paint(painter, orig_transform)

    def metric(self, metric: qt.QPaintDevice.PaintDeviceMetric) -> int:
        if metric == qt.QPaintDevice.PaintDeviceMetric.PdmWidth:
            return self._width
        elif metric == qt.QPaintDevice.PaintDeviceMetric.PdmHeight:
            return self._height
        elif metric == qt.QPaintDevice.PaintDeviceMetric.PdmWidthMM:
            return int(self._width * INCH_MM / self.dpix)
        elif metric == qt.QPaintDevice.PaintDeviceMetric.PdmHeightMM:
            return int(self._height * INCH_MM / self.dpiy)
        elif metric == qt.QPaintDevice.PaintDeviceMetric.PdmNumColors:
            return sys.maxsize
        elif metric == qt.QPaintDevice.PaintDeviceMetric.PdmDepth:
            return 24
        elif metric in (
            qt.QPaintDevice.PaintDeviceMetric.PdmDpiX,
            qt.QPaintDevice.PaintDeviceMetric.PdmPhysicalDpiX,
        ):
            return round(self.dpix)
        elif metric in (
            qt.QPaintDevice.PaintDeviceMetric.PdmDpiY,
            qt.QPaintDevice.PaintDeviceMetric.PdmPhysicalDpiY,
        ):
            return round(self.dpiy)
        elif metric == qt.QPaintDevice.PaintDeviceMetric.PdmDevicePixelRatio:
            return 1
        else:
            # fallback
            return super().metric(metric)

    def drawItemCount(self) -> int:
        return self._engine.drawItemCount()

    def addElement(self, el: PaintElement) -> None:
        """add an element to the list of maintained elements"""
        self._elements.append(el)
