from typing import Callable, Iterable

from ... import qtall as qt

# tolerance for points being the same
TOL: float = 1e-5


def interceptVert(horzval: float, pt1: qt.QPointF, pt2: qt.QPointF) -> qt.QPointF:
    gradient: float = (pt2.y() - pt1.y()) / (pt2.x() - pt1.x())
    return qt.QPointF(horzval, (horzval - pt1.x()) * gradient + pt1.y())


def interceptHorz(vertval: float, pt1: qt.QPointF, pt2: qt.QPointF) -> qt.QPointF:
    gradient: float = (pt2.x() - pt1.x()) / (pt2.y() - pt1.y())
    return qt.QPointF((vertval - pt1.y()) * gradient + pt1.x(), vertval)


def gtclose(v1: float, v2: float) -> bool:
    """greater than or close"""
    return v1 > v2 or abs(v1 - v2) < TOL


def ltclose(v1: float, v2: float) -> bool:
    """less than or close"""
    return v1 < v2 or abs(v1 - v2) < TOL


class State:
    def _clip_edge(
        self,
        edge: str,
        is_inside: Callable[[qt.QPointF], bool],
        intercept: Callable[[float, qt.QPointF, qt.QPointF], qt.QPointF],
        _next: Callable[[qt.QPointF], None],
    ) -> Callable[[qt.QPointF], None]:
        def edge_clip_point(pt: qt.QPointF) -> None:
            last_pt: qt.QPointF = self.__getattribute__(f"{edge}_last")
            if self.__getattribute__(f"{edge}_is_1st"):
                # do nothing
                self.__setattr__(f"{edge}_1st", pt)
                self.__setattr__(f"{edge}_is_1st", False)
            else:
                if is_inside(pt):
                    if not is_inside(last_pt):
                        # this point inside and last point outside
                        _next(
                            intercept(
                                self.clip.__getattribute__(edge)(),
                                pt,
                                last_pt,
                            )
                        )
                    _next(pt)
                else:
                    if is_inside(last_pt):
                        # this point outside and last point inside
                        _next(
                            intercept(
                                self.clip.__getattribute__(edge)(),
                                pt,
                                last_pt,
                            )
                        )
                    # else do nothing if both outside

            self.__setattr__(f"{edge}_last", pt)

        return edge_clip_point

    def __init__(self, rect: qt.QRectF, out: qt.QPolygonF):
        self.clip: qt.QRectF = rect  # location of corners of clip rectangle

        # output points are added here
        self.output: qt.QPolygonF = out

        # last points added
        self.left_last: qt.QPointF = qt.QPointF()
        self.right_last: qt.QPointF = qt.QPointF()
        self.top_last: qt.QPointF = qt.QPointF()
        self.bottom_last: qt.QPointF = qt.QPointF()

        # first point for each stage
        self.left_1st: qt.QPointF = qt.QPointF()
        self.right_1st: qt.QPointF = qt.QPointF()
        self.top_1st: qt.QPointF = qt.QPointF()
        self.bottom_1st: qt.QPointF = qt.QPointF()

        # whether this is the 1st point through
        self.left_is_1st: bool = True
        self.right_is_1st: bool = True
        self.top_is_1st: bool = True
        self.bottom_is_1st: bool = True

        # add functions for clipping to each edge
        self.bottomClipPoint = self._clip_edge(
            "bottom", self.insideBottom, interceptHorz, self.writeClipPoint
        )
        self.topClipPoint = self._clip_edge(
            "top", self.insideTop, interceptHorz, self.bottomClipPoint
        )
        self.rightClipPoint = self._clip_edge(
            "right", self.insideRight, interceptVert, self.topClipPoint
        )
        self.leftClipPoint = self._clip_edge(
            "left", self.insideLeft, interceptVert, self.rightClipPoint
        )

    # tests for whether point is inside of outside each side
    def insideBottom(self, pt: qt.QPointF) -> bool:
        return ltclose(pt.y(), self.clip.bottom())

    def insideTop(self, pt: qt.QPointF) -> bool:
        return gtclose(pt.y(), self.clip.top())

    def insideRight(self, pt: qt.QPointF) -> bool:
        return ltclose(pt.x(), self.clip.right())

    def insideLeft(self, pt: qt.QPointF) -> bool:
        return gtclose(pt.x(), self.clip.left())

    # finally writes to output
    def writeClipPoint(self, pt: qt.QPointF):
        # don't add the same point
        if (
            self.output.isEmpty()
            or abs(pt.x() - self.output.last().x()) > TOL
            or abs(pt.y() - self.output.last().y()) > TOL
        ):
            self.output.append(pt)


def polygonClip(
    in_polygon: Iterable[qt.QPointF] | qt.QPolygonF,
    clip_rectangle: qt.QRectF,
    out_polygon: qt.QPolygonF,
) -> None:
    # construct initial state
    state: State = State(clip_rectangle, out_polygon)

    # do the clipping
    for pt in in_polygon:
        state.leftClipPoint(pt)

    # complete
    state.leftClipPoint(state.left_1st)
    state.rightClipPoint(state.right_1st)
    state.topClipPoint(state.top_1st)
    state.bottomClipPoint(state.bottom_1st)


def plotClippedPolygon(
    painter: qt.QPainter,
    rect: qt.QRectF,
    in_polygon: qt.QPolygonF,
    auto_expand: bool = True,
):
    if auto_expand:
        lw: float = painter.pen().widthF()
        if painter.pen().style() != qt.Qt.PenStyle.NoPen:
            rect.adjust(-lw, -lw, lw, lw)

    plt = qt.QPolygonF()
    polygonClip(in_polygon, rect, plt)
    painter.drawPolygon(plt)
