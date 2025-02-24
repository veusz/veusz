import enum

import numpy as np
from numpy.typing import NDArray

from ._constants import _EPS


class ISect(enum.Enum):
    LINE_NOOVERLAP = 0
    LINE_CROSS = 1
    LINE_OVERLAP = 2


def twodPolyArea(poly: list[NDArray]) -> float:
    """area of polygon (+ve -> clockwise)"""
    s: int = len(poly)
    tot: float = 0.0
    for i in range(s):
        tot += poly[i][0] * poly[(i + 1) % s][1] - poly[(i + 1) % s][0] * poly[i][1]
    return 0.5 * tot


def twodPolyMakeClockwise(poly: list[NDArray]):
    """ensure polygon is clockwise"""
    if twodPolyArea(poly) < 0:
        poly.reverse()


def ptInside(p: NDArray, cp0: NDArray, cp1: NDArray) -> bool:
    """is `a` to the left of p1->p2?"""
    return (cp1[0] - cp0[0]) * (p[1] - cp0[1]) > (cp1[1] - cp0[1]) * (p[0] - cp0[0])


def ptInsideTol(p: NDArray, cp0: NDArray, cp1: NDArray) -> int:
    """
    version of above with tolerance of points on line
    0: on line, -1: outside, +1: inside
    """
    d: float = (cp1[0] - cp0[0]) * (p[1] - cp0[1]) - (cp1[1] - cp0[1]) * (p[0] - cp0[0])
    return 1 if d > _EPS else -1 if d < -_EPS else 0


def twodLineIntersect(
    p1: NDArray, p2: NDArray, q1: NDArray, q2: NDArray
) -> (int, NDArray, NDArray):
    """
    Do the two line segments p1->p2, q1->q2 cross or overlap?
    return LINE_NOOVERLAP if no overlap
           LINE_CROSS if they cross somewhere
           LINE_OVERLAP if they lie on top of each other partially
    if position1 != 0, return crossing position if LINE_CROSS
    if LINE_OVERLAP the two end points of overlap are returned in position1 and position2
    Assumes that the line segments are finite.
    """

    dp: NDArray = p2 - p1
    dq: NDArray = q2 - q1
    dpq: NDArray = p1 - q1
    denom: float = float(np.cross(dp, dq))

    position1: NDArray = np.zeros(2)
    position2: NDArray = np.zeros(2)

    # parallel vectors or points below
    if abs(denom) < _EPS:
        if abs(np.cross(dp, dpq)) > _EPS or abs(np.cross(dq, dpq)) > _EPS:
            return ISect.LINE_NOOVERLAP, position1, position2

        # collinear segments - do they overlap?
        u0: float
        u1: float
        dpq2: NDArray = p2 - q1
        if abs(dq[0]) > abs(dq[1]):
            u0 = dpq[0] / dq[0]
            u1 = dpq2[0] / dq[0]
        else:
            u0 = dpq[1] / dq[1]
            u1 = dpq2[1] / dq[1]

        if u0 > u1:
            u0, u1 = u1, u0

        if u0 > (1 + _EPS) or u1 < -_EPS:
            return ISect.LINE_NOOVERLAP, position1, position2

        u0 = max(u0, 0.0)
        u1 = min(u1, 1.0)
        position1 = q1 + dq * u0
        if abs(u0 - u1) < _EPS:
            return ISect.LINE_CROSS, position1, position2
        position2 = q1 + dq * u1
        return ISect.LINE_OVERLAP, position1, position2

    s: float = np.cross(dq, dpq) * (1 / denom)
    if s < -_EPS or s > (1 + _EPS):
        return ISect.LINE_NOOVERLAP, position1, position2
    t: float = np.cross(dp, dpq) * (1 / denom)
    if t < -_EPS or t > (1 + _EPS):
        return ISect.LINE_NOOVERLAP, position1, position2

    position1 = p1 + dp * max(min(s, 1.0), 0.0)

    return ISect.LINE_CROSS, position1, position2


def twodLineIntersectPolygon(p1: NDArray, p2: NDArray, poly: list[NDArray]) -> bool:
    """does line cross polygon? (make sure poly is defined clockwise)"""
    s: int = len(poly)
    inside1: bool = True
    inside2: bool = True

    for i in range(s):
        e1: NDArray = poly[i]
        e2: NDArray = poly[(i + 1) % s]

        # are any of the points inside?
        insidep1: int = ptInsideTol(p1, e1, e2)
        if insidep1 != 1:
            inside1 = False
        insidep2: int = ptInsideTol(p2, e1, e2)
        if insidep2 != 1:
            inside2 = False

        # check for line intersection if one of the edges doesn't touch
        # an edge
        if insidep1 != 0 and insidep2 != 0:
            two_d_line_intersection, position1, position2 = twodLineIntersect(
                p1, p2, e1, e2
            )
            if two_d_line_intersection == ISect.LINE_CROSS:
                return True

    # one of the points is inside
    return inside1 or inside2
