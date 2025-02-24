"""
Copyright (C) 2010 Jeremy S. Sanders
Email: Jeremy Sanders <jeremy@jeremysanders.net>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

from .beziers import sp_bezier_fit_cubic, sp_bezier_fit_cubic_r
from ... import qtall as qt

__all__ = [
    "bezier_fit_cubic_single",
    "bezier_fit_cubic_multi",
    "bezier_fit_cubic_tight",
]


def bezier_fit_cubic_single(data: qt.QPolygonF, error: float) -> qt.QPolygonF:
    out: qt.QPolygonF = qt.QPolygonF(4)
    retn: int = sp_bezier_fit_cubic(out.data(), data.data(), error)
    if retn >= 0:
        return out
    else:
        return qt.QPolygonF()


def bezier_fit_cubic_multi(
    data: qt.QPolygonF,
    error: float,
    max_beziers: int,
) -> qt.QPolygonF:
    out: qt.QPolygonF = qt.QPolygonF(4 * max_beziers)
    retn: int = sp_bezier_fit_cubic_r(out.data(), data.data(), error, max_beziers)

    if retn >= 0:
        # get rid of unused points
        if retn * 4 < out.count():
            out.remove(retn * 4, out.count() - retn * 4)
        return out
    else:
        return qt.QPolygonF()


def bezier_fit_cubic_tight(data: qt.QPolygonF, looseness: float) -> qt.QPolygonF:
    """
    MS Excel-like cubic Bezier fitting formulated by Brian Murphy.
    (http://www.xlrotor.com/Smooth_curve_bezier_example_file.zip)

    4 bezier control points (ctrls[0]-ctrl[3]) are created for each line
    segment. Positions of ctrls are determined by 4 nearest data points
    (pt0-pt3) with following rules:
    ctrls[0]: same position as pt0.
    ctrls[1]: on a line through pt1 parallel to pt0-pt2,
              at a distance from pt1 = f1 * |pt0-pt2|
    ctrls[2]: on a line through pt2 parallel to pt1-pt3,
              at a distance from pt2 = f2 * |pt1-pt3|
    ctrls[3]: same position as pt3
    The magic numbers (f1 and f2) are determined by length ratio of the
    3 line segments and “looseness” with some additional rules.
    looseness: artificial parameter to control “tension” of the Bézier
               curve. Larger value gives more curved connection.
               In MS Excell, this value is set as “0.5”.
    """
    length: int = data.size()
    if length < 2:
        return qt.QPolygonF()
    elif length == 2:
        bezier_ctrls: qt.QPolygonF = qt.QPolygonF(4)
        bezier_ctrls << data[0] << data[0] << data[1] << data[1]
        return bezier_ctrls
    else:
        bezier_ctrls: qt.QPolygonF = qt.QPolygonF(4 * (length - 1))
        for i in range(length):
            ctrls: qt.QPolygonF = qt.QPolygonF(4)
            pt0: qt.QPointF
            pt1: qt.QPointF = data[i - 1]
            pt2: qt.QPointF = data[i]
            pt3: qt.QPointF
            ctrls[0] = pt1
            ctrls[3] = pt2
            f1: float
            f2: float
            if i == 1:
                pt0 = data[i - 1]
                pt3 = data[i + 1]
                f1 = looseness / 1.5
                f2 = looseness / 3.0
            elif i == length - 1:
                pt0 = data[i - 2]
                pt3 = data[i]
                f1 = looseness / 3.0
                f2 = looseness / 1.5
            else:
                pt0 = data[i - 2]
                pt3 = data[i + 1]
                f1 = looseness / 3.0
                f2 = looseness / 3.0

            d02: float = qt.QLineF(pt0, pt2).length()
            d12: float = qt.QLineF(pt1, pt2).length()
            d13: float = qt.QLineF(pt1, pt3).length()
            b1: bool = d02 < d12 * 3.0
            b2: bool = d13 < d12 * 3.0
            if not (b1 and b2):
                f1 = d12 / d02 / 2.0
                f2 = d12 / d13 / 2.0
                if b1:
                    f1 = f2

                if b2:
                    f2 = f1

            ctrls[1] = pt1 + (pt2 - pt0) * f1
            ctrls[2] = pt2 + (pt1 - pt3) * f2
            bezier_ctrls += ctrls

        return bezier_ctrls
