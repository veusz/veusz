import sys
from math import hypot, isnan, sqrt

import numpy as np
from numpy.typing import NDArray

from ... import qtall as qt

# \file
# Bezier interpolation for inkscape drawing code.
#
# Original code published in:
#   An Algorithm for Automatically Fitting Digitized Curves
#   by Philip J. Schneider
#  "Graphics Gems", Academic Press, 1990
#
# Authors:
#   Philip J. Schneider
#   Lauris Kaplinski <lauris@kaplinski.com>
#   Peter Moulder <pmoulder@mail.csse.monash.edu.au>
#
# Copyright (C) 1990 Philip J. Schneider
# Copyright (C) 2001 Lauris Kaplinski
# Copyright (C) 2001 Ximian, Inc.
# Copyright (C) 2003,2004 Monash University
#
# Released under GNU GPL, read the file 'COPYING' for more information
# Modified to be based around QPointF by Jeremy Sanders (2007)

SP_HUGE: float = 1e5


def unit_vector(pt: qt.QPointF) -> qt.QPointF:
    mag: float = hypot(pt.x(), pt.y())
    return pt / mag


def dot(a: qt.QPointF, b: qt.QPointF) -> float:
    return a.x() * b.x() + a.y() * b.y()


def lensq(p: qt.QPointF) -> float:
    return dot(p, p)


# Compute the L2, or euclidean, norm of :param p.


def L2(p: qt.QPointF) -> float:
    return hypot(p.x(), p.y())


def rot90(p: qt.QPointF) -> qt.QPointF:
    return qt.QPointF(-p.y(), p.x())


def copy_without_nans_or_adjacent_duplicates(
    src: list[qt.QPointF],
) -> list[qt.QPointF]:
    """
    Filter out points containing NaN and adjacent points with equal x and y.
    """
    dest: list[qt.QPointF] = [
        qt.QPointF(src[si])
        for si in range(len(src))
        if not (isnan(src[si].x()) or isnan(src[si].y()))
    ]
    if not dest:
        return dest
    return [dest[0]] + [
        dest[di] for di in range(1, len(dest)) if dest[di] != dest[di - 1]
    ]


unconstrained_tangent: qt.QPointF = qt.QPointF(0.0, 0.0)


BezierCurve = list[qt.QPointF]

# B0, B1, B2, B3: Bezier multipliers


def B0(
    u: float | np.float64 | NDArray[np.float64],
) -> float | np.float64 | NDArray[np.float64]:
    return np.power(1.0 - u, 3)


def B1(
    u: float | np.float64 | NDArray[np.float64],
) -> float | np.float64 | NDArray[np.float64]:
    return 3.0 * np.dot(u, np.square(1.0 - u))


def B2(
    u: float | np.float64 | NDArray[np.float64],
) -> float | np.float64 | NDArray[np.float64]:
    return 3.0 * np.dot(np.power(u, 2), (1.0 - u))


def B3(
    u: float | np.float64 | NDArray[np.float64],
) -> float | np.float64 | NDArray[np.float64]:
    return np.power(u, 3)


def chord_length_parameterize(d: list[qt.QPointF]) -> NDArray[np.float64]:
    """
    Assign parameter values to digitized points using relative distances between points.
    """
    if len(d) < 2:
        return np.empty(0)

    # First let u[i] equal the distance travelled along the path from d[0] to d[i].
    u: NDArray[np.float64] = np.cumsum(
        [L2(d[i] - d[i - 1]) for i in range(1, len(d))],
        dtype=np.float64,
    )

    # Then scale to [0.0 .. 1.0].
    if np.isfinite(u[-1]):
        u /= u[-1]
    else:
        # We could do better, but this probably never happens anyway.
        u /= len(d)

    # todo
    #  It's been reported that u[len - 1] can differ from 1.0 on some
    #  systems (amd64), despite it having been calculated as x / x where x
    #  is isFinite and non-zero.
    if u[-1] != 1.0:
        diff: np.float64 | NDArray[np.float64] = u[-1] - 1.0
        if np.abs(diff) > 1e-13:
            print(
                f"u[-1] = {u[- 1]:19g} (= 1 + {diff:19g}), expecting exactly 1",
                file=sys.stderr,
            )
        u[-1] = 1.0
    return u


def compute_max_error_ratio(
    d: list[qt.QPointF],
    u: NDArray[np.float64],
    bezCurve: BezierCurve,
    tolerance: float,
) -> tuple[float, int]:
    """
    Find the maximum squared distance of digitized points to fitted curve, and (if this maximum
    error is non-zero) return the corresponding index.

    (2 <= len(d)), (u[0] == 0), and (u[len(d) - 1] == 1.0).
    ((ret == 0.0) || ((split_point < len(d) - 1) && (split_point != 0 || ret < 0.0))).
    """
    assert 2 <= len(d)
    assert bezCurve[0] == d[0]
    assert bezCurve[3] == d[-1]
    assert u[0] == 0.0
    assert u[-1] == 1.0
    # I.e. assert that the error for the first & last points is zero.
    # Otherwise, we should include those points in the below loop.
    # The assertion is also necessary to ensure 0 < split_point < last.

    max_distsq: float = 0.0  # Maximum error
    max_hook_ratio: float = 0.0
    snap_end: int = 0
    prev: qt.QPointF = bezCurve[0]
    distsq: float
    split_point: int = -1
    for i in range(1, len(d)):
        curr: qt.QPointF = bezier_pt(3, bezCurve, u[i])
        distsq = lensq(curr - d[i])
        if distsq > max_distsq:
            max_distsq = distsq
            split_point = i
        hook_ratio: float = compute_hook(
            prev, curr, 0.5 * (u[i - 1] + u[i]), bezCurve, tolerance
        )
        if max_hook_ratio < hook_ratio:
            max_hook_ratio = hook_ratio
            snap_end = i
        prev = curr

    dist_ratio: float = sqrt(max_distsq) / tolerance
    ret: float
    if max_hook_ratio <= dist_ratio:
        ret = dist_ratio
    else:
        assert 0 < snap_end
        ret = -max_hook_ratio
        split_point = snap_end - 1
    assert ret == 0.0 or (
        (split_point < len(d) - 1) and (split_point != 0 or ret < 0.0)
    )
    return ret, split_point


def bezier_pt(
    degree: int,
    V: list[qt.QPointF],
    t: float | np.float64 | NDArray[np.float64],
) -> qt.QPointF:
    """
    Evaluate a Bézier curve at parameter value :param t.

    :param degree: The degree of the Bézier curve: 3 for cubic, 2 for quadratic etc.
    :param V: The control points for the Bézier curve.  Must have (:param degree+1)
       elements.
    :param t: The “parameter” value, specifying whereabouts along the curve to
       evaluate.  Typically, in the range [0.0, 1.0].
        Let s = 1 - t.
    BezierII(1, V) gives (s, t) * V, i.e. t of the way
    from V[0] to V[1].
    BezierII(2, V) gives (s**2, 2*s*t, t**2) * V.
    BezierII(3, V) gives (s**3, 3 s**2 t, 3s t**2, t**3) * V.
        The derivative of BezierII(i, V) with respect to t
    is i * BezierII(i-1, V'), where for all j, V'[j] =
    V[j + 1] - V[j].
    """
    ...


def estimate_lengths(
    data: list[qt.QPointF],
    uPrime: NDArray[np.float64],
    tHat1: qt.QPointF,
    tHat2: qt.QPointF,
) -> list[qt.QPointF]:
    bezier: list[qt.QPointF] = [qt.QPointF() for _ in range(4)]

    c: NDArray[np.float64] = np.zeros((2, 2))
    x: NDArray[np.float64] = np.zeros(2)

    # First and last control points of the Bézier curve are positioned exactly at the first and
    # last data points.
    bezier[0] = data[0]
    bezier[3] = data[-1]
    # Bezier control point coefficients.
    b0: NDArray[np.float64] = B0(uPrime)
    b1: NDArray[np.float64] = B1(uPrime)
    b2: NDArray[np.float64] = B2(uPrime)
    b3: NDArray[np.float64] = B3(uPrime)
    for i in range(len(data)):
        # rhs for eqn
        a1 = b1 * tHat1
        a2 = b2 * tHat2

        c[0, 0] += dot(a1, a1)
        c[0, 1] += dot(a1, a2)
        c[1, 0] = c[0, 1]
        c[1, 1] += dot(a2, a2)

        # Additional offset to the data point from the predicted point if we were to set bezier[1]
        # to bezier[0] and bezier[2] to bezier[3].
        shortfall: qt.QPointF = (
            data[i] - ((b0 + b1) * bezier[0]) - ((b2 + b3) * bezier[3])
        )
        x[0] += dot(a1, shortfall)
        x[1] += dot(a2, shortfall)

    # We've constructed a pair of equations in the form of a matrix product c * alpha = x.
    # Now solve for alpha.
    alpha_l: float | np.float64 | NDArray[np.float64]
    alpha_r: float | np.float64 | NDArray[np.float64]

    # Compute the determinants of c and x.
    det_c0_c1: float | np.float64 | NDArray[np.float64] = (
        c[0, 0] * c[1, 1] - c[1, 0] * c[0, 1]
    )
    if det_c0_c1 != 0.0:
        # Apparently Kramer's rule.
        det_c0_x: float | np.float64 | NDArray[np.float64] = (
            c[0][0] * x[1] - c[0][1] * x[0]
        )
        det_x_c1: float | np.float64 | NDArray[np.float64] = (
            x[0] * c[1][1] - x[1] * c[0][1]
        )
        alpha_l = det_x_c1 / det_c0_c1
        alpha_r = det_c0_x / det_c0_c1
    else:
        # The matrix is under-determined.  Try requiring alpha_l == alpha_r.
        #
        # One way of implementing the constraint alpha_l == alpha_r is to treat them as the same
        # variable in the equations.  We can do this by adding the columns of c to form a single
        # column, to be multiplied by alpha to give the column vector x.
        #
        # We try each row in turn.
        c0: float | np.float64 | NDArray[np.float64] = c[0][0] + c[0][1]
        if c0 != 0.0:
            alpha_l = alpha_r = x[0] / c0
        else:
            c1: float | np.float64 | NDArray[np.float64] = c[1][0] + c[1][1]
            if c1 != 0:
                alpha_l = alpha_r = x[1] / c1
            else:
                # Let the below code handle this.
                alpha_l = alpha_r = 0.0

    # If alpha negative, use the Wu/Barsky heuristic (see text).  (If alpha is 0, you get
    # coincident control points that lead to divide by zero in any subsequent
    # NewtonRaphsonRootFind() call.)
    # todo Check whether this special-casing is necessary now that
    #      NewtonRaphsonRootFind handles non-positive denominator.
    if alpha_l < 1.0e-6 or alpha_r < 1.0e-6:
        alpha_l = alpha_r = L2(data[-1] - data[0]) * (1.0 / 3.0)

    # Control points 1 and 2 are positioned an alpha distance out on the tangent vectors, left and
    # right, respectively.
    bezier[1] = alpha_l * tHat1 + bezier[0]
    bezier[2] = alpha_r * tHat2 + bezier[3]

    return bezier


def estimate_bi(
    bezier: list[qt.QPointF],
    ei: int,
    data: list[qt.QPointF],
    u: NDArray[np.float64],
) -> list[qt.QPointF]:
    if not (1 <= ei <= 2):
        return bezier

    oi: int = 3 - ei
    num: list[float] = [0.0, 0.0]
    den: float = 0.0
    for i in range(len(data)):
        ui = u[i]
        b: list[float | np.float64 | NDArray[np.float64]] = [
            B0(ui),
            B1(ui),
            B2(ui),
            B3(ui),
        ]
        num[0] += b[ei] * (
            b[0] * bezier[0].x()
            + b[oi] * bezier[0].x()
            + b[3] * bezier[3].x()
            + -data[i].x()
        )
        num[1] += b[ei] * (
            b[0] * bezier[0].y()
            + b[oi] * bezier[0].y()
            + b[3] * bezier[3].y()
            + -data[i].y()
        )
        den -= b[ei] ** 2

    if den != 0.0:
        bezier[ei] = qt.QPointF(*num) / den
    else:
        bezier[ei] = (oi * bezier[0] + ei * bezier[3]) * (1.0 / 3.0)

    return bezier


def generate_bezier(
    data: list[qt.QPointF],
    u: NDArray[np.float64],
    t_hat1: qt.QPointF,
    t_hat2: qt.QPointF,
    tolerance_sq: float,
) -> list[qt.QPointF]:
    """
    Fill in :param bezier[] based on the given data and tangent requirements, using
    a least-squares fit.

    Each of t_hat1 and t_hat2 should be either a zero vector or a unit vector.
    If it is zero, then bezier[1 or 2] is estimated without constraint; otherwise,
    it bezier[1 or 2] is placed in the specified direction from bezier[0 or 3].

    :param tolerance_sq Used only for an initial guess as to tangent directions
      when :param t_hat1 or :param t_hat2 is zero.
    """
    bezier: list[qt.QPointF]

    est1: bool = t_hat1.isNull()
    est2: bool = t_hat2.isNull()
    est_t_hat1: qt.QPointF = (
        sp_darray_left_tangent(data, tolerance_sq) if est1 else t_hat1
    )
    est_t_hat2: qt.QPointF = (
        sp_darray_right_tangent(data, tolerance_sq) if est2 else t_hat2
    )
    bezier = estimate_lengths(data, u, est_t_hat1, est_t_hat2)
    # We find that sp_darray_right_tangent tends to produce better results
    # for our current freehand tool than full estimation.
    if est1:
        bezier = estimate_bi(bezier, 1, data, u)
        if bezier[1] != bezier[0]:
            est_t_hat1 = unit_vector(bezier[1] - bezier[0])
        bezier = estimate_lengths(data, u, est_t_hat1, est_t_hat2)
    return bezier


def newton_raphson_root_find(
    q: BezierCurve,
    p: qt.QPointF,
    u: float | np.float64 | NDArray[np.float64],
) -> float | np.float64 | NDArray[np.float64]:
    """
    Use Newton-Raphson iteration to find better root.

    :param q  Current fitted curve
    :param p  Digitized point
    :param u  Parameter value for "P"

    :return Improved u
    """
    assert 0.0 <= u <= 1.0
    # Generate control vertices for q'.
    q1: list[qt.QPointF] = [
        3.0 * (q[1] - q[0]),
        3.0 * (q[2] - q[1]),
        3.0 * (q[3] - q[2]),
    ]

    # Generate control vertices for q''.
    q2: list[qt.QPointF] = [
        2.0 * (q1[1] - q1[0]),
        2.0 * (q1[2] - q1[1]),
    ]

    # Compute q(u), q'(u) and q''(u).
    q_u: qt.QPointF = bezier_pt(3, q, u)
    q1_u: qt.QPointF = bezier_pt(2, q1, u)
    q2_u: qt.QPointF = bezier_pt(1, q2, u)

    # Compute f(u)/f'(u), where f is the derivative wrt u of distsq(u) = 0.5 * the square of the
    # distance from p to q(u).  Here we're using Newton-Raphson to find a stationary point in the
    # distsq(u), hopefully corresponding to a local minimum in distsq (and hence a local minimum
    # distance from p to q(u)).
    diff: qt.QPointF = q_u - p
    numerator: float = dot(diff, q1_u)
    denominator: float = dot(q1_u, q1_u) + dot(diff, q2_u)

    improved_u: float | np.float64 | NDArray[np.float64]
    if denominator > 0.0:
        # One iteration of Newton-Raphson:
        # improved_u = u - f(u)/f'(u)
        improved_u = u - (numerator / denominator)
    else:
        # Using Newton-Raphson would move in the wrong direction (towards a local maximum rather
        # than local minimum), so we move an arbitrary amount in the right direction.
        if numerator > 0.0:
            improved_u = u * 0.98 - 0.01
        elif numerator < 0.0:
            # Deliberately asymmetrical, to reduce the chance of cycling.
            improved_u = 0.031 + u * 0.98
        else:
            improved_u = u

    if not np.isfinite(improved_u):
        improved_u = u
    elif improved_u < 0.0:
        improved_u = 0.0
    elif improved_u > 1.0:
        improved_u = 1.0

    # Ensure that improved_u isn't actually worse.
    diff_lensq: float = lensq(diff)
    proportion: float = 0.125
    while True:
        if lensq(bezier_pt(3, q, improved_u) - p) > diff_lensq:
            if proportion > 1.0:
                improved_u = u
                break
            improved_u = (1 - proportion) * improved_u + proportion * u
        else:
            break
        proportion += 0.125
    return improved_u


def reparameterize(
    d: list[qt.QPointF],
    u: NDArray[np.float64],
    bezCurve: BezierCurve,
) -> NDArray[np.float64]:
    """
    Given set of points and their parameterization, try to find a better assignment of parameter
    values for the points.

    :param d  Array of digitized points.
    :param u  Current parameter values.
    :param bezCurve  Current fitted curve.
                 Also, the size of the array that is allocated for return.
    """
    assert 2 <= len(d)
    assert u.size == len(d)
    assert bezCurve[0] == d[0]
    assert bezCurve[3] == d[-1]
    assert u[0] == 0.0
    assert u[-1] == 1.0
    # Otherwise, consider including 0 and last in the below loop.

    for i in range(1, len(d) - 1):
        u[i] = newton_raphson_root_find(bezCurve, d[i], u[i])
        u[i] = newton_raphson_root_find(bezCurve, d[i], u[i])

    return u


def sp_bezier_fit_cubic(
    bezier: list[qt.QPointF],
    data: list[qt.QPointF],
    error: float,
) -> int:
    """
    Fit a single-segment Bézier curve to a set of digitized points.

    :return Number of segments generated, or -1 on error.
    """
    return sp_bezier_fit_cubic_r(bezier, data, error, 1)


def sp_bezier_fit_cubic_r(
    bezier: list[qt.QPointF],
    data: list[qt.QPointF],
    error: float,
    max_beziers: int,
) -> int:
    """
    Fit a multi-segment Bézier curve to a set of digitized points, with
    possible weedout of identical points and NaNs.

    :param max_beziers: Maximum number of generated segments

    :return Number of segments generated, or -1 on error.
    """
    if not data:
        return -1
    if max_beziers >= (1 << (31 - 2 - 1 - 3)):
        return -1

    uniqued_data: qt.QPolygonF = qt.QPolygonF(
        copy_without_nans_or_adjacent_duplicates(data)
    )
    if uniqued_data.size() < 2:
        return 0

    # Call fit-cubic function with recursion.
    return sp_bezier_fit_cubic_full(
        bezier,
        [],
        uniqued_data.data(),
        unconstrained_tangent,
        unconstrained_tangent,
        error,
        max_beziers,
    )


def sp_bezier_fit_cubic_full(
    bezier: list[qt.QPointF],
    split_points: list[int],
    data: list[qt.QPointF],
    t_hat1: qt.QPointF,
    t_hat2: qt.QPointF,
    error: float,
    max_beziers: int,
) -> int:
    """
    Fit a multi-segment Bézier curve to a set of digitized points, without
    possible weedout of identical points and NaNs.

    :param data: should be uniqued, i.e. not exist i: data[i] == data[i + 1].
    :param max_beziers: Maximum number of generated segments
    """

    #  Max times to try iterating
    max_iterations: int = 4

    if not data:
        return -1
    if len(data) < 2:
        return 0
    if len(data) == 2:
        # We have 2 points, which can be fitted trivially.
        bezier[0] = data[0]
        bezier[3] = data[-1]
        dist: float = L2(data[-1] - data[0]) * (1.0 / 3.0)
        if isnan(dist):
            # Numerical problem, fall back to straight line segment.
            bezier[1] = bezier[0]
            bezier[2] = bezier[3]
        else:
            bezier[1] = (
                ((2 * bezier[0] + bezier[3]) * (1.0 / 3.0))
                if t_hat1.isNull()
                else (bezier[0] + dist * t_hat1)
            )
            bezier[2] = (
                ((bezier[0] + 2 * bezier[3]) * (1.0 / 3.0))
                if t_hat2.isNull()
                else (bezier[3] + dist * t_hat2)
            )
        return 1

    # Parameterize points, and attempt to fit curve
    splitPoint: int  # Point to split point set at.
    u: NDArray[np.float64] = chord_length_parameterize(data)
    if u[-1] == 0.0:
        # Zero-length path: every point in data[] is the same.
        #
        # (Clients aren't allowed to pass such data; handling the case is defensive
        # programming.)
        #
        return 0

    bezier = generate_bezier(data, u, t_hat1, t_hat2, error)
    u = reparameterize(data, u, bezier)

    # Find max deviation of points to fitted curve.
    tolerance: float = sqrt(error + 1e-9)
    maxErrorRatio: float
    maxErrorRatio, splitPoint = compute_max_error_ratio(data, u, bezier, tolerance)

    if abs(maxErrorRatio) <= 1.0:
        return 1

    # If error not too large, then try some reparameterization and iteration.
    if 0.0 <= maxErrorRatio <= 3.0:
        for i in range(max_iterations):
            bezier = generate_bezier(data, u, t_hat1, t_hat2, error)
            u = reparameterize(data, u, bezier)
            maxErrorRatio, splitPoint = compute_max_error_ratio(
                data, u, bezier, tolerance
            )
            if abs(maxErrorRatio) <= 1.0:
                return 1

    is_corner: bool = maxErrorRatio < 0

    if is_corner:
        assert splitPoint < len(data)
        if splitPoint == 0:
            if t_hat1.isNull():
                # Got spike even with unconstrained initial tangent.
                splitPoint += 1
            else:
                return sp_bezier_fit_cubic_full(
                    bezier,
                    split_points,
                    data,
                    unconstrained_tangent,
                    t_hat2,
                    error,
                    max_beziers,
                )
        elif splitPoint == (len(data) - 1):
            if t_hat2.isNull():
                # Got spike even with unconstrained final tangent.
                splitPoint -= 1
            else:
                return sp_bezier_fit_cubic_full(
                    bezier,
                    split_points,
                    data,
                    t_hat1,
                    unconstrained_tangent,
                    error,
                    max_beziers,
                )

    if 1 < max_beziers:
        #
        # Fitting failed -- split at max error point and fit recursively
        #
        rec_max_beziers1: int = max_beziers - 1

        rec_t_hat2: qt.QPointF
        rec_t_hat1: qt.QPointF
        if is_corner:
            if not (0 < splitPoint < (len(data) - 1)):
                return -1
            rec_t_hat1 = rec_t_hat2 = unconstrained_tangent
        else:
            # Unit tangent vector at splitPoint.
            rec_t_hat2 = sp_darray_center_tangent(data, splitPoint)
            rec_t_hat1 = -rec_t_hat2
        nsegs1: int = sp_bezier_fit_cubic_full(
            bezier,
            split_points,
            data[: (splitPoint + 1)],
            t_hat1,
            rec_t_hat2,
            error,
            rec_max_beziers1,
        )
        if nsegs1 < 0:
            return -1
        assert nsegs1 != 0
        split_points[nsegs1 - 1] = splitPoint
        rec_max_beziers2: int = max_beziers - nsegs1
        nsegs2: int = sp_bezier_fit_cubic_full(
            bezier[nsegs1 * 4 :],
            split_points[nsegs1:],
            data[splitPoint:],
            rec_t_hat1,
            t_hat2,
            error,
            rec_max_beziers2,
        )
        if nsegs2 < 0:
            return -1

        return nsegs1 + nsegs2
    else:
        return -1


def sp_darray_left_tangent(
    d: list[qt.QPointF],
    tolerance_sq: float | None = None,
) -> qt.QPointF:
    """
    Estimate the (forward) tangent at point d[first + 0.5].

    Unlike the center and right versions, this calculates the tangent in
    the way one might expect, i.e., wrt increasing index into d.

    (2 <= len(d)), (d[0] != d[1]), and all[p in d] in_svg_plane(p).
    is_unit_vector(ret).
    """
    assert len(d) >= 2
    assert d[0] != d[1]
    if tolerance_sq is None:
        return unit_vector(d[1] - d[0])

    assert 0 <= tolerance_sq
    distsq: float = 0.0
    for i in range(1, len(d)):
        t: qt.QPointF = d[i] - d[0]
        distsq = lensq(t)
        if tolerance_sq < distsq:
            return unit_vector(t)
    else:
        return sp_darray_left_tangent(d) if distsq == 0.0 else unit_vector(d[-1] - d[0])


def sp_darray_right_tangent(
    d: list[qt.QPointF],
    tolerance_sq: float | None = None,
) -> qt.QPointF:
    """
    Estimates the (backward) tangent at d[last].

    The tangent is “backwards”, i.e. it is with respect to
    decreasing index rather than increasing index.

    (2 <= len), (d[-1] != d[-2]), and all[p in d] in_svg_plane(p).
    is_unit_vector(ret).
    """
    assert 2 <= len(d)
    assert d[-1] != d[-2]
    if tolerance_sq is None:
        return unit_vector(d[-2] - d[-2])

    assert 0 <= tolerance_sq
    distsq: float = 0.0
    for i in range(len(d) - 2, -1, -1):
        t: qt.QPointF = d[i] - d[-1]
        distsq = lensq(t)
        if tolerance_sq < distsq:
            return unit_vector(t)
    else:
        return (
            sp_darray_right_tangent(d) if distsq == 0.0 else unit_vector(d[-1] - d[0])
        )


def sp_darray_center_tangent(d: list[qt.QPointF], center: int) -> qt.QPointF:
    """
    Estimates the (backward) tangent at d[center], by averaging the two
    segments connected to d[center] (and then normalizing the result).

    The tangent is “backwards”, i.e. it is with respect to
    decreasing index rather than increasing index.

    (0 < center < len(d) - 1) and :param d is unique (at least in
    the immediate vicinity of :param center).
    """
    assert center != 0
    assert center < len(d) - 1

    ret: qt.QPointF
    if d[center + 1] == d[center - 1]:
        # Rotate 90 degrees in an arbitrary direction.
        diff: qt.QPointF = d[center] - d[center - 1]
        ret = rot90(diff)
    else:
        ret = d[center - 1] - d[center + 1]
    return unit_vector(ret)


def compute_hook(
    a: qt.QPointF,
    b: qt.QPointF,
    u: float | np.float64 | NDArray[np.float64],
    bezCurve: BezierCurve,
    tolerance: float,
) -> float:
    """
    Whereas compute_max_error_ratio() checks for itself that each data point
    is near some point on the curve, this function checks that each point on
    the curve is near some data point (or near some point on the polyline
    defined by the data points, or something like that: we allow for a
    "reasonable curviness" from such a polyline).  "Reasonable curviness"
    means we draw a circle centred at the midpoint of a..b, of radius
    proportional to the length |a - b|, and require that each point on the
    segment of bezCurve between the parameters of a and b be within that circle.
    If any point p on the bezCurve segment is outside of that allowable
    region (circle), then we return some metric that increases with the
    distance from p to the circle.

     Given that this is a fairly arbitrary criterion for finding appropriate
     places for sharp corners, we test only one point on bezCurve, namely
     the point on bezCurve with parameter halfway between our estimated
     parameters for a and b.  (Alternatives are taking the farthest of a
     few parameters between those of a and b, or even using a variant of
     NewtonRaphsonFindRoot() for finding the maximum rather than minimum
     distance.)
    """
    p: qt.QPointF = bezier_pt(3, bezCurve, u)
    diff: qt.QPointF = 0.5 * (a + b) - p
    dist: float = L2(diff)
    if dist < tolerance:
        return 0

    # factor of 0.2 introduced by JSS to stop more hooks
    allowed: float = L2(b - a) * 0.2 + tolerance
    return dist / allowed
    # todo
    #  efficiency: Hooks are very rare.  We could start by comparing
    #  `distsq`, only resorting to the more expensive `L2` in cases of
    #  uncertainty.
