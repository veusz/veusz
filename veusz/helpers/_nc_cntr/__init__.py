from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

Z_VALUE: Final[int] = 0x0003
ZONE_EX: Final[int] = 0x0004
I_BNDY: Final[int] = 0x0008
J_BNDY: Final[int] = 0x0010
I0_START: Final[int] = 0x0020
I1_START: Final[int] = 0x0040
J0_START: Final[int] = 0x0080
J1_START: Final[int] = 0x0100
START_ROW: Final[int] = 0x0200
SLIT_UP: Final[int] = 0x0400
SLIT_DN: Final[int] = 0x0800
OPEN_END: Final[int] = 0x1000
ALL_DONE: Final[int] = 0x2000


# some helpful macros to find points relative to a given directed
# edge -- points are designated 0, 1, 2, 3 CCW around zone with 0 and
# 1 the endpoints of the current edge
def FORWARD(left, ix) -> int:
    return (1 if left > 1 else -ix) if left > 0 else (-1 if left < -1 else ix)


def POINT0(edge, fwd) -> int:
    return edge - (fwd if fwd > 0 else 0)


def POINT1(edge, fwd) -> int:
    return edge + (fwd if fwd < 0 else 0)


def IS_JEDGE(left) -> bool:
    return (left > 1) if left > 0 else (left < -1)


ANY_START: Final[int] = I0_START | I1_START | J0_START | J1_START


def START_MARK(left) -> int:
    return (
        (J1_START if left > 1 else I1_START)
        if left > 0
        else (J0_START if left < -1 else I0_START)
    )


@dataclass
class Csite:
    """
    here is the minimum structure required to tell where we are in the
    mesh sized data array
    """

    edge: int = 0  # ij of current edge
    left: int = (
        0  # +-1 or +-imax as the zone is to right, left, below, or above the edge
    )
    imax: int = 0  # imax for the mesh
    jmax: int = 0  # jmax for the mesh
    n: int = 0  # number of points marked on this curve so far
    count: int = 0  # count of start markers visited

    # starting site on this curve for closure
    edge0: int = 0
    left0: int = 0
    level0: int = 0  # starting level for closure
    edge00: int = 0  # site needing START_ROW mark

    triangle: NDArray[np.int64] | None = None  # triangulation array for the mesh
    reg: NDArray[np.int64] | None = None  # region array for the mesh

    # the data about edges, zones, and points -- boundary or not, exists
    #  or not, z value 0, 1, or 2 -- is kept in a mesh sized data array
    #  the array consists of the following bits:
    #  Z_VALUE     (2 bits) 0, 1, or 2 function value at point
    #  ZONE_EX     1 zone exists, 0 zone doesn't exist
    #  I_BNDY      this i-edge (i=constant edge) is a mesh boundary
    #  J_BNDY      this j-edge (i=constant edge) is a mesh boundary
    #  I0_START    this i-edge is a start point into zone to left
    #  I1_START    this i-edge is a start point into zone to right
    #  J0_START    this j-edge is a start point into zone below
    #  J1_START    this j-edge is a start point into zone above
    #  START_ROW   next start point is in current row (accelerates 2nd pass)
    #  SLIT_UP     marks this i-edge as the beginning of a slit upstroke
    #  SLIT_DN     marks this i-edge as the beginning of a slit downstroke
    #  OPEN_END    marks an i-edge start point whose other endpoint is
    #              on a boundary for the single level case
    #  ALL_DONE    marks final start point
    data: NDArray[np.int64] | None = None  # added by EF
    # making the actual marks requires a bunch of other stuff
    # mesh coordinates and function values
    x: NDArray[np.float64] | None = None
    y: NDArray[np.float64] | None = None
    z: NDArray[np.float64] | None = None
    # output contour points
    xcp: NDArray[np.float64] | None = None
    ycp: NDArray[np.float64] | None = None

    # contour levels, zlevel[1]<=zlevel[0] signals single level case
    zlevel0: float = 0.0
    zlevel1: float = 0.0


def data_init(site: Csite, region: int, nchunk: int) -> Csite:
    """
    this initializes the data array for curve_tracer

    The sole function of the `region` argument is to specify the
    value in Csite.reg that denotes a missing zone. We always
    use zero.
    """

    data: NDArray[np.int64] = site.data
    imax: int = site.imax
    jmax: int = site.jmax
    z: NDArray[np.float64] | None = site.z
    zlev0: float = site.zlevel0
    zlev1: float = site.zlevel1
    two_levels: bool = zlev1 > zlev0
    reg: NDArray[np.int64] | None = site.reg
    count: int = 0
    started: bool = False

    icsize: int = imax - 1
    jcsize: int = jmax - 1

    if nchunk and two_levels:
        # figure out chunk sizes
        #  -- input nchunk is square root of maximum allowed zones per chunk
        #  -- start points for single level case are wrong, so don't try it
        inum: int = (nchunk**2) // (jmax - 1)
        jnum: int = (nchunk**2) // (imax - 1)
        if inum < nchunk:
            inum = nchunk
        if jnum < nchunk:
            jnum = nchunk
        # ijnum= actual number of chunks,
        # ijrem= number of those chunks needing one more zone (ijcsize+1)
        inum = (imax - 2) // inum + 1
        icsize = (imax - 1) // inum
        irem = (imax - 1) % inum
        jnum = (jmax - 2) // jnum + 1
        jcsize = (jmax - 1) // jnum
        jrem = (jmax - 1) % jnum
        # convert ijrem into value of i or j at which to begin adding an
        # extra zone
        irem = (inum - irem) * icsize
        jrem = (jnum - jrem) * jcsize
    else:
        irem = imax
        jrem = jmax

    # do everything in a single pass through the data array to
    # minimize cache faulting (z, reg, and data are potentially
    # very large arrays)
    # access to the z and reg arrays is strictly sequential,
    # but we need two rows (+-imax) of the data array at a time
    if z[0, 0] > zlev0:
        data[0, 0] = 2 if two_levels and z[0, 0] > zlev1 else 1
    else:
        data[0, 0] = 0
    jchunk: int = 0
    for j in range(jmax):
        ichunk: int = 0
        i_was_chunk: int = 0
        for i in range(imax):
            # transfer zonal existence from reg to data array
            # -- get these for next row so we can figure existence of
            #    points and j-edges for this row
            data[i, j + 1] = 0
            if reg is not None:
                if (reg[i, j + 1] == region) if region else (reg[i, j + 1] != 0):
                    data[i, j + 1] = ZONE_EX
            else:
                if i < imax - 1 and j < jmax - 1:
                    data[i, j + 1] = ZONE_EX

            # translate z values to 0, 1, 2 flags
            if j == 0 and i < imax:
                data[i + 1, j] = 0
            if i < imax - 1 and j < jmax and z[i + 1, j] > zlev0:
                data[i + 1, j] |= 2 if (two_levels and z[i + 1, j] > zlev1) else 1

            # apply edge boundary marks
            ibndy = i == ichunk or (data[i, j] & ZONE_EX) != (data[i + 1, j] & ZONE_EX)
            jbndy = j == jchunk or (data[i, j] & ZONE_EX) != (data[i, j + 1] & ZONE_EX)
            if ibndy:
                data[i, j] |= I_BNDY
            if jbndy:
                data[i, j] |= J_BNDY

            # apply i-edge start marks
            # -- i-edges are only marked when actually cut
            # -- no mark is necessary if one of the j-edges which share
            #    the lower endpoint is also cut
            # -- no I0 mark necessary unless filled region below some cut,
            #    no I1 mark necessary unless filled region above some cut
            if j > 0:
                v0: int = data[i, j] & Z_VALUE
                vb: int = data[i, j - 1] & Z_VALUE
                if v0 != vb:
                    # i-edge is cut
                    if ibndy:
                        if data[i, j] & ZONE_EX:
                            data[i, j] |= I0_START
                            count += 1
                        if data[i + 1, j] & ZONE_EX:
                            data[i, j] |= I1_START
                            count += 1
                    else:
                        va: int = data[i - 1, j] & Z_VALUE
                        vc: int = data[i + 1, j] & Z_VALUE
                        vd: int = data[i + 1, j - 1] & Z_VALUE
                        if (
                            v0 != 1
                            and va != v0
                            and (vc != v0 or vd != v0)
                            and (data[i, j] & ZONE_EX)
                        ):
                            data[i, j] |= I0_START
                            count += 1
                        if (
                            vb != 1
                            and va == vb
                            and (vc == vb or vd == vb)
                            and (data[i + 1, j] & ZONE_EX)
                        ):
                            data[i, j] |= I1_START
                            count += 1
            # apply j-edge start marks
            # -- j-edges are only marked when they are boundaries
            # -- all cut boundary edges marked
            # -- for two level case, a few uncut edges must be marked
            if i and jbndy:
                v0: int = data[i, j] & Z_VALUE
                vb: int = data[i - 1, j] & Z_VALUE
                if v0 != vb:
                    if data[i, j] & ZONE_EX:
                        data[i, j] |= J0_START
                        count += 1
                    if data[i, j + 1] & ZONE_EX:
                        data[i, j] |= J1_START
                        count += 1
                elif two_levels and v0 == 1:
                    if data[i, j + 1] & ZONE_EX:
                        if i_was_chunk or not (data[i - 1, j + 1] & ZONE_EX):
                            # lower left is a drawn part of boundary
                            data[i, j] |= J1_START
                            count += 1
                    elif data[i, j] & ZONE_EX:
                        if data[i - 1, j + 1] & ZONE_EX:
                            # weird case of open hole at lower left
                            data[i, j] |= J0_START
                            count += 1

            i_was_chunk = i == ichunk
            if i_was_chunk:
                ichunk += icsize + (ichunk >= irem)

        if j == jchunk:
            jchunk += jcsize + (jchunk >= jrem)

        # place first START_ROW marker
        if count and not started:
            data[0, j - 1] |= START_ROW
            started = True

    # place immediate stop mark if nothing found
    if not count:
        data[0, 0] |= ALL_DONE

    # initialize site
    site.edge0 = site.edge00 = site.edge = 0
    site.left0 = site.left = 0
    site.n = 0
    site.count = count

    return site


def mask_zones(
    i_max: int,
    j_max: int,
    mask: NDArray[np.bool],
    reg: NDArray[np.int64],
) -> None:
    """
    reg should have the same dimensions as data, which
    has an extra i_max + 1 points relative to Z.
    It differs from mask in being the opposite (True
    where a region exists, versus the mask, which is True
    where a data point is bad), and in that it marks
    zones, not points.  All four zones sharing a bad
    point must be marked as not existing.
    """

    reg[i_max, :] = 1

    for j in range(j_max):
        for i in range(i_max):
            if i == 0 or j == 0:
                reg[i, j] = 0
            if mask[i, j]:
                reg[i, j] = 0
                reg[i + 1, j] = 0
                reg[i, j + 1] = 0
                reg[i + 1, j + 1] = 0


def slit_cutter(site: Csite, up: int, pass2: int) -> int:
    """
    -- slit_cutter is never called for single level case
    """

    data: list[int] = site.data
    imax: int = site.imax
    n: int = site.n

    x: NDArray[np.float64] | None = site.x if pass2 else None
    y: NDArray[np.float64] | None = site.y if pass2 else None
    xcp: list[float] = site.xcp if pass2 else None
    ycp: list[float] = site.ycp if pass2 else None

    if up:
        # upward stroke of slit proceeds up left side of slit until
        # it hits a boundary or a point not between the contour levels
        # -- this never happens on the first pass */
        p1: int = site.edge
        z1: int
        while True:
            z1 = data[p1] & Z_VALUE
            if z1 != 1:
                site.edge = p1
                site.left = -1
                site.n = n
                return z1 != 0
            elif (
                data[p1] & J_BNDY
            ):  # this is very unusual case of closing on a mesh hole
                site.edge = p1
                site.left = -imax
                site.n = n
                return 2
            xcp[n] = x[p1]
            ycp[n] = y[p1]
            n += 1
            p1 += imax

    else:
        # downward stroke proceeds down right side of slit until it
        # hits a boundary or point not between the contour levels
        p0: int = site.edge
        z0: int
        # at beginning of first pass, mark first i-edge with SLIT_DN
        data[p0] |= SLIT_DN
        p0 -= imax
        while True:
            z0 = data[p0] & Z_VALUE
            if not pass2:
                if z0 != 1 or (data[p0] & I_BNDY) or (data[p0 + 1] & J_BNDY):
                    # at end of first pass, mark final i-edge with SLIT_UP
                    data[p0 + imax] |= SLIT_UP
                    # one extra count for splicing at outer curve
                    site.n = n + 1
                    return 4
                    # return same special value as for OPEN_END
            else:
                if z0 != 1:
                    site.edge = p0 + imax
                    site.left = 1
                    site.n = n
                    return z0 != 0
                elif data[p0 + 1] & J_BNDY:
                    site.edge = p0 + 1
                    site.left = imax
                    site.n = n
                    return 2
                elif data[p0] & I_BNDY:
                    site.edge = p0
                    site.left = 1
                    site.n = n
                    return 2
            if pass2:
                xcp[n] = x[p0]
                ycp[n] = y[p0]
                n += 1
            else:
                # on first pass need to count for upstroke as well
                n += 2
            p0 -= imax


def zone_crosser(site: Csite, level: int, pass2: int) -> int:
    """
    zone_crosser assumes you are sitting at a cut edge about to cross
    the current zone.  It always marks the initial point, crosses at
    least one zone, and marks the final point.  On non-boundary i-edges,
    it is responsible for removing start markers on the first pass.
    """
    data: list[int] = site.data
    edge: int = site.edge
    left: int = site.left
    n: int = site.n
    fwd: int = FORWARD(left, site.imax)
    jedge: bool = IS_JEDGE(left)
    edge0: int = site.edge0
    left0: int = site.left0
    level0: bool = site.level0 == level
    two_levels: bool = site.zlevel1 > site.zlevel0
    triangle: list[int] = site.triangle

    x: NDArray[np.float64] | None = site.x if pass2 else None
    y: NDArray[np.float64] | None = site.y if pass2 else None
    z: NDArray[np.float64] | None = site.z if pass2 else None
    zlevel: float = {0: site.zlevel0, 1: site.zlevel1}[level] if pass2 else 0.0
    xcp: list[float] = site.xcp if pass2 else None
    ycp: list[float] = site.ycp if pass2 else None

    keep_left: int = 0  # flag to try to minimize curvature in saddles
    done: int = 0

    if level:
        level = 2

    while True:
        # set edge endpoints
        p0 = POINT0(edge, fwd)
        p1 = POINT1(edge, fwd)

        # always mark cut on current edge
        if pass2:
            # second pass actually computes and stores the point
            zcp: float = (zlevel - z[p0]) / (z[p1] - z[p0])
            xcp[n] = zcp * (x[p1] - x[p0]) + x[p0]
            ycp[n] = zcp * (y[p1] - y[p0]) + y[p0]
        if not done and not jedge:
            if n:
                # if this is not the first point on the curve, and we're
                # not done, and this is an i-edge, check several things
                if not two_levels and not pass2 and (data[edge] & OPEN_END):
                    # reached an OPEN_END mark, skip the n++
                    done = 4
                    # same return value 4 used below
                    break

                # check for curve closure -- if not, erase any start mark
                if edge == edge0 and left == left0:
                    # may signal closure on a downstroke
                    if level0:
                        done = 5 if (not pass2 and two_levels and left < 0) else 3
                elif not pass2:
                    start: int = data[edge] & (I0_START if fwd > 0 else I1_START)
                    if start:
                        data[edge] &= ~start
                        site.count -= 1
                    if not two_levels:
                        start = data[edge] & (I1_START if fwd > 0 else I0_START)
                        if start:
                            data[edge] &= ~start
                            site.count -= 1
        n += 1
        if done:
            break

        # cross current zone to another cut edge
        z0 = (data[p0] & Z_VALUE) != level
        # 1 if fill toward p0
        z1 = not z0
        # know level cuts edge
        z2 = (data[p1 + left] & Z_VALUE) != level
        z3 = (data[p0 + left] & Z_VALUE) != level
        if z0 == z2:
            bkwd: bool = False
            if z1 == z3:
                # this is a saddle zone, need triangle to decide
                # -- set triangle if not already decided for this zone
                zone: int = edge + (left if left > 0 else 0)
                if triangle:
                    if not triangle[zone]:
                        if keep_left:
                            triangle[zone] = -1 if jedge else 1
                        else:
                            triangle[zone] = 1 if jedge else -1
                    if not jedge if triangle[zone] > 0 else jedge:
                        bkwd = True
                else:
                    if keep_left:
                        bkwd = True
            if bkwd:
                # bend backward (left along curve)
                keep_left = 0
                jedge = not jedge
                edge = p0 + (left if left > 0 else 0)
                fwd, left = left, -fwd
            else:
                # bend forward (right along curve)
                keep_left = 1
                jedge = not jedge
                edge = p1 + (left if left > 0 else 0)
                fwd, left = -left, fwd
        elif z1 == z3:
            # bend backward (left along curve)
            keep_left = 0
            jedge = not jedge
            edge = p0 + (left if left > 0 else 0)
            fwd, left = left, -fwd
        else:
            # straight across to opposite edge
            edge += left
        # after crossing zone, edge/left/fwd is oriented CCW relative to
        # the next zone, assuming we will step there

        # now that we've taken a step, check for the downstroke
        # of a slit on the second pass (upstroke checked above)
        # -- taking step first avoids a race condition
        if pass2 and two_levels and not jedge:
            if left > 0:
                if data[edge] & SLIT_UP:
                    done = 6
            else:
                if data[edge] & SLIT_DN:
                    done = 5

        if not done:
            # finally, check if we are on a boundary
            if data[edge] & (J_BNDY if jedge else I_BNDY):
                done = 2 if two_levels else 4
                # flip back into the zone that exists
                left = -left
                fwd = -fwd
                if not pass2 and (edge != edge0 or left != left0):
                    start: int = data[edge] & START_MARK(left)
                    if start:
                        data[edge] &= ~start
                        site.count -= 1

    site.edge = edge
    site.n = n
    site.left = left
    return slit_cutter(site, done - 5, pass2) if done > 4 else done


def edge_walker(site: Csite, pass2: int) -> int:
    """
    edge_walker assumes that the current edge is being drawn CCW
    around the current zone.  Since only boundary edges are drawn
    and we always walk around with the filled region to the left,
    no edge is ever drawn CW.  We attempt to advance to the next
    edge on this boundary, but if current second endpoint is not
    between the two contour levels, we exit back to zone_crosser.
    Note that we may wind up marking no points.
    -- edge_walker is never called for single level case
    """

    data: list[int] = site.data
    edge: int = site.edge
    left: int = site.left
    n: int = site.n
    fwd: int = FORWARD(left, site.imax)
    p0: int = POINT0(edge, fwd)
    p1: int = POINT1(edge, fwd)
    jedge: bool = IS_JEDGE(left)
    edge0: int = site.edge0
    left0: int = site.left0
    level0: bool = site.level0 == 2
    marked: int

    x: NDArray[np.float64] | None = site.x if pass2 else None
    y: NDArray[np.float64] | None = site.y if pass2 else None
    xcp: list[float] = site.xcp if pass2 else None
    ycp: list[float] = site.ycp if pass2 else None

    z0: int
    z1: int
    heads_up: int = 0

    while True:
        # mark endpoint 0 only if value is 1 there, and this is a
        # two level task
        z0 = data[p0] & Z_VALUE
        z1 = data[p1] & Z_VALUE
        marked = 0
        if z0 == 1:
            # mark current boundary point
            if pass2:
                xcp[n] = x[p0]
                ycp[n] = y[p0]
            marked = 1
        elif not n:
            # if this is the first point is not between the levels
            # must do the job of the zone_crosser and mark the first cut here,
            # so that it will be marked again by zone_crosser as it closes
            if pass2:
                zcp: float = {0: site.zlevel0, 1: site.zlevel1}[(z0 != 0)]
                zcp = (zcp - site.z[p0]) / (site.z[p1] - site.z[p0])
                xcp[n] = zcp * (x[p1] - x[p0]) + x[p0]
                ycp[n] = zcp * (y[p1] - y[p0]) + y[p0]
            marked = 1
        if n:
            # check for closure
            if level0 and edge == edge0 and left == left0:
                site.edge = edge
                site.left = left
                site.n = n + marked
                # if the curve is closing on a hole, need to make a downslit
                if fwd < 0 and not (data[edge] & (J_BNDY if jedge else I_BNDY)):
                    return slit_cutter(site, 0, pass2)
                return 3
            elif pass2:
                if heads_up or (fwd < 0 and (data[edge] & SLIT_DN)):
                    site.edge = edge
                    site.left = left
                    site.n = n + marked
                    return slit_cutter(site, heads_up, pass2)
            else:
                # if this is not first point, clear start mark for this edge
                start: int = data[edge] & START_MARK(left)
                if start:
                    data[edge] &= ~start
                    site.count -= 1
        if marked:
            n += 1

        # if next endpoint not between levels, need to exit to zone_crosser
        if z1 != 1:
            site.edge = edge
            site.left = left
            site.n = n
            return z1 != 0
            # return level closest to p1

        # step to p1 and find next edge
        # -- turn left if possible, else straight, else right
        # -- check for upward slit beginning at same time */
        edge = p1 + (left if left > 0 else 0)
        if pass2 and jedge and fwd > 0 and (data[edge] & SLIT_UP):
            jedge = not jedge
            heads_up = 1
        elif data[edge] & (I_BNDY if jedge else J_BNDY):
            fwd, left = left, -fwd
            jedge = not jedge
        else:
            edge = p1 + (fwd if fwd > 0 else 0)
            if pass2 and not jedge and fwd > 0 and (data[edge] & SLIT_UP):
                heads_up = 1
            elif not (data[edge] & (J_BNDY if jedge else I_BNDY)):
                edge = p1 - (left if left < 0 else 0)
                jedge = not jedge
                fwd, left = -left, fwd
        p0 = p1
        p1 = POINT1(edge, fwd)


def curve_tracer(site: Csite, pass2: int) -> int:
    """
    curve_tracer finds the next starting point, then traces the curve,
    returning the number of points on this curve
    -- in a two level trace, the return value is negative on the
       first pass if the curve closed on a hole
    -- in a single level trace, the return value is negative on the
       first pass if the curve is an incomplete open curve
    -- a return value of 0 indicates no more curves
    """

    data: NDArray[np.int64] = site.data
    imax: int = site.imax
    edge0: int = site.edge0
    i0, j0 = edge0 % imax, edge0 // imax
    left0: int = site.left0
    edge00: int = site.edge00
    i00, j00 = edge00 % imax, edge00 // imax
    two_levels: bool = site.zlevel1 > site.zlevel0

    # it is possible for a single i-edge to serve as two actual start
    # points, one to the right and one to the left
    # -- for the two level case, this happens on the first pass for
    #    a doubly cut edge, or on a chunking boundary
    # -- for single level case, this is impossible, but a similar
    #    situation involving open curves is handled below
    # a second two start possibility is when the edge0 zone does not
    # exist and both the i-edge and j-edge boundaries are cut
    # yet another possibility is three start points at a junction
    # of chunk cuts
    # -- sigh, several other rare possibilities,
    #    allow for general case, just go in order i1, i0, j1, j0

    two_starts: int
    # print("curve_tracer pass %d\n" % pass2)
    # print_Csite(site)
    if left0 == 1:
        two_starts = data[i0, j0] & (I0_START | J1_START | J0_START)
    elif left0 == -1:
        two_starts = data[i0, j0] & (J1_START | J0_START)
    elif left0 == imax:
        two_starts = data[i0, j0] & J0_START
    else:
        two_starts = 0

    if pass2 or (i0, j0) == (0, 0):
        # zip up to row marked on first pass (or by data_init if edge0==0)
        # -- but not for double start case
        if not two_starts:
            # final start point marked by ALL_DONE marker
            first: bool = (i0, j0) == (0, 0) and not pass2
            e0: tuple[int, int] = i0, j0
            if data[i0, j0] & ALL_DONE:
                return 0
            while not (data[i0, j0] & START_ROW):
                j0 += 1
            if e0 == (i0, j0):
                i0 += 1
                # two starts handled specially
            if first:
                # if this is the very first start point, we want to remove
                # the START_ROW marker placed by data_init
                data[0, j0] &= ~START_ROW

    else:
        # first pass ends when all potential start points visited
        if site.count <= 0:
            # place ALL_DONE marker for second pass
            data[i00, j00] |= ALL_DONE
            # reset initial site for second pass
            site.edge0 = site.edge00 = site.left0 = 0
            return 0
        if not two_starts:
            i0 += 1

    level: int
    if two_starts:
        # trace second curve with this start immediately
        if left0 == 1 and (data[i0, j0] & I0_START):
            left0 = -1
            level = 2 if (data[i0, j0] & I_BNDY) else 0
        elif (left0 == 1 or left0 == -1) and (data[i0, j0] & J1_START):
            left0 = imax
            level = 2
        else:
            left0 = -imax
            level = 2

    else:
        # usual case is to scan for next start marker
        # -- on second pass, this is at most one row of mesh, but first
        #    pass hits nearly every point of the mesh, since it can't
        #    know in advance which potential start marks removed
        while not (data[i0, j0] & ANY_START):
            i0 += 1

        if data[i0, j0] & I1_START:
            left0 = 1
        elif data[i0, j0] & I0_START:
            left0 = -1
        elif data[i0, j0] & J1_START:
            left0 = imax
        else:  # data[i0, j0]&J0_START
            left0 = -imax

        if data[i0, j0] & (I1_START | I0_START):
            level = 2 if (data[i0, j0] & I_BNDY) else 0
        else:
            level = 2

    # this start marker will not be unmarked, but it has been visited
    if not pass2:
        site.count -= 1

    # if this curve starts on a non-boundary i-edge, we need to
    # determine the level
    if not level and two_levels:
        level = (
            int((data[i0, j0 - 1] & Z_VALUE) != 0)
            if left0 > 0
            else int((data[i0, j0] & Z_VALUE) != 0)
        )

    # initialize site for this curve
    site.edge = site.edge0 = i0 + j0 * imax
    site.left = site.left0 = left0
    site.level0 = level0 = level
    # for open curve detection only

    # single level case just uses zone_crosser
    if not two_levels:
        level = 0

    # to generate the curve, alternate between zone_crosser and
    # edge_walker until closure or first call to edge_walker in
    # single level case
    site.n = 0
    while True:
        if level < 2:
            level = zone_crosser(site, level, pass2)
        elif level < 3:
            level = edge_walker(site, pass2)
        else:
            break
    n = site.n

    # single level case may have ended at a boundary rather than closing
    # -- need to recognize this case here in order to place the
    #    OPEN_END mark for zone_crosser, remove this start marker,
    #    and be sure not to make a START_ROW mark for this case
    # two level case may close with slit_cutter, in which case start
    #    must also be removed and no START_ROW mark made
    # -- change sign of return n to inform caller
    if not pass2 and level > 3 and (two_levels or level0 == 0):
        if not two_levels:
            data[i0, j0] |= OPEN_END
        data[i0, j0] &= ~(I1_START if left0 > 0 else I0_START)
        mark_row = 0
        # do not mark START_ROW */
        n = -n
    else:
        if two_levels:
            mark_row = not two_starts
        else:
            mark_row = 1

    # on first pass, must apply START_ROW mark in column above previous
    # start marker
    # -- but skip if we just did second of two start case
    if not pass2 and mark_row:
        data[0, j0 - j00] |= START_ROW
        site.edge00 = i0 + j0 * imax

    return n


def build_cntr_list_p(nseg, xp, yp) -> list[list[tuple[float, float]]]:
    """
    Build a list of lists of points, where each point is an (x,y)
    tuple.
    """

    start: int
    end: int = 0

    all_contours: list[list[tuple[float, float]]] = []
    contour_list: list[tuple[float, float]] = list(zip(xp, yp, strict=True))
    for p in nseg:
        start = end
        end += p
        all_contours.append(contour_list[start:end])

    return all_contours


def build_cntr_list_v2(nseg, xp, yp) -> list[NDArray[np.float64]]:
    """
    Build a list of XY 2-D arrays, shape (N,2)
    """

    xyv: NDArray[np.float64]
    start: int
    end: int = 0

    all_contours: list[NDArray[np.float64]] = []

    xyv = np.asarray(zip(xp, yp, strict=True))
    for p in nseg:
        start = end
        end += p
        all_contours.append(xyv[start:end])

    return all_contours


class Cntr:
    def __init__(
        self,
        xpa: NDArray[np.float64],
        ypa: NDArray[np.float64],
        zpa: NDArray[np.float64],
        mask: NDArray[np.bool] | None = None,
    ) -> None:
        i_max: int = zpa.shape[0]
        j_max: int = zpa.shape[1]
        if (
            xpa.shape != zpa.shape
            or ypa.shape != zpa.shape
            or (mask is not None and mask.shape != zpa.shape)
        ):
            raise ValueError(
                "Arguments x, y, z, mask (if present) must have the same dimensions."
            )

        ijmax: tuple[int, int] = i_max, j_max
        nreg: tuple[int, int] = (i_max + 1, j_max + 1)

        self.site: Csite = Csite(
            imax=i_max,
            jmax=j_max,
            data=np.zeros(nreg, dtype=np.int64),
            triangle=np.zeros(ijmax, dtype=np.int64),
            reg=np.zeros(nreg) if mask is not None else None,
        )

        if mask is not None:
            mask_zones(i_max, j_max, mask, self.site.reg)

        #  I don't think we need to initialize site.data.

        self.site.x = xpa
        self.site.y = ypa
        self.site.z = zpa
        self.site.mpa = mask

    def trace(
        self,
        *levels: float,
        points: bool = False,
        nchunk: int = 0,
    ) -> list[list[tuple[float, float]]] | list[NDArray[np.float64]]:
        """
        The function is called once per contour level or level pair.
        If len(levels) is 1, a set of contour lines will be returned; if len(levels)
        is 2, the set of polygons bounded by the levels will be returned.
        If points is True, the lines will be returned as a list of list
        of points; otherwise, as a list of tuples of vectors.
        """

        site = self.site

        c_list: list[list[tuple[float, float]]] | list[NDArray[np.float64]]

        nparts: int = 0
        ntotal: int = 0
        nparts2: int = 0
        ntotal2: int = 0

        site.zlevel0 = levels[0]

        if len(levels) == 2:
            site.zlevel1 = levels[1]
        else:
            site.zlevel1 = levels[0]

        site.n = site.count = 0

        site = data_init(site, 0, nchunk)

        # make first pass to compute required sizes for second pass
        while True:
            n = curve_tracer(site, 0)

            if not n:
                break
            if n > 0:
                nparts += 1
                ntotal += n
            else:
                ntotal -= n

        xp0: list[float] = []
        yp0: list[float] = []
        nseg0: list[int] = []

        # second pass
        site.xcp = xp0
        site.ycp = yp0
        iseg = 0

        while True:
            n = curve_tracer(site, 1)
            if ntotal2 + n > ntotal:
                raise RuntimeError(
                    "curve_tracer: ntotal2, pass 2 exceeds ntotal, pass 1"
                )
            if n == 0:
                break
            if n > 0:
                # could add array bounds checking
                nseg0[iseg] = n
                site.xcp += n
                site.ycp += n
                ntotal2 += n
                nparts2 += 1
            else:
                raise RuntimeError("Negative n from curve_tracer in pass 2")

        if points:
            c_list = build_cntr_list_p(nseg0, xp0, yp0)
        else:
            c_list = build_cntr_list_v2(nseg0, xp0, yp0)

        return c_list
