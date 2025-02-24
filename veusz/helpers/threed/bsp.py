import numpy as np
from numpy.typing import NDArray

from ._constants import EPS
from .fragment import Fragment

EMPTY_BSP_IDX: int = int.from_bytes(b"\xff" * 4, "little", signed=False)


class BSPRecord:
    def __init__(self):
        # fragments stored in this node, in terms of the index to an array
        # of indexes, frag_idxs
        self.minfragidxidx: int = 0
        self.nfrags: int = 0
        # indices in bsp_recs to the BSPRecord items in front and behind
        self.frontidx: int = EMPTY_BSP_IDX
        self.backidx: int = EMPTY_BSP_IDX


class BSPBuilder:
    """
    This class defines a specialised Binary Space Paritioning (BSP)
    buliding routine. 3D space is split recursively by planes to
    separate objects into front and back entries. The idea is to only
    use the BSP tree _once_, which is unlike normal uses of BSP. It is
    used to create a robust back->front ordering for a particular
    viewing direction. To avoid lots of dynamic memory allocation and
    to reduce overheads, the nodes in the BSP tree are stored in a
    vector.
    """

    def __init__(self, fragvec: list[Fragment], viewdirn: NDArray):
        """
        construct the BSP tree from the fragments given and a particular
        viewing direction
        """
        # the nodes in the tree
        # initial record
        self.bsp_recs: list[BSPRecord] = [BSPRecord()]
        # vector of indices to the fragments vector
        self.frag_idxs: list[int] = []

        # add every non-empty fragment onto a list of fragments to process
        to_process: list[int] = []
        for i in range(len(fragvec)):
            if fragvec[i].type != Fragment.FragmentType.FR_NONE:
                to_process.append(i)

        # these are where indices for the front and back side of the plane
        idxback: list[int] = []
        idxfront: list[int] = []
        planepts: list[NDArray]

        # stack of items to process
        stack: list[BSPStackItem] = [BSPStackItem(0, len(to_process))]

        while stack:
            stackitem: BSPStackItem = stack[-1]
            stack = stack[:-1]

            # this is the bsp record with which the items are associated
            rec: BSPRecord = self.bsp_recs[stackitem.bspidx]
            rec.minfragidxidx = len(self.frag_idxs)  # where the items get added

            # if more than item to process then choose a plane, then split
            planepts_valid, planepts = findPlane(
                to_process, len(to_process) - stackitem.nidxs, fragvec
            )
            if stackitem.nidxs > 1 and planepts_valid:
                # norm of plane (making sure it points to observer)
                norm: NDArray = np.cross(
                    planepts[1] - planepts[0], planepts[2] - planepts[0]
                )
                if np.dot(norm, viewdirn) < 0:
                    norm = -norm
                # approximately normalise
                norm /= abs(norm[0]) + abs(norm[1]) + abs(norm[2])

                to_process_size: int = len(to_process)
                for i in range(to_process_size - stackitem.nidxs, to_process_size):
                    fidx: int = to_process[i]
                    if fragvec[fidx].type == Fragment.FragmentType.FR_PATH:
                        handlePath(
                            norm,
                            planepts[0],
                            fragvec,
                            fidx,
                            self.frag_idxs,
                            idxfront,
                            idxback,
                        )
                    elif fragvec[fidx].type == Fragment.FragmentType.FR_LINESEG:
                        handleLine(
                            norm,
                            planepts[0],
                            fragvec,
                            fidx,
                            self.frag_idxs,
                            idxfront,
                            idxback,
                        )
                    elif fragvec[fidx].type == Fragment.FragmentType.FR_TRIANGLE:
                        handleTriangle(
                            norm,
                            planepts[0],
                            fragvec,
                            fidx,
                            self.frag_idxs,
                            idxfront,
                            idxback,
                        )

                # number added to this node
                rec.nfrags = len(self.frag_idxs) - rec.minfragidxidx
                # remove items to process
                to_process = to_process[: -stackitem.nidxs]

                if rec.nfrags == 0:
                    if not idxfront and idxback:
                        self.frag_idxs.extend(idxback)
                        rec.nfrags = len(idxback)
                        idxback = []
                    elif not idxback and idxfront:
                        self.frag_idxs.extend(idxfront)
                        rec.nfrags = len(idxfront)
                        idxfront = []

                # push_back invalidates rec, so we don't use it below
                if idxfront:
                    newbspidx: int = len(self.bsp_recs)
                    self.bsp_recs[stackitem.bspidx].frontidx = newbspidx
                    self.bsp_recs.append(BSPRecord())
                    stack.append(BSPStackItem(newbspidx, len(idxfront)))
                    to_process.extend(idxfront)
                    idxfront = []

                if idxback:
                    newbspidx: int = len(self.bsp_recs)
                    self.bsp_recs[stackitem.bspidx].backidx = newbspidx
                    # add the record to be processed
                    self.bsp_recs.append(BSPRecord())
                    # new set of items to process
                    stack.append(BSPStackItem(newbspidx, len(idxback)))
                    # and add onto to process list
                    to_process.extend(idxback)
                    idxback = []
            else:
                # single item to process or plane couldn't be found
                self.frag_idxs.extend(to_process[: -stackitem.nidxs])
                to_process = to_process[: -stackitem.nidxs]
                rec.nfrags = stackitem.nidxs

    def getFragmentIdxs(self, fragvec: list[Fragment]) -> list[int]:
        """
        return a vector of fragment indexes in drawing order

        This is a non-recursive function to walk the tree. We keep a
        "stack" to do the walking. Because we have to walk the back before
        the current items and then the front, we have two types of stack
        items: WALK_START and WALK_RECS
        """
        retn: list[int] = []

        stack: list[WalkStackItem] = [WalkStackItem(0, 0)]

        temp: list[int]

        while stack:
            stackitem: WalkStackItem = stack[-1]
            stack = stack[:-1]

            rec: BSPRecord = self.bsp_recs[stackitem.bsp_idx]

            if stackitem.stage == 0:
                if rec.frontidx != EMPTY_BSP_IDX:
                    stack.append(WalkStackItem(rec.frontidx, 0))
                stack.append(WalkStackItem(stackitem.bsp_idx, 1))
                if rec.backidx != EMPTY_BSP_IDX:
                    stack.append(WalkStackItem(rec.backidx, 0))
            else:
                # Sort images in plane by Z. This is helpful for points
                # which may overlap.
                temp = self.frag_idxs[
                    rec.minfragidxidx : (rec.minfragidxidx + rec.nfrags)
                ]

                temp.sort(key=lambda x: fragZ(fragvec[x]))

                for _type in (
                    Fragment.FragmentType.FR_TRIANGLE,
                    Fragment.FragmentType.FR_LINESEG,
                    Fragment.FragmentType.FR_PATH,
                ):
                    for i in temp:
                        if fragvec[i].type == _type:
                            retn.append(i)

        return retn


def triAreaSqd2D(pts: list[NDArray]) -> float:
    """2d triangle area squared (only considering X/Y)"""
    a: float = (
        pts[0][0] * pts[1][1]
        - pts[0][1] * pts[1][0]
        + pts[1][0] * pts[2][1]
        - pts[1][1] * pts[2][0]
        + pts[2][0] * pts[0][1]
        - pts[2][1] * pts[0][0]
    )
    return a**2


def findPlane(idxs: list[int], startidx: int, frags: list[Fragment]) -> (bool, NDArray):
    """
    Find set of three points to define a plane (pts).
    Needs to find points which are not the same return true if ok

    Algorithm is to find the biggest triangle on the plane of the
    image (X/Y), then if none, the set of 3 points with the largest
    area. We prefer triangles as splitting them is expensive in
    terms of numbers of fragments.

    Plane of image triangles are preferred as splitting in the
    opposite directions gives a weird viewing order for points

    Using larger triangles is a heuristic to prevent lots of split
    triangles. Empirically it seems to reduce the number of final
    fragments by quite a lot.
    """

    maxtriarea2: float = -1.0
    besttri: int = EMPTY_BSP_IDX

    temppts: list[NDArray] = [np.zeros(3)] * 3
    ptct: int = 0
    maxptsarea2: float = -1.0

    endidx: int = len(idxs)
    pts: NDArray = np.zeros(3)
    for i in range(startidx, endidx):
        f: Fragment = frags[idxs[i]]
        if f.type == Fragment.FragmentType.FR_TRIANGLE:
            areasqd: float = triAreaSqd2D(f.points)
            if areasqd > maxtriarea2:
                maxtriarea2 = areasqd
                besttri = i
        elif besttri == EMPTY_BSP_IDX:
            # only bother looking at other points if we haven't found a
            # triangle yet
            # this is a crude rotating buffer looking for larger triangles
            # in the plane of the image
            if (
                f.type == Fragment.FragmentType.FR_LINESEG
                or f.type == Fragment.FragmentType.FR_PATH
            ):
                temppts[ptct % 3] = f.points[0]
                ptct += 1
            if f.type == Fragment.FragmentType.FR_LINESEG:
                temppts[ptct % 3] = f.points[1]
                ptct += 1
            if ptct >= 3:
                areasqd: float = triAreaSqd2D(temppts)
                if areasqd > maxptsarea2:
                    for j in range(3):
                        pts[j] = temppts[j]
                    maxptsarea2 = areasqd

    # return triangle
    if besttri != EMPTY_BSP_IDX:
        for i in range(3):
            pts[i] = frags[idxs[besttri]].points[i]
        return True, pts
    else:
        # is the returned triangle valid?
        return maxptsarea2 > EPS, pts


def dotsign(dot) -> int:
    """sign of calculated dot"""
    return 1 if dot > EPS else -1 if dot < -EPS else 0


def handlePath(
    norm: NDArray,
    plane0: NDArray,
    v: list[Fragment],
    fidx: int,
    idxsame: list[int],
    idxfront: list[int],
    idxback: list[int],
):
    """is path in front, on or behind plane?"""
    sign: int = dotsign(np.dot(norm, v[fidx].points[0] - plane0))
    if sign == 1:
        idxfront.append(fidx)
    elif sign == -1:
        idxback.append(fidx)
    else:
        idxsame.append(fidx)


def handleLine(
    norm: NDArray,
    plane0: NDArray,
    fragvec: list[Fragment],
    fidx: int,
    idxsame: list[int],
    idxfront: list[int],
    idxback: list[int],
):
    """is line in front, on or behind plane"""
    f: Fragment = fragvec[fidx]

    dot0: float = float(np.dot(norm, f.points[0] - plane0))
    sign0: int = dotsign(dot0)
    sign1: int = dotsign(np.dot(norm, f.points[1] - plane0))
    signsum: int = sign0 + sign1

    # first cases are that the line is simply on one side
    if sign0 == 0 and sign1 == 0:
        idxsame.append(fidx)
    elif signsum > 0:
        idxfront.append(fidx)
    elif signsum < 0:
        idxback.append(fidx)
    else:
        # split line. Note: we change original, then push a copy, as
        # a push invalidates the original reference
        linevec: NDArray = f.points[1] - f.points[0]
        d: float = -dot0 / np.dot(linevec, norm)
        newpt: NDArray = f.points[0] + linevec * d
        fcpy: Fragment = f

        # overwrite original with +ve part
        f.points[sign0 >= 0] = newpt
        idxfront.append(fidx)

        # write copy with -ve part
        fcpy.points[sign0 < 0] = newpt
        idxback.append(len(fragvec))
        fragvec.append(fcpy)


def handleTriangle(
    norm: NDArray,
    plane0: NDArray,
    fragvec: list[Fragment],
    fidx: int,
    idxsame: list[int],
    idxfront: list[int],
    idxback: list[int],
):
    """is triangle in front, behind or on plane?"""
    f: Fragment = fragvec[fidx]

    dots: list[float] = []
    signs: list[int] = []
    for i in range(3):
        dots.append(float(np.dot(norm, f.points[i] - plane0)))
        signs.append(dotsign(dots[i]))
    signsum: int = sum(signs)
    nzero: int = signs.count(0)

    if nzero == 3:
        # all on plane
        idxsame.append(fidx)
    elif signsum + nzero == 3:
        # all +ve or on plane
        idxfront.append(fidx)
    elif signsum - nzero == -3:
        # all -ve or on plane
        idxback.append(fidx)
    elif nzero == 1:
        # split triangle into two as one point is on the plane and
        # the other two are either side
        # index of point on plane
        idx0: int = 0 if signs[0] == 0 else 1 if signs[1] == 0 else 2

        linevec: NDArray = f.points[(idx0 + 2) % 3] - f.points[(idx0 + 1) % 3]
        d: float = -dots[(idx0 + 1) % 3] / np.dot(linevec, norm)
        newpt: NDArray = f.points[(idx0 + 1) % 3] + linevec * d

        fcpy: Fragment = f

        # modify original
        f.points[(idx0 + 2) % 3] = newpt
        if dots[(idx0 + 1) % 3] > 0:
            idxfront.append(fidx)
        else:
            idxback.append(fidx)

        # then make a copy for the other side
        fcpy.points[(idx0 + 1) % 3] = newpt
        if dots[(idx0 + 2) % 3] > 0:
            idxfront.append(len(fragvec))
        else:
            idxback.append(len(fragvec))
        fragvec.append(fcpy)
    else:  # nzero==0
        # split triangle into three, as no points are on the plane

        # point index by itself on one side of plane
        diffidx: int = 0 if signs[1] == signs[2] else 1 if signs[0] == signs[2] else 2

        # new points on plane
        linevec_p1: NDArray = f.points[(diffidx + 1) % 3] - f.points[diffidx]
        d_p1: float = -dots[diffidx] / np.dot(linevec_p1, norm)
        newpt_p1: NDArray = f.points[diffidx] + linevec_p1 * d_p1
        linevec_p2: NDArray = f.points[(diffidx + 2) % 3] - f.points[diffidx]
        d_p2: float = -dots[diffidx] / np.dot(linevec_p2, norm)
        newpt_p2: NDArray = f.points[diffidx] + linevec_p2 * d_p2

        # now make one triangle on one side and two on the other
        fcpy1: Fragment = f
        fcpy2: Fragment = f

        # modify original: triangle by itself on one side
        f.points[(diffidx + 1) % 3] = newpt_p1
        f.points[(diffidx + 2) % 3] = newpt_p2
        (idxfront if dots[diffidx] > 0 else idxback).append(fidx)

        # then add the other two on the other side
        fcpy1.points[diffidx] = newpt_p1
        fcpy1.points[(diffidx + 2) % 3] = newpt_p2
        (idxfront if dots[diffidx] < 0 else idxback).append(len(fragvec))
        fragvec.append(fcpy1)
        fcpy2.points[diffidx] = newpt_p2
        (idxfront if dots[diffidx] < 0 else idxback).append(len(fragvec))
        fragvec.append(fcpy2)


class BSPStackItem:
    def __init__(self, bspidx: int, nidxs: int):
        # BSPRecord we are working on here
        self.bspidx: int = bspidx
        # Number of fragment indices in to_process
        self.nidxs: int = nidxs


def fragZ(f: Fragment) -> float:
    """
    get Z component of fragment, nudging points and lines forward
    Z decreases away from viewer
    """
    if f.type == Fragment.FragmentType.FR_TRIANGLE:
        return min(f.points[0][2], f.points[1][2], f.points[2][2])
    elif f.type == Fragment.FragmentType.FR_LINESEG:
        return min(f.points[0][2], f.points[1][2]) + 1e-5
    elif f.type == Fragment.FragmentType.FR_PATH:
        return f.points[0][2] + 2e-5
    return np.nan


class FragZCompare:
    def __init__(self, v: list[Fragment]):
        self.v = v[:]

    def relation(self, a: int, b: int) -> bool:
        return fragZ(self.v[a]) < fragZ(self.v[b])


class WalkStackItem:
    """
    Similar stack item to keep track of the tree navigation while
    "drawing" the tree
    """

    def __init__(self, idx: int, stage: int):
        self.bsp_idx: int = idx
        self.stage: int = stage
