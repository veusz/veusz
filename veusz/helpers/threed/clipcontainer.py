import numpy as np
from numpy.typing import NDArray

from ._constants import _EPS
from .fragment import Fragment
from .mmaths import vec3to4, vec4to3
from .objects import ObjectContainer


def clipLine(
    f: Fragment,
    onplane: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> None:
    """clip line by plane"""
    dot0: float = float(np.dot(f.points[0] - onplane, normal))
    bad0: bool = dot0 < -_EPS
    dot1: float = float(np.dot(f.points[1] - onplane, normal))
    bad1: bool = dot1 < -_EPS

    if not bad0 and not bad1:
        # both points on good side of plane
        pass  # do nothing
    elif bad0 and bad1:
        # both line points on bad side of plane
        f.type = Fragment.FragmentType.FR_NONE
    else:
        # clip line to plane (bad1 or bad2, not both)
        linevec: NDArray = f.points[1] - f.points[0]
        d: float = -dot0 / np.dot(linevec, normal)
        f.points[0 if bad0 else 1] = f.points[0] + linevec * d


def clipTriangle(
    v: list[Fragment],
    idx: int,
    onplane: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> None:
    """clip triangle by plane"""
    f: Fragment = v[idx]
    dotv: list[float] = []
    bad: list[bool] = []
    for i in range(3):
        dotv.append(float(np.dot(f.points[i] - onplane, normal)))
        bad.append(dotv[i] < -_EPS)
    badsum = sum(bad)

    if badsum == 0:
        # all points ok
        pass  # do nothing
    elif badsum == 1:
        # two points are good, one is bad
        badidx: int = 0 if bad[0] else 1 if bad[1] else 2

        # calculate where vectors from good to bad points
        # intercept plane
        good1: NDArray[np.float64] = f.points[(badidx + 1) % 3]
        linevec1: NDArray[np.float64] = good1 - f.points[badidx]
        d1: float = -dotv[badidx] / np.dot(linevec1, normal)
        icept1: NDArray[np.float64] = f.points[badidx] + linevec1 * d1

        good2: NDArray[np.float64] = f.points[(badidx + 2) % 3]
        linevec2: NDArray[np.float64] = good2 - f.points[badidx]
        d2: float = -dotv[badidx] / np.dot(linevec2, normal)
        icept2: NDArray[np.float64] = f.points[badidx] + linevec2 * d2

        # break into two triangles from good points to intercepts
        # note: the push back invalidates the original, so we have
        # to make a copy
        f.points[0] = good2
        f.points[1] = icept2
        f.points[2] = good1
        fcpy: Fragment = f
        fcpy.points[0] = good1
        fcpy.points[1] = icept1
        fcpy.points[2] = icept2
        v.append(fcpy)
    elif badsum == 2:
        # one point is ok, the other two are bad
        goodidx: int = 0 if not bad[0] else 1 if not bad[1] else 2

        # work out where vectors from ok point intercept with plane
        linevec1: NDArray[np.float64] = f.points[(goodidx + 1) % 3] - f.points[goodidx]
        d1: float = -dotv[goodidx] / np.dot(linevec1, normal)
        f.points[(goodidx + 1) % 3] = f.points[goodidx] + linevec1 * d1

        linevec2: NDArray[np.float64] = f.points[(goodidx + 2) % 3] - f.points[goodidx]
        d2: float = -dotv[goodidx] / np.dot(linevec2, normal)
        f.points[(goodidx + 2) % 3] = f.points[goodidx] + linevec2 * d2
    elif badsum == 3:
        # all points are bad
        f.type = Fragment.FragmentType.FR_NONE


def clipFragments(
    v: list[Fragment],
    start: int,
    onplane: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> None:
    """clip all fragments to the plane given"""

    n_frags: int = len(v)
    for _i in range(start, n_frags):
        f: Fragment = v[_i]
        if f.type == Fragment.FragmentType.FR_PATH:
            # point on wrong side of plane
            if np.dot(f.points[0] - onplane, normal) < -_EPS:
                f.type = Fragment.FragmentType.FR_NONE

        elif f.type == Fragment.FragmentType.FR_LINESEG:
            clipLine(f, onplane, normal)

        elif f.type == Fragment.FragmentType.FR_TRIANGLE:
            clipTriangle(v, _i, onplane, normal)


class ClipContainer(ObjectContainer):
    """container which clips children in a 3D box"""

    def __init__(self, minpt: NDArray[np.float64], maxpt: NDArray[np.float64]) -> None:
        super().__init__()
        self.minpt: NDArray[np.float64] = minpt
        self.maxpt: NDArray[np.float64] = maxpt

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        """get fragments for children (and range in vector)"""

        v: list[Fragment] = []

        for i in range(len(self.objects)):
            v.extend(self.objects[i].getFragments(persp_m, outer_m))

        # these are the points defining the clipping cube
        pts: list[NDArray[np.float64]] = [
            self.minpt,
            np.array([self.minpt[0], self.minpt[1], self.maxpt[2]]),
            np.array([self.minpt[0], self.maxpt[1], self.minpt[2]]),
            np.array([self.minpt[0], self.maxpt[1], self.maxpt[2]]),
            np.array([self.maxpt[0], self.minpt[1], self.minpt[2]]),
            np.array([self.maxpt[0], self.minpt[1], self.maxpt[2]]),
            np.array([self.maxpt[0], self.maxpt[1], self.minpt[2]]),
            self.maxpt,
        ]

        # convert cube coordinates to outer coordinates
        for i in range(8):
            pts[i] = vec4to3(np.dot(outer_m, vec3to4(pts[i])))

        # clip with plane point and normal
        # dotting points with plane with these will give all >= 0 if in cube
        clipFragments(v, 0, pts[0], np.cross(pts[2] - pts[0], pts[1] - pts[0]))
        clipFragments(v, 0, pts[0], np.cross(pts[1] - pts[0], pts[4] - pts[0]))
        clipFragments(v, 0, pts[0], np.cross(pts[4] - pts[0], pts[2] - pts[0]))
        clipFragments(v, 0, pts[7], np.cross(pts[5] - pts[7], pts[3] - pts[7]))
        clipFragments(v, 0, pts[7], np.cross(pts[3] - pts[7], pts[6] - pts[7]))
        clipFragments(v, 0, pts[7], np.cross(pts[6] - pts[7], pts[5] - pts[7]))

        return v

    def pointInBounds(self, pt: NDArray[np.float64]) -> bool:
        return all(
            (
                self.minpt[0] <= pt[0] <= self.maxpt[0],
                self.minpt[1] <= pt[1] <= self.maxpt[1],
                self.minpt[2] <= pt[2] <= self.maxpt[2],
            )
        )
