import enum

import numpy as np
from numpy.typing import NDArray

from .mmaths import calcProjVec
from .properties import LineProp, SurfaceProp
from ... import qtall as qt

LINE_DELTA_DEPTH: float = 1e-6


class FragmentPathParameters:
    def __init__(self) -> None:
        self.path: qt.QPainterPath = qt.QPainterPath()
        self.scaleline: bool = False
        self.scalepersp: bool = False
        self.runcallback: bool = False

    # optional callback function if runcallback is set
    def callback(
        self,
        painter: qt.QPainter,
        pt1: qt.QPointF,
        pt2: qt.QPointF,
        pt3: qt.QPointF,
        index: int,
        scale: float,
        linescale: float,
    ) -> None:
        pass


class Fragment:
    """created by drawing Objects to draw to screen"""

    class FragmentType(enum.Enum):
        FR_NONE = 0
        FR_TRIANGLE = 3
        FR_LINESEG = 2
        FR_PATH = 1

    def __init__(self):
        from .objects import Object

        # 3D points
        self.points: list[NDArray] = [np.zeros(3)] * 3

        # projected points associated with fragment
        self.proj: list[NDArray] = [np.zeros(3)] * 3

        # pointer to object, to avoid self-comparison.
        self.object: Object | None = None

        # optional pointer to a parameters object
        self.params: FragmentPathParameters = FragmentPathParameters()

        # drawing style
        self.surfaceprop: SurfaceProp | None = None
        self.lineprop: LineProp | None = None

        # for path
        self.pathsize: float = 0.0

        # calculated color from lighting
        self.calccolor: int = 0

        # number of times this has been split
        self.splitcount: int = 0

        # passed to path plotting or as index to color bar
        self.index: int = 0

        # type of fragment
        self.type: Fragment.FragmentType = self.FragmentType.FR_NONE

        # use calculated color
        self.usecalccolor: bool = False

    def nPointsVisible(self) -> int:
        """number of (visible) points used by fragment type"""
        return {
            Fragment.FragmentType.FR_TRIANGLE: 3,
            Fragment.FragmentType.FR_LINESEG: 2,
            Fragment.FragmentType.FR_PATH: 1,
            Fragment.FragmentType.FR_NONE: 0,
        }[self.type]

    # number of points used by fragment, including hidden ones
    # FR_PATH has an optional 2nd point for keeping track of a baseline
    def nPointsTotal(self) -> int:
        return {
            Fragment.FragmentType.FR_TRIANGLE: 3,
            Fragment.FragmentType.FR_LINESEG: 2,
            Fragment.FragmentType.FR_PATH: 3,
            Fragment.FragmentType.FR_NONE: 0,
        }[self.type]

    def minDepth(self) -> float:
        return {
            Fragment.FragmentType.FR_TRIANGLE: min(
                self.proj[0][2], self.proj[1][2], self.proj[2][2]
            ),
            Fragment.FragmentType.FR_LINESEG: min(self.proj[0][2], self.proj[1][2])
            - LINE_DELTA_DEPTH,
            Fragment.FragmentType.FR_PATH: self.proj[0][2] - 2 * LINE_DELTA_DEPTH,
            Fragment.FragmentType.FR_NONE: np.nan,
        }[self.type]

    def maxDepth(self) -> float:
        return {
            Fragment.FragmentType.FR_TRIANGLE: max(
                self.proj[0][2], max(self.proj[1][2], self.proj[2][2])
            ),
            Fragment.FragmentType.FR_LINESEG: max(self.proj[0][2], self.proj[1][2])
            - LINE_DELTA_DEPTH,
            Fragment.FragmentType.FR_PATH: self.proj[0][2] - 2 * LINE_DELTA_DEPTH,
            Fragment.FragmentType.FR_NONE: np.nan,
        }[self.type]

    def meanDepth(self) -> float:
        return {
            Fragment.FragmentType.FR_TRIANGLE: (
                self.proj[0][2] + self.proj[1][2] + self.proj[2][2]
            )
            / 3.0,
            Fragment.FragmentType.FR_LINESEG: (self.proj[0][2] + self.proj[1][2]) * 0.5
            - LINE_DELTA_DEPTH,
            Fragment.FragmentType.FR_PATH: self.proj[0][2] - 2 * LINE_DELTA_DEPTH,
            Fragment.FragmentType.FR_NONE: np.nan,
        }[self.type]

    def updateProjCoords(self, proj_m: NDArray):
        """recalculate projected coordinates"""
        for i in range(self.nPointsTotal()):
            self.proj[i] = calcProjVec(proj_m, self.points[i])

    # is fragment visible based on transparency?
    def isVisible(self) -> bool:
        vis: bool = False
        if (
            self.type == Fragment.FragmentType.FR_TRIANGLE
            or self.type == Fragment.FragmentType.FR_PATH
        ) and self.surfaceprop is not None:
            if self.surfaceprop.color(self.index).alpha() > 0:
                vis = True

        if (
            self.type == Fragment.FragmentType.FR_LINESEG
            or self.type == Fragment.FragmentType.FR_PATH
        ) and self.lineprop is not None:
            if self.lineprop.color(self.index).alpha() > 0:
                vis = True
        return vis
