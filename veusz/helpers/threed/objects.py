import abc
import copy
import enum
import sys

import numpy as np
from numpy.typing import NDArray

from .fragment import Fragment, FragmentPathParameters
from .mmaths import calcProjVec, vec3to2, vec3to4, vec4to3
from .properties import LineProp, SurfaceProp
from .twod import twodLineIntersectPolygon, twodPolyMakeClockwise
from ... import qtall as qt


class Object:
    def __init__(self) -> None:
        self.widget_id: int = 0  # id of widget which generated object

    def assignWidgetId(self, new_id: int) -> None:
        self.widget_id = new_id

    @abc.abstractmethod
    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]: ...


class Triangle(Object):
    # Triangle()
    # : Object(), surfaceprop(0)
    # {}
    def __init__(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        prop: SurfaceProp | None = None,
    ):
        super().__init__()
        self.points: list[NDArray] = [a, b, c]
        self.surfaceprop = copy.deepcopy(prop)

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        f: Fragment = Fragment()
        f.type = Fragment.FragmentType.FR_TRIANGLE
        f.surfaceprop = copy.deepcopy(self.surfaceprop)
        f.lineprop = None
        for i in range(3):
            f.points[i] = vec4to3(np.dot(outer_m, vec3to4(self.points[i])))
        f.object = self

        return [f]


class PolyLine(Object):
    def __init__(self, *args: NDArray[np.float64] | LineProp | None) -> None:
        super().__init__()
        self.lineprop: LineProp | None = args[-1]
        self.points: list[NDArray[np.float64]] = []
        if len(args) == 4:
            self.addPoints(args[0], args[1], args[2])
        elif len(args) != 1:
            raise TypeError("Unsupported signature")

    def addPoint(self, v: NDArray[np.float64]) -> None:
        self.points.append(v)

    def addPoints(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
    ) -> None:
        self.points.extend(np.array([x_i, y_i, z_i]) for x_i, y_i, z_i in zip(x, y, z))

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        assert persp_m.shape == (4, 4)
        assert outer_m.shape == (4, 4)

        f: Fragment = Fragment()
        f.type = Fragment.FragmentType.FR_LINESEG
        f.surfaceprop = None
        f.lineprop = self.lineprop
        f.object = self

        v: list[Fragment] = []

        # iterators use many more instructions here...
        for i in range(len(self.points)):
            f.points[1] = f.points[0]
            f.points[0] = vec4to3(np.dot(outer_m, vec3to4(self.points[i])))
            f.index = i

            if i > 0 and np.isfinite(np.sum(f.points[0] + f.points[1])):
                v.append(f)

        return v


class LineSegments(Object):
    def __init__(self, *args: NDArray | LineProp) -> None:
        super().__init__()
        self.lineprop: LineProp = args[-1]
        args = args[:-1]
        self.points: list[NDArray] = []
        size: int = min(x.size for x in args)
        if len(args) == 2:
            pts1: NDArray
            pts2: NDArray
            pts1, pts2 = args
            for i in range(0, size, 3):
                self.points.append(np.array([pts1[i], pts1[i + 1], pts1[i + 2]]))
                self.points.append(np.array([pts2[i], pts2[i + 1], pts2[i + 2]]))
        elif len(args) == 6:
            x1: NDArray
            y1: NDArray
            z1: NDArray
            x2: NDArray
            y2: NDArray
            z2: NDArray
            x1, y1, z1, x2, y2, z2 = args
            for i in range(size):
                self.points.append(np.array([x1[i], y1[i], z1[i]]))
                self.points.append(np.array([x2[i], y2[i], z2[i]]))

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        f: Fragment = Fragment()
        f.type = Fragment.FragmentType.FR_LINESEG
        f.surfaceprop = None
        f.lineprop = copy.deepcopy(self.lineprop)
        f.object = self

        v: list[Fragment] = []

        s: int = len(self.points)
        for i in range(0, s, 2):
            f.points[0] = vec4to3(np.dot(outer_m, vec3to4(self.points[i])))
            f.points[1] = vec4to3(np.dot(outer_m, vec3to4(self.points[i + 1])))
            f.index = i
            v.append(f)

        return v


class Mesh(Object):
    """
    a grid of height values on a regular mesh with grid points given
    heights has M*np elements where M and np are the length of pos1 & pos2
    """

    # X_DIRN: heights is X, `a` is Y, `b` is Z
    # Y_DIRN: heights is Y, `a` is Z. `b` is X
    # Z_DIRN: heights is Z, `a` is X, `b` is Y
    class Direction(enum.Enum):
        X_DIRN = 0
        Y_DIRN = 1
        Z_DIRN = 2

    X_DIRN = Direction.X_DIRN
    Y_DIRN = Direction.Y_DIRN
    Z_DIRN = Direction.Z_DIRN

    def __init__(
        self,
        pos1: NDArray[np.float64],
        pos2: NDArray[np.float64],
        heights: NDArray[np.float64],
        dirn: Direction,
        lprop: LineProp | None = None,
        sprop: SurfaceProp | None = None,
        hidehorzline: bool = False,
        hidevertline: bool = False,
    ) -> None:
        super().__init__()
        self.pos1: NDArray[np.float64] = pos1
        self.pos2: NDArray[np.float64] = pos2
        self.heights: NDArray[np.float64] = heights
        self.dirn: Mesh.Direction = dirn
        self.lineprop: LineProp | None = lprop
        self.surfaceprop: SurfaceProp | None = sprop
        self.hidehorzline: bool = hidehorzline
        self.hidevertline: bool = hidevertline

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        return self.getLineFragments(outer_m) + self.getSurfaceFragments(outer_m)

    def getSurfaceFragments(self, outer_m: NDArray[np.float64]) -> list[Fragment]:
        assert outer_m.shape == (4, 4)

        if self.surfaceprop is None:
            return []

        vidx_h, vidx_1, vidx_2 = self.getVecIdxs()

        fs: Fragment = Fragment()
        fs.type = Fragment.FragmentType.FR_TRIANGLE
        fs.surfaceprop = self.surfaceprop
        fs.lineprop = None
        fs.object = self

        # for each grid point we alternatively draw one of two sets of
        # triangles, to make a symmetric diamond pattern, which looks
        # better when striped
        tidxs: list[list[list[int]]] = [[[0, 1, 2], [3, 1, 2]], [[1, 0, 3], [2, 0, 3]]]

        n1: int = self.pos1.size
        n2: int = self.pos2.size

        v: list[Fragment] = []

        p: list[NDArray[np.float64]] = [np.zeros(4)] * 4
        pproj: list[NDArray[np.float64]] = [np.zeros(3)] * 4
        p[0][3] = p[1][3] = p[2][3] = p[3][3] = 1
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                # update coordinates of corners of square and project
                for i in range(4):
                    j1: int = i1 + i % 2
                    j2: int = i2 + i // 2
                    p[i][vidx_h] = self.heights[j1 * n2 + j2]
                    p[i][vidx_1] = self.pos1[j1]
                    p[i][vidx_2] = self.pos2[j2]
                    pproj[i] = vec4to3(np.dot(outer_m, p[i]))

                # add two triangles, using indices of corners
                for tri in range(2):
                    indexes: list[int] = tidxs[(i1 + i2) % 2][tri]
                    if np.isfinite(
                        np.sum(p[indexes[0]] + p[indexes[1]] + p[indexes[2]])
                    ):
                        for i in range(3):
                            fs.points[i] = pproj[indexes[i]]
                        v.append(fs)

                fs.index += 1

        return v

    def getLineFragments(self, outer_m: NDArray[np.float64]) -> list[Fragment]:
        assert outer_m.shape == (4, 4)

        if self.lineprop is None:
            return []

        vidx_h, vidx_1, vidx_2 = self.getVecIdxs()

        fl: Fragment = Fragment()
        fl.type = Fragment.FragmentType.FR_LINESEG
        fl.surfaceprop = None
        fl.lineprop = self.lineprop
        fl.object = self

        n2: int = self.pos2.size
        pt: NDArray[np.float64] = np.array([0, 0, 0, 1])

        v: list[Fragment] = []

        for step_index in (0, 1):
            if self.hidehorzline and step_index == 0:
                continue
            if self.hidevertline and step_index == 1:
                continue

            vec_step: NDArray[np.float64] = self.pos1 if step_index == 0 else self.pos2
            vec_const: NDArray[np.float64] = self.pos2 if step_index == 0 else self.pos1
            vidx_step: int = vidx_1 if step_index == 0 else vidx_2
            vidx_const: int = vidx_2 if step_index == 0 else vidx_1

            for const_i in range(vec_const.size):
                pt[vidx_const] = vec_const[const_i]
                for step_i in range(vec_step.size):
                    heights_val: float = self.heights[
                        (
                            step_i * n2 + const_i
                            if step_index == 0
                            else const_i * n2 + step_i
                        )
                    ]
                    pt[vidx_step] = vec_step[step_i]
                    pt[vidx_h] = heights_val

                    # shuffle new to old positions and calculate new new
                    fl.points[1] = fl.points[0]
                    fl.points[0] = vec4to3(np.dot(outer_m, pt))

                    if step_i > 0 and np.isfinite(np.sum(fl.points[0] + fl.points[1])):
                        v.append(fl)
                    fl.index += 1

        return v

    def getVecIdxs(self) -> tuple[int, int, int]:
        """get indices into vector for coordinates in height, pos1 and pos2 directions"""
        if self.dirn == Mesh.Direction.Y_DIRN:
            return 1, 2, 0
        elif self.dirn == Mesh.Direction.Z_DIRN:
            return 2, 0, 1
        elif self.dirn == Mesh.Direction.X_DIRN:
            return 0, 1, 2
        else:
            raise ValueError(f"Invalid direction: {self.dirn}")


class DataMesh(Object):
    """
    Grid of data values, where the centres of the bins are specified.
    There should be 1 more values along edges than values in array.
    idxval, edge1, edge2 give the index of the axis (x=0,y=1,z=2) for
    that direction.
    """

    def __init__(
        self,
        edges1: NDArray[np.float64],
        edges2: NDArray[np.float64],
        vals: NDArray[np.float64],
        idxval: int,
        idxedge1: int,
        idxedge2: int,
        highres: bool,
        lprop: LineProp | None = None,
        sprop: SurfaceProp | None = None,
        hidehorzline: bool = False,
        hidevertline: bool = False,
    ):
        super().__init__()
        self.edges1: NDArray[np.float64] = edges1
        self.edges2: NDArray[np.float64] = edges2
        self.vals: NDArray[np.float64] = vals
        self.idxval: int = idxval
        self.idxedge1: int = idxedge1
        self.idxedge2: int = idxedge2
        self.highres: bool = highres
        self.lineprop: LineProp | None = lprop
        self.surfaceprop: SurfaceProp | None = sprop
        self.hidehorzline: bool = hidehorzline
        self.hidevertline: bool = hidevertline

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        assert persp_m.shape == (4, 4)
        assert outer_m.shape == (4, 4)

        # check indices
        found: list[bool] = [False] * 3
        idxs: list[int] = [self.idxval, self.idxedge1, self.idxedge2]
        for i in range(3):
            if idxs[i] <= 2:
                found[idxs[i]] = True
        if not found[0] or not found[1] or not found[2]:
            print("DataMesh: invalid indices", file=sys.stderr)
            return []

        # check that data sizes agree
        if (self.edges1.size - 1) * (self.edges2.size - 1) != self.vals.size:
            print("DataMesh: invalid size", file=sys.stderr)
            return []

        # nothing to draw
        if self.lineprop is None and self.surfaceprop is None:
            return []

        ft: Fragment

        # used to draw the grid and surface
        ft = Fragment()
        ft.type = Fragment.FragmentType.FR_TRIANGLE
        ft.surfaceprop = self.surfaceprop
        ft.lineprop = None
        ft.object = self

        fl = Fragment()
        fl.type = Fragment.FragmentType.FR_LINESEG
        fl.surfaceprop = None
        fl.lineprop = self.lineprop
        fl.object = self

        # these are the corner indices used for drawing low and high resolution surfaces
        trilist_highres: list[list[int]] = [
            [8, 0, 1],
            [8, 1, 2],
            [8, 2, 3],
            [8, 3, 4],
            [8, 4, 5],
            [8, 5, 6],
            [8, 6, 7],
            [8, 7, 0],
        ]
        # there are two low resolution triangle lists, as we want to
        # alternate them in each grid point to make a symmetric pattern
        trilist_lowres1: list[list[int]] = [[0, 2, 4], [0, 6, 4]]
        trilist_lowres2: list[list[int]] = [[2, 0, 6], [2, 4, 6]]
        linelist_lowres: list[list[int]] = [[0, 2], [0, 6], [4, 2], [4, 6]]
        linelist_highres: list[list[int]] = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 0],
        ]

        # This is to avoid double-drawing lines. Lines are given an x/yindex to say which
        # side of the grid cell is being drawn and a lineidx which is unique for sub-lines
        # xidx, yidx, lineidx
        linecell_lowres: list[list[int]] = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1]]
        linecell_highres: list[list[int]] = [
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 2],
            [1, 0, 3],
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 3],
            [0, 0, 2],
        ]
        # whether lines are vertical or horizontal
        dirn_lowres: list[int] = [0, 1, 0, 1]
        dirn_highres: list[int] = [1, 1, 0, 0, 1, 1, 0, 0]

        # select list above depending on high or low resolution
        lines: list[list[int]] = linelist_highres if self.highres else linelist_lowres
        linecells: list[list[int]] = (
            linecell_highres if self.highres else linecell_lowres
        )
        linedirn: list[int] = dirn_highres if self.highres else dirn_lowres

        # store corners and neighbouring cell values
        neigh: list[float] = [0.0] * 9
        corners: list[NDArray[np.float64]] = [np.array([0, 0, 0, 1])] * 9  # 4d corners
        corners3: list[NDArray[np.float64]]  # 3d version of above

        # don't draw lines twice by keeping track which edges of which
        # cells have been drawn already
        linetracker: LineCellTracker = LineCellTracker(
            self.edges1.size, self.edges2.size
        )

        n1: int = self.edges1.size - 1
        n2: int = self.edges2.size - 1

        v: list[Fragment] = []

        # loop over 2d array
        for i1 in range(n1):
            for i2 in range(n2):
                # skip bad data values
                if not np.isfinite(self.vals[i1 * n2 + i2]):
                    continue

                # get values of neighbouring cells (clipping at edges)
                # -1,-1 -1,0 -1,1   0,-1 0,0 0,1   1,-1 1,0 1,1
                for d1 in (-1, 0, 1):
                    for d2 in (-1, 0, 1):
                        clip1: int = max(min(i1 + d1, n1 - 1), 0)
                        clip2: int = max(min(i2 + d2, n2 - 1), 0)
                        val: float = self.vals[clip1 * n2 + clip2]
                        neigh[(d1 + 1) * 3 + (d2 + 1)] = val

                # compute “corners” - these are the clockwise corners and
                # edge centres from the top left (d1==d2==-1), followed by
                # the cell centre
                corners[0][idxs[0]] = np.nanmean(
                    [neigh[0], neigh[3], neigh[4], neigh[1]]
                )
                corners[0][idxs[1]] = self.edges1[i1]
                corners[0][idxs[2]] = self.edges2[i2]

                corners[1][idxs[0]] = np.nanmean([neigh[4], neigh[3]])
                corners[1][idxs[1]] = 0.5 * (self.edges1[i1] + self.edges1[i1 + 1])
                corners[1][idxs[2]] = self.edges2[i2]

                corners[2][idxs[0]] = np.nanmean(
                    [neigh[3], neigh[6], neigh[7], neigh[4]]
                )
                corners[2][idxs[1]] = self.edges1[i1 + 1]
                corners[2][idxs[2]] = self.edges2[i2]

                corners[3][idxs[0]] = np.nanmean([neigh[4], neigh[7]])
                corners[3][idxs[1]] = self.edges1[i1 + 1]
                corners[3][idxs[2]] = 0.5 * (self.edges2[i2] + self.edges2[i2 + 1])

                corners[4][idxs[0]] = np.nanmean(
                    [neigh[4], neigh[7], neigh[8], neigh[5]]
                )
                corners[4][idxs[1]] = self.edges1[i1 + 1]
                corners[4][idxs[2]] = self.edges2[i2 + 1]

                corners[5][idxs[0]] = np.nanmean([neigh[4], neigh[5]])
                corners[5][idxs[1]] = 0.5 * (self.edges1[i1] + self.edges1[i1 + 1])
                corners[5][idxs[2]] = self.edges2[i2 + 1]

                corners[6][idxs[0]] = np.nanmean(
                    [neigh[1], neigh[4], neigh[5], neigh[2]]
                )
                corners[6][idxs[1]] = self.edges1[i1]
                corners[6][idxs[2]] = self.edges2[i2 + 1]

                corners[7][idxs[0]] = np.nanmean([neigh[4], neigh[1]])
                corners[7][idxs[1]] = self.edges1[i1]
                corners[7][idxs[2]] = 0.5 * (self.edges2[i2] + self.edges2[i2 + 1])

                corners[8][idxs[0]] = neigh[4]
                corners[8][idxs[1]] = 0.5 * (self.edges1[i1] + self.edges1[i1 + 1])
                corners[8][idxs[2]] = 0.5 * (self.edges2[i2] + self.edges2[i2 + 1])

                # convert to 3d coordinates
                corners3 = [vec4to3(np.dot(outer_m, corner)) for corner in corners]

                # draw triangles
                if ft.surfaceprop is not None:
                    # alternate triangle list to make a symmetric pattern for lowres
                    tris: list[list[int]] = (
                        trilist_highres
                        if self.highres
                        else (
                            trilist_lowres1 if (i1 + i2) % 2 == 0 else trilist_lowres2
                        )
                    )

                    ft.index = i1 * n2 + i2
                    for tris_i in tris:
                        ft.points[0] = corners3[tris_i[0]]
                        ft.points[1] = corners3[tris_i[1]]
                        ft.points[2] = corners3[tris_i[2]]
                        v.append(ft)

                # draw lines (if they haven't been drawn before)
                if fl.lineprop is not None:
                    fl.index = i1 * n2 + i2
                    for i in range(len(lines)):
                        # skip lines which are in wrong direction
                        if (self.hidehorzline and linedirn[i] == 0) or (
                            self.hidevertline and linedirn[i] == 1
                        ):
                            continue

                        if not linetracker.isLineSet(
                            i1 + linecells[i][0], i2 + linecells[i][1], linecells[i][2]
                        ):
                            fl.points[0] = corners3[lines[i][0]]
                            fl.points[1] = corners3[lines[i][1]]
                            if (
                                np.isfinite(fl.points[0]).all()
                                and np.isfinite(fl.points[1]).all()
                            ):
                                v.append(fl)
                            linetracker.setLine(
                                i1 + linecells[i][0],
                                i2 + linecells[i][1],
                                linecells[i][2],
                            )

        return v


class MultiCuboid(Object):
    """multiple cuboids"""

    def __init__(
        self,
        xmin: NDArray[np.float64],
        xmax: NDArray[np.float64],
        ymin: NDArray[np.float64],
        ymax: NDArray[np.float64],
        zmin: NDArray[np.float64],
        zmax: NDArray[np.float64],
        lprop: LineProp | None = None,
        sprop: SurfaceProp | None = None,
    ) -> None:
        super().__init__()
        self.xmin: NDArray[np.float64] = xmin
        self.xmax: NDArray[np.float64] = xmax
        self.ymin: NDArray[np.float64] = ymin
        self.ymax: NDArray[np.float64] = ymax
        self.zmin: NDArray[np.float64] = zmin
        self.zmax: NDArray[np.float64] = zmax
        self.lineprop: LineProp | None = lprop
        self.surfaceprop: SurfaceProp | None = sprop

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        assert persp_m.shape == (4, 4)
        assert outer_m.shape == (4, 4)

        # nothing to draw
        if self.lineprop is None and self.surfaceprop is None:
            return []

        ft: Fragment
        # used to draw the grid and surface
        ft = Fragment()
        ft.type = Fragment.FragmentType.FR_TRIANGLE
        ft.surfaceprop = self.surfaceprop
        ft.lineprop = None
        ft.object = self

        fl = Fragment()
        fl.type = Fragment.FragmentType.FR_LINESEG
        fl.surfaceprop = None
        fl.lineprop = self.lineprop
        fl.object = self

        # triangles for drawing surface of cube
        triidx: list[list[list[int]]] = [
            [[0, 0, 0], [0, 0, 1], [1, 0, 0]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 1], [0, 0, 1]],
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [0, 0, 0], [1, 0, 0]],
            [[0, 1, 1], [1, 0, 1], [0, 0, 1]],
            [[0, 1, 1], [1, 1, 1], [1, 0, 1]],
            [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[1, 0, 1], [1, 0, 0], [0, 0, 1]],
            [[1, 0, 1], [1, 1, 0], [1, 0, 0]],
            [[1, 0, 1], [1, 1, 1], [1, 1, 0]],
            [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
        ]

        # lines for drawing edges of cube
        edgeidx: list[list[list[int]]] = [
            [[0, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [1, 0, 1]],
            [[0, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [1, 1, 0]],
            [[0, 1, 1], [1, 1, 1]],
            [[1, 0, 0], [1, 0, 1]],
            [[1, 0, 0], [1, 1, 0]],
            [[1, 0, 1], [1, 1, 1]],
            [[1, 1, 0], [1, 1, 1]],
        ]

        # maximum size of array
        sizex: int = min(self.xmin.size, self.xmax.size)
        sizey: int = min(self.ymin.size, self.ymax.size)
        sizez: int = min(self.zmin.size, self.zmax.size)
        size: int = min(sizex, sizey, sizez)

        v: list[Fragment] = []

        for i in range(size):
            x: list[float] = [self.xmin[i], self.xmax[i]]
            y: list[float] = [self.ymin[i], self.ymax[i]]
            z: list[float] = [self.zmin[i], self.zmax[i]]

            if ft.surfaceprop is not None and not ft.surfaceprop.hide:
                ft.index = i

                # iterate over triangles in cube
                for tri in range(12):
                    # points for triangle
                    for pt in range(3):
                        ft.points[pt] = vec4to3(
                            np.dot(
                                outer_m,
                                np.array(
                                    [
                                        x[triidx[tri][pt][0]],
                                        y[triidx[tri][pt][1]],
                                        z[triidx[tri][pt][2]],
                                        1.0,
                                    ]
                                ),
                            )
                        )
                    if ft.isVisible():
                        v.append(ft)

            if fl.lineprop is not None and not fl.lineprop.hide:
                fl.index = i

                # iterate over edges
                for edge in range(12):
                    # points for line
                    for pt in range(2):
                        fl.points[pt] = vec4to3(
                            np.dot(
                                outer_m,
                                np.array(
                                    [
                                        x[edgeidx[edge][pt][0]],
                                        y[edgeidx[edge][pt][1]],
                                        z[edgeidx[edge][pt][2]],
                                        1.0,
                                    ]
                                ),
                            )
                        )
                    if fl.isVisible():
                        v.append(fl)

        return v


class Points(Object):
    """a set of points to plot"""

    def __init__(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
        pp: qt.QPainterPath,
        pointedge: LineProp | None = None,
        pointfill: SurfaceProp | None = None,
    ):
        super().__init__()
        self.x: NDArray[np.float64] = x
        self.y: NDArray[np.float64] = y
        self.z: NDArray[np.float64] = z
        self.sizes: NDArray[np.float64] = np.zeros(x.shape)
        self.path: qt.QPainterPath = pp
        self.scaleline: bool = True
        self.scalepersp: bool = True
        self.lineedge: LineProp | None = pointedge
        self.surfacefill: SurfaceProp | None = pointfill
        self.fragparams: FragmentPathParameters = FragmentPathParameters()

    def setSizes(self, sizes: NDArray[np.float64]) -> None:
        self.sizes = sizes

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        assert persp_m.shape == (4, 4)
        assert outer_m.shape == (4, 4)

        self.fragparams.path = self.path
        self.fragparams.scaleline = self.scaleline
        self.fragparams.scalepersp = self.scalepersp
        self.fragparams.runcallback = False

        fp: Fragment = Fragment()
        fp.type = Fragment.FragmentType.FR_PATH
        fp.object = self
        fp.params = self.fragparams
        fp.surfaceprop = self.surfacefill
        fp.lineprop = self.lineedge
        fp.pathsize = 1

        size: int = min(self.x.size, self.y.size, self.z.size)
        if self.sizes.size:
            size = min(size, self.sizes.size)

        v: list[Fragment] = []

        for i in range(size):
            fp.points[0] = vec4to3(
                np.dot(outer_m, np.array([self.x[i], self.y[i], self.z[i], 1]))
            )
            if self.sizes.size:
                fp.pathsize = self.sizes[i]
            fp.index = i

            if not np.isinf(fp.points[0]).any():
                v.append(fp)

        return v


class Text(Object):
    """a "text" class which calls back draw() when drawing is requested"""

    class TextPathParameters(FragmentPathParameters):
        def __init__(self):
            super().__init__()
            self.text: Text | None = None

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
            if self.text is not None:
                self.text.draw(painter, pt1, pt2, pt3, index, scale, linescale)

    # pos1 and pos2 contain a list of x,y,z values
    def __init__(self, pos1: NDArray[np.float64], pos2: NDArray[np.float64]) -> None:
        super().__init__()
        self.pos1: NDArray[np.float64] = pos1
        self.pos2: NDArray[np.float64] = pos2
        self.fragparams: Text.TextPathParameters = Text.TextPathParameters()
        self.fragparams.text = self
        self.fragparams.path = qt.QPainterPath()
        self.fragparams.scaleline = False
        self.fragparams.scalepersp = False
        self.fragparams.runcallback = True

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        assert persp_m.shape == (4, 4)
        assert outer_m.shape == (4, 4)

        fp: Fragment = Fragment()
        fp.type = Fragment.FragmentType.FR_PATH
        fp.object = self
        fp.params = self.fragparams
        fp.surfaceprop = None
        fp.lineprop = None
        fp.pathsize = 1

        v: list[Fragment] = []
        numitems: int = min(self.pos1.size, self.pos2.size) // 3
        for i in range(numitems):
            base: int = i * 3
            pt1 = np.array([self.pos1[base], self.pos1[base + 1], self.pos1[base + 2]])
            fp.points[0] = vec4to3(np.dot(outer_m, pt1))
            pt2 = np.array([self.pos2[base], self.pos2[base + 1], self.pos2[base + 2]])
            fp.points[1] = vec4to3(np.dot(outer_m, pt2))
            fp.index = i
            v.append(fp)

        return v

    def draw(
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


"""

private:
  class TextPathParameters : public FragmentPathParameters
  {
  public:
    void callback(QPainter* painter, QPointF pt1, QPointF pt2, QPointF pt3,
                  int index,  double scale, double linescale);
    Text* text;
  };

  TextPathParameters fragparams;

public:
  ValVector pos1, pos2;
};
"""


class TriangleFacing(Triangle):
    """A triangle only visible if its norm (translated to viewing space) is +ve"""

    def __init__(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        prop: SurfaceProp | None = None,
    ) -> None:
        super().__init__(a, b, c, prop)

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        assert persp_m.shape == (4, 4)
        assert outer_m.shape == (4, 4)

        torigin: NDArray[np.float64] = vec4to3(np.dot(outer_m, np.array([0, 0, 0, 1])))
        norm: NDArray[np.float64] = np.cross(
            self.points[1] - self.points[0],
            self.points[2] - self.points[0],
        )
        tnorm: NDArray[np.float64] = vec4to3(np.dot(outer_m, vec3to4(norm)))

        v: list[Fragment] = []

        # norm points towards +z
        if tnorm[2] > torigin[2]:
            v = super().getFragments(persp_m, outer_m)

        return v


class ObjectContainer(Object):
    """
    container of objects with transformation matrix of children
    Note: object pointers passed to object will be deleted when this
    container is deleted
    """

    def __init__(self) -> None:
        super().__init__()
        self.objM: NDArray[np.float64] = np.eye(4)
        self.objects: list[Object] = []

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        assert persp_m.shape == (4, 4)
        assert outer_m.shape == (4, 4)

        tot_m: NDArray[np.float64] = np.dot(outer_m, self.objM)
        assert tot_m.shape == (4, 4)
        return sum((o.getFragments(persp_m, tot_m) for o in self.objects), start=[])

    def addObject(self, obj: Object) -> None:
        self.objects.append(obj)

    # recursive set id of child objects
    def assignWidgetId(self, new_id: int) -> None:
        for o in self.objects:
            o.assignWidgetId(new_id)


class FacingContainer(ObjectContainer):
    """
    container which only draws contents if the norm is pointing in +ve
    z direction
    """

    def __init__(self, norm: NDArray[np.float64]) -> None:
        super().__init__()
        self.norm: NDArray[np.float64] = norm

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        assert persp_m.shape == (4, 4)
        assert outer_m.shape == (4, 4)

        v: list[Fragment] = []
        origin: NDArray[np.float64] = vec4to3(np.dot(outer_m, np.array([0, 0, 0, 1])))
        assert origin.shape == (3,)
        tnorm: NDArray[np.float64] = vec4to3(np.dot(outer_m, vec3to4(self.norm)))
        assert tnorm.shape == (3,)

        # norm points towards +z
        if tnorm[2] > origin[2]:
            v = super().getFragments(persp_m, outer_m)

        return v


class AxisLabels(Object):
    """This class draws tick labels with correct choice of axis"""

    class PathParameters(FragmentPathParameters):
        def __init__(self) -> None:
            super().__init__()
            self.tl: AxisLabels | None = None
            self.axangle: float = 0.0

        def callback(
            self,
            painter: qt.QPainter,
            pt: qt.QPointF,
            ax1: qt.QPointF,
            ax2: qt.QPointF,
            index: int,
            scale: float,
            linescale: float,
        ) -> None:
            painter.save()
            self.tl.drawLabel(painter, index, pt, ax1, ax2, self.axangle)
            painter.restore()

    # cube defined to be between these corners
    def __init__(
        self,
        box1: NDArray[np.float64],
        box2: NDArray[np.float64],
        tickfracs: NDArray[np.float64],
        labelfrac: float,
    ) -> None:
        super().__init__()
        self.box1: NDArray[np.float64] = box1
        self.box2: NDArray[np.float64] = box2
        self.tickfracs: NDArray = tickfracs
        self.labelfrac: float = labelfrac
        self.starts: list[NDArray[np.float64]] = []
        self.ends: list[NDArray[np.float64]] = []
        self.fragparams: AxisLabels.PathParameters = AxisLabels.PathParameters()

    def addAxisChoice(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
    ) -> None:
        self.starts.append(start)
        self.ends.append(end)

    # override this: draw requested label at origin, with alignment
    # given
    # (if index==-1, then draw axis label)
    def drawLabel(
        self,
        painter: qt.QPainter,
        index: int,
        pt: qt.QPointF,
        ax1: qt.QPointF,
        ax2: qt.QPointF,
        axangle: float,
    ) -> None:
        pass

    def getFragments(
        self,
        persp_m: NDArray[np.float64],
        outer_m: NDArray[np.float64],
    ) -> list[Fragment]:
        """
        algorithm:

        Take possible axis positions
        Find those which do not overlap on the screen with body of cube
         - make cube faces
         - look for endpoints which are somewhere on a face (not edge)
        Prefer those axes to bottom left
        Determine from faces, which side of the axis is inside and which outside
        Setup drawLabel for the right axis
        """
        num_entries: int = min(len(self.starts), len(self.ends))
        if num_entries == 0:
            return []

        box_points: tuple[NDArray[np.float64], NDArray[np.float64]] = (
            self.box1,
            self.box2,
        )

        # compute corners of cube in projected coordinates
        # (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
        proj_corners = [np.zeros(3)] * 8
        for i0 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    pt = np.array(
                        [
                            box_points[i0][0],
                            box_points[i1][1],
                            box_points[i2][2],
                        ]
                    )
                    proj_corners[i2 + i1 * 2 + i0 * 4] = calcProjVec(
                        persp_m,
                        np.dot(outer_m, vec3to4(pt)),
                    )

        # point indices for faces of cube
        faces: NDArray[np.float64] = np.array(
            [
                [0, 1, 3, 2],  # x==0
                [4, 5, 7, 6],  # x==1
                [0, 1, 5, 4],  # y==0
                [2, 3, 7, 6],  # y==1
                [0, 4, 6, 2],  # z==0
                [1, 5, 7, 3],  # z==1
            ]
        )

        # scene and projected coordinates of axis ends
        proj_starts: list[NDArray[np.float64]] = []
        proj_ends: list[NDArray[np.float64]] = []
        for axis in range(num_entries):
            # ends shifted slightly inwards for overlap checks
            # (fixes issues in ends exactly overlapping with
            #  face edges causing overlap to fail)
            delta: NDArray[np.float64] = self.ends[axis] - self.starts[axis]
            start_in: NDArray[np.float64] = vec3to4(self.starts[axis] + delta * 0.001)
            end_in: NDArray[np.float64] = vec3to4(self.starts[axis] + delta * 0.999)

            proj_starts.append(calcProjVec(persp_m, np.dot(outer_m, start_in)))
            proj_ends.append(calcProjVec(persp_m, np.dot(outer_m, end_in)))

        # find axes which don't overlap with faces in 2D
        axchoices: list[int] = []

        facepts: list[NDArray[np.float64]]
        for axis in range(num_entries):
            linept1: list[NDArray[np.float64]] = vec3to2(proj_starts[axis])
            linept2: list[NDArray[np.float64]] = vec3to2(proj_ends[axis])

            overlap: bool = False

            # does this overlap with any faces?
            for face in range(6):
                facepts = [vec3to2(proj_corners[faces[face][i]]) for i in range(4)]
                twodPolyMakeClockwise(facepts)

                if twodLineIntersectPolygon(linept1, linept2, facepts):
                    overlap = True
                    break

            if not overlap:
                axchoices.append(axis)

        # if none are suitable, prefer all
        if not axchoices:
            for axis in range(num_entries):
                axchoices.append(axis)

        # get projected cube centre
        proj_cent: NDArray[np.float64] = calcProjVec(
            persp_m,
            np.dot(outer_m, vec3to4((self.box1 + self.box2) * 0.5)),
        )

        # currently-prefered axis number
        bestaxis: int = 0
        # axes are scored to prefer front, bottom, left axes
        bestscore: int = -1

        for choice in axchoices:
            av: NDArray[np.float64] = (proj_starts[choice] + proj_ends[choice]) * 0.5

            # score is weighted towards front, then bottom, then left
            score: int = (
                10 * (av[0] <= proj_cent[0])
                + 11 * (av[1] > proj_cent[1])
                + 12 * (av[2] < proj_cent[2])
            )
            if score > bestscore:
                bestscore = score
                bestaxis = choice

        # initialise PathParameters with best axis
        self.fragparams.tl = self
        self.fragparams.path = qt.QPainterPath()
        self.fragparams.scaleline = False
        self.fragparams.scalepersp = False
        self.fragparams.runcallback = True

        self.fragparams.axangle = np.degrees(
            np.atan2(
                (proj_starts[bestaxis][1] + proj_ends[bestaxis][1]) * 0.5
                - proj_cent[1],
                (proj_starts[bestaxis][0] + proj_ends[bestaxis][0]) * 0.5
                - proj_cent[0],
            )
        )

        axstart: NDArray[np.float64] = self.starts[bestaxis]
        delta: NDArray[np.float64] = self.ends[bestaxis] - axstart

        # scene coordinates of axis ends
        axstart_scene: NDArray[np.float64] = vec4to3(np.dot(outer_m, vec3to4(axstart)))
        axend_scene: NDArray[np.float64] = vec4to3(
            np.dot(outer_m, vec3to4(self.ends[bestaxis]))
        )

        # ok, now we add the number fragments for the best choice of axis
        fp: Fragment = Fragment()
        fp.type = Fragment.FragmentType.FR_PATH
        fp.object = self
        fp.params = self.fragparams
        fp.surfaceprop = None
        fp.lineprop = None
        fp.pathsize = 1

        fp.points[1] = axstart_scene
        fp.points[2] = axend_scene

        v: list[Fragment] = []

        # add tick labels
        for i in range(self.tickfracs.size):
            fp.index = i
            pt: NDArray[np.float64] = axstart + delta * (self.tickfracs[i])

            fp.points[0] = vec4to3(np.dot(outer_m, vec3to4(pt)))
            v.append(fp)

        # add main axis label
        if self.labelfrac >= 0:
            fp.index = -1
            pt: NDArray[np.float64] = axstart + delta * self.labelfrac

            fp.points[0] = vec4to3(np.dot(outer_m, vec3to4(pt)))
            v.append(fp)

        return v


class LineCellTracker:
    """
    keep track of which lines are drawn in the grid, so they aren't
    drawn again. We have a grid point for each edge, and a line
    index (0-3)
    """

    MAXLINEIDX: int = 4

    def __init__(self, n1: int, n2: int) -> None:
        self.n1: int = n1
        self.n2: int = n2
        self.data: list[bool] = [False] * (n1 * n2 * LineCellTracker.MAXLINEIDX)

    def setLine(self, i1: int, i2: int, lineidx: int) -> None:
        self.data[(i1 * self.n2 + i2) * LineCellTracker.MAXLINEIDX + lineidx] = True

    def isLineSet(self, i1: int, i2: int, lineidx: int) -> bool:
        return self.data[(i1 * self.n2 + i2) * LineCellTracker.MAXLINEIDX + lineidx]
