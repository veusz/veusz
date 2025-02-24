import abc
import enum

import numpy as np
from numpy.typing import NDArray

from .bsp import BSPBuilder
from .camera import Camera
from .fragment import Fragment, LINE_DELTA_DEPTH
from .mmaths import calcProjVec, projVecToScreen, scaleM3, translateM3, vec4to3
from .objects import Object
from .properties import LineProp, SurfaceProp
from .. import threed
from ... import qtall as qt


def breakLongLines(fragments: list[Fragment], maxlen: float):
    maxlen2: float = maxlen**2
    size = len(fragments)
    for ifrag in range(size):
        f = fragments[ifrag]
        if f.type == Fragment.FragmentType.FR_LINESEG:
            len2: float = float(np.sum(np.square(f.points[1] - f.points[0])))
            if len2 > maxlen2:
                nbits: int = int(np.sqrt(len2 / maxlen2)) + 1
                delta: NDArray = (f.points[1] - f.points[0]) * (1.0 / nbits)

                # set original to be first segment
                f.points[1] = f.points[0] + delta

                # add nbits-1 copies for next segments
                temp_f: Fragment = f
                for ic in range(1, nbits):
                    temp_f.points[0] = temp_f.points[1]
                    temp_f.points[1] += delta
                    fragments.append(temp_f)


def makeScreenM(
    frags: list[Fragment],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> NDArray:
    # get range of projected points in x and y
    minx: float = np.nan
    miny: float = np.nan
    maxx: float = -np.nan
    maxy: float = -np.nan

    for f in frags:
        for p in range(f.nPointsVisible()):
            x: float = f.proj[p][0]
            y: float = f.proj[p][1]
            if np.isfinite(x) and np.isfinite(y):
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)

    # catch bad values or empty arrays
    if maxx == minx or np.isinf(minx) or np.isinf(maxx):
        maxx = 1
        minx = 0
    if maxy == miny or np.isinf(miny) or np.isinf(maxy):
        maxy = 1
        miny = 0

    # now make matrix to scale to range x1->x2,y1->y2
    minscale: float = min((x2 - x1) / (maxx - minx), (y2 - y1) / (maxy - miny))
    return (
        translateM3(0.5 * (x1 + x2), 0.5 * (y1 + y2))
        * scaleM3(minscale)
        * translateM3(-0.5 * (minx + maxx), -0.5 * (miny + maxy))
    )


def makeScreenMFixed(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float,
) -> NDArray:
    scaling: float = 0.5 * min(x2 - x1, y2 - y1) * scale

    return translateM3(0.5 * (x1 + x2), 0.5 * (y1 + y2)) * scaleM3(scaling)


class Scene:
    class RenderMode(enum.Enum):
        RENDER_PAINTERS = 0
        RENDER_BSP = 1

    RENDER_PAINTERS = RenderMode.RENDER_PAINTERS
    RENDER_BSP = RenderMode.RENDER_BSP

    init_fragments_size: int = 512

    # internal light color and position
    class Light:
        posn = np.zeros(3)
        r: float
        g: float
        b: float

    # if passed to drawing routine, this is called after drawing each
    # fragment
    class DrawCallback:
        @abc.abstractmethod
        def drawnFragment(self, frag: Fragment) -> None: ...

    def __init__(self, _mode: int) -> None:
        self.mode: int = _mode
        # last screen matrix
        self.screenM: NDArray[np.float64] = np.zeros((3, 3))

        self.fragments: list[Fragment] = []
        self.draworder: list[int] = []
        self.lights: list[Scene.Light] = []

    def addLight(
        self,
        posn: NDArray[np.float64],
        col: qt.QColor,
        intensity: float,
    ) -> None:
        """add a light to a list"""
        light = Scene.Light()
        light.posn = posn
        light.r = col.redF() * intensity
        light.g = col.greenF() * intensity
        light.b = col.blueF() * intensity
        self.lights.append(light)

    def render(
        self,
        root: Object,
        painter: qt.QPainter,
        cam: Camera,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        scale: float,
    ) -> None:
        """
        render scene to painter in coordinate range given
        (if scale<=0, then automatic scaling)
        """
        self.render_internal(root, painter, cam, x1, y1, x2, y2, scale)

    def idPixel(
        self,
        root: Object,
        painter: qt.QPainter,
        cam: Camera,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        scale: float,
        scaling: float,
        x: int,
        y: int,
    ) -> int:
        """find widget id of pixel painted by drawing scene at (x, y)"""
        box: int = 3

        # class to keep a small pixmap of the image and keep looking for
        # changes
        class IdDrawCallback(Scene.DrawCallback):
            def __init__(self) -> None:
                super().__init__()
                self.lastwidgetid: int = 0
                self.pixrender: qt.QPixmap = qt.QPixmap(2 * box + 1, 2 * box + 1)
                self.pixrender.fill(qt.QColor(254, 254, 254))
                self.lastimage: qt.QImage = self.pixrender.toImage()

            def drawnFragment(self, frag: Fragment) -> None:
                # has the image changed since the last time?
                image: qt.QImage = self.pixrender.toImage()

                # Should only be a relatively small number of
                # comparisons. Alternatively, it could use a checksum.
                if image != self.lastimage:
                    if frag.object != 0:
                        self.lastwidgetid = frag.object.widget_id
                    self.lastimage = image

        callback: IdDrawCallback = IdDrawCallback()

        painter.begin(callback.pixrender)
        painter.scale(scaling, scaling)
        painter.setWindow(x - box, y - box, box * 2 + 1, box * 2 + 1)
        self.render_internal(root, painter, cam, x1, y1, x2, y2, scale, callback)
        painter.end()

        return callback.lastwidgetid

    def calcLighting(self) -> None:
        """calculate lighting norms for triangles"""
        # lighting is full on
        if not self.lights:
            return

        for frag in self.fragments:
            if frag.type == Fragment.FragmentType.FR_TRIANGLE:
                if frag.surfaceprop is not None:
                    self.calcLightingTriangle(frag)
            elif frag.type == Fragment.FragmentType.FR_LINESEG:
                if frag.lineprop is not None:
                    self.calcLightingLine(frag)

    def calcLightingTriangle(self, frag: Fragment) -> None:
        """
        Calculate triangle norm. Make sure norm points towards
        the viewer @ (0,0,0)
        """
        tripos = (frag.points[0] + frag.points[1] + frag.points[2]) / 3.0
        norm = np.cross(
            frag.points[1] - frag.points[0], frag.points[2] - frag.points[0]
        )
        if np.dot(tripos, norm) < 0:
            norm = -norm
        norm /= np.linalg.norm(norm)

        # get color of surface
        prop: SurfaceProp | None = frag.surfaceprop
        if prop.refl == 0.0:
            return

        r: float
        g: float
        b: float
        a: float
        if prop.hasRGBs():
            rgb: qt.QColor = prop.rgbs[min(frag.index, len(prop.rgbs) - 1)]
            r = rgb.redF()
            g = rgb.greenF()
            b = rgb.blueF()
            a = rgb.alphaF()
        else:
            r = prop.r
            g = prop.g
            b = prop.b
            a = 1.0 - prop.trans

        # add lighting contributions
        for light in self.lights:
            # Now dot vector from light source to triangle with norm
            light2tri = tripos - light.posn
            light2tri /= np.linalg.norm(light2tri)

            # add new lighting index
            dotprod: float = max(0.0, float(np.dot(light2tri, norm)))

            delta: float = prop.refl * dotprod
            r += delta * light.r
            g += delta * light.g
            b += delta * light.b

        frag.calccolor = qt.qRgba(
            np.clip(int(r * 255), 0, 255),
            np.clip(int(g * 255), 0, 255),
            np.clip(int(b * 255), 0, 255),
            np.clip(int(a * 255), 0, 255),
        )
        frag.usecalccolor = True

    def calcLightingLine(self, frag: Fragment) -> None:
        prop: LineProp | None = frag.lineprop
        if prop.refl == 0.0:
            return

        r: float
        g: float
        b: float
        a: float
        if prop.hasRGBs():
            rgb: qt.QColor = prop.rgbs[min(frag.index, len(prop.rgbs) - 1)]
            r = rgb.redF()
            g = rgb.greenF()
            b = rgb.blueF()
            a = rgb.alphaF()
        else:
            r = prop.r
            g = prop.g
            b = prop.b
            a = 1.0 - prop.trans

        pmid = (frag.points[0] + frag.points[1]) * 0.5
        linevec = frag.points[1] - frag.points[0]
        linevec /= np.linalg.norm(linevec)

        # add lighting contributions
        for light in self.lights:
            light_to_pmid = light.posn - pmid
            light_to_pmid /= np.linalg.norm(light_to_pmid)
            # this is sin of angle between line segment and light
            sintheta: float = np.sqrt(
                np.sum(np.square(np.cross(linevec, light_to_pmid)))
            )
            delta: float = prop.refl * sintheta
            r += delta * light.r
            g += delta * light.g
            b += delta * light.b

        frag.calccolor = qt.qRgba(
            np.clip(int(r * 255), 0, 255),
            np.clip(int(g * 255), 0, 255),
            np.clip(int(b * 255), 0, 255),
            np.clip(int(a * 255), 0, 255),
        )
        frag.usecalccolor = True

    def projectFragments(self, cam: Camera):
        """compute projected coordinates"""
        # convert 3d to 2d coordinates using the Camera
        for f in self.fragments:
            for pi in range(f.nPointsTotal()):
                f.proj[pi] = calcProjVec(cam.perspM, f.points[pi])

    def doDrawing(
        self,
        painter: qt.QPainter,
        screenM: NDArray[np.float64],
        linescale: float,
        cam: Camera,
        callback: DrawCallback | None = None,
    ):
        # draw fragments
        lline: LineProp | None = None
        lsurf: SurfaceProp | None = None
        ltype: Fragment.FragmentType = Fragment.FragmentType.FR_NONE

        # distance to centre of plot
        dist0: float = np.sqrt(
            np.sum(np.square(vec4to3(np.dot(cam.viewM, threed.Vec4(0, 0, 0)))))
        )

        no_pen = qt.QPen(qt.Qt.PenStyle.NoPen)
        no_brush = qt.QBrush(qt.Qt.BrushStyle.NoBrush)
        painter.setPen(no_pen)
        painter.setBrush(no_brush)

        proj_pts: list[qt.QPointF] = [qt.QPointF(), qt.QPointF(), qt.QPointF()]

        for i in range(len(self.draworder)):
            frag = self.fragments[self.draworder[i]]

            # convert projected points to screen
            for pi in range(frag.nPointsTotal()):
                p = projVecToScreen(screenM, frag.proj[pi])
                proj_pts[pi].setX(p[0])
                proj_pts[pi].setY(p[1])

            if frag.type == Fragment.FragmentType.FR_TRIANGLE:
                if frag.surfaceprop is not None and not frag.surfaceprop.hide:
                    if (
                        ltype != frag.type
                        or lsurf != frag.surfaceprop
                        or (
                            frag.surfaceprop is not None
                            and (frag.surfaceprop.hasRGBs() or frag.usecalccolor)
                        )
                    ):
                        lsurf = frag.surfaceprop
                        painter.setBrush(self.surfaceProp2QBrush(frag))

                        # use a pen if the surface is not transparent, to
                        # fill up the gaps between triangles when there is
                        # anti-aliasing
                        if frag.surfaceprop.trans == 0:
                            painter.setPen(self.surfaceProp2QPen(frag))
                        else:
                            painter.setPen(no_pen)

                    painter.drawPolygon(
                        proj_pts
                    )  # BUG: `Qt.FillRule` doesn't have a value of 3

            elif frag.type == Fragment.FragmentType.FR_LINESEG:
                if frag.lineprop is not None and not frag.lineprop.hide:
                    if ltype != frag.type or lsurf is not None:
                        painter.setBrush(no_brush)
                        lsurf = None
                    if (
                        ltype != frag.type
                        or lline != frag.lineprop
                        or (
                            frag.lineprop is not None
                            and (frag.lineprop.hasRGBs() or frag.usecalccolor)
                        )
                    ):
                        lline = frag.lineprop
                        painter.setPen(self.lineProp2QPen(frag, linescale))
                    painter.drawLine(proj_pts[0], proj_pts[1])

            elif frag.type == Fragment.FragmentType.FR_PATH:
                if (
                    ltype != frag.type
                    or lline != frag.lineprop
                    or (frag.lineprop is not None and frag.lineprop.hasRGBs())
                ):
                    lline = frag.lineprop
                    painter.setPen(self.lineProp2QPen(frag, linescale))
                if (
                    ltype != frag.type
                    or lsurf != frag.surfaceprop
                    or (
                        frag.surfaceprop is not None
                        and (frag.surfaceprop.hasRGBs() or frag.usecalccolor)
                    )
                ):
                    lsurf = frag.surfaceprop
                    painter.setBrush(self.surfaceProp2QBrush(frag))

                # ratio of distance for size scaling
                distinvratio: float = np.sqrt(np.sum(np.square(dist0 / frag.points[0])))

                self.drawPath(
                    painter,
                    frag,
                    proj_pts[0],
                    proj_pts[1],
                    proj_pts[2],
                    linescale,
                    distinvratio,
                )

            if callback is not None:
                callback.drawnFragment(frag)

            ltype = frag.type

    def drawPath(
        self,
        painter: qt.QPainter,
        frag: Fragment,
        pt1: qt.QPointF,
        pt2: qt.QPointF,
        pt3: qt.QPointF,
        linescale: float,
        distscale: float,
    ):
        pars = frag.params
        scale: float = frag.pathsize * linescale

        if pars.scalepersp:
            scale *= distscale

        # hook into drawing routine
        if pars.runcallback:
            pars.callback(painter, pt1, pt2, pt3, frag.index, scale, linescale)
            return

        if pars.scaleline:
            painter.save()
            painter.translate(pt1.x(), pt1.y())
            painter.scale(scale, scale)
            painter.drawPath(pars.path)
            painter.restore()
        else:
            # scale point and relocate
            path = qt.QPainterPath(pars.path)
            for i in range(path.elementCount()):
                el: qt.QPainterPath.Element = path.elementAt(i)
                path.setElementPositionAt(
                    i, el.x * scale + pt1.x(), el.y * scale + pt1.y()
                )
            painter.drawPath(path)

    # different rendering modes
    def renderPainters(self, cam: Camera):
        self.calcLighting()

        breakLongLines(self.fragments, 0.25)
        self.projectFragments(cam)

        # simple painter's algorithm
        self.draworder.extend(range(len(self.fragments)))

        self.draworder.sort(
            key=lambda draworder: self.fragments[draworder].maxDepth(),
            reverse=True,
        )

    def renderBSP(self, cam: Camera):
        self.calcLighting()

        # print("\nFragment size 1", length(self.fragments))

        # This is a hack to force lines to be rendered in front of
        # triangles and paths to be rendered in front of lines. Suggestions
        # to fix this are welcome.
        for f in self.fragments:
            if f.type == 2:
                f.points[0][2] += LINE_DELTA_DEPTH
                f.points[1][2] += LINE_DELTA_DEPTH
            if f.type == 1:
                f.points[0][2] += 2 * LINE_DELTA_DEPTH
                f.points[1][2] += 2 * LINE_DELTA_DEPTH

        bsp: BSPBuilder = BSPBuilder(self.fragments, np.array([0, 0, 1]))
        self.draworder = bsp.getFragmentIdxs(self.fragments)

        # print("BSP recs size ", bsp.bsp_recs.size())
        # print("Fragment size 2", fragments.size())

        self.projectFragments(cam)

    # render scene to painter in coordinate range given
    # (if scale<=0 then automatic scaling)
    def render_internal(
        self,
        root: Object,
        painter: qt.QPainter,
        cam: Camera,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        scale: float,
        callback: DrawCallback | None = None,
    ):
        self.draworder = []

        # get fragments for whole scene
        self.fragments = root.getFragments(cam.perspM, cam.viewM)

        if self.mode == self.RenderMode.RENDER_BSP:
            self.renderBSP(cam)
        elif self.mode == self.RenderMode.RENDER_PAINTERS:
            self.renderPainters(cam)

        # how to transform projected points to screen (screenM is member)
        self.screenM = (
            makeScreenM(self.fragments, x1, y1, x2, y2)
            if scale <= 0
            else makeScreenMFixed(x1, y1, x2, y2, scale)
        )

        linescale: float = max(abs(x2 - x1), abs(y2 - y1)) / 1000.0

        # finally, draw items
        self.doDrawing(painter, self.screenM, linescale, cam, callback)

        # don't decrease size of fragments unnecessarily, unless it is large
        self.init_fragments_size = len(self.fragments)
        while self.init_fragments_size > 65536:
            self.init_fragments_size /= 2

    # create pens/brushes
    def lineProp2QPen(self, frag: Fragment, linescale: float) -> qt.QPen:
        p: LineProp = frag.lineprop
        if p is None or p.hide:
            return qt.QPen(qt.Qt.PenStyle.NoPen)

        col: qt.QColor
        if frag.usecalccolor:
            col = qt.QColor.fromRgba(frag.calccolor)
        else:
            col = p.color(frag.index)

        pen: qt.QPen = qt.QPen(qt.QBrush(col), p.width * linescale, p.style)

        if p.dashpattern:
            pen.setDashPattern(p.dashpattern)

        return pen

    def surfaceProp2QColor(self, frag: Fragment) -> qt.QColor:
        """calculate color, including reflection"""
        if frag.usecalccolor:
            return qt.QColor.fromRgba(frag.calccolor)
        return frag.surfaceprop.color(frag.index)

    def surfaceProp2QBrush(self, frag: Fragment) -> qt.QBrush:
        if frag.surfaceprop is None or frag.surfaceprop.hide:
            return qt.QBrush()
        else:
            return qt.QBrush(self.surfaceProp2QColor(frag))

    def surfaceProp2QPen(self, frag: Fragment) -> qt.QPen:
        if frag.surfaceprop is None or frag.surfaceprop.hide:
            return qt.QPen(qt.Qt.PenStyle.NoPen)
        else:
            return qt.QPen(self.surfaceProp2QColor(frag))
