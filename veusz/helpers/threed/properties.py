from ... import qtall as qt

"""
These classes describe the color and properties of a surface or line

A reference counting scheme (PropSmartPtr) is used to keep track of
when to delete them. PropSmartPtr is an intrusive pointer, which
uses a reference count in the object to keep track of how many
copies are used.
"""


def _qimage2rgbvec(img: qt.QImage) -> list[qt.QColor]:
    """helper to convert images to list of rgbs"""
    # TODO: is it only the 1st line?
    return [img.pixelColor(x, 0) for x in range(img.width())]


class SurfaceProp:
    def __init__(
        self,
        r: float = 0.5,
        g: float = 0.5,
        b: float = 0.5,
        refl: float = 0.5,
        trans: float = 0,
        hide: bool = False,
    ) -> None:
        self.r: float = r
        self.g: float = g
        self.b: float = b
        self.refl: float = refl
        self.trans: float = trans
        self.hide: bool = hide
        self.rgbs: list[qt.QColor] = []
        # used to reference count usages by Object() instances
        self._ref_cnt: int = 0

    def hasRGBs(self) -> bool:
        return bool(self.rgbs)

    def setRGBs(self, img: qt.QImage) -> None:
        self.rgbs = _qimage2rgbvec(img)

    def color(self, idx: int) -> qt.QColor:
        if not self.rgbs:
            return qt.QColor(
                int(self.r * 255),
                int(self.g * 255),
                int(self.b * 255),
                int((1 - self.trans) * 255),
            )
        else:
            return self.rgbs[min(len(self.rgbs) - 1, idx)]


class LineProp:
    def __init__(
        self,
        r: float = 0,
        g: float = 0,
        b: float = 0,
        trans: float = 0,
        refl: float = 0,
        width: float = 1,
        hide: bool = 0,
        style: qt.Qt.PenStyle = qt.Qt.PenStyle.SolidLine,
    ):
        self.r: float = r
        self.g: float = g
        self.b: float = b
        self.trans: float = trans
        self.refl: float = refl
        self.width: float = width
        self.hide: bool = hide
        self.style: qt.Qt.PenStyle = style
        self.rgbs: list = []
        self.dashpattern: list[float] = []
        # used to reference count usages by Object() instances
        self._ref_cnt: int = 0

    def hasRGBs(self) -> bool:
        return bool(self.rgbs)

    def setRGBs(self, img: qt.QImage):
        self.rgbs = _qimage2rgbvec(img)

    def color(self, idx) -> qt.QColor:
        if not self.rgbs:
            return qt.QColor(
                int(self.r * 255),
                int(self.g * 255),
                int(self.b * 255),
                int((1 - self.trans) * 255),
            )
        else:
            return qt.QColor.fromRgba(self.rgbs[min(len(self.rgbs) - 1, idx)])

    def setDashPattern(self, vec: list[float]):
        self.dashpattern = vec[:]
