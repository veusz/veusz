import numpy as np
from numpy.typing import NDArray

from .camera import Camera
from .clipcontainer import ClipContainer
from .mmaths import rotate3M4, scaleM4, translationM4
from .objects import (
    AxisLabels,
    DataMesh,
    FacingContainer,
    LineSegments,
    Mesh,
    MultiCuboid,
    ObjectContainer,
    Points,
    PolyLine,
    TriangleFacing,
)
from .properties import LineProp, SurfaceProp
from .scene import Scene

__all__ = [
    "Camera",
    "ClipContainer",
    "rotate3M4",
    "translationM4",
    "scaleM4",
    "AxisLabels",
    "DataMesh",
    "FacingContainer",
    "Mesh",
    "LineSegments",
    "MultiCuboid",
    "ObjectContainer",
    "Points",
    "PolyLine",
    "TriangleFacing",
    "LineProp",
    "SurfaceProp",
    "Scene",
    "Vec3",
    "Vec4",
    "ValVector",
]


def Vec3(x: float = 0, y: float = 0, z: float = 0) -> NDArray[np.float64]:
    return np.asarray([x, y, z])


def Vec4(*args: float) -> NDArray[np.float64]:
    if not args:
        return np.zeros(4)
    if len(args) == 3:
        return np.asarray([*args, 1])
    if len(args) == 4:
        return np.asarray([*args])
    raise ValueError


ValVector = np.asarray
