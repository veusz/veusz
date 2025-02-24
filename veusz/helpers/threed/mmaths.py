import numpy as np
from numpy.typing import NDArray


def vec4to3(v: NDArray[np.float64]) -> NDArray[np.float64]:
    assert v.shape == (4,)
    return v[:3] / v[3]


def vec3to4(v: NDArray[np.float64]) -> NDArray[np.float64]:
    assert v.shape == (3,)
    return np.array([v[0], v[1], v[2], 1.0])


def vec3to2(v: NDArray[np.float64]) -> NDArray[np.float64]:
    assert v.shape == (3,)
    return v[:2]


def rotateM4(angle: float, vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """create a rotation matrix"""
    assert vec.shape == (3,)
    c: float = np.cos(angle)
    s: float = np.sin(angle)

    a: NDArray[np.float64] = np.linalg.norm(vec)
    assert a.shape == (3,)
    t: NDArray[np.float64] = a * (1 - c)
    assert t.shape == (3,)

    m: NDArray[np.float64] = np.eye(4)
    m[0, 0] = c + t[0] * a[0]
    m[0, 1] = 0 + t[1] * a[0] - s * a[2]
    m[0, 2] = 0 + t[2] * a[0] + s * a[1]

    m[1, 0] = 0 + t[0] * a[1] + s * a[2]
    m[1, 1] = c + t[1] * a[1]
    m[1, 2] = 0 + t[2] * a[1] - s * a[0]

    m[2, 0] = 0 + t[0] * a[2] - s * a[1]
    m[2, 1] = 0 + t[1] * a[2] + s * a[0]
    m[2, 2] = c + t[2] * a[2]

    return m


# rotation matrix in terms of sin and cos of three angles in x,y,z
# doing z rotation, then y, then x
def rotate3M4_cs(
    sx: float,
    cx: float,
    sy: float,
    cy: float,
    sz: float,
    cz: float,
) -> NDArray[np.float64]:
    m: NDArray[np.float64] = np.eye(4)

    m[0, 0] = cy * cz
    m[0, 1] = cz * sx * sy - cx * sz
    m[0, 2] = cx * cz * sy + sx * sz

    m[1, 0] = cy * sz
    m[1, 1] = cx * cz + sx * sy * sz
    m[1, 2] = cx * sy * sz - cz * sx

    m[2, 0] = -sy
    m[2, 1] = cy * sx
    m[2, 2] = cx * cy

    return m


def rotate3M4(ax: float, ay: float, az: float) -> NDArray[np.float64]:
    return rotate3M4_cs(
        np.sin(ax),
        np.cos(ax),
        np.sin(ay),
        np.cos(ay),
        np.sin(az),
        np.cos(az),
    )


def scaleM3(s: float) -> NDArray[np.float64]:
    m: NDArray[np.float64] = np.eye(3)
    m[0, 0] = s
    m[1, 1] = s
    return m


def scaleM4(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """create a scaling matrix"""
    m: NDArray[np.float64] = np.eye(4)
    m[0, 0] = s[0]
    m[1, 1] = s[1]
    m[2, 2] = s[2]
    return m


def translateM3(dx: float, dy: float) -> NDArray[np.float64]:
    m: NDArray[np.float64] = np.eye(3)
    m[0, 2] = dx
    m[1, 2] = dy
    return m


def translationM4(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """create a translation matrix"""
    assert vec.shape == (3,)
    m: NDArray[np.float64] = np.eye(4)
    m[0, 3] = vec[0]
    m[1, 3] = vec[1]
    m[2, 3] = vec[2]
    return m


def calcProjVec(
    proj_m: NDArray[np.float64],
    v: NDArray[np.float64],
) -> NDArray[np.float64]:
    """do projection, getting x,y coordinate and depth"""
    assert proj_m.shape == (4, 4)
    if v.shape == (4,):
        nv = np.dot(proj_m, v)
    elif v.shape == (3,):
        nv = np.dot(proj_m, np.array([v[0], v[1], v[2], 1.0]))
    else:
        raise ValueError("Invalid shape:", v.shape)
    assert nv.shape == (4,)
    return nv[:3] / nv[3]


def projVecToScreen(
    screen_m: NDArray[np.float64],
    vec: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    convert projected coordinates to screen coordinates using screen matrix
    makes (x,y,depth) -> screen coordinates
    """
    assert screen_m.shape == (3, 3)
    assert vec.shape == (3,)
    mult: NDArray[np.float64] = np.dot(screen_m, np.array([vec[0], vec[1], 1]))
    assert mult.shape == (3,)
    return mult[:2] / mult[2]
