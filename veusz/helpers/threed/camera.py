import numpy as np
from numpy.typing import NDArray


class Camera:
    def __init__(self):
        self.viewM: NDArray[np.float64] = np.zeros((4, 4))  # view matrix
        self.perspM: NDArray[np.float64] = np.zeros((4, 4))  # perspective matrix
        self.combM: NDArray[np.float64] = np.zeros((4, 4))  # combined matrix
        self.eye: NDArray[np.float64] = np.zeros(3)  # location of eye

        self.setPointing(np.array([0, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]))
        self.setPerspective()

    def setPointing(self, eye, target, up):
        """
        Look at target position from eye, given up vector.
        See glm code lookAt
        """
        # is it this one or the one below?
        # http://3dgep.com/?p=1700

        if not isinstance(eye, np.ndarray):
            eye = np.asarray(eye)
        if not isinstance(target, np.ndarray):
            target = np.asarray(target)
        if not isinstance(up, np.ndarray):
            up = np.asarray(up)

        _ = target - eye
        f = _ / np.linalg.norm(_)
        u = up / np.linalg.norm(up)
        _ = np.cross(f, u)
        s = _ / np.linalg.norm(_)
        u = np.cross(s, f)

        self.viewM[0, 0] = s[0]
        self.viewM[0, 1] = s[1]
        self.viewM[0, 2] = s[2]
        self.viewM[0, 3] = -np.dot(s, eye)

        self.viewM[1, 0] = u[0]
        self.viewM[1, 1] = u[1]
        self.viewM[1, 2] = u[2]
        self.viewM[1, 3] = -np.dot(u, eye)

        self.viewM[2, 0] = -f[0]
        self.viewM[2, 1] = -f[1]
        self.viewM[2, 2] = -f[2]
        self.viewM[2, 3] = np.dot(f, eye)

        self.viewM[3, 0] = 0
        self.viewM[3, 1] = 0
        self.viewM[3, 2] = 0
        self.viewM[3, 3] = 1

        self.combM = self.perspM * self.viewM

    def setPerspective(
        self,
        fov_degrees: float = 90,
        znear: float = 0.1,
        zfar: float = 100,
    ):
        """
        fovy_degrees: total field of view in degrees
        znear: np.clip things nearer than this (should be as big as
               possible for precision)
        zfar: far clipping plane.
        """
        # matrix from Scratchapixel 2
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix

        scale: float = 1 / np.tan(np.degrees(fov_degrees) / 2)

        self.perspM[0, 0] = scale
        self.perspM[1, 0] = 0
        self.perspM[2, 0] = 0
        self.perspM[3, 0] = 0

        self.perspM[0, 1] = 0
        self.perspM[1, 1] = scale
        self.perspM[2, 1] = 0
        self.perspM[3, 1] = 0

        self.perspM[0, 2] = 0
        self.perspM[1, 2] = 0
        self.perspM[2, 2] = -zfar / (zfar - znear)
        self.perspM[3, 2] = -1

        self.perspM[0, 3] = 0
        self.perspM[1, 3] = 0
        self.perspM[2, 3] = -zfar * znear / (zfar - znear)
        self.perspM[3, 3] = 0

        self.combM = self.perspM * self.viewM
