from funcs import normalise
import numpy as N

class Camera(object):
    """Camera object defines the view matrix

    Member matrices:
     viewM: view matrix
     perspM: perspective matrix
     combM: perspM*viewM
    """

    def __init__(self):
        self.viewM = self.perspM = N.identity(4)
        self.setPointing( (0,0,0), (0,0,1), (0,1,0) )
        self.setPerspective()

    def setPointing(self, eye, target, up):
        """Look at target position from eye, given up vector.

        See glm code lookAt
        """

        # is it this one or the one below?
        # http://3dgep.com/?p=1700

        self.eye = eye

        eye = N.array(eye)
        target = N.array(target)
        up = N.array(up)

        f = normalise(target - eye)
        u = normalise(up)
        s = normalise(N.cross(f, u))
        u = N.cross(s, f)

        self.viewM = N.array([
                [  s[0],  s[1],  s[2], -s.dot(eye) ],
                [  u[0],  u[1],  u[2], -u.dot(eye) ],
                [ -f[0], -f[1], -f[2],  f.dot(eye) ],
                [  0,     0,     0,     1          ]])

        self.combM = self.perspM.dot(self.viewM)

    def setPerspective(self, fovy_degrees=45., aspect=1., znear=0.1, zfar=100.):
        """Set perspective transform matrix.

        fovy_degrees: field of view in y direction in degrees
        aspect: aspect ratio
        znear: clip things nearer than this (should be as big as
               possible for precision)
        zfar: far clipping plane.
        """

        r = N.tan(fovy_degrees * N.pi/180. / 2.) * znear
        left = -r * aspect
        right = r * aspect
        bottom = -r
        top = r

        self.perspM = N.array([
                [ (2*znear)/(right-left), 0, 0, 0 ],
                [ 0, (2*znear)/(top-bottom), 0, 0 ],
                [ 0, 0, -(zfar+znear)/(zfar-znear), -1 ],
                [ 0, 0, -(2*zfar*znear)/(zfar-znear), 0 ]
                ])
        self.combM = self.perspM.dot(self.viewM)
