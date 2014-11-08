#    Copyright (C) 2013 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

from __future__ import print_function

from . import funcs
from . import objects
import numpy as N

class LightSource(object):
    def __init__(self, posn, color):
        self.posn = posn
        self.color = color

class Scene(object):
    def __init__(self):
        self.objects = []
        self.lightsource = []

    def depthsort(self, cam):
        """Return objects sorted by decreasing distance from observer."""
        outobjects = []
        # get points in view coordinates
        camT = cam.perspM.T
        for obj in self.objects:
            for obj, points in obj.getPoints(cam.viewM):
                projpoints = points.dot(camT)
                pp2d = projpoints[:,:2] / projpoints[:,3:4]
                depth = -projpoints[:,2] / projpoints[:,3]
                outobjects.append( [depth.min(), depth.max(), obj, pp2d] )

        sortmin = sorted(outobjects, reverse=True)
        sortmax = sorted(outobjects, key=lambda x: -x[1])
        return sortmin, sortmax

    def render(self, painter, cam, outwin):
        """Render objects to painter.

        outwin = (minx, miny, maxx, maxy)
        """
        sortmin, sortmax = self.depthsort(cam)

        for mindepth, maxdepth, obj, points in sortmin:

            winpts = ( (points*0.5+0.5) *
                       N.array((outwin[2]-outwin[0], outwin[3]-outwin[1])) +
                       N.array((outwin[0], outwin[1])) )
            obj.draw(painter, winpts)

"""
import random
import camera
import objects
from funcs import pt_3_4
import veusz.qtall as qt4

class MyWin(qt4.QWidget):

    def __init__(self):
        qt4.QWidget.__init__(self)

        objs = []
        surface = objects.SurfaceProp()

        for i, face in enumerate((
            ( (0,0,0), (0,1,0), (0,0,1) ),
            ( (0,1,1), (0,1,0), (0,0,1) ),

            ( (1,0,0), (1,1,0), (1,0,1) ),
            ( (1,1,1), (1,1,0), (1,0,1) ),

            ( (0,0,0), (1,0,0), (0,1,0) ),
            ( (1,1,0), (1,0,0), (0,1,0) ),

            ( (0,0,1), (1,0,1), (0,1,1) ),
            ( (1,1,1), (1,0,1), (0,1,1) ),

            ( (0,0,0), (1,0,0), (0,0,1) ),
            ( (1,0,1), (1,0,0), (0,0,1) ),

            ( (0,1,0), (1,1,0), (0,1,1) ),
            ( (1,1,1), (1,1,0), (0,1,1) ),

            )):

            if i % 2 == 1:
                s = objs[i-1].surfaceprop
            else:
                s = objects.SurfaceProp()
                s.color = ( random.random(), random.random(), random.random() )
                s.trans = random.random()
            t = objects.Triangle([pt_3_4(p) for p in face], s)
            objs.append(t)

        self.cube = objects.Compound(objs)
        self.cube.sceneM = funcs.translationM((-0.5,-0.5,-0.5))

        self.scene = Scene()
        self.scene.objects.append(self.cube)
        self.cam = camera.Camera()
        self.cam.setPointing( (0,0,-20), (0,0,1), (0,1,0) )

        self.timer = qt4.QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.ontimeout)
        self.timer.start()

    def ontimeout(self):
        self.cube.sceneM = funcs.rotateM(0.01, (0,1,0)).dot(funcs.rotateM(0.025,(1,1,1))).dot(self.cube.sceneM)
        self.update()

    def paintEvent(self, evt):
        painter = qt4.QPainter(self)
        #painter.setRenderHint( qt4.QPainter.Antialiasing )

        size = min(self.width(), self.height())-10

        painter.drawRect(0, 0, size, size)

        self.scene.render(painter, self.cam, (0,0,size,size))

        painter.end()


def main():
    app = qt4.QApplication([])

    win = MyWin()
    win.show()

    app.exec_()

if __name__ == '__main__':
    main()
"""
