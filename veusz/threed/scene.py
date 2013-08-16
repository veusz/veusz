from . import funcs
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
        objpoints = []
        # get points in view coordinates
        for obj in self.objects:
            objpoints += obj.getPoints(cam.viewM)

        outobjects = []
        for obj, points in objpoints:
            objprojpts = []
            avdepth = 0.
            for p in points:
                # calculate radius of point
                r = N.sqrt( p[0]**2 + p[1]**2 + p[2]**2 ) / p[3]
                avdepth += r

                # convert to projected coordinate
                prend = cam.perspM.dot(p)
                objprojpts.append( (prend[0]/prend[3], prend[1]/prend[3]) )
            avdepth *= (1./len(obj.points))

            outobjects.append( (avdepth, obj, N.array(objprojpts)) )

        outobjects.sort(reverse=True)
        return outobjects

    def render(self, painter, cam, outwin):
        """Render objects to painter.

        outwin = (minx, miny, maxx, maxy)
        """
        sortobjects = self.depthsort(cam)

        for depth, obj, points in sortobjects:
            winpts = ( (points*0.5+0.5) *
                       N.array((outwin[2]-outwin[0], outwin[3]-outwin[1])) +
                       N.array((outwin[0], outwin[1])) )
            obj.draw(painter, winpts)

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
