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

import numpy as N

from .. import utils
from .. import qtall as qt4

class SurfaceProp(object):
    """Describe surface properties."""
    def __init__(self):
        self.color = (1.,0.,0.)
        self.specular = 0.5
        self.diffuse = 0.5
        self.trans = 0.

    def calcQColor(self):
        return qt4.QColor(int(self.color[0]*255),
                          int(self.color[1]*255),
                          int(self.color[2]*255),
                          int((1-self.trans)*255))

    def makeQBrush(self):
        return qt4.QBrush(self.calcQColor())

class LineProp(object):
    """Describe line properties."""
    def __init__(self):
        self.color = (0.,0.,0.)
        self.specular = 0.5
        self.diffuse = 0.5
        self.trans = 0.
        self.thickness = 1.

    def calcQColor(self):
        return qt4.QColor(int(self.color[0]*255),
                          int(self.color[1]*255),
                          int(self.color[2]*255),
                          int((1-self.trans)*255))

    def makeQPen(self):
        return qt4.QPen(qt4.QBrush(self.calcQColor), self.thickness)

class Object(object):
    """Object in scene."""
    def __init__(self):
        self.sceneM = N.identity(4)

    def getPoints(self, outersceneM):
        """Return a list of (object, points) for children.

        Each of these children should be able to have .draw() called on them
        """
        return []

class Primary(Object):
    """Object in scene which consists of points."""
    def __init__(self, points):
        Object.__init__(self)
        self.points = points

    def getPoints(self, outersceneM):
        totM = outersceneM.dot(self.sceneM)
        outpts = []
        for pt in self.points:
            outpts.append( totM.dot(pt) )
        return [(self, N.array(outpts))]

class Triangle(Primary):
    def __init__(self, points, surfaceprop):
        Primary.__init__(self, points)
        self.surfaceprop = surfaceprop

    def draw(self, painter, projpts):
        painter.setBrush( self.surfaceprop.makeQBrush() )
        painter.setPen( qt4.QPen(qt4.Qt.NoPen) )
        painter.drawPolygon(
            qt4.QPolygonF([ qt4.QPointF(projpts[0][0], projpts[0][1]),
                            qt4.QPointF(projpts[1][0], projpts[1][1]),
                            qt4.QPointF(projpts[2][0], projpts[2][1]) ])
            )

class Polyline(Primary):
    def __init__(self, points, lineprop):
        Primary.__init__(self, points)
        self.lineprop = lineprop

    def draw(self, painter, projpts):
        painter.setBrush( qt4.QBrush() )
        painter.setPen( self.lineprop.makeQPen() )
        poly = qt4.QPolygonF()
        utils.addNumpyToPolygonF(poly, projpts[:,0], projpts[:,1])
        painter.drawPolygon(poly)

class Compound(Object):
    """Object in scene made of other objects."""

    def __init__(self, objs):
        Object.__init__(self)
        self.objs = objs

    def getPoints(self, outersceneM):
        totM = outersceneM.dot(self.sceneM)
        allpts = []
        for obj in self.objs:
            allpts += obj.getPoints(totM)
        return allpts
