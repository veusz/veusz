import qtall as qt4
import numpy as N

from nonorthgraph import NonOrthGraph
import veusz.document as document
import veusz.setting as setting

class Polar(NonOrthGraph):

    typename='polar'
    allowusercreation = True
    description = 'Polar graph'

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        NonOrthGraph.addSettings(s)

        s.add( setting.FloatOrAuto('maxradius', 'Auto',
                                   descr='Maximum value of radius',
                                   usertext='Max radius') )
        s.add( setting.Choice('angle',
                              ('degrees', 'radians'), 
                              'degrees', 
                              descr = 'Angular units to use',
                              usertext='Angular units') )

    def graphToPlotCoords(self, coorda, coordb):
        '''Convert coordinates in r, theta to x, y.'''

        print self._maxradius

        cb = coordb
        if self.settings.angle == 'degrees':
            cb = coordb * (N.pi/180.)
        ca = coorda / self._maxradius

        x = self._xc + ca * N.cos(cb) * self._xscale
        y = self._yc + ca * N.sin(cb) * self._yscale
        return x, y

    def drawGraph(self, painter, bounds, datarange, outerbounds=None):
        '''Plot graph area and axes.'''

        s = self.settings
        if s.maxradius == 'Auto':
            self._maxradius = datarange[1]
        else:
            self._maxradius = s.maxradius
    
        self._xscale = (bounds[2]-bounds[0])*0.5
        self._yscale = (bounds[3]-bounds[1])*0.5
        self._xc = 0.5*(bounds[0]+bounds[2])
        self._yc = 0.5*(bounds[3]+bounds[1])

        painter.setPen( s.Border.makeQPenWHide(painter) )
        painter.setBrush( s.Background.makeQBrushWHide() )
        painter.drawEllipse( qt4.QRectF( qt4.QPointF(bounds[0], bounds[1]),
                                         qt4.QPointF(bounds[2], bounds[3]) ) )


    def setClip(self, painter, bounds):
        '''Set clipping for graph.'''
        p = qt4.QPainterPath()
        p.addEllipse( qt4.QRectF( qt4.QPointF(bounds[0], bounds[1]),
                                  qt4.QPointF(bounds[2], bounds[3]) ) )
        painter.setClipPath(p)

    def drawAxes(self, painter, bounds, datarange, outerbounds=None):
        '''Plot axes.'''


document.thefactory.register(Polar)
