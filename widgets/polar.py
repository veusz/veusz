import qtall as qt4
import numpy as N

from nonorthgraph import NonOrthGraph
from axisticks import AxisTicks
from axis import TickLabel

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

class Tick(setting.Line):
    '''Tick settings.'''

    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)
        self.add( setting.DistancePt( 'length',
                                      '6pt',
                                      descr = 'Length of major ticks',
                                      usertext='Length') )
        self.add( setting.Int( 'number',
                               6,
                               descr = 'Number of major ticks to aim for',
                               usertext='Number') )
        self.add( setting.Bool('hidespokes', False,
                               descr = 'Hide radial spokes',
                               usertext = 'Hide spokes') )
        self.add( setting.Bool('hideannuli', False,
                               descr = 'Hide annuli',
                               usertext = 'Hide annuli') )

    def getLength(self, painter):
        '''Return tick length in painter coordinates'''
        
        return self.get('length').convert(painter)

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
        s.add( setting.Choice('dataunits',
                              ('degrees', 'radians'), 
                              'degrees', 
                              descr = 'Angular units of data',
                              usertext='Data units') )
        s.add( setting.Choice('direction',
                              ('clockwise', 'anticlockwise'),
                              'clockwise',
                              descr = 'Angle direction',
                              usertext = 'Direction') )


        s.add( TickLabel('TickLabels', descr = 'Tick labels',
                    usertext='Tick labels'),
               pixmap='settings_axisticklabels' )
        s.add( Tick('Tick', descr = 'Tick line',
                    usertext='Tick'),
               pixmap='settings_axismajorticks' )

    def graphToPlotCoords(self, coorda, coordb):
        '''Convert coordinates in r, theta to x, y.'''

        cb = coordb
        if self.settings.dataunits == 'degrees':
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

        s = self.settings
        t = s.Tick
        atick = AxisTicks(0, self._maxradius, t.number,
                          t.number*4,
                          extendbounds=False,  extendzero=False)
        minval, maxval, majtick, mintick, tickformat = atick.getTicks()

        # draw ticks as circles
        if not t.hideannuli:
            painter.setPen( s.Tick.makeQPenWHide(painter) )
            painter.setBrush( qt4.QBrush() )      
            for tick in majtick[1:]:
                radius = tick / self._maxradius

                painter.drawEllipse(qt4.QRectF(
                        qt4.QPointF( self._xc - radius*self._xscale,
                                     self._yc - radius*self._yscale ),
                        qt4.QPointF( self._xc + radius*self._xscale,
                                     self._yc + radius*self._yscale ) ))

        tl = s.TickLabels
        scale, format = tl.scale, tl.format
        if format == 'Auto':
            format = tickformat
        painter.setPen( tl.makeQPen() )
        font = tl.makeQFont(painter)
        for tick in majtick[1:]:
            num = utils.formatNumber(tick*scale, format)
            x = tick / self._maxradius * self._xscale + self._xc
            r = utils.Renderer(painter, font, x, self._yc, num, alignhorz=-1,
                               alignvert=-1, usefullheight=True)
            r.render()

        # draw spokes
        if not t.hidespokes:
            angle = 2 * N.pi / 12
            lines = []
            for i in xrange(12):
                x = self._xc +  N.cos(angle*i) * self._xscale
                y = self._yc +  N.sin(angle*i) * self._yscale
                lines.append( qt4.QLineF(qt4.QPointF(self._xc, self._yc),
                                         qt4.QPointF(x, y)) )
            painter.drawLines(lines)


document.thefactory.register(Polar)
