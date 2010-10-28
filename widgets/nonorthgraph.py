import controlgraph
from widget import Widget
from page import Page
from grid import Grid

import veusz.setting as setting

class NonOrthGraph(Widget):
    '''Non-orthogonal graph base widget.'''

    allowedparenttypes = [Page, Grid]

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        Widget.addSettings(s)

        s.add( setting.Distance( 'leftMargin',
                                 '1.7cm',
                                 descr='Distance from left of graph to edge',
                                 usertext='Left margin',
                                 formatting=True) )
        s.add( setting.Distance( 'rightMargin',
                                 '0.2cm',
                                 descr='Distance from right of graph to edge',
                                 usertext='Right margin',
                                 formatting=True) )
        s.add( setting.Distance( 'topMargin',
                                 '0.2cm',
                                 descr='Distance from top of graph to edge',
                                 usertext='Top margin',
                                 formatting=True) )
        s.add( setting.Distance( 'bottomMargin',
                                 '1.7cm',
                                 descr='Distance from bottom of graph to edge',
                                 usertext='Bottom margin',
                                 formatting=True) )
        s.add( setting.GraphBrush( 'Background',
                                   descr = 'Background plot fill',
                                   usertext='Background'),
               pixmap='settings_bgfill' )
        s.add( setting.Line('Border', descr = 'Graph border line',
                            usertext='Border'),
               pixmap='settings_border')

    def graphToPlotCoords(self, coorda, coordb):
        '''Convert graph to plotting coordinates.
        Returns (plta, pltb) coordinates
        '''
    
    def drawGraph(self, painter, bounds, datarange, outerbounds=None):
        '''Plot graph area.'''

    def drawAxes(self, painter, bounds, datarange, outerbounds=None):
        '''Plot axes.'''

    def setClip(self, painter, bounds):
        '''Set clipping for graph.'''

    def getDataRange(self):
        """Get automatic data range."""
        drange = [1e199, -1e199, 1e199, -1e199]
        for c in self.children:
            c.updateDataRanges(drange)
        return drange
            
    def draw(self, parentposn, painter, outerbounds=None):
        '''Update the margins before drawing.'''

        s = self.settings

        margins = ( s.get('leftMargin').convert(painter),
                    s.get('topMargin').convert(painter),
                    s.get('rightMargin').convert(painter),
                    s.get('bottomMargin').convert(painter) )
        bounds = self.computeBounds(parentposn, painter, margins=margins)
        maxbounds = self.computeBounds(parentposn, painter)

        # controls for adjusting graph margins
        self.controlgraphitems = [
            controlgraph.ControlMarginBox(self, bounds, maxbounds, painter)
            ]

        # do no painting if hidden
        if s.hide:
            return bounds

        # plot graph
        painter.beginPaintingWidget(self, bounds)
        datarange = self.getDataRange()
        self.drawGraph(painter, bounds, datarange, outerbounds=outerbounds)
        painter.endPaintingWidget()

        # paint children
        painter.save()
        self.setClip(painter, bounds)
        for c in reversed(self.children):
            c.draw(bounds, painter, outerbounds=outerbounds)
        painter.restore()

        # draw axes
        painter.beginPaintingWidget(self, bounds)
        self.drawAxes(painter, bounds, datarange, outerbounds=outerbounds)
        painter.endPaintingWidget()

        return bounds

    def updateControlItem(self, cgi):
        """Graph resized or moved - call helper routine to move self."""
        cgi.setWidgetMargins()
