#    Copyright (C) 2010 Jeremy S. Sanders
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

"""Non orthogonal graph root."""

import controlgraph
from widget import Widget
from page import Page
from grid import Grid

import veusz.setting as setting

filloptions = ('center', 'outside', 'top', 'bottom', 'left', 'right',
               'polygon')

class FillBrush(setting.BrushExtended):
    '''Brush for filling point region.'''
    def __init__(self, *args, **argsv):
        setting.BrushExtended.__init__(self, *args, **argsv)
        self.add( setting.Choice('filltype', filloptions, 'center',
                                 descr='Fill to this edge/position',
                                 usertext='Fill type') )
        self.get('hide').newDefault(True)

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

    def coordRanges(self):
        '''Return coordinate ranges of plot.
        This is in the form [[mina, maxa], [minb, maxb]].'''

    def drawFillPts(self, painter, extfill, bounds, ptsx, ptsy):
        '''Draw set of points for filling.
        extfill: extended fill brush
        bounds: usual tuple (minx, miny, maxx, maxy)
        ptsx, ptsy: translated plotter coordinates
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
            
    def draw(self, parentposn, phelper, outerbounds=None):
        '''Update the margins before drawing.'''

        s = self.settings

        margins = ( s.get('leftMargin').convert(phelper),
                    s.get('topMargin').convert(phelper),
                    s.get('rightMargin').convert(phelper),
                    s.get('bottomMargin').convert(phelper) )
        bounds = self.computeBounds(parentposn, phelper, margins=margins)
        maxbounds = self.computeBounds(parentposn, phelper)

        painter = phelper.painter(self, bounds)

        # controls for adjusting margins
        phelper.setControlGraph(self, [
                controlgraph.ControlMarginBox(self, bounds, maxbounds, phelper)])

        # do no painting if hidden
        if s.hide:
            return bounds

        # plot graph
        datarange = self.getDataRange()
        self.drawGraph(painter, bounds, datarange, outerbounds=outerbounds)
        self.drawAxes(painter, bounds, datarange, outerbounds=outerbounds)

        # paint children
        for c in reversed(self.children):
            c.draw(bounds, phelper, outerbounds=outerbounds)

        return bounds

    def updateControlItem(self, cgi):
        """Graph resized or moved - call helper routine to move self."""
        cgi.setWidgetMargins()
