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

from __future__ import division
from . import controlgraph
from .widget import Widget

from .. import qtall as qt4
from .. import setting

filloptions = ('center', 'outside', 'top', 'bottom', 'left', 'right',
               'polygon')

def _(text, disambiguation=None, context='NonOrthGraph'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class FillBrush(setting.BrushExtended):
    '''Brush for filling point region.'''
    def __init__(self, *args, **argsv):
        setting.BrushExtended.__init__(self, *args, **argsv)
        self.add( setting.Choice('filltype', filloptions, 'center',
                                 descr=_('Fill to this edge/position'),
                                 usertext=_('Fill type')) )
        self.get('hide').newDefault(True)

class NonOrthGraph(Widget):
    '''Non-orthogonal graph base widget.'''

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        Widget.addSettings(s)

        s.add( setting.Distance( 'leftMargin',
                                 '1.7cm',
                                 descr=_('Distance from left of graph to edge'),
                                 usertext=_('Left margin'),
                                 formatting=True) )
        s.add( setting.Distance( 'rightMargin',
                                 '0.2cm',
                                 descr=_('Distance from right of graph to edge'),
                                 usertext=_('Right margin'),
                                 formatting=True) )
        s.add( setting.Distance( 'topMargin',
                                 '0.2cm',
                                 descr=_('Distance from top of graph to edge'),
                                 usertext=_('Top margin'),
                                 formatting=True) )
        s.add( setting.Distance( 'bottomMargin',
                                 '1.7cm',
                                 descr=_('Distance from bottom of graph to edge'),
                                 usertext=_('Bottom margin'),
                                 formatting=True) )
        s.add( setting.GraphBrush( 'Background',
                                   descr = _('Background plot fill'),
                                   usertext=_('Background')),
               pixmap='settings_bgfill' )
        s.add( setting.Line('Border', descr = _('Graph border line'),
                            usertext=_('Border')),
               pixmap='settings_border')

    @classmethod
    def allowedParentTypes(klass):
        from . import page, grid
        return (page.Page, grid.Grid)

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
        '''Plot graph area.
        datarange is  [mina, maxa, minb, maxb] or None
        '''

    def drawAxes(self, painter, bounds, datarange, outerbounds=None):
        '''Plot axes.
        datarange is  [mina, maxa, minb, maxb] or None
        '''

    def setClip(self, painter, bounds):
        '''Set clipping for graph.'''

    def getDataRange(self):
        """Get automatic data range. Return None if no data."""

        drange = [1e199, -1e199, 1e199, -1e199]
        for c in self.children:
            if hasattr(c, 'updateDataRanges'):
                c.updateDataRanges(drange)

        # no data
        if drange[0] > drange[1] or drange[2] > drange[3]:
            drange = None

        return drange

    def getMargins(self, painthelper):
        """Use settings to compute margins."""
        s = self.settings
        return ( s.get('leftMargin').convert(painthelper),
                 s.get('topMargin').convert(painthelper),
                 s.get('rightMargin').convert(painthelper),
                 s.get('bottomMargin').convert(painthelper) )

    def draw(self, parentposn, phelper, outerbounds=None):
        '''Update the margins before drawing.'''

        s = self.settings

        bounds = self.computeBounds(parentposn, phelper)
        maxbounds = self.computeBounds(parentposn, phelper, withmargin=False)

        # do no painting if hidden
        if s.hide:
            return bounds

        painter = phelper.painter(self, bounds)
        with painter:
            # reset counter and compute automatic colors
            phelper.autoplottercount = 0
            for c in self.children:
                c.setupAutoColor(painter)

            # plot graph
            datarange = self.getDataRange()
            self.drawGraph(painter, bounds, datarange, outerbounds=outerbounds)
            self.drawAxes(painter, bounds, datarange, outerbounds=outerbounds)

            # paint children
            for c in reversed(self.children):
                c.draw(bounds, phelper, outerbounds=outerbounds)

        # controls for adjusting margins
        phelper.setControlGraph(self, [
                controlgraph.ControlMarginBox(self, bounds, maxbounds, phelper)])

        return bounds

    def updateControlItem(self, cgi):
        """Graph resized or moved - call helper routine to move self."""
        cgi.setWidgetMargins()
