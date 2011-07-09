# graph widget for containing other sorts of widget

#    Copyright (C) 2004 Jeremy S. Sanders
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

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.utils as utils
import veusz.document as document

import widget
import axis
import page
import grid
import controlgraph

class Graph(widget.Widget):
    """Graph for containing other sorts of widgets"""
    
    typename='graph'
    allowedparenttypes = [page.Page, grid.Grid]
    allowusercreation = True
    description = 'Base graph'

    def __init__(self, parent, name=None):
        """Initialise object and create axes."""

        widget.Widget.__init__(self, parent, name=name)
        s = self.settings
        self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

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
        
    def addDefaultSubWidgets(self):
        """Add axes automatically."""

        if self.parent.getChild('x') is None:
            axis.Axis(self, name='x')
        if self.parent.getChild('y') is None:
            axis.Axis(self, name='y')

    def getAxes(self, axesnames):
        """Get the axes for widgets to plot against.
        names is a list of names to find."""

        widgets = {}
        # recursively go back up the tree to find axes 
        w = self
        while w is not None and len(widgets) < len(axesnames):
            for c in w.children:
                name = c.name
                if ( name in axesnames and name not in widgets and
                     isinstance(c, axis.Axis) ):
                    widgets[name] = c
            w = w.parent

        # didn't find everything...
        if w is None:
            for name in axesnames:
                if name not in widgets:
                    widgets[name] = None

        # return list of found widgets
        return [widgets[n] for n in axesnames]

    def draw(self, parentposn, painthelper, outerbounds = None):
        '''Update the margins before drawing.'''

        s = self.settings

        margins = ( s.get('leftMargin').convert(painthelper),
                    s.get('topMargin').convert(painthelper),
                    s.get('rightMargin').convert(painthelper),
                    s.get('bottomMargin').convert(painthelper) )

        bounds = self.computeBounds(parentposn, painthelper, margins=margins)
        maxbounds = self.computeBounds(parentposn, painthelper)

        # controls for adjusting graph margins
        painter = painthelper.painter(self, self.parent, bounds)
        painthelper.setControlGraph(self, [
                controlgraph.ControlMarginBox(self, bounds, maxbounds,
                                              painthelper) ])

        # do no painting if hidden
        if s.hide:
            return bounds

        # set graph rectangle attributes
        painter.setBrush( s.get('Background').makeQBrushWHide() )
        painter.setPen( s.get('Border').makeQPenWHide(painthelper) )

        # draw graph rectangle
        painter.drawRect( qt4.QRectF(qt4.QPointF(bounds[0], bounds[1]),
                                     qt4.QPointF(bounds[2], bounds[3])) )

        painter.end()

        # do normal drawing of children
        # iterate over children in reverse order
        for c in reversed(self.children):
            c.draw(bounds, painthelper, outerbounds=outerbounds)

        # now need to find axes which aren't children, and draw those again
        axestodraw = set()
        childrennames = set()
        for c in self.children:
            childrennames.add(c.name)
            try:
                for axis in c.getAxesNames():
                    axestodraw.add(axis)
            except AttributeError:
                pass

        axestodraw = axestodraw - childrennames
        if axestodraw:
            # now redraw all these axes if they aren't children of us
            axeswidgets = self.getAxes(axestodraw)
            for w in axeswidgets:
                if w is not None:
                    w.draw(bounds, painthelper, outerbounds=outerbounds)

        return bounds

    def updateControlItem(self, cgi):
        """Graph resized or moved - call helper routine to move self."""
        cgi.setWidgetMargins()

# allow users to make Graph objects
document.thefactory.register( Graph )

