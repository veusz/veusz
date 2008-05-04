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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.utils as utils
import veusz.document as document

import widget
import axis
import page
import containers

class Graph(widget.Widget):
    """Graph for containing other sorts of widgets"""
    
    typename='graph'
    allowedparenttypes = [page.Page, containers.Grid]
    allowusercreation = True
    description = 'Base graph'

    def __init__(self, parent, name=None):
        """Initialise object and create axes."""

        widget.Widget.__init__(self, parent, name=name)
        s = self.settings
        s.add( setting.Distance( 'leftMargin', '1.7cm', descr=
                                 'Distance from left of graph to '
                                 'edge of page',
                                 usertext='Left margin',
                                 formatting=True) )
        s.add( setting.Distance( 'rightMargin', '0.1cm', descr=
                                 'Distance from right of graph to '
                                 'edge of page',
                                 usertext='Right margin',
                                 formatting=True) )
        s.add( setting.Distance( 'topMargin', '0.1cm', descr=
                                 'Distance from top of graph to '
                                 'edge of page',
                                 usertext='Top margin',
                                 formatting=True) )
        s.add( setting.Distance( 'bottomMargin', '1.7cm', descr=
                                 'Distance from bottom of graph'
                                 'to edge of page',
                                 usertext='Bottom margin',
                                 formatting=True) )
        s.add( setting.GraphBrush( 'Background',
                                   descr = 'Background plot fill',
                                   usertext='Background'), pixmap='bgfill' )
        s.add( setting.Line('Border', descr = 'Graph border line',
                            usertext='Border'), pixmap='border')
        
        self.readDefaults()

    def addDefaultSubWidgets(self):
        """Add axes automatically."""

        if self.parent.getChild('x') is None:
            axis.Axis(self, name='x')
        if self.parent.getChild('y') is None:
            axis.Axis(self, name='y')

    def getAxes(self, names):
        """Get the axes for widgets to plot against.
        names is a list of names to find."""

        widgets = {}
        for n in names:
            widgets[n] = None

        remain = len(names)

        # recursively go back up the tree to find axes 
        w = self
        while w is not None and remain > 0:
            for c in w.children:
                name = c.name
                if name in widgets and widgets[name] is None:
                    widgets[name] = c
                    remain -= 1
            w = w.parent

        # return list of found widgets
        return [widgets[n] for n in names]

    def getAxesNames(self):
        """Return list of axes names used by children of this widget."""
        
        axes = []
        for c in self.children:
            try:
                axes += c.getAxesNames()
            except AttributeError:
                pass
        return axes

    def autoAxis(self, axisname, bounds):
        """If axis is used by children, update bounds."""

        # skip if another axis overrides this one
        for c in self.children:
            if c.name == axisname:
                return

        # update bounds for each of the children
        for c in self.children:
            c.autoAxis(axisname, bounds)

    def draw(self, parentposn, painter, outerbounds = None):
        '''Update the margins before drawing.'''

        s = self.settings

        margins = ( s.get('leftMargin').convert(painter),
                    s.get('topMargin').convert(painter),
                    s.get('rightMargin').convert(painter),
                    s.get('bottomMargin').convert(painter) )
        bounds = self.computeBounds(parentposn, painter, margins=margins)

        # allow use to move bounds of graph
        #self.controlpts = {
        #    0: (bounds[0], bounds[1]),
        #    1: (bounds[2], bounds[1]),
        #    2: (bounds[0], bounds[3]),
        #    3: (bounds[2], bounds[3])
        #    }

        # do no painting if hidden
        if s.hide:
            return bounds

        painter.beginPaintingWidget(self, bounds)

        # if there's a background
        if not s.Background.hide:
            brush = s.get('Background').makeQBrush()
            painter.fillRect( qt4.QRectF(bounds[0], bounds[1],
                                         bounds[2]-bounds[0],
                                         bounds[3]-bounds[1]), brush )

        # if there's a border
        if not s.Border.hide:
            painter.setPen( s.get('Border').makeQPen(painter) )
            painter.drawRect( qt4.QRectF(bounds[0], bounds[1],
                                         bounds[2]-bounds[0],
                                         bounds[3]-bounds[1]) )

        # work out outer bounds
        ob = list(parentposn)
        if outerbounds is not None:
            # see whether margin, is zero, and borrow from above if so
            for i in range(4):
                if margins[i] == 0.:
                    ob[i] = outerbounds[i]

        painter.endPaintingWidget()
        
        # do normal drawing of children
        # iterate over children in reverse order
        for i in utils.reverse(self.children):
            i.draw(bounds, painter, outerbounds=outerbounds)

        # now need to find axes which aren't children, and draw those again
        axestodraw = {}
        childrennames = {}
        for c in self.children:
            childrennames[c.name] = True
            try:
                for i in c.getAxesNames():
                    axestodraw[i] = True
            except AttributeError:
                pass

        # FIXME: this code is terrible - find a better way to do this

        # if there are any
        if len(axestodraw) != 0:
            # now redraw all these axes if they aren't children of us
            axestodraw = [ i for i in axestodraw.keys() if i not in
                           childrennames ]

            # nasty, as we have to work out whether we're on the edge of
            # a collection
            edgezero = [ abs(a-b)<2 for a, b in zip(bounds, parentposn) ]

            axeswidgets = self.getAxes(axestodraw)
            for w in axeswidgets:
                if w is None:
                    continue
                    
                # find which side the axis is against
                edge = w.againstWhichEdge()

                # if it's in the middle of the plot (edges is None)
                # or the distance to the edge is not zero,
                # and the margin is zero, suppress text
                
                showtext = ( edge is None or edgezero[edge] or
                             margins[edge] != 0 )
                
                w.draw( bounds, painter, suppresstext = not showtext,
                        outerbounds=ob )

        return bounds

    def updateControlPoint(self, name, pos, bounds):
        """Update position of point given new name and vals."""

        s = self.settings


                            
# allow users to make Graph objects
document.thefactory.register( Graph )

