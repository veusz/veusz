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

import widget
import widgetfactory
import axis
import page
import containers
import setting
import utils

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
        s.add( setting.Distance( 'leftMargin', '10%', descr=
                                 'Distance from left of graph to '
                                 'edge of page') )
        s.add( setting.Distance( 'rightMargin', '5%', descr=
                                 'Distance from right of graph to '
                                 'edge of page') )
        s.add( setting.Distance( 'topMargin', '5%', descr=
                                 'Distance from top of graph to '
                                 'edge of page') )
        s.add( setting.Distance( 'bottomMargin', '10%', descr=
                                 'Distance from bottom of graph'
                                 'to edge of page') )
        s.add( setting.GraphBrush( 'Background',
                                   descr = 'Background plot fill' ) )
        s.add( setting.Line('Border',
                            descr = 'Graph border line') )

        s.readDefaults()

    def addDefaultSubWidgets(self):
        """Add axes automatically."""

        if self.parent.getChild('x') == None:
            axis.Axis(self, name='x')
        if self.parent.getChild('y') == None:
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
        while w != None and remain > 0:
            for c in w.children:
                name = c.name
                if name in widgets and widgets[name] == None:
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

    def draw(self, parentposn, painter):
        '''Update the margins before drawing.'''

        s = self.settings
        self.margins = [s.leftMargin, s.topMargin,
                        s.rightMargin, s.bottomMargin]

        bounds = self.computeBounds(parentposn, painter)

        # if there's a background
        if not s.Background.hide:
            brush = s.get('Background').makeQBrush()
            painter.fillRect( bounds[0], bounds[1], bounds[2]-bounds[0]+1,
                              bounds[3]-bounds[1]+1, brush )

        # if there's a border
        if not s.Border.hide:
            painter.setPen( s.get('Border').makeQPen(painter) )
            painter.drawRect( bounds[0], bounds[1], bounds[2]-bounds[0]+1,
                              bounds[3]-bounds[1]+1 )

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

        # if there are any
        if len(axestodraw) != 0:
            # now redraw all these axes if they aren't children of us
            axestodraw = [ i for i in axestodraw.keys()
                           if i not in childrennames ]

            # nasty, as we have to work out whether we're on the edge of
            # a collection
            edgezero = [ (a==b) for a, b in zip(bounds, parentposn) ]
            margins = utils.cnvtDists(self.margins, painter)

            axeswidgets = self.getAxes(axestodraw)
            for w in axeswidgets:
                # find which side the axis is against
                edge = w.againstWhichEdge()

                # if it's in the middle of the plot (edges == None)
                # or the distance to the edge is not zero,
                # and the margin is zero, suppress text
                
                showtext = edge == None or edgezero[edge] or margins[edge] != 0
                
                w.draw(bounds, painter, suppresstext=not showtext)

        widget.Widget.draw(self, parentposn, painter)

# allow users to make Graph objects
widgetfactory.thefactory.register( Graph )

