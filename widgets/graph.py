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
        
        # make axes
        ax = axis.Axis(self, name='x')
        ay = axis.Axis(self, name='y')
        ay.settings.direction = 'vertical'

        # these widgets shouldn't be remade
        self.autoadd += ['x', 'y']

    def draw(self, parentposn, painter):
        '''Update the margins before drawing.'''

        s = self.settings
        self.margins = [s.leftMargin, s.topMargin,
                        s.rightMargin, s.bottomMargin]

        # get parent's position
        x1, y1, x2, y2 = parentposn
        dx, dy = x2-x1, y2-y1

        # get our position
        x1, y1, x2, y2 = ( x1+dx*self.position[0], y1+dy*self.position[1],
                           x1+dx*self.position[2], y1+dy*self.position[3] )
        dx, dy = x2-x1, y2-y1

        # convert margin to physical units and subtract
        deltas = utils.cnvtDists( self.margins, painter )
        bounds = ( int(x1+deltas[0]), int(y1+deltas[1]),
                   int(x2-deltas[2]), int(y2-deltas[3]) )

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

        widget.Widget.draw(self, parentposn, painter)

# allow users to make Graph objects
widgetfactory.thefactory.register( Graph )
 
