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

        widget.Widget.draw(self, parentposn, painter)

# allow users to make Graph objects
widgetfactory.thefactory.register( Graph )
 
