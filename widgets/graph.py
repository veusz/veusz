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

class Graph(widget.Widget):
    """Graph for containing other sorts of widgets"""

    typename='graph'

    def __init__(self, parent, name=None):
        """Initialise object and create axes."""

        widget.Widget.__init__(self, parent, name=name)

        # make axes
        ax = axis.Axis(self, name='x')
        ay = axis.Axis(self, name='y')
        ay.direction = 1

# allow users to make Graph objects
widgetfactory.thefactory.register( Graph )
 
