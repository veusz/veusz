#    key symbol plotting

#    Copyright (C) 2005 Jeremy S. Sanders
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
###############################################################################

# $Id$

import plotters
import widget
import graph
import widgetfactory
import setting

class Key(widget.Widget):
    """Key on graph."""

    typename = 'key'
    description = "Plot key"
    allowedparenttypes = [graph.Graph]
    allowusercreation = True

    def __init__(self, parent, name=None):
        widget.Widget.__init__(self, parent, name=name)

        s = self.settings
        s.add( setting.Text('Text',
                            descr = 'Text settings') )
        s.readDefaults()

    def draw(self, parentposn, painter):
        """Plot the key on a plotter."""

        widget.Widget.draw(self, parentposn, painter)

widgetfactory.thefactory.register( Key )
