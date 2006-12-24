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

"""Widget that represents a page in the document."""

import veusz.qtall as qt4

import veusz.document as document

import widget
import root

class Page(widget.Widget):
    """A class for representing a page of plotting."""

    typename='page'
    allowusercreation = True
    allowedparenttypes = [root.Root]
    description='Blank page'

    def __init__(self, parent, name=None):
        """Initialise object."""

        widget.Widget.__init__(self, parent, name=name)

    def draw(self, parentposn, painter, outerbounds=None):
        """Draw the plotter. Clip graph inside bounds."""

        # document should pass us the page bounds
        x1, y1, x2, y2 = parentposn

        if self.settings.hide:
            bounds = self.computeBounds(parentposn, painter)
            return bounds

        painter.beginPaintingWidget(self, parentposn)
        painter.save()
        painter.veusz_page_size = (x2-x1, y2-y1)
        painter.setClipRect( qt4.QRect(x1, y1, x2-x1, y2-y1) )
        bounds = widget.Widget.draw(self, parentposn, painter,
                                    parentposn)
        painter.restore()
        painter.endPaintingWidget()

        return bounds

# allow the factory to instantiate this
document.thefactory.register( Page )
    
