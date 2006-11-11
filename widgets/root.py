# root.py
# Represents the root widget for plotting the document

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

import veusz.document as document
import veusz.setting as setting

import widget

class Root(widget.Widget):
    """Root widget class for plotting the document."""

    typename='document'
    allowusercreation = False
    allowedparenttypes = [None]

    def __init__(self, parent, name=None, document=None):
        """Initialise object."""

        widget.Widget.__init__(self, parent, name=name)
        s = self.settings
        s.add( setting.Distance('width', '15cm',
                                descr='Width of the pages',
                                usertext='Page width') )
        s.add( setting.Distance('height', '15cm',
                                descr='Height of the pages',
                                usertext='Page height') )
        s.add( setting.StyleSheet(descr='Master settings for document',
                                  usertext='Style sheet') )
        self.document = document

        if type(self) == Root:
            self.readDefaults()
            
    def getSize(self, painter):
        """Get dimensions of widget in painter coordinates."""
        return ( self.settings.get('width').convert(painter),
                 self.settings.get('height').convert(painter) )
            
    def draw(self, parentposn, painter, outerbounds = None):
        """Draw the plotter. Clip graph inside bounds."""

        x1, y1, x2, y2 = parentposn

        painter.beginPaintingWidget(self, parentposn)
        painter.save()
        painter.setClipRect( qt4.QRect(x1, y1, x2-x1+1, y2-y1+1) )
        bounds = widget.Widget.draw(self, parentposn, painter,
                                    outerbounds = parentposn)
        painter.restore()
        painter.endPaintingWidget()

        return bounds

# allow the factory to instantiate this
document.thefactory.register( Root )
