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
import controlgraph

class Root(widget.Widget):
    """Root widget class for plotting the document."""

    typename='document'
    allowusercreation = False
    allowedparenttypes = [None]

    def __init__(self, parent, name=None, document=None):
        """Initialise object."""

        widget.Widget.__init__(self, parent, name=name)
        s = self.settings

        # don't want user to be able to hide entire document
        s.remove('hide')

        s.add( setting.Distance('width', '15cm',
                                descr='Width of the pages',
                                usertext='Page width',
                                formatting=True) )
        s.add( setting.Distance('height', '15cm',
                                descr='Height of the pages',
                                usertext='Page height',
                                formatting=True) )
        s.add( setting.StyleSheet(descr='Master settings for document',
                                  usertext='Style sheet') )
        self.document = document

        if type(self) == Root:
            self.readDefaults()
            
    def getSize(self, painter):
        """Get dimensions of widget in painter coordinates."""
        return ( self.settings.get('width').convert(painter),
                 self.settings.get('height').convert(painter) )
            
    def draw(self, painter, pagenum):
        """Draw the page requested on the painter."""

        xw, yw = self.getSize(painter)
        posn = [0, 0, xw, yw]
        painter.beginPaintingWidget(self, posn)
        page = self.children[pagenum]
        page.draw( posn, painter )

        self.controlgraphitems = [
            controlgraph.ControlMarginBox(self, posn,
                                          [-10000, -10000,
                                            10000,  10000],
                                          painter,
                                          ismovable = False)
            ]

        painter.endPaintingWidget()

    def updateControlItem(self, cgi):
        """Graph resized or moved - call helper routine to move self."""

        s = self.settings

        # get margins in pixels
        width = cgi.posn[2] - cgi.posn[0]
        height = cgi.posn[3] - cgi.posn[1]

        # set up fake painter containing veusz scalings
        fakepainter = qt4.QPainter()
        fakepainter.veusz_page_size = cgi.page_size
        fakepainter.veusz_scaling = cgi.scaling
        fakepainter.veusz_pixperpt = cgi.pixperpt

        # convert to physical units
        width = s.get('width').convertInverse(width, fakepainter)
        height = s.get('height').convertInverse(height, fakepainter)

        # modify widget margins
        operations = (
            document.OperationSettingSet(s.get('width'), width),
            document.OperationSettingSet(s.get('height'), height),
            )
        self.document.applyOperation(
            document.OperationMultiple(operations, descr='change page size'))

# allow the factory to instantiate this
document.thefactory.register( Root )
