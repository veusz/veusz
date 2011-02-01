#    Copyright (C) 2010 Jeremy S. Sanders
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

# $Id$

"""A paint engine for doing self-tests."""

import sys
import veusz.qtall as qt4

import svg_export

class SelfTestPaintEngine(svg_export.SVGPaintEngine):
    """Paint engine class for self testing output."""

    def __init__(self, width_in, height_in):
        """Create the class, using width and height as size of canvas
        in inches."""

        svg_export.SVGPaintEngine.__init__(self, width_in, height_in)
        self.imageformat = 'bmp'

    def drawTextItem(self, pt, textitem):
        """Convert text to a path and draw it.
        """
        self.doStateUpdate()
        self.fileobj.write(
            '<text x="%s" y="%s" font-size="%gpt" fill="%s">' % (
                svg_export.fltStr(pt.x()),
                svg_export.fltStr(pt.y()),
                textitem.font().pointSize(),
                self.pen.color().name()
                )
            )
        self.fileobj.write( textitem.text().toUtf8() )
        self.fileobj.write('</text>\n')

class SelfTestPaintDevice(svg_export.SVGPaintDevice):
     """Paint device for SVG paint engine."""

     def __init__(self, fileobj, width_in, height_in):
          qt4.QPaintDevice.__init__(self)
          self.engine = SelfTestPaintEngine(width_in, height_in)
          self.fileobj = fileobj
