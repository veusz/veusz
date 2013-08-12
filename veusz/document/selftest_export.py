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

"""A paint engine for doing self-tests."""

from __future__ import division
from . import svg_export

class SelfTestPaintEngine(svg_export.SVGPaintEngine):
    """Paint engine class for self testing output."""

    def __init__(self, width_in, height_in):
        """Create the class, using width and height as size of canvas
        in inches."""

        svg_export.SVGPaintEngine.__init__(self, width_in, height_in)
        # ppm images are simple and should be same on all platforms
        self.imageformat = 'ppm'

    def drawTextItem(self, pt, textitem):
        """Write text directly in self test mode."""

        text = textitem.text().encode('ascii', 'xmlcharrefreplace').decode(
            'ascii')
        svg_export.SVGElement(self.celement, 'text',
                              'x="%s" y="%s" font-size="%gpt" fill="%s"' %
                              (svg_export.fltStr(pt.x()*svg_export.scale),
                               svg_export.fltStr(pt.y()*svg_export.scale),
                               textitem.font().pointSize(),
                               self.pen.color().name()),
                              text=text)

class SelfTestPaintDevice(svg_export.SVGPaintDevice):
     """Paint device for SVG paint engine."""

     def __init__(self, fileobj, width_in, height_in):
          svg_export.SVGPaintDevice.__init__(self, fileobj, width_in, height_in)
          self.engine = SelfTestPaintEngine(width_in, height_in)
