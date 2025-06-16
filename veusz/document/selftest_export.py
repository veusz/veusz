#    Copyright (C) 2010 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This file is part of Veusz.
#
#    Veusz is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    Veusz is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################

"""A paint engine for doing self-tests."""

from . import svg_export

class SelfTestPaintEngine(svg_export.SVGPaintEngine):
    """Paint engine class for self testing output."""

    def __init__(self):
        svg_export.SVGPaintEngine.__init__(self)
        # ppm images are simple and should be same on all platforms
        self.imageformat = 'ppm'

    def drawTextItem(self, pt, textitem):
        """Write text directly in self test mode."""

        text = textitem.text().encode('ascii', 'xmlcharrefreplace').decode(
            'ascii')
        svg_export.SVGElement(
            self.celement, 'text',
            'x="%s" y="%s" font-size="%gpt" fill="%s"' % (
                svg_export.fltStr(pt.x()),
                svg_export.fltStr(pt.y()),
                textitem.font().pointSize(),
                self.pen.color().name()
            ),
            text=text
        )

class SelfTestPaintDevice(svg_export.SVGPaintDevice):
     """Paint device for SVG paint engine.

     Note: this device is different to SVGPaintDevice because it
     switches scaling to 1 by default.
     """

     def __init__(self, fileobj, width_in, height_in, dpi=90):
         """Initialise with output file, and dimensions in inches."""
         svg_export.SVGPaintDevice.__init__(
             self, fileobj, width_in, height_in, dpi=dpi, scale=1)
         self.engine = SelfTestPaintEngine()
