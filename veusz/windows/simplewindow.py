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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

from __future__ import division
from .. import qtall as qt4

from .. import document
from .. import dataimport
from . import plotwindow

"""
A simple window class for wrapping a plotwindow
"""

class SimpleWindow(qt4.QMainWindow):
    """ The main window class for the application."""

    def __init__(self, title, doc=None):
        qt4.QMainWindow.__init__(self)

        self.setWindowTitle(title)

        self.document = doc
        if not doc:
            self.document = document.Document()

        self.plot = plotwindow.PlotWindow(self.document, self)
        self.toolbar = None

        self.setCentralWidget( self.plot )

    def enableToolbar(self, enable=True):
        """Enable or disable the zoom toolbar in this window."""

        if self.toolbar is None and enable:
            self.toolbar = self.plot.createToolbar(self, None)
            self.toolbar.show()

        if self.toolbar is not None and not enable:
            self.toolbar.close()
            self.toolbar = None
            
    def setZoom(self, zoom):
        """Zoom(zoom)

        Set the plot zoom level:
        This is a number to for the zoom from 1:1 or
        'page': zoom to page
        'width': zoom to fit width
        'height': zoom to fit height
        """
        if zoom == 'page':
            self.plot.slotViewZoomPage()
        elif zoom == 'width':
            self.plot.slotViewZoomWidth()
        elif zoom == 'height':
            self.plot.slotViewZoomHeight()
        else:
            self.plot.setZoomFactor(zoom)

    def setAntiAliasing(self, ison):
        """AntiAliasing(ison)

        Switches on or off anti aliasing in the plot."""

        self.plot.antialias = ison
        self.plot.actionForceUpdate()
