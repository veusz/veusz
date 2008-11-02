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
##############################################################################

# $Id$

import veusz.qtall as qt4

import veusz.document as document
import plotwindow

"""
A simple window class for wrapping a plotwindow
"""

class SimpleWindow(qt4.QMainWindow):
    """ The main window class for the application."""

    def __init__(self, title):
        qt4.QMainWindow.__init__(self)
        self.setWindowTitle(title)

        self.document = document.Document()

        self.plot = plotwindow.PlotWindow(self.document, self)
        self.plotzoom = 0
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
            
