# plotwindow.py
# the main window for showing plots
 
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

import qt

class PlotWindow(qt.QScrollView):
    """ Class to show the plot(s)."""
    
    def __init__(self, document, *args):
        """Initialise the window."""

        qt.QScrollView.__init__(self, *args)
        self.viewport().setBackgroundMode( qt.Qt.NoBackground )

        self.document = document
        document.addModifiedCallback( self.__callbackModifiedDoc )
        self.modified = True

        self.setOutputSize(1000, 1000)

    def setOutputSize(self, xwidth, ywidth):
        """Set the ouput display size."""

        self.size = (xwidth, ywidth)
        self.bufferpixmap = qt.QPixmap( *self.size )
        self.resizeContents( *self.size )

    def __callbackModifiedDoc(self, ismodified):
        """Called when the document has been modified."""
        if ismodified:
            self.modified = True

    def drawContents(self, painter, clipx=-1, clipy=-1, clipw=-1, cliph=-1):
        """Called when the contents need repainting."""

        # draw data into background pixmap if modified
        if self.modified:
            self.bufferpixmap.fill( qt.Qt.white )
            p = qt.QPainter( self.bufferpixmap )
            widget = self.document.getBaseWidget()
            widget.draw( (50, 50, 250, 250), p )
            self.modified = False

        # blt the pixmap into the image
        painter.drawPixmap(0, 0, self.bufferpixmap)
        
        
        
