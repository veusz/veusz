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

# FIXME: this needs to be set somewhere when installed
_logolocation='images/logo.png'

class PlotWindow( qt.QScrollView ):
    """Class to show the plot(s) in a scrollable window."""
    
    def __init__(self, document, *args):
        """Initialise the window."""

        qt.QScrollView.__init__(self, *args)
        self.viewport().setBackgroundMode( qt.Qt.NoBackground )

        # set up so if document is modified we are notified
        self.document = document
        self.connect( self.document, qt.PYSIGNAL("sigModified"),
                      self.slotModifiedDoc )
        self.modified = True
        self.connect( self.document, qt.PYSIGNAL("sigResize"),
                      self.slotResizeDoc )

        self.setOutputSize(10, 10)

    def setOutputSize(self, xwidth_inch, ywidth_inch ):
        """Set the ouput display size."""

        self.realsize = ( xwidth_inch, ywidth_inch )

        # convert physical units into pixels
        metrics = qt.QPaintDeviceMetrics( self )
        pixwidth = int( xwidth_inch * metrics.logicalDpiX() )
        pixheight = int( ywidth_inch * metrics.logicalDpiY() )

        # set the acutal pixel size
        self.size = (pixwidth, pixheight)
        self.bufferpixmap = qt.QPixmap( *self.size )
        self.resizeContents( *self.size )

    def slotModifiedDoc(self, ismodified):
        """Called when the document has been modified."""

        if ismodified:
            self.modified = True
            # repaint window without redrawing background
            self.updateContents()

    def slotResizeDoc(self, width_inch, height_inch):
        """Called when the document is resized."""

        self.setOutputSize( width_inch, height_inch )
        self.slotModifiedDoc( True )

    def drawLogo(self, painter):
        """Draw the Veusz logo in centre of window."""

        logo = qt.QPixmap( _logolocation )
        painter.drawPixmap( self.visibleWidth()/2 - logo.width()/2,
                            self.visibleHeight()/2 - logo.height()/2,
                            logo )

    def drawContents(self, painter, clipx=-1, clipy=-1, clipw=-1, cliph=-1):
        """Called when the contents need repainting."""

        widget = self.document.getBaseWidget()

        # draw data into background pixmap if modified
        if self.modified:
            
            # fill pixmap with proper background colour
            self.bufferpixmap.fill( self.colorGroup().base() )

            # make a QPainter to draw into the buffer pixmap
            p = qt.QPainter( self.bufferpixmap )
            # draw the data into the buffer
            widget.draw( (50, 50, 250, 250), p )

            self.modified = False

        # blt the pixmap into the image
        painter.drawPixmap(0, 0, self.bufferpixmap)

        # add logo if no children
        if len(widget.getChildren()) == 0:
            self.drawLogo(painter)
        
        
