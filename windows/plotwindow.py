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

import os
import qt

import utils

mdir = os.path.dirname(__file__)
_logolocation='%s/../images/logo.png' % mdir

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
        self.outdated = True

        self.size = (-1, -1)
        self.oldzoom = -1.
        self.zoomfactor = 1.
        self.pagenumber = 0

        # set up redrawing timer
        self.timer = qt.QTimer(self)
        self.connect( self.timer, qt.SIGNAL('timeout()'),
                      self.slotTimeout )
        self.timer.start(1000)

    def setOutputSize(self):
        """Set the ouput display size."""

        # convert distances into pixels
        painter = qt.QPainter( self )
        painter.veusz_scaling = self.zoomfactor
        size = utils.cnvtDists(self.document.getSize(), painter )
        painter.end()

        # make new buffer and resize widget
        if size != self.size:
            self.size = size
            self.bufferpixmap = qt.QPixmap( *self.size )
            self.resizeContents( *self.size )

    def setZoomFactor(self, zoomfactor):
        """Set the zoom factor of the window."""
        self.zoomfactor = float(zoomfactor)
        self.updateContents()

    def setPageNumber(self, pageno):
        """Move the the selected page."""

        # we don't need to
        if self.pagenumber == pageno and not self.outdated:
            return

        # keep within bounds
        pageno = min(pageno, self.document.getNumberPages()-1)
        pageno = max(0, pageno)

        self.pagenumber = pageno
        self.outdated = True
        self.updateContents()

    def getPageNumber(self):
        """Get the the selected page."""
        return self.pagenumber

    def slotModifiedDoc(self, ismodified):
        """Called when the document has been modified."""

        if ismodified:
            # this is picked up by a timer
            self.outdated = True

    def drawLogo(self, painter):
        """Draw the Veusz logo in centre of window."""

        logo = qt.QPixmap( _logolocation )
        painter.drawPixmap( self.visibleWidth()/2 - logo.width()/2,
                            self.visibleHeight()/2 - logo.height()/2,
                            logo )

    def slotTimeout(self):
        """Called after timer times out, to check for updates to window."""

        # no threads, so can't get interrupted here
        if self.outdated:
            self.updateContents()

    def drawContents(self, painter, clipx=0, clipy=0, clipw=-1, cliph=-1):
        """Called when the contents need repainting."""

        widget = self.document.basewidget

        # draw data into background pixmap if modified
        if self.outdated or self.zoomfactor != self.oldzoom:
            self.setOutputSize()
            
            # fill pixmap with proper background colour
            self.bufferpixmap.fill( self.colorGroup().base() )

            self.pagenumber = min( self.document.getNumberPages() - 1,
                                   self.pagenumber )
            if self.pagenumber >= 0:
                # draw the data into the buffer
                self.document.printTo( self.bufferpixmap, [self.pagenumber],
                                       self.zoomfactor )
            else:
                self.pagenumber = 0

            self.emit( qt.PYSIGNAL("sigUpdatePage"), (self.pagenumber,) )

            self.outdated = False
            self.oldzoom = self.zoomfactor

            # redraw whole window
            clipx = clipy = 0
            clipw = cliph = -1

        # blt the visible part of the pixmap into the image
        painter.drawPixmap(clipx, clipy, self.bufferpixmap,
                           clipx, clipy, clipw, cliph)

        # annoyingly we have to draw the surrounding grey ourselves
        dim = ( self.contentsX(), self.contentsY(),
                self.visibleWidth(), self.visibleHeight() )

        if dim[0]+dim[2] > self.size[0]:
            painter.fillRect(self.size[0], dim[1],
                             dim[0]+dim[2]-self.size[0], dim[3],
                             qt.QBrush( self.colorGroup().dark() ))

        if dim[1]+dim[3] > self.size[1]:
            painter.fillRect(dim[0], self.size[1],
                             dim[2], dim[1]+dim[3]-self.size[1],
                             qt.QBrush( self.colorGroup().dark() ))

        # add logo if no children
        if len(widget.children) == 0:
            self.drawLogo(painter)
        
        
