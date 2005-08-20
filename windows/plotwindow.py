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
import sys
import itertools

import qt

import setting
import dialogs.exceptiondialog
import widgets

mdir = os.path.dirname(__file__)
_logolocation='%s/../images/logo.png' % mdir

class ClickPainter(widgets.Painter):
    """A variant of a painter which checks to see whether a certain
    sized area is drawn over each time a widget is drawn. This allows
    the program to identify clicks with a widget.

    The painter monitors a certain sized region in the output pixmap
    """

    def __init__(self, pixmap, xmin, ymin, xw, yw):
        """Monitor the region from (xmin, ymin) to (xmin+xw, ymin+yw).

        pixmap is the region the painter monitors
        """
        
        widgets.Painter.__init__(self)

        self.pixmap = pixmap
        self.xmin = xmin
        self.ymin = ymin
        self.xw = xw
        self.yw = yw

        # a stack keeping track of the widgets being painted currently
        self.widgets = []
        # a stack of starting state pixmaps of the widgets
        self.pixmaps = []
        # a list of widgets which change the region
        self.foundwidgets = []

        # we hope this color isn't actually used by the user
        # if a pixel changes from this color, a widget has drawn something
        self.specialcolor = qt.QColor(254, 255, 254)
        self.pixmap.fill(self.specialcolor)
        self.begin(self.pixmap)

    def beginPaintingWidget(self, widget):
        self.widgets.append(widget)

        # make a small pixmap of the starting state of the image
        # we can compare this after the widget is painted
        pixmap = qt.QPixmap(self.xw, self.yw, 24)
        qt.copyBlt(pixmap, 0, 0, self.pixmap, self.xmin, self.ymin,
                   self.xw, self.yw)
        self.pixmaps.append(pixmap)

    def endPaintingWidget(self):
        """When a widget has finished."""

        oldpixmap = self.pixmaps.pop()
        widget = self.widgets.pop()

        # compare current pixmap for region with initial contents
        self.flush()
        newpixmap = qt.QPixmap(self.xw, self.yw, 24)
        qt.copyBlt(newpixmap, 0, 0, self.pixmap, self.xmin, self.ymin,
                   self.xw, self.yw)

        if oldpixmap.convertToImage() != newpixmap.convertToImage():
            # drawn here, so make a note
            self.foundwidgets.append(widget)

            # copy back original
            qt.copyBlt(self.pixmap, self.xmin, self.ymin,
                       oldpixmap, 0, 0, self.xw, self.yw)

    def getFoundWidget(self):
        """Return the widget lowest in the tree near the click of the mouse.
        """

        if self.foundwidgets:
            return self.foundwidgets[-1]
        else:
            return None

class PlotWindow( qt.QScrollView ):
    """Class to show the plot(s) in a scrollable window."""
    
    def __init__(self, document, *args):
        """Initialise the window."""

        qt.QScrollView.__init__(self, *args)
        self.viewport().setBackgroundMode( qt.Qt.NoBackground )

        # set up so if document is modified we are notified
        self.document = document
        self.docchangeset = -100

        self.size = (1, 1)
        self.oldzoom = -1.
        self.zoomfactor = 1.
        self.pagenumber = 0
        self.forceupdate = False
        self.setOutputSize()

        # set up redrawing timer
        self.timer = qt.QTimer(self)
        self.connect( self.timer, qt.SIGNAL('timeout()'),
                      self.slotTimeout )

        # for drag scrolling
        self.grabPos = None

        # get update period from setting database
        if 'plot_updateinterval' in setting.settingdb:
            self.interval = setting.settingdb['plot_updateinterval']
        else:
            self.interval = 1000

        if self.interval != None:
            self.timer.start(self.interval)

        # allow window to get foucs, to allow context menu
        self.setFocusPolicy(qt.QWidget.StrongFocus)

    def contentsMousePressEvent(self, event):
        """Allow user to drag window around."""

        if event.button() == qt.Qt.MidButton:
            # store start position
            event.accept()
            qt.QApplication.setOverrideCursor(qt.QCursor(qt.Qt.SizeAllCursor))
            self.grabPos = (event.globalX(), event.globalY())

    def contentsMouseMoveEvent(self, event):
        """Scroll window by how much the mouse has moved since last time."""

        if self.grabPos != None:
            event.accept()
            pos = (event.globalX(), event.globalY())
            self.scrollBy(self.grabPos[0]-pos[0], self.grabPos[1]-pos[1])
            self.grabPos = pos

    def contentsMouseReleaseEvent(self, event):
        """If the mouse button is released, check whether the mouse
        clicked on a widget, and emit a sigWidgetClicked(widget)."""

        button = event.button()
        if button == qt.Qt.LeftButton:
            # work out where the mouse clicked
            event.accept()
            x, y = event.x(), event.y()

            # now crazily draw the whole thing again
            # see which widgets change the region in the small box given below
            bufferpixmap = qt.QPixmap( *self.size )
            painter = ClickPainter(bufferpixmap, x-2, y-2, 5, 5)
            painter.veusz_scaling = self.zoomfactor

            pagenumber = min( self.document.getNumberPages() - 1,
                              self.pagenumber )
            if pagenumber >= 0:
                self.document.paintTo(painter, self.pagenumber)
            painter.end()

            widget = painter.getFoundWidget()
            if widget != None:
                # tell connected caller that widget was clicked
                self.emit( qt.PYSIGNAL('sigWidgetClicked'), (widget,) )

        elif button == qt.Qt.MidButton:
            # restore cursor to normal state after dragging
            event.accept()
            qt.QApplication.restoreOverrideCursor()
            self.grabPos = None

        else:
            qt.QScrollView.contentsMouseReleaseEvent(self, event)

    def setOutputSize(self):
        """Set the ouput display size."""

        # convert distances into pixels
        painter = widgets.Painter(self)
        painter.veusz_scaling = self.zoomfactor
        size = self.document.basewidget.getSize(painter)
        painter.end()

        # make new buffer and resize widget
        if size != self.size:
            self.size = size
            self.bufferpixmap = qt.QPixmap( *self.size )
            self.bufferpixmap.fill( self.colorGroup().base() )
            self.resizeContents( *self.size )

    def setZoomFactor(self, zoomfactor):
        """Set the zoom factor of the window."""
        self.zoomfactor = float(zoomfactor)
        self.updateContents()

    def zoomWidth(self):
        """Make the zoom factor so that the plot fills the whole width."""

        # FIXME zoomWidth/height/page routines fail to take into account
        # width of scroll bars

        width = self.visibleWidth()
        mult = width/float(self.size[0])
        self.setZoomFactor( mult * self.zoomfactor )
        
    def zoomHeight(self):
        """Make the zoom factor so that the plot fills the whole width."""

        height = self.visibleHeight()
        mult = height/float(self.size[1])
        self.setZoomFactor( mult * self.zoomfactor )

    def zoomPage(self):
        """Make the zoom factor correct to show the whole page."""

        width = self.visibleWidth()
        height = self.visibleHeight()

        multw = width/float(self.size[0])
        multh = height/float(self.size[1])
        self.setZoomFactor( min(multw, multh)*self.zoomfactor)

    def setPageNumber(self, pageno):
        """Move the the selected page."""

        # we don't need to
        if (self.pagenumber == pageno and
            self.document.changeset == self.docchangeset):
            return

        # keep within bounds
        pageno = min(pageno, self.document.getNumberPages()-1)
        pageno = max(0, pageno)

        self.pagenumber = pageno
        self.forceupdate = True
        self.updateContents()

    def getPageNumber(self):
        """Get the the selected page."""
        return self.pagenumber

    def drawLogo(self, painter):
        """Draw the Veusz logo in centre of window."""

        logo = qt.QPixmap( _logolocation )
        painter.drawPixmap( self.visibleWidth()/2 - logo.width()/2,
                            self.visibleHeight()/2 - logo.height()/2,
                            logo )

    def slotTimeout(self):
        """Called after timer times out, to check for updates to window."""

        # no threads, so can't get interrupted here
        # draw data into background pixmap if modified
        if ( self.zoomfactor != self.oldzoom or
             self.document.changeset != self.docchangeset or
             self.forceupdate ):

            self.setOutputSize()
            
            # fill pixmap with proper background colour
            self.bufferpixmap.fill( self.colorGroup().base() )

            self.pagenumber = min( self.document.getNumberPages() - 1,
                                   self.pagenumber )
            if self.pagenumber >= 0:
                # draw the data into the buffer
                # errors cause an exception window to pop up
                try:
                    self.document.printTo( self.bufferpixmap,
                                           [self.pagenumber],
                                           self.zoomfactor )
                except Exception:
                    dialogs.exceptiondialog.showException(sys.exc_info())
                    
            else:
                self.pagenumber = 0

            self.emit( qt.PYSIGNAL("sigUpdatePage"), (self.pagenumber,) )

            self.oldzoom = self.zoomfactor
            self.forceupdate = False
            self.docchangeset = self.document.changeset

            self.updateContents()
            
    def drawContents(self, painter, clipx=0, clipy=0, clipw=-1, cliph=-1):
        """Called when the contents need repainting."""

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
        widget = self.document.basewidget
        if len(widget.children) == 0:
            self.drawLogo(painter)

    def contextMenuEvent(self, event):
        """A context to change update periods, or disable updates."""

        popup = qt.QPopupMenu(self)
        popup.setCheckable(True)

        # option to force an update
        popup.insertItem('Force update', 1)
        popup.insertItem('Disable updates', 2)
        if self.interval == None:
            popup.setItemChecked(2, True)
        popup.insertSeparator()

        # populate menu with update periods
        intervals = [100, 250, 500, 1000, 2000, 5000, 10000]
        for i, id in itertools.izip(intervals, itertools.count()):
            popup.insertItem('Update every %gs' % (i * 0.001), 100+id)
            if i == self.interval:
                popup.setItemChecked(100+id, True)

        # show menu
        ret = popup.exec_loop( event.globalPos() )

        if ret == 1:
            # force an update
            self.docchangeset = -100
            self.slotTimeout()
        elif ret == 2:
            # stop updates
            self.interval = None
            self.timer.stop()
        elif ret >= 100:
            # change interval to one selected
            self.interval = intervals[ret-100]
            self.timer.changeInterval(self.interval)
            # start timer if it was stopped
            if not self.timer.isActive():
                self.timer.start()

        # update setting database
        if ret > 0:
            setting.settingdb['plot_updateinterval'] = self.interval
