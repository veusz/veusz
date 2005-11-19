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
import os.path
import sys
import itertools

import qt
import numarray as N

import setting
import dialogs.exceptiondialog
import widgets
import action

class PointPainter(widgets.Painter):
    """A simple painter variant which works out the last widget
    to overlap with the point specified."""

    def __init__(self, pixmap, x, y):
        """Watch the point x, y."""
        widgets.Painter.__init__(self)
        self.x = x
        self.y = y
        self.widget = None
        self.bounds = {}

        self.pixmap = pixmap
        self.begin(pixmap)

    def beginPaintingWidget(self, widget, bounds):

        if (isinstance(widget, widgets.Graph) and
            bounds[0] <= self.x and bounds[1] <= self.y and
            bounds[2] >= self.x and bounds[3] >= self.y):
            self.widget = widget

        # record bounds of each widget
        self.bounds[widget] = bounds

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

    def beginPaintingWidget(self, widget, bounds):
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

        # show splash logo until timer runs out (3s)
        self.showlogo = True
        qt.QTimer.singleShot(3000, self.slotSplashDisable)

        # set up so if document is modified we are notified
        self.document = document
        self.docchangeset = -100

        self.size = (1, 1)
        self.oldzoom = -1.
        self.zoomfactor = 1.
        self.pagenumber = 0
        self.forceupdate = False
        self.setOutputSize()

        # mode for clicking
        self.clickmode = 'select'
        self.currentclickmode = None

        # set up redrawing timer
        self.timer = qt.QTimer(self)
        self.connect( self.timer, qt.SIGNAL('timeout()'),
                      self.slotTimeout )

        # for drag scrolling
        self.grabPos = None
        self.scrolltimer = qt.QTimer(self)

        # for turning clicking into scrolling after a period
        self.connect( self.scrolltimer, qt.SIGNAL('timeout()'),
                      self.slotBecomeScrollClick )

        # get update period from setting database
        if 'plot_updateinterval' in setting.settingdb:
            self.interval = setting.settingdb['plot_updateinterval']
        else:
            self.interval = 1000

        if self.interval != None:
            self.timer.start(self.interval)

        # allow window to get foucs, to allow context menu
        self.setFocusPolicy(qt.QWidget.StrongFocus)

        # optional view toolbar
        self.viewtoolbar = None
        self.viewactions = None

    def createToolbar(self, parent, menu=None):
        """Make a view toolbar, and optionally update menu."""

        self.zoomtoolbar = qt.QToolBar(parent, "viewtoolbar")
        self.zoomtoolbar.setLabel("View toolbar - Veusz")

        items = [
            ('viewzoomin', 'Zoom into the plot', 'Zoom &In', 'view',
             self.slotViewZoomIn, 'stock-zoom-in.png', False, 'Ctrl++'),
            ('viewzoomout', 'Zoom out of the plot', 'Zoom &Out', 'view',
             self.slotViewZoomOut, 'stock-zoom-out.png', False, 'Ctrl+-'),
            ('viewzoom11', 'Restore plot to natural size', 'Zoom 1:1', 'view',
             self.slotViewZoom11, 'stock-zoom-100.png', False, 'Ctrl+1'),
            ('viewzoomwidth', 'Zoom plot to show whole width',
             'Zoom to width', 'view', self.slotViewZoomWidth,
             'stock_zoom-page-width.png', False, ''),
            ('viewzoomheight', 'Zoom plot to show whole height',
             'Zoom to height', 'view', self.slotViewZoomHeight,
             'stock_zoom-page-height.png', False, ''),
            ('viewzoompage', 'Zoom plot to show whole page',
             'Zoom to page', 'view', self.slotViewZoomPage,
             'stock_zoom-page.png', False, ''),
            ('view',),
            ('viewprevpage', 'Move to the previous page', '&Previous page',
             'view', self.slotViewPreviousPage, 'stock_previous-page.png',
             True, 'Ctrl+PgUp'),
            ('viewnextpage', 'Move to the next page', '&Next page',
             'view', self.slotViewNextPage, 'stock_next-page.png',
             True, 'Ctrl+PgDown'),
            ('viewzoomgraph', 'Zoom into graph', 'Zoom graph',
             'view', self.slotViewZoomGraph, 'zoom_graph.png',
             True, '')
            ]

        menus = None
        if menu != None:
            menus = {}
            menus['view'] = menu

        self.viewactions = action.populateMenuToolbars(items, self.zoomtoolbar,
                                                       menus)
                                                   
        zoomtb = qt.QToolButton(self.zoomtoolbar)
        zoomicon = os.path.join(os.path.dirname(__file__), 'icons',
                                'zoom-options.png')
        zoomtb.setIconSet(qt.QIconSet( qt.QPixmap(zoomicon) ))

        # drop down zoom button on toolbar
        zoompop = qt.QPopupMenu(zoomtb)
        for act in ('viewzoomin', 'viewzoomout', 'viewzoom11',
                    'viewzoomwidth', 'viewzoomheight', 'viewzoompage'):
            self.viewactions[act].addTo(zoompop)
        zoomtb.setPopup(zoompop)
        zoomtb.setPopupDelay(0)

    def contentsMousePressEvent(self, event):
        """Allow user to drag window around."""

        if event.button() == qt.Qt.LeftButton:

            self.grabPos = (event.globalX(), event.globalY())
            if self.clickmode == 'select':
                # we set this to true unless the timer runs out (400ms),
                # then it becomes a scroll click
                # scroll clicks drag the window around, and selecting clicks
                # select widgets!
                self.scrolltimer.start(400, True)

            elif self.clickmode == 'scroll':
                qt.QApplication.setOverrideCursor(
                    qt.QCursor(qt.Qt.SizeAllCursor))

            elif self.clickmode == 'graphzoom':
                qt.QApplication.setOverrideCursor(
                    qt.QCursor(qt.Qt.CrossCursor))
                self._drawZoomRect(self.grabPos)

            # record what mode we were clicked in
            self.currentclickmode = self.clickmode

    def _drawZoomRect(self, pos):
        """Draw a dotted rectangle xored."""

        # convert global coordinates of edges into viewport coordinates
        self._currentzoomrect = pos
        minx = min(self.grabPos[0], pos[0])
        maxx = max(self.grabPos[0], pos[0])
        miny = min(self.grabPos[1], pos[1])
        maxy = max(self.grabPos[1], pos[1])
        w = maxx - minx + 1
        h = maxy - miny + 1

        # draw the rectangle on the viewport
        view = self.viewport()
        pt = view.mapFromGlobal( qt.QPoint(minx, miny) )
        painter = qt.QPainter(view, True)
        painter.setPen(qt.QPen(qt.QColor('black'), 0, qt.Qt.DotLine))
        painter.drawRect(pt.x(), pt.y(), w, h)

    def _hideZoomRect(self):
        """Remove the zoom rectangle painted by _drawZoomRect."""

        # convert bounds of old zoom rect (in global coords)
        # to contents coordinates
        pos = self._currentzoomrect
        minx = min(self.grabPos[0], pos[0])
        maxx = max(self.grabPos[0], pos[0])
        miny = min(self.grabPos[1], pos[1])
        maxy = max(self.grabPos[1], pos[1])
        w = maxx - minx + 1
        h = maxy - miny + 1

        view = self.viewport()
        pt1 = view.mapFromGlobal(qt.QPoint(minx, miny))
        pt2 = view.mapFromGlobal(qt.QPoint(maxx, maxy))
        pt1x, pt1y = self.viewportToContents(pt1.x(), pt1.y())
        pt2x, pt2y = self.viewportToContents(pt2.x(), pt2.y())

        # repaint the contents along the edges of the zoom rect
        self.repaintContents(pt1x, pt1y, w, 1, False)
        self.repaintContents(pt1x, pt1y, 1, h, False)
        self.repaintContents(pt1x, pt2y, w, 1, False)
        self.repaintContents(pt2x, pt1y, 1, h, False)

    def doZoomRect(self):
        """Take the zoom rectangle drawn by the user and do the zooming.

        This is pretty messy - first we have to work out the graph associated
        to the first point

        Then we have to iterate over each of the plotters, identify their
        axes, and change the range of the axes to match the screen region
        selected.
        """

        pt1 = self.viewport().mapFromGlobal(qt.QPoint(*self.grabPos))
        pt2 = self.viewport().mapFromGlobal(qt.QPoint(*self._currentzoomrect))
        pt1 = self.viewportToContents(pt1)
        pt2 = self.viewportToContents(pt2)

        # try to work out in which widget the first point is in
        bufferpixmap = qt.QPixmap( *self.size )
        painter = PointPainter(bufferpixmap, pt1.x(), pt1.y())
        painter.veusz_scaling = self.zoomfactor
        pagenumber = min( self.document.getNumberPages() - 1,
                          self.pagenumber )
        if pagenumber >= 0:
            self.document.paintTo(painter, self.pagenumber)
        painter.end()

        # get widget
        widget = painter.widget
        if widget != None:

            # convert points on plotter to points on axis for each axis
            xpts = N.array( [pt1.x(), pt2.x()] )
            ypts = N.array( [pt1.y(), pt2.y()] )

            # iterate over children, to look for plotters
            for c in widget.children:
                if isinstance(c, widgets.GenericPlotter):

                    # get axes associated with plotter
                    axes = c.parent.getAxes( (c.settings.xAxis,
                                              c.settings.yAxis) )

                    # iterate over each, and update the ranges
                    for axis in [a for a in axes if a != None]:
                        s = axis.settings
                        if s.direction == 'horizontal':
                            p = xpts
                        else:
                            p = ypts

                        # convert points on plotter to axis coordinates
                        # FIXME: Need To Trap Conversion Errors!
                        r = axis.plotterToGraphCoords(painter.bounds[axis], p)

                        # invert if min and max are inverted
                        if r[1] < r[0]:
                            r[1], r[0] = r[0], r[1]

                        # actually set the axis
                        if s.min != r[0]:
                            s.min = r[0]
                        if s.max != r[1]:
                            s.max = r[1]

    def slotBecomeScrollClick(self):
        """If the click is still down when this timer is reached then
        we turn the click into a scrolling click."""

        if self.currentclickmode == 'select':
            qt.QApplication.setOverrideCursor(qt.QCursor(qt.Qt.SizeAllCursor))
            self.currentclickmode = 'scroll'

    def contentsMouseMoveEvent(self, event):
        """Scroll window by how much the mouse has moved since last time."""

        if self.currentclickmode == 'scroll':
            event.accept()
            pos = (event.globalX(), event.globalY())
            self.scrollBy(self.grabPos[0]-pos[0], self.grabPos[1]-pos[1])
            self.grabPos = pos
        elif self.currentclickmode == 'graphzoom':
            # get rid of current rectangle
            self._hideZoomRect()
            self._drawZoomRect((event.globalX(), event.globalY()))

    def contentsMouseReleaseEvent(self, event):
        """If the mouse button is released, check whether the mouse
        clicked on a widget, and emit a sigWidgetClicked(widget)."""

        if event.button() == qt.Qt.LeftButton:
            event.accept()
            self.scrolltimer.stop()
            if self.currentclickmode == 'select':
                # work out where the mouse clicked and choose widget
                self.locateClickWidget(event.x(), event.y())
            elif self.currentclickmode == 'scroll':
                # return the cursor to normal after scrolling
                qt.QApplication.restoreOverrideCursor()
            elif self.currentclickmode == 'graphzoom':
                self._hideZoomRect()
                qt.QApplication.restoreOverrideCursor()
                self.doZoomRect()
        else:
            qt.QScrollView.contentsMouseReleaseEvent(self, event)

    def locateClickWidget(self, x, y):
        """Work out which widget was clicked, and if necessary send
        a sigWidgetClicked(widget) signal."""
        
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

    def slotSplashDisable(self):
        """Disable drawing the splash logo."""
        self.showlogo = False
        self.updateContents()

    def drawLogo(self, painter):
        """Draw the Veusz logo in centre of window."""

        logolocation = os.path.join(os.path.dirname(__file__),
                                    '..', 'images', 'logo.png')
        logo = qt.QPixmap( logolocation )
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
            self.updatePageToolbar()

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
        if len(widget.children) == 0 and self.showlogo:
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

    def setZoomFactor(self, zoomfactor):
        """Set the zoom factor of the window."""
        self.zoomfactor = float(zoomfactor)
        self.updateContents()

    def slotViewZoomIn(self):
        """Zoom into the plot."""
        self.setZoomFactor(self.zoomfactor * N.sqrt(2.))

    def slotViewZoomOut(self):
        """Zoom out of the plot."""
        self.setZoomFactor(self.zoomfactor / N.sqrt(2.))

    def slotViewZoomWidth(self):
        """Make the zoom factor so that the plot fills the whole width."""

        # FIXME zoomWidth/height/page routines fail to take into account
        # width of scroll bars

        width = self.visibleWidth()
        mult = width/float(self.size[0])
        self.setZoomFactor(self.zoomfactor * mult)
        
    def slotViewZoomHeight(self):
        """Make the zoom factor so that the plot fills the whole width."""

        height = self.visibleHeight()
        mult = height/float(self.size[1])
        self.setZoomFactor(self.zoomfactor * mult)

    def slotViewZoomPage(self):
        """Make the zoom factor correct to show the whole page."""

        width = self.visibleWidth()
        height = self.visibleHeight()

        multw = width/float(self.size[0])
        multh = height/float(self.size[1])
        self.setZoomFactor(self.zoomfactor * min(multw, multh))

    def slotViewZoom11(self):
        """Restore the zoom to 1:1"""
        self.setZoomFactor(1.)

    def slotViewPreviousPage(self):
        """View the previous page."""
        self.setPageNumber( self.pagenumber - 1 )
 
    def slotViewNextPage(self):
        """View the next page."""
        self.setPageNumber( self.pagenumber + 1 )

    def slotViewZoomGraph(self):
        """Zoom into graph."""

    def updatePageToolbar(self):
        """Update page number when the plot window says so."""

        # disable previous and next page actions
        if self.viewactions != None:
            np = self.document.getNumberPages()
            self.viewactions['viewprevpage'].setEnabled(self.pagenumber != 0)
            self.viewactions['viewnextpage'].setEnabled(self.pagenumber < np-1)
