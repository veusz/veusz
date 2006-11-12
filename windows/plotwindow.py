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

import veusz.qtall as qt4
import numarray as N

import veusz.setting as setting
import veusz.dialogs.exceptiondialog as exceptiondialog
import veusz.widgets as widgets
import veusz.document as document
import veusz.utils as utils

import action

class PointPainter(document.Painter):
    """A simple painter variant which works out the last widget
    to overlap with the point specified."""

    def __init__(self, pixmap, x, y):
        """Watch the point x, y."""
        document.Painter.__init__(self)
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

class ClickPainter(document.Painter):
    """A variant of a painter which checks to see whether a certain
    sized area is drawn over each time a widget is drawn. This allows
    the program to identify clicks with a widget.

    The painter monitors a certain sized region in the output pixmap
    """

    def __init__(self, pixmap, xmin, ymin, xw, yw):
        """Monitor the region from (xmin, ymin) to (xmin+xw, ymin+yw).

        pixmap is the region the painter monitors
        """
        
        document.Painter.__init__(self)

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
        self.specialcolor = qt4.QColor(254, 255, 254)
        self.pixmap.fill(self.specialcolor)
        self.begin(self.pixmap)

    def beginPaintingWidget(self, widget, bounds):
        self.widgets.append(widget)

        # make a small pixmap of the starting state of the image
        # we can compare this after the widget is painted
        pixmap = self.pixmap.copy(self.xmin, self.ymin, self.xw, self.yw)
        self.pixmaps.append(pixmap)

    def endPaintingWidget(self):
        """When a widget has finished."""

        oldpixmap = self.pixmaps.pop()
        widget = self.widgets.pop()

        # compare current pixmap for region with initial contents
        # hope this is not needed
        #self.flush()
        newpixmap = self.pixmap.copy(self.xmin, self.ymin, self.xw, self.yw)

        if oldpixmap.toImage() != newpixmap.toImage():
            # drawn here, so make a note
            self.foundwidgets.append(widget)

            # copy back original
            self.drawPixmap(qt4.QRect(self.xmin, self.ymin, self.xw, self.yw),
                            oldpixmap,
                            qt4.QRect(0, 0, self.xw, self.yw))

    def getFoundWidget(self):
        """Return the widget lowest in the tree near the click of the mouse.
        """

        if self.foundwidgets:
            return self.foundwidgets[-1]
        else:
            return None

class DisplayWidget(qt4.QLabel):
    """A widget for displaying the plot, embedded in scrollable area."""
    
    def __init__(self, *args):
        qt4.QLabel.__init__(self, *args)

        # no zoom rectangle initially
        self._zoomrect = None

        # show splash logo until timer runs out (3s)
        self._showlogo = True
        qt4.QTimer.singleShot(3000, self.slotSplashDisable)

    def slotSplashDisable(self):
        """Disable drawing the splash logo."""
        self._showlogo = False
        self.update()

    def paintEvent(self, event):
        """Paint display widget."""

        qt4.QLabel.paintEvent(self, event)

        if self._zoomrect:
            # draw zoom rectangle if any shown
            painter = qt4.QPainter(self)
            painter.setPen(qt4.QPen(qt4.QColor('black'), 0, qt4.Qt.DotLine))
            painter.drawRect(*self._zoomrect)

        if self._showlogo:
            # show logo until timer runs out
            painter = qt4.QPainter(self)
            logo = action.getPixmap('logo.png')
            painter.drawPixmap(self.width()/2 - logo.width()/2,
                               self.height()/2 - logo.height()/2,
                               logo)

    def drawRect(self, pt1, pt2):
        """Draw a zoom rectangle from QPoint pt1 to pt2."""

        if self._zoomrect:
            self.hideRect()

        minx = min(pt1.x(), pt2.x())
        maxx = max(pt1.x(), pt2.x())
        miny = min(pt1.y(), pt2.y())
        maxy = max(pt1.y(), pt2.y())
        w = maxx - minx
        h = maxy - miny
        self._zoomrect = (minx, miny, w, h)
        self._repaintRect(self._zoomrect)

    def _repaintRect(self, rect):
        """Repaint rectangle region."""

        minx, miny, w, h = rect
        maxx = minx + w
        maxy = miny + h
        self.repaint(minx, miny, w, 1)
        self.repaint(minx, maxy, w, 1)
        self.repaint(maxx, miny, 1, h)
        self.repaint(minx, miny, 1, h)

    def hideRect(self):
        """Hide any shown zoom rectangle."""

        if self._zoomrect:
            old = self._zoomrect
            self._zoomrect = None
            self._repaintRect(old)

class PlotWindow( qt4.QScrollArea ):
    """Class to show the plot(s) in a scrollable window."""

    def __init__(self, document, parent, menu=None):
        """Initialise the window.

        menu gives a menu to add any menu items to
        """

        qt4.QScrollArea.__init__(self, parent)
        self.label = DisplayWidget()
        self.setWidget(self.label)
        self.setBackgroundRole(qt4.QPalette.Dark)
        self.label.setSizePolicy(qt4.QSizePolicy.Fixed, qt4.QSizePolicy.Fixed)

        # set up so if document is modified we are notified
        self.document = document
        self.docchangeset = -100

        self.size = (1, 1)
        self.oldzoom = -1.
        self.zoomfactor = 1.
        self.pagenumber = 0
        self.forceupdate = False

        # work out dpi
        self.widgetdpi = self.logicalDpiY()

        # convert size to pixels
        self.setOutputSize()

        # mode for clicking
        self.clickmode = 'select'
        self.currentclickmode = None

        # set up redrawing timer
        self.timer = qt4.QTimer(self)
        self.connect( self.timer, qt4.SIGNAL('timeout()'),
                      self.slotTimeout )

        # for drag scrolling
        self.grabPos = None
        self.scrolltimer = qt4.QTimer(self)
        self.scrolltimer.setSingleShot(True)

        # for turning clicking into scrolling after a period
        self.connect( self.scrolltimer, qt4.SIGNAL('timeout()'),
                      self.slotBecomeScrollClick )

        # get update period from setting database
        self.interval = setting.settingdb['plot_updateinterval']

        # load antialias settings
        self.antialias = setting.settingdb['plot_antialias']

        if self.interval > 0:
            self.timer.start(self.interval)

        # allow window to get foucs, to allow context menu
        self.setFocusPolicy(qt4.Qt.StrongFocus)

        # create toolbar in main window (urgh)
        self.createToolbar(parent, menu)

        # make the context menu object
        self._constructContextMenu()

    def showToolbar(self, show=True):
        """Show or hide toolbar"""
        self.viewtoolbar.setVisible(show)

    def createToolbar(self, parent, menu=None):
        """Make a view toolbar, and optionally update menu."""

        self.viewtoolbar = qt4.QToolBar("View toolbar - Veusz", parent)
        self.viewtoolbar.setObjectName('veuszviewtoolbar')
        self.viewtoolbar.hide()
        parent.addToolBar(qt4.Qt.TopToolBarArea, self.viewtoolbar)

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
            ('viewselect', 'Select items from the graph or scroll',
             'Select items or scroll', 'view', None,
             'pointer.png', True, ''),
            ('viewzoomgraph', 'Zoom into graph', 'Zoom graph',
             'view', None, 'zoom_graph.png',
             True, '')
            ]

        menus = None
        if menu is not None:
            menus = {}
            menus['view'] = menu

        self.viewactions = action.populateMenuToolbars(items, self.viewtoolbar,
                                                       menus)

        # a button for the zoom icon
        zoomtb = qt4.QToolButton(self.viewtoolbar)
        zoomtb.setIcon( action.getIcon('zoom-options.png') )

        # drop down zoom button on toolbar
        zoompop = qt4.QMenu(zoomtb)
        for act in ('viewzoomin', 'viewzoomout', 'viewzoom11',
                    'viewzoomwidth', 'viewzoomheight', 'viewzoompage'):
            zoompop.addAction(self.viewactions[act])
        zoomtb.setMenu(zoompop)
        zoomtb.setPopupMode(qt4.QToolButton.InstantPopup)
        self.viewtoolbar.addWidget(zoomtb)

        # define action group for various different selection models
        g = self.selectactiongrp = qt4.QActionGroup(self)
        g.setExclusive(True)
        for a in [self.viewactions[i] for i in
                  ('viewselect', 'viewzoomgraph')]:
            a.setActionGroup(g)
            a.setCheckable(True)
        self.viewactions['viewselect'].setChecked(True)
        self.connect(g, qt4.SIGNAL('triggered(QAction*)'), self.slotSelectMode)

        return self.viewtoolbar

    def doZoomRect(self, endpos):
        """Take the zoom rectangle drawn by the user and do the zooming.
        endpos is a QPoint end point

        This is pretty messy - first we have to work out the graph associated
        to the first point

        Then we have to iterate over each of the plotters, identify their
        axes, and change the range of the axes to match the screen region
        selected.
        """

        # get points corresponding to corners of rectangle
        pt1 = qt4.QPoint(self.grabPos)
        pt2 = qt4.QPoint(endpos)

        # work out whether it's worthwhile to zoom: only zoom if there
        # are >=5 pixels movement
        if abs((pt2-pt1).x()) < 10 or abs((pt2-pt1).y()) < 10:
            return

        # try to work out in which widget the first point is in
        bufferpixmap = qt4.QPixmap( *self.size )
        painter = PointPainter(bufferpixmap, pt1.x(), pt1.y())
        pagenumber = min( self.document.getNumberPages() - 1,
                          self.pagenumber )
        if pagenumber >= 0:
            self.document.paintTo(painter, self.pagenumber,
                                  scaling=self.zoomfactor, dpi=self.widgetdpi)
        painter.end()

        # get widget
        widget = painter.widget
        if widget is None:
            return
        
        # convert points on plotter to points on axis for each axis
        xpts = N.array( [pt1.x(), pt2.x()] )
        ypts = N.array( [pt1.y(), pt2.y()] )

        # build up operation list to do zoom
        operations = []
        
        # iterate over children, to look for plotters
        for c in [i for i in widget.children if
                  isinstance(i, widgets.GenericPlotter)]:

            # get axes associated with plotter
            axes = c.parent.getAxes( (c.settings.xAxis,
                                      c.settings.yAxis) )

            # iterate over each, and update the ranges
            for axis in [a for a in axes if a is not None]:
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

                # build up operations to change axis
                if s.min != r[0]:
                    operations.append( document.OperationSettingSet(s.get('min'),
                                                                    r[0]) )
                if s.max != r[1]:
                    operations.append( document.OperationSettingSet(s.get('max'),
                                                                    r[1]) )


        # finally change the axes
        self.document.applyOperation( document.OperationMultiple(operations, descr='zoom axes') )
                    
    def slotBecomeScrollClick(self):
        """If the click is still down when this timer is reached then
        we turn the click into a scrolling click."""

        if self.currentclickmode == 'select':
            qt4.QApplication.setOverrideCursor(qt4.QCursor(qt4.Qt.SizeAllCursor))
            self.currentclickmode = 'scroll'

    def mousePressEvent(self, event):
        """Allow user to drag window around."""

        if event.button() == qt4.Qt.LeftButton:

            # need to copy position, otherwise it gets reused!
            self.winPos = qt4.QPoint(event.pos())
            self.grabPos = self.widget().mapFromParent(self.winPos)

            if self.clickmode == 'select':
                # we set this to true unless the timer runs out (400ms),
                # then it becomes a scroll click
                # scroll clicks drag the window around, and selecting clicks
                # select widgets!
                self.scrolltimer.start(400)

            elif self.clickmode == 'scroll':
                qt4.QApplication.setOverrideCursor(
                    qt4.QCursor(qt4.Qt.SizeAllCursor))

            elif self.clickmode == 'graphzoom':
                self.label.drawRect(self.grabPos, self.grabPos)

            # record what mode we were clicked in
            self.currentclickmode = self.clickmode

    def mouseMoveEvent(self, event):
        """Scroll window by how much the mouse has moved since last time."""

        if self.currentclickmode == 'scroll':
            event.accept()

            # move scroll bars by amount
            pos = event.pos()
            dx = self.winPos.x()-pos.x()
            scrollx = self.horizontalScrollBar()
            scrollx.setValue( scrollx.value() + dx )

            dy = self.winPos.y()-pos.y()
            scrolly = self.verticalScrollBar()
            scrolly.setValue( scrolly.value() + dy )

            # need to copy point
            self.winPos = qt4.QPoint(event.pos())

        elif self.currentclickmode == 'graphzoom':
            # get rid of current rectangle
            pos = self.widget().mapFromParent(event.pos())
            self.label.drawRect(self.grabPos, pos)

    def mouseReleaseEvent(self, event):
        """If the mouse button is released, check whether the mouse
        clicked on a widget, and emit a sigWidgetClicked(widget)."""

        if event.button() == qt4.Qt.LeftButton:
            event.accept()
            self.scrolltimer.stop()
            if self.currentclickmode == 'select':
                # work out where the mouse clicked and choose widget
                pos = self.widget().mapFromParent(event.pos())
                self.locateClickWidget(pos.x(), pos.y())
            elif self.currentclickmode == 'scroll':
                # return the cursor to normal after scrolling
                qt4.QApplication.restoreOverrideCursor()
            elif self.currentclickmode == 'graphzoom':
                self.label.hideRect()
                self.doZoomRect(self.widget().mapFromParent(event.pos()))
            elif self.currentclickmode == 'viewgetclick':
                self.clickmode = 'select'
        else:
            qt4.QLabel.contentsMouseReleaseEvent(self, event)

    def locateClickWidget(self, x, y):
        """Work out which widget was clicked, and if necessary send
        a sigWidgetClicked(widget) signal."""
        
        # now crazily draw the whole thing again
        # see which widgets change the region in the small box given below
        bufferpixmap = qt4.QPixmap( *self.size )
        painter = ClickPainter(bufferpixmap, x-3, y-3, 7, 7)

        pagenumber = min( self.document.getNumberPages() - 1,
                          self.pagenumber )
        if pagenumber >= 0:
            self.document.paintTo(painter, self.pagenumber,
                                  scaling=self.zoomfactor, dpi=self.widgetdpi)
        painter.end()

        widget = painter.getFoundWidget()
        if widget:
            # tell connected objects that widget was clicked
            self.emit( qt4.SIGNAL('sigWidgetClicked'), widget )

    def setOutputSize(self):
        """Set the ouput display size."""

        # convert distances into pixels
        pix = qt4.QPixmap(1, 1)
        painter = document.Painter(pix)
        painter.veusz_scaling = self.zoomfactor
        painter.veusz_pixperpt = self.widgetdpi / 72.
        size = self.document.basewidget.getSize(painter)
        painter.end()

        # make new buffer and resize widget
        if size != self.size:
            self.size = size
            self.bufferpixmap = qt4.QPixmap( *self.size )
            self.forceupdate = True
            self.label.resize(*size)

    def setPageNumber(self, pageno):
        """Move the the selected page."""

        # we don't need to do anything
        if (self.pagenumber == pageno and
            self.document.changeset == self.docchangeset):
            return

        # keep within bounds
        pageno = min(pageno, self.document.getNumberPages()-1)
        pageno = max(0, pageno)

        self.pagenumber = pageno
        self.forceupdate = True

    def getPageNumber(self):
        """Get the the selected page."""
        return self.pagenumber

    def slotTimeout(self):
        """Called after timer times out, to check for updates to window."""

        # no threads, so can't get interrupted here
        # draw data into background pixmap if modified
        if ( self.zoomfactor != self.oldzoom or
             self.document.changeset != self.docchangeset or
             self.forceupdate ):

            self.setOutputSize()
            
            # fill pixmap with proper background colour
            self.bufferpixmap.fill( self.palette().color(qt4.QPalette.Base) )

            self.pagenumber = min( self.document.getNumberPages() - 1,
                                   self.pagenumber )
            if self.pagenumber >= 0:
                # draw the data into the buffer
                # errors cause an exception window to pop up
                try:
                    self.document.printTo( self.bufferpixmap,
                                           [self.pagenumber],
                                           scaling = self.zoomfactor,
                                           dpi = self.widgetdpi,
                                           antialias = self.antialias )
                except Exception:
                    # stop updates this time round and show exception dialog
                    d = exceptiondialog.ExceptionDialog(sys.exc_info(), self)
                    self.oldzoom = self.zoomfactor
                    self.forceupdate = False
                    self.docchangeset = self.document.changeset
                    d.exec_()
                    
            else:
                self.pagenumber = 0

            self.emit( qt4.SIGNAL("sigUpdatePage"), self.pagenumber )
            self.updatePageToolbar()

            self.oldzoom = self.zoomfactor
            self.forceupdate = False
            self.docchangeset = self.document.changeset

            self.label.setPixmap(self.bufferpixmap)

    def _constructContextMenu(self):
        """Construct the context menu."""

        menu = self.contextmenu = qt4.QMenu(self)

        # add some useful entries
        menu.addAction( self.viewactions['viewzoomin'] )
        menu.addAction( self.viewactions['viewzoomout'] )
        menu.addSeparator()
        menu.addAction( self.viewactions['viewprevpage'] )
        menu.addAction( self.viewactions['viewnextpage'] )
        menu.addSeparator()

        # update NOW!
        menu.addAction('Force update', self.actionForceUpdate)

        # Update submenu
        submenu = menu.addMenu('Updates')
        intgrp = qt4.QActionGroup(self)

        intervals = [0, 100, 250, 500, 1000, 2000, 5000, 10000]
        inttext = ['Disable']
        for intv in intervals[1:]:
            inttext.append('Every %gs' % (intv * 0.001))

        # need to keep copies of bound objects otherwise they are collected
        self._intfuncs = []

        # bind interval options to actions
        for intv, text in itertools.izip(intervals, inttext):
            act = intgrp.addAction(text)
            act.setCheckable(True)
            fn = utils.BoundCaller(self.actionSetTimeout, intv)
            self._intfuncs.append(fn)
            self.connect(act, qt4.SIGNAL('triggered(bool)'), fn)
            if intv == self.interval:
                act.setChecked(True)
            submenu.addAction(act)

        # antialias
        menu.addSeparator()
        act = menu.addAction('Antialias', self.actionAntialias)
        act.setCheckable(True)
        act.setChecked(self.antialias)
        
    def contextMenuEvent(self, event):
        """Show context menu."""
        self.contextmenu.exec_(qt4.QCursor.pos())

    def actionForceUpdate(self):
        """Force an update for the graph."""
        self.docchangeset = -100
        self.slotTimeout()

    def actionSetTimeout(self, interval, checked):
        """Called by setting the interval."""

        if interval == 0:
            # stop updates
            self.interval = 0
            if self.timer.isActive():
                self.timer.stop()
        else:
            # change interval to one selected
            self.interval = interval
            self.timer.setInterval(interval)
            # start timer if it was stopped
            if not self.timer.isActive():
                self.timer.start()

        # remember changes for next time
        setting.settingdb['plot_updateinterval'] = self.interval

    def actionAntialias(self):
        """Toggle antialias."""
        self.antialias = not self.antialias
        setting.settingdb['plot_antialias'] = self.antialias
        self.actionForceUpdate()

    def setZoomFactor(self, zoomfactor):
        """Set the zoom factor of the window."""
        self.zoomfactor = float(zoomfactor)
        self.update()

    def slotViewZoomIn(self):
        """Zoom into the plot."""
        self.setZoomFactor(self.zoomfactor * N.sqrt(2.))

    def slotViewZoomOut(self):
        """Zoom out of the plot."""
        self.setZoomFactor(self.zoomfactor / N.sqrt(2.))

    def slotViewZoomWidth(self):
        """Make the zoom factor so that the plot fills the whole width."""

        # need to take account of scroll bars when deciding size
        viewportsize = self.maximumViewportSize()
        aspectwin = viewportsize.width()*1./viewportsize.height()
        aspectplot = self.size[0]*1./self.size[1]

        width = viewportsize.width()
        if aspectwin > aspectplot:
            # take account of scroll bar
            width -= self.verticalScrollBar().width()
            
        mult = width*1./self.size[0]
        self.setZoomFactor(self.zoomfactor * mult)
        
    def slotViewZoomHeight(self):
        """Make the zoom factor so that the plot fills the whole width."""

        # need to take account of scroll bars when deciding size
        viewportsize = self.maximumViewportSize()
        aspectwin = viewportsize.width()*1./viewportsize.height()
        aspectplot = self.size[0]*1./self.size[1]

        height = viewportsize.height()
        if aspectwin < aspectplot:
            # take account of scroll bar
            height -= self.horizontalScrollBar().height()
            
        mult = height*1./self.size[1]
        self.setZoomFactor(self.zoomfactor * mult)

    def slotViewZoomPage(self):
        """Make the zoom factor correct to show the whole page."""

        viewportsize = self.maximumViewportSize()
        multw = viewportsize.width()*1./self.size[0]
        multh = viewportsize.height()*1./self.size[1]
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

    def updatePageToolbar(self):
        """Update page number when the plot window says so."""

        # disable previous and next page actions
        if self.viewactions is not None:
            np = self.document.getNumberPages()
            self.viewactions['viewprevpage'].setEnabled(self.pagenumber != 0)
            self.viewactions['viewnextpage'].setEnabled(self.pagenumber < np-1)

    def slotSelectMode(self, action):
        """Called when the selection mode has changed."""

        modecnvt = { self.viewactions['viewselect'] : 'select',
                     self.viewactions['viewzoomgraph'] : 'graphzoom' }
        
        # convert action into clicking mode
        self.clickmode = modecnvt[action]

        if self.clickmode == 'select':
            self.label.setCursor(qt4.Qt.ArrowCursor)
        elif self.clickmode == 'graphzoom':
            self.label.setCursor(qt4.Qt.CrossCursor)
        
    def getClick(self):
        """Return a click point from the graph."""

        # FIXME does not work for qt4 probably

        # wait for click from user
        qt4.QApplication.setOverrideCursor(qt4.QCursor(qt4.Qt.CrossCursor))
        oldmode = self.clickmode
        self.clickmode = 'viewgetclick'
        while self.clickmode == 'viewgetclick':
            qt4.qApp.processEvents()
        self.clickmode = oldmode
        qt4.QApplication.restoreOverrideCursor()

        # take clicked point and convert to coords of scrollview
        pt = qt4.QPoint(*self.grabPos)
        pt = self.viewport().mapFromGlobal(pt)
        pt = self.viewportToContents(pt)

        # try to work out in which widget the first point is in
        bufferpixmap = qt4.QPixmap( *self.size )
        painter = PointPainter(bufferpixmap, pt.x(), pt.y())
        pagenumber = min( self.document.getNumberPages() - 1,
                          self.pagenumber )
        if pagenumber >= 0:
            self.document.paintTo(painter, self.pagenumber,
                                  scaling=self.zoomfactor, dpi=self.widgetdpi)
        painter.end()

        # get widget
        widget = painter.widget
        if widget is None:
            return []
        
        # convert points on plotter to points on axis for each axis
        xpts = N.array( [pt.x()] )
        ypts = N.array( [pt.y()] )

        axesretn = []
        # iterate over children, to look for plotters
        for c in [i for i in widget.children if
                  isinstance(i, widgets.GenericPlotter)]:

            # get axes associated with plotter
            axes = c.parent.getAxes( (c.settings.xAxis,
                                      c.settings.yAxis) )

            # iterate over each, and update the ranges
            for axis in [a for a in axes if a is not None]:
                s = axis.settings
                if s.direction == 'horizontal':
                    p = xpts
                else:
                    p = ypts

                # convert point on plotter to axis coordinate
                # FIXME: Need To Trap Conversion Errors!
                r = axis.plotterToGraphCoords(painter.bounds[axis], p)

                axesretn.append( (axis.path, r[0]) )

        return axesretn
