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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

from __future__ import division
import sys
import traceback

from ..compat import crange
from .. import qtall as qt
import numpy as N

from .. import setting
from ..dialogs import exceptiondialog
from .. import document
from .. import utils
from .. import widgets

def _(text, disambiguation=None, context='PlotWindow'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class PickerCrosshairItem( qt.QGraphicsPathItem ):
    """The picker cross widget: it moves from point to point and curve to curve
       with the arrow keys, and hides itself when it looses focus"""
    def __init__(self, parent=None):
        path = qt.QPainterPath()
        path.addRect(-4, -4, 8, 8)
        path.addRect(-5, -5, 10, 10)
        path.moveTo(-8, 0)
        path.lineTo(8, 0)
        path.moveTo(0, -8)
        path.lineTo(0, 8)

        qt.QGraphicsPathItem.__init__(self, path, parent)
        self.setBrush(qt.QBrush(qt.Qt.black))
        self.setFlags(self.flags() | qt.QGraphicsItem.ItemIsFocusable)

    def paint(self, painter, option, widget):
        """Override this to enforce the global antialiasing setting"""
        aa = setting.settingdb['plot_antialias']
        painter.save()
        painter.setRenderHint(qt.QPainter.Antialiasing, aa)
        qt.QGraphicsPathItem.paint(self, painter, option, widget)
        painter.restore()

    def focusOutEvent(self, event):
        qt.QGraphicsPathItem.focusOutEvent(self, event)
        self.hide()

class RenderControl(qt.QObject):
    """Object for rendering plots in a separate thread."""

    # emitted when new item on plot queue
    sigQueueChange = qt.pyqtSignal(int)

    # when a rendering job is finished
    signalRenderFinished = qt.pyqtSignal(
        int, qt.QImage, document.PaintHelper)

    def __init__(self, plotwindow):
        """Start up numthreads rendering threads."""
        qt.QObject.__init__(self)
        self.sem = qt.QSemaphore()
        self.mutex = qt.QMutex()
        self.threads = []
        self.exit = False
        self.latestjobs = []
        self.latestaddedjob = -1
        self.latestdrawnjob = -1
        self.plotwindow = plotwindow

        self.updateNumberThreads()

    def updateNumberThreads(self, num=None):
        """Changes the number of rendering threads."""
        if num is None:
            if qt.QFontDatabase.supportsThreadedFontRendering():
                # use number of threads in preference
                num = setting.settingdb['plot_numthreads']
            else:
                # disable threads
                num = 0

        if self.threads:
            # delete old ones
            self.exit = True
            self.sem.release(len(self.threads))
            for t in self.threads:
                t.wait()
            del self.threads[:]
            self.exit = False

        # start new ones
        for i in crange(num):
            t = RenderThread(self)
            t.start()
            self.threads.append(t)

    def exitThreads(self):
        """Exit threads started."""
        self.updateNumberThreads(num=0)

    def processNextJob(self):
        """Take a job from the queue and process it.

        emits renderfinished(jobid, img, painthelper)
        when done, if job has not been superseded
        """

        self.mutex.lock()
        jobid, helper = self.latestjobs[-1]
        del self.latestjobs[-1]
        lastadded = self.latestaddedjob
        self.mutex.unlock()

        # don't process jobs which have been superseded
        if lastadded == jobid:
            img = qt.QImage(
                int(helper.rawpagesize[0]), int(helper.rawpagesize[1]),
                qt.QImage.Format_ARGB32_Premultiplied)
            img.fill( setting.settingdb.color('page').rgb() )

            painter = qt.QPainter(img)
            aa = self.plotwindow.antialias
            painter.setRenderHint(qt.QPainter.Antialiasing, aa)
            painter.setRenderHint(qt.QPainter.TextAntialiasing, aa)
            helper.renderToPainter(painter)
            painter.end()

            self.mutex.lock()
            # just throw away result if it older than the latest one
            if jobid > self.latestdrawnjob:
                self.signalRenderFinished.emit(jobid, img, helper)
                self.latestdrawnjob = jobid
            self.mutex.unlock()

        # tell any listeners that a job has been processed
        self.sigQueueChange.emit(-1)

    def addJob(self, helper):
        """Process drawing job in PaintHelper given."""

        # indicate that there is a new item to be processed to listeners
        self.sigQueueChange.emit(1)

        # add the job to the queue
        self.mutex.lock()
        self.latestaddedjob += 1
        self.latestjobs.append( (self.latestaddedjob, helper) )
        self.mutex.unlock()

        if self.threads:
            # tell a thread to process job
            self.sem.release(1)
        else:
            # process job in current thread if multithreading disabled
            self.processNextJob()

class RenderThread( qt.QThread ):
    """A thread for processing rendering jobs.
    This is controlled by a RenderControl object
    """

    def __init__(self, rendercontrol):
        qt.QThread.__init__(self)
        self.rc = rendercontrol

    def run(self):
        """Repeat forever until told to exit.
        If it aquires 1 resource from the semaphore it will process
        the next job.
        """
        while True:
            # wait until we can aquire the resources
            self.rc.sem.acquire(1)
            if self.rc.exit:
                break
            try:
                self.rc.processNextJob()
            except Exception:
                sys.stderr.write(_("Error in rendering thread\n"))
                traceback.print_exc(file=sys.stderr)

class ControlGraphRoot(qt.QGraphicsItem):
    """Control graph items are connected to this root item.
    We don't use a group here as it would swallow parent events."""
    def __init__(self):
        qt.QGraphicsItem.__init__(self)
    def paint(self, painter, option, widget=None):
        pass
    def boundingRect(self):
        return qt.QRectF()

class PlotWindow( qt.QGraphicsView ):
    """Class to show the plot(s) in a scrollable window."""

    # emitted when new item on plot queue
    sigQueueChange = qt.pyqtSignal(int)
    # on drawing a page
    sigUpdatePage = qt.pyqtSignal(int)
    # point picked on plot
    sigPointPicked = qt.pyqtSignal(object)
    # picker enabled
    sigPickerEnabled = qt.pyqtSignal(bool)
    # axis values update from moving mouse
    sigAxisValuesFromMouse = qt.pyqtSignal(dict)
    # gives widget clicked
    sigWidgetClicked = qt.pyqtSignal(object, str)

    # how often the document can update
    updateintervals = (
        (0, _('Disable')),
        (-1, _('On document change')),
        (100, _('Every 0.1s')),
        (250, _('Every 0.25s')),
        (500, _('Every 0.5s')),
        (1000, _('Every 1s')),
        (2000, _('Every 2s')),
        (5000, _('Every 5s')),
        (10000, _('Every 10s')),
        )

    def __init__(self, document, parent, menu=None):
        """Initialise the window.

        menu gives a menu to add any menu items to
        """

        qt.QGraphicsView.__init__(self, parent)
        self.setBackgroundRole(qt.QPalette.Dark)
        self.scene = qt.QGraphicsScene()
        self.setScene(self.scene)

        # this graphics scene item is the actual graph
        pixmap = qt.QPixmap(1, 1)
        self.dpi = (pixmap.logicalDpiX(), pixmap.logicalDpiY())
        self.pixmapitem = self.scene.addPixmap(pixmap)

        # whether full screen mode
        self.isfullscreen = False

        # set to be parent's actions
        self.vzactions = None

        # for controlling plot elements
        self.controlgraphroot = ControlGraphRoot()
        self.scene.addItem(self.controlgraphroot)

        # zoom rectangle for zooming into graph (not shown normally)
        self.zoomrect = self.scene.addRect(
            0, 0, 100, 100, qt.QPen(qt.Qt.DotLine))
        self.zoomrect.setZValue(2.)
        self.zoomrect.hide()

        # picker graphicsitem for marking the picked point
        self.pickeritem = PickerCrosshairItem()
        self.scene.addItem(self.pickeritem)
        self.pickeritem.setZValue(2.)
        self.pickeritem.hide()

        # all the widgets that picker key-navigation might cycle through
        self.pickerwidgets = []

        # the picker state
        self.pickerinfo = widgets.PickInfo()

        # set up so if document is modified we are notified
        self.document = document
        self.docchangeset = -100
        self.oldpagenumber = -1
        self.document.signalModified.connect(self.slotDocModified)

        # state of last plot from painthelper
        self.painthelper = None

        self.lastwidgetsselected = []
        self.oldzoom = -1.
        self.zoomfactor = 1.
        self.pagenumber = 0
        self.ignoreclick = False

        # for rendering plots in separate threads
        self.rendercontrol = RenderControl(self)
        self.rendercontrol.signalRenderFinished.connect(
            self.slotRenderFinished)
        self.rendercontrol.sigQueueChange.connect(
            self.sigQueueChange)

        # mode for clicking
        self.clickmode = 'select'
        self.currentclickmode = None

        # wheel zooming/scrolling accumulator
        self.sumwheeldelta = 0

        # set up redrawing timer
        self.timer = qt.QTimer(self)
        self.timer.timeout.connect(self.checkPlotUpdate)

        # for drag scrolling
        self.grabpos = None
        self.scrolltimer = qt.QTimer(self)
        self.scrolltimer.setSingleShot(True)

        # for turning clicking into scrolling after a period
        self.scrolltimer.timeout.connect(self.slotBecomeScrollClick)

        # get plot view updating policy
        #  -1: update on document changes
        #   0: never update automatically
        #  >0: check for updates every x ms
        self.interval = setting.settingdb['plot_updatepolicy']

        # if using a time-based document update checking, start timer
        if self.interval > 0:
            self.timer.start(self.interval)

        # load antialias settings
        self.antialias = setting.settingdb['plot_antialias']

        # allow window to get focus, to allow context menu
        self.setFocusPolicy(qt.Qt.StrongFocus)

        # get mouse move events if mouse is not pressed
        self.setMouseTracking(True)

        # create toolbar in main window (urgh)
        self.createToolbar(parent, menu)

    def hideEvent(self, event):
        """Window closing, so exit rendering threads."""
        self.rendercontrol.exitThreads()
        qt.QGraphicsView.hideEvent(self, event)

    def sizeHint(self):
        """Return size hint for window."""
        p = self.pixmapitem.pixmap()
        if p.width() <= 1 and p.height() <= 1:
            # if the document has been uninitialized, get the doc size
            return qt.QSize(*self.document.docSize())
        return p.size()

    def showToolbar(self, show=True):
        """Show or hide toolbar"""
        self.viewtoolbar.setVisible(show)

    def createToolbar(self, parent, menu=None):
        """Make a view toolbar, and optionally update menu."""

        self.viewtoolbar = qt.QToolBar(_("View toolbar - Veusz"), parent)
        self.viewtoolbar.setObjectName('veuszviewtoolbar')
        iconsize = setting.settingdb['toolbar_size']
        self.viewtoolbar.setIconSize(qt.QSize(iconsize, iconsize))
        self.viewtoolbar.hide()
        if parent:
            parent.addToolBar(qt.Qt.TopToolBarArea, self.viewtoolbar)

        if parent and hasattr(parent, 'vzactions'):
            # share actions with parent if possible
            # as plot windows can be isolated from mainwindows, we need this
            self.vzactions = actions = parent.vzactions
        else:
            self.vzactions = actions = {}

        a = utils.makeAction
        actions.update({
                'view.zoomin':
                    a(self, _('Zoom into the plot'), _('Zoom &In'),
                      self.slotViewZoomIn,
                      icon='kde-zoom-in', key='Ctrl++'),
                'view.zoomout':
                    a(self, _('Zoom out of the plot'), _('Zoom &Out'),
                      self.slotViewZoomOut,
                      icon='kde-zoom-out', key='Ctrl+-'),
                'view.zoom11':
                    a(self, _('Restore plot to natural size'), _('Zoom 1:1'),
                      self.slotViewZoom11,
                      icon='kde-zoom-1-veuszedit', key='Ctrl+1'),
                'view.zoomwidth':
                    a(self, _('Zoom plot to show whole width'), _('Zoom to width'),
                      self.slotViewZoomWidth,
                      icon='kde-zoom-width-veuszedit'),
                'view.zoomheight':
                    a(self, _('Zoom plot to show whole height'), _('Zoom to height'),
                      self.slotViewZoomHeight,
                      icon='kde-zoom-height-veuszedit'),
                'view.zoompage':
                    a(self, _('Zoom plot to show whole page'), _('Zoom to page'),
                      self.slotViewZoomPage,
                      icon='kde-zoom-page-veuszedit'),
                'view.zoommenu':
                    a(self, _('Zoom functions menu'), _('Zoom'),
                      None,
                      icon='kde-zoom-veuszedit'),
                'view.prevpage':
                    a(self, _('Move to the previous page'), _('&Previous page'),
                      self.slotViewPreviousPage,
                      icon='kde-go-previous', key='Ctrl+PgUp'),
                'view.nextpage':
                    a(self, _('Move to the next page'), _('&Next page'),
                      self.slotViewNextPage,
                      icon='kde-go-next', key='Ctrl+PgDown'),
                'view.select':
                    a(self, _('Select items from the graph or scroll'),
                      _('Select items or scroll'),
                      None,
                      icon='kde-mouse-pointer'),
                'view.pick':
                    a(self, _('Read data points on the graph'),
                      _('Read data points'),
                      None,
                      icon='veusz-pick-data'),
                'view.zoomgraph':
                    a(self, _('Zoom into graph'), _('Zoom graph'),
                      None,
                      icon='veusz-zoom-graph'),
                'view.fullscreen':
                    a(self, _('View plot full screen'), _('Full screen'),
                      self.slotFullScreen,
                      icon='veusz-view-fullscreen', key='Ctrl+F11'),
                })

        if menu:
            # only construct menu if required
            menuitems = [
                ('view', '', [
                        'view.zoomin', 'view.zoomout',
                        'view.zoom11', 'view.zoomwidth',
                        'view.zoomheight', 'view.zoompage',
                        '',
                        'view.prevpage', 'view.nextpage',
                        'view.fullscreen',
                        '',
                        'view.select', 'view.pick', 'view.zoomgraph',
                        ]),
                ]
            utils.constructMenus(menu, {'view': menu}, menuitems,
                                 actions)

        # populate menu on zoom toolbar icon
        utils.makeMenuGroupSaved(
            'view.zoommenu', self, actions, (
                'view.zoomin', 'view.zoomout', 'view.zoom11',
                'view.zoomwidth', 'view.zoomheight', 'view.zoompage',
            ))

        # add items to toolbar
        utils.addToolbarActions(
            self.viewtoolbar, actions, (
                'view.prevpage', 'view.nextpage',
                'view.fullscreen',
                'view.select', 'view.pick',
                'view.zoomgraph', 'view.zoommenu',
            ))

        # define action group for various different selection models
        grp = self.selectactiongrp = qt.QActionGroup(self)
        grp.setExclusive(True)
        for a in ('view.select', 'view.pick', 'view.zoomgraph'):
            actions[a].setActionGroup(grp)
            actions[a].setCheckable(True)
        actions['view.select'].setChecked(True)
        grp.triggered.connect(self.slotSelectMode)

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

        # safety net
        if self.grabpos is None or endpos is None:
            return

        # get points corresponding to corners of rectangle
        pt1 = self.grabpos
        pt2 = endpos

        # work out whether it's worthwhile to zoom: only zoom if there
        # are >=5 pixels movement
        if abs((pt2-pt1).x()) < 10 or abs((pt2-pt1).y()) < 10:
            return

        # scale into graph coordinates
        pt1 /= self.painthelper.cgscale
        pt2 /= self.painthelper.cgscale

        # try to work out in which widget the first point is in
        widget = self.painthelper.pointInWidgetBounds(
            pt1.x(), pt1.y(), widgets.Graph)
        if widget is None:
            return

        # convert points on plotter to points on axis for each axis
        # we also add a neighbouring pixel for the rounding calculation
        xpts = N.array( [pt1.x(), pt2.x(), pt1.x()+1, pt2.x()-1] )
        ypts = N.array( [pt1.y(), pt2.y(), pt1.y()+1, pt2.y()-1] )

        # build up operation list to do zoom
        operations = []

        axes = {}
        # iterate over children, to look for plotters
        for c in [i for i in widget.children if
                  isinstance(i, widgets.GenericPlotter)]:

            # get axes associated with plotter
            caxes = c.parent.getAxes( (c.settings.xAxis,
                                      c.settings.yAxis) )

            for a in caxes:
                if a:
                    axes[a] = True

        # iterate over each axis, and update the ranges
        for axis in axes:
            s = axis.settings
            if s.direction == 'horizontal':
                p = xpts
            else:
                p = ypts

            # convert points on plotter to axis coordinates
            # FIXME: Need To Trap Conversion Errors!
            try:
                r = axis.plotterToGraphCoords(
                    self.painthelper.widgetBounds(axis), p)
            except KeyError:
                continue

            # invert if min and max are inverted
            if r[1] < r[0]:
                r[1], r[0] = r[0], r[1]
                r[3], r[2] = r[2], r[3]

            # build up operations to change axis
            if s.min != r[0]:
                operations.append( document.OperationSettingSet(
                        s.get('min'),
                        utils.round2delt(r[0], r[2])) )

            if s.max != r[1]:
                operations.append( document.OperationSettingSet(
                        s.get('max'),
                        utils.round2delt(r[1], r[3])) )

        # finally change the axes
        self.document.applyOperation(
            document.OperationMultiple(operations,descr=_('zoom axes')) )

    def axesForPoint(self, mousepos):
        """Find all the axes which contain the given mouse position"""

        if self.painthelper is None:
            return []

        pos = self.mapToScene(mousepos)
        px = pos.x() / self.painthelper.cgscale
        py = pos.y() / self.painthelper.cgscale

        axes = []
        for widget, bounds in self.painthelper.widgetBoundsIterator(
            widgettype=widgets.Axis):
            # if widget is axis, and point lies within bounds
            if ( px>=bounds[0] and px<=bounds[2] and
                 py>=bounds[1] and py<=bounds[3] ):

                # convert correct pointer position
                if widget.settings.direction == 'horizontal':
                    val = px
                else:
                    val = py
                coords=widget.plotterToGraphCoords(bounds, N.array([val]))
                axes.append( (widget, coords[0]) )

        return axes

    def emitPicked(self, pickinfo):
        """Report that a new point has been picked"""

        self.pickerinfo = pickinfo
        self.pickeritem.setPos(
            pickinfo.graphpos[0] * self.painthelper.cgscale,
            pickinfo.graphpos[1] * self.painthelper.cgscale)
        self.sigPointPicked.emit(pickinfo)

    def doPick(self, mousepos):
        """Find the point on any plot-like widget closest to the cursor"""

        self.pickerwidgets = []

        pickinfo = widgets.PickInfo()
        # get scalable graph coordinates for mouse point
        pos = self.mapToScene(mousepos)
        pos /= self.painthelper.cgscale

        for w, bounds in self.painthelper.widgetBoundsIterator():
            # ask the widget for its (visually) closest point to the cursor
            try:
                info = w.pickPoint(pos.x(), pos.y(), bounds)
            except AttributeError:
                # widget isn't pickable
                continue

            if info:
                # this is a pickable widget, so remember it for future
                # key navigation
                self.pickerwidgets.append(w)

                if info.distance < pickinfo.distance:
                    # and remember the overall closest
                    pickinfo = info

        if not pickinfo:
            self.pickeritem.hide()
            return

        self.emitPicked(pickinfo)

    def slotBecomeScrollClick(self):
        """If the click is still down when this timer is reached then
        we turn the click into a scrolling click."""

        if self.currentclickmode == 'select':
            qt.QApplication.setOverrideCursor(qt.QCursor(qt.Qt.SizeAllCursor))
            self.currentclickmode = 'scroll'

    def mousePressEvent(self, event):
        """Allow user to drag window around."""

        qt.QGraphicsView.mousePressEvent(self, event)
        if self.painthelper is None:
            return

        # work out whether user is clicking on a control point
        items = self.items(event.pos())
        self.ignoreclick = ( len(items)==0 or
                             items[0] is not self.pixmapitem or
                             self.painthelper is None )

        if event.button() == qt.Qt.LeftButton and not self.ignoreclick:

            # need to copy position, otherwise it gets reused!
            self.winpos = qt.QPoint(event.pos())
            self.grabpos = self.mapToScene(self.winpos)

            if self.clickmode == 'select':
                # we set this to true unless the timer runs out (400ms),
                # then it becomes a scroll click
                # scroll clicks drag the window around, and selecting clicks
                # select widgets!
                self.scrolltimer.start(400)

            elif self.clickmode == 'pick':
                self.pickeritem.show()
                self.pickeritem.setFocus(qt.Qt.MouseFocusReason)
                self.doPick(event.pos())

            elif self.clickmode == 'scroll':
                qt.QApplication.setOverrideCursor(
                    qt.QCursor(qt.Qt.SizeAllCursor))

            elif self.clickmode == 'graphzoom':
                self.zoomrect.setRect(
                    self.grabpos.x(), self.grabpos.y(),
                    0, 0)
                self.zoomrect.show()

                #self.label.drawRect(self.grabpos, self.grabpos)

            # record what mode we were clicked in
            self.currentclickmode = self.clickmode

    def mouseMoveEvent(self, event):
        """Scroll window by how much the mouse has moved since last time."""

        qt.QGraphicsView.mouseMoveEvent(self, event)
        if self.painthelper is None:
            return

        if self.currentclickmode == 'scroll':
            event.accept()

            # move scroll bars by amount
            pos = event.pos()
            dx = self.winpos.x()-pos.x()
            scrollx = self.horizontalScrollBar()
            scrollx.setValue( scrollx.value() + dx )

            dy = self.winpos.y()-pos.y()
            scrolly = self.verticalScrollBar()
            scrolly.setValue( scrolly.value() + dy )

            # need to copy point
            self.winpos = qt.QPoint(event.pos())

        elif self.currentclickmode == 'graphzoom' and self.grabpos is not None:
            pos2 = self.mapToScene(event.pos())
            self.zoomrect.setRect(qt.QRectF(
                qt.QPointF(
                    min(self.grabpos.x(), pos2.x()),
                    min(self.grabpos.y(), pos2.y())),
                qt.QPointF(
                    max(self.grabpos.x(), pos2.x()),
                    max(self.grabpos.y(), pos2.y())),
                ))

        elif self.clickmode == 'select' or self.clickmode == 'pick':
            # find axes which map to this position
            axes = self.axesForPoint(event.pos())
            vals = dict([ (a[0].name, a[1]) for a in axes ])

            self.sigAxisValuesFromMouse.emit(vals)

            if self.currentclickmode == 'pick':
                # drag the picker around
                self.doPick(event.pos())

    def mouseReleaseEvent(self, event):
        """If the mouse button is released, check whether the mouse
        clicked on a widget, and emit a sigWidgetClicked(widget,mode)."""

        qt.QGraphicsView.mouseReleaseEvent(self, event)
        if self.painthelper is None:
            return

        if event.button() == qt.Qt.LeftButton and not self.ignoreclick:
            event.accept()
            self.scrolltimer.stop()
            if self.currentclickmode == 'select':
                # work out where the mouse clicked and choose widget
                pos = self.mapToScene(event.pos())
                self.identifyAndClickWidget(pos.x(), pos.y(), event.modifiers())
            elif self.currentclickmode == 'scroll':
                # return the cursor to normal after scrolling
                self.clickmode = 'select'
                self.currentclickmode = None
                qt.QApplication.restoreOverrideCursor()
            elif self.currentclickmode == 'graphzoom':
                self.zoomrect.hide()
                self.doZoomRect(self.mapToScene(event.pos()))
                self.grabpos = None
            elif self.currentclickmode == 'viewgetclick':
                self.clickmode = 'select'
            elif self.currentclickmode == 'pick':
                self.currentclickmode = None

    def keyPressEvent(self, event):
        """Keypad motion moves the picker if it has focus"""
        if self.pickeritem.hasFocus():

            k = event.key()
            if k == qt.Qt.Key_Left or k == qt.Qt.Key_Right:
                # navigate to the previous or next point on the curve
                event.accept()
                dir = 'right' if k == qt.Qt.Key_Right else 'left'
                ix = self.pickerinfo.index
                pickinfo = self.pickerinfo.widget.pickIndex(
                    ix, dir, self.painthelper.widgetBounds(
                        self.pickerinfo.widget))
                if pickinfo:
                    # more points visible in this direction
                    self.emitPicked(pickinfo)
                return

            elif k == qt.Qt.Key_Up or k == qt.Qt.Key_Down:
                # navigate to the next plot up or down on the screen
                event.accept()
                p = self.pickeritem.pos()

                oldw = self.pickerinfo.widget
                pickinfo = widgets.PickInfo()

                dist = float('inf')
                for w in self.pickerwidgets:
                    if w == oldw:
                        continue

                    # ask the widgets to pick their point which is closest horizontally
                    # to the last (screen) x value picked
                    pi = w.pickPoint(
                        self.pickerinfo.graphpos[0], p.y(),
                        self.painthelper.widgetBounds(w),
                        distance='horizontal')
                    if not pi:
                        continue

                    dy = p.y() - pi.graphpos[1]

                    # take the new point which is closest vertically to the current
                    # one and either above or below it as appropriate
                    if abs(dy) < dist and (
                            (k == qt.Qt.Key_Up and dy > 0) or
                            (k == qt.Qt.Key_Down and dy < 0) ):
                        pickinfo = pi
                        dist = abs(dy)

                if pickinfo:
                    oldx = self.pickerinfo.graphpos[0]
                    self.emitPicked(pickinfo)

                    # restore the previous x-position, so that vertical navigation
                    # stays repeatable
                    pickinfo.graphpos = (oldx, pickinfo.graphpos[1])

                return

        # handle up-stream
        qt.QGraphicsView.keyPressEvent(self, event)

    def wheelEvent(self, event):
        """For zooming in or moving."""

        if event.modifiers() & qt.Qt.ControlModifier:
            # zoom in/out with ctrl held down
            d = event.angleDelta()
            delta = d.x() if d.x() != 0 else d.y()
            self.sumwheeldelta += delta
            while self.sumwheeldelta <= -120:
                self.slotViewZoomOut()
                self.sumwheeldelta += 120
            while self.sumwheeldelta >= 120:
                self.slotViewZoomIn()
                self.sumwheeldelta -= 120
            event.accept()
        elif event.modifiers() & qt.Qt.ShiftModifier:
            # scroll horizontally if shift is held down
            d = event.pixelDelta()
            if d.isNull():
                # Fallback mode to angleDelta
                d = event.angleDelta()
            delta = d.x() if d.x() != 0 else d.y()
            scrollx = self.horizontalScrollBar()
            scrollx.setValue(scrollx.value() + delta)
            event.accept()
        else:
            qt.QGraphicsView.wheelEvent(self, event)

    def identifyAndClickWidget(self, x, y, modifier):
        """Work out which widget was clicked, and if necessary send
        a sigWidgetClicked(widget) signal."""

        if self.document.getNumberPages() == 0:
            return

        widget = self.painthelper.identifyWidgetAtPoint(
            x, y, antialias=self.antialias)
        if widget is None:
            # select page if nothing clicked
            widget = self.document.basewidget.getPage(self.pagenumber)

        # tell connected objects that widget was clicked
        if widget is not None:
            if modifier & qt.Qt.ControlModifier:
                mode = 'toggle'
            elif modifier & qt.Qt.ShiftModifier:
                mode = 'add'
            else:
                mode = 'new'

            self.sigWidgetClicked.emit(widget, mode)

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
        if self.pagenumber != self.oldpagenumber and self.interval != 0:
            self.checkPlotUpdate()

    def getPageNumber(self):
        """Get the the selected page."""
        return self.pagenumber

    @qt.pyqtSlot(int)
    def slotDocModified(self, ismodified):
        """Update plot on document being modified."""
        # only update if doc is modified and the update policy is set
        # to update on document updates
        if ismodified and self.interval == -1:
            self.checkPlotUpdate()

    def getDevicePixelRatio(self):
        """Get device pixel ratio for window."""
        # ugly, as we have to get it from QWindow
        widget = self
        while widget:
            window = widget.windowHandle()
            if window is not None:
                try:
                    return window.devicePixelRatio()
                except AttributeError:
                    return 1
            widget = widget.parent()
        return 1

    def checkPlotUpdate(self):
        """Check whether plot needs updating."""

        # no threads, so can't get interrupted here
        # draw data into background pixmap if modified

        # is an update required?
        if ( self.zoomfactor == self.oldzoom and
             self.document.changeset == self.docchangeset and
             self.pagenumber == self.oldpagenumber ):
            return

        self.pickeritem.hide()

        # do we need the following line?
        self.pagenumber = min(
            self.document.getNumberPages()-1, self.pagenumber)
        self.oldpagenumber = self.pagenumber

        if self.pagenumber >= 0:
            devicepixelratio = self.getDevicePixelRatio()
            scaling = self.zoomfactor*devicepixelratio
            size = self.document.pageSize(
                self.pagenumber, scaling=scaling, integer=False)

            # draw the data into the buffer
            # errors cause an exception window to pop up
            try:
                phelper = document.PaintHelper(
                    self.document, size,
                    scaling=scaling,
                    dpi=self.dpi,
                    devicepixelratio=devicepixelratio)
                self.document.paintTo(phelper, self.pagenumber)

            except Exception:
                # stop updates this time round and show exception dialog
                d = exceptiondialog.ExceptionDialog(sys.exc_info(), self)
                self.oldzoom = self.zoomfactor
                self.docchangeset = self.document.changeset
                d.exec_()

            self.painthelper = phelper
            self.rendercontrol.addJob(phelper)
        else:
            self.painthelper = None
            self.pagenumber = 0
            size = self.document.docSize()
            pixmap = qt.QPixmap(*size)
            pixmap.fill( setting.settingdb.color('page') )
            self.setSceneRect(0, 0, *size)
            self.pixmapitem.setPixmap(pixmap)

        self.sigUpdatePage.emit(self.pagenumber)
        self.updatePageToolbar()

        self.updateControlGraphs(self.lastwidgetsselected)
        self.oldzoom = self.zoomfactor
        self.docchangeset = self.document.changeset

    def slotRenderFinished(self, jobid, img, helper):
        """Update image on display if rendering (usually in other
        thread) finished."""
        dpr = helper.devicepixelratio
        bufferpixmap = qt.QPixmap.fromImage(img)
        bufferpixmap.setDevicePixelRatio(dpr)
        self.setSceneRect(0, 0, bufferpixmap.width()/dpr, bufferpixmap.height()/dpr)
        self.pixmapitem.setPixmap(bufferpixmap)

    def updatePlotSettings(self):
        """Update plot window settings from settings."""
        self.setTimeout(setting.settingdb['plot_updatepolicy'])
        self.antialias = setting.settingdb['plot_antialias']
        self.rendercontrol.updateNumberThreads()
        self.actionForceUpdate()

    def contextMenuEvent(self, event):
        """Show context menu."""

        menu = qt.QMenu(self)

        # add some useful entries
        menu.addAction( self.vzactions['view.zoommenu'] )
        menu.addSeparator()
        menu.addAction( self.vzactions['view.prevpage'] )
        menu.addAction( self.vzactions['view.nextpage'] )
        menu.addSeparator()

        # force an update now menu item
        menu.addAction(_('Force update'), self.actionForceUpdate)

        if self.isfullscreen:
            menu.addAction(_('Close full screen'), self.slotFullScreen)
        else:
            menu.addAction( self.vzactions['view.fullscreen'] )

        # Update policy submenu
        submenu = menu.addMenu(_('Updates'))
        intgrp = qt.QActionGroup(self)

        # bind interval options to actions
        for intv, text in self.updateintervals:
            act = intgrp.addAction(text)
            act.setCheckable(True)
            def setfn(interval):
                return lambda checked: self.actionSetTimeout(interval, checked)
            act.triggered.connect(setfn(intv))
            if intv == self.interval:
                act.setChecked(True)
            submenu.addAction(act)

        # antialias
        menu.addSeparator()
        act = menu.addAction(_('Antialias'), self.actionAntialias)
        act.setCheckable(True)
        act.setChecked(self.antialias)

        menu.exec_(qt.QCursor.pos())

    def actionForceUpdate(self):
        """Force an update for the graph."""
        self.docchangeset = -100
        self.checkPlotUpdate()

    def slotFullScreen(self):
        """Show window full screen or not."""
        if not self.isfullscreen:
            self._fullscreenwindow = FullScreenPlotWindow(
                self.document, self.pagenumber)
        else:
            # cheesy way of closing full screen window
            p = self
            while p.parent() is not None:
                p = p.parent()
            p.close()

    def setTimeout(self, interval):
        """Change timer setting without changing save value."""
        self.interval = interval
        if interval <= 0:
            # stop updates
            if self.timer.isActive():
                self.timer.stop()
        else:
            # change interval to one selected
            self.timer.setInterval(interval)
            # start timer if it was stopped
            if not self.timer.isActive():
                self.timer.start()

    def actionSetTimeout(self, interval, checked):
        """Called by setting the interval."""

        self.setTimeout(interval)

        # remember changes for next time
        setting.settingdb['plot_updatepolicy'] = self.interval

    def actionAntialias(self):
        """Toggle antialias."""
        self.antialias = not self.antialias
        setting.settingdb['plot_antialias'] = self.antialias
        self.actionForceUpdate()

    def setZoomFactor(self, zoomfactor):
        """Set the zoom factor of the window."""
        zoomfactor = max(0.05, min(20, zoomfactor))
        self.zoomfactor = float(zoomfactor)
        self.checkPlotUpdate()

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
        aspectwin = viewportsize.width() / viewportsize.height()
        r = self.pixmapitem.boundingRect()
        aspectplot = r.width() / r.height()

        width = viewportsize.width()
        if aspectwin > aspectplot:
            # take account of scroll bar
            width -= self.verticalScrollBar().width()

        mult = width / r.width()
        self.setZoomFactor(self.zoomfactor * mult)

    def slotViewZoomHeight(self):
        """Make the zoom factor so that the plot fills the whole width."""

        viewportsize = self.maximumViewportSize()
        pixrect = self.pixmapitem.boundingRect()

        try:
            aspectwin = viewportsize.width() / viewportsize.height()
            aspectplot = pixrect.width() / pixrect.height()
        except ZeroDivisionError:
            return

        height = viewportsize.height()
        if aspectwin < aspectplot:
            # take account of scroll bar
            height -= self.horizontalScrollBar().height()

        mult = height / pixrect.height()
        self.setZoomFactor(self.zoomfactor * mult)

    def slotViewZoomPage(self):
        """Make the zoom factor correct to show the whole page."""

        viewportsize = self.maximumViewportSize()
        r = self.pixmapitem.boundingRect()
        if r.width() != 0 and r.height() != 0:
            multw = viewportsize.width() / r.width()
            multh = viewportsize.height() / r.height()
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
        if self.vzactions is not None:
            np = self.document.getNumberPages()
            self.vzactions['view.prevpage'].setEnabled(self.pagenumber != 0)
            self.vzactions['view.nextpage'].setEnabled(self.pagenumber < np-1)

    def slotSelectMode(self, action):
        """Called when the selection mode has changed."""

        modecnvt = { self.vzactions['view.select'] : 'select',
                     self.vzactions['view.pick'] : 'pick',
                     self.vzactions['view.zoomgraph'] : 'graphzoom' }

        # close the current picker
        self.pickeritem.hide()
        self.sigPickerEnabled.emit(False)

        # convert action into clicking mode
        self.clickmode = modecnvt[action]

        if self.clickmode == 'select':
            self.pixmapitem.unsetCursor()
        elif self.clickmode == 'graphzoom':
            self.pixmapitem.setCursor(qt.Qt.CrossCursor)
        elif self.clickmode == 'pick':
            self.pixmapitem.setCursor(qt.Qt.CrossCursor)
            self.sigPickerEnabled.emit(True)

    def getClick(self):
        """Return a click point from the graph."""

        # wait for click from user
        qt.QApplication.setOverrideCursor(qt.QCursor(qt.Qt.CrossCursor))
        oldmode = self.clickmode
        self.clickmode = 'viewgetclick'
        while self.clickmode == 'viewgetclick':
            qt.qApp.processEvents()
        self.clickmode = oldmode
        qt.QApplication.restoreOverrideCursor()

        # take clicked point and convert to coords of scrollview
        pt = self.grabpos

        # try to work out in which widget the first point is in
        widget = self.painthelper.pointInWidgetBounds(
            pt.x(), pt.y(), widgets.Graph)
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
                r = axis.plotterToGraphCoords(
                    self.painthelper.widgetBounds(axis), p)

                axesretn.append( (axis.path, r[0]) )

        return axesretn

    def selectedWidgets(self, widgets):
        """Update control items on screen associated with widget.
        Called when widgets have been selected in the tree edit window
        """
        self.updateControlGraphs(widgets)
        self.lastwidgetsselected = widgets

    def updateControlGraphs(self, widgets):
        """Add control graphs for the widgets given."""

        # delete old items from root
        for c in list(self.controlgraphroot.childItems()):
            self.scene.removeItem(c)

        # add each item to the root
        if self.painthelper:
            for widget in widgets:
                cgis = self.painthelper.getControlGraph(widget)
                if cgis:
                    for control in cgis:
                        control.createGraphicsItem(self.controlgraphroot)

class FullScreenPlotWindow(qt.QScrollArea):
    """Window for showing plot in full-screen mode."""

    def __init__(self, document, pagenumber):
        qt.QScrollArea.__init__(self)
        self.setFrameShape(qt.QFrame.NoFrame)
        self.setWidgetResizable(True)

        # window which shows plot
        self.document = document
        pw = self.plotwin = PlotWindow(document, None)
        pw.isfullscreen = True
        pw.pagenumber = pagenumber
        self.setWidget(pw)
        pw.setFocus()

        self.showFullScreen()

        self.toolbar = qt.QToolBar(_("Full screen toolbar"), self)
        self.toolbar.addAction(utils.getIcon("kde-window-close"), _("Close"),
                               self.close)
        for a in ('view.zoom11', 'view.zoomin', 'view.zoomout',
                  'view.zoomwidth', 'view.zoomheight',
                  'view.zoompage', 'view.prevpage', 'view.nextpage'):
            self.toolbar.addAction( pw.vzactions[a] )
        self.toolbar.show()

    def resizeEvent(self, event):
        """Make zoom fit screen."""

        qt.QScrollArea.resizeEvent(self, event)

        # size graph to fill screen
        pagesize = self.document.pageSize(self.plotwin.pagenumber,
                                          dpi=self.plotwin.dpi)
        screensize = self.plotwin.size()

        aspectw = screensize.width() / pagesize[0]
        aspecth = screensize.height() / pagesize[1]

        self.plotwin.zoomfactor = min(aspectw, aspecth)
        self.plotwin.checkPlotUpdate()

    def keyPressEvent(self, event):

        k = event.key()
        if k == qt.Qt.Key_Escape:
            event.accept()
            self.close()
            return
        qt.QScrollArea.keyPressEvent(self, event)
