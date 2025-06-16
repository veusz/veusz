#    Copyright (C) 2011 Jeremy S. Sanders
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

"""Helper for doing the plotting of the document.
"""

from .. import qtall as qt
from .. import utils

try:
    from ..helpers.recordpaint import RecordPaintDevice
except ImportError:
    # fallback to this if we don't get the native recorded
    def RecordPaintDevice(width, height, dpix, dpiy):
        return qt.QPicture()

class DrawState:
    """Each widget plotted has a recorded state in this object."""

    def __init__(self, widget, bounds, clip, helper):
        """Initialise state for widget.
        bounds: tuple of (x1, y1, x2, y2)
        clip: if clipping should be done, another tuple."""

        self.widget = widget
        self.record = RecordPaintDevice(
            int(helper.pagesize[0]), int(helper.pagesize[1]),
            int(helper.dpi[0]), int(helper.dpi[1]))
        self.bounds = bounds
        self.clip = clip

        # controlgraphs belonging to widget
        self.cgis = []

        # list of child widgets states
        self.children = []

class PainterRoot(qt.QPainter):
    """Base class for painting of widgets."""

    def updateMetaData(self, helper):
        """Update metadeta from helper

        These values are used during plotting."""

        self.helper = helper
        self.document = helper.document
        self.colors = self.document.evaluate.colors
        self.scaling = helper.scaling
        self.pixperpt = helper.pixperpt
        self.dpi = helper.dpi[1]
        self.pagesize = helper.pagesize
        self.maxdim = max(*self.pagesize)
        self.textrects = helper.textrects

    def docColor(self, name):
        """Return color from document."""
        return self.colors.get(name)

    def docColorAuto(self, index):
        """Return automatic doc color given index."""
        return self.colors.getIndex(index+1)

    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass

class DirectPainter(PainterRoot):
    """Painter class for direct painting with PaintHelper below.
    """

class RecordPainter(PainterRoot):
    """This is the painter subclass for rendering in Veusz, which keeps
    track of which widget is being painted."""

    def __init__(self, widget, outdev):
        PainterRoot.__init__(self, outdev)
        self.widget = widget

    def __enter__(self):
        #print ' '*len(self.helper.widgetstack), self.widget
        self.helper.widgetstack.append(self.widget)

    def __exit__(self, exc_type, exc_value, traceback):
        self.helper.widgetstack.pop()

class PaintHelper:
    """Helper used when painting widgets.

    Designed to be used for a particular page.

    Provides a QPainter/RecordPainter to each widget for plotting.
    Records the controlgraphs for each widget.
    Holds the scaling, dpi and size of the page.

    """

    def __init__(self, document, pagesize,
                 scaling=1, devicepixelratio=1, dpi=(100, 100),
                 directpaint=None):
        """
        pagesize: tuple (pixelw, pixelh), which can be float.
         This is the page size in the coordinates presented to graph drawing.
        scaling: scaling from graph coordinates to native coordinates
        devicepixelratio: high DPI scaling factor from logical to native pixels
        dpi: tuple of X and Y dpi for graph coordinates
        directpaint: use this painter directly, rather than using RecordPainter
          to store each widget painting
        """

        self.document = document
        self.dpi = dpi
        self.scaling = scaling
        # scaling factor, excluding high-DPI factor (for controlgraphs)
        self.cgscale = scaling / devicepixelratio
        self.devicepixelratio = devicepixelratio
        self.pixperpt = self.dpi[1] / 72.

        # page size in native pixels (without default zoom)
        self.rawpagesize = max(pagesize[0], 1), max(pagesize[1], 1)
        # page size in graph pixels
        self.pagesize = self.rawpagesize[0]/scaling, self.rawpagesize[1]/scaling

        # keep track of states of all widgets
        # maps (widget, layer) to DrawState
        self.states = {}

        # axis to plotter mappings
        self.axisplottermap = {}
        self.plotteraxismap = {}

        # whether to directly render to a painter or make new layers
        self.directpaint = directpaint

        # state for root widget
        self.rootstate = None

        # keep track of last widget being plotted
        self.widgetstack = []

        # current index for each plotter (if wanting automatic colors)
        self.autoplottercount = 0
        self.autoplottermap = {}

        # to avoid overlapping text
        self.textrects = utils.RectangleOverlapTester()

    @property
    def maxdim(self):
        """Return maximum page dimension (using PaintHelper's DPI)."""
        return max(*self.pagesize)

    def sizeAtDpi(self, dpi):
        """Return a tuple size for the page given an output device dpi."""
        return (
            int(self.pagesize[0]/self.dpi[0] * dpi),
            int(self.pagesize[1]/self.dpi[1] * dpi)
        )

    def painter(self, widget, bounds, clip=None, layer=None):
        """Return a painter for use when drawing the widget.
        widget: widget object
        bounds: tuple (x1, y1, x2, y2) of widget bounds
        clip: a QRectF, if set
        layer: layer to plot widget, or None to get next automatically
        """

        # automatically add a layer if not given
        if layer is None:
            layer = 0
            while (widget, layer) in self.states:
                layer += 1

        s = self.states[(widget, layer)] = DrawState(widget, bounds, clip, self)

        if self.widgetstack:
            self.states[(self.widgetstack[-1], 0)].children.append(s)
        else:
            self.rootstate = s

        if self.directpaint is None:
            # save to multiple recorded layers
            p = RecordPainter(widget, s.record)
        else:
            # only paint to one output painter
            p = self.directpaint
            # make sure we get the same state each time
            p.restore()
            p.save()

        if clip is not None:
            # have to clip before scaling, avoiding a qt bug where the clipping
            # seems to happen in the wrong place
            p.setClipRect(qt.QRectF(
                clip.topLeft()*self.scaling, clip.bottomRight()*self.scaling))

        # scale (used for zooming)
        if self.scaling != 1:
            p.scale(self.scaling, self.scaling)

        p.updateMetaData(self)

        return p

    def setControlGraph(self, widget, cgis):
        """Records the control graph list for the widget given."""
        self.states[(widget,0)].cgis = cgis

    def getControlGraph(self, widget):
        """Return control graph for widget (or None)."""
        try:
            return self.states[(widget,0)].cgis
        except KeyError:
            return None

    def renderToPainter(self, painter):
        """Render saved output to painter.
        """
        self._renderState(self.rootstate, painter)

    def _renderState(self, state, painter, indent=0):
        """Render state to painter."""

        painter.save()
        state.record.play(painter)
        painter.restore()

        for child in state.children:
            #print '  '*indent, child.widget
            self._renderState(child, painter, indent=indent+1)

    def identifyWidgetAtPoint(self, x, y, antialias=True):
        """What widget has drawn at the point x,y?

        Returns the widget drawn last on the point, or None if it is
        an empty part of the page.
        root is the root widget to recurse from
        if antialias is true, do test for antialiased drawing
        """

        # reset text rectangles for painting so not to lose text
        self.textrects.reset()

        # convert screen to bitmap coordinates
        x *= self.devicepixelratio
        y *= self.devicepixelratio

        # make a small image filled with a specific color
        box = 3
        specialcolor = qt.QColor(254, 255, 254)
        origpix = qt.QPixmap(2*box+1, 2*box+1)
        origpix.fill(specialcolor)
        origimg = origpix.toImage()
        # store most recent widget here
        lastwidget = [None]

        def rendernextstate(state):
            """Recursively draw painter.

            Checks whether drawing a widgetchanges the small image
            around the point given.
            """

            pixmap = qt.QPixmap(origpix)
            painter = qt.QPainter(pixmap)
            painter.setRenderHint(qt.QPainter.RenderHint.Antialiasing, antialias)
            painter.setRenderHint(qt.QPainter.RenderHint.TextAntialiasing, antialias)
            # this makes the small image draw from x-box->x+box, y-box->y+box
            # translate would get overriden by coordinate system playback
            painter.setWindow(int(x-box), int(y-box), int(box*2+1), int(box*2+1))
            state.record.play(painter)
            painter.end()
            newimg = pixmap.toImage()

            if newimg != origimg:
                lastwidget[0] = state.widget

            for child in state.children:
                rendernextstate(child)

        rendernextstate(self.rootstate)
        widget = lastwidget[0]

        # need to re-render 3d scene to look for clicks
        if widget and widget.typename == 'scene3d':
            bounds = [0,0,100,100]
            for w in self.states:
                if w[0] is widget:
                    bounds = self.states[w].bounds
            return widget.identifyWidgetAtPoint(self, bounds, self.scaling, x, y)
        else:
            return widget

    def pointInWidgetBounds(self, x, y, widgettype):
        """Which graph widget plots at point x,y?

        Recurse from widget root
        widgettype is the class of widget to get
        """

        widget = [None]

        def recursestate(state):
            if isinstance(state.widget, widgettype):
                b = state.bounds
                if x >= b[0] and y >= b[1] and x <= b[2] and y <= b[3]:
                    # most recent widget drawing on point
                    widget[0] = state.widget

            for child in state.children:
                recursestate(child)

        recursestate(self.rootstate)
        return widget[0]

    def widgetBounds(self, widget):
        """Return bounds of widget."""
        return self.states[(widget,0)].bounds

    def widgetBoundsIterator(self, widgettype=None):
        """Returns bounds for each widget.
        Set widgettype to be a widget type to filter returns
        Yields (widget, bounds)
        """

        # this is a recursive algorithm turned into an iterative one
        # which makes creation of a generator easier
        stack = [self.rootstate]
        while stack:
            state = stack[0]
            if widgettype is None or isinstance(state.widget, widgettype):
                yield state.widget, state.bounds
            # remove the widget itself from the stack and insert children
            stack = state.children + stack[1:]

    def autoColorIndex(self, key):
        """Return automatic color index for key given."""
        if key not in self.autoplottermap:
            self.autoplottermap[key] = self.autoplottercount
            self.autoplottercount += 1
        return self.autoplottermap[key]
