#    Copyright (C) 2011 Jeremy S. Sanders
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

"""Helper for doing the plotting of the document.
"""

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.utils as utils
from veusz.helpers.qtloops import RecordPaintDevice

class DrawState(object):
    """Each widget plotted has a recorded state in this object."""

    def __init__(self, widget, bounds, clip, helper):
        """Initialise state for widget.
        bounds: tuple of (x1, y1, x2, y2)
        clip: if clipping should be done, another tuple."""

        self.widget = widget
        self.record = RecordPaintDevice(
            helper.pagesize[0], helper.pagesize[1],
            helper.dpi[0], helper.dpi[1])
        self.bounds = bounds
        self.clip = clip

        # controlgraphs belonging to widget
        self.cgis = []

        # list of child widgets states
        self.children = []

class PaintHelper(object):
    """Helper used when painting widgets.

    Provides a QPainter to each widget for plotting.
    Records the controlgraphs for each widget.
    Holds the scaling, dpi and size of the page.
    """

    def __init__(self, pagesize, scaling=1., dpi=(100, 100),
                 directpaint=None):
        """Initialise using page size (tuple of pixelw, pixelh).

        If directpaint is set to a painter, use this directly rather
        than creating separate layers for rendering later. The user
        will need to call restore() on the painter before ending, if
        using this mode, however.
        """

        self.dpi = dpi
        self.scaling = scaling
        self.pixperpt = self.dpi[1] / 72.
        self.pagesize = pagesize

        # keep track of states of all widgets
        self.states = {}

        # axis to plotter mappings
        self.axisplottermap = {}

        self.directpaint = directpaint
        self.directpainting = False

    @property
    def maxsize(self):
        """Return maximum page dimension (using PaintHelper's DPI)."""
        return max(*self.pagesize)

    def sizeAtDpi(self, dpi):
        """Return a tuple size for the page given an output device dpi."""
        return ( int(self.pagesize[0]/self.dpi[0] * dpi),
                 int(self.pagesize[1]/self.dpi[1] * dpi) )

    def updatePageSize(self, pagew, pageh):
        """Update page size to value given (in user text units."""
        self.pagesize = ( setting.Distance.convertDistance(self, pagew),
                          setting.Distance.convertDistance(self, pageh) )

    def painter(self, widget, bounds, clip=None):
        """Return a painter for use when drawing the widget.
        widget: widget object
        bounds: tuple (x1, y1, x2, y2) of widget bounds
        clip: another tuple, if set clips drawing to this rectangle
        """
        s = self.states[widget] = DrawState(widget, bounds, clip, self)
        if widget.parent is not None:
            self.states[widget.parent].children.append(s)

        if self.directpaint:
            # only paint to one output painter
            p = self.directpaint
            if self.directpainting:
                p.restore()
            self.directpainting = True
            p.save()
        else:
            # save to multiple recorded layers
            p = qt4.QPainter(s.record)

        p.scaling = self.scaling
        p.pixperpt = self.pixperpt
        p.pagesize = self.pagesize
        p.dpi = self.dpi[1]

        if clip:
            p.setClipRect(clip)

        return p

    def setControlGraph(self, widget, cgis):
        """Records the control graph list for the widget given."""
        self.states[widget].cgis = cgis

    def renderToPainter(self, root, painter):
        """Render saved output to painter.
        root is the root widget
        """
        self._renderState(self.states[root], painter)

    def _renderState(self, state, painter):
        """Render state to painter."""

        painter.save()
        state.record.play(painter)
        painter.restore()

        for child in state.children:
            self._renderState(child, painter)
