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

class DrawState(object):
    """Each widget plotted has a recorded state in this object."""

    def __init__(self, widget, bounds, clip):
        """Initialise state for widget.
        bounds: tuple of (x1, y1, x2, y2)
        clip: if clipping should be done, another tuple."""

        self.widget = widget
        self.picture = qt4.QPicture()
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

    def __init__(self, pagesize, scaling=1., dpi=75):
        """Initialise using page size (tuple of pixelw, pixelh)."""

        self.dpi = dpi
        self.scaling = scaling
        self.pixperpt = dpi / 72.
        self.pagesize = pagesize

        # keep track of states of all widgets
        self.states = {}

        # axis to plotter mappings
        self.axisplottermap = {}

        # latest painter
        self.lpainter = None

    @property
    def maxsize(self):
        """Return maximum page dimension."""
        return max(*self.pagesize)

    def updatePageSize(self, pagew, pageh):
        """Update page size to value given (in user text units."""
        self.pagesize = ( setting.Distance.convertDistance(self, pagew),
                          setting.Distance.convertDistance(self, pageh) )

    def painter(self, widget, bounds, clip=None):
        self.states[widget] = s = DrawState(widget, bounds, clip)
        if widget.parent is not None:
            self.states[widget.parent].children.append(s)

        p = qt4.QPainter(s.picture)
        self.lpainter = p
        return p

    def setControlGraph(self, widget, cgis):
        self.states[widget].cgis = cgis

    def renderToPainter(self, root, painter):
        """Render saved QPictures to painter.
        root is the root widget
        """
        self._renderState(self.states[root], painter)

    def _renderState(self, state, painter):
        """Render state to painter."""
        if state.clip:
            painter.save()
            painter.setClipRect(state.clip)

        painter.drawPicture(0, 0, state.picture)

        for child in state.children:
            self._renderState(child, painter)

        if state.clip:
            painter.restore()
