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

"""The grid class allows graphs to be arranged in a regular grid.
The graphs may share axes if they are stored in the grid widget.
"""

from __future__ import division
from ..compat import crange
from .. import document
from .. import setting
from .. import qtall as qt

from . import widget
from . import graph
from . import controlgraph

def _(text, disambiguation=None, context='Grid'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class _gridengine:
    """Internal class to build up grid of widgets."""

    def __init__(self, columns, rows):
        """Initialise allocater, with N columns or rows (other set to None)."""

        # we don't allocate space until it is used
        self.alloced = []
        self.rows = rows
        self.columns = columns

        # starting position
        self.row = 0
        self.col = 0

    def isAlloced(self, c, r):
        """Returns whether element (c,r) allocated."""
        if r >= len(self.alloced):
            return False
        row = self.alloced[r]
        if c >= len( row ):
            return False
        else:
            return row[c]

    def isAllocedBlock(self, c, r, w, h):
        """Is the block (c,r) -> (c+w,r+h) allocated?"""
        for y in crange(h):
            for x in crange(w):
                if self.isAlloced(c+x, y+r):
                    return True
        return False

    def setAlloced(self, c, r):
        """Set element (c,r) as allocated."""
        while r >= len(self.alloced):
            self.alloced.append( [] )
        row = self.alloced[r]
        while c >= len(row):
            row.append( False )

        row[c] = True

    def setAllocedBlock(self, c, r, w, h):
        """Set block (c,r)->(c+w,r+h) as allocated."""
        for y in crange(h):
            for x in crange(w):
                self.setAlloced(x+c, y+r)

    def add(self, width, height):
        """Add a block of width x height, returning position as tuple."""
        if self.columns is not None:
            # wrap around if item too wide
            # (providing we didn't request more columns than we have -
            #  in that case we ignore the request)
            if ((self.col + width) > self.columns) and (width <= self.columns):
                self.col = 0
                self.row += 1

            # increase column until we can allocate the block
            # if we run out of columns, move to the next row
            while self.isAllocedBlock(self.col, self.row, width, height):
                self.col += 1
                if (self.col + width > self.columns) and \
                       (width <= self.columns):
                    self.col = 0
                    self.row += 1

            # save position
            c = self.col
            r = self.row
            self.col += width

        else:
            # work in row based layout now
            if ((self.row + height) > self.rows) and (height <= self.rows):
                self.row = 0
                self.col += 1

            # increase row until we can allocate the next block
            # if we run out of rows, move to the next column
            while self.isAllocedBlock(self.col, self.row, width, height):
                self.row += 1
                if (self.row + height > self.rows) and (height <= self.rows):
                    self.row = 0
                    self.col += 1
                    
            # save position
            c = self.col
            r = self.row
            self.row += height

        # allocate and return block position
        self.setAllocedBlock(c, r, width, height)
        return (c, r)

    def getAllocedDimensions(self):
        """Return the columns x rows allocated."""
        # assumes blocks don't get unset
        h = len(self.alloced)
        w = 0
        for l in self.alloced:
            w = max(w, len(l))
        return (w, h)
    
class Grid(widget.Widget):
    """Class to hold plots in a grid arrangement.

    The idea is we either specify how many rows or columns to use.
    If we specify no of rows, then we fill vertically until we exceed rows,
    then we add another column.
    The same is true if cols is specified.
    """

    typename='grid'
    allowusercreation=True
    description=_('Arrange graphs in a grid')

    def __init__(self, parent, name=None):
        """Initialise the grid.
        """

        widget.Widget.__init__(self, parent, name=name)

        self.addAction( widget.Action(
                'zeroMargins', self.actionZeroMargins,
                descr = _('Zero margins of graphs in grid'),
                usertext = _('Zero margins')) )

        # calculated positions for children
        self.childpositions = {}

        # watch for changes to these variables to decide whether to
        # recalculate positions
        self.lastdimensions = None
        self.lastscalings = None
        self.lastchildren = None

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        s.add(setting.Int('rows', 2,
                          descr = _('Number of rows in grid'),
                          usertext=_('Number of rows')) )
        s.add(setting.Int('columns', 2,
                          descr = _('Number of columns in grid'),
                          usertext=_('Number of columns')) )

        s.add( setting.FloatList(
                'scaleRows',
                [],
                descr = _('Row scaling factors. A sequence'
                          ' of values\nby which to scale rows '
                          'relative to each other.'),
                usertext=_('Row scalings')) )
        s.add( setting.FloatList(
                'scaleCols',
                [],
                descr = _('Column scaling factors. A sequence'
                          ' of values\nby which to scale columns'
                          ' relative to each other.'),
                usertext=_('Column scalings')) )

        s.add( setting.Distance(
                'leftMargin', '1.7cm',
                descr=_('Distance from left of grid to edge of page'),
                usertext=_('Left margin'),
                formatting=True) )
        s.add( setting.Distance(
                'rightMargin', '0.2cm',
                descr=_('Distance from right of grid to edge of page'),
                usertext=_('Right margin'),
                formatting=True) )
        s.add( setting.Distance(
                'topMargin', '0.2cm',
                descr=_('Distance from top of grid to edge of page'),
                usertext=_('Top margin'),
                formatting=True) )
        s.add( setting.Distance(
                'bottomMargin', '1.7cm',
                descr=_('Distance from bottom of grid to edge of page'),
                usertext=_('Bottom margin'),
                formatting=True) )
        s.add( setting.Distance(
                'internalMargin', '0cm',
                descr=_('Gap between grid members'),
                usertext=_('Internal margin'),
                formatting=True) )

    @classmethod
    def allowedParentTypes(klass):
        from . import page
        return (page.Page, Grid)

    @property
    def userdescription(self):
        """User friendly description."""
        s = self.settings
        return "%(rows)i rows, %(columns)i columns"  % s

    def _recalcPositions(self):
        """(internal) recalculate the positions of the children."""

        # class to handle management
        ge = _gridengine(self.settings.columns, self.settings.rows)

        # copy children, and remove any which are axes
        children = [ c for c in self.children if
                     not c.isaxis ]
        child_dimensions = {}
        child_posns = {}
        for c in children:
            dims = (1, 1)
            child_dimensions[c] = dims
            child_posns[c] = ge.add(*dims)

        nocols, norows = ge.getAllocedDimensions()
        self.dims = (nocols, norows)

        # exit if there aren't any children
        if nocols == 0 or norows == 0:
            return

        # get total scaling factors for cols
        scalecols = list(self.settings.scaleCols[:nocols])
        scalecols += [1.]*(nocols-len(scalecols))
        totscalecols = sum(scalecols)
        if totscalecols == 0.:
            totscalecols = 1.

        # fractional starting positions of columns
        last = 0.
        startcols = [last]
        for scale in scalecols:
            last += scale/totscalecols
            startcols.append(last)

        # similarly get total scaling factors for rows
        scalerows = list(self.settings.scaleRows[:norows])
        scalerows += [1.]*(norows-len(scalerows))
        totscalerows = sum(scalerows)
        if totscalerows == 0.:
            totscalerows = 1.

        # fractional starting positions of rows
        last = 0.
        startrows = [last]
        for scale in scalerows:
            last += scale/totscalerows
            startrows.append(last)

        # iterate over children, and modify positions
        self.childpositions.clear()
        for child in children:
            dims = child_dimensions[child]
            pos = child_posns[child]
            self.childpositions[child] = (
                ( pos[0],
                  pos[1] ),
                ( startcols[pos[0]],
                  startrows[pos[1]],
                  startcols[pos[0]+dims[0]],
                  startrows[pos[1]+dims[1]] ),
                )

    def actionZeroMargins(self):
        """Zero margins of plots inside this grid."""

        operations = []
        for c in self.children:
            if isinstance(c, graph.Graph):
                s = c.settings
                for v in ('leftMargin', 'topMargin', 'rightMargin',
                          'bottomMargin'):
                    operations.append(
                        document.OperationSettingSet(s.get(v), '0cm') )
                    
        self.document.applyOperation(
            document.OperationMultiple(operations, descr='zero margins') )

    def _drawChild(self, phelper, child, bounds, parentposn):
        """Draw child at correct position, with correct bounds."""

        # default positioning
        coutbound = newbounds = parentposn

        if child in self.childpositions:
            intmargin = self.settings.get('internalMargin').convert(phelper)
            cidx, cpos = self.childpositions[child]

            # calculate size after margins
            dx = bounds[2]-bounds[0]
            dy = bounds[3]-bounds[1]
            marx = intmargin*max(0, self.dims[0]-1)
            mary = intmargin*max(0, self.dims[1]-1)
            if dx > marx and dy > mary:
                dx -= marx
                dy -= mary
            else:
                # margins too big
                intmargin = 0

            # bounds for child
            newbounds = [ bounds[0]+dx*cpos[0]+intmargin*cidx[0],
                          bounds[1]+dy*cpos[1]+intmargin*cidx[1],
                          bounds[0]+dx*cpos[2]+intmargin*cidx[0],
                          bounds[1]+dy*cpos[3]+intmargin*cidx[1] ]
            # bounds the axes can spread into
            coutbound = list(newbounds)

            # adjust outer bounds to half the internal margin space
            if cidx[0] > 0:
                coutbound[0] -= intmargin/2.
            if cidx[1] > 0:
                coutbound[1] -= intmargin/2.
            if cidx[0] < self.dims[0]-1:
                coutbound[2] += intmargin/2.
            if cidx[1] < self.dims[1]-1:
                coutbound[3] += intmargin/2.

            # work out bounds for graph in box
            # this is the space available for axes, etc
            # FIXME: should consider case if no graphs to side
            if cidx[0] == 0:
                coutbound[0] = parentposn[0]
            if cidx[1] == 0:
                coutbound[1] = parentposn[1]
            if cidx[0] == self.dims[0]-1:
                coutbound[2] = parentposn[2]
            if cidx[1] == self.dims[1]-1:
                coutbound[3] = parentposn[3]

        # draw widget
        child.draw(newbounds, phelper, outerbounds=coutbound)

    def getMargins(self, painthelper):
        """Use settings to compute margins."""
        s = self.settings
        return ( s.get('leftMargin').convert(painthelper),
                 s.get('topMargin').convert(painthelper),
                 s.get('rightMargin').convert(painthelper),
                 s.get('bottomMargin').convert(painthelper) )

    def draw(self, parentposn, phelper, outerbounds=None):
        """Draws the widget's children."""

        s = self.settings

        # if the contents have been modified, recalculate the positions
        dimensions = (s.columns, s.rows)
        scalings = (s.scaleRows, s.scaleCols)
        if ( self.children != self.lastchildren or
             self.lastdimensions != dimensions or
             self.lastscalings != scalings ):
            
            self._recalcPositions()
            self.lastchildren = list(self.children)
            self.lastdimensions = dimensions
            self.lastscalings = scalings

        bounds = self.computeBounds(parentposn, phelper)
        maxbounds = self.computeBounds(parentposn, phelper, withmargin=False)

        painter = phelper.painter(self, bounds)

        # controls for adjusting grid margins
        phelper.setControlGraph(self,[
                controlgraph.ControlMarginBox(self, bounds, maxbounds, phelper)])

        with painter:
            for child in self.children:
                if not child.isaxis:
                    self._drawChild(phelper, child, bounds, parentposn)

        # do not call widget.Widget.draw, do not collect 200 pounds
        pass

    def updateControlItem(self, cgi):
        """Grid resized or moved - call helper routine to move self."""
        cgi.setWidgetMargins()

# allow the factory to instantiate a grid
document.thefactory.register( Grid )
