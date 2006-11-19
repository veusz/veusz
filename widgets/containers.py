# containers.py
# plot containers, for holding other plotting elements

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

"""Module containing container widget classes.

Classes include
 Grid: Class to plot a grid of plots
"""

import itertools

import veusz.document as document
import veusz.setting as setting

import widget
import axis
import page
import graph

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
        for y in xrange(h):
            for x in xrange(w):
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
        for y in xrange(h):
            for x in xrange(w):
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
    description='Arrange graphs in a grid'
    allowedparenttypes=[page.Page]

    def __init__(self, parent, name=None):
        """Initialise the container.

        Specify rows to grow horizontal, but fill vertically
        Specify columnss to grow vertically, but fill horizontally

        By default columns will be 1.
        If both are specified then we ignore rows
        """

        widget.Widget.__init__(self, parent, name=name)

        s = self.settings
        s.add(setting.Int('rows', 2,
                          descr = 'Number of rows in grid',
                          usertext='Number of rows') )
        s.add(setting.Int('columns', 2,
                          descr = 'Number of columns in grid',
                          usertext='Number of columns') )

        s.add( setting.Distance( 'leftMargin', '1.7cm', descr=
                                 'Distance from left of grid to '
                                 'edge of page',
                                 usertext='Left margin') )
        s.add( setting.Distance( 'rightMargin', '0.1cm', descr=
                                 'Distance from right of grid to '
                                 'edge of page',
                                 usertext='Right margin') )
        s.add( setting.Distance( 'topMargin', '0.1cm', descr=
                                 'Distance from top of grid to '
                                 'edge of page',
                                 usertext='Top margin') )
        s.add( setting.Distance( 'bottomMargin', '1.7cm', descr=
                                 'Distance from bottom of grid'
                                 'to edge of page',
                                 usertext='Bottom margin') )

        # we're not descended from
        if type(self) == Grid:
            self.readDefaults()

        self.addAction( widget.Action('zeroMargins', self.actionZeroMargins,
                                      descr = 'Zero margins of graphs in grid',
                                      usertext = 'Zero margins') )

        self.lastdimensions = (-1, -1)
        self.childpositions = {}

        # maintain copy of children to check for mods
        self._old_children = list(self.children)

    def _recalcPositions(self):
        """(internal) recalculate the positions of the children."""

        # class to handle management
        ge = _gridengine(self.settings.columns, self.settings.rows)

        # copy children, and remove any which are axes
        children = [i for i in self.children if not isinstance(i, axis.Axis)]
        child_dimensions = {}
        child_posns = {}
        for c in children:
            dims = (1, 1)
            child_dimensions[c] = dims
            child_posns[c] = ge.add(*dims)

        nocols, norows = ge.getAllocedDimensions()

        # exit if there aren't any children
        if nocols == 0 or norows == 0:
            return

        # fractions per col and row
        invc, invr = 1./nocols, 1./norows

        # iterate over children, and modify positions
        self.childpositions.clear()
        for child in children:
            dims = child_dimensions[child]
            pos = child_posns[child]
            self.childpositions[child] = ( pos[0]*invc, pos[1]*invr,
                                           (pos[0]+dims[0])*invc,
                                           (pos[1]+dims[1])*invr )

    def actionZeroMargins(self):
        """Zero margins of plots inside this grid."""

        operations = []
        for c in self.children:
            if isinstance(c, graph.Graph):
                s = c.settings
                for v in ('leftMargin', 'topMargin', 'rightMargin', 'bottomMargin'):
                    operations.append( document.OperationSettingSet(s.get(v), '0cm') )
                    
        self.document.applyOperation( document.OperationMultiple(operations, descr='zero margins') )

    def draw(self, parentposn, painter, outerbounds=None):
        """Draws the widget's children."""

        s = self.settings

        # if the contents have been modified, recalculate the positions
        dimensions = (s.columns, s.rows)
        if ( self.children != self._old_children or
             self.lastdimensions != dimensions ):
            
            self._old_children = list(self.children)
            self._recalcPositions()
            self.lastdimensions = dimensions

        margins = ( s.get('leftMargin').convert(painter),
                    s.get('topMargin').convert(painter),
                    s.get('rightMargin').convert(painter),
                    s.get('bottomMargin').convert(painter) )

        bounds = self.computeBounds(parentposn, painter, margins=margins)
        for c in self.children:
            if not isinstance(c, axis.Axis):
                # save old position, then update with calculated
                oldposn = c.position
                c.position = self.childpositions[c]
                # draw widget
                c.draw(bounds, painter, outerbounds=parentposn)
                # restore position
                c.position = oldposn

        # do not call widget.Widget.draw

# allow the factory to instantiate a grid
document.thefactory.register( Grid )
       
class Splitter(widget.Widget):
    """Class to hold plots with the page split into bits
    """

    typename='splitter'
    allowusercreation=True
    description='Split page into separate graphs'
    allowedparenttypes=[page.Page]

    def __init__(self, *args, **optargs):
        """Initialise the Splitter.

        User can specify 'Horizontal' or 'Vertical' to split
        in different directions."""
        
        widget.Widget.__init__(self, *args, **optargs)

        s = self.settings
        s.add( setting.Choice('direction',
                              ['horizontal', 'vertical'],
                              'vertical',
                              descr = 'Split horizonally or vertically',
                              usertext='Split direction') )

        s.add( setting.FloatList('sizes',
                                 [],
                                 descr = 'List of relative sizes for each graph'
                                 'in the splitter.\nThese are in terms of the '
                                 'ratio to the total.\nAssumed to be 1 by default.',
                                 usertext='Relative sizes'))
        
        s.add( setting.Distance( 'leftMargin', '1.7cm', descr=
                                 'Distance from left of splitter to '
                                 'edge of page',
                                 usertext='Left margin') )
        s.add( setting.Distance( 'rightMargin', '0.1cm', descr=
                                 'Distance from right of splitter to '
                                 'edge of page',
                                 usertext='Right margin') )
        s.add( setting.Distance( 'topMargin', '0.1cm', descr=
                                 'Distance from top of splitter to '
                                 'edge of page',
                                 usertext='Top margin') )
        s.add( setting.Distance( 'bottomMargin', '1.7cm', descr=
                                 'Distance from bottom of splitter'
                                 'to edge of page',
                                 usertext='Bottom margin') )

        if type(self) == Splitter:
            self.readDefaults()

        self.addAction( widget.Action('zeroMargins', self.actionZeroMargins,
                                      descr = 'Zero margins of graphs in splitter',
                                      usertext = 'Zero margins') )

    def actionZeroMargins(self):
        """Zero margins of plots inside this splitter."""

        operations = []
        for c in self.children:
            if isinstance(c, graph.Graph):
                s = c.settings
                for v in ('leftMargin', 'topMargin', 'rightMargin', 'bottomMargin'):
                    operations.append( document.OperationSettingSet(s.get(v), '0cm') )
                    
        self.document.applyOperation( document.OperationMultiple(operations, descr='zero margins') )

    def draw(self, parentposn, painter, outerbounds=None):
        """Draws the widget's children."""

        s = self.settings

        margins = ( s.get('leftMargin').convert(painter),
                    s.get('topMargin').convert(painter),
                    s.get('rightMargin').convert(painter),
                    s.get('bottomMargin').convert(painter) )

        graphs = [c for c in self.children if not isinstance(c, axis.Axis)]
        sizes = s.sizes

        # add up total size of graphs in splitter
        totalsize = 0.
        for i in xrange(len(graphs)):
            if i < len(sizes):
                totalsize += sizes[i]
            else:
                totalsize += 1.0

        # work out how to split margins
        if s.direction == 'horizontal':
            splitindices = (0, 2)
        else:
            splitindices = (1, 3)

        # get bounds of splitter, and draw children who aren't axes
        bounds = self.computeBounds(parentposn, painter, margins=margins)

        runningsize = 0.
        for i, child in enumerate(graphs):
            if i < len(sizes):
                thissize = sizes[i]
            else:
                thissize = 1.

            oldposition = child.position
            
            # work out position of child graph in splitter
            scaling = 1./totalsize
            childbounds = [0., 0., 1., 1.]
            childbounds[splitindices[0]] = scaling*runningsize
            childbounds[splitindices[1]] = scaling*(runningsize+thissize)

            child.position = childbounds

            # actually draw graph
            child.draw(bounds, painter, outerbounds=parentposn)

            # retain position again
            child.position = oldposition

            runningsize += thissize

# allow the factory to instantiate a splitter
document.thefactory.register( Splitter )

