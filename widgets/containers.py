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
 GridContainer: Class to plot a grid of plots
"""

import widget
import widgetfactory
import region

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
        for y in range(h):
            for x in range(w):
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
        for y in range(h):
            for x in range(w):
                self.setAlloced(x+c, y+r)

    def add(self, width, height):
        """Add a block of width x height, returning position as tuple."""
        if self.columns != None:
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
    
class GridContainer(widget.Widget):
    """Class to hold plots in a grid arrangement.

    The idea is we either specify how many rows or columns to use.
    If we specify no of rows, then we fill vertically until we exceed rows,
    then we add another column.
    The same is true if cols is specified.
    """

    typename='grid'
    allowusercreation=True
    description='Arrange graphs in a grid'
    allowedparenttypes=[region.Region]

    def __init__(self, parent, rows=None, columns=None, name=None):
        """Initialise the container.

        Specify rows to grow horizontal, but fill vertically
        Specify columnss to grow vertically, but fill horizontally

        By default columns will be 1.
        If both are specified then we ignore rows
        """

        widget.Widget.__init__(self, parent, name=name)
        self.readPrefs()

        self.child_dimensions = {}
        self.changeShape(rows, columns)

        # maintain copy of children to check for mods
        self._old_children = list(self.children)

    def changeShape(self, rows=None, columns=None):
        """Modify the shape of the container (as specified above)."""
        
        if rows == None and columns == None:
            columns = 1

        self.rows = rows
        self.columns = columns

        # force a recalculation of the positions of the children
        self.recalc = 1

    def _recalcPositions(self):
        """(internal) recalculate the positions of the children."""

        # class to handle management
        ge = _gridengine(self.columns, self.rows)

        # iterate over children, and collect dimensions of children
        # (a tuple width nocols x norows)

        childrenposns = []
        for c in self.children:
            name = c.getName()

            if name not in self.child_dimensions:
                self.child_dimensions[name] = (1, 1)

            childrenposns.append( ge.add( * self.child_dimensions[name] ) )

        nocols, norows = ge.getAllocedDimensions()

        # exit if there aren't any children
        if nocols == 0 or norows == 0:
            return

        # fractions per col and row
        invc, invr = 1./nocols, 1./norows

        # iterate over children, and modify positions
        for child, pos in zip(self.children, childrenposns):
            dim = self.child_dimensions[child.getName()]
            child.position = ( pos[0]*invc, pos[1]*invr,
                               (pos[0]+dim[0])*invc, (pos[1]+dim[1])*invr )

    def draw(self, posn, painter):
        """Draws the widget's children."""

        # if the contents have been modified, recalculate the positions
        if self.children != self._old_children:
            self._old_children = list(self.children)
            self._recalcPositions()

        widget.Widget.draw(self, posn, painter)

# allow the factory to instantiate a grid
widgetfactory.thefactory.register( GridContainer )

       
