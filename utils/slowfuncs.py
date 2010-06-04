#    Copyright (C) 2009 Jeremy S. Sanders
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

# $Id$

"""
These are slow versions of routines also implemented in C++
"""

from itertools import izip
import veusz.qtall as qt4
import numpy as N

def addNumpyToPolygonF(poly, *args):
    """Add a set of numpy arrays to a QPolygonF.

    The first argument is the QPolygonF to add to
    Subsequent arguments should be pairs of x and y coordinate arrays
    """
    
    # we stick the datasets next to each other horizontally, then
    # reshape it into an array of x, y pairs
    minlen = min([x.shape[0] for x in args])
    cols = N.hstack([N.reshape(x[:minlen], (minlen, 1)) for x in args])
    points = N.reshape(cols, (minlen*len(args)/2, 2))

    # finally draw the points
    pappend = poly.append
    qpointf = qt4.QPointF
    for p in points:
        pappend( qpointf(*p) )

def plotPathsToPainter(painter, path, x, y):
    """Plot array of x, y points."""

    # more copying things into local variables
    t = painter.translate
    d = painter.drawPath
    for xp, yp in izip(x, y):
        t(xp, yp)
        d(path)
        t(-xp, -yp)

def plotLinesToPainter(painter, x1, y1, x2, y2):
    """Plot lines given in numpy arrays to painter."""
    lines = []
    lappend = lines.append
    qlinef = qt4.QLineF
    for p in izip(x1, y1, x2, y2):
        lappend( qlinef(*p) )
    painter.drawLines(lines)
