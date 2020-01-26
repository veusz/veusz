#    Copyright (C) 2012 Jeremy S. Sanders
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
###############################################################################

"""Paint fills with extended brush class.

Paints solid, hatching and various qt brushes
"""

from __future__ import division, print_function
import numpy as N
from .. import qtall as qt
import math

from ..helpers.qtloops import plotLinesToPainter, polygonClip

def dumppath(p):
    i =0
    while i < p.elementCount():
        e = p.elementAt(i)
        if e.isLineTo():
            print(" l(%i,%i)" %( e.x, e.y), end=' ')
            i += 1
        elif e.isMoveTo():
            print(" m(%i,%i)" %(e.x, e.y), end=' ')
            i += 1
        else:
            print(" c(%i,%i,%i,%i,%i,%i)" %(e.x, e.y, p.elementAt(i+1).x, p.elementAt(i+1).y, p.elementAt(i+2).x, p.elementAt(i+2).y), end=' ')
            i += 3
    print()

def _hatcher(painter, pen, painterpath, spacing, hatchlist):
    """Draw hatching on painter path given."""

    painter.save()
    painter.setPen(pen)

    # debugging
    # dumppath(painterpath)

    painter.setClipPath(painterpath, qt.Qt.IntersectClip)

    # this is the bounding box of the path
    bb = painter.clipPath().boundingRect()

    for params in hatchlist:
        # scale values
        offsetx, offsety, deltax, deltay = [x*spacing for x in params]

        # compute max number of steps
        numsteps = 0
        if deltax != 0:
            numsteps += int(bb.width() / abs(deltax)) + 4
        if deltay != 0:
            numsteps += int(bb.height() / abs(deltay)) + 4

        # calculate repeat position in x and y
        # we start the positions of lines at multiples of these distances
        # to ensure the pattern doesn't shift if the area does

        # distances between lines
        mag = math.sqrt(deltax**2 + deltay**2)
        # angle to x of perpendicular lines
        theta = math.atan2(deltay, deltax)
        s, c = math.sin(theta), math.cos(theta)
        intervalx = intervaly = 1.
        if abs(c) > 1e-4: intervalx = abs(mag / c)
        if abs(s) > 1e-4: intervaly = abs(mag / s)

        startx = int(bb.left() / intervalx) * intervalx + offsetx
        starty = int(bb.top()  / intervaly) * intervaly + offsety

        # normalise line vector
        linedx, linedy = -deltay/mag, deltax/mag
        # scale of lines from start position
        scale = max( bb.width(), bb.height() ) * 4

        # construct points along lines
        # this is scales to ensure the lines are bigger than the box
        idx = N.arange(-numsteps, numsteps)
        x = idx*deltax + startx
        y = idx*deltay + starty
        x1 = x - scale*linedx
        x2 = x + scale*linedx
        y1 = y - scale*linedy
        y2 = y + scale*linedy

        # plot lines, clipping to bb
        plotLinesToPainter(painter, x1, y1, x2, y2, bb)

    painter.restore()

# list of fill styles
extfillstyles = (
    'solid', 'horizontal', 'vertical', 'cross',
    'forward diagonals', 'backward diagonals',
    'diagonal cross',
    'forward 2', 'backward 2', 'forward 3',
    'backward 3', 'forward 4', 'backward 4',
    'forward 5', 'backward 5',
    'diagonal cross 2', 'diagonal cross 3',
    'diagonal cross 4', 'diagonal cross 5',
    'vertical forward', 'vertical backward',
    'horizontal forward', 'horizontal backward',
    'star',
    'triangles 1', 'triangles 2', 'triangles 3', 'triangles 4',
    'horizontal double', 'vertical double',
    'forward double', 'backward double',
    'double cross', 'double diagonal cross',
    '94% dense', '88% dense', '63% dense', '50% dense',
    '37% dense', '12% dense', '6% dense',
    )

# hatching styles
# map names to lists of (offsetx, offsety, gradx, grady)
_hatchmap = {
    'horizontal': ((0, 0, 0, 1), ),
    'vertical': ((0, 0, 1, 0), ),
    'cross': ( (0, 0, 0, 1), (0, 0, 1, 0), ),
    'forward diagonals': ( (0, 0, 0.7071, -0.7071), ),
    'backward diagonals': ( (0, 0, 0.7071, 0.7071), ),
    'diagonal cross': ( (0, 0, 0.7071, 0.7071), (0, 0, 0.7071, -0.7071), ),
    'forward 2': ( (0, 0, 0.8660, -0.5), ),
    'backward 2': ( (0, 0, 0.8660, 0.5), ),
    'forward 3': ( (0, 0, 0.5, -0.8660), ),
    'backward 3': ( (0, 0, 0.5, 0.8660), ),
    'forward 4': ( (0, 0, 0.2588, -0.9659), ),
    'backward 4': ( (0, 0, 0.2588, 0.9659), ),
    'forward 5': ( (0, 0, 0.9659, -0.2588), ),
    'backward 5': ( (0, 0, 0.9659, 0.2588), ),
    'diagonal cross 2': ( (0, 0, 0.8660, -0.5), (0, 0, 0.8660, 0.5), ),
    'diagonal cross 3': ( (0, 0, 0.5, -0.8660), (0, 0, 0.5, 0.8660), ),
    'diagonal cross 4': ( (0, 0, 0.9659, -0.2588), (0, 0, 0.9659, 0.2588), ),
    'diagonal cross 5': ( (0, 0, 0.2588, 0.9659), (0, 0, 0.2588, -0.9659), ),
    'vertical forward': ( (0, 0, 1, 0), (0, 0, 0.7071, -0.7071), ),
    'vertical backward': ( (0, 0, 1, 0), (0, 0, 0.7071, 0.7071), ),
    'horizontal forward': ( (0, 0, 0, 1), (0, 0, 0.7071, -0.7071), ),
    'horizontal backward': ( (0, 0, 0, 1), (0, 0, 0.7071, 0.7071), ),
    'star': ( (0, 0, 2, 0), (0, 0, 0, 2), (0, 0, -1, 1), (0, 0, 1, 1), ),
    'triangles 1': ( (0, 0, 2, 0), (0, 0, 0, 2), (0, 0, -1, 1),),
    'triangles 2': ( (0, 0, 2, 0), (0, 0, 0, 2), (0, 0, 1, 1),),
    'triangles 3': ( (0, 0, 0, 1), (0, 0, -1, 1), (0, 0, 1, 1), ),
    'triangles 4': ( (0, 0, 1, 0), (0, 0, -1, 1), (0, 0, 1, 1), ),
    'horizontal double': ((0, 0, 0, 1), (0, 0.2828, 0, 1), ),
    'vertical double': ((0, 0, 1, 0), (0.2828, 0, 1, 0), ),
    'forward double': ((0, 0, 0.7071, -0.7071), (0.4,0, 0.7071, -0.7071)),
    'backward double': ((0, 0, 0.7071, 0.7071), (0.4,0, 0.7071, 0.7071)),
    'double cross': ((0, 0, 0, 1), (0, 0.2828, 0, 1),
                     (0, 0, 1, 0), (0.2828, 0, 1, 0),),
    'double diagonal cross': ((0, 0, 0.7071, -0.7071), (0.4,0, 0.7071, -0.7071),
                              (0, 0, 0.7071, 0.7071), (0.4,0, 0.7071, 0.7071)),
    }

# convert qt-specific fill styles into qt styles
_fillcnvt = {
    'solid': qt.Qt.SolidPattern,
    '94% dense': qt.Qt.Dense1Pattern,
    '88% dense': qt.Qt.Dense2Pattern,
    '63% dense': qt.Qt.Dense3Pattern,
    '50% dense': qt.Qt.Dense4Pattern,
    '37% dense': qt.Qt.Dense5Pattern,
    '12% dense': qt.Qt.Dense6Pattern,
    '6% dense': qt.Qt.Dense7Pattern
    }

def brushExtFillPath(painter, extbrush, path, ignorehide=False,
                     stroke=None, dataindex=0):
    """Use an BrushExtended settings object to fill a path on painter.
    If ignorehide is True, ignore the hide setting on the brush object.
    stroke is an optional QPen for stroking outline of path
    """

    if extbrush.hide and not ignorehide:
        if stroke is not None:
            painter.strokePath(path, stroke)
        return

    style = extbrush.style
    if style in _fillcnvt:
        # standard fill: use Qt styles for painting
        color = extbrush.get('color').color(painter, dataindex=dataindex)
        if extbrush.transparency >= 0:
            if extbrush.transparency == 100:
                # skip filling fully transparent areas
                return
            color.setAlphaF((100-extbrush.transparency) / 100.)
        brush = qt.QBrush(color, _fillcnvt[style])
        if stroke is None:
            painter.fillPath(path, brush)
        else:
            painter.save()
            painter.setPen(stroke)
            painter.setBrush(brush)
            painter.drawPath(path)
            painter.restore()

    elif style in _hatchmap:
        # fill with hatching

        if not extbrush.backhide:
            # background brush
            color = extbrush.get('backcolor').color(
                painter, dataindex=dataindex)
            if extbrush.backtransparency > 0:
                color.setAlphaF((100-extbrush.backtransparency) / 100.)
            brush = qt.QBrush(color)
            painter.fillPath(path, brush)

        color = extbrush.get('color').color(painter, dataindex=dataindex)
        if extbrush.transparency >= 0:
            color.setAlphaF((100-extbrush.transparency) / 100.)
        width = extbrush.get('linewidth').convert(painter)
        lstyle, dashpattern = extbrush.get('linestyle')._linecnvt[
            extbrush.linestyle]
        pen = qt.QPen(color, width, lstyle)

        if dashpattern:
            pen.setDashPattern(dashpattern)

        # do hatching with spacing
        spacing = extbrush.get('patternspacing').convert(painter)
        if spacing > 0:
            _hatcher(painter, pen, path, spacing, _hatchmap[style])

        if stroke is not None:
            painter.strokePath(path, stroke)

def brushExtFillPolygon(painter, extbrush, cliprect, polygon, ignorehide=False):
    """Fill a polygon with an extended brush."""
    clipped = qt.QPolygonF()
    polygonClip(polygon, cliprect, clipped)
    path = qt.QPainterPath()
    path.addPolygon(clipped)
    brushExtFillPath(painter, extbrush, path, ignorehide=ignorehide)
