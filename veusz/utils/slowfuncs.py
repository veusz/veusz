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

"""
These are slow versions of routines also implemented in C++
"""

from __future__ import division
from itertools import izip, count
import sys
import struct

import veusz.qtall as qt4
import numpy as N

sys.stderr.write("Warning: Using slow substitutes for some functions. "
                 "Bezier curves are disabled.\n"
                 "Compile helpers to avoid this warning - "
                 "see INSTALL document\n")

def addNumpyToPolygonF(poly, *args):
    """Add a set of numpy arrays to a QPolygonF.

    The first argument is the QPolygonF to add to
    Subsequent arguments should be pairs of x and y coordinate arrays
    """
    
    # we stick the datasets next to each other horizontally, then
    # reshape it into an array of x, y pairs
    minlen = min([x.shape[0] for x in args])
    cols = N.hstack([N.reshape(x[:minlen], (minlen, 1)) for x in args])
    points = N.reshape(cols, (minlen*len(args)//2, 2))

    # finally draw the points
    pappend = poly.append
    qpointf = qt4.QPointF
    for p in points:
        pappend( qpointf(*p) )

def addNumpyPolygonToPath(path, clip, *args):
    """Add a set of polygons to a path, clipping (optionally)."""
    if clip is None:
        clip = qt4.QRectF(qt4.QPointF(-32767,-32767),qt4.QPointF(32767,32767))
    else:
        clip = qt4.QRectF(clip)

    for i in count():
        p = qt4.QPolygonF()
        dobreak = True
        for c in xrange(len(args)//2):
            if i < len(args[c*2]) and i < len(args[c*2+1]):
                p.append(qt4.QPointF(args[c*2][i], args[c*2+1][i]))
                dobreak = False
        if dobreak:
            break
        else:
            if clip is None:
                path.addPolygon(p)
            else:
                cp = qt4.QPolygonF()
                polygonClip(p, clip, cp)
                path.addPolygon(cp)
            path.closeSubpath()

def plotPathsToPainter(painter, path, x, y, scaling=None,
                       clip=None, colorimg=None):
    """Plot array of x, y points."""

    if clip is None:
        clip = qt4.QRectF(qt4.QPointF(-32767,-32767),qt4.QPointF(32767,32767))
    else:
        clip = qt4.QRectF(clip)

    # adjust bounding box by size of path
    pathbox = path.boundingRect()
    clip.adjust(pathbox.left(), pathbox.top(),
                pathbox.bottom(), pathbox.right())

    # draw the paths
    numpts = min(len(x), len(y))
    if scaling is not None:
        numpts = min(numpts, len(scaling))
    if colorimg is not None:
        numpts = min(numpts, colorimg.width())

    origtrans = painter.worldTransform()
    for i in xrange(numpts):
        pt = qt4.QPointF(x[i], y[i])
        if clip.contains(pt):
            painter.translate(pt)
            # scale if wanted
            if scaling is not None:
                painter.scale(scaling[i], scaling[i])
            # set color if given
            if colorimg is not None:
                b = qt4.QBrush( qt4.QColor.fromRgba(colorimg.pixel(i, 0)) )
                painter.setBrush(b)

            painter.drawPath(path)
            painter.setWorldTransform(origtrans)

def plotLinesToPainter(painter, x1, y1, x2, y2, clip=None, autoexpand=True):
    """Plot lines given in numpy arrays to painter."""
    lines = []
    lappend = lines.append
    qlinef = qt4.QLineF
    for p in izip(x1, y1, x2, y2):
        lappend( qlinef(*p) )
    painter.drawLines(lines)

def plotClippedPolyline(painter, cliprect, pts, autoexpand=True):
    """Draw a polyline, trying to clip the points.
    
    The python version does nothing really as it would be too hard.
    """

    ptsout = qt4.QPolygonF()
    for p in pts:
        x = max( min(p.x(), 32767.), -32767.)
        y = max( min(p.y(), 32767.), -32767.)
        ptsout.append( qt4.QPointF(x, y) )
        
    painter.drawPolyline(ptsout)

def polygonClip(inpoly, rect, outpoly):
    """Clip a polygon to the rectangle given, writing to outpoly
    
    The python version does nothing really as it would be too hard.
    """

    for p in inpoly:
        x = max( min(p.x(), 32767.), -32767.)
        y = max( min(p.y(), 32767.), -32767.)
        outpoly.append( qt4.QPointF(x, y) )

def plotClippedPolygon(painter, inrect, inpoly, autoexpand=True):
    """Plot a polygon, clipping if necessary."""

    outpoly = qt4.QPolygonF()
    polygonClip(inpoly, inrect, outpoly)
    painter.drawPolygon(outpoly)

def plotBoxesToPainter(painter, x1, y1, x2, y2, clip=None, autoexpand=True):
    """Plot a set of rectangles."""
    if clip is None:
        clip = qt4.QRectF(qt4.QPointF(-32767,-32767), qt4.QPointF(32767,32767))

    # expand clip by line width
    if autoexpand:
        clip = qt4.QRectF(clip)
        lw = painter.pen().widthF()
        clip.adjust(-lw, -lw, lw, lw)

    # construct rectangle list
    rects = []
    for ix1, iy1, ix2, iy2 in izip(x1, y1, x2, y2):
        rect = qt4.QRectF( qt4.QPointF(ix1, iy1), qt4.QPointF(ix2, iy2) )
        if clip.intersects(rect):
            rects.append(clip.intersected(rect))

    # paint it
    if rects:
        painter.drawRects(rects)

def slowNumpyToQImage(img, cmap, transparencyimg):
    """Slow version of routine to convert numpy array to QImage
    This is hard work in Python, but it was like this originally.

    img: numpy array to convert to QImage
    cmap: 2D array of colors (BGRA rows)
    forcetrans: force image to have alpha component."""

    if struct.pack("h", 1) == "\000\001":
        # have to swap colors for big endian architectures
        cmap2 = cmap.copy()
        cmap2[:,0] = cmap[:,3]
        cmap2[:,1] = cmap[:,2]
        cmap2[:,2] = cmap[:,1]
        cmap2[:,3] = cmap[:,0]
        cmap = cmap2

    fracs = N.clip(N.ravel(img), 0., 1.)

    # Work out which is the minimum colour map. Assumes we have <255 bands.
    numbands = cmap.shape[0]-1
    bands = (fracs*numbands).astype(N.uint8)
    bands = N.clip(bands, 0, numbands-1)

    # work out fractional difference of data from band to next band
    deltafracs = (fracs - bands * (1./numbands)) * numbands

    # need to make a 2-dimensional array to multiply against triplets
    deltafracs.shape = (deltafracs.shape[0], 1)

    # calculate BGRalpha quadruplets
    # this is a linear interpolation between the band and the next band
    quads = (deltafracs*cmap[bands+1] +
             (1.-deltafracs)*cmap[bands]).astype(N.uint8)

    # apply transparency if a transparency image is set
    if transparencyimg is not None and transparencyimg.shape == img.shape:
        quads[:,3] = ( N.clip(N.ravel(transparencyimg), 0., 1.) *
                       quads[:,3] ).astype(N.uint8)

    # convert 32bit quads to a Qt QImage
    s = quads.tostring()

    fmt = qt4.QImage.Format_RGB32
    if N.any(cmap[:,3] != 255) or transparencyimg is not None:
        # any transparency
        fmt = qt4.QImage.Format_ARGB32

    img = qt4.QImage(s, img.shape[1], img.shape[0], fmt)
    img = img.mirrored()

    # hack to ensure string isn't freed before QImage
    img.veusz_string = s
    return img

