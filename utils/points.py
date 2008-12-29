# points.py - functions to plot points

#    Copyright (C) 2003 Jeremy S. Sanders
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
###############################################################################

# $Id$

import itertools

import veusz.qtall as qt4
import numpy as N

"""This is the symbol plotting part of Veusz

There are actually several different ways symbols are plotted.
We choose the most appropriate one for the shape:

QPainterPath symbols plotted with _plotPathSymbols
line symbols are plotted with _plotLineSymbols
ploygon symbols are plotted wiht _plotPolygonSymbols

Many of these are implemented as paths internally, and drawn using
QPainterPaths
"""

#######################################################################
## draw symbols which are sets of line segments

_linesymbols = {
    'asterisk': ( ((-0.707, -0.707), (0.707,  0.707)),
                  ((-0.707,  0.707), (0.707, -0.707)),
                  ((-1, 0), (1, 0)), ((0, -1), (0, 1)) ),
    'lineplus': ( ((-1, 0), (1, 0)), ((0, -1), (0, 1)) ),
    'linecross': ( ((-0.707, -0.707), (0.707,  0.707)),
                   ((-0.707,  0.707), (0.707, -0.707)) ),
    'linehorz': ( ((-1, 0), (1, 0)), ),
    'linevert': ( ((0, -1), (0, 1)), ),
    'arrowleft': ( ((1, -1), (0, 0), (1, 1)), ((2, 0), (0, 0)) ),
    'arrowleftaway': ( ((-1, -1), (-2, 0), (-1, 1)), ((-2, 0), (0, 0)) ),
    'arrowright': ( ((-1, -1), (0, 0), (-1, 1)), ((-2, 0), (0, 0)) ),
    'arrowrightaway': ( ((1, -1), (2, 0), (1, 1)), ((2, 0), (0, 0)) ),
    'arrowup': ( ((-1, 1), (0, 0), (1, 1)), ((0, 2), (0, 0)) ),
    'arrowupaway': ( ((-1, -1), (0, -2), (1, -1)), ((0, 0), (0, -2)) ),
    'arrowdown': ( ((-1, -1), (0, 0), (1, -1)), ((0, -2), (0, 0)) ),
    'arrowdownaway': ( ((-1, 1), (0, 2), (1, 1)), ((0, 0), (0, 2)) ),
    'limitlower': ( ((-1, -1), (0, 0), (1, -1)), ((0, -2), (0, 0)),
                    ((-1, 0), (1, 0)) ),
    'limitupper': ( ((-1, 1), (0, 0), (1, 1)), ((0, 2), (0, 0)),
                    ((-1, 0), (1, 0)) ),
    'limitleft': ( ((1, -1), (0, 0), (1, 1)), ((2, 0), (0, 0)),
                   ((0, -1), (0, 1)) ),
    'limitright': ( ((-1, -1), (0, 0), (-1, 1)), ((-2, 0), (0, 0)),
                    ((0, -1), (0, 1)) ),
    'limitupperaway': ( ((-1, -1), (0, -2), (1, -1)), ((0, 0), (0, -2)),
                        ((-1, 0), (1, 0)) ),
    'limitloweraway': ( ((-1, 1), (0, 2), (1, 1)), ((0, 0), (0, 2)),
                        ((-1, 0), (1, 0)) ),
    'limitleftaway': ( ((-1, -1), (-2, 0), (-1, 1)), ((-2, 0), (0, 0)),
                       ((0, -1), (0, 1)) ),
    'limitrightaway': ( ((1, -1), (2, 0), (1, 1)), ((2, 0), (0, 0)),
                        ((0, -1), (0, 1)) ),

    # for arrows
    '_linearrow': ( ((-1.8, -1), (0, 0), (-1.8, 1)), ),
    '_linearrowreverse': ( ((1.8, -1), (0, 0), (1.8, 1)), ),
    }

def _plotLineSymbols(painter, name, xpos, ypos, size):
    '''Plots shapes which are made out of line segments'''

    # make a set of point arrays for each plotted symbol
    ptarrays = []
    for lines in _linesymbols[name]:
        pa = qt4.QPolygonF()
        for x, y in lines:
            pa.append(qt4.QPointF(x*size, y*size))
        ptarrays.append(pa)

    painter.save()
    for x, y in itertools.izip(xpos, ypos):
        painter.translate(x, y)
        for pa in ptarrays:
            painter.drawPolyline(pa)
        painter.resetTransform()
    painter.restore()

#######################################################################
## draw symbols which are polygons

# X and Y pts for corners of polygons
_polygons = {
    # make the diamond the same area as the square
    'diamond': ( (0., 1.414), (1.414, 0.), (0., -1.414), (-1.414, 0.) ),
    'barhorz': ( (-1, -0.5), (1, -0.5), (1, 0.5), (-1, 0.5) ),
    'barvert': ( (-0.5, -1), (0.5, -1), (0.5, 1), (-0.5, 1) ),
    'plus': ( (0.4, 1), (0.4, 0.4), (1, 0.4), (1, -0.4),
              (0.4, -0.4), (0.4, -1), (-0.4, -1), (-0.4, -0.4),
              (-1, -0.4), (-1, 0.4), (-0.4, 0.4), (-0.4, 1) ),
    'octogon': ( (0.414, 1), (1, 0.414), (1, -0.414), (0.414, -1),
                 (-0.414, -1), (-1, -0.414), (-1, 0.414), (-0.414, 1) ),
    'triangle': ( (0, -1.2), (1.0392, 0.6), (-1.0392, 0.6) ),
    'triangledown': ( (0, 1.2), (1.0392, -0.6), (-1.0392, -0.6) ),
    'triangleleft': ( (-1.2, 0), (0.6, 1.0392), (0.6, -1.0392) ),
    'triangleright': ( (1.2, 0), (-0.6, 1.0392), (-0.6, -1.0392) ),
    'cross': ( (-0.594, 1.1028), (0, 0.5088), (0.594, 1.1028),
               (1.1028, 0.594), (0.5088, -0), (1.1028, -0.594),
               (0.594, -1.1028), (-0, -0.5088), (-0.594, -1.1028),
               (-1.1028, -0.594), (-0.5088, 0), (-1.1028, 0.594) ),
    'star': ( (0, -1.2), (-0.27, -0.3708), (-1.1412, -0.3708),
              (-0.4356, 0.1416), (-0.7056, 0.9708), (-0, 0.4584),
              (0.7056, 0.9708), (0.4356, 0.1416), (1.1412, -0.3708),
              (0.27, -0.3708) ),
    'pentagon': ((0, -1.2), (1.1412, -0.3708), (0.6936, 0.9708),
                 (-0.6936, 0.9708), (-1.1412, -0.3708)),
    'tievert': ( (-1, -1), (1, -1), (-1, 1), (1, 1) ),
    'tiehorz': ( (-1, -1), (-1, 1), (1, -1), (1, 1) ),
    'lozengehorz': ( (0, 0.707), (1.414, 0), (0, -0.707), (-1.414, 0) ),
    'lozengevert': ( (0, 1.414), (0.707, 0), (0, -1.414), (-0.707, 0) ),

    # special arrow symbols
    '_arrow': ( (0, 0), (-1.8, 1), (-1.4, 0), (-1.8, -1) ),
    '_arrowtriangle': ( (0, 0), (-1.8, 1), (-1.8, -1) ),
    '_arrownarrow': ( (0, 0), (-1.8, 0.5), (-1.8, -0.5) ),
    '_arrowreverse': ( (0, 0), (1.8, 1), (1.4, 0), (1.8, -1) ),
    }

def _addPolyPath( path, vals ):
    """Add a polygon with the list of x,y pts as tuples in vals."""
    poly = qt4.QPolygonF()
    for x, y in vals:
        poly.append( qt4.QPointF(x, y) )
    path.addPolygon(poly)
    path.closeSubpath()

def _plotPolygonSymbols(painter, name, xpos, ypos, size):
    '''Plots shapes which are polygons'''

    path = qt4.QPainterPath()
    _addPolyPath(path, N.array(_polygons[name])*size)

    # optimise by looking up functions first
    t = painter.translate
    d = painter.drawPath
    r = painter.resetTransform

    painter.save()
    for x, y in itertools.izip(xpos, ypos):
        t(x, y)
        d(path)
        r()
    painter.restore()

#######################################################################
## draw symbols using a QPainterPath

def _squarePath(painter, path, size):
    """Square path of size given."""
    path.addRect( qt4.QRectF(-size, -size, size*2, size*2) )

def _circlePath(painter, path, size):
    """Circle path of size given."""
    path.addEllipse( qt4.QRectF(-size, -size, size*2, size*2) )

def _ellipseHorzPath(painter, path, size):
    """Horizontal ellipse path."""
    path.addEllipse( qt4.QRectF(-size, -size*0.5, size*2, size) )

def _ellipseVertPath(painter, path, size):
    """Vertical ellipse path."""
    path.addEllipse( qt4.QRectF(-size*0.5, -size, size, size*2) )

def _circleHolePath(painter, path, size):
    """Circle with centre missing."""
    _circlePath(painter, path, size)
    _circlePath(painter, path, size*0.5)

def _squareHolePath(painter, path, size):
    """Square with centre missing."""
    path.addRect( qt4.QRectF(-size, -size, size*2, size*2) )
    path.addRect( qt4.QRectF(-size*0.5, -size*0.5, size, size) )

def _diamondHolePath(painter, path, size):
    """Diamond with centre missing."""
    pts = N.array(_polygons['diamond'])*size
    _addPolyPath(path, pts)
    _addPolyPath(path, pts*0.5)

def _pentagonHolePath(painter, path, size):
    """Pentagon with centre missing."""
    pts = N.array(_polygons['pentagon'])*size
    _addPolyPath(path, pts)
    _addPolyPath(path, pts*0.5)

def _squareRoundedPath(painter, path, size):
    """A square with rounded corners."""
    path.addRoundRect( qt4.QRectF(-size, -size, size*2, size*2), 50, 50 )

def _dotPath(painter, path, size):
    """Draw a dot."""
    w = painter.pen().widthF()
    path.addEllipse( qt4.QRectF(-w*0.5, -w*0.5, w, w) )

def _bullseyePath(painter, path, size):
    """A filled circle inside a filled circle."""
    path.setFillRule(qt4.Qt.WindingFill)
    _circlePath(painter, path, size)
    _circlePath(painter, path, size*0.5)

def _circleDotPath(painter, path, size):
    """A dot inside a circle."""
    path.setFillRule(qt4.Qt.WindingFill)
    _circlePath(painter, path, size)
    _dotPath(painter, path, size)

_pathsymbols = {
    'square': _squarePath,
    'circle': _circlePath,
    'ellipsehorz': _ellipseHorzPath,
    'ellipsevert': _ellipseVertPath,
    'circlehole': _circleHolePath,
    'squarehole': _squareHolePath,
    'diamondhole': _diamondHolePath,
    'pentagonhole': _pentagonHolePath,
    'squarerounded': _squareRoundedPath,
    'dot': _dotPath,
    'bullseye': _bullseyePath,
    'circledot': _circleDotPath,
    }

def _plotPathSymbols(painter, name, xpos, ypos, size):
    """Draw symbols using a QPainterPath."""

    path = qt4.QPainterPath()
    _pathsymbols[name](painter, path, size)

    # optimise by looking up functions first
    t = painter.translate
    d = painter.drawPath
    r = painter.resetTransform

    painter.save()
    for x, y in itertools.izip(xpos, ypos):
        t(x, y)
        d(path)
        r()
    painter.restore()

#######################################################################
## external interfaces

# list of codes
MarkerCodes = ( 'none',
                'circle', 'diamond', 'square',
                'cross', 'plus', 'star',
                'barhorz', 'barvert',
                'octogon', 'pentagon', 'tievert', 'tiehorz',
                'triangle', 'triangledown', 'triangleleft', 'triangleright',
                'dot', 'circledot', 'bullseye',
                'circlehole', 'squarehole', 'diamondhole', 'pentagonhole',
                'squarerounded',
                'ellipsehorz', 'ellipsevert',
                'lozengehorz', 'lozengevert',
                'asterisk',
                'lineplus', 'linecross',
                'linevert', 'linehorz',
                'arrowleft', 'arrowright', 'arrowup', 'arrowdown',
                'arrowleftaway', 'arrowrightaway',
                'arrowupaway', 'arrowdownaway',
                'limitupper', 'limitlower', 'limitleft', 'limitright',
                'limitupperaway', 'limitloweraway',
                'limitleftaway', 'limitrightaway',
                )

def plotMarkers(painter, xpos, ypos, markername, markersize):
    """Funtion to plot an array of markers on painter
    """

    if markername in _polygons:
        _plotPolygonSymbols(painter, markername, xpos, ypos, markersize)
    elif markername in _linesymbols:
        _plotLineSymbols(painter, markername, xpos, ypos, markersize)
    elif markername in _pathsymbols:
        _plotPathSymbols(painter, markername, xpos, ypos, markersize)
    elif markername == 'none':
        pass
    else:
        raise ValueError, "Invalid marker name %s" % markername

def plotMarker(painter, xpos, ypos, markername, markersize):
    """Function to plot a marker on a painter, posn xpos, ypos, type and size
    """
    plotMarkers(painter, (xpos,), (ypos,), markername, markersize)



# translate arrow shapes to point types (we reuse them)
arrow_translate = {
    'none': 'none',
    'arrow': '_arrow',
    'arrownarrow': '_arrownarrow',
    'arrowtriangle': '_arrowtriangle',
    'arrowreverse': '_arrowreverse',
    'linearrow': '_linearrow',
    'linearrowreverse': '_linearrowreverse',
    'bar': 'linevert',
    'linecross': 'linecross',
    'asterisk': 'asterisk',
    'circle': 'circle',
    'square': 'square',
    'diamond': 'diamond',
}

# codes of allowable arrows
ArrowCodes = ( 'none', 'arrow', 'arrownarrow',
               'arrowtriangle',
               'arrowreverse',
               'linearrow', 'linearrowreverse',
               'bar', 'linecross',
               'asterisk',
               'circle', 'square', 'diamond',
               )

def plotLineArrow(painter, xpos, ypos, length, angle,
                  arrowsize=0,
                  arrowleft='none', arrowright='none'):
    """Plot a line or arrow.
    
    xpos, ypos is the starting point of the line
    angle is the angle to the horizontal (degrees)
    arrowleft and arrowright are arrow codes."""

    painter.save()
    painter.translate(xpos, ypos)
    painter.rotate(angle)

    # draw line between points
    painter.drawLine( qt4.QPointF(0., 0.),
                      qt4.QPointF(length, 0.) )

    # plot marker at one end of line
    plotMarker(painter, length, 0., 
               arrow_translate[arrowright], arrowsize)

    # plot reversed marker at other end
    painter.scale(-1, 1)
    plotMarker(painter, 0., 0., 
               arrow_translate[arrowleft], arrowsize)

    painter.restore()
