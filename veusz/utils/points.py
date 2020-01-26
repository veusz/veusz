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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
###############################################################################

from __future__ import division
from .. import qtall as qt
import numpy as N

from ..helpers.qtloops import plotPathsToPainter

from . import colormap

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

linesymbols = {
    'asterisk': ( ((-0.707, -0.707), (0.707,  0.707)),
                  ((-0.707,  0.707), (0.707, -0.707)),
                  ((-1, 0), (1, 0)), ((0, -1), (0, 1)) ),
    'lineplus': ( ((-1, 0), (1, 0)), ((0, -1), (0, 1)) ),
    'linecross': ( ((-0.707, -0.707), (0.707,  0.707)),
                   ((-0.707,  0.707), (0.707, -0.707)) ),
    'plushair': ( ((-1, 0), (-0.5, 0)), ((1, 0), (0.5, 0)),
                  ((0, -1), (0, -0.5)), ((0, 1), (0, 0.5)) ),
    'crosshair': ( ((-0.707, -0.707), (-0.354, -0.354)),
                   (( 0.707,  0.707), ( 0.354,  0.354)),
                   (( 0.707, -0.707), ( 0.354, -0.354)),
                   ((-0.707,  0.707), (-0.354,  0.354)) ),
    'asteriskhair': ( ((-1, 0), (-0.5, 0)), ((1, 0), (0.5, 0)),
                      ((0, -1), (0, -0.5)), ((0, 1), (0, 0.5)),
                      ((-0.707, -0.707), (-0.354, -0.354)),
                      (( 0.707,  0.707), ( 0.354,  0.354)),
                      (( 0.707, -0.707), ( 0.354, -0.354)),
                      ((-0.707,  0.707), (-0.354,  0.354)) ),
    'linehorz': ( ((-1, 0), (1, 0)), ),
    'linevert': ( ((0, -1), (0, 1)), ),
    'linehorzgap': ( ((-1, 0), (-0.5, 0)), ((1, 0), (0.5, 0)) ),
    'linevertgap': ( ((0, -1), (0, -0.5)), ((0, 1), (0, 0.5)) ),

    # arrows
    'arrowleft': ( ((1, -0.8), (0, 0), (1, 0.8)), ((2, 0), (0, 0)) ),
    'arrowleftaway': ( ((-1, -0.8), (-2, 0), (-1, 0.8)), ((-2, 0), (0, 0)) ),
    'arrowright': ( ((-1, -0.8), (0, 0), (-1, 0.8)), ((-2, 0), (0, 0)) ),
    'arrowrightaway': ( ((1, -0.8), (2, 0), (1, 0.8)), ((2, 0), (0, 0)) ),
    'arrowup': ( ((-0.8, 1), (0, 0), (0.8, 1)), ((0, 2), (0, 0)) ),
    'arrowupaway': ( ((-0.8, -1), (0, -2), (0.8, -1)), ((0, 0), (0, -2)) ),
    'arrowdown': ( ((-0.8, -1), (0, 0), (0.8, -1)), ((0, -2), (0, 0)) ),
    'arrowdownaway': ( ((-0.8, 1), (0, 2), (0.8, 1)), ((0, 0), (0, 2)) ),

    # limits
    'limitlower': ( ((-0.8, -1), (0, 0), (0.8, -1)), ((0, -2), (0, 0)),
                    ((-1, 0), (1, 0)) ),
    'limitupper': ( ((-0.8, 1), (0, 0), (0.8, 1)), ((0, 2), (0, 0)),
                    ((-1, 0), (1, 0)) ),
    'limitleft': ( ((1, -0.8), (0, 0), (1, 0.8)), ((2, 0), (0, 0)),
                   ((0, -1), (0, 1)) ),
    'limitright': ( ((-1, -0.8), (0, 0), (-1, 0.8)), ((-2, 0), (0, 0)),
                    ((0, -1), (0, 1)) ),
    'limitupperaway': ( ((-0.8, -1), (0, -2), (0.8, -1)), ((0, 0), (0, -2)),
                        ((-1, 0), (1, 0)) ),
    'limitloweraway': ( ((-0.8, 1), (0, 2), (0.8, 1)), ((0, 0), (0, 2)),
                        ((-1, 0), (1, 0)) ),
    'limitleftaway': ( ((-1, -0.8), (-2, 0), (-1, 0.8)), ((-2, 0), (0, 0)),
                       ((0, -1), (0, 1)) ),
    'limitrightaway': ( ((1, -0.8), (2, 0), (1, 0.8)), ((2, 0), (0, 0)),
                        ((0, -1), (0, 1)) ),

    'arrowlowerleftaway':( ((-0.8, 1), (0, 2), (0.8, 1)),
                           ((0, 2), (0, 0), (-2, 0)),
                           ((-1, -0.8), (-2, 0), (-1, 0.8)) ),
    'arrowlowerrightaway': ( ((1, -0.8), (2, 0), (1, 0.8)),
                             ((2, 0), (0, 0), (0, 2)),
                             ((-0.8, 1), (0, 2), (0.8, 1)) ),
    'arrowupperleftaway':( ((-0.8, -1), (0, -2), (0.8, -1)),
                           ((0, -2), (0, 0), (-2, 0)),
                           ((-1, -0.8), (-2, 0), (-1, 0.8)) ),
    'arrowupperrightaway': ( ((-0.8, -1), (0, -2), (0.8, -1)),
                             ((2, 0), (0, 0), (0, -2)),
                             ((1, -0.8), (2, 0), (1, 0.8)) ),

    'lineup': ( ((0, 0), (0, -1)), ),
    'linedown': ( ((0, 0), (0, 1)), ),
    'lineleft': ( ((0, 0), (-1, 0)), ),
    'lineright': ( ((0, 0), (1, 0)), ),

    # for arrows
    '_linearrow': ( ((-1.8, -1), (0, 0), (-1.8, 1)), ),
    '_linearrowreverse': ( ((1.8, -1), (0, 0), (1.8, 1)), ),
    }

def getLinePainterPath(name, size):
    """Get a painter path for line like objects."""
    path = qt.QPainterPath()
    for lines in linesymbols[name]:
        path.moveTo(lines[0][0]*size, lines[0][1]*size)
        for x, y in lines[1:]:
            path.lineTo(x*size, y*size)
    return path

#######################################################################
## draw symbols which are polygons

# X and Y pts for corners of polygons
polygons = {
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

    'star3': ( (0., -1.), (0.173, -0.1), (0.866, 0.5), (0, 0.2),
               (-0.866, 0.5), (-0.173, -0.1) ),
    'star4': ( (0.000, 1.000), (-0.354, 0.354), (-1.000, 0.000),
               (-0.354, -0.354), (0.000, -1.000), (0.354, -0.354),
               (1.000, -0.000), (0.354, 0.354), ),
    'star6': ( (0.000, 1.000), (-0.250, 0.433), (-0.866, 0.500),
               (-0.500, 0.000), (-0.866, -0.500), (-0.250, -0.433),
               (-0.000, -1.000), (0.250, -0.433), (0.866, -0.500),
               (0.500, 0.000), (0.866, 0.500), (0.250, 0.433), ),
    'star8': ( (0.000, 1.000), (-0.191, 0.462), (-0.707, 0.707),
               (-0.462, 0.191), (-1.000, 0.000), (-0.462, -0.191),
               (-0.707, -0.707), (-0.191, -0.462), (0.000, -1.000),
               (0.191, -0.462), (0.707, -0.707), (0.462, -0.191),
               (1.000, -0.000), (0.462, 0.191), (0.707, 0.707),
               (0.191, 0.462), ),
    'hexagon': ( (0, 1), (0.866, 0.5), (0.866, -0.5),
                 (0, -1), (-0.866, -0.5), (-0.866, 0.5), ),
    'starinvert': ( (0, 1.2), (-0.27, 0.3708), (-1.1412, 0.3708),
                    (-0.4356, -0.1416), (-0.7056, -0.9708), (0, -0.4584),
                    (0.7056, -0.9708), (0.4356, -0.1416), (1.1412, 0.3708),
                    (0.27, 0.3708) ),
    'squashbox': ( (-1, 1), (0, 0.5), (1, 1), (0.5, 0),
                   (1, -1), (0, -0.5), (-1, -1), (-0.5, 0) ),
    'plusnarrow': ( (0.2, 1), (0.2, 0.2), (1, 0.2), (1, -0.2),
                    (0.2, -0.2), (0.2, -1), (-0.2, -1), (-0.2, -0.2),
                    (-1, -0.2), (-1, 0.2), (-0.2, 0.2), (-0.2, 1) ),
    'crossnarrow': ( (-0.566, 0.849), (0, 0.283), (0.566, 0.849),
                     (0.849, 0.566), (0.283, 0), (0.849, -0.566),
                     (0.566, -0.849), (0, -0.283), (-0.566, -0.849),
                     (-0.849, -0.566), (-0.283, 0), (-0.849, 0.566) ),

    'limitupperaway2': ( (-1, 0), (0, 0), (0, -1), (-1, -1), (0, -2),
                         (1, -1), (0, -1), (0, 0), (1, 0) ),
    'limitloweraway2': ( (-1, 0), (0, 0), (0, 1), (-1, 1), (0, 2),
                         (1, 1), (0, 1), (0, 0), (1, 0) ),
    'limitleftaway2':  ( (0, -1), (0, 0) , (-1, 0), (-1, -1), (-2, 0),
                         (-1, 1), (-1, 0), (0, 0), (0, 1) ),
    'limitrightaway2': ( (0, -1), (0, 0), (1, 0), (1, -1), (2, 0),
                         (1, 1), (1, 0), (0, 0), (0, 1) ),

    # special arrow symbols
    '_arrow': ( (0, 0), (-1.8, 1), (-1.4, 0), (-1.8, -1) ),
    '_arrowtriangle': ( (0, 0), (-1.8, 1), (-1.8, -1) ),
    '_arrownarrow': ( (0, 0), (-1.8, 0.5), (-1.8, -0.5) ),
    '_arrowreverse': ( (0, 0), (1.8, 1), (1.4, 0), (1.8, -1) ),
    }

def addPolyPath(path, vals):
    """Add a polygon with the list of x,y pts as tuples in vals."""
    poly = qt.QPolygonF()
    for x, y in vals:
        poly.append( qt.QPointF(x, y) )
    path.addPolygon(poly)
    path.closeSubpath()

def getPolygonPainterPath(name, size):
    """Create a poly path for a polygon."""
    path = qt.QPainterPath()
    addPolyPath(path, N.array(polygons[name])*size)
    return path

#######################################################################
## draw symbols using a QPainterPath

def squarePath(path, size, linewidth):
    """Square path of size given."""
    path.addRect( qt.QRectF(-size, -size, size*2, size*2) )

def circlePath(path, size, linewidth):
    """Circle path of size given."""
    path.addEllipse( qt.QRectF(-size, -size, size*2, size*2) )

def circlePlusPath(path, size, linewidth):
    """Circle path with plus."""
    path.addEllipse( qt.QRectF(-size, -size, size*2, size*2) )
    path.moveTo(0, -size)
    path.lineTo(0, size)
    path.moveTo(-size, 0)
    path.lineTo(size, 0)

def circleCrossPath(path, size, linewidth):
    """Circle path with cross."""
    path.addEllipse( qt.QRectF(-size, -size, size*2, size*2) )
    m = N.sqrt(2.)*size*0.5
    path.moveTo(-m, -m)
    path.lineTo(m, m)
    path.moveTo(-m, m)
    path.lineTo(m, -m)

def circlePairPathHorz(path, size, linewidth):
    """2 circles next to each other (horizontal)."""
    path.addEllipse( qt.QRectF(-size, -size*0.5, size, size) )
    path.addEllipse( qt.QRectF(0,  -size*0.5, size, size) )

def circlePairPathVert(path, size, linewidth):
    """2 circles next to each other (vertical)."""
    path.addEllipse( qt.QRectF(-size*0.5, -size, size, size) )
    path.addEllipse( qt.QRectF(-size*0.5, 0, size, size) )

def ellipseHorzPath(path, size, linewidth):
    """Horizontal ellipse path."""
    path.addEllipse( qt.QRectF(-size, -size*0.5, size*2, size) )

def ellipseVertPath(path, size, linewidth):
    """Vertical ellipse path."""
    path.addEllipse( qt.QRectF(-size*0.5, -size, size, size*2) )

def circleHolePath(path, size, linewidth):
    """Circle with centre missing."""
    circlePath(path, size, linewidth)
    circlePath(path, size*0.5, linewidth)

def squarePlusPath(path, size, linewidth):
    """Square with plus sign."""
    path.addRect( qt.QRectF(-size, -size, size*2, size*2) )
    path.moveTo(0, -size)
    path.lineTo(0, size)
    path.moveTo(-size, 0)
    path.lineTo(size, 0)

def squareCrossPath(path, size, linewidth):
    """Square with cross sign."""
    path.addRect( qt.QRectF(-size, -size, size*2, size*2) )
    path.moveTo(-size, -size)
    path.lineTo(size, size)
    path.moveTo(-size, size)
    path.lineTo(size, -size)

def squareHolePath(path, size, linewidth):
    """Square with centre missing."""
    path.addRect( qt.QRectF(-size, -size, size*2, size*2) )
    path.addRect( qt.QRectF(-size*0.5, -size*0.5, size, size) )

def diamondHolePath(path, size, linewidth):
    """Diamond with centre missing."""
    pts = N.array(polygons['diamond'])*size
    addPolyPath(path, pts)
    addPolyPath(path, pts*0.5)

def pentagonHolePath(path, size, linewidth):
    """Pentagon with centre missing."""
    pts = N.array(polygons['pentagon'])*size
    addPolyPath(path, pts)
    addPolyPath(path, pts*0.5)

def squareRoundedPath(path, size, linewidth):
    """A square with rounded corners."""
    path.addRoundedRect(
        qt.QRectF(-size, -size, size*2, size*2),
        50, 50, qt.Qt.RelativeSize)

def dotPath(path, size, linewidth):
    """Draw a dot."""
    path.addEllipse(qt.QRectF(
        -linewidth*0.5, -linewidth*0.5, linewidth, linewidth))

def bullseyePath(path, size, linewidth):
    """A filled circle inside a filled circle."""
    path.setFillRule(qt.Qt.WindingFill)
    circlePath(path, size, linewidth)
    circlePath(path, size*0.5, linewidth)

def circleDotPath(path, size, linewidth):
    """A dot inside a circle."""
    path.setFillRule(qt.Qt.WindingFill)
    circlePath(path, size, linewidth)
    dotPath(path, size, linewidth)

pathsymbols = {
    'square': squarePath,
    'circle': circlePath,
    'circleplus': circlePlusPath,
    'circlecross': circleCrossPath,
    'circlepairhorz': circlePairPathHorz,
    'circlepairvert': circlePairPathVert,
    'ellipsehorz': ellipseHorzPath,
    'ellipsevert': ellipseVertPath,
    'circlehole': circleHolePath,
    'squareplus': squarePlusPath,
    'squarecross': squareCrossPath,
    'squarehole': squareHolePath,
    'diamondhole': diamondHolePath,
    'pentagonhole': pentagonHolePath,
    'squarerounded': squareRoundedPath,
    'dot': dotPath,
    'bullseye': bullseyePath,
    'circledot': circleDotPath,
    }

def getSymbolPainterPath(name, size, linewidth):
    """Get a painter path for a symbol shape."""
    path = qt.QPainterPath()
    pathsymbols[name](path, size, linewidth)
    return path

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

#######################################################################
## external interfaces

def getPointPainterPath(name, size, linewidth):
    """Return a painter path for the name and size.

    Returns (painterpath, enablefill)."""
    if name in linesymbols:
        return getLinePainterPath(name, size), False
    elif name in polygons:
        return getPolygonPainterPath(name, size), True
    elif name in pathsymbols:
        return getSymbolPainterPath(name, size, linewidth), True
    elif name == 'none':
        return qt.QPainterPath(), True
    else:
        raise ValueError("Invalid marker name %s" % name)

# list of codes supported
MarkerCodes = (
    'none',
    'circle', 'diamond', 'square',
    'cross', 'plus', 'star',
    'barhorz', 'barvert',
    'pentagon', 'hexagon', 'octogon', 'tievert', 'tiehorz',
    'triangle', 'triangledown', 'triangleleft', 'triangleright',
    'dot', 'circledot', 'bullseye',
    'circlehole', 'squarehole', 'diamondhole', 'pentagonhole',
    'squarerounded', 'squashbox',
    'ellipsehorz', 'ellipsevert',
    'lozengehorz', 'lozengevert',
    'plusnarrow', 'crossnarrow',
    'circleplus', 'circlecross', 'squareplus', 'squarecross',
    'star3', 'star4', 'star6', 'star8', 'starinvert',
    'circlepairhorz', 'circlepairvert',
    'asterisk', 'lineplus', 'linecross',
    'plushair', 'crosshair', 'asteriskhair',
    'linevert', 'linehorz', 'linevertgap', 'linehorzgap',
    'arrowleft', 'arrowright', 'arrowup', 'arrowdown',
    'arrowleftaway', 'arrowrightaway',
    'arrowupaway', 'arrowdownaway',
    'limitupper', 'limitlower', 'limitleft', 'limitright',
    'limitupperaway', 'limitloweraway',
    'limitleftaway', 'limitrightaway',
    'limitupperaway2', 'limitloweraway2',
    'limitleftaway2', 'limitrightaway2',
    'arrowupperleftaway', 'arrowupperrightaway',
    'arrowlowerrightaway', 'arrowlowerleftaway',
    'lineup', 'linedown', 'lineleft', 'lineright',
    )

def plotMarkers(painter, xpos, ypos, markername, markersize, scaling=None,
                clip=None, cmap=None, colorvals=None, scaleline=False):
    """Funtion to plot an array of markers on a painter.

    painter: QPainter
    xpos, ypos: iterable item of positions
    markername: name of marker from MarkerCodes
    markersize: size of marker to plot
    scaling: scale size of markers by array, or don't in None
    clip: rectangle if clipping wanted
    cmap: colormap to use if colorvals is set
    colorvals: color values 0-1 of each point if used
    scaleline: if scaling, scale border line width with scaling
    """

    # minor optimization
    if markername == 'none':
        return
    
    painter.save()

    # get sharper angles and more exact positions using these settings
    pen = painter.pen()
    pen.setJoinStyle( qt.Qt.MiterJoin )
    painter.setPen(pen)

    # get path to draw and whether to fill
    path, fill = getPointPainterPath(
        markername, markersize, painter.pen().widthF())
    if not fill:
        # turn off brush
        painter.setBrush( qt.QBrush() )

    # if using colored points
    colorimg = None
    if colorvals is not None:
        # convert colors to rgb values via a 2D image and pass to function
        trans = (1-painter.brush().color().alphaF())*100
        color2d = colorvals.reshape( 1, len(colorvals) )
        colorimg = colormap.applyColorMap(
            cmap, 'linear', color2d, 0., 1., trans)

    plotPathsToPainter(painter, path, xpos, ypos, scaling, clip, colorimg,
                       scaleline)

    painter.restore()

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
    'lineup': 'lineup',
    'linedown': 'linedown',
    'lineextend': 'lineright',
}

# codes of allowable arrows
ArrowCodes = ( 'none', 'arrow', 'arrownarrow',
               'arrowtriangle',
               'arrowreverse',
               'linearrow', 'linearrowreverse',
               'bar', 'linecross',
               'asterisk',
               'circle', 'square', 'diamond',
               'lineup', 'linedown',
               'lineextend',
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

    # plot marker at one end of line
    plotMarker(painter, length, 0., arrow_translate[arrowright], arrowsize)

    # plot reversed marker at other end
    painter.scale(-1, 1)
    plotMarker(painter, 0, 0, arrow_translate[arrowleft], arrowsize)

    pen = painter.pen()
    pen.setCapStyle(qt.Qt.FlatCap)
    painter.setPen(pen)
    painter.scale(-1, 1)
    painter.drawLine(qt.QPointF(0, 0), qt.QPointF(length, 0))

    painter.restore()
