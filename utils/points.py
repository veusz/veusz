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

def _plotNone(painter, xpos, ypos, size):
    """ (internal) function plot nothing!
    """
    pass

def _plotDot(painter, xpos, ypos, size):
    """ (internal) function to plot a dot
    """

    # draw dot as circle with same radius as line thickness
    #  - much more sensible than actual dot routine
    w = painter.pen().width() / 2
    painter.drawEllipse( xpos-w, ypos-w, w*2+1, w*2+1)

def _plotCircle(painter, xpos, ypos, size):
    """ (internal) function to plot a circle marker
    size is the radius of the circle
    """

    # qt uses a bounding rectangle, so we have to do this the hard way
    painter.drawEllipse( qt4.QRectF(xpos - size, ypos - size,
                                    size*2, size*2) )

def _plotEllipseVert(painter, xpos, ypos, size):
    """ (internal) function to plot a vertical ellipse marker
    size is the radius of the ellipse on the long end
    """

    # qt uses a bounding rectangle, so we have to do this the hard way
    painter.drawEllipse( qt4.QRectF(xpos - size*0.5, ypos - size,
                                    size+1, size*2+1 ) )

def _plotEllipseHorz(painter, xpos, ypos, size):
    """ (internal) function to plot a horizontal ellipse marker
    size is the radius of the ellipse on the lon end
    """

    # qt uses a bounding rectangle, so we have to do this the hard way
    painter.drawEllipse( qt4.QRectF(xpos - size, ypos - size*0.5,
                                    size*2+1, size+1) )

def _plotCircleDot(painter, xpos, ypos, size):
    """ (internal) function to plot a circle marker with dot at core
    size is the radius of the circle
    """

    # qt uses a bounding rectangle, so we have to do this the hard way
    painter.drawEllipse( qt4.QRectF(xpos - size, ypos - size,
                                    size*2+1, size*2+1) )
    w = painter.pen().width()*0.5
    painter.drawEllipse( qt4.QRectF(xpos-w, ypos-w, w*2+1, w*2+1) )

def _plotBullseye(painter, xpos, ypos, size):
    """ (internal) function to plot a bullseye shape
    size is radius
    """

    painter.drawEllipse( qt4.QRectF(xpos - size, ypos - size,
                                    size*2+1, size*2+1) )
    painter.drawEllipse( qt4.QRectF(xpos - size*0.5, ypos - size*0.5,
                                    size+1, size+1) )

# functions to call for special shapes
_markerlookup = { 'none': _plotNone,
                  'circle': _plotCircle,
                  'dot': _plotDot,
                  'circledot': _plotCircleDot,
                  'bullseye': _plotBullseye,
                  'ellipsehorz': _plotEllipseHorz,
                  'ellipsevert': _plotEllipseVert
                  }

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

    lastx = 0.
    lasty = 0.
    for x, y in itertools.izip(xpos, ypos):
        for pa in ptarrays:
            pa2 = qt4.QPolygonF(pa)
            pa2.translate(x, y)
            painter.drawPolyline(pa2)
    
# X and Y pts for corners of polygons
_polygons = {
    'square': ( (-1, -1), (1, -1), (1, 1), (-1, 1) ),
    # make the diamond the same area as the square
    'diamond': ( (0., 1.414), (1.414, 0.), (0., -1.414), (-1.414, 0.) ),
    'barhorz': ( (-1, -0.5), (1, -0.5), (1, 0.5), (-1, 0.5) ),
    'barvert': ( (-0.5, -1), (0.5, -1), (0.5, 1), (-0.5, 1) ),
    'plus': ( (0.3, 1), (0.3, 0.3), (1, 0.3), (1, -0.3),
              (0.3, -0.3), (0.3, -1), (-0.3, -1), (-0.3, -0.3),
              (-1, -0.3), (-1, 0.3), (-0.3, 0.3), (-0.3, 1) ),
    'octogon': ( (0.414, 1), (1, 0.414), (1, -0.414), (0.414, -1),
                 (-0.414, -1), (-1, -0.414), (-1, 0.414), (-0.414, 1) ),
    'triangle': ( (0, -1.2), (1.0392, 0.6), (-1.0392, 0.6) ),
    'triangledown': ( (0, 1.2), (1.0392, -0.6), (-1.0392, -0.6) ),
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
    'lozengevert': ( (0, 1.414), (0.707, 0), (0, -1.414), (-0.707, 0) )
    }

def _plotPolygons(painter, name, xpos, ypos, size):
    '''Plots shapes which are polygons'''

    pgn = qt4.QPolygonF()
    for x, y in _polygons[name]:
        pgn.append(qt4.QPointF(x*size, y*size))

    for x, y in itertools.izip(xpos, ypos):
        pgn2 = qt4.QPolygonF(pgn)
        pgn2.translate(x, y)
        painter.drawPolygon(pgn2)

MarkerCodes = ( 'none', 'cross', 'plus', 'star', 'circle',
                'diamond', 'square', 'barhorz', 'barvert',
                'octogon', 'pentagon', 'tievert', 'tiehorz',
                'triangle', 'triangledown',
                'dot', 'circledot', 'bullseye',
                'ellipsehorz', 'ellipsevert',
                'lozengehorz', 'lozengevert',
                'asterisk',
                'lineplus', 'linecross',
                'linevert', 'linehorz',
                'arrowleft', 'arrowright', 'arrowup',
                'arrowdown',
                'arrowleftaway', 'arrowrightaway',
                'arrowupaway', 'arrowdownaway',
                'limitupper', 'limitlower', 'limitleft', 'limitright',
                'limitupperaway', 'limitloweraway' )

def plotMarker(painter, xpos, ypos, markercode, markersize):
    """Function to plot a marker on a painter, posn xpos, ypos, type and size
    """
    if markercode in _polygons:
        _plotPolygons(painter, markercode, (xpos,), (ypos,), markersize)
    elif markercode in _linesymbols:
        _plotLineSymbols(painter, markercode, (xpos,), (ypos,), markersize)
    else:
        _markerlookup[markercode](painter, xpos, ypos, markersize)

def plotMarkers(painter, xpos, ypos, markername, markersize):
    """Funtion to plot an array of markers on painter
    """

    if markername in _polygons:
        _plotPolygons(painter, markername, xpos, ypos, markersize)
    elif markername in _linesymbols:
        _plotLineSymbols(painter, markername, xpos, ypos, markersize)
    elif markername == 'none':
        # little optimization
        return
    else:
        fn = _markerlookup[markername]
        for x, y in itertools.izip(xpos, ypos):
            fn(painter, x, y, markersize)
