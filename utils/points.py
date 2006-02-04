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

import qt
import numarray as N

def _plot_none(painter, xpos, ypos, size):
    """ (internal) function plot nothing!
    """
    pass

def _plot_dot(painter, xpos, ypos, size):
    """ (internal) function to plot a dot
    """

    # draw dot as circle with same radius as line thickness
    #  - much more sensible than actual dot routine
    w = painter.pen().width() / 2
    painter.drawEllipse( xpos-w, ypos-w, w*2+1, w*2+1)

def _plot_circle(painter, xpos, ypos, size):
    """ (internal) function to plot a circle marker
    size is the radius of the circle
    """

    # qt uses a bounding rectangle, so we have to do this the hard way
    painter.drawEllipse( xpos - size, ypos - size , size*2+1, size*2+1 )

def _plot_ellipse_vert(painter, xpos, ypos, size):
    """ (internal) function to plot a vertical ellipse marker
    size is the radius of the ellipse on the long end
    """

    # qt uses a bounding rectangle, so we have to do this the hard way
    painter.drawEllipse( xpos - size/2, ypos - size , size+1, size*2+1 )

def _plot_ellipse_horz(painter, xpos, ypos, size):
    """ (internal) function to plot a horizontal ellipse marker
    size is the radius of the ellipse on the lon end
    """

    # qt uses a bounding rectangle, so we have to do this the hard way
    painter.drawEllipse( xpos - size, ypos - size/2 , size*2+1, size+1 )

def _plot_asterisk(painter, xpos, ypos, size):
    """ (internal) function to plot a *
    """

    s2 = int(size*0.707107)
    painter.drawLine( xpos - s2, ypos - s2, xpos + s2, ypos + s2 )
    painter.drawLine( xpos - s2, ypos + s2, xpos + s2, ypos - s2 )
    painter.drawLine( xpos - size, ypos, xpos + size, ypos )
    painter.drawLine( xpos, ypos - size, xpos, ypos + size )

def _plot_line_plus(painter, xpos, ypos, size):
    """ (internal) function to plot a + in lines.
    """
    painter.drawLine( xpos - size, ypos, xpos + size, ypos )
    painter.drawLine( xpos, ypos - size, xpos, ypos + size )

def _plot_line_cross(painter, xpos, ypos, size):
    """ (internal) function to plot a x in lines.
    """
    s2 = int(size*0.707107)
    painter.drawLine( xpos - s2, ypos - s2, xpos + s2, ypos + s2 )
    painter.drawLine( xpos - s2, ypos + s2, xpos + s2, ypos - s2 )

def _plot_circle_dot(painter, xpos, ypos, size):
    """ (internal) function to plot a circle marker with dot at core
    size is the radius of the circle
    """

    # qt uses a bounding rectangle, so we have to do this the hard way
    painter.drawEllipse( xpos - size, ypos - size , size*2+1, size*2+1 )
    w = painter.pen().width() / 2
    painter.drawEllipse( xpos-w, ypos-w, w*2+1, w*2+1)
   
def _plot_bullseye(painter, xpos, ypos, size):
    """ (internal) function to plot a bullseye shape
    size is radius
    """

    painter.drawEllipse( xpos - size, ypos - size, size*2+1, size*2+1 )
    painter.drawEllipse( xpos - size/2, ypos - size/2, size+1, size+1 )

def _plot_line_horz(painter, xpos, ypos, size):
    """ (internal) function to plot a horizontal line
    """

    painter.drawLine( xpos-size, ypos, xpos+size, ypos)

def _plot_line_vert(painter, xpos, ypos, size):
    """ (internal) function to plot a horizontal line
    """

    painter.drawLine( xpos, ypos-size, xpos, ypos+size)

def _plot_arrow_left(painter, xpos, ypos, size):
    """ (internal) function to plot a left arrow
    """
    painter.drawLine(xpos, ypos, xpos+size*2, ypos)
    painter.drawLine(xpos, ypos, xpos+size, ypos+size)
    painter.drawLine(xpos, ypos, xpos+size, ypos-size)

def _plot_arrow_right(painter, xpos, ypos, size):
    """ (internal) function to plot a right arrow
    """
    painter.drawLine(xpos, ypos, xpos-size*2, ypos)
    painter.drawLine(xpos, ypos, xpos-size, ypos+size)
    painter.drawLine(xpos, ypos, xpos-size, ypos-size)

def _plot_arrow_up(painter, xpos, ypos, size):
    """ (internal) function to plot an up arrow
    """
    painter.drawLine(xpos, ypos, xpos, ypos+size*2)
    painter.drawLine(xpos, ypos, xpos-size, ypos+size)
    painter.drawLine(xpos, ypos, xpos+size, ypos+size)

def _plot_arrow_down(painter, xpos, ypos, size):
    """ (internal) function to plot a down arrow
    """
    painter.drawLine(xpos, ypos, xpos, ypos-size*2)
    painter.drawLine(xpos, ypos, xpos-size, ypos-size)
    painter.drawLine(xpos, ypos, xpos+size, ypos-size)

def _plot_limit_lower(painter, xpos, ypos, size):
    """Plot lower limit."""
    _plot_arrow_down(painter, xpos, ypos, size)
    _plot_line_horz(painter, xpos, ypos, size)

def _plot_limit_upper(painter, xpos, ypos, size):
    """Plot upper limit."""
    _plot_arrow_up(painter, xpos, ypos, size)
    _plot_line_horz(painter, xpos, ypos, size)
    
def _plot_limit_left(painter, xpos, ypos, size):
    """Plot left limit."""
    _plot_arrow_left(painter, xpos, ypos, size)
    _plot_line_vert(painter, xpos, ypos, size)

def _plot_limit_right(painter, xpos, ypos, size):
    """Plot right limit."""
    _plot_arrow_right(painter, xpos, ypos, size)
    _plot_line_vert(painter, xpos, ypos, size)
    
MarkerCodes = ( 'none', 'cross', 'plus', 'star', 'circle',
                'diamond', 'square', 'barhorz', 'barvert',
                'octogon', 'pentagon', 'tievert', 'tiehorz',
                'triangle', 'triangledown',
                'dot', 'circledot', 'bullseye',
                'ellipsehorz', 'ellipsevert',
                'asterisk',
                'lineplus', 'linecross',
                'linevert', 'linehorz',
                'arrowleft', 'arrowright', 'arrowup',
                'arrowdown',
                'limitupper', 'limitlower', 'limitleft', 'limitright')

# functions to call for special shapes
_MarkerLookup = { 'none': _plot_none,
                  'circle': _plot_circle,
                  'dot': _plot_dot,
                  'circledot': _plot_circle_dot,
                  'bullseye': _plot_bullseye,
                  'ellipsehorz': _plot_ellipse_horz,
                  'ellipsevert': _plot_ellipse_vert,
                  'asterisk': _plot_asterisk,
                  'lineplus': _plot_line_plus,
                  'linecross': _plot_line_cross,
                  'linehorz': _plot_line_horz,
                  'linevert': _plot_line_vert,
                  'arrowleft': _plot_arrow_left,
                  'arrowright': _plot_arrow_right,
                  'arrowup': _plot_arrow_up,
                  'arrowdown': _plot_arrow_down,
                  'limitupper': _plot_limit_upper,
                  'limitlower': _plot_limit_lower,
                  'limitleft': _plot_limit_left,
                  'limitright': _plot_limit_right
                  }

# X and Y pts for corners of polygons
_Polygons = {
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
    'tiehorz': ( (-1, -1), (-1, 1), (1, -1), (1, 1) )
    }

def _plotPolygons(painter, name, xpos, ypos, size):
    '''Plots shapes which are polygons'''

    # make a polygon of the correct size
    pgn = (N.array(_Polygons[name], N.Float32) * size).astype(N.Int32)
    pa = qt.QPointArray( pgn.flat.tolist() )

    xlast = 0
    ylast = 0
    for x, y in itertools.izip(xpos, ypos):
        pa.translate(x-xlast, y-ylast)
        painter.drawPolygon(pa)
        xlast = x
        ylast = y

def plotMarker(painter, xpos, ypos, markercode, markersize):
    """Function to plot a marker on a painter, posn xpos, ypos, type and size
    """
    if markercode in _Polygons:
        _plotPolygons(painter, markercode, (xpos,), (ypos,), markersize)
    else:
        _MarkerLookup[markercode](painter, xpos, ypos, markersize)

def plotMarkers(painter, xpos, ypos, markername, markersize):
    """Funtion to plot an array of markers on painter
    """

    if markername in _Polygons:
        _plotPolygons(painter, markername, xpos, ypos, markersize)
    elif markername == 'none':
        # little optimization
        return
    else:
        fn = _MarkerLookup[markername]
        for x, y in zip(xpos, ypos):
            fn(painter, x, y, markersize)
