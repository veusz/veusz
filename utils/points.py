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

import qt
import numarray as N

def _plot_none(painter, xpos, ypos, size):
    """ (internal) function plot nothing!
    """
    pass

def _plot_point(painter, xpos, ypos, size):
    """ (internal) function to plot a point
    """
    painter.drawPoint( xpos, ypos )

def _plot_circle(painter, xpos, ypos, size):
    """ (internal) function to plot a circle marker
    size is the radius of the circle
    """

    # qt uses a bounding rectangle, so we have to do this the hard way
    painter.drawEllipse( xpos - size, ypos - size , size*2+1, size*2+1 )

def _plot_star(painter, xpos, ypos, size):
    """ (internal) function to plot a *
    """

    painter.drawLine( xpos - size, ypos - size, xpos + size, ypos + size )
    painter.drawLine( xpos - size, ypos + size, xpos + size, ypos - size )
    painter.drawLine( xpos - size, ypos, xpos + size, ypos )
    painter.drawLine( xpos, ypos - size, xpos, ypos + size )

def _plot_circle_dot(painter, xpos, ypos, size):
    """ (internal) function to plot a circle marker with dot at core
    size is the radius of the circle
    """

    # qt uses a bounding rectangle, so we have to do this the hard way
    painter.drawEllipse( xpos - size, ypos - size , size*2+1, size*2+1 )
    painter.drawPoint( xpos, ypos )
   
def _plot_bullseye(painter, xpos, ypos, size):
    """ (internal) function to plot a bullseye shape
    size is radius
    """

    painter.drawEllipse( xpos - size/2, ypos - size/2 , size, size )
    painter.drawEllipse( xpos - size, ypos - size , size*2+1, size*2+1 )

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

MarkerCodes = ['none', 'cross', 'plus', 'star', 'circle',
               'circledot', 'diamond', 'square', 'barhorz', 'barvert',
               'octogon', 'pentagon', 'tievert', 'tiehorz',
               'bullseye', 'triangle',
               'point', 'horzbar', 'vertbar',
               'arrowleft', 'arrowright', 'arrowup',
               'arrowdown']

_MarkerFunctions = [_plot_none, None, None, None, _plot_circle,
                    _plot_circle_dot, None, None, None, None,
                    None, None, None, None,
                    _plot_bullseye, None,
                    _plot_point, _plot_line_horz, _plot_line_vert,
                    _plot_arrow_left, _plot_arrow_right, _plot_arrow_up,
                    _plot_arrow_down]

_MarkerLookup = {}
for code, fn in zip(MarkerCodes, _MarkerFunctions):
    _MarkerLookup[code] = fn

# list of markers to be automatically iterated through on new data
AutoMarkers = [ 'X', '+', '*', 'O', 'Odot',
                'box', 'boxdot', 'bullseye',
                'triangle', 'triangledot',
                '.' ]

# where we are in the above list
_automarker = 0

# default colours
AutoColors = [ 'black', 'red', 'blue', 'green', 'magenta',
               'cyan', 'DarkRed', 'DarkBlue', 'DarkGreen',
               'DarkMagenta', 'DarkCyan', 'purple' ]

_autocolor = 0

def getAutoColor():
    """ Get the next available colour."""
    global _autocolor
    return AutoColors[_autocolor]

def getAutoMarker():
    """ Get the next automatic marker."""
    global _automarker
    return AutoMarkers[_automarker]

def nextAutos():
    """ Move to the next colour and marker."""
    global _autocolor
    global _automarker

    _autocolor += 1
    if _autocolor == len(AutoColors):
        _autocolor = 0

    _automarker += 1
    if _automarker == len(AutoMarkers):
        _automarker = 0


# X and Y pts for corners of polygons
_Polygons = {
    'square': ( (-1, -1), (1, -1), (1, 1), (-1, 1) ),
    'diamond': ( (0, 1), (1, 0), (0, -1), (-1, 0) ),
    'barhorz': ( (-1, -0.5), (1, -0.5), (1, 0.5), (-1, 0.5) ),
    'barvert': ( (-0.5, -1), (0.5, -1), (0.5, 1), (-0.5, 1) ),
    'plus': ( (0.3, 1), (0.3, 0.3), (1, 0.3), (1, -0.3),
              (0.3, -0.3), (0.3, -1), (-0.3, -1), (-0.3, -0.3),
              (-1, -0.3), (-1, 0.3), (-0.3, 0.3), (-0.3, 1) ),
    'octogon': ( (0.414, 1), (1, 0.414), (1, -0.414), (0.414, -1),
                 (-0.414, -1), (-1, -0.414), (-1, 0.414), (-0.414, 1) ),
    'triangle': ( (0, -1), (0.866, 0.5), (-0.866, 0.5) ),
    'cross': ( (-0.495, 0.919), (0.000, 0.424), (0.495, 0.919), (0.919, 0.495),
               (0.424, -0.000), (0.919, -0.495), (0.495, -0.919),
               (-0.000, -0.424), (-0.495, -0.919), (-0.919, -0.495),
               (-0.424, 0.000), (-0.919, 0.495) ),
    'star': ( (0.000, -1.000), (-0.225, -0.309), (-0.951, -0.309),
              (-0.363, 0.118), (-0.588, 0.809), (-0.000, 0.382),
              (0.588, 0.809), (0.363, 0.118), (0.951, -0.309),
              (0.225, -0.309) ),
    'pentagon': ( (0, -1), (0.951, -0.309), (0.578, 0.809),
                  (-0.578, 0.809), (-0.951, -0.309) ),
    'tievert': ( (-1, -1), (1, -1), (-1, 1), (1, 1) ),
    'tiehorz': ( (-1, -1), (-1, 1), (1, -1), (1, 1) )
    }

def _plotPolygons(painter, name, xpos, ypos, size):
    '''Plots shapes which are polygons'''

    # make a polygon of the correct size
    pgn = ( N.array(_Polygons[name], N.Float32) * size ).astype(N.Int32)

    for x, y in zip(xpos, ypos):
        pts = pgn + N.array( (x, y) )
        pa = qt.QPointArray( pts.flat.tolist() )
        painter.drawPolygon(pa)

def plotMarker(painter, xpos, ypos, markercode, markersize):
    """Function to plot a marker on a painter, posn xpos, ypos, type and size
    """
    if markercode in _Polygons:
        _plotPolygons(painter, markercode, xpos, ypos, markersize)
    else:
        _MarkerLookup[markercode](painter, xpos, ypos, markersize)

def plotMarkers(painter, xpos, ypos, markername, markersize):
    """Funtion to plot an array of markers on painter
    """

    if markername in _Polygons:
        _plotPolygons(painter, markername, xpos, ypos, markersize)
    else:
        fn = _MarkerLookup[markername]
        for x, y in zip(xpos, ypos):
            fn(painter, x, y, markersize)
