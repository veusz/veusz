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

import qt

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

def _plot_plus(painter, xpos, ypos, size):
    """ (internal) function to plot a +
    size is the length of one arm
    """

    # horizontal
    painter.drawLine( xpos - size, ypos, xpos + size, ypos )
    # vertical
    painter.drawLine( xpos, ypos - size, xpos, ypos + size )

def _plot_X(painter, xpos, ypos, size):
    """ (internal) function to plot a X
    size is the length in terms of DX and DY
    """

    painter.drawLine( xpos - size, ypos - size, xpos + size, ypos + size )
    painter.drawLine( xpos - size, ypos + size, xpos + size, ypos - size )

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
   
def _plot_box(painter, xpos, ypos, size):
    """ (internal) function to plot a box shape
    size is half the length of a side
    """

    painter.drawRect( xpos - size, ypos - size, size*2+1, size*2+1 )

def _plot_box_dot(painter, xpos, ypos, size):
    """ (internal) function to plot a box shape with dot at centre
    size is half the length of a side
    """

    painter.drawRect( xpos - size, ypos - size, size*2+1, size*2+1 )
    painter.drawPoint( xpos, ypos )

def _plot_bullseye(painter, xpos, ypos, size):
    """ (internal) function to plot a bullseye shape
    size is radius
    """

    painter.drawEllipse( xpos - size/2, ypos - size/2 , size, size )
    painter.drawEllipse( xpos - size, ypos - size , size*2+1, size*2+1 )

def _plot_triangle_dot(painter, xpos, ypos, size):
    """ (internal) function to plot a triangle shape with a dot at the core
    """


    dy = int(0.577*size)
    pts = qt.QPointArray( [xpos, ypos-dy*2, xpos-size, ypos+dy,
                           xpos+size, ypos+dy] )
    painter.drawConvexPolygon( pts )
    painter.drawPoint( xpos, ypos )

def _plot_triangle(painter, xpos, ypos, size):
    """ (internal) function to plot a triangle shape
    """

    dy = int(0.577*size)
    pts = qt.QPointArray( [xpos, ypos-dy*2, xpos-size, ypos+dy,
                           xpos+size, ypos+dy] )
    painter.drawConvexPolygon( pts )

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

# list of markers and the functions to plot them
MarkerCodes = {
    'none': _plot_none,
    'X': _plot_X,
    '+': _plot_plus,
    '*': _plot_star,
    'O': _plot_circle,
    'Odot': _plot_circle_dot,
    'box': _plot_box,
    'boxdot': _plot_box_dot,
    'bullseye': _plot_bullseye,
    'triangle': _plot_triangle,
    'triangledot': _plot_triangle_dot,
    '.': _plot_point,
    '-': _plot_line_horz,
    '|': _plot_line_vert,
    'arrowleft': _plot_arrow_left,
    'arrowright': _plot_arrow_right,
    'arrowup': _plot_arrow_up,
    'arrowdown': _plot_arrow_down
}

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

def plotMarker(painter, xpos, ypos, markercode, markersize):
    """Function to plot a marker on a painter, posn xpos, ypos, type and size
    """
    MarkerCodes[markercode](painter, xpos, ypos, markersize)

def plotMarkers(painter, xpos, ypos, markername, markersize):
    """Funtion to plot an array of markers on painter
    """
    noitems = len(xpos)
    fn = MarkerCodes[markername]
    for i in xrange(noitems):
        fn(painter, xpos[i], ypos[i], markersize)
