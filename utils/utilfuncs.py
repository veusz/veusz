# utilfuncs.py
# utility functions

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

import string
import qt
import math

def split_on_colons(str):
    """Break text into bits between colons.

    Variables in the text are ignored (between $ signs)
    """

    invar = False
    items = []
    part = ''

    for c in str + ':':
        if c == ':' and not invar:
            items.append(part)
            part = ''
        else:
            part += c

        if c == '$':
            invar = not invar

    return items

def pythonise(text):
    """Turn an expression of the form 'A b c d' into 'A(b,c,d)'.

    This is for 'pythonising' commands from a command-line interface
    to make it easier for users. We also have to take account of quotes
    and backslashes.
    """
    
    out = ''
    insingle = False    # in a single quote section
    indouble = False    # in a double quote section
    firstnonws = False  # have we reached first non WS char
    firstpart = True    # have we appended the first part of the expr
    lastbslash = False  # was the last character a back-slash

    # iterate over characters
    for c in text:

        # keep leading WS
        if c in string.whitespace and not firstnonws:
            out += c
            continue
        firstnonws = True

        # this character isn't escaped
        if not lastbslash:

            # quoted section
            if c == "'":
                insingle = not insingle
            elif c == '"':
                indouble = not indouble

            elif c == '\\':
                lastbslash = True
                continue

            # spacing between parts
            if c == ' ' and not insingle and not indouble:
                if firstpart:
                    out += '('
                    firstpart = False
                else:
                    out += ','
            else:
                out += c
        else:
            out += '\\' + c
            lastbslash = False

    # we're still in the first part
    if firstpart:
        out += '('

    # we can add a right bracket
    if not insingle and not indouble:
        out += ')'

    return out

def textout(painter, px, py, text, alignhorz = 0, alignvert = 0):
    r = painter.fontMetrics().boundingRect(text)
    
    x = px
    y = py

    if alignhorz == 0: # centre
        x -= r.width() / 2
    elif alignhorz == 1: # right
        x -= r.width()

    if alignvert  == 0: # centre
        y += r.height() / 2
    elif alignvert == -1: # top
        y += r.height()

    painter.drawText(x, y, text)

def formatNumber(num, format):
    """ Format a number in different ways.

    Format types are
    e - 1.23e20
    g - automatic change from f to e
    f - 12.34

    g* \  Change e20 to x 10^20
    e* /
    """

    if format == 'e' or format == 'e*':
        text = '%e' % num
    elif format == 'f':
        text = '%f'
    else:
        text = '%g' % num

    # split around exponential (if any)
    parts = string.split(text, 'e')

    # remove trailing zeros before an exponential and after decimal pt
    fp = parts[0]
    hitdec = False
    lastnonzero = -1
    for i in xrange(len(fp)):
        c = fp[i]
        if c != '0':
            lastnonzero = i
        if c == '.':
            hitdec = True
            if i != 0:
                lastnonzero = i-1

    if hitdec:
        fp = fp[:lastnonzero+1]

    # put back the exponential part
    if len(parts) != 1:
        sp = parts[1]

        # get rid of + on exponential and strip a leading zero
        if sp[0] == '+':
            sp = sp[1:]
            if len(sp) > 1 and sp[0] == '0':
                sp = sp[1:]
        elif sp[0] == '-':
            if len(sp) > 2 and sp[1] == '0':
                sp = sp[0:1] + sp[2]

        # change 1.2e20 to 1.2\times10^29
        if format == 'g*' or format == 'e*':

            # get rid of 1x before anything
            if fp != '1':
                fp += u'\u00d7'
            else:
                fp = ''
            fp += '10^{%s}' % sp
        else:
            fp += 'e' + sp

    return fp

def getPixelsPerPoint(painter):
    """ Returns number of pixels per point on the painter."""

    return float( painter.fontInfo().pixelSize() ) / \
           painter.fontInfo().pointSize()

def clipper(xpts, ypts, bounds):
    """ Clip points that are safe to remove.

    Takes points in xpts, ypts.
    If any are clippable by the bounds and they lie between clipped points
    then those points are remove.

       -1,-1 | 0,-1  | 1,-1
       ---------------------
       -1,0  | 0, 0  | 1, 0
       ---------------------
       -1,1  | 0, 1  | 1, 1
       
    Data are returned in an array in the form (x1,y1, x2, y2...)
    """

    x1, y1, x2, y2 = bounds

    clipx = []
    clipy = []

    # find out whether points are clippable
    for x, y in zip(xpts, ypts):

        # is clippable?
        xclip = 0
        yclip = 0

        if   x<x1: xclip = -1
        elif x>x2: xclip = 1
        if   y<y1: yclip = -1
        elif y>y2: yclip = 1

        clipx.append(xclip)
        clipy.append(yclip)

    outx = []
    outy = []

    # now go through and collect the points we need...
    nopts = len(xpts)
    for i in xrange(nopts):
        cx = clipx[i]
        cy = clipy[i]

        # unclipped
        if cx == 0 and cy == 0:
            outx.append( xpts[i] )
            outy.append( ypts[i] )
            
        else:
            dx = abs( clipx[i+1] - cx )
            dy = abs( clipy[i+1] - cy )

            # set if it may be true we may see this pt
            visible = True
            if (dx == 0 and dy == 0) or \
               (dx == 0 and cx != 0) or \
               (dy == 0 and cy != 0):
                visible = False

##     lastclip = False
##     firstpass = True
##     for x,y in zip(xpts,ypts):


##         if xclip == 0 and yclip == 0:
##             if lastclip:
##                 pts.append(lastclippedx)
##                 pts.append(lastclippedy)
##                 lastclip = False
##             pts.append(x)
##             pts.append(y)
##         else:
##             if not lastclip:
##                 pts.append(x)
##                 pts.append(y)
##                 lastclip = True
##                 oldxclip = xclip
                
##             else:

##             deltax = abs( xclip - oldxclip )
##             deltay = abs( yclip - oldyclip )


##             if (deltax != 0 and deltay == 0) or \
##                (deltax == 0 and deltay != 0):

##         if x<x1 or x>x2 or y<y1 or y>y2:
##             lastx = x
##             lasty = y
##             if not lastclip and not firstpass:
##                 pts.append(x)
##                 pts.append(y)
##             lastclip = True
##         else:
##             # non-clippable:
##             # put back the last clipped point
##             if lastclip:
##                 if len(pts) < 2 or pts[-2] != lastx or pts[-1] != lasty:
##                     pts.append(lastx)
##                     pts.append(lasty)
##                 lastclip = False
##             # add the point
##             pts.append(x)
##             pts.append(y)
##         firstpass = False

##     return pts
