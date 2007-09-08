# textrender.py
# module to render text, tries to understand a basic LateX-like syntax

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

import math
import re

import numpy as N
import veusz.qtall as qt4

# lookup table for special symbols
symbols = {
    # escaped characters
    r'\_': '_',
    r'\^': '^',
    r'\{': '{',
    r'\}': '}',
    r'\[': '[',
    r'\]': ']',

    # symbols
    r'\times': u'\u00d7',
    r'\pm': u'\u00b1',
    r'\deg': u'\u00b0',
    r'\divide': u'\u00f7',
    r'\dagger': u'\u2020',
    r'\ddagger': u'\u2021',
    r'\bullet': u'\u2022',
    r'\AA': u'\u212b',
    r'\sqrt': u'\u221a',
    r'\propto': u'\u221d',
    r'\infty': u'\u221e',
    r'\int': u'\u222b',
    r'\sim': u'\u223c',
    r'\odot': u'\u0e4f',
    r'\leftarrow': u'\u2190',
    r'\uparrow': u'\u2191',
    r'\rightarrow': u'\u2192',
    r'\downarrow': u'\u2193',
    r'\circ': u'\u0e50',
    
    # lower case greek letters
    r'\alpha': u'\u03b1',
    r'\beta': u'\u03b2',
    r'\gamma': u'\u03b3',
    r'\delta': u'\u03b4',
    r'\epsilon': u'\u03b5',
    r'\zeta': u'\u03b6',
    r'\eta': u'\u03b7',
    r'\theta': u'\u03b8',
    r'\iota': u'\u03b9',
    r'\kappa': u'\u03ba',
    r'\lambda': u'\u03bb',
    r'\mu': u'\u03bc',
    r'\nu': u'\u03bd',
    r'\xi': u'\u03be',
    r'\omicron': u'\u03bf',
    r'\pi': u'\u03c0',
    r'\rho': u'\u03c1',
    r'\stigma': u'\u03c2',
    r'\sigma': u'\u03c3',
    r'\tau': u'\u03c4',
    r'\upsilon': u'\u03c5',
    r'\phi': u'\u03c6',
    r'\chi': u'\u03c7',
    r'\psi': u'\u03c8',
    r'\omega': u'\u03c9',
    
    # upper case greek letters
    r'\Alpha': u'\u0391',
    r'\Beta': u'\u0392',
    r'\Gamma': u'\u0393',
    r'\Delta': u'\u0394',
    r'\Epsilon': u'\u0395',
    r'\Zeta': u'\u0396',
    r'\Eta': u'\u0397',
    r'\Theta': u'\u0398',
    r'\Iota': u'\u0399',
    r'\Kappa': u'\u039a',
    r'\Lambda': u'\u039b',
    r'\Mu': u'\u039c',
    r'\Nu': u'\u039d',
    r'\Xi': u'\u039e',
    r'\Omicron': u'\u039f',
    r'\Pi': u'\u03a0',
    r'\Rho': u'\u03a1',
    r'\Sigma': u'\u03a3',
    r'\Tau': u'\u03a4',
    r'\Upsilon': u'\u03a5',
    r'\Phi': u'\u03a6',
    r'\Chi': u'\u03a7',
    r'\Psi': u'\u03a8',
    r'\Omega': u'\u03a9'
    }

class RenderState:
    """Holds the state of the rendering."""
    def __init__(self, font, painter, x, y,
                 actually_render=True):
        self.font = font
        self.painter = painter
        self.x = x
        self.y = y
        self.actually_render = actually_render

class Part:
    """Represents a part of the text to be rendered, made up of smaller parts."""
    def __init__(self, children):
        self.children = children

    def render(self, state):
        for p in self.children:
            p.render(state)

class PartText(Part):
    """Fundamental bit of text to be rendered: some text."""
    def __init__(self, text):
        self.text = text
    
    def render(self, state):
        """Render some text."""

        width = qt4.QFontMetricsF(state.font,
                                  state.painter.device()).width(self.text)

        # actually write the text if requested
        if state.actually_render:
            state.painter.drawText( qt4.QPointF(state.x, state.y), self.text )
            
        # move along, nothing to see
        state.x += width

class PartSuperScript(Part):
    """Represents superscripted part."""
    def render(self, state):
        font = state.font
        painter = state.painter

        # change text height
        oldheight = qt4.QFontMetricsF(font, painter.device()).height()
        size = font.pointSizeF()
        font.setPointSizeF(size*0.6)
        painter.setFont(font)

        # set position
        oldy = state.y
        state.y -= ( oldheight -
                     qt4.QFontMetricsF(font, painter.device()).height() )

        # draw children
        Part.render(self, state)

        # restore font and position
        state.y = oldy
        font.setPointSizeF(size)
        painter.setFont(font)

class PartSubScript(Part):
    """Represents subscripted part."""
    def render(self, state):
        font = state.font
        painter = state.painter

        # change text height
        size = font.pointSizeF()
        font.setPointSizeF(size*0.6)
        painter.setFont(font)

        # set position
        oldy = state.y
        state.y += qt4.QFontMetricsF(font, state.painter.device()).descent()

        # draw children
        Part.render(self, state)

        # restore font and position
        state.y = oldy
        font.setPointSizeF(size)
        painter.setFont(font)

class PartItalic(Part):
    """Represents italic part."""
    def render(self, state):
        font = state.font
        painter = state.painter

        font.setItalic( not font.italic() )
        state.painter.setFont(font)
                
        Part.render(self, state)
                
        font.setItalic( not font.italic() )
        state.painter.setFont(font)

class PartBold(Part):
    """Represents bold part."""
    def render(self, state):
        font = state.font
        painter = state.painter

        font.setBold( not font.bold() )
        state.painter.setFont(font)
                
        Part.render(self, state)
                
        font.setBold( not font.bold() )
        state.painter.setFont(font)

class PartUnderline(Part):
    """Represents underlined part."""
    def render(self, state):
        font = state.font
        painter = state.painter

        font.setUnderline( not font.underline() )
        state.painter.setFont(font)

        Part.render(self, state)
                
        font.setUnderline( not font.underline() )
        state.painter.setFont(font)

class PartFont(Part):
    """Change font name in part."""
    def __init__(self, children):
        try:
            self.fontname = children[0].text
        except AttributeError:
            self.fontname = ''
        self.children = children[1:]

    def render(self, state):
        font = state.font
        oldfamily = font.family()
        font.setFamily(self.fontname)
        state.painter.setFont(font)

        Part.render(self, state)

        font.setFamily(oldfamily)
        state.painter.setFont(font)

class PartSize(Part):
    """Change font size in part."""
    def __init__(self, children):
        try:
            self.size = float(children[0].text)
        except AttributeError, ValueError:
            self.size = 0.
        self.children = children[1:]

    def render(self, state):
        font = state.font
        oldsize = font.pointSizeF()
        font.setPointSizeF(oldsize+self.size)
        state.painter.setFont(font)

        Part.render(self, state)

        font.setPointSizeF(oldsize)
        state.painter.setFont(font)

# a dict of latex commands, the part object they correspond to,
# and the number of arguments
part_commands = {
    '^': (PartSuperScript, 1),
    '_': (PartSubScript, 1),
    r'\italic': (PartItalic, 1),
    r'\emph': (PartItalic, 1),
    r'\bold': (PartBold, 1),
    r'\underline': (PartUnderline, 1),
    r'\textbf': (PartBold, 1),
    r'\textit': (PartItalic, 1),
    r'\font': (PartFont, 2),
    r'\size': (PartSize, 2)
    }

# split up latex expression into bits
splitter_re = re.compile(r'''
(
\\[A-Za-z]+[ ]* |   # normal latex command
\\\{ | \\\} |       # escaped {} brackets
\\\[ | \\\] |       # escaped [] brackets
\\\\ |              # line end
\{ |                # begin block
\} |                # end block
\^ |                # power
_                   # subscript
)
''', re.VERBOSE)

def makePartList(text):
    """Make list of parts from text"""
    parts = []
    parents = [parts]
    
    def doAdd(p):
        """Add the part at the correct level."""
        parents[-1].append(p)
        return p

    for p in splitter_re.split(text):
        if p[:1] == '\\':
            # we may need to drop excess spaces after \foo commands
            p = p.rstrip()
            if p in symbols:
                # convert to symbol if possible
                doAdd(symbols[p])
            else:
                # add as possible command
                doAdd(p)
        elif p == '{':
            # add a new level
            parents.append( doAdd([]) )
        elif p == '}':
            if len(parents) > 1:
                parents.pop()
        elif p:
            # if not blank, keep it
            doAdd(p)
    return parts

def makePartTree(partlist):
    """Make a tree of parts from the part list."""

    itemlist = []
    length = len(partlist)
    i = 0
    while i < length:
        p = partlist[i]
        if isinstance(p, basestring):
            if p in part_commands:
                klass, numargs = part_commands[p]
                partargs = [makePartTree(k) for k in partlist[i+1:i+numargs+1]]
                itemlist.append( klass(partargs) )
                i += numargs
            else:
                itemlist.append( PartText(p) )
        else:
            itemlist.append( makePartTree(p) )
        i += 1

    if len(itemlist) == 1:
        # try to flatten any excess layers
        return itemlist[0]
    else:
        return Part(itemlist)

class Renderer:
    """A class for rendering text.

    The class emulates latex-like formatting, allows rotated text, and alignment
    """



    def __init__(self, painter, font, x, y, text,
                 alignhorz = -1, alignvert = -1, angle = 0,
                 usefullheight = False):
        """Initialise the renderer.

        painter is the painter to draw on
        font is the starting font to use
        x and y are the x and y positions to draw the text at
        alignhorz = (-1, 0, 1) for (left, centre, right) alignment
        alignvert = (-1, 0, 1) for (above, centre, below) alignment
        angle is the angle to draw the text at
        usefullheight means include descenders in calculation of height
          of thext

        alignment is in the painter frame, not the text frame
        """

        # save things we'll need later
        self.painter = painter
        self.font = font
        self.alignhorz = alignhorz
        self.alignvert = alignvert
        self.angle = angle
        self.usefullheight = usefullheight

        partlist = makePartList(text)
        self.parttree = makePartTree(partlist)

        self.x = x
        self.y = y
        self.calcbounds = None

    def getBounds(self):
        """Get bounds of text on screen."""

        if self.calcbounds is not None:
            return self.calcbounds

        # no text
        #if len(self.parts) == 2:
        #    self.xi = self.x
        #    self.yi = self.y
        #    self.calcbounds = [self.x, self.y, self.x, self.y]
        #    return self.calcbounds

        # work out total width and height
        self.painter.setFont(self.font)

        # work out height of box, and
        # make the bounding box a bit bigger if we want to include descents

        fm = qt4.QFontMetricsF(self.font, self.painter.device())
        
        if self.usefullheight:
            totalheight = fm.ascent()
            dy = fm.descent()
        else:
            if self.alignvert == 0:
                # if want vertical centering, better to centre around middle
                # of typical letter
                totalheight = fm.boundingRect(qt4.QChar('0')).height()
                
            else:
                # if top/bottom alignment, better to use maximum letter height
                totalheight = fm.ascent()
            dy = 0

        # work out width
        state = RenderState(self.font, self.painter, 0, 0,
                            actually_render = False)
        self.parttree.render(state)
        totalwidth = state.x

        # in order to work out text position, we rotate a bounding box
        # in fact we add two extra points to account for descent if reqd
        tw = totalwidth / 2
        th = totalheight / 2
        coordx = N.array( [-tw,  tw,  tw, -tw, -tw,    tw   ] )
        coordy = N.array( [ th,  th, -th, -th,  th+dy, th+dy] )
        
        # rotate angles by theta
        theta = -self.angle * (math.pi / 180.)
        c = math.cos(theta)
        s = math.sin(theta)
        newx = coordx*c + coordy*s
        newy = coordy*c - coordx*s

        # calculate bounding box
        newbound = (newx.min(), newy.min(), newx.max(), newy.max())

        # use rotated bounding box to find position of start text posn
        if self.alignhorz < 0:
            xr = ( self.x, self.x+(newbound[2]-newbound[0]) )
            self.xi = self.x + (newx[0] - newbound[0])
        elif self.alignhorz > 0:
            xr = ( self.x-(newbound[2]-newbound[0]), self.x )
            self.xi = self.x + (newx[0] - newbound[2])
        else:
            xr = ( self.x+newbound[0], self.x+newbound[2] )
            self.xi = self.x + newx[0]

        # y alignment
        # adjust y by these values to ensure proper alignment
        if self.alignvert < 0:
            yr = ( self.y + (newbound[1]-newbound[3]), self.y )
            self.yi = self.y + (newy[0] - newbound[3])
        elif self.alignvert > 0:
            yr = ( self.y, self.y + (newbound[3]-newbound[1]))
            self.yi = self.y + (newy[0] - newbound[1])
        else:
            yr = ( self.y+newbound[1], self.y+newbound[3] )
            self.yi = self.y + newy[0]

        self.calcbounds = [xr[0], yr[0], xr[1], yr[1]]
        return self.calcbounds

    def ensureInBox(self, minx = -32767, maxx = 32767,
                    miny = -32767, maxy = 32767, extraspace = False):
        """Adjust position of text so that it is within this box."""

        if self.calcbounds is None:
            self.getBounds()

        cb = self.calcbounds

        # add a small amount of extra room if requested
        if extraspace:
            self.painter.setFont(self.font)
            l = qt4.QFontMetricsF(self.font, self.painter.device()).leading()
            minx += l
            maxx -= l
            miny += l
            maxy -= l

        # twiddle positions and bounds
        if cb[2] > maxx:
            dx = cb[2] - maxx
            self.xi -= dx
            cb[2] -= dx
            cb[0] -= dx

        if cb[0] < minx:
            dx = minx - cb[0]
            self.xi += dx
            cb[2] += dx
            cb[0] += dx

        if cb[3] > maxy:
            dy = cb[3] - maxy
            self.yi -= dy
            cb[3] -= dy
            cb[1] -= dy

        if cb[1] < miny:
            dy = miny - cb[1]
            self.yi += dy
            cb[3] += dy
            cb[1] += dy

    def getDimensions(self):
        """Get the (w, h) of the bounding box."""

        if self.calcbounds is None:
            self.getBounds()
        cb = self.calcbounds
        return (cb[2]-cb[0]+1, cb[3]-cb[1]+1)

    def render(self):
        """Render the text."""

        if self.calcbounds is None:
            self.getBounds()

        state = RenderState(self.font, self.painter,
                            self.xi, self.yi)

        # if the text is rotated, change the coordinate frame
        if self.angle != 0:
            self.painter.save()
            self.painter.translate( qt4.QPointF(state.x, state.y) )
            self.painter.rotate(self.angle)
            state.x = 0
            state.y = 0

        # actually paint the string
        self.painter.setFont(self.font)
        self.parttree.render(state)

        # restore coordinate frame if text was rotated
        if self.angle != 0:
            self.painter.restore()

        # caller might want this information
        return self.calcbounds
