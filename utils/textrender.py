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

import numarray as N
import qt

# constants for special parts of an expression
_BlockStart = 1
_BlockEnd = 2
_SuperScript = 3
_SubScript = 4

_FontItalic = 5
_FontBold = 6
_FontUnderline = 7

_Modifiers = {
    '\\italic': _FontItalic,
    '\\emph':   _FontItalic,
    '\\bold':   _FontBold,
    '\\underline': _FontUnderline,
    '\\textbf': _FontBold,
    '\\textit': _FontItalic
    }

# lookup table for specials
_Specials = {
    '{': _BlockStart,
    '}': _BlockEnd,
    '^': _SuperScript,
    '_': _SubScript
    }

# lookup table for special symbols
_Symbols = {
    '\\times': u'\u00d7',
    '\\pm': u'\u00b1',
    '\\deg': u'\u00b0',
    '\\divide': u'\u00f7',
    '\\dagger': u'\u2020',
    '\\ddagger': u'\u2021',
    '\\bullet': u'\u2022',
    '\\AA': u'\u212b',
    '\\sqrt': u'\u221a',
    '\\propto': u'\u221d',
    '\\infty': u'\u221e',
    '\\int': u'\u222b',
    '\\sim': u'\u223c',
    
    # lower case greek letters
    '\\alpha': u'\u03b1',
    '\\beta': u'\u03b2',
    '\\gamma': u'\u03b3',
    '\\delta': u'\u03b4',
    '\\epsilon': u'\u03b5',
    '\\zeta': u'\u03b6',
    '\\eta': u'\u03b7',
    '\\theta': u'\u03b8',
    '\\iota': u'\u03b9',
    '\\kappa': u'\u03ba',
    '\\lambda': u'\u03bb',
    '\\mu': u'\u03bc',
    '\\nu': u'\u03bd',
    '\\xi': u'\u03be',
    '\\omicron': u'\u03bf',
    '\\pi': u'\u03c0',
    '\\rho': u'\u03c1',
    '\\stigma': u'\u03c2',
    '\\sigma': u'\u03c3',
    '\\tau': u'\u03c4',
    '\\upsilon': u'\u03c5',
    '\\phi': u'\u03c6',
    '\\chi': u'\u03c7',
    '\\psi': u'\u03c8',
    '\\omega': u'\u03c9',
    
    # upper case greek letters
    '\\Alpha': u'\u0391',
    '\\Beta': u'\u0392',
    '\\Gamma': u'\u0393',
    '\\Delta': u'\u0394',
    '\\Epsilon': u'\u0395',
    '\\Zeta': u'\u0396',
    '\\Eta': u'\u0397',
    '\\Theta': u'\u0398',
    '\\Iota': u'\u0399',
    '\\Kappa': u'\u039a',
    '\\Lambda': u'\u039b',
    '\\Mu': u'\u039c',
    '\\Nu': u'\u039d',
    '\\Xi': u'\u039e',
    '\\Omicron': u'\u039f',
    '\\Pi': u'\u03a0',
    '\\Rho': u'\u03a1',
    '\\Sigma': u'\u03a3',
    '\\Tau': u'\u03a4',
    '\\Upsilon': u'\u03a5',
    '\\Phi': u'\u03a6',
    '\\Chi': u'\u03a7',
    '\\Psi': u'\u03a8',
    '\\Omega': u'\u03a9'
    }

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
        self._makeParts(text)

        self.x = x
        self.y = y
        self.calcbounds = None

    def getBounds(self):
        """Get bounds of text on screen."""

        if self.calcbounds != None:
            return self.calcbounds

        # no text
        if len(self.parts) == 2:
            self.xi = self.x
            self.yi = self.y
            self.calcbounds = [self.x, self.y, self.x, self.y]
            return self.calcbounds

        # work out total width and height
        self.painter.setFont(self.font)

        fm = self.painter.fontMetrics()
        totalheight = fm.boundingRect('0').height()

        # make the bounding box a bit bigger if we want to include descents
        if self.usefullheight:
            dy = fm.descent()
        else:
            dy = 0

        self.xpos = 0
        self.ypos = 0
        self._renderPart(0, self.font, render=0)
        totalwidth = self.xpos

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
            self.xi = self.x + int(newx[0] - newbound[0])
        elif self.alignhorz > 0:
            xr = ( self.x-(newbound[2]-newbound[0]), self.x )
            self.xi = self.x + int(newx[0] - newbound[2])
        else:
            xr = ( self.x+newbound[0], self.x+newbound[2] )
            self.xi = self.x + int(newx[0])

        # y alignment
        # adjust y by these values to ensure proper alignment
        if self.alignvert < 0:
            yr = ( self.y + (newbound[1]-newbound[3]), self.y )
            self.yi = self.y + int(newy[0] - newbound[3])
        elif self.alignvert > 0:
            yr = ( self.y, self.y + (newbound[3]-newbound[1]))
            self.yi = self.y + int(newy[0] - newbound[1])
        else:
            yr = ( self.y+newbound[1], self.y+newbound[3] )
            self.yi = self.y + int(newy[0])

        self.calcbounds = [xr[0], yr[0], xr[1], yr[1]]
        return self.calcbounds

    def ensureInBox(self, minx = -32767, maxx = 32767,
                    miny = -32767, maxy = 32767, extraspace = False):
        """Adjust position of text so that it is within this box."""

        if self.calcbounds == None:
            self.getBounds()

        cb = self.calcbounds

        # add a small amount of extra room if requested
        if extraspace:
            self.painter.setFont(self.font)
            l = self.painter.fontMetrics().leading()
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

    def overlapsRegion(self, region):
        """Do the bounds of the text overlap with the qt QRegion given?"""

        if self.calcbounds == None:
            self.getBounds()

        cb = self.calcbounds
        return region.contains( qt.QRect(cb[0], cb[1],
                                         cb[2]-cb[0]+1, cb[3]-cb[1]+1) )

    def getDimensions(self):
        """Get the (w, h) of the bounding box."""

        if self.calcbounds == None:
            self.getBounds()
        cb = self.calcbounds
        return (cb[2]-cb[0]+1, cb[3]-cb[1]+1)

    def render(self):
        """Render the text."""
        
        if self.calcbounds == None:
            self.getBounds()

        self.xpos = self.xi
        self.ypos = self.yi

        # if the text is rotated, change the coordinate frame
        if self.angle != 0:
            self.painter.save()
            self.painter.translate(self.xpos, self.ypos)
            self.painter.rotate(self.angle)
            self.xpos = 0
            self.ypos = 0

        # actually paint the string
        self.painter.setFont(self.font)
        self._renderPart(0, self.font, render=1)

        # restore coordinate frame if text was rotated
        if self.angle != 0:
            self.painter.restore()

        # caller might want this information
        return self.calcbounds

    def _makeParts(self, text):
        """Text is broken up into 'parts', representing text or attributes.

        This should probably be broken up into a tree of objects.
        """

        self.parts = [_BlockStart]

        part = ''
        l = len(text)
        inmodifier = False

        i = 0
        while i < l:
            # keep track of this character and the next
            c = text[i]
            next = ''
            if i < (l-1):
                next = text[i+1]

            # end of modifier
            if inmodifier and (not c.isalpha()):
                self._addPart(part)
                part = ''
                inmodifier = False

            # handle special characters
            if c in _Specials:
                self._addPart(part)
                part = ''
                self.parts.append( _Specials[c] )
                i += 1
                continue
        
            # new modifier or escape
            elif c == '\\':
                # escaped character
                if next in '\\{}':
                    part += next
                    i += 2
                    continue

                # new modifier
                self._addPart(part)
                part = ''
                inmodifier = True

            # add character to current part and move onto the next char
            part += c
            i += 1

        # add any incomplete part
        self._addPart(part)
        self.parts.append(_BlockEnd)

    def _addPart(self, part):
        """ Adds the built part to the part list."""
        
        if len(part) != 0:

            # look up modifiers
            if part in _Modifiers:
                self.parts.append( _Modifiers[part] )

            else:
                # expand symbols
                if part in _Symbols:
                    part = _Symbols[part]

                # append to previous if previous not a modifier
                # assumption is that's it's quicker to do this that write
                # two out separately
                if len(self.parts) != 0 and \
                   not isinstance(self.parts[-1], int):
                    self.parts[-1] += part
                else:
                    self.parts.append(part)

    def _renderPart(self, partno, font, render=False):
        """ Render or measure the specified part.

        Blocks are iterated over, and recursively descended
        Returns the next part no to be rendered
        """

        p = self.parts[partno]

        # is this a modifier or just text?
        if isinstance(p, int):
            
            # if we start a new block
            if p == _BlockStart:
                # iterate over block entries
                noparts = len(self.parts)
                partno += 1
            
                while partno < noparts:
                    if self.parts[partno] == _BlockEnd:
                        partno += 1
                        break
                
                    partno = self._renderPart(partno, font, render)

            # start a superscript part
            elif p == _SuperScript:
                oldascent = self.painter.fontMetrics().ascent()
                size = font.pointSizeFloat()
                font.setPointSizeFloat(size*0.6)
                self.painter.setFont(font)

                oldy = self.ypos
                self.ypos -= (oldascent - self.painter.fontMetrics().ascent())
            
                partno = self._renderPart(partno+1, font, render)

                self.ypos = oldy
                font.setPointSizeFloat(size)
                self.painter.setFont(font)

            # start a subscript part
            elif p == _SubScript:
                size = font.pointSizeFloat()
                font.setPointSizeFloat(size*0.6)
                
                self.painter.setFont(font)
                
                partno = self._renderPart(partno+1, font, render)
                
                font.setPointSizeFloat(size)
                self.painter.setFont(font)
            
            elif p == _FontItalic:
                # toggle italic
                font.setItalic( not font.italic() )
                self.painter.setFont(font)
                
                partno = self._renderPart(partno+1, font, render)
                
                font.setItalic( not font.italic() )
                self.painter.setFont(font)
                
            elif p == _FontBold:
                # toggle bold
                font.setBold( not font.bold() )
                self.painter.setFont(font)
                
                partno = self._renderPart(partno+1, font, render)
                
                font.setBold( not font.bold() )
                self.painter.setFont(font)

            elif p == _FontUnderline:
                # toggle underline
                font.setUnderline( not font.underline() )
                self.painter.setFont(font)

                partno = self._renderPart(partno+1, font, render)

                font.setUnderline( not font.underline() )
                self.painter.setFont(font)

            # should throw something instead
            else:
                assert False

            # next part
            return partno

        else:
            # write text
            # how far do we need to advance?
            fm = self.painter.fontMetrics()
            width = fm.width( p )

            # actually write the text if requested
            if render:
                self.painter.drawText( self.xpos, self.ypos, p )
                
            # move along, nothing to see
            self.xpos += width

            # return next part
            return partno + 1
