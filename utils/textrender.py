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

class _PartBuilder:
    """ A class to handle LaTeX-like expressions."""
    
    # constants for special parts of an expression
    BlockStart = 1
    BlockEnd = 2
    SuperScript = 3
    SubScript = 4

    FontItalic = 5
    FontBold = 6
    FontUnderline = 7

    Modifiers = {
        '\\italic': FontItalic,
        '\\emph': FontItalic,
        '\\bold': FontBold,
        '\\underline': FontUnderline
        }

    # lookup table for specials
    Specials = { '{': BlockStart, '}': BlockEnd,
                 '^': SuperScript, '_': SubScript }

    # lookup table for special symbols
    Symbols = { '\\times': u'\u00d7',
                '\\pm': u'\u00b1',
                '\\deg': u'\u00b0',
                '\\divide': u'\u00f7',

                # greek letters
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

                # special symbols
                '\\dagger': u'\u2020',
                '\\ddagger': u'\u2021',
                '\\bullet': u'\u2022',
                '\\AA': u'\u212b',
                '\\sqrt': u'\u221a',
                '\\propto': u'\u221d',
                '\\infty': u'\u221e',
                '\\int': u'\u222b',
                '\\sim': u'\u223c' }
    
    def addpart(self):
        """ Adds the built part to the part list."""
        
        if len(self.part) != 0:

            # look up modifiers
            if self.part in _PartBuilder.Modifiers:
                self.parts.append( _PartBuilder.Modifiers[self.part] )

            else:
                # expand symbols
                if self.part in _PartBuilder.Symbols:
                    self.part = _PartBuilder.Symbols[self.part]

                # append to previous if previous not a modifier
                # assumption is that's it's quicker to do this that write
                # two out separately
                if len(self.parts) != 0 and \
                   not isinstance(self.parts[-1], int):
                    self.parts[-1] += self.part
                else:
                    self.parts.append(self.part)

            self.part = ''

    def __init__(self, text):
        """ Split text into its separate LaTeX-like parts."""

        self.parts = []
        self.part = ''
        self.parts.append(_PartBuilder.BlockStart)
        self.reset_bounds()
        self.x = self.y = 0
        
        l = len(text)
        inmodifier = 0

        i = 0
        while i<l:
            # keep track of this character and the next
            c = text[i]
            next = ''
            if i<(l-1):
                next = text[i+1]

            # end of modifier
            if inmodifier and (not c.isalpha()):
                self.addpart()
                inmodifier = 0

            # handle special characters
            if c in _PartBuilder.Specials:
                self.addpart()
                self.parts.append( _PartBuilder.Specials[c] )
                i += 1
                continue
        
            # new modifier or escape
            elif c == '\\':
                # escaped character
                if next in '\\{}':
                    self.part += next
                    i += 2
                    continue

                # new modifier
                self.addpart()
                inmodifier = 1

            # add character to current part and move onto the next char
            self.part += c
            i += 1

        # add any incomplete part
        self.addpart()
        self.parts.append(_PartBuilder.BlockEnd)

    def reset_bounds(self):
        """Reset bounds to nothing."""
        self.bounds = [ 10000000, 10000000, -10000000, -10000000 ]

    def _update_bounds(self, x, y):
        """Increase bounds if pixel outside them."""
        if x < self.bounds[0]: self.bounds[0] = x
        if y < self.bounds[1]: self.bounds[1] = y
        if x > self.bounds[2]: self.bounds[2] = x
        if y > self.bounds[3]: self.bounds[3] = y

    def getparts(self):
        """ Return the separate parts."""
        return self.parts

    def renderpart(self, painter, partno, font, render=0):
        """ Render or measure the specified part.

        Blocks are iterated over, and recursively descended
        Returns the next part no to be rendered
        """

        p = self.parts[partno]

        # is this a modifier or just text?
        if isinstance(p, int):
            
            # if we start a new block
            if p == _PartBuilder.BlockStart:
                # iterate over block entries
                noparts = len(self.parts)
                partno += 1
            
                while partno < noparts:
                    if self.parts[partno] == _PartBuilder.BlockEnd:
                        partno += 1
                        break
                
                    partno = self.renderpart(painter, partno, font, render)

            # start a superscript part
            elif p == _PartBuilder.SuperScript:
                oldascent = painter.fontMetrics().ascent()
                size = font.pointSizeFloat()
                font.setPointSizeFloat( size*0.6 )
                painter.setFont(font)

                ytemp = self.y
                self.y -= (oldascent - painter.fontMetrics().ascent())
            
                partno = self.renderpart(painter, partno+1, \
                                         font, render)

                self.y = ytemp
                font.setPointSizeFloat( size )
                painter.setFont(font)

            # start a subscript part
            elif p == _PartBuilder.SubScript:
                size = font.pointSizeFloat()
                font.setPointSizeFloat( size*0.6 )
                
                painter.setFont(font)
                
                partno = self.renderpart(painter, partno+1, \
                                         font, render)
                
                font.setPointSizeFloat( size )
                painter.setFont(font)
            
            elif p == _PartBuilder.FontItalic:
                # toggle italic
                font.setItalic( not font.italic() )
                painter.setFont(font)
                
                partno = self.renderpart(painter, partno+1, \
                                         font, render)
                
                font.setItalic( not font.italic() )
                painter.setFont(font)
                
            elif p == _PartBuilder.FontBold:
                # toggle bold
                font.setBold( not font.bold() )
                painter.setFont(font)
                
                partno = self.renderpart(painter, partno+1, \
                                         font, render)
                
                font.setBold( not font.bold() )
                painter.setFont(font)

            elif p == _PartBuilder.FontUnderline:
                # toggle underline
                font.setUnderline( not font.underline() )
                painter.setFont(font)

                partno = self.renderpart(painter, partno+1, \
                                         font, render)

                font.setUnderline( not font.underline() )
                painter.setFont(font)

            # should throw something instead
            else:
                print "Error! - Unknown code"

            return partno

        # just write some text

        # how far do we need to advance?
        width = painter.fontMetrics().width( p )

        # actually write the text if requested
        self._update_bounds(self.x, self.y)
        if render:
            painter.drawText( self.x, self.y, p )

        # move along, nothing to see
        self.x += width

        # update bounds with edge of this text
        self._update_bounds( self.x, self.y - painter.fontMetrics().ascent() )

        # return next part
        return partno + 1

    def get_bounds(self):
        return self.bounds

def render(painter, font, x, y, text, alignhorz = -1, alignvert = -1,
           angle = 0):
    """ Render text at a certain position using a certain font.

    painter is the QPainter device
    font is the QFont to use
    x, y are the QPainter coordinates
    text is the text
    angle is in degrees clockwise
    alignhorz is horzontal alignment (-1=left, 0=centre, 1=right)
    alignvert is vertical alignment (-1=bottom, 0=centre, 1=top)

    Returns [x1, y1, x2, y2] bounds
    """

    if len(text) == 0:
        return [x, y, x, y]

    pb = _PartBuilder(text)

    # if the text is rotated, change the coordinate frame
    if angle != 0:
        painter.save()
        painter.translate(x, y)
        painter.rotate(angle)
        origcoord = (x, y)
        x = 0
        y = 0
        
    painter.setFont(font)

    # centred or top alignment
    if alignvert >= 0:
        # why can't I use ascent() here??
        h = painter.fontMetrics().boundingRect('0').height()
        if alignvert == 0:
            y += h/2  # centred
        else:
            y += h    # top

    # centred or right alignment
    if alignhorz >= 0:
        # measure the width of the output first
        pb.x = 0
        pb.y = 0
        pb.renderpart(painter, 0, font, render=0)
        totalwidth = pb.x

        if alignhorz == 0:
            x -= totalwidth / 2 # centred
        else:
            x -= totalwidth     # right

    # actually paint the string
    pb.x = x
    pb.y = y
    pb.reset_bounds()
    pb.renderpart(painter, 0, font, render=1)
    totalwidth = pb.x - x

    retbound = pb.get_bounds()

    # restore coordinate frame if text was rotated
    if angle != 0:
        painter.restore()

        # rotate bounds in opposite direction to get real bounds...
        # hope this stuff is correct
        theta = -angle * (math.pi / 180.)
        c = math.cos(theta)
        s = math.sin(theta)

        # rotate coordinates
        b = ( retbound[0]*c + retbound[1]*s,
              retbound[1]*c - retbound[0]*s,
              retbound[2]*c + retbound[3]*s,
              retbound[3]*c - retbound[2]*s )

        # find bounding box of rotated rectangle
        # we have to add back on the original coordinates
        retbound = [ int( math.floor( min(b[0], b[2]) ) + origcoord[0] ),
                     int( math.floor( min(b[1], b[3]) ) + origcoord[1] ),
                     int( math.ceil( max(b[0], b[2]) ) + origcoord[0] ),
                     int( math.ceil( max(b[1], b[3]) ) + origcoord[1] ) ]
        
    # we might need this later
    return retbound
