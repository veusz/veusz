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

import qt
import sys

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
                '\\alpha': u'\u03b1',
                '\\beta': u'\u03b2',
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

    def split(self, text):
        """ Split text into its separate LaTeX-like parts."""

        self.parts = []
        self.part = ''
        self.parts.append(_PartBuilder.BlockStart)
        
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
                font.setPointSizeFloat( size*0.5 )
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
                font.setPointSizeFloat( size*0.5 )
                
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
        if render:
            painter.drawText( self.x, self.y, p )
            #painter.drawPoint( self.x, self.y )

        # move along, nothing to see
        self.x += width

        # return next part
        return partno + 1

def render(painter, font, x, y, text, alignhorz = -1, alignvert = -1, angle = 0):
    """ Render text at a certain position using a certain font.

    painter is the QPainter device
    font is the QFont to use
    x, y are the QPainter coordinates
    text is the text
    angle is in degrees clockwise
    alignhorz is horzontal alignment (-1=left, 0=centre, 1=right)
    alignvert is vertical alignment (-1=bottom, 0=centre, 1=top)
    """

    if len(text) == 0:
        return

    pb = _PartBuilder()
    pb.split(text)

    # if the text is rotated, change the coordinate frame
    if angle != 0:
        painter.save()
        painter.translate(x, y)
        painter.rotate(angle)
        x = 0
        y = 0
        
    painter.setFont(font)
    #painter.drawRect(x, y, 3, 3)

    # centred or top alignment
    if alignvert >= 0:
        ascent = painter.fontMetrics().ascent()
        if alignvert == 0:
            y += ascent/2  # centred
        else:
            y += ascent    # top

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
    pb.renderpart(painter, 0, font, render=1)
    totalwidth = pb.x - x

    # restore coordinate frame if text was rotated
    if angle != 0:
        painter.restore()

    # we might need this later
    return totalwidth
