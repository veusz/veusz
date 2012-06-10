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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
###############################################################################

import math
import re
import sys

import numpy as N
import veusz.qtall as qt4

import points

mmlsupport = True
try:
    import veusz.helpers.qtmml as qtmml
    import veusz.helpers.recordpaint as recordpaint
except ImportError:
    mmlsupport = False

# this definition is monkey-patched when veusz is running in self-test
# mode as we need to hack the metrics - urgh
FontMetrics = qt4.QFontMetricsF

# lookup table for special symbols
symbols = {
    # escaped characters
    r'\_': '_',
    r'\^': '^',
    r'\{': '{',
    r'\}': '}',
    r'\[': '[',
    r'\]': ']',
    r'\backslash' : u'\u005c',

    # operators
    r'\pm': u'\u00b1',
    r'\mp': u'\u2213',
    r'\times': u'\u00d7',
    r'\cdot': u'\u22c5',
    r'\ast': u'\u2217',
    r'\star': u'\u22c6',
    r'\deg': u'\u00b0',
    r'\divide': u'\u00f7',
    r'\dagger': u'\u2020',
    r'\ddagger': u'\u2021',
    r'\cup': u'\u22c3',
    r'\cap': u'\u22c2',
    r'\uplus': u'\u228e',
    r'\vee': u'\u22c1',
    r'\wedge': u'\u22c0',
    r'\nabla': u'\u2207',
    r'\lhd': u'\u22b2',
    r'\rhd': u'\u22b3',
    r'\unlhd': u'\u22b4',
    r'\unrhd': u'\u22b5',

    r'\oslash': u'\u2298',
    r'\odot': u'\u0e4f',
    r'\oplus': u'\u2295',
    r'\ominus': u'\u2296',
    r'\otimes': u'\u2297',

    r'\diamond': u'\u22c4',
    r'\bullet': u'\u2022',
    r'\AA': u'\u212b',
    r'\sqrt': u'\u221a',
    r'\propto': u'\u221d',
    r'\infty': u'\u221e',
    r'\int': u'\u222b',
    r'\leftarrow': u'\u2190',
    r'\Leftarrow': u'\u21d0',
    r'\uparrow': u'\u2191',
    r'\rightarrow': u'\u2192',
    r'\to': u'\u2192',
    r'\Rightarrow': u'\u21d2',
    r'\downarrow': u'\u2193',
    r'\leftrightarrow': u'\u2194',
    r'\Leftrightarrow': u'\u21d4',
    r'\circ': u'\u0e50',

    # relations
    r'\le': u'\u2264',
    r'\ge': u'\u2265',
    r'\neq': u'\u2260',
    r'\sim': u'\u223c',
    r'\ll': u'\u226a',
    r'\gg': u'\u226b',
    r'\doteq': u'\u2250',
    r'\simeq': u'\u2243',
    r'\subset': u'\u2282',
    r'\supset': u'\u2283',
    r'\approx': u'\u2248',
    r'\asymp': u'\u224d',
    r'\subseteq': u'\u2286',
    r'\supseteq': u'\u2287',
    r'\sqsubset': u'\u228f',
    r'\sqsupset': u'\u2290',
    r'\sqsubseteq': u'\u2291',
    r'\sqsupseteq': u'\u2292',
    r'\in': u'\u2208',
    r'\ni': u'\u220b',
    r'\equiv': u'\u2261',
    r'\prec': u'\u227a',
    r'\succ': u'\u227b',
    r'\preceq': u'\u227c',
    r'\succeq': u'\u227d',
    r'\bowtie': u'\u22c8',
    r'\vdash': u'\u22a2',
    r'\dashv': u'\u22a3',
    r'\models': u'\u22a7',
    r'\perp': u'\u22a5',
    r'\parallel': u'\u2225',
    r'\umid': u'\u2223',

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

class RenderState(object):
    """Holds the state of the rendering."""
    def __init__(self, font, painter, x, y, alignhorz,
                 actually_render=True):
        self.font = font
        self.painter = painter
        self.device = painter.device()
        self.x = x     # current x position
        self.y = y     # current y position
        self.alignhorz = alignhorz
        self.actually_render = actually_render
        self.maxlines = 1 # maximim number of lines drawn

    def fontMetrics(self):
        """Returns font metrics object."""
        return FontMetrics(self.font, self.device)

    def getPixelsPerPt(self):
        """Return number of pixels per point in the rendering."""
        painter = self.painter
        pixperpt = painter.device().logicalDpiY() / 72.
        try:
            pixperpt *= painter.scaling
        except AttributeError:
            pass
        return pixperpt

class Part(object):
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
    
    def addText(self, text):
        self.text += text
    
    def render(self, state):
        """Render some text."""

        width = state.fontMetrics().width(self.text)

        # actually write the text if requested
        if state.actually_render:
            state.painter.drawText( qt4.QPointF(state.x, state.y), self.text )
            
        # move along, nothing to see
        state.x += width

class PartLines(Part):
    """Render multiple lines."""

    def __init__(self, children):
        Part.__init__(self, children)
        self.widths = []

    def render(self, state):
        """Render multiple lines."""
        # record widths of individual lines
        if not state.actually_render:
            self.widths = []

        height = state.fontMetrics().height()
        inity = state.y
        initx = state.x

        state.y -= height*(len(self.children)-1)

        # iterate over lines (reverse as we draw from bottom up)
        for i, part in enumerate(self.children):
            if state.actually_render and self.widths:
                xwidth = max(self.widths)
                # if we're rendering, use max width to justify line
                if state.alignhorz < 0:
                    # left alignment
                    state.x = initx
                elif state.alignhorz == 0:
                    # centre alignment
                    state.x = initx + (xwidth - self.widths[i])*0.5
                elif state.alignhorz > 0:
                    # right alignment
                    state.x = initx + (xwidth - self.widths[i])
            else:
                # if not, just left justify to get widths
                state.x = initx

            # render the line itself
            part.render(state)

            # record width if we're not rendering
            if not state.actually_render:
                self.widths.append( state.x - initx )
            # move up a line
            state.y += height

        # move on x posn
        if self.widths:
            state.x = initx + max(self.widths)
        else:
            state.x = initx
        state.y = inity
        # keep track of number of lines rendered
        state.maxlines = max(state.maxlines, len(self.children))

class PartSuperScript(Part):
    """Represents superscripted part."""
    def render(self, state):
        font = state.font
        painter = state.painter

        # change text height
        oldheight = state.fontMetrics().height()
        size = font.pointSizeF()
        font.setPointSizeF(size*0.6)
        painter.setFont(font)

        # set position
        oldy = state.y
        state.y -= oldheight*0.4

        # draw children
        Part.render(self, state)

        # restore font and position
        state.y = oldy
        font.setPointSizeF(size)
        painter.setFont(font)

class PartFrac(Part):
    """"A fraction, do latex \frac{a}{b}."""

    def render(self, state):
        if len(self.children) != 2:
            return

        font = state.font
        painter = state.painter

        # make font half size
        size = font.pointSizeF()
        font.setPointSizeF(size*0.5)
        painter.setFont(font)

        # keep track of width above and below line
        if not state.actually_render:
            self.widths = []

        initx = state.x
        inity = state.y

        # render bottom of fraction
        if state.actually_render and len(self.widths) == 2:
            # centre line
            state.x = initx + (max(self.widths) - self.widths[0])*0.5
        self.children[1].render(state)
        if not state.actually_render:
            # get width if not rendering
            self.widths.append(state.x - initx)

        # render top of fraction
        m = state.fontMetrics()
        state.y -= (m.ascent() + m.descent())
        if state.actually_render and len(self.widths) == 2:
            # centre line
            state.x = initx + (max(self.widths) - self.widths[1])*0.5
        else:
            state.x = initx
        self.children[0].render(state)
        if not state.actually_render:
            self.widths.append(state.x - initx)

        state.x = initx + max(self.widths)
        state.y = inity

        # restore font
        font.setPointSizeF(size)
        painter.setFont(font)
        height = state.fontMetrics().ascent()

        # draw line between lines with 0.5pt thickness
        painter.save()
        pen = painter.pen()
        painter.setPen( qt4.QPen(painter.pen().brush(),
                                 state.getPixelsPerPt()*0.5) )
        painter.setPen(pen)

        painter.drawLine(qt4.QPointF(initx,
                                     inity-height/2.),
                         qt4.QPointF(initx+max(self.widths),
                                     inity-height/2))

        painter.restore()

class PartSubScript(Part):
    """Represents subscripted part."""
    def render(self, state):
        font = state.font

        # change text height
        size = font.pointSizeF()
        font.setPointSizeF(size*0.6)
        state.painter.setFont(font)

        # set position
        oldy = state.y
        state.y += state.fontMetrics().descent()

        # draw children
        Part.render(self, state)

        # restore font and position
        state.y = oldy
        font.setPointSizeF(size)
        state.painter.setFont(font)

class PartMultiScript(Part):
    """Represents multiple parts with the same starting x, e.g. a combination of
       super- and subscript parts."""
    def render(self, state):
        oldx = state.x
        newx = oldx
        for p in self.children:
            state.x = oldx
            p.render(state)
            newx = max([state.x, newx])
        state.x = newx
    
    def append(self, p):
        self.children.append(p)

class PartItalic(Part):
    """Represents italic part."""
    def render(self, state):
        font = state.font

        font.setItalic( not font.italic() )
        state.painter.setFont(font)
                
        Part.render(self, state)
                
        font.setItalic( not font.italic() )
        state.painter.setFont(font)

class PartBold(Part):
    """Represents bold part."""
    def render(self, state):
        font = state.font

        font.setBold( not font.bold() )
        state.painter.setFont(font)
                
        Part.render(self, state)
                
        font.setBold( not font.bold() )
        state.painter.setFont(font)

class PartUnderline(Part):
    """Represents underlined part."""
    def render(self, state):
        font = state.font

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
        self.size = None
        self.deltasize = None

        # convert size
        try:
            size = children[0].text.replace('pt', '') # crap code
            if size[:1] in '+-':
                # is a modification of font size
                self.deltasize = float(size)
            else:
                # is an absolute font size
                self.size = float(size)
        except (AttributeError, ValueError):
            self.deltasize = 0.

        self.children = children[1:]

    def render(self, state):
        font = state.font
        size = oldsize = font.pointSizeF()

        if self.size:
            # absolute size
            size = self.size
        elif self.deltasize:
            # change of size
            size = max(size+self.deltasize, 0.1)
        
        font.setPointSizeF(size)
        state.painter.setFont(font)

        Part.render(self, state)

        font.setPointSizeF(oldsize)
        state.painter.setFont(font)

class PartBar(Part):
    """Draw a bar over text."""

    def render(self, state):
        initx = state.x

        # draw material under bar
        Part.render(self, state)

        # draw line over text with 0.5pt thickness
        painter = state.painter
        height = state.fontMetrics().ascent()

        painter.save()
        penw = state.getPixelsPerPt()*0.5
        painter.setPen( qt4.QPen(painter.pen().brush(), penw) )
        painter.drawLine(qt4.QPointF(initx,
                                     state.y-height+penw),
                         qt4.QPointF(state.x,
                                     state.y-height+penw))
        painter.restore()

class PartDot(Part):
    """Draw a dot over text."""

    def render(self, state):
        initx = state.x

        # draw material under bar
        Part.render(self, state)

        # draw circle over text with 1pt radius
        painter = state.painter
        height = state.fontMetrics().ascent()

        painter.save()
        circsize = state.getPixelsPerPt()
        painter.setBrush( qt4.QBrush(painter.pen().color()) )
        painter.setPen( qt4.QPen(qt4.Qt.NoPen) )

        x = 0.5*(initx + state.x)
        y = state.y-height + circsize
        painter.drawEllipse( qt4.QRectF(
                qt4.QPointF(x-circsize,y-circsize),
                qt4.QPointF(x+circsize,y+circsize)) )
        painter.restore()

class PartMarker(Part):
    """Draw a marker symbol."""

    def render(self, state):
        painter = state.painter
        size = state.fontMetrics().ascent()

        painter.save()
        pen = painter.pen()
        pen.setWidthF( state.getPixelsPerPt() * 0.5 )
        painter.setPen(pen)

        try:
            points.plotMarker(
                painter, state.x + size/2.,
                state.y - size/2.,
                self.children[0].text, size*0.3)
        except ValueError:
            pass

        painter.restore()

        state.x += size

class PartColor(Part):
    def __init__(self, children):
        try:
            self.colorname = children[0].text
        except AttributeError:
            self.colorname = ''
        self.children = children[1:]

    def render(self, state):
        painter = state.painter
        pen = painter.pen()
        oldcolor = pen.color()

        pen.setColor( qt4.QColor(self.colorname) )
        painter.setPen(pen)

        Part.render(self, state)

        pen.setColor(oldcolor)
        painter.setPen(pen)

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
    r'\size': (PartSize, 2),
    r'\frac': (PartFrac, 2),
    r'\bar': (PartBar, 1),
    r'\overline': (PartBar, 1),
    r'\dot': (PartDot, 1),
    r'\marker': (PartMarker, 1),
    r'\color': (PartColor, 2),
    }

# split up latex expression into bits
splitter_re = re.compile(r'''
(
\\[A-Za-z]+[ ]* |   # normal latex command
\\[\[\]{}_^] |      # escaped special characters
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
            ps = p.rstrip()
            if ps in symbols:
                # it will become a symbol, so preserve whitespace
                doAdd(ps)
                if ps != p:
                    doAdd(p[len(ps)-len(p):])
            else:
                # add as possible command, so drop excess whitespace
                doAdd(ps)
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

    lines = []
    itemlist = []
    length = len(partlist)
    
    def addText(text):
        """Try to merge consecutive text items for better rendering."""
        if itemlist and isinstance(itemlist[-1], PartText):
            itemlist[-1].addText(text)
        else:
            itemlist.append( PartText(text) )
    
    i = 0
    while i < length:
        p = partlist[i]
        if p == r'\\':
            lines.append( Part(itemlist) )
            itemlist = []
        elif isinstance(p, basestring):
            if p in symbols:
                addText(symbols[p])
            elif p in part_commands:
                klass, numargs = part_commands[p]
                if numargs == 1 and len(partlist) > i+1 and isinstance(partlist[i+1], basestring):
                    # coerce a single argument to a partlist so that things
                    # like "A^\dagger" render correctly without needing
                    # curly brackets
                    partargs = [makePartTree([partlist[i+1]])]
                else:
                    partargs = [makePartTree(k) for k in partlist[i+1:i+numargs+1]]
                
                if (p == '^' or p == '_'):
                    if len(itemlist) > 0 and (
                        isinstance(itemlist[-1], PartSubScript) or
                        isinstance(itemlist[-1], PartSuperScript) or
                        isinstance(itemlist[-1], PartMultiScript)):
                        # combine sequences of multiple sub-/superscript parts into
                        # a MultiScript item so that a single text item can have 
                        # both super and subscript indicies
                        # e.g. X^{(q)}_{i}
                        if isinstance(itemlist[-1], PartMultiScript):
                            itemlist.append( klass(partargs) )
                        else:
                            itemlist[-1] = PartMultiScript([itemlist[-1], klass(partargs)])
                    else:
                        itemlist.append( klass(partargs) )
                else:
                    itemlist.append( klass(partargs) )
                i += numargs
            else:
                addText(p)
        else:
            itemlist.append( makePartTree(p) )
        i += 1
    # remaining items
    lines.append( Part(itemlist) )

    if len(lines) == 1:
        # single line, so optimize (itemlist == lines[0] still)
        if len(itemlist) == 1:
            # try to flatten any excess layers
            return itemlist[0]
        else:
            return lines[0]
    else:
        return PartLines(lines)

class _Renderer:
    """Different renderer types based on this."""

    def __init__(self, painter, font, x, y, text,
                 alignhorz = -1, alignvert = -1, angle = 0,
                 usefullheight = False):

        self.painter = painter
        self.font = font
        self.alignhorz = alignhorz
        self.alignvert = alignvert
        self.angle = angle
        self.usefullheight = usefullheight

        # x and y are the original coordinates
        # xi and yi are adjusted for alignment
        self.x = self.xi = x
        self.y = self.yi = y
        self.calcbounds = None

        self._initText(text)

    def _initText(self, text):
        """Override this to set up renderer with text."""

    def ensureInBox(self, minx = -32767, maxx = 32767,
                    miny = -32767, maxy = 32767, extraspace = False):
        """Adjust position of text so that it is within this box."""

        if self.calcbounds is None:
            self.getBounds()

        cb = self.calcbounds

        # add a small amount of extra room if requested
        if extraspace:
            self.painter.setFont(self.font)
            l = FontMetrics(
                self.font,
                self.painter.device()).height()*0.2
            miny += l

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

    def _getWidthHeight(self):
        """Calculate the width and height of rendered text.

        Return totalwidth, totalheight, dy
        dy is a descent to add, to include in the alignment, if wanted
        """

    def getBounds(self):
        """Get bounds in standard version."""

        if self.calcbounds is not None:
            return self.calcbounds

        totalwidth, totalheight, dy = self._getWidthHeight()

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
            self.xi += (newx[0] - newbound[0])
        elif self.alignhorz > 0:
            xr = ( self.x-(newbound[2]-newbound[0]), self.x )
            self.xi += (newx[0] - newbound[2])
        else:
            xr = ( self.x+newbound[0], self.x+newbound[2] )
            self.xi += newx[0]

        # y alignment
        # adjust y by these values to ensure proper alignment
        if self.alignvert < 0:
            yr = ( self.y + (newbound[1]-newbound[3]), self.y )
            self.yi += (newy[0] - newbound[3])
        elif self.alignvert > 0:
            yr = ( self.y, self.y + (newbound[3]-newbound[1]) )
            self.yi += (newy[0] - newbound[1])
        else:
            yr = ( self.y+newbound[1], self.y+newbound[3] )
            self.yi += newy[0]

        self.calcbounds = [xr[0], yr[0], xr[1], yr[1]]
        return self.calcbounds

class _StdRenderer(_Renderer):
    """Standard rendering class."""

    def _initText(self, text):
        # make internal tree
        partlist = makePartList(text)
        self.parttree = makePartTree(partlist)

    def _getWidthHeight(self):
        """Get size of box around text."""

        # work out total width and height
        self.painter.setFont(self.font)

        # work out height of box, and
        # make the bounding box a bit bigger if we want to include descents

        state = RenderState(self.font, self.painter, 0, 0,
                            self.alignhorz,
                            actually_render = False)
        fm = state.fontMetrics()

        if self.usefullheight:
            totalheight = fm.ascent()
            dy = fm.descent()
        else:
            if self.alignvert == 0:
                # if want vertical centering, better to centre around middle
                # of typical letter (i.e. where strike position is)
                #totalheight = fm.strikeOutPos()*2
                totalheight = fm.boundingRect(qt4.QChar('0')).height()
            else:
                # if top/bottom alignment, better to use maximum letter height
                totalheight = fm.ascent()
            dy = 0

        # work out width
        self.parttree.render(state)
        totalwidth = state.x
        # add number of lines for height
        totalheight += fm.height()*(state.maxlines-1)

        return totalwidth, totalheight, dy

    def render(self):
        """Render the text."""

        if self.calcbounds is None:
            self.getBounds()

        state = RenderState(self.font, self.painter,
                            self.xi, self.yi,
                            self.alignhorz)

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

class _MmlRenderer(_Renderer):
    """MathML renderer."""

    def _initText(self, text):
        """Setup MML document and draw it in recording paint device."""

        self.size = qt4.QSize(1, 1)
        if not mmlsupport:
            self.mmldoc = None
            sys.stderr.write('Error: MathML support not built\n')
            return

        self.mmldoc = doc = qtmml.QtMmlDocument()
        try:
            self.mmldoc.setContent(text)
        except ValueError, e:
            self.mmldoc = None
            sys.stderr.write('Error interpreting MathML: %s\n' %
                             unicode(e))
            return

        # this is pretty horrible :-(

        # We write the mathmml document to a RecordPaintDevice device
        # at the same DPI as the screen, because the MML code breaks
        # for other DPIs. We then repaint the output to the real
        # device, scaling to make the size correct.

        screendev = qt4.QApplication.desktop()
        self.record = recordpaint.RecordPaintDevice(
            1024, 1024, screendev.logicalDpiX(), screendev.logicalDpiY())

        rpaint = qt4.QPainter(self.record)
        # painting code relies on these attributes of the painter
        rpaint.pixperpt = screendev.logicalDpiY() / 72.
        rpaint.scaling = 1.0

        # Upscale any drawing by this factor, then scale back when
        # drawing. We have to do this to get consistent output at
        # different zoom factors (I hate this code).
        upscale = 5.

        doc.setFontName( qtmml.QtMmlWidget.NormalFont, self.font.family() )

        ptsize = self.font.pointSizeF()
        if ptsize < 0:
            ptsize = self.font.pixelSize() / self.painter.pixperpt
        ptsize /= self.painter.scaling

        doc.setBaseFontPointSize(ptsize * upscale)

        # the output will be painted finally scaled
        self.drawscale = (
            self.painter.scaling * self.painter.dpi / screendev.logicalDpiY()
            / upscale )
        self.size = doc.size() * self.drawscale

        doc.paint(rpaint, qt4.QPoint(0, 0))
        rpaint.end()

    def _getWidthHeight(self):
        return self.size.width(), self.size.height(), 0

    def render(self):
        """Render the text."""

        if self.calcbounds is None:
            self.getBounds()

        if self.mmldoc is not None:
            p = self.painter
            p.save()
            p.translate(self.xi, self.yi)
            p.rotate(self.angle)
            # is drawn from bottom of box, not top
            p.translate(0, -self.size.height())
            p.scale(self.drawscale, self.drawscale)
            self.record.play(p)
            p.restore()

        return self.calcbounds

# identify mathml text
mml_re = re.compile(r'^\s*<math.*</math\s*>\s*$', re.DOTALL)

def Renderer(painter, font, x, y, text,
                alignhorz = -1, alignvert = -1, angle = 0,
                usefullheight = False):
    """Return an appropriate Renderer object depending on the text.
    This looks like a class name, because it was a class originally.

    painter is the painter to draw on
    font is the starting font to use
    x and y are the x and y positions to draw the text at
    alignhorz = (-1, 0, 1) for (left, centre, right) alignment
    alignvert = (-1, 0, 1) for (above, centre, below) alignment
    angle is the angle to draw the text at
    usefullheight means include descenders in calculation of height
    of text

    alignment is in the painter frame, not the text frame
    """

    if mml_re.match(text):
        r = _MmlRenderer
    else:
        r = _StdRenderer

    return r(
        painter, font, x, y, text,
        alignhorz=alignhorz, alignvert=alignvert,
        angle=angle, usefullheight=usefullheight
        )
