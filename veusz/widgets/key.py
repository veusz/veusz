#    key symbol plotting

#    Copyright (C) 2005 Jeremy S. Sanders
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

from __future__ import division
import math

from ..compat import crange, citems
from .. import qtall as qt4
from .. import document
from .. import setting
from .. import utils

from . import widget
from . import controlgraph

def _(text, disambiguation=None, context='Key'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

#############################################################################
# classes for controlling key position interactively

class ControlKey(object):
    """Control the position of a key on a plot."""

    def __init__( self, widget, parentposn,
                  boxposn, boxdims,
                  textheight ):
        """widget is widget to adjust
        parentposn: posn of parent on plot
        xpos, ypos: position of key
        width. height: size of key
        textheight: 
        """
        self.widget = widget
        self.parentposn = tuple(parentposn)
        self.posn = tuple(boxposn)
        self.dims = tuple(boxdims)
        self.textheight = textheight

    def createGraphicsItem(self, parent):
        return _GraphControlKey(parent, self)

class _GraphControlKey(qt4.QGraphicsRectItem):
    """The graphical rectangle which is dragged around to reposition
    the key."""

    def __init__(self, parent, params):
        qt4.QGraphicsRectItem.__init__(
            self,
            params.posn[0], params.posn[1],
            params.dims[0], params.dims[1],
            parent)
        self.params = params

        self.setCursor(qt4.Qt.SizeAllCursor)
        self.setZValue(1.)
        self.setFlag(qt4.QGraphicsItem.ItemIsMovable)
        self.highlightpen = qt4.QPen(qt4.Qt.red, 2, qt4.Qt.DotLine)

        pposn, dims = params.parentposn, params.dims
        th = params.textheight

        # special places on the plot
        xposn = {
            'left':   pposn[0] + th,
            'centre': pposn[0] + 0.5*(pposn[2]-pposn[0]-dims[0]),
            'right':  pposn[2] - th - dims[0]
            }
        yposn = {
            'top':    pposn[1] + th,
            'centre': pposn[1] + 0.5*(pposn[3]-pposn[1]-dims[1]),
            'bottom': pposn[3] - th - dims[1]
            }

        # these are special places where the key is aligned
        self.highlightpoints = {}
        for xname, xval in citems(xposn):
            for yname, yval in citems(yposn):
                self.highlightpoints[(xname, yname)] = qt4.QPointF(xval, yval)

        self.updatePen()

    def checkHighlight(self):
        """Check to see whether box is over hightlight area.
        Returns (x, y) name or None if not."""

        rect = self.rect()
        rect.translate(self.pos())

        highlight = None
        highlightrect = qt4.QRectF(rect.left()-10, rect.top()-10, 20, 20)
        for name, point in citems(self.highlightpoints):
            if highlightrect.contains(point):
                highlight = name
                break
        return highlight

    def updatePen(self):
        """Update color of rectangle if it is over a hightlight area."""
        if self.checkHighlight():
            self.setPen(self.highlightpen)
        else:
            self.setPen(controlgraph.controlLinePen())

    def mouseMoveEvent(self, event):
        """Set correct pen for box."""
        qt4.QGraphicsRectItem.mouseMoveEvent(self, event)
        self.updatePen()

    def mouseReleaseEvent(self, event):
        """Update widget with position."""
        qt4.QGraphicsRectItem.mouseReleaseEvent(self, event)
        highlight = self.checkHighlight()
        if highlight:
            # in a highlight zone so use highlight zone name to set position
            hp, vp = highlight
            hm, vm = 0., 0.
        else:
            # calculate the position of the box to work out Manual fractions
            rect = self.rect()
            rect.translate(self.pos())
            pposn = self.params.parentposn

            hp, vp = 'manual', 'manual'
            hm = (rect.left() - pposn[0]) / (pposn[2] - pposn[0])
            vm = (pposn[3] - rect.bottom()) / (pposn[3] - pposn[1])

        # update widget with positions
        s = self.params.widget.settings
        operations = (
            document.OperationSettingSet(s.get('horzPosn'), hp),
            document.OperationSettingSet(s.get('vertPosn'), vp),
            document.OperationSettingSet(s.get('horzManual'), hm),
            document.OperationSettingSet(s.get('vertManual'), vm),
            )
        self.params.widget.document.applyOperation(
            document.OperationMultiple(operations, descr=_('move key')))

############################################################################

class Key(widget.Widget):
    """Key on graph."""

    typename = 'key'
    description = _('Plot key')
    allowusercreation = True

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        s.add( setting.Text('Text',
                            descr = _('Text settings'),
                            usertext=_('Text')),
               pixmap = 'settings_axislabel' )
        s.add( setting.KeyBrush('Background',
                                descr = _('Key background fill'),
                                usertext=_('Background')),
               pixmap = 'settings_bgfill' )
        s.add( setting.Line('Border',
                            descr = _('Key border line'),
                            usertext=_('Border')),
               pixmap = 'settings_border' )

        s.add( setting.Str('title', '',
                           descr=_('Key title text'),
                           usertext=_('Title')) )

        s.add( setting.AlignHorzWManual( 'horzPosn',
                                         'right',
                                         descr = _('Horizontal key position'),
                                         usertext=_('Horz posn'),
                                         formatting=True) )
        s.add( setting.AlignVertWManual( 'vertPosn',
                                         'bottom',
                                         descr = _('Vertical key position'),
                                         usertext=_('Vert posn'),
                                         formatting=True) )
                               
        s.add( setting.Distance('keyLength',
                                '1cm',
                                descr = _('Length of line to show in sample'),
                                usertext=_('Key length'),
                                formatting=True) )
        
        s.add( setting.AlignVert( 'keyAlign',
                                  'top',
                                  descr = _('Alignment of key symbols relative to text'),
                                  usertext = _('Key alignment'),
                                  formatting = True) )

        s.add( setting.Float( 'horzManual',
                              0.,
                              descr = _('Manual horizontal fractional position'),
                              usertext=_('Horz manual'),
                              formatting=True) )
        s.add( setting.Float( 'vertManual',
                              0.,
                              descr = _('Manual vertical fractional position'),
                              usertext=_('Vert manual'),
                              formatting=True) )

        s.add( setting.Float( 'marginSize',
                              1.,
                              minval = 0.,
                              descr = _('Width of margin in characters'),
                              usertext=_('Margin size'),
                              formatting=True) )

        s.add( setting.Int( 'columns',
                            1,
                            descr = _('Number of columns in key'),
                            usertext = _('Columns'),
                            minval = 1,
                            maxval = 100,
                            formatting = True) )

        s.add( setting.Bool(
            'symbolswap',
            False,
            descr=_('Put key symbol on right and text on left'),
            usertext=_('Swap symbol'),
            formatting=True) )

    @classmethod
    def allowedParentTypes(klass):
        from . import graph
        return (graph.Graph,)

    @staticmethod
    def _layoutChunk(entries, start, dims):
        """Layout the entries into the given box, starting at start"""
        row, col = start
        numrows, numcols = dims
        colstats = [0] * numcols
        layout = []
        for (plotter, num, lines) in entries:
            if row+lines > numrows:
                # this item doesn't fit in this column, so move to the next
                col += 1
                row = 0
            if col >= numcols:
                # this layout failed, suggest expanding the box by 1 row
                return ([], [], numrows+1)
            if lines > numrows:
                # this layout failed, suggest expanding the box to |lines|
                return ([], [], lines)
            
            # col -> yp, row -> xp
            layout.append( (plotter, num, col, row, lines) )
            row += lines
            colstats[col] += 1
        
        return (layout, colstats, numrows)
    
    def _layout(self, entries, totallines):
        """Layout the items, trying to keep the box as small as possible
        while still filling the columns"""
        
        maxcols = self.settings.columns
        numcols = min(maxcols, max(len(entries), 1))
        
        if not entries:
            return (list(), (0, 0))
        
        # start with evenly-sized rows and expand to fit
        numrows = totallines // numcols
        layout = []
        
        while not layout:
            # try to do a first cut of the layout, and expand the box until
            # everything fits
            (layout, colstats, newrows) = self._layoutChunk(entries, (0, 0), (numrows, numcols))
            if not layout:
                numrows = newrows
            
        # ok, we've got a layout where everything fits, now pull items right
        # to fill the remaining columns, if need be
        while colstats[-1] == 0:
            # shift 1 item to the right, up to the first column that has
            # excess items
            meanoccupation = max(1, sum(colstats)/numcols)
            
            # loop until we find a victim item which can be safely moved
            victimcol = numcols
            while True:
                # find the right-most column with excess occupation number
                for i in reversed(crange(victimcol)):
                    if colstats[i] > meanoccupation:
                        victimcol = i
                        break
                
                # find the last item in the victim column
                victim = 0
                for i in reversed(crange(len(layout))):
                    if layout[i][2] == victimcol:
                        victim = i
                        break
                
                # try to relayout with the victim item shoved to the next column
                (newlayout, newcolstats, newrows) = self._layoutChunk(entries[victim:],
                                                        (0, victimcol+1), (numrows, numcols))
                if newlayout:
                    # the relayout worked, so accept it
                    layout = layout[0:victim] + newlayout
                    colstats[victimcol] -= 1
                    del colstats[victimcol+1:]
                    colstats += newcolstats[victimcol+1:]
                    break
                
                # if we've run out of potential victims, just return what we have
                if victimcol == 0:
                    return (layout, (numrows, numcols))
        
        return (layout, (numrows, numcols))
    
    def draw(self, parentposn, phelper, outerbounds = None):
        """Plot the key on a plotter."""

        s = self.settings
        if s.hide:
            return

        painter = phelper.painter(self, parentposn)
        with painter:
            self._doDrawing(painter, phelper, parentposn)

    def _doDrawing(self, painter, phelper, parentposn):
        """Do the actual drawing."""

        s = self.settings
        font = s.get('Text').makeQFont(painter)
        painter.setFont(font)
        height = utils.FontMetrics(font, painter.device()).height()
        margin = s.marginSize * height

        showtext = not s.Text.hide

        # total number of layout lines required
        totallines = 0

        # reserve space for the title
        titlewidth, titleheight = 0, 0
        if s.title != '':
            titlefont = qt4.QFont(font)
            titlefont.setPointSize(max(font.pointSize() * 1.2, font.pointSize() + 2))
            titlewidth, titleheight = utils.Renderer(
                painter, titlefont,
                0, 0, s.title,
                doc=self.document).getDimensions()
            titleheight += 0.5*margin

        # maximum width of text required
        maxwidth = 1

        entries = []
        # iterate over children and find widgets which are suitable
        for c in self.parent.children:
            try:
                num = c.getNumberKeys()
            except AttributeError:
                continue
            if not c.settings.hide:
                # add an entry for each key entry for each widget
                for i in crange(num):
                    lines = 1
                    if showtext:
                        w, h = utils.Renderer(
                            painter, font, 0, 0,
                            c.getKeyText(i),
                            doc=self.document).getDimensions()
                        maxwidth = max(maxwidth, w)
                        lines = max(1, math.ceil(h/height))

                    totallines += lines
                    entries.append( (c, i, lines) )

        # layout the box
        layout, (numrows, numcols) = self._layout(entries, totallines)

        # width of key part of key
        symbolwidth = s.get('keyLength').convert(painter)
        keyswidth = ( (maxwidth + height + symbolwidth)*numcols +
                      height*(numcols-1) )
        # total width of box
        totalwidth = max(keyswidth, titlewidth)

        totalheight = numrows * height + titleheight
        if not s.Border.hide:
            totalwidth += 2*margin
            totalheight += margin

        # work out horizontal position
        h = s.horzPosn
        if h == 'left':
            x = parentposn[0] + height
        elif h == 'right':
            x = parentposn[2] - height - totalwidth
        elif h == 'centre':
            x = ( parentposn[0] +
                  0.5*(parentposn[2] - parentposn[0] - totalwidth) )
        elif h == 'manual':
            x = parentposn[0] + (parentposn[2]-parentposn[0])*s.horzManual

        # work out vertical position
        v = s.vertPosn
        if v == 'top':
            y = parentposn[1] + height
        elif v == 'bottom':
            y = parentposn[3] - totalheight - height
        elif v == 'centre':
            y = ( parentposn[1] +
                  0.5*(parentposn[3] - parentposn[1] - totalheight) )
        elif v == 'manual':
            y = ( parentposn[3] -
                  (parentposn[3]-parentposn[1])*s.vertManual - totalheight )

        # for controlgraph
        boxposn = (x, y)
        boxdims = (totalwidth, totalheight)

        # draw surrounding box
        boxpath = qt4.QPainterPath()
        boxpath.addRect(qt4.QRectF(x, y, totalwidth, totalheight))
        if not s.Background.hide:
            utils.brushExtFillPath(painter, s.Background, boxpath)
        if not s.Border.hide:
            painter.strokePath(boxpath, s.get('Border').makeQPen(painter) )
            y += margin*0.5

        # center and draw the title
        if s.title:
            xpos = x + (totalwidth-titlewidth)/2
            utils.Renderer(
                painter, titlefont, xpos, y, s.title,
                alignvert=1, doc=self.document).render()
            y += titleheight

        # centres key below title
        x += (totalwidth-keyswidth)/2

        textpen = s.get('Text').makeQPen(painter)

        swap = s.symbolswap

        # plot dataset entries
        for (plotter, num, xp, yp, lines) in layout:
            xpos = x + xp*(maxwidth+2*height+symbolwidth)
            ypos = y + yp*height

            # plot key symbol
            painter.save()
            keyoffset = 0
            if s.keyAlign == 'centre':
                keyoffset = (lines-1)*height/2.0
            elif s.keyAlign == 'bottom':
                keyoffset = (lines-1)*height

            sx = xpos
            if swap:
                sx += maxwidth + height
            plotter.drawKeySymbol(num, painter, sx, ypos+keyoffset,
                                  symbolwidth, height)
            painter.restore()

            # write key text
            if showtext:
                painter.setPen(textpen)
                if swap:
                    lx = xpos + maxwidth
                    alignx = 1
                else:
                    lx = xpos + height + symbolwidth
                    alignx = -1

                utils.Renderer(
                    painter, font,
                    lx, ypos,
                    plotter.getKeyText(num),
                    alignx, 1,
                    doc=self.document).render()

        phelper.setControlGraph(
            self, [ControlKey(self, parentposn, boxposn, boxdims, height)] )

document.thefactory.register( Key )
