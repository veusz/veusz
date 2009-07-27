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

# $Id$

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import plotters
import widget
import graph
import controlgraph

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

    def createGraphicsItem(self):
        return _GraphControlKey(self)

class _GraphControlKey(qt4.QGraphicsRectItem):
    """The graphical rectangle which is dragged around to reposition
    the key."""

    def __init__(self, params):
        qt4.QGraphicsRectItem.__init__(self,
                                       params.posn[0], params.posn[1],
                                       params.dims[0], params.dims[1])
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
        for xname, xval in xposn.iteritems():
            for yname, yval in yposn.iteritems():
                self.highlightpoints[(xname, yname)] = qt4.QPointF(xval, yval)

        self.updatePen()

    def checkHighlight(self):
        """Check to see whether box is over hightlight area.
        Returns (x, y) name or None if not."""

        rect = self.rect()
        rect.translate(self.pos())

        highlight = None
        highlightrect = qt4.QRectF(rect.left()-10, rect.top()-10, 20, 20)
        for name, point in self.highlightpoints.iteritems():
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
            document.OperationMultiple(operations, descr='move key'))

############################################################################

class Key(widget.Widget):
    """Key on graph."""

    typename = 'key'
    description = "Plot key"
    allowedparenttypes = [graph.Graph]
    allowusercreation = True

    def __init__(self, parent, name=None):
        widget.Widget.__init__(self, parent, name=name)

        if type(self) == Key:
            self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        s.add( setting.Text('Text',
                            descr = 'Text settings',
                            usertext='Text'),
               pixmap = 'settings_axislabel' )
        s.add( setting.KeyBrush('Background',
                                descr = 'Key background fill',
                                usertext='Background'),
               pixmap = 'settings_bgfill' )
        s.add( setting.Line('Border',
                            descr = 'Key border line',
                            usertext='Border'),
               pixmap = 'settings_border' )

        s.add( setting.AlignHorzWManual( 'horzPosn',
                                         'right',
                                         descr = 'Horizontal key position',
                                         usertext='Horz posn',
                                         formatting=True) )
        s.add( setting.AlignVertWManual( 'vertPosn',
                                         'bottom',
                                         descr = 'Vertical key position',
                                         usertext='Vert posn',
                                         formatting=True) )
                               
        s.add( setting.Distance('keyLength',
                                '1cm',
                                descr = 'Length of line to show in sample',
                                usertext='Key length',
                                formatting=True) )

        s.add( setting.Float( 'horzManual',
                              0.,
                              descr = 'Manual horizontal fractional position',
                              usertext='Horz manual',
                              formatting=True) )
        s.add( setting.Float( 'vertManual',
                              0.,
                              descr = 'Manual vertical fractional position',
                              usertext='Vert manual',
                              formatting=True) )

        s.add( setting.Int( 'columns',
                            1,
                            descr = 'Number of columns in key',
                            usertext = 'Columns',
                            minval = 1,
                            maxval = 100,
                            formatting = True) )

    def draw(self, parentposn, painter, outerbounds = None):
        """Plot the key on a plotter."""

        s = self.settings
        if s.hide:
            return

        painter.beginPaintingWidget(self, parentposn)
        painter.save()

        font = s.get('Text').makeQFont(painter)
        painter.setFont(font)
        height = painter.fontMetrics().height()

        showtext = not s.Text.hide

        # keep track of widgets to place
        keywidgets = []
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
                for i in xrange(num):
                    entries.append( (c, i) )

                    if showtext:
                        w, h = utils.Renderer(painter, font, 0, 0,
                                              c.getKeyText(i)).getDimensions()
                        maxwidth = max(maxwidth, w)

        # get number of columns
        count = len(entries)
        numcols = min(s.columns, max(count, 1))
        numrows = count / numcols
        if count % numcols != 0:
            numrows += 1

        # total size of box
        symbolwidth = s.get('keyLength').convert(painter)
        totalwidth = ( (maxwidth + height + symbolwidth)*numcols +
                       height*(numcols-1) )
        totalheight = numrows * height
        if not s.Border.hide:
            totalwidth += 2*height
            totalheight += height

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
        if not s.Background.hide:
            brush = s.get('Background').makeQBrush()
            painter.fillRect( qt4.QRectF(x, y, totalwidth, totalheight), brush)
        if not s.Border.hide:
            painter.setPen( s.get('Border').makeQPen(painter) )
            painter.drawRect( qt4.QRectF(x, y, totalwidth, totalheight) )
            x += height
            y += height*0.5

        textpen = s.get('Text').makeQPen()

        # plot dataset entries
        for index, (plotter, num) in enumerate(entries):
            xp, yp = index / numrows, index % numrows
            xpos = x + xp*(maxwidth+2*height+symbolwidth)
            ypos = y + yp*height

            # plot key symbol
            painter.save()
            plotter.drawKeySymbol(num, painter, xpos, ypos,
                                  symbolwidth, height)
            painter.restore()

            # write key text
            if showtext:
                painter.setPen(textpen)
                utils.Renderer(painter, font,
                               xpos + height + symbolwidth, ypos,
                               plotter.getKeyText(num),
                               -1, 1).render()

        self.controlgraphitems = [
            ControlKey(self, parentposn, boxposn, boxdims, height)
            ]

        painter.restore()
        painter.endPaintingWidget()

document.thefactory.register( Key )
