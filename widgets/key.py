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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
###############################################################################

# $Id$

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import plotters
import widget
import graph

class Key(widget.Widget):
    """Key on graph."""

    typename = 'key'
    description = "Plot key"
    allowedparenttypes = [graph.Graph]
    allowusercreation = True

    def __init__(self, parent, name=None):
        widget.Widget.__init__(self, parent, name=name)

        s = self.settings
        s.add( setting.Text('Text',
                            descr = 'Text settings',
                            usertext='Text'),
               pixmap = 'axislabel' )
        s.add( setting.KeyBrush('Background',
                                descr = 'Key background fill',
                                usertext='Background'),
               pixmap = 'bgfill' )
        s.add( setting.Line('Border',
                            descr = 'Key border line',
                            usertext='Border'),
               pixmap = 'border' )

        s.add( setting.Choice( 'horzPosn',
                               ('left', 'centre', 'right', 'manual'),
                               'right',
                               descr = 'Horizontal key position',
                               usertext='Horz posn',
                               formatting=True) )
        s.add( setting.Choice( 'vertPosn',
                               ('top', 'centre', 'bottom', 'manual'),
                               'bottom',
                               descr = 'Vertical key position',
                               usertext='Vert posn',
                               formatting=True) )
                               
        s.add( setting.Distance('keyLength', '1cm',
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

        s.add( setting.Int( 'columns', 1,
                            descr = 'Number of columns in key',
                            usertext = 'Columns',
                            minval = 1,
                            maxval = 100,
                            formatting = True) )

        if type(self) == Key:
            self.readDefaults()

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

        # iterate over children and find widgets which are suitable
        for c in self.parent.children:
            childset = c.settings
            if childset.isSetting('key') and childset.key and not childset.hide:
                keywidgets.append(c)
                if showtext:
                    w, h = utils.Renderer(painter, font, 0, 0,
                                          childset.key).getDimensions()
                    maxwidth = max(maxwidth, w)

        # get number of columns
        count = len(keywidgets)
        numcols = min(s.columns, count)
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
                  0.5*(parentposn[3] - parentposn[1]) - 0.5*totalheight)
        elif v == 'manual':
            y = ( parentposn[3] -
                  (parentposn[3]-parentposn[1])*s.vertManual - totalheight )

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
        for index, plotter in enumerate(keywidgets):
            xp, yp = index / numrows, index % numrows
            xpos = x + xp*(maxwidth+2*height+symbolwidth)
            ypos = y + yp*height

            # plot key symbol
            painter.save()
            plotter.drawKeySymbol(painter, xpos, ypos,
                                  symbolwidth, height)
            painter.restore()

            # write key text
            if showtext:
                painter.setPen(textpen)
                utils.Renderer(painter, font,
                               xpos + height + symbolwidth,
                               ypos,
                               plotter.settings.key,
                               -1, 1).render()

        painter.restore()
        painter.endPaintingWidget()

document.thefactory.register( Key )
