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

import plotters
import widget
import graph
import widgetfactory
import setting
import utils

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
                            descr = 'Text settings') )
        s.add( setting.KeyBrush('Background',
                                descr = 'Key background fill') )
        s.add( setting.Line('Border',
                            descr = 'Key border line') )

        s.add( setting.Choice( 'horzPosn',
                               ('left', 'centre', 'right', 'manual'),
                               'right',
                               descr = 'Horizontal key position' ) )
        s.add( setting.Choice( 'vertPosn',
                               ('top', 'centre', 'bottom', 'manual'),
                               'bottom',
                               descr = 'Vertical key position' ) )
                               
        s.add( setting.Float( 'horzManual',
                              0.,
                              descr = 'Manual horizontal fractional position'))
        s.add( setting.Float( 'vertManual',
                              0.,
                              descr = 'Manual vertical fractional position') )
        s.readDefaults()

    def draw(self, parentposn, painter):
        """Plot the key on a plotter."""

        widget.Widget.draw(self, parentposn, painter)

        painter.save()

        s = self.settings
        font = s.get('Text').makeQFont(painter)
        painter.setFont(font)
        height = painter.fontMetrics().height()

        # count number of keys to draw
        number = 0
        maxwidth = 10
        maxsymbolwidth = 1
        for c in self.parent.children:
            try:
                c.drawKeySymbol
            except AttributeError:
                pass
            else:
                if c.settings.key != '':
                    number += 1
                    w, h = utils.getTextDimensions(painter, font,
                                                   c.settings.key)
                    maxwidth = max(maxwidth, w)
                    maxsymbolwidth = max(c.getKeySymbolWidth(height),
                                         maxsymbolwidth)

        # total size of box
        totalwidth = maxwidth + 3*height + maxsymbolwidth
        totalheight = (number+1)*height

        # work out horizontal position
        h = s.horzPosn
        if h == 'left':
            x = parentposn[0] + height
        elif h == 'right':
            x = parentposn[2] - height - totalwidth
        elif h == 'centre':
            x = (parentposn[0] +
                 (parentposn[2] - parentposn[0])/2 - totalwidth/2)
        elif h == 'manual':
            x = int( parentposn[0] +
                     (parentposn[2]-parentposn[0])*s.horzManual )

        # work out vertical position
        v = s.vertPosn
        if v == 'top':
            y = parentposn[1] + height
        elif v == 'bottom':
            y = parentposn[3] - height - totalheight
        elif v == 'centre':
            y = (parentposn[1] +
                 (parentposn[3] - parentposn[1])/2 - totalheight/2)
        elif v == 'manual':
            y = int( parentposn[3] -
                     (parentposn[3]-parentposn[1])*s.vertManual - totalheight)

        # draw surrounding box
        if not s.Background.hide:
            brush = s.get('Background').makeQBrush()
            painter.fillRect(x, y, totalwidth, totalheight, brush)
        if not s.Border.hide:
            painter.setPen( s.get('Border').makeQPen(painter) )
            painter.drawRect(x, y, totalwidth, totalheight)

        textpen = s.get('Text').makeQPen()

        # plot dataset entries
        ypos = y + height/2
        for c in self.parent.children:
            try:
                c.drawKeySymbol
            except AttributeError:
                pass
            else:
                if c.settings.key != '':
                    # plot key symbol
                    c.drawKeySymbol(painter, x+height, ypos,
                                    maxsymbolwidth, height)

                    # write key text
                    if not s.Text.hide:
                        painter.setPen(textpen)
                        utils.render(painter,
                                     font, x + height*2 + maxsymbolwidth,
                                     ypos+height/2,
                                     c.settings.key, -1, 0)
                    ypos += height

        painter.restore()

widgetfactory.thefactory.register( Key )
