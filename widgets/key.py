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
                               
        s.add( setting.Distance('keyLength', '1cm',
                                descr = 'Length of line to show in sample') )

        s.add( setting.Float( 'horzManual',
                              0.,
                              descr = 'Manual horizontal fractional position'))
        s.add( setting.Float( 'vertManual',
                              0.,
                              descr = 'Manual vertical fractional position') )

        if type(self) == Key:
            self.readDefaults()

    def draw(self, parentposn, painter, outerbounds = None):
        """Plot the key on a plotter."""

        painter.beginPaintingWidget(self, parentposn)
        painter.save()

        s = self.settings
        font = s.get('Text').makeQFont(painter)
        painter.setFont(font)
        height = painter.fontMetrics().height()

        showtext = not s.Text.hide

        # count number of keys to draw
        number = 0
        maxwidth = 1
        for c in self.parent.children:
            if c.settings.isSetting('key') and c.settings.key != '':
                number += 1
                if showtext:
                    w, h = utils.Renderer(painter, font, 0, 0,
                                          c.settings.key).getDimensions()
                    maxwidth = max(maxwidth, w)

        # total size of box
        symbolwidth = s.get('keyLength').convert(painter)
        totalwidth = maxwidth + height + symbolwidth
        totalheight = (number+1)*height
        if not s.Border.hide:
            totalwidth += height*2

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
            y = parentposn[1]
            if not s.Border.hide:
                y += height
        elif v == 'bottom':
            y = parentposn[3] - totalheight
            if not s.Border.hide:
                y -= height
        elif v == 'centre':
            y = (parentposn[1] +
                 (parentposn[3] - parentposn[1])/2 - totalheight/2)
        elif v == 'manual':
            y = int( parentposn[3] -
                     (parentposn[3]-parentposn[1])*s.vertManual - totalheight)

        # position of text in x
        symbxpos = x

        # draw surrounding box
        if not s.Background.hide:
            brush = s.get('Background').makeQBrush()
            painter.fillRect(x, y, totalwidth, totalheight, brush)
        if not s.Border.hide:
            painter.setPen( s.get('Border').makeQPen(painter) )
            painter.drawRect(x, y, totalwidth, totalheight)
            symbxpos += height

        textpen = s.get('Text').makeQPen()

        # plot dataset entries
        ypos = y + height/2
        for c in self.parent.children:
            if c.settings.isSetting('key') and c.settings.key != '':
                # plot key symbol
                painter.save()
                c.drawKeySymbol(painter, symbxpos, ypos,
                                symbolwidth, height)
                painter.restore()
                
                # write key text
                if showtext:
                    painter.setPen(textpen)
                    utils.Renderer(painter, font,
                                   symbxpos + height + symbolwidth,
                                   ypos + height/2, c.settings.key,
                                   -1, 0).render()

                ypos += height

        painter.restore()
        painter.endPaintingWidget()

widgetfactory.thefactory.register( Key )
