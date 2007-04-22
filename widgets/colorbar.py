#    Copyright (C) 2007 Jeremy S. Sanders
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
##############################################################################

# $Id:$

"""A colorbar widget for the image widget. Should show the scale of
the image."""

import veusz.qtall as qt4
import numpy as N

import veusz.document as document
import veusz.setting as setting

import graph
import containers
import image
import axis
import widget
import axis

class ColorBar(axis.Axis):
    """Color bar for showing scale of image.

    This naturally is descended from an axis
    """

    typename='colorbar'
    allowedparenttypes = [graph.Graph, containers.Grid]
    allowusercreation = True
    description = 'Image color bar'

    def __init__(self, parent, name=None):
        """Initialise object and create axes."""

        axis.Axis.__init__(self, parent, name=name)
        s = self.settings

        s.add( setting.Image('image', '',
                             descr = 'Corresponding image',
                             usertext = 'Image'), 0 )

        s.get('log').readonly = True

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
        s.add( setting.Distance('width', '5cm',
                                descr = 'Width of colorbar',
                                usertext='Width',
                                formatting=True) )
        s.add( setting.Distance('height', '1cm',
                                descr = 'Height of colorbar',
                                usertext='Height',
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

        s.add( setting.Line('Border', descr = 'Colorbar border line',
                            usertext='Border'), pixmap='border')

        if type(self) == ColorBar:
            self.readDefaults()
        
        self._cachedrange = [None, None]

    def chooseName(self):
        """Get name of widget."""

        # override axis naming of x and y
        return widget.Widget.chooseName(self)

    def _autoLookupRange(self):
        """Get automatic minimum and maximum of axis."""

        # this is pretty hacky -
        # this variable is updated every draw
        return self._cachedrange

    def draw(self, parentposn, painter, outerbounds = None):
        '''Update the margins before drawing.'''

        s = self.settings

        totalwidth = s.get('width').convert(painter)
        totalheight = s.get('height').convert(painter)

        font = s.get('Label').makeQFont(painter)
        painter.setFont(font)
        fontheight = painter.fontMetrics().height()

        # work out horizontal position
        h = s.horzPosn
        pp = list(parentposn)
        if h == 'left':
            pp[0] += fontheight
            pp[2] = pp[0] + totalwidth
        elif h == 'right':
            pp[2] -= fontheight
            pp[0] = pp[2] - totalwidth
        elif h == 'centre':
            delta = (pp[2]-pp[0]-totalwidth)/2.
            pp[0] += delta
            pp[2] -= delta
        elif h == 'manual':
            pp[0] += (pp[2]-pp[0])*s.horzManual
            pp[2] = pp[0] + totalwidth

        # work out vertical position
        v = s.vertPosn
        if v == 'top':
            pp[1] += fontheight
            pp[3] = pp[1] + totalheight 
        elif v == 'bottom':
            pp[3] -= fontheight
            pp[1] = pp[3] - totalheight 
        elif v == 'centre':
            delta = (pp[3]-pp[1]-totalheight)/2.
            pp[1] += delta
            pp[3] -= delta
        elif v == 'manual':
            pp[1] += (pp[3]-pp[1])*s.vertManual
            pp[3] = pp[1] + totalheight

        bounds = self.computeBounds(pp, painter)

        # do no painting if hidden or no image
        imgwidget = s.get('image').findImage()
        if s.hide or not imgwidget:
            return bounds

        # update image if necessary with new settings

        minval, maxval, axisscale, img = \
                imgwidget.makeColorbarImage(s.direction)
        self._cachedrange = [minval, maxval]
        s.get('log').set(axisscale == 'log')            

        painter.beginPaintingWidget(self, bounds)

        # if there's a border
        if not s.Border.hide:
            painter.setPen( s.get('Border').makeQPen(painter) )
            painter.drawRect( bounds[0], bounds[1], bounds[2]-bounds[0],
                              bounds[3]-bounds[1] )

        # now draw image on axis...
        minpix, maxpix = self.graphToPlotterCoords( bounds,
                                                    N.array([minval, maxval]) )

        if s.direction == 'horizontal':
            c = [ minpix, bounds[1], maxpix, bounds[3] ]
        else:
            c = [ bounds[0], maxpix, bounds[2], minpix ]
        r = qt4.QRect(c[0], c[1], c[2]-c[0]+1, c[3]-c[1]+1)
        
        painter.drawImage(r, img)
        axis.Axis.draw(self, bounds, painter)

        painter.endPaintingWidget()
            
# allow the factory to instantiate a colorbar
document.thefactory.register( ColorBar )
