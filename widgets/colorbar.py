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
                             usertext = 'Image') )

        s.get('log').readonly = True

        s.add( setting.Distance( 'leftMargin', '0.1cm', descr=
                                 'Distance from left of colorbar to '
                                 'edge of parent',
                                 usertext='Left margin',
                                 formatting=True) )
        s.add( setting.Distance( 'rightMargin', '0.1cm', descr=
                                 'Distance from right of colorbar to '
                                 'edge of parent',
                                 usertext='Right margin',
                                 formatting=True) )
        s.add( setting.Distance( 'topMargin', '1cm', descr=
                                 'Distance from top of colorbar to '
                                 'edge of parent',
                                 usertext='Top margin',
                                 formatting=True) )
        s.add( setting.Distance( 'bottomMargin', '1cm', descr=
                                 'Distance from bottom of colorbar'
                                 'to edge of parent',
                                 usertext='Bottom margin',
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

        margins = ( s.get('leftMargin').convert(painter),
                    s.get('topMargin').convert(painter),
                    s.get('rightMargin').convert(painter),
                    s.get('bottomMargin').convert(painter) )
        bounds = self.computeBounds(parentposn, painter, margins=margins)

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
