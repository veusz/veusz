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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

# $Id$

"""A colorbar widget for the image widget. Should show the scale of
the image."""

import veusz.qtall as qt4
import numpy as N

import veusz.document as document
import veusz.setting as setting

import graph
import grid
import widget
import axis

class ColorBar(axis.Axis):
    """Color bar for showing scale of image.

    This naturally is descended from an axis
    """

    typename='colorbar'
    allowedparenttypes = [graph.Graph, grid.Grid]
    allowusercreation = True
    description = 'Image color bar'

    def __init__(self, parent, name=None):
        """Initialise object and create axes."""

        axis.Axis.__init__(self, parent, name=name)
        if type(self) == ColorBar:
            self.readDefaults()
        
    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        axis.Axis.addSettings(s)

        s.add( setting.Image('image', '',
                             descr = 'Corresponding image',
                             usertext = 'Image'), 0 )

        s.get('log').readonly = True
        s.get('datascale').readonly = True

        s.add( setting.AlignHorzWManual( 'horzPosn',
                                         'right',
                                         descr = 'Horizontal position',
                                         usertext='Horz posn',
                                         formatting=True) )
        s.add( setting.AlignVertWManual( 'vertPosn',
                                         'bottom',
                                         descr = 'Vertical position',
                                         usertext='Vert posn',
                                         formatting=True) )
        s.add( setting.DistanceOrAuto('width', 'Auto',
                                      descr = 'Width of colorbar',
                                      usertext='Width',
                                      formatting=True) )
        s.add( setting.DistanceOrAuto('height', 'Auto',
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
                            usertext='Border'),
               pixmap='settings_border')

    def chooseName(self):
        """Get name of widget."""

        # override axis naming of x and y
        return widget.Widget.chooseName(self)

    def draw(self, parentposn, painter, outerbounds = None):
        '''Update the margins before drawing.'''

        s = self.settings

        # exit if hidden
        if s.hide:
            return

        # get height of label font
        font = s.get('Label').makeQFont(painter)
        painter.setFont(font)
        fontheight = painter.fontMetrics().height()

        bounds = self.computeBounds(parentposn, painter)
        horz = s.direction == 'horizontal'

        # use above to estimate width and height if necessary
        w = s.get('width')
        if w.isAuto():
            if horz:
                totalwidth = bounds[2] - bounds[0] - 2*fontheight
            else:
                totalwidth = fontheight
        else:
            totalwidth = w.convert(painter)

        h = s.get('height')
        if h.isAuto():
            if horz:
                totalheight = fontheight
            else:
                totalheight = bounds[3] - bounds[1] - 2*fontheight
        else:
            totalheight = h.convert(painter)

        # work out horizontal position
        h = s.horzPosn
        if h == 'left':
            bounds[0] += fontheight
            bounds[2] = bounds[0] + totalwidth
        elif h == 'right':
            bounds[2] -= fontheight
            bounds[0] = bounds[2] - totalwidth
        elif h == 'centre':
            delta = (bounds[2]-bounds[0]-totalwidth)/2.
            bounds[0] += delta
            bounds[2] -= delta
        elif h == 'manual':
            bounds[0] += (bounds[2]-bounds[0])*s.horzManual
            bounds[2] = bounds[0] + totalwidth

        # work out vertical position
        v = s.vertPosn
        if v == 'top':
            bounds[1] += fontheight
            bounds[3] = bounds[1] + totalheight 
        elif v == 'bottom':
            bounds[3] -= fontheight
            bounds[1] = bounds[3] - totalheight 
        elif v == 'centre':
            delta = (bounds[3]-bounds[1]-totalheight)/2.
            bounds[1] += delta
            bounds[3] -= delta
        elif v == 'manual':
            bounds[1] += (bounds[3]-bounds[1])*s.vertManual
            bounds[3] = bounds[1] + totalheight

        # do no painting if hidden or no image
        imgwidget = s.get('image').findImage()
        if s.hide or not imgwidget:
            return bounds

        # update image if necessary with new settings
        (minval, maxval,
         axisscale, img) = imgwidget.makeColorbarImage(s.direction)
        self.setAutoRange([minval, maxval])

        s.get('log').setSilent(axisscale == 'log')            

        painter.beginPaintingWidget(self, bounds)

        # now draw image on axis...
        minpix, maxpix = self.graphToPlotterCoords( bounds,
                                                    N.array([minval, maxval]) )

        if s.direction == 'horizontal':
            c = [ minpix, bounds[1], maxpix, bounds[3] ]
        else:
            c = [ bounds[0], maxpix, bounds[2], minpix ]
        r = qt4.QRectF(c[0], c[1], c[2]-c[0], c[3]-c[1])
        painter.drawImage(r, img)

        # if there's a border
        if not s.Border.hide:
            painter.setPen( s.get('Border').makeQPen(painter) )
            painter.setBrush( qt4.QBrush() )
            painter.drawRect( qt4.QRectF(bounds[0], bounds[1],
                                         bounds[2]-bounds[0],
                                         bounds[3]-bounds[1]) )

        # actually draw axis
        # we have to force position to full, as otherwise computeBounds
        # will mess up range if called twice
        savedposition = self.position
        self.position = (0., 0., 1., 1.)
        axis.Axis.draw(self, bounds, painter)
        self.position = savedposition

        painter.endPaintingWidget()
            
# allow the factory to instantiate a colorbar
document.thefactory.register( ColorBar )
