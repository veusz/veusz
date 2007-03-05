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

# $Id: graph.py 556 2007-01-11 17:31:27Z jeremysanders $

"""A colorbar widget for the image widget. Should show the scale of
the image."""

import veusz.document as document
import veusz.setting as setting

import graph
import containers
import image
import axis
import widget

class ColorBar(widget.Widget):
    """Color bar for showing scale of image."""

    typename='colorbar'
    allowedparenttypes = [graph.Graph, containers.Grid]
    allowusercreation = True
    description = 'Image color bar'

    def __init__(self, parent, name=None):
        """Initialise object and create axes."""

        widget.Widget.__init__(self, parent, name=name)
        s = self.settings

        s.add( setting.Text('Text',
                            descr = 'Text settings',
                            usertext='Text'),
               pixmap = 'axislabel' )
        
        s.add( setting.Image('image', '',
                             descr = 'Corresponding image',
                             usertext = 'Image') )

        s.add( setting.Choice('direction',
                              ['horizontal', 'vertical'],
                              'horizontal',
                              descr = 'Direction of colorbar',
                              usertext='Direction') )

        s.add( setting.Distance( 'leftMargin', '1.7cm', descr=
                                 'Distance from left of graph to '
                                 'edge of page',
                                 usertext='Left margin',
                                 formatting=True) )
        s.add( setting.Distance( 'rightMargin', '0.1cm', descr=
                                 'Distance from right of graph to '
                                 'edge of page',
                                 usertext='Right margin',
                                 formatting=True) )
        s.add( setting.Distance( 'topMargin', '0.1cm', descr=
                                 'Distance from top of graph to '
                                 'edge of page',
                                 usertext='Top margin',
                                 formatting=True) )
        s.add( setting.Distance( 'bottomMargin', '1.7cm', descr=
                                 'Distance from bottom of graph'
                                 'to edge of page',
                                 usertext='Bottom margin',
                                 formatting=True) )

        # the colorbar contains a hidden axis to plot the numbers
        #self.hiddenaxis = axis.Axis(None, 'hidden')

        # scale containing (min, max, scaling)
        self.cacheScale = (None, None, None)

    def draw(self, parentposn, painter, outerbounds = None):
        '''Update the margins before drawing.'''

        s = self.settings

        margins = ( s.get('leftMargin').convert(painter),
                    s.get('topMargin').convert(painter),
                    s.get('rightMargin').convert(painter),
                    s.get('bottomMargin').convert(painter) )
        bounds = self.computeBounds(parentposn, painter, margins=margins)

        # do no painting if hidden
        if s.hide:
            return bounds

        painter.beginPaintingWidget(self, bounds)

        img = s.get('image').findImage()
        print img

        painter.endPaintingWidget()
            
# allow the factory to instantiate a colorbar
document.thefactory.register( ColorBar )
