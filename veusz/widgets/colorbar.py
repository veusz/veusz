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

"""A colorbar widget for the image widget. Should show the scale of
the image."""

from __future__ import division
from .. import qtall as qt4
import numpy as N

from .. import document
from .. import setting
from .. import utils

from . import widget
from . import axis

def _(text, disambiguation=None, context='ColorBar'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class ColorBar(axis.Axis):
    """Color bar for showing scale of image.

    This naturally is descended from an axis
    """

    typename='colorbar'
    allowusercreation = True
    description = _('Image color bar')
    isaxis = False

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        axis.Axis.addSettings(s)

        s.add( setting.WidgetChoice('widgetName', '',
                                    descr=_('Corresponding widget'),
                                    widgettypes=('image', 'xy', 'nonorthpoint'),
                                    usertext = _('Widget')), 0 )

        s.get('log').readonly = True
        s.get('datascale').readonly = True

        s.add( setting.AlignHorzWManual( 'horzPosn',
                                         'right',
                                         descr = _('Horizontal position'),
                                         usertext=_('Horz posn'),
                                         formatting=True) )
        s.add( setting.AlignVertWManual( 'vertPosn',
                                         'bottom',
                                         descr = _('Vertical position'),
                                         usertext=_('Vert posn'),
                                         formatting=True) )
        s.add( setting.DistanceOrAuto('width', 'Auto',
                                      descr = _('Width of colorbar'),
                                      usertext=_('Width'),
                                      formatting=True) )
        s.add( setting.DistanceOrAuto('height', 'Auto',
                                      descr = _('Height of colorbar'),
                                      usertext=_('Height'),
                                      formatting=True) )
        
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

        s.add( setting.Line('Border', descr = _('Colorbar border line'),
                            usertext=_('Border')),
               pixmap='settings_border')

        s.add( setting.SettingBackwardCompat('image', 'widgetName', None) )

    @classmethod
    def allowedParentTypes(klass):
        from . import graph, grid, nonorthgraph
        return (graph.Graph, grid.Grid, nonorthgraph.NonOrthGraph)

    @property
    def userdescription(self):
        return _("widget='%s', label='%s'") % (
            self.settings.widgetName, self.settings.label)

    def chooseName(self):
        """Get name of widget."""

        # override axis naming of x and y
        return widget.Widget.chooseName(self)

    def _axisDraw(self, posn, parentposn, outerbounds, painter, phelper):
        """Do actual drawing."""

        s = self.settings

        # get height of label font
        bounds = self.computeBounds(parentposn, phelper)

        font = s.get('Label').makeQFont(phelper)
        painter.setFont(font)
        fontheight = utils.FontMetrics(font, painter.device()).height()

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

        # FIXME: this is ugly - update bounds in helper state
        phelper.states[(self,0)].bounds = bounds

        # do no painting if hidden or no image
        imgwidget = s.get('widgetName').findWidget()
        if s.hide:
            return bounds

        self.updateAxisLocation(bounds)

        # update image if necessary with new settings
        if imgwidget is not None:
            minval, maxval, axisscale, cmapname, trans, invert = \
                imgwidget.getColorbarParameters()

            cmap = self.document.evaluate.getColormap(cmapname, invert)

            img = utils.makeColorbarImage(
                minval, maxval, axisscale, cmap, trans,
                direction=s.direction)
        else:
            # couldn't find widget
            minval, maxval, axisscale = 0., 1., 'linear'
            img = None

        s.get('log').setSilent(axisscale == 'log')
        self.setAutoRange([minval, maxval])
        self.computePlottedRange(force=True)

        # now draw image on axis...
        minpix, maxpix = self.graphToPlotterCoords(
            bounds, N.array([minval, maxval]) )

        routside = qt4.QRectF(
            bounds[0], bounds[1],
            bounds[2]-bounds[0], bounds[3]-bounds[1] )

        # really draw the img
        if img is not None:
            # coordinates to draw image and to clip rectangle
            if s.direction == 'horizontal':
                c = [ minpix, bounds[1], maxpix, bounds[3] ]
                cl = [ self.coordParr1, bounds[1], self.coordParr2, bounds[3] ]
            else:
                c = [ bounds[0], maxpix, bounds[2], minpix ]
                cl = [ bounds[0], self.coordParr1, bounds[2], self.coordParr2 ]
            r = qt4.QRectF(c[0], c[1], c[2]-c[0], c[3]-c[1])
            rclip = qt4.QRectF(cl[0], cl[1], cl[2]-cl[0], cl[3]-cl[1])

            painter.save()
            painter.setClipRect(rclip & routside)
            painter.drawImage(r, img)
            painter.restore()

        # if there's a border
        if not s.Border.hide:
            painter.setPen( s.get('Border').makeQPen(painter) )
            painter.setBrush( qt4.QBrush() )
            painter.drawRect( routside )

        # actually draw axis
        axis.Axis._axisDraw(self, bounds, parentposn, None, painter,
                            phelper)

# allow the factory to instantiate a colorbar
document.thefactory.register( ColorBar )
