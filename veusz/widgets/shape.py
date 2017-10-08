#    Copyright (C) 2008 Jeremy S. Sanders
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

"""For plotting shapes."""

from __future__ import division, print_function
import codecs
import itertools
import os

from ..compat import czip, cbytes
from .. import qtall as qt4
from .. import setting
from .. import document
from .. import utils
from . import widget
from . import controlgraph
from . import plotters

def _(text, disambiguation=None, context='Shape'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class Shape(plotters.FreePlotter):
    """A shape on a page/graph."""

    def __init__(self, parent, name=None):
        plotters.FreePlotter.__init__(self, parent, name=name)

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.FreePlotter.addSettings(s)

        s.add( setting.ShapeFill('Fill',
                                 descr = _('Shape fill'),
                                 usertext=_('Fill')),
               pixmap = 'settings_bgfill' )
        s.add( setting.Line('Border',
                            descr = _('Shape border'),
                            usertext=_('Border')),
               pixmap = 'settings_border' )
        s.add( setting.Bool('clip', False,
                            descr=_('Clip shape to its container'),
                            usertext=_('Clip'),
                            formatting=True) )

class BoxShape(Shape):
    """For drawing box-like shapes."""

    def __init__(self, parent, name=None):
        Shape.__init__(self, parent, name=name)

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        Shape.addSettings(s)

        s.add( setting.DatasetExtended(
                'width', [0.1],
                descr=_('List of fractional widths, dataset or expression'),
                usertext=_('Widths'),
                formatting=False), 3 )
        s.add( setting.DatasetExtended(
                'height', [0.1],
                descr=_('List of fractional heights, dataset or expression'),
                usertext=_('Heights'),
                formatting=False), 4 )
        s.add( setting.DatasetExtended(
                'rotate', [0.],
                descr=_('Rotation angles of shape, dataset or expression'),
                usertext=_('Rotate'),
                formatting=False), 5 )

    def drawShape(self, painter, rect):
        pass

    def draw(self, posn, phelper, outerbounds = None):
        """Plot the key on a plotter."""

        s = self.settings
        d = self.document
        if s.hide:
            return

        # get positions of shapes
        width = s.get('width').getFloatArray(d)
        height = s.get('height').getFloatArray(d)
        rotate = s.get('rotate').getFloatArray(d)
        if width is None or height is None or rotate is None:
            return

        # translate coordinates from axes or relative values
        xpos, ypos = self._getPlotterCoords(posn)
        if xpos is None or ypos is None:
            # we can't calculate coordinates
            return

        # if a dataset is used, we can't use control items
        isnotdataset = ( not s.get('xPos').isDataset(d) and
                         not s.get('yPos').isDataset(d) and
                         not s.get('width').isDataset(d) and
                         not s.get('height').isDataset(d) and
                         not s.get('rotate').isDataset(d) )
        controlgraphitems = []

        clip = None
        if s.clip:
            clip = qt4.QRectF( qt4.QPointF(posn[0], posn[1]),
                               qt4.QPointF(posn[2], posn[3]) )
        painter = phelper.painter(self, posn, clip=clip)
        with painter:
            # drawing settings for shape
            if not s.Border.hide:
                painter.setPen( s.get('Border').makeQPen(painter) )
            else:
                painter.setPen( qt4.QPen(qt4.Qt.NoPen) )

            # iterate over positions
            index = 0
            dx, dy = posn[2]-posn[0], posn[3]-posn[1]
            x = y = w = h = r = None
            for x, y, w, h, r in czip(xpos, ypos,
                                      itertools.cycle(width),
                                      itertools.cycle(height),
                                      itertools.cycle(rotate)):
                wp, hp = dx*w, dy*h
                painter.save()
                painter.translate(x, y)
                if r != 0:
                    painter.rotate(r)
                self.drawShape(painter, qt4.QRectF(-wp*0.5, -hp*0.5, wp, hp))
                painter.restore()

        if x is not None and isnotdataset:
            cgi = controlgraph.ControlResizableBox(
                self, [x, y], [wp, hp], r, allowrotate=True)
            cgi.index = index
            cgi.widgetposn = posn
            index += 1
            controlgraphitems.append(cgi)

        phelper.setControlGraph(self, controlgraphitems)

    def updateControlItem(self, cgi):
        """If control item is moved or resized, this is called."""
        s = self.settings

        # calculate new position coordinate for item
        xpos, ypos = self._getGraphCoords(cgi.widgetposn,
                                          cgi.posn[0], cgi.posn[1])
        if xpos is None or ypos is None:
            return

        xw = abs(cgi.dims[0] / (cgi.widgetposn[2]-cgi.widgetposn[0]))
        yw = abs(cgi.dims[1] / (cgi.widgetposn[1]-cgi.widgetposn[3]))

        # actually do the adjustment on the document
        xp = list(s.get('xPos').getFloatArray(self.document))
        yp = list(s.get('yPos').getFloatArray(self.document))
        w = list(s.get('width').getFloatArray(self.document))
        h = list(s.get('height').getFloatArray(self.document))
        r = list(s.get('rotate').getFloatArray(self.document))

        xp[min(cgi.index, len(xp)-1)] = xpos
        yp[min(cgi.index, len(yp)-1)] = ypos
        w[min(cgi.index, len(w)-1)] = xw
        h[min(cgi.index, len(h)-1)] = yw
        r[min(cgi.index, len(r)-1)] = cgi.angle

        operations = (
            document.OperationSettingSet(s.get('xPos'), xp),
            document.OperationSettingSet(s.get('yPos'), yp),
            document.OperationSettingSet(s.get('width'), w),
            document.OperationSettingSet(s.get('height'), h),
            document.OperationSettingSet(s.get('rotate'), r)
            )
        self.document.applyOperation(
            document.OperationMultiple(operations, descr=_('adjust shape')) )

class Rectangle(BoxShape):
    """Draw a rectangle, or rounded rectangle."""
    typename = 'rect'
    description = _('Rectangle')
    allowusercreation = True

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        BoxShape.addSettings(s)

        s.add( setting.Int('rounding', 0,
                           minval=0, maxval=100,
                           descr=_('Round corners with this percentage'),
                           usertext=_('Rounding corners'),
                           formatting=True) )

    def drawShape(self, painter, rect):
        s = self.settings
        path = qt4.QPainterPath()
        if s.rounding == 0:
            path.addRect(rect)
        else:
            path.addRoundedRect(rect, s.rounding, s.rounding)

        utils.brushExtFillPath(painter, s.Fill, path, stroke=painter.pen())

class Ellipse(BoxShape):
    """Draw an ellipse."""

    typename = 'ellipse'
    description = _('Ellipse')
    allowusercreation = True

    def drawShape(self, painter, rect):
        s = self.settings
        path = qt4.QPainterPath()
        path.addEllipse(rect)
        utils.brushExtFillPath(painter, s.Fill, path, stroke=painter.pen())

class ImageFile(BoxShape):
    """Draw an image."""

    typename = 'imagefile'
    description = _('Image file')
    allowusercreation = True

    def __init__(self, parent, name=None):
        BoxShape.__init__(self, parent, name=name)

        self.cacheimage = None
        self.cachefilename = None
        self.cachestat = None
        self.cacheembeddata = None

        self.addAction( widget.Action('embed', self.actionEmbed,
                                      descr = _('Embed image in Veusz document '
                                                'to remove dependency on external file'),
                                      usertext = _('Embed image')) )

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        BoxShape.addSettings(s)

        s.add( setting.ImageFilename('filename', '',
                                     descr=_('Image filename'),
                                     usertext=_('Filename'),
                                     formatting=False),
               posn=0 )

        s.add( setting.Str('embeddedImageData', '',
                           descr=_('Embedded base 64-encoded image data, '
                                   'used if filename set to {embedded}'),
                           usertext=_('Embedded data'),
                           hidden=True) )

        s.add( setting.Bool('aspect', True,
                            descr=_('Preserve aspect ratio'),
                            usertext=_('Preserve aspect'),
                            formatting=True),
               posn=0 )
        s.Border.get('hide').newDefault(True)

    def actionEmbed(self):
        """Embed external image into veusz document."""

        s = self.settings

        if s.filename == '{embedded}':
            print("Data already embedded")
            return

        # get data from external file
        try:
            f = open(s.filename, 'rb')
            data = f.read()
            f.close()
        except EnvironmentError:
            print("Could not find file. Not embedding.")
            return

        # convert to base 64 to make it nicer in the saved file
        encoded = codecs.encode(data, 'base64').decode('ascii')

        # now put embedded data in hidden setting
        ops = [
            document.OperationSettingSet(s.get('filename'), '{embedded}'),
            document.OperationSettingSet(s.get('embeddedImageData'),
                                         encoded)
            ]
        self.document.applyOperation(
            document.OperationMultiple(ops, descr=_('embed image')) )

    def updateCachedImage(self):
        """Update cache."""
        s = self.settings
        self.cachestat = os.stat(s.filename)
        self.cacheimage = qt4.QImage(s.filename)
        self.cachefilename = s.filename

    def updateCachedEmbedded(self):
        """Update cached image from embedded data."""
        s = self.settings
        self.cacheimage = qt4.QImage()

        # convert the embedded data from base64 and load into the image
        decoded = s.embeddedImageData.encode('ascii').decode('base64')
        self.cacheimage.loadFromData(decoded)

        # we cache the data we have decoded
        self.cacheembeddata = s.embeddedImageData

    def drawShape(self, painter, rect):
        """Draw image."""
        s = self.settings

        # draw border and fill
        painter.drawRect(rect)

        # check to see whether image needs reloading
        image = None
        if s.filename != '' and os.path.isfile(s.filename):
            if (self.cachefilename != s.filename or
                os.stat(s.filename) != self.cachestat):
                # update the image cache
                self.updateCachedImage()
                # clear any embedded image data
                self.settings.get('embeddedImageData').set('')
            image = self.cacheimage

        # or needs recreating from embedded data
        if s.filename == '{embedded}':
            if s.embeddedImageData is not self.cacheembeddata:
                self.updateCachedEmbedded()
            image = self.cacheimage

        # if no image, then use default image
        if ( not image or image.isNull() or
             image.width() == 0 or image.height() == 0 ):
            # load replacement image
            fname = os.path.join(utils.imagedir, 'button_imagefile.svg')
            r = qt4.QSvgRenderer(fname)
            r.render(painter, rect)

        else:
            # image rectangle
            irect = qt4.QRectF(image.rect())

            # preserve aspect ratio
            if s.aspect:
                xr = rect.width() / irect.width()
                yr = rect.height() / irect.height()

                if xr > yr:
                    rect = qt4.QRectF(
                        rect.left()+(rect.width()-irect.width()*yr)*0.5,
                        rect.top(), irect.width()*yr, rect.height())
                else:
                    rect = qt4.QRectF(
                        rect.left(),
                        rect.top()+(rect.height()-irect.height()*xr)*0.5,
                        rect.width(), irect.height()*xr)

            # finally draw image
            painter.drawImage(rect, image, irect)

document.thefactory.register( Ellipse )
document.thefactory.register( Rectangle )
document.thefactory.register( ImageFile )
