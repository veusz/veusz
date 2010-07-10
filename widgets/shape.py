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

# $Id$

"""For plotting shapes."""

import itertools
import os

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.document as document
import veusz.utils as utils

import controlgraph
import plotters

class Shape(plotters.FreePlotter):
    """A shape on a page/graph."""

    def __init__(self, parent, name=None):
        plotters.FreePlotter.__init__(self, parent, name=name)

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.FreePlotter.addSettings(s)

        s.add( setting.ShapeFill('Fill',
                                 descr = 'Shape fill',
                                 usertext='Fill'),
               pixmap = 'settings_bgfill' )
        s.add( setting.Line('Border',
                            descr = 'Shape border',
                            usertext='Border'),
               pixmap = 'settings_border' )

class BoxShape(Shape):
    """For drawing box-like shapes."""

    def __init__(self, parent, name=None):
        Shape.__init__(self, parent, name=name)

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        Shape.addSettings(s)

        s.add( setting.DatasetOrFloatList('width', [0.1],
                                          descr='List of fractional '
                                          'widths or dataset',
                                          usertext='Widths',
                                          formatting=False), 3 )
        s.add( setting.DatasetOrFloatList('height', [0.1],
                                          descr='List of fractional '
                                          'heights or dataset',
                                          usertext='Heights',
                                          formatting=False), 4 )
        s.add( setting.DatasetOrFloatList('rotate', [0.],
                                          descr='Rotation angles of '
                                          'shape or dataset',
                                          usertext='Rotate',
                                          formatting=False), 5 )


    def drawShape(self, painter, rect):
        pass

    def draw(self, posn, painter, outerbounds = None):
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
        self.controlgraphitems = []

        painter.beginPaintingWidget(self, posn)
        painter.save()

        # drawing settings for shape
        if not s.Border.hide:
            painter.setPen( s.get('Border').makeQPen(painter) )
        else:
            painter.setPen( qt4.QPen(qt4.Qt.NoPen) )
        if not s.Fill.hide:
            painter.setBrush( s.get('Fill').makeQBrush() )
        else:
            painter.setBrush( qt4.QBrush() )

        # iterate over positions
        index = 0
        dx, dy = posn[2]-posn[0], posn[3]-posn[1]
        for x, y, w, h, r in itertools.izip(xpos, ypos,
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

            if isnotdataset:
                cgi = controlgraph.ControlResizableBox(
                    self, [x, y], [wp, hp], r, allowrotate=True)
                cgi.index = index
                cgi.widgetposn = posn
                index += 1
                self.controlgraphitems.append(cgi)

        painter.restore()
        painter.endPaintingWidget()

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
        xp, yp = list(s.xPos), list(s.yPos)
        w, h, r = list(s.width), list(s.height), list(s.rotate)
        xp[cgi.index] = xpos
        yp[cgi.index] = ypos
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
            document.OperationMultiple(operations, descr='adjust shape') )

class Rectangle(BoxShape):
    """Draw a rectangle, or rounded rectangle."""
    typename = 'rect'
    description = 'Rectangle'
    allowusercreation = True

    def __init__(self, parent, name=None):
        BoxShape.__init__(self, parent, name=name)
        if type(self) == Rectangle:
            self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        BoxShape.addSettings(s)

        s.add( setting.Int('rounding', 0,
                           minval=0, maxval=100,
                           descr='Round corners with this percentage',
                           usertext='Rounding corners',
                           formatting=True) )

    def drawShape(self, painter, rect):
        s = self.settings
        if s.rounding == 0:
            painter.drawRect(rect)
        else:
            painter.drawRoundRect(rect, s.rounding, s.rounding )

class Ellipse(BoxShape):
    """Draw an ellipse."""

    typename = 'ellipse'
    description = 'Ellipse'
    allowusercreation = True

    def __init__(self, parent, name=None):
        BoxShape.__init__(self, parent, name=name)
        if type(self) == Ellipse:
            self.readDefaults()

    def drawShape(self, painter, rect):
        painter.drawEllipse(rect)

class ImageFile(BoxShape):
    """Draw an image."""

    typename = 'imagefile'
    description = 'Image file'
    allowusercreation = True

    def __init__(self, parent, name=None):
        BoxShape.__init__(self, parent, name=name)
        if type(self) == ImageFile:
            self.readDefaults()

        self.cachepixmap = None
        self.cachefilename = None
        self.cachestat = None

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        BoxShape.addSettings(s)

        s.add( setting.ImageFilename('filename', '',
                                     descr='Image filename',
                                     usertext='Filename',
                                     formatting=False),
               posn=0 )
        s.add( setting.Bool('aspect', True,
                            descr='Preserve aspect ratio',
                            usertext='Preserve aspect',
                            formatting=True),
               posn=0 )
        s.Border.get('hide').newDefault(True)

    def updateCachedPixmap(self):
        """Update cache."""
        s = self.settings
        self.cachestat = os.stat(s.filename)
        self.cachepixmap = qt4.QPixmap(s.filename)
        self.cachefilename = s.filename
        return self.cachepixmap

    def drawShape(self, painter, rect):
        """Draw pixmap."""
        s = self.settings

        # draw border and fill
        painter.drawRect(rect)

        # cache pixmap
        pixmap = None
        if s.filename != '' and os.path.isfile(s.filename):
            if (self.cachefilename != s.filename or 
                os.stat(s.filename) != self.cachestat):
                self.updateCachedPixmap()
            pixmap = self.cachepixmap

        # if no pixmap, then use default image
        if not pixmap or pixmap.width() == 0 or pixmap.height() == 0:
            pixmap = utils.getIcon('button_imagefile').pixmap(64, 64)
        
        # pixmap rectangle
        prect = qt4.QRectF(pixmap.rect())

        # preserve aspect ratio
        if s.aspect:
            xr = rect.width() / prect.width()
            yr = rect.height() / prect.height()

            if xr > yr:
                rect = qt4.QRectF(rect.left()+(rect.width()-
                                               prect.width()*yr)*0.5,
                                  rect.top(),
                                  prect.width()*yr,
                                  rect.height())
            else:
                rect = qt4.QRectF(rect.left(),
                                  rect.top()+(rect.height()-
                                              prect.height()*xr)*0.5,
                                  rect.width(),
                                  prect.height()*xr)

        # finally draw pixmap
        painter.drawPixmap(rect, pixmap, prect)

document.thefactory.register( Ellipse )
document.thefactory.register( Rectangle )
document.thefactory.register( ImageFile )
