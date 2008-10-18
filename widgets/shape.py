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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
###############################################################################

# $Id$

"""For plotting shapes."""

import itertools
import os
import numpy as N

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.document as document
import veusz.utils as utils

import widget
import page
import graph
import controlgraph

class Shape(widget.Widget):
    """A shape on a page/graph."""

    allowedparenttypes = [graph.Graph, page.Page]

    def __init__(self, parent, name=None):
        widget.Widget.__init__(self, parent, name=name)
        s = self.settings

        s.add( setting.ShapeFill('Fill',
                                 descr = 'Shape fill',
                                 usertext='Fill'),
               pixmap = 'bgfill' )
        s.add( setting.Line('Border',
                            descr = 'Shape border',
                            usertext='Border'),
               pixmap = 'border' )

class BoxShape(Shape):
    """For drawing box-like shapes."""

    def __init__(self, parent, name=None):
        Shape.__init__(self, parent, name=name)

        s = self.settings

        s.add( setting.DatasetOrFloatList('xPos', 0.5,
                                          descr='List of fractional X '
                                          'coordinates or dataset',
                                          usertext='X position',
                                          formatting=False) )
        s.add( setting.DatasetOrFloatList('yPos', 0.5,
                                          descr='List of fractional Y '
                                          'coordinates or dataset',
                                          usertext='Y position',
                                          formatting=False) )
        s.add( setting.DatasetOrFloatList('width', 0.1,
                                          descr='List of fractional '
                                          'widths or dataset',
                                          usertext='Width',
                                          formatting=False) )
        s.add( setting.DatasetOrFloatList('height', 0.1,
                                          descr='List of fractional '
                                          'heights or dataset',
                                          usertext='Height',
                                          formatting=False) )
        s.add( setting.DatasetOrFloatList('rotate', 0.,
                                          descr='Rotation angle of '
                                          'shape heights or dataset',
                                          usertext='Rotate',
                                          formatting=False) )

        s.add( setting.Choice('positioning',
                              ['axes', 'relative'], 'relative',
                              descr='Use axes or fractional '
                              'position to place label',
                              usertext='Position mode',
                              formatting=False) )
        s.add( setting.Axis('xAxis', 'x', 'horizontal',
                            descr = 'Name of X-axis to use',
                            usertext='X axis') )
        s.add( setting.Axis('yAxis', 'y', 'vertical',
                            descr = 'Name of Y-axis to use',
                            usertext='Y axis') )

    def drawShape(self, painter, rect):
        pass

    def draw(self, posn, painter, outerbounds = None):
        """Plot the key on a plotter."""

        s = self.settings
        d = self.document
        if s.hide:
            return

        # get positions of shapes
        xpos = s.get('xPos').getFloatArray(d)
        ypos = s.get('yPos').getFloatArray(d)
        width = s.get('width').getFloatArray(d)
        height = s.get('height').getFloatArray(d)
        rotate = s.get('rotate').getFloatArray(d)

        if (xpos is None or ypos is None or width is None or height is None
            or rotate is None):
            return

        self.lastposn = posn

        # translate coordinates from axes or relative values
        if s.positioning == 'axes':
            if hasattr(self.parent, 'getAxes'):
                axes = self.parent.getAxes( (s.xAxis, s.yAxis) )
            else:
                return
            if None in axes:
                return
            xpos = axes[0].graphToPlotterCoords(posn, xpos)
            ypos = axes[1].graphToPlotterCoords(posn, ypos)
        else:
            xpos = posn[0] + (posn[2]-posn[0])*xpos
            ypos = posn[3] - (posn[3]-posn[1])*ypos

        # if a dataset is used, we can't use control items
        isnotdataset = ( not s.get('xPos').isDataset(d) and 
                         not s.get('yPos').isDataset(d) and
                         not s.get('width').isDataset(d) and
                         not s.get('height').isDataset(d) and
                         not s.get('rotate').isDataset(d) )
        del self.controlgraphitems[:]

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
                cgi = controlgraph.ControlGraphResizableBox(
                    self, [x, y], [wp, hp], r, allowrotate=True)
                cgi.index = index
                index += 1
                self.controlgraphitems.append(cgi)

        painter.restore()
        painter.endPaintingWidget()

    def updateControlItem(self, cgi):
        """If control item is moved or resized, this is called."""
        s = self.settings
        try:
            cgiindex = self.controlgraphitems.index(cgi)
        except ValueError:
            return

        # calculate new position coordinate for item
        if s.positioning == 'axes':
            if hasattr(self.parent, 'getAxes'):
                axes = self.parent.getAxes( (s.xAxis, s.yAxis) )
            else:
                return
            if None in axes:
                return
            
            xpos = axes[0].plotterToGraphCoords(self.lastposn,
                                                N.array(cgi.posn[0]))
            ypos = axes[1].plotterToGraphCoords(self.lastposn,
                                                N.array(cgi.posn[1]))
        else:
            xpos = ((cgi.posn[0] - self.lastposn[0]) /
                    (self.lastposn[2]-self.lastposn[0]))
            ypos = ((cgi.posn[1] - self.lastposn[3]) /
                    (self.lastposn[1]-self.lastposn[3]))

        xw = abs(cgi.dims[0] / (self.lastposn[2]-self.lastposn[0]))
        yw = abs(cgi.dims[1] / (self.lastposn[1]-self.lastposn[3]))

        # actually do the adjustment on the document
        xp, yp = list(s.xPos), list(s.yPos)
        w, h, r = list(s.width), list(s.height), list(s.rotate)
        xp[cgiindex] = xpos
        yp[cgiindex] = ypos
        w[min(cgiindex, len(w)-1)] = xw
        h[min(cgiindex, len(h)-1)] = yw
        r[min(cgiindex, len(r)-1)] = cgi.angle

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
        self.settings.add( setting.Int('rounding', 0,
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

    def drawShape(self, painter, rect):
        painter.drawEllipse(rect)

class ImageFile(BoxShape):
    """Draw an image."""

    typename = 'imagefile'
    description = 'Image file'
    allowusercreation = True

    def __init__(self, parent, name=None):
        BoxShape.__init__(self, parent, name=name)
        self.settings.add( setting.Filename('filename', '',
                                            descr='Image filename',
                                            usertext='Filename',
                                            formatting=False),
                           posn=0 )
        self.settings.add( setting.Bool('aspect', True,
                                        descr='Preserve aspect ratio',
                                        usertext='Preserve aspect',
                                        formatting=True),
                           posn=0 )
        self.settings.Border.get('hide').newDefault(True)

        self.cachepixmap = None
        self.cachefilename = None
        self.cachestat = None

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
        if s.filename != '' and os.path.exists(s.filename):
            if (self.cachefilename != s.filename or 
                os.stat(s.filename) != self.cachestat):
                self.updateCachedPixmap()
            pixmap = self.cachepixmap

        # if no pixmap, then use default image
        if not pixmap:
            pixmap = utils.getPixmap('button_imagefile.png')
        
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
