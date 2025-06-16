#    Copyright (C) 2005 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This file is part of Veusz.
#
#    Veusz is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    Veusz is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################

"""Image plotting from 2d datasets."""

import numpy as N

from .. import qtall as qt
from .. import setting
from .. import document
from .. import utils
from ..helpers import qtloops
from . import plotters

def _(text, disambiguation=None, context='Image'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

def cropLinearImageToBox(image, pltx, plty, posn):
    """Given a plotting range pltx[0]->pltx[1], plty[0]->plty[1] and
    plotting bounds posn, return an image which is cropped to posn.

    Returns:
     - updated pltx range
     - updated plty range
     - cropped image
    """

    x1, y1, x2, y2 = posn
    pltx1, pltx2 = pltx
    pltw = pltx2-pltx1
    plty2, plty1 = plty
    plth = plty2-plty1

    imw = image.width()
    imh = image.height()
    pixw = pltw / imw
    pixh = plth / imh
    cutr = [0, 0, imw-1, imh-1]

    # work out where image intercepts posn, and make sure image
    # fills at least that area

    # need to chop left
    if pltx1 < x1:
        d = int((x1-pltx1) / pixw)
        cutr[0] += d
        pltx[0] += d*pixw

    # need to chop right
    if pltx2 > x2:
        d = max(0, int((pltx2-x2) / pixw) - 1)
        cutr[2] -= d
        pltx[1] -= d*pixw

    # chop top
    if plty1 < y1:
        d = int((y1-plty1) / pixh)
        cutr[1] += d
        plty[1] += d*pixh

    # chop bottom
    if plty2 > y2:
        d = max(0, int((plty2-y2) / pixh) - 1)
        cutr[3] -= d
        plty[0] -= d*pixh

    # create chopped-down image
    newimage = image.copy(
        cutr[0], cutr[1], cutr[2]-cutr[0]+1, cutr[3]-cutr[1]+1)

    # return new image coordinates and image
    return pltx, plty, newimage

def cropGridImageToBox(image, gridx, gridy, posn):
    """Given an image, pixel coordinates and box, crop image to box."""

    def trimGrid(grid, p1, p2):
        """Trim grid to bounds given, returning index range."""

        if grid[0] < grid[-1]:
            # fwd order
            i1 = max(N.searchsorted(grid, p1, side='right')-1, 0)
            i2 = min(N.searchsorted(grid, p2, side='left'), len(grid)) + 1

        else:
            # reverse order of grid
            gridr = grid[::-1]

            i1 = max(len(grid)-N.searchsorted(gridr, p2, side='left')-1, 0)
            i2 = min(
                len(grid)-N.searchsorted(gridr, p1, side='right'), len(grid))+1

        return i1, i2

    def trimEdge(grid, minval, maxval):
        """Trim outer gridpoints to minval and maxval."""
        if grid[0] < grid[-1]:
            grid[0] = max(grid[0], minval)
            grid[-1] = min(grid[-1], maxval)
        else:
            grid[0] = min(grid[0], maxval)
            grid[-1] = max(grid[-1], minval)

    # see whether cropping necessary
    x1, x2 = trimGrid(gridx, posn[0], posn[2])
    y1, y2 = trimGrid(gridy, posn[1], posn[3])

    if x1 > 0 or y1 > 0 or x2 < len(gridx)-1 or y2 < len(gridy)-1:
        # do cropping
        image = image.copy(x1, len(gridy)-y2, x2-x1-1, y2-y1-1)
        gridx = N.array(gridx[x1:x2])
        gridy = N.array(gridy[y1:y2])

        # trim outer grid point to viewable range
        trimEdge(gridx, posn[0], posn[2])
        trimEdge(gridy, posn[1], posn[3])

    return gridx, gridy, image

class Image(plotters.GenericPlotter):
    """A class which plots an image on a graph with a specified
    coordinate system."""

    typename='image'
    allowusercreation=True
    description=_('Plot a 2d dataset as an image')

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.GenericPlotter.addSettings(s)

        s.add( setting.DatasetExtended(
            'data', '',
            dimensions=2,
            descr=_('Dataset to plot'),
            usertext=_('Dataset')), 0 )
        s.add( setting.FloatOrAuto(
            'min', 'Auto',
            descr=_('Minimum value of image scale'),
            usertext=_('Min. value')), 1 )
        s.add( setting.FloatOrAuto(
            'max', 'Auto',
            descr=_('Maximum value of image scale'),
            usertext=_('Max. value')), 2 )
        s.add( setting.Choice(
            'colorScaling',
            ['linear', 'sqrt', 'log', 'squared'],
            'linear',
            descr=_('Scaling to transform numbers to color'),
            usertext=_('Scaling')), 3 )

        s.add( setting.DatasetExtended(
            'transparencyData', '',
            dimensions=2,
            descr=_('Dataset to use for transparency (0 to 1)'),
            usertext=_('Trans. data')), 4 )

        s.add( setting.Choice(
            'mapping',
            ('pixels', 'bounds'),
            'pixels',
            descr=_('Map image using pixels or bound coordinates'),
            usertext=_('Mapping')), 5 )

        s.add( setting.Colormap(
            'colorMap',
            'grey',
            descr=_('Set of colors to plot data with'),
            usertext=_('Colormap'),
            formatting=True), 5 )
        s.add( setting.Bool(
            'colorInvert', False,
            descr=_('Invert color map'),
            usertext=_('Invert colormap'),
            formatting=True), 6 )
        s.add( setting.Int(
            'transparency', 0,
            descr=_('Transparency percentage'),
            usertext=_('Transparency'),
            minval=0,
            maxval=100,
            formatting=True), 7 )

        s.add( setting.Choice(
            'drawMode',
            ['default', 'resample-pixels', 'resample-smooth', 'rectangles'],
            'default',
            descr=_('Method for drawing output'),
            usertext=_('Draw Mode'),
            formatting=True ) )

        # translate smooth to drawMode
        s.add( setting.SettingBackwardCompat(
            'smooth',
            'drawMode',
            False,
            translatefn=lambda x: {
                True: 'resample-smooth',
                False: 'default'
            }[x],
            formatting=True,
        ) )

    @property
    def userdescription(self):
        """User friendly description."""
        s = self.settings
        out = []
        if s.data:
            out.append(s.data)
        out += [s.colorScaling, s.colorMap]
        return ', '.join(out)

    def getDataValueRange(self, data):
        """Update data range from data."""

        s = self.settings
        minval = s.min
        if minval == 'Auto':
            if data is not None and len(data.data) != 0:
                minval = N.nanmin(data.data)
            else:
                minval = 0.
        maxval = s.max
        if maxval == 'Auto':
            if data is not None and len(data.data) != 0:
                maxval = N.nanmax(data.data)
            else:
                maxval = minval + 1

        # this is used currently by colorbar objects
        return (minval, maxval)

    def affectsAxisRange(self):
        """Range information provided by widget."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def getRange(self, axis, depname, axrange):
        """Automatically determine the ranges of variable on the axes."""

        # this is copied from Image, probably should combine
        s = self.settings
        d = self.document

        # return if no data
        data = s.get('data').getData(d)
        if data is None or data.dimensions != 2:
            return

        xr, yr = data.getDataRanges()
        if depname == 'sx':
            axrange[0] = min( axrange[0], xr[0] )
            axrange[1] = max( axrange[1], xr[1] )
        elif depname == 'sy':
            axrange[0] = min( axrange[0], yr[0] )
            axrange[1] = max( axrange[1], yr[1] )

    def getColorbarParameters(self):
        """Return parameters for colorbar."""

        s = self.settings
        d = self.document
        data = s.get('data').getData(d)
        minval, maxval = self.getDataValueRange(data)

        return (
            minval, maxval,
            s.colorScaling,
            s.colorMap,
            s.transparency,
            s.colorInvert,
        )

    def drawNonlinearImage(self, painter, axes, posn, data, image):
        """Draw an image where the image data are non-linear, or the
        axes are non-linear."""

        drawmode = self.settings.drawMode
        # get pixel edges, converted to plotter coordinates
        xedgep, yedgep = data.getPixelEdges(
            scalefnx=lambda v: axes[0].dataToPlotterCoords(posn, v),
            scalefny=lambda v: axes[1].dataToPlotterCoords(posn, v))

        if drawmode == 'default' or drawmode == 'rectangles':
            # simply draw everything as boxes
            qtloops.plotNonlinearImageAsBoxes(
                painter, image,
                xedgep, yedgep)
        else:
            # map image to a linear QImage
            # crop any pixels completely outside posn
            xedgep, yedgep, image = cropGridImageToBox(
                image, xedgep, yedgep, posn)
            x0 = int(min(xedgep[0], xedgep[-1]))
            x1 = int(max(xedgep[0], xedgep[-1]))
            y0 = int(min(yedgep[0], yedgep[-1]))
            y1 = int(max(yedgep[0], yedgep[-1]))

            print(painter.device().logicalDpiY())
            if drawmode == 'resample-pixels':
                # resample image to a flat bitmap
                image = qtloops.resampleNonlinearImage(
                    image, x0, y0, x1, y1, xedgep, yedgep)

            elif drawmode == 'resample-smooth':
                # render smaller and scale up to smooth
                s = 4
                image = qtloops.resampleNonlinearImage(
                    image, x0//s, y0//s, x1//s, y1//s,
                    xedgep/s, yedgep/s)
                image = image.scaled(
                    x1-x0, y1-y0,
                    qt.Qt.AspectRatioMode.IgnoreAspectRatio,
                    qt.Qt.TransformationMode.SmoothTransformation
                )
            else:
                raise RuntimeError('Invalid draw mode')

            imgposn = qt.QRectF(x0, y0, x1-x0, y1-y0)
            painter.drawImage(imgposn, image)

        # Debug position of pixels
        # painter.setPen(qt.QPen(qt.QBrush(qt.QColor("black")), 0.125))
        # for x in range(len(xedgep)-1):
        #     for y in range(len(yedgep)-1):
        #         painter.drawRect(qt.QRectF(
        #             xedgep[x], yedgep[y],
        #             xedgep[x+1]-xedgep[x],
        #             yedgep[y+1]-yedgep[y]))

    def dataDraw(self, painter, axes, posn, clip):
        """Draw image."""

        s = self.settings
        d = self.document

        data = s.get('data').getData(d)
        if s.hide or data is None or data.dimensions != 2:
            return

        transimg = s.get('transparencyData').getData(d)
        if transimg is not None:
            transimg = transimg.data

        rangex, rangey = data.getDataRanges()
        pltrangex = axes[0].dataToPlotterCoords(posn, N.array(rangex))
        pltrangey = axes[1].dataToPlotterCoords(posn, N.array(rangey))

        # abort if coordinate range is too small
        if(abs(pltrangex[0]-pltrangex[1])<1e-2 or
           abs(pltrangey[0]-pltrangey[1])<1e-2):
            return

        # make QImage from data
        cmap = d.evaluate.getColormap(s.colorMap, s.colorInvert)
        datavaluerange = self.getDataValueRange(data)
        image = utils.applyColorMap(
            cmap,
            s.colorScaling,
            data.data,
            datavaluerange[0], datavaluerange[1],
            s.transparency, transimg=transimg,
        )

        drawmode = s.drawMode

        # if data are non linear, or axes are non linear in pixel
        # mode, switch to non linear drawing
        if not data.isLinearImage() or ((
                not axes[0].isLinear() or not axes[1].isLinear()) and
                s.mapping == 'pixels'):
            self.drawNonlinearImage(painter, axes, posn, data, image)
            return

        # linearly spaced grid

        # avoid drawing pixels outside of axis range
        if ( pltrangex[0]<posn[0] or pltrangex[1]>posn[2] or
             pltrangey[0]<posn[1] or pltrangey[1]>posn[3] ):
            # need to crop image
            pltrangex, pltrangey, image = cropLinearImageToBox(
                image, pltrangex, pltrangey, posn)

        # invert output drawing if axes go from positive->negative
        # we only translate the coordinate system if this is the case
        xw = pltrangex[1]-pltrangex[0]
        yw = pltrangey[0]-pltrangey[1]
        xscale = 1 if xw>0 else -1
        yscale = 1 if yw>0 else -1

        painter.save()

        xp = pltrangex[0]
        yp = pltrangey[1]
        if xscale != 1 or yscale != 1:
            painter.translate(xp, yp)
            xp = yp = 0
            painter.scale(xscale, yscale)

        imgposn = qt.QRectF(
            xp, yp, abs(pltrangex[0]-pltrangex[1]),
            abs(pltrangey[0]-pltrangey[1]))

        drawmode = s.drawMode
        if drawmode == 'rectangles' or (drawmode =='default' and (
                image.width()<30 or image.height()<30)):
            # draw low res images as rectangles
            qtloops.plotImageAsRects(painter, imgposn, image)

        else:
            # upscale if requested
            if drawmode == 'resample-pixels':
                image = image.scaled(
                    int(pltrangex[1]-pltrangex[0]),
                    int(pltrangey[0]-pltrangey[1]),
                    qt.Qt.AspectRatioMode.IgnoreAspectRatio, qt.Qt.TransformationMode.FastTransformation)
            elif drawmode == 'resample-smooth':
                image = image.scaled(
                    int(pltrangex[1]-pltrangex[0]),
                    int(pltrangey[0]-pltrangey[1]),
                    qt.Qt.AspectRatioMode.IgnoreAspectRatio, qt.Qt.TransformationMode.SmoothTransformation)

            painter.drawImage(imgposn, image)

        painter.restore()

# allow the factory to instantiate an image
document.thefactory.register(Image)
