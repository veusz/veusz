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

"""Image plotting from 2d datasets."""

import string
import os.path

import veusz.qtall as qt4
import numarray as N

import veusz.setting as setting
import veusz.document as document

import plotters

def applyScaling(data, mode, minval, maxval):
    """Apply a scaling transformation on the data.

    mode is one of 'linear', 'sqrt', 'log', or 'squared'
    minval is the minimum value of the scale
    maxval is the maximum value of the scale

    returns transformed (data, minval, maxval)
    """

    if mode == 'linear':
        pass

    elif mode == 'sqrt':
        minval = max(0., minval)
        maxval = max(0., maxval)

        # replace illegal values
        belowzero = data < 0.
        data = data.copy()
        data[belowzero] = minval

        # calculate transform
        data = N.sqrt(data)
        minval = N.sqrt(minval)
        maxval = N.sqrt(maxval)

    elif mode == 'log':
        minval = max(1e-200, minval)
        maxval = max(1e-200, maxval)

        # replace illegal values
        bad = data < 1e-200
        data = data.copy()
        data[bad] = minval

        # transform
        data = N.log(data)
        minval = N.log(minval)
        maxval = N.log(maxval)

    elif mode == 'squared':
        # very simple to do this...
        minval = minval**2
        maxval = maxval**2

        data = data**2

    else:
        raise RuntimeError, 'Invalid scaling mode "%s"' % mode

    if minval > maxval:
        minval, maxval = maxval, minval
    if minval == maxval:
        minval = 0.
        maxval = 1.

    return (data, minval, maxval)

class Image(plotters.GenericPlotter):
    """A class which plots an image on a graph with a specified
    coordinate system."""

    # a dict of colormaps loaded in from external file
    colormaps = None

    typename='image'
    allowusercreation=True
    description='Plot a 2d dataset as an image'

    def __init__(self, parent, name=None):
        """Initialise plotter with axes."""

        plotters.GenericPlotter.__init__(self, parent, name=name)

        # lazy read of colormap file (Let's help startup times)
        if Image.colormaps == None:
            Image.readColorMaps()

        s = self.settings
        s.add( setting.Dataset('data', '',
                               dimensions = 2,
                               descr = 'Dataset to plot' ),
               0 )
        s.add( setting.FloatOrAuto('min', 'Auto',
                                   descr = 'Minimum value of image scale'),
               1 )
        s.add( setting.FloatOrAuto('max', 'Auto',
                                   descr = 'Maximum value of image scale'),
               2 )
        s.add( setting.Choice('colorScaling',
                              ['linear', 'sqrt', 'log', 'squared'],
                              'linear',
                              descr = 'Scaling to transform numbers to color'),
               3 )

        s.add( setting.Choice('colorMap', Image.colormapnames,
                              'grey',
                              descr = 'Set of colors to plot data with'),
               4 )
        s.add( setting.Bool('colorInvert', False,
                            descr = 'Invert color map'),
               5 )

        self.lastcolormap = None
        self.lastdataset = None
        self.schangeset = -1

    def readColorMaps(cls):
        """Read color maps data file (a class method)

        File is made up of:
          comments (prefaced by # on separate line)
          colormapname
          list of colors with B G R alpha order from 0->255 on separate lines
          [colormapname ...]
        """

        name = ''
        vals = []
        cls.colormaps = {}

        # locate file holding colormap data
        dir = os.path.dirname( os.path.abspath(__file__) )
        filename = os.path.join(dir, 'data', 'colormaps.dat')

        # iterate over file
        for l in open(filename):
            p = l.split()
            if len(p) == 0 or p[0][0] == '#':
                # blank or commented line
                pass
            elif p[0][0] not in string.digits:
                # new colormap follows
                if name != '':
                    cls.colormaps[name] = N.array(vals).astype(N.UInt8)
                name = p[0]
                vals = []
            else:
                # add value to current colormap
                assert name != ''
                assert len(p) == 4
                vals.append( [int(i) for i in p] )

        # add on final colormap
        if name != '':
            cls.colormaps[name] = N.array(vals).astype(N.UInt8)

        # collect names and sort alphabetically
        names = cls.colormaps.keys()
        names.sort()
        cls.colormapnames = names

    readColorMaps = classmethod(readColorMaps)

    def applyColourMap(self, cmap, scaling, datain, minval, maxval):
        """Apply a colour map to the 2d data given.

        cmap is the color map (numarray of BGRalpha quads)
        scaling is scaling mode => 'linear', 'sqrt', 'log' or 'squared'
        data are the imaging data
        minval and maxval are the extremes of the data for the colormap
        Returns a QImage
        """

        # apply scaling of data
        data, minval, maxval = applyScaling(datain, scaling, minval, maxval)

        # calculate fraction between min and max of data
        fracs = (N.ravel(data)-minval) * (1./(maxval-minval))
        fracs = N.clip(fracs, 0., 1.)

        # Work out which is the minimum colour map. Assumes we have <255 bands.
        numbands = cmap.shape[0]-1
        bands = (fracs*numbands).astype(N.UInt8)
        bands = N.clip(bands, 0, numbands-1)

        # work out fractional difference of data from band to next band
        deltafracs = (fracs - bands * (1./numbands)) * numbands

        # need to make a 2-dimensional array to multiply against triplets
        deltafracs.shape = (deltafracs.shape[0], 1)

        # calculate BGRalpha quadruplets
        # this is a linear interpolation between the band and the next band
        quads = (deltafracs*cmap[bands+1] +
                 (1.-deltafracs)*cmap[bands]).astype(N.UInt8)

        # convert 32bit quads to a Qt QImage
        # FIXME: Does this assume C-style array layout??
        s = quads.tostring()
        img = qt4.QImage(s, data.shape[1], data.shape[0],
                         qt4.QImage.Format_RGB32)
        img = img.mirrored()

        # hack to ensure string isn't freed before QImage
        img.veusz_string = s

        # done!
        return img

    def updateImage(self):
        """Update the image with new contents."""

        s = self.settings
        d = self.document
        data = d.data[s.data]

        minval = s.min
        if minval == 'Auto':
            minval = data.data.min()
        maxval = s.max
        if maxval == 'Auto':
            maxval = data.data.max()

        cmap = self.colormaps[s.colorMap]
        if s.colorInvert:
            cmap = cmap[::-1]

        self.image = self.applyColourMap(cmap, s.colorScaling,
                                         data.data,
                                         minval, maxval)

    def autoAxis(self, name, bounds):
        """Automatically determine the ranges of variable on the axes."""

        s = self.settings
        d = self.document

        # return if no data
        if s.data not in d.data:
            return

        # return if the dataset isn't two dimensional
        data = d.data[s.data]
        if data.dimensions != 2:
            return

        xrange = data.xrange
        yrange = data.yrange

        if name == s.xAxis:
            bounds[0] = min( bounds[0], xrange[0] )
            bounds[1] = max( bounds[1], xrange[1] )
        elif name == s.yAxis:
            bounds[0] = min( bounds[0], yrange[0] )
            bounds[1] = max( bounds[1], yrange[1] )

    def cutImageToFit(self, pltx, plty, posn):
        x1, y1, x2, y2 = posn
        pltx1, pltx2 = pltx
        pltw = pltx2-pltx1
        plty1, plty2 = plty
        plth = plty2-plty1

        imw = self.image.width()
        imh = self.image.height()
        pixw = pltw / float(imw)
        pixh = plth / float(imh)
        cutr = [0, 0, imw-1, imh-1]

        # work out where image intercepts posn, and make sure image
        # fills at least that area

        # need to chop left
        if pltx1 < x1:
            d = int((x1-pltx1) / pixw)
            cutr[0] += d
            pltx[0] += int(d*pixw)

        # need to chop right
        if pltx2 > x2:
            d = max(0, int((pltx2-x2) / pixw) - 1)
            cutr[2] -= d
            pltx[1] += int(d*pixh)

        # chop top
        if plty1 < y1:
            d = int((y1-plty1) / pixh)
            cutr[1] += d
            plty[0] += int(d*pixh)
            
        # chop bottom
        if plty2 > y2:
            d = max(0, int((plty2-y2) / pixh) - 1)
            cutr[3] -= d
            plty[1] -= int(d*pixh)

        # create chopped-down image
        newimage = self.image.copy(cutr[0], cutr[1],
                                   cutr[2]-cutr[0]+1, cutr[3]-cutr[1]+1)

        # return new image coordinates and image
        return pltx, plty, newimage

    def draw(self, parentposn, painter, outerbounds = None):
        """Draw the image."""

        posn = plotters.GenericPlotter.draw(self, parentposn, painter,
                                            outerbounds = outerbounds)
        x1, y1, x2, y2 = posn
        s = self.settings
        d = self.document
        
        # get axes widgets
        axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

        # return if there's no proper axes
        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' or
             s.data not in d.data ):
            return

        # return if the dataset isn't two dimensional
        data = d.data[s.data]
        if data.dimensions != 2:
            return

        # recalculate pixmap if image has changed
        if data != self.lastdataset or self.schangeset != s.changeset:
            self.updateImage()
            self.lastdataset = data
            self.schangeset = s.changeset

        # find coordinates of image coordinate bounds
        rangex, rangey = data.getDataRanges()

        # translate coordinates to plotter coordinates
        coordsx = axes[0].graphToPlotterCoords(posn, N.array(rangex))
        coordsy = axes[1].graphToPlotterCoords(posn, N.array(rangey))

        # truncate image down if necessary
        # This assumes linear pixels!
        if ( coordsx[0] < x1 or coordsx[1] > x2 or
             coordsy[0] < y1 or coordsy[1] > y2 ):

            coordsx, coordsy, image = self.cutImageToFit(coordsx, coordsy,
                                                         posn)
        else:
            image = self.image

        # clip data within bounds of plotter
        painter.beginPaintingWidget(self, posn)
        painter.save()
        painter.setClipRect( qt4.QRect(x1, y1, x2-x1, y2-y1) )

        # now draw pixmap
        painter.drawImage( qt4.QRect(coordsx[0], coordsy[1],
                                     coordsx[1]-coordsx[0]+1,
                                     coordsy[0]-coordsy[1]+1),
                           image )

        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate an image
document.thefactory.register( Image )
