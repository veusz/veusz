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

import qt
import numarray as N

import setting
import widgetfactory
import plotters

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
        s.add( setting.Dataset('data', '', self.document,
                               descr = 'Dataset to plot' ),
               0 )
        s.add( setting.FloatOrAuto('min', 'Auto',
                                   descr = 'Minimum value of image scale'),
               1 )
        s.add( setting.FloatOrAuto('max', 'Auto',
                                   descr = 'Maximum value of image scale'),
               2 )
        s.add( setting.Choice('colorScaling',
                              ['linear', 'sqrt', 'log'],
                              'linear',
                              descr = 'Scaling to transform numbers to color'),
               3 )

        s.add( setting.Choice('colorMap', Image.colormapnames,
                              'grey',
                              descr = 'Set of colors to plot data with'),
               4 )

        self.lastcolormap = None
        self.lastdataset = None
        self.schangeset = -1

    def readColorMaps(C):
        """Read color maps data file (a class method)

        File is made up of:
          comments (prefaced by # on separate line)
          colormapname
          list of colors with B G R alpha order from 0->255 on separate lines
          [colormapname ...]
        """

        name = ''
        vals = []
        C.colormaps = {}

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
                    C.colormaps[name] = N.array(vals).astype(N.UInt8)
                name = p[0]
                vals = []
            else:
                # add value to current colormap
                assert name != ''
                assert len(p) == 4
                vals.append( [int(i) for i in p] )

        # add on final colormap
        if name != '':
            C.colormaps[name] = N.array(vals).astype(N.UInt8)

        # collect names and sort alphabetically
        names = C.colormaps.keys()
        names.sort()
        C.colormapnames = names

    readColorMaps = classmethod(readColorMaps)

    def applyColourMap(self, cmap, data, minval, maxval):
        """Apply a colour map to the 2d data given.

        cmap is the color map (numarray of BGRalpha quads)
        data are the imaging data
        minval and maxval are the extremes of the data for the colormap
        Returns a QImage
        """

        # calculate fraction between min and max of data
        fracs = (N.ravel(data)-minval) * (1./(maxval-minval))
        fracs = N.clip(fracs, 0., 1.)

        # number of bands to split the data between (take account of end point)
        c = self.colormaps[cmap]
        numbands = c.shape[0]-1

        # Work out which is the minimum colour map. Assumes we have <255 bands.
        bands = (fracs*numbands).astype(N.UInt8)
        bands = N.clip(bands, 0, numbands-1)

        # work out fractional difference of data from band to next band
        deltafracs = (fracs - bands * (1./numbands)) * numbands

        # need to make a 2-dimensional array to multiply against triplets
        deltafracs.shape = (deltafracs.shape[0], 1)

        # calculate BGRalpha quadruplets
        # this is a linear interpolation between the band and the next band
        quads = (deltafracs*c[bands+1] +
                 (1.-deltafracs)*c[bands]).astype(N.UInt8)

        # convert 32bit quads to a Qt QImage
        # FIXME: Does this assume C-style array layout??
        s = quads.tostring()
        img = qt.QImage(s, data.shape[1], data.shape[0], 32, None, 0,
                        qt.QImage.IgnoreEndian)

        # convert QImage to QPixmap and return
        # docs suggest pixmaps are quicker for plotting...
        pixmap = qt.QPixmap(img)
        return pixmap

    def updatePixmap(self):
        """Update the pixmap with new contents."""

        s = self.settings
        d = self.document
        data = d.getData(s.data)

        minval = s.min
        if minval == 'Auto':
            minval = data.data.min()
        maxval = s.max
        if maxval == 'Auto':
            maxval = data.data.max()

        self.pixmap = self.applyColourMap(s.colorMap, data.data,
                                          minval, maxval)

    def autoAxis(self, name, bounds):
        """Automatically determine the ranges of variable on the axes."""

        s = self.settings
        d = self.document

        # return if no data
        if not d.hasData(s.data):
            return

        # return if the dataset isn't two dimensional
        data = d.getData(s.data)
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
             not d.hasData(s.data) ):
            return

        # return if the dataset isn't two dimensional
        data = d.getData(s.data)
        if data.dimensions != 2:
            return

        # recalculate pixmap if image has changed
        if data != self.lastdataset or self.schangeset != s.changeset:
            self.updatePixmap()
            self.lastdataset = data
            self.schangeset = s.changeset

        # find coordinates of image coordinate bounds
        rangex, rangey = data.getDataRanges()

        # translate coordinates to plotter coordinates
        coordsx = axes[0].graphToPlotterCoords(posn, N.array(rangex))
        coordsy = axes[1].graphToPlotterCoords(posn, N.array(rangey))

        # clip data within bounds of plotter
        painter.save()
        painter.setClipRect( qt.QRect(x1, y1, x2-x1, y2-y1) )

        # now draw pixmap
        painter.drawPixmap( qt.QRect(coordsx[0], coordsy[1],
                                     coordsx[1]-coordsx[0]+1,
                                     coordsy[0]-coordsy[1]+1),
                            self.pixmap )

        painter.restore()

# allow the factory to instantiate an axis
widgetfactory.thefactory.register( Image )
