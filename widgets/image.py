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

import numarray as N

import setting
import widgetfactory
import plotters

class Image(plotters.GenericPlotter):
    """A class which plots an image on a graph with a specified
    coordinate system."""

    colormaps = None

    typename='image'
    allowusercreation=True
    description='Plot a 2d dataset as an image'

    def readColorMaps(self):
        """Read color maps data file.

        File is made up of:
          comments (prefaced by # on separate line)
          colormapname
          list of colors with B G R alpha order from 0->255 on separate lines
          [colormapname ...]
        """

        name = ''
        vals = []
        Image.colormaps = {}

        dir = os.path.dirname( os.path.abspath(__file__) )
        filename = os.path.join(dir, 'data', 'colormaps.dat')
        f = open(filename)
        for l in f:
            p = l.split()
            if len(p) == 0 or p[0][0] == '#':
                continue
            if p[0][0] not in string.digits:
                if len(vals) != 0:
                    Image.colormaps[name] = N.array(vals).astype(N.UInt8)
                name = p[0]
                vals = []
            else:
                assert name != ''
                vals.append( [int(i) for i in p] )
        if name != '':
            Image.colormaps[name] = N.array(vals).astype(N.UInt8)

        # collect names and sort alphabetically
        names = Image.colormaps.keys()
        names.sort()
        Image.colormapnames = names

    def __init__(self, parent, name=None):
        """Initialise plotter with axes."""

        plotters.GenericPlotter.__init__(self, parent, name=name)

        # lazy read of colormap file
        if Image.colormaps == None:
            self.readColorMaps()

        s = self.settings
        s.add( setting.Dataset('data', 'd', self.document,
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
        c = Image.colormaps[cmap]
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
        s = quads.tostring()
        img = qt.QImage(s, data.shape[1], data.shape[0], 32, None, 0,
                        qt.QImage.IgnoreEndian)

        # we need to store the string while the image exists
        # this is a hacky, but simple way of ensuring this
        img._pythonstring = s

        return img

# allow the factory to instantiate an axis
widgetfactory.thefactory.register( Image )
