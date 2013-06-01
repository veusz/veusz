#    Copyright (C) 2011 Jeremy S. Sanders
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

import numpy as N

# use fast or slow helpers
slowfuncs = False
try:
    from veusz.helpers.qtloops import numpyToQImage, applyImageTransparancy
except ImportError:
    slowfuncs = True
    from slowfuncs import slowNumpyToQImage

# Default colormaps used by widgets.
# Each item in this dict is a colormap entry, with the key the name.

# The values in the dict are tuples of (B, G, R, alpha).  B, G, R and
# alpha go from 0 to 255

# Colors are linearly interpolated in this space, unless they start
# with (-1,0,0,0) which enables a step mode (this first value is
# ignored)

defaultcolormaps = {
    'blank': (
        (0,   0,   0,   0),
        (0,   0,   0,   0),
        ),
    'heat': (
        (0,   0,   0,   255),
        (0,   0,   186, 255),
        (50,  139, 255, 255),
        (19,  239, 248, 255),
        (255, 255, 255, 255),
        ),
    'spectrum2': (
        (0,   0,   255, 255),
        (0,   255, 255, 255),
        (0,   255, 0,   255),
        (255, 255, 0,   255),
        (255, 0,   0,   255),
        ),
    'spectrum2-step': (
        (-1,  0,   0,   0),
        (0,   0,   255, 255),
        (0,   255, 255, 255),
        (0,   255, 0,   255),
        (255, 255, 0,   255),
        (255, 0,   0,   255),
        ),
    'spectrum': (
        (0,   0,   0,   255),
        (0,   0,   255, 255),
        (0,   255, 255, 255),
        (0,   255, 0,   255),
        (255, 255, 0,   255),
        (255, 0,   0,   255),
        (255, 255, 255, 255),
        ),
    'spectrum-step': (
        (-1,  0,   0,   0),
        (0,   0,   0,   255),
        (0,   0,   255, 255),
        (0,   255, 255, 255),
        (0,   255, 0,   255),
        (255, 255, 0,   255),
        (255, 0,   0,   255),
        (255, 255, 255, 255),
        ),
    'grey': (
        (0,   0,   0,   255),
        (255, 255, 255, 255),
        ),
    'grey-step5': (
        (-1,  0,   0,   0),
        (0,   0,   0,   255),
        (64,  64,  64,  255),
        (128, 128, 128, 255),
        (191, 191, 191, 255),
        (255, 255, 255, 255),
        ),
    'grey-step6': (
        (-1,  0,   0,   0),
        (0,   0,   0,   255),
        (51,  51,  51,  255),
        (102, 102, 102, 255),
        (153, 153, 153, 255),
        (204, 204, 204, 255),
        (255, 255, 255, 255),
        ),
    'blue': (
        (0,   0,   0,   255),
        (255, 0,   0,   255),
        (255, 255, 255, 255),
        ),
    'red': (
        (0,   0,   0,   255),
        (0,   0,   255, 255),
        (255, 255, 255, 255),
        ),
    'green': (
        (0,   0,   0,   255),
        (0,   255, 0,   255),
        (255, 255, 255, 255),
        ),
    'bluegreen': (
        (0,   0,   0,   255),
        (255, 123, 0,   255),
        (255, 226, 72,  255),
        (161, 255, 0,   255),
        (255, 255, 255, 255),
        ),
    'bluegreen-step': (
        (-1,  0,   0,   0),
        (0,   0,   0,   255),
        (255, 123, 0,   255),
        (255, 226, 72,  255),
        (161, 255, 0,   255),
        (255, 255, 255, 255),
        ),
    'transblack': (
        (0,   0,   0,   255),
        (0,   0,   0,   0),
        ),
    'transblack-step5': (
        (-1,  0,   0,   0),
        (0,   0,   0,   255),
        (0,   0,   0,   191),
        (0,   0,   0,   128),
        (0,   0,   0,   64),
        (0,   0,   0,   0),
        ),
    'royal': (
        (0,   0,   0,   255),
        (128, 0,   0,   255),
        (255, 0,   128, 255),
        (0,   255, 255, 255),
        (255, 255, 255, 255),
        ),
    'royal-step': (
        (-1,  0,   0,   0),
        (0,   0,   0,   255),
        (128, 0,   0,   255),
        (255, 0,   128, 255),
        (0,   255, 255, 255),
        (255, 255, 255, 255),
        ),
    'complement': (
        (0,   0,   0,   255),
        (0,   255, 0,   255),
        (255, 0,   255, 255),
        (0,   0,   255, 255),
        (0,   255, 255, 255),
        (255, 255, 255, 255),
        ),
    'complement-step': (
        (-1,  0,   0,   0),
        (0,   0,   0,   255),
        (0,   255, 0,   255),
        (255, 0,   255, 255),
        (0,   0,   255, 255),
        (0,   255, 255, 255),
        (255, 255, 255, 255),
        ),
    }

def applyScaling(data, mode, minval, maxval):
    """Apply a scaling transformation on the data.
    data is a numpy array
    mode is one of 'linear', 'sqrt', 'log', or 'squared'
    minval is the minimum value of the scale
    maxval is the maximum value of the scale

    returns transformed data, valid between 0 and 1
    """

    # catch naughty people by hardcoding a range
    if minval == maxval:
        minval, maxval = 0., 1.
        
    if mode == 'linear':
        # linear scaling
        data = (data - minval) / (maxval - minval)

    elif mode == 'sqrt':
        # sqrt scaling
        # translate into fractions of range
        data = (data - minval) / (maxval - minval)
        # clip off any bad sqrts
        data[data < 0.] = 0.
        # actually do the sqrt transform
        data = N.sqrt(data)

    elif mode == 'log':
        # log scaling of image
        # clip any values less than lowermin
        lowermin = data < minval
        data = N.log(data - (minval - 1)) / N.log(maxval - (minval - 1))
        data[lowermin] = 0.

    elif mode == 'squared':
        # squared scaling
        # clip any negative values
        lowermin = data < minval
        data = (data-minval)**2 / (maxval-minval)**2
        data[lowermin] = 0.

    else:
        raise RuntimeError, 'Invalid scaling mode "%s"' % mode

    return data

def applyColorMap(cmap, scaling, datain, minval, maxval,
                  trans, transimg=None):
    """Apply a colour map to the 2d data given.

    cmap is the color map (numpy of BGRalpha quads)
    scaling is scaling mode => 'linear', 'sqrt', 'log' or 'squared'
    data are the imaging data
    minval and maxval are the extremes of the data for the colormap
    trans is a number from 0 to 100
    transimg is an optional image to apply transparency from
    Returns a QImage
    """

    cmap = N.array(cmap, dtype=N.intc)

    # invert colour map if min and max are swapped
    if minval > maxval:
        minval, maxval = maxval, minval
        if cmap[0,0] >= 0:
            # reverse standard colormap
            cmap = cmap[::-1]
        else:
            # uses flag signal at start of array for stepped maps
            # ignore this in reverse
            cmap[1:] = cmap[-1:0:-1]

    # apply transparency
    if trans != 0:
        cmap = cmap.copy()
        cmap[:,3] = (cmap[:,3].astype(N.float32) * (100-trans) /
                     100.).astype(N.intc)

    # apply scaling of data
    fracs = applyScaling(datain, scaling, minval, maxval)

    if not slowfuncs:
        img = numpyToQImage(fracs, cmap, transimg is not None)
        if transimg is not None:
            applyImageTransparancy(img, transimg)
    else:
        img = slowNumpyToQImage(fracs, cmap, transimg)
    return img

def makeColorbarImage(minval, maxval, scaling, cmap, transparency,
                      direction='horz'):
    """Make a colorbar for the scaling given."""

    barsize = 128

    if scaling in ('linear', 'sqrt', 'squared'):
        # do a linear color scaling
        vals = N.arange(barsize)/(barsize-1.0)*(maxval-minval) + minval
        colorscaling = scaling
        coloraxisscale = 'linear'
    else:
        assert scaling == 'log'

        # a logarithmic color scaling
        # we cheat here by actually plotting a linear colorbar
        # and telling veusz to put a log axis along it
        # (as we only care about the endpoints)
        # maybe should do this better...

        vals = N.arange(barsize)/(barsize-1.0)*(maxval-minval) + minval
        colorscaling = 'linear'
        coloraxisscale = 'log'

    # convert 1d array to 2d image
    if direction == 'horizontal':
        vals = vals.reshape(1, barsize)
    else:
        assert direction == 'vertical'
        vals = vals.reshape(barsize, 1)

    img = applyColorMap(cmap, colorscaling, vals,
                        minval, maxval, transparency)

    return img
