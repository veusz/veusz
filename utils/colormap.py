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
import veusz.qtall as qt4

# use fast or slow helpers
slowfuncs = False
try:
    from veusz.helpers.qtloops import numpyToQImage, applyImageTransparancy
except ImportError:
    slowfuncs = True
    from slowfuncs import slowNumpyToQImage

# Default colormaps used by widgets.
# Each item in this dict is a colormap entry, with the key the name.
# The values in the dict are tuples of (B, G, R, alpha).
# B, G, R and alpha go from 0 to 255
# Colors are linearly interpolated in this space.

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
    'spectrum': (
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
    'transblack': (
        (0,   0,   0,   255),
        (0,   0,   0,   0),
        ),
    'royal': (
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
        cmap = cmap[::-1]

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
