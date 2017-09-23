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

from __future__ import division
import re
import numpy as N

from .. import qtall as qt4

# use fast or slow helpers
slowfuncs = False
try:
    from ..helpers.qtloops import numpyToQImage, applyImageTransparancy
except ImportError:
    slowfuncs = True
    from .slowfuncs import slowNumpyToQImage

# Default colormaps used by widgets.
# Each item in this dict is a colormap entry, with the key the name.

# The values in the dict are tuples of (B, G, R, alpha).  B, G, R and
# alpha go from 0 to 255

# Colors are linearly interpolated in this space, unless they start
# with (-1,0,0,0) which enables a step mode (this first value is
# ignored)

_defaultmaps = {
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

    # from http://www.kennethmoreland.com/color-maps/
    'cool-warm': (
        (59, 76, 192, 255), 
        (68, 90, 204, 255), 
        (77, 104, 215, 255), 
        (87, 117, 225, 255), 
        (98, 130, 234, 255), 
        (108, 142, 241, 255), 
        (119, 154, 247, 255), 
        (130, 165, 251, 255), 
        (141, 176, 254, 255), 
        (152, 185, 255, 255), 
        (163, 194, 255, 255), 
        (174, 201, 253, 255), 
        (184, 208, 249, 255), 
        (194, 213, 244, 255), 
        (204, 217, 238, 255), 
        (213, 219, 230, 255), 
        (221, 221, 221, 255), 
        (229, 216, 209, 255), 
        (236, 211, 197, 255), 
        (241, 204, 185, 255), 
        (245, 196, 173, 255), 
        (247, 187, 160, 255), 
        (247, 177, 148, 255), 
        (247, 166, 135, 255), 
        (244, 154, 123, 255), 
        (241, 141, 111, 255), 
        (236, 127, 99, 255), 
        (229, 112, 88, 255), 
        (222, 96, 77, 255), 
        (213, 80, 66, 255), 
        (203, 62, 56, 255), 
        (192, 40, 47, 255), 
        (180, 4, 38, 255),
        ),

    # from http://geog.uoregon.edu/datagraphics/color_scales.htm
    'green-magenta': (
        (0, 80, 0, 255),
        (0, 134, 0, 255),
        (0, 187, 0, 255),
        (0, 241, 0, 255),
        (80, 255, 80, 255),
        (134, 255, 134, 255),
        (187, 255, 187, 255),
        (255, 255, 255, 255),
        (255, 241, 255, 255),
        (255, 187, 255, 255),
        (255, 134, 255, 255),
        (255, 80, 255, 255),
        (241, 0, 241, 255),
        (187, 0, 187, 255),
        (134, 0, 134, 255),
        (80, 0, 80, 255)
        ),

    'blue-darkred': (
        (216, 0, 36, 255),
        (247, 28, 24, 255),
        (255, 87, 40, 255),
        (255, 135, 61, 255),
        (255, 176, 86, 255),
        (255, 211, 117, 255),
        (255, 234, 153, 255),
        (255, 249, 188, 255),
        (255, 255, 234, 255),
        (234, 255, 255, 255),
        (188, 241, 255, 255),
        (153, 214, 255, 255),
        (117, 172, 255, 255),
        (86, 120, 255, 255),
        (61, 61, 255, 255),
        (53, 39, 247, 255),
        (47, 21, 216, 255),
        (33, 0, 165, 255)
        ),

    'blue-darkorange': (
        (102, 102, 0, 255),
        (153, 153, 0, 255),
        (204, 204, 0, 255),
        (255, 255, 0, 255),
        (255, 255, 51, 255),
        (255, 255, 101, 255),
        (255, 255, 153, 255),
        (255, 255, 178, 255),
        (255, 255, 203, 255),
        (255, 255, 229, 255),
        (203, 229, 255, 255),
        (153, 202, 255, 255),
        (101, 173, 255, 255),
        (51, 142, 255, 255),
        (0, 110, 255, 255),
        (0, 85, 204, 255),
        (0, 61, 153, 255),
        (0, 39, 102, 255)
        ),

    'brown-blue': (
        (0, 25, 51, 255),
        (0, 47, 102, 255),
        (53, 96, 153, 255),
        (122, 155, 204, 255),
        (151, 175, 216, 255),
        (205, 218, 242, 255),
        (255, 253, 204, 255),
        (255, 248, 153, 255),
        (255, 239, 101, 255),
        (255, 227, 50, 255),
        (204, 169, 0, 255),
        (153, 122, 0, 255)
        ),

    'blue-orange': (
        (255, 42, 0, 255),
        (255, 101, 25, 255),
        (255, 153, 50, 255),
        (255, 204, 101, 255),
        (255, 237, 153, 255),
        (255, 255, 204, 255),
        (204, 255, 255, 255),
        (153, 238, 255, 255),
        (101, 204, 255, 255),
        (50, 153, 255, 255),
        (25, 102, 255, 255),
        (0, 42, 255, 255)
        ),

    'seq': (
        (15, 15, 153, 255),
        (44, 44, 178, 255),
        (81, 81, 204, 255),
        (126, 126, 229, 255),
        (178, 178, 255, 255),
        (15, 84, 153, 255),
        (44, 111, 178, 255),
        (81, 142, 204, 255),
        (126, 177, 229, 255),
        (178, 216, 255, 255),
        (15, 153, 107, 255),
        (44, 178, 133, 255),
        (81, 204, 163, 255),
        (126, 229, 195, 255),
        (178, 255, 229, 255),
        (153, 107, 15, 255),
        (178, 133, 44, 255),
        (204, 163, 81, 255),
        (229, 195, 126, 255),
        (255, 229, 178, 255),
        (153, 15, 38, 255),
        (178, 44, 66, 255),
        (204, 81, 101, 255),
        (229, 126, 143, 255),
        (255, 178, 191, 255),
        ),
    
    # from https://github.com/visit-vis/VisIt/blob/master/avt/Pipeline/Data/avtColorTables.C 
    'hot_desaturated': (
        (219, 71,  71,  255),
        (91,  0,   0,   255),
        (255, 255, 0,   255),
        (0,   127, 0,   255),
        (0,   255, 255, 255),
        (0,   96,  255, 255),
        (0,   0,   107, 255),
        (76,  76,  224, 255),
        ),
    
    # exported from VisIt color table (http://www.visitusers.org)
    'yellow-green': (
        (229, 255, 255, 255),
        (185, 252, 247, 255),
        (163, 240, 217, 255),
        (142, 221, 173, 255),
        (121, 198, 120, 255),
        (93,  171, 65,  255),
        (67,  132, 35,  255),
        (55,  104, 0,   255),
        (41,  69,  0,   255),
        ),
    'yellow-green-blue': (
        (217, 255, 255, 255),
        (177, 248, 237, 255),
        (180, 233, 199, 255),
        (187, 205, 127, 255),
        (196, 182, 65,  255),
        (192, 145, 29,  255),
        (168, 94,  34,  255),
        (148, 52,  37,  255),
        (88,  29,  8,   255),
        ),
    'yellow-orange-brown': (
        (229, 255, 255, 255),
        (188, 247, 255, 255),
        (145, 227, 254, 255),
        (79,  196, 254, 255),
        (41,  153, 254, 255),
        (20,  112, 236, 255),
        (2,   76,  204, 255),
        (4,   52,  153, 255),
        (6,   37,  102, 255),
        ),
    'yellow-orange-red': (
        (204, 255, 255, 255),
        (160, 237, 255, 255),
        (118, 217, 254, 255),
        (76,  178, 254, 255),
        (60,  141, 253, 255),
        (42,  78,  252, 255),
        (28,  26,  227, 255),
        (38,  0,   189, 255),
        (38,  0,   128, 255),
        ),

    }

def cubehelix(start, rots, hue, gamma, nlev=64):
    """Return a cube helix color scheme.
    See https://www.mrao.cam.ac.uk/~dag/CUBEHELIX/
    Green, D. A., 2011, `A colour scheme for the display of astronomical
    intensity images', Bulletin of the Astronomical Society of India, 39, 28
    """

    fract = N.linspace(0, 1, nlev)
    angle = 2*N.pi*(start/3.+1.+rots*fract)
    fract = fract**gamma
    amp = 0.5*hue*fract*(1-fract)
    c, s = N.cos(angle), N.sin(angle)
    red   = fract+amp*(-0.14861*c+1.78277*s)
    green = fract+amp*(-0.29227*c-0.90649*s)
    blue  = fract+amp*( 1.97294*c)

    r = N.clip(red*255, 0, 255)
    g = N.clip(green*255, 0, 255)
    b = N.clip(blue*255, 0, 255)
    a = N.zeros(nlev)+255

    return N.column_stack( (b,g,r,a) ).astype(N.intc)

def stepCMap(cmap, n):
    """Give color map, interpolate to produce n steps and return stepped
    colormap."""

    if n == 0:
        return N.vstack( ([-1,0,0,0], cmap) ).astype(N.intc)

    cmap = N.array(cmap, dtype=N.float64)
    x = N.linspace(0, 1, n)
    xp = N.linspace(0, 1, len(cmap))

    b = N.interp(x, xp, cmap[:,0])
    g = N.interp(x, xp, cmap[:,1])
    r = N.interp(x, xp, cmap[:,2])
    a = N.interp(x, xp, cmap[:,3])

    return N.vstack( ([-1,0,0,0], N.column_stack((b,g,r,a))) ).astype(N.intc)

class ColorMaps(object):
    """Class representing defined color maps.

    This is initialised from the default list.

    Also supported are functional color maps,
    e.g. cubehelix(start[,rotations[,hue[,gamma]]])

    Colormaps with steps -stepN where N is an integer or missing are
    also automatically generated.
    """

    def __init__(self):
        self.wipe()

    def wipe(self):
        """Reset colormaps to default state."""
        self.maps = dict(_defaultmaps)

    def get(self, idx, default=None):
        try:
            return self[idx]
        except KeyError:
            return default

    def __getitem__(self, key):
        """Lookup and return colormap."""

        origkey = key = key.strip()

        if key in self.maps:
            return self.maps[key]

        # does the name end in stepXXX ?
        step = None
        sm = re.match(r'^(.+)-step([0-9]*)$', key)
        if sm is not None:
            if sm.group(2):
                step = int(sm.group(2))
            else:
                step = 0
            key = sm.group(1)

        cmap = None
        if key in self.maps:
            cmap = self.maps[key]
        else:
            # match cubehelix(a,b,c,d), where b, c and d are optional numerics
            # giving start, rotations, hue and gamma
            cm = re.match(
                r'^cubehelix\s*\('
                r'(?:\s*(-?[0-9.]+))?'
                r'(?:\s*,\s*(-?[0-9.]+))?'
                r'(?:\s*,\s*(-?[0-9.]+))?'
                r'(?:\s*,\s*(-?[0-9.]+))?'
                r'\s*\)$',
                key)

            if cm is not None:
                vals = []
                for i, v in enumerate(cm.groups()):
                    try:
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        vals.append((0,1,1,1)[i])
                cmap = cubehelix(*vals)

        if cmap is None:
            raise KeyError('Invalid colormap name')

        # apply steps to colormap
        if step is not None:
            cmap = stepCMap(cmap, step)

        # cache result and return
        self.maps[origkey] = cmap
        return cmap

    def __setitem__(self, key, val):
        self.maps[key] = val

    def __contains__(self, key):
        return self.get(key) is not None

    def __iter__(self):
        items = set(self.maps)
        items.update([
            'cubehelix(0.5,-1.5,1,1)',
            'bluegreen-step',
            'complement-step',
            'grey-step5',
            'grey-step6',
            'royal-step',
            'spectrum-step',
            'transblack-step5',
            'green-magenta-step16',
            'blue-darkred-step12',
            'blue-darkorange-step12',
            'brown-blue-step12',
            'blue-orange-step12',
            'seq-step25',
        ])
        return iter(items)

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
        with N.errstate(invalid='ignore', divide='ignore'):
            invrange = 1./(N.log(maxval)-N.log(minval))
            data = (N.log(data)-N.log(minval)) * invrange
        data[~N.isfinite(data)] = N.nan

    elif mode == 'squared':
        # squared scaling
        # clip any negative values
        lowermin = data < minval
        data = (data-minval)**2 / (maxval-minval)**2
        data[lowermin] = 0.

    else:
        raise RuntimeError('Invalid scaling mode "%s"' % mode)

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
                      direction='horizontal', barsize=128):
    """Make a colorbar for the scaling given."""

    if scaling in ('linear', 'sqrt', 'squared'):
        # do a linear color scaling
        vals = N.arange(barsize)/(barsize-1.0)*(maxval-minval) + minval
        colorscaling = scaling
    elif scaling == 'log':
        # a logarithmic color scaling
        # we cheat here by actually plotting a linear colorbar
        # and telling veusz to put a log axis along it
        # (as we only care about the endpoints)
        # maybe should do this better...

        vals = N.arange(barsize)/(barsize-1.0)*(maxval-minval) + minval
        colorscaling = 'linear'
    else:
        raise RuntimeError('Invalid scaling')

    # convert 1d array to 2d image
    if direction == 'horizontal':
        vals = vals.reshape(1, barsize)
    else:
        assert direction == 'vertical'
        vals = vals.reshape(barsize, 1)

    img = applyColorMap(cmap, colorscaling, vals,
                        minval, maxval, transparency)

    return img

def getColormapArray(cmap, nvals):
    """Get [R,G,B,alpha] array of nvals for colormap array given.

    number of values to return is given by nvals
    """

    img = makeColorbarImage(0, 1, 'linear', cmap, 0, barsize=nvals)

    # ensure data are stored in the correct order
    fmt = qt4.QImage.Format_ARGB32
    if img.format() != fmt:
        img = img.convertToFormat(fmt)

    # convert image pointer to numpy array (assumes nvals*4 bytes)
    cv = N.array(img.constScanLine(0).asarray(nvals*4)).reshape((nvals,4))

    # swap into RGBA
    cv = N.column_stack(( cv[:,2], cv[:,1], cv[:,0], cv[:,3] ))

    return cv
