#    Copyright (C) 2016 Jeremy S. Sanders
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

def _(text, disambiguation=None, context="Datasets"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

def convertNumpy(a, dims=1):
    """Convert to a numpy double if possible.

    dims is number of dimensions to check for
    """
    if a is None:
        # leave as None
        return None
    elif isinstance(a, N.ndarray):
        # make conversion if numpy type is not correct
        if a.dtype != N.float64:
            a = a.astype(N.float64)
    else:
        # convert to numpy array
        a = N.array(a, dtype=N.float64)

    if a.ndim != dims:
        if a.ndim == 0:
            if dims == 1:
                a = a.reshape((1,))
            elif dims == 2:
                a = a.reshape((1,1))
            else:
                raise RuntimeError()
        else:
            raise ValueError("Only %i-dimensional arrays or lists allowed" % dims)
    return a

def convertNumpyAbs(a):
    """Convert to numpy 64 bit positive values, if possible."""
    if a is None:
        return None
    else:
        return N.abs( convertNumpy(a) )

def convertNumpyNegAbs(a):
    """Convert to numpy 64 bit negative values, if possible."""
    if a is None:
        return None
    else:
        return -N.abs( convertNumpy(a) )

def copyOrNone(a):
    """Return a copy if not None, or None."""
    if a is None:
        return None
    elif isinstance(a, N.ndarray):
        return N.array(a)
    elif isinstance(a, list):
        return list(a)

def datasetNameToDescriptorName(name):
    """Return descriptor name for dataset."""
    if re.match('^[0-9A-Za-z_]+$', name):
        return name
    else:
        return '`%s`' % name

def dsPreviewHelper(d):
    """Get preview of numpy data d."""
    if d.shape[0] <= 6:
        line1 = ', '.join( ['%.3g' % x for x in d] )
    else:
        line1 = ', '.join( ['%.3g' % x for x in d[:3]] +
                           [ '...' ] +
                           ['%.3g' % x for x in d[-3:]] )

    try:
        line2 = _('mean: %.3g, min: %.3g, max: %.3g') % (
            N.nansum(d) / N.isfinite(d).sum(),
            N.nanmin(d),
            N.nanmax(d))
    except (ValueError, ZeroDivisionError):
        # nanXXX returns error if no valid data points
        return line1
    return line1 + '\n' + line2
