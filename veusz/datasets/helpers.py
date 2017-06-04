#    Copyright (C) 2004 Jeremy S. Sanders
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
##############################################################################

"""Functions to be used by module users as helpers."""

from __future__ import division
import numpy as N

from .base import DatasetBase
from .oned import Dataset
from .twod import Dataset2D
from .text import DatasetText

from ..compat import cstr

def valsToDataset(vals, datatype, dimensions):
    """Return a dataset given a numpy array of values."""

    if datatype == 'numeric':
        try:
            nvals = N.array(vals, dtype=N.float64)

            if nvals.ndim == dimensions:
                if nvals.ndim == 1:
                    return Dataset(data=nvals)
                elif nvals.ndim == 2:
                    return Dataset2D(nvals)
        except ValueError:
            pass

    elif datatype == 'text':
        try:
            return DatasetText([cstr(x) for x in vals])
        except ValueError:
            pass

    raise RuntimeError('Invalid array')

def generateValidDatasetParts(datasets, breakds=True):
    """Generator to return array of valid parts of datasets.

    if breakds is True:
      Yields new datasets between rows which are invalid
    else:
      Yields single, filtered dataset
    """

    # find NaNs and INFs in input dataset
    invalid = datasets[0].invalidDataPoints()
    minlen = invalid.shape[0]
    for ds in datasets[1:]:
        if isinstance(ds, DatasetBase) and not ds.empty():
            nextinvalid = ds.invalidDataPoints()
            minlen = min(nextinvalid.shape[0], minlen)
            invalid = N.logical_or(invalid[:minlen], nextinvalid[:minlen])

    if breakds:
        # return multiple datasets, breaking at invalid values

        # get indexes of invalid points
        indexes = invalid.nonzero()[0].tolist()

        # no bad points: optimisation
        if not indexes:
            yield datasets
            return

        # add on shortest length of datasets
        indexes.append(minlen)

        lastindex = 0
        for index in indexes:
            if index != lastindex:
                retn = []
                for ds in datasets:
                    if ds is not None and (
                            not isinstance(ds, DatasetBase) or
                            not ds.empty()):
                        retn.append(ds[lastindex:index])
                    else:
                        retn.append(None)
                yield retn
            lastindex = index+1

    else:
        # in this mode we return single datasets where the invalid
        # values are masked out

        if not N.any(invalid):
            yield datasets
            return

        valid = N.logical_not(invalid)
        retn = []
        for ds in datasets:
            if ds is None:
                retn.append(None)
            else:
                retn.append(ds[valid])
        yield retn
