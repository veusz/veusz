# -*- coding: utf-8 -*-
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

"""N dimensional datasets."""

import numpy as N

from ..compat import crepr
from .. import utils

from .commonfn import _, dsPreviewHelper
from .base import DatasetConcreteBase

class DatasetNDBase(DatasetConcreteBase):
    """N-dimensional datasets."""

    dimensions = -1
    dstype = _('nD')
    editable = False

    def userSize(self):
        return u'×'.join(str(x) for x in self.data.shape)

    def userPreview(self):
        return dsPreviewHelper(N.ravel(self.data))

    def description(self):
        return _('ND (%s), numeric') % self.userSize()

    def returnCopy(self):
        return DatasetND(data=self.data)

    def returnCopyWithNewData(self, **args):
        return DatasetND(**args)

    def empty(self):
        """Is the data defined?"""
        return len(self.data) == 0

    def datasetAsText(self, fmt='%g', join='\t'):
        """Dataset as text for copy, paste, etc."""
        def fmtrecurse(arr):
            if arr.ndim == 0:
                return fmt % arr + '\n'
            elif arr.ndim == 1:
                out = [fmt % v for v in arr]
                return join.join(out)
            else:
                out = []
                for v in arr:
                    out.append(fmtrecurse(v))
                out.append('')
                return '\n'.join(out)

        return fmtrecurse(self.data)

class DatasetND(DatasetNDBase):
    def __init__(self, data=None):
        """data is a numpy array of N dimensions."""

        DatasetNDBase.__init__(self)

        if isinstance(data, N.ndarray):
            self.data = data.astype(N.float64)
        elif isinstance(data, list) or isinstance(data, tuple):
            self.data = N.array(dtype=N.float64)
        else:
            raise ValueError("Could not convert data to nD numpy array.")

    def saveDataDumpToText(self, fileobj, name):
        """Save data to vsz in form of text."""

        fileobj.write("ImportStringND(%s, '''\n" % crepr(name))
        if self.data.shape[0] == 1:
            # unfortunately it's hard to decode a single dimension
            # here so we record this unambiguously
            shape = ' '.join((str(d) for d in self.data.shape))
            fileobj.write("shape %s\n" % shape)
        fileobj.write(self.datasetAsText(fmt='%e', join=' '))
        fileobj.write("''')\n")

    def saveDataDumpToHDF5(self, group, name):
        """Save dataset to VSZ HDF5 format."""

        escname = utils.escapeHDFDataName(name)
        group[escname] = self.data
        group[escname].attrs['vsz_datatype'] = 'nd'
        group[escname].attrs['vsz_name'] = name.encode('utf-8')
