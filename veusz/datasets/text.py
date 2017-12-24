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

"""Text datasets."""

from __future__ import division

from .commonfn import _
from .base import DatasetConcreteBase

from ..compat import cstr, crepr
from .. import utils

class DatasetText(DatasetConcreteBase):
    """Represents a text dataset: holding an array of strings."""

    dimensions = 1
    datatype = displaytype = 'text'
    columns = ('data',)
    column_descriptions = (_('Data'),)
    dstype = _('Text')
    editable = True

    def __init__(self, data=None, linked=None):
        """Initialise dataset with data given. Data are a list of strings."""

        DatasetConcreteBase.__init__(self, linked=linked)
        self.data = list(data)

    def description(self):
        return _('Text (length %i)') % len(self.data)

    def userSize(self):
        """Size of dataset."""
        return str( len(self.data) )

    def changeValues(self, type, vals):
        if type == 'data':
            self.data = list(vals)
        else:
            raise ValueError('type does not contain an allowed value')

        self.document.modifiedData(self)

    def uiConvertToDataItem(self, val):
        """Return a value cast to this dataset data type."""
        return cstr(val)

    def uiDataItemToData(self, val):
        """Return val converted to data."""
        return val

    def saveDataDumpToText(self, fileobj, name):
        '''Save data to file.
        '''
        fileobj.write("SetDataText(%s, [\n" % crepr(name))
        for line in self.data:
            fileobj.write("    %s,\n" % crepr(line))
        fileobj.write("])\n")

    def saveDataDumpToHDF5(self, group, name):
        """Save text data to hdf5 file."""
        tgrp = group.create_group(utils.escapeHDFDataName(name))
        tgrp.attrs['vsz_datatype'] = 'text'
        # make sure data are encoded
        encdata = [x.encode('utf-8') for x in self.data]
        tgrp['data'] = encdata
        tgrp['data'].attrs['vsz_name'] = name.encode('utf-8')

    def datasetAsText(self, fmt=None, join=None):
        """Return data as text."""
        lines = list(self.data)
        lines.append('')
        return '\n'.join(lines)

    def deleteRows(self, row, numrows):
        """Delete numrows rows starting from row.
        Returns deleted rows as a dict of {column:data, ...}
        """
        retn = {'data': self.data[row:row+numrows]}
        del self.data[row:row+numrows]

        self.document.modifiedData(self)
        return retn

    def insertRows(self, row, numrows, rowdata):
        """Insert numrows rows starting from row.
        rowdata is a dict of {column: data}.
        """
        data = rowdata.get('data', [])

        insdata = data + (['']*(numrows-len(data)))
        for d in insdata[::-1]:
            self.data.insert(row, d)

        self.document.modifiedData(self)

    def returnCopy(self):
        """Returns version of dataset with no linking."""
        return DatasetText(self.data)

    def returnCopyWithNewData(self, **args):
        """Return dataset of same type using the column data given."""
        return DatasetText(**args)
