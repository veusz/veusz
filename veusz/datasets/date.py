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

"""Date time datasets."""

from __future__ import division

import numpy as N

from .. import utils
from ..compat import cbasestr, cstr, crepr

from .commonfn import _, convertNumpy, datasetNameToDescriptorName
from .oned import Dataset1DBase

class DatasetDateTimeBase(Dataset1DBase):
    """Dataset holding dates and times."""

    columns = ('data',)
    column_descriptions = (_('Data'),)

    dstype = _('Date')
    displaytype = 'date'

    def description(self):
        return _('Date/time (length %i)') % len(self.data)

    def returnCopy(self):
        """Returns version of dataset with no linking."""
        return DatasetDateTime(data=N.array(self.data))

    def returnCopyWithNewData(self, **args):
        """Return dataset of same type using the column data given."""
        return DatasetDateTime(**args)

    def uiConvertToDataItem(self, val):
        """Return a value cast to this dataset data type."""
        if isinstance(val, cbasestr):
            v = utils.dateStringToDate( cstr(val) )
            if not N.isfinite(v):
                try:
                    v = float(val)
                except ValueError:
                    pass
            return v
        else:
            return N.nan

    def uiDataItemToData(self, val):
        """Return val converted to data."""
        return utils.dateFloatToString(val)

    def datasetAsText(self, fmt=None, join=None):
        """Return data as text."""
        lines = [ utils.dateFloatToString(val) for val in self.data ]
        lines.append('')
        return '\n'.join(lines)

class DatasetDateTime(DatasetDateTimeBase):
    """Standard date/time class for use by humans."""

    editable = True

    def __init__(self, data=None, linked=None):
        DatasetDateTimeBase.__init__(self, linked=linked)

        self.data = convertNumpy(data)
        self.perr = self.nerr = self.serr = None

    def saveDataDumpToText(self, fileobj, name):
        '''Save data to file.
        '''
        descriptor = datasetNameToDescriptorName(name) + '(date)'
        fileobj.write( "ImportString(%s,'''\n" % crepr(descriptor) )
        fileobj.write( self.datasetAsText() )
        fileobj.write( "''')\n" )

    def saveDataDumpToHDF5(self, group, name):
        """Save date data to hdf5 file."""
        dgrp = group.create_group(utils.escapeHDFDataName(name))
        dgrp.attrs['vsz_datatype'] = 'date'
        dgrp['data'] = self.data
        data = dgrp['data']
        data.attrs['vsz_convert_datetime'] = 1
        data.attrs['vsz_name'] = name.encode('utf-8')

    def deleteRows(self, row, numrows):
        """Delete numrows rows starting from row.
        Returns deleted rows as a dict of {column:data, ...}
        """
        retn = {
            'data': self.data[row:row+numrows],
        }
        self.data = N.delete(self.data, N.s_[row:row+numrows])
        self.document.modifiedData(self)
        return retn

    def insertRows(self, row, numrows, rowdata):
        """Insert numrows rows starting from row.
        rowdata is a dict of {column: data}.
        """
        data = N.zeros(numrows)
        if 'data' in rowdata:
            data[:len(rowdata['data'])] = N.array(rowdata['data'])
        self.data =  N.insert(self.data, [row]*numrows, data)
        self.document.modifiedData(self)

    def changeValues(self, thetype, vals):
        """Change the requested part of the dataset to vals.

        thetype == data
        """
        if thetype != 'data':
            raise ValueError('invalid column %s' % thetype)

        self.data = N.array(vals)

        # tell the document that we've changed
        self.document.modifiedData(self)
