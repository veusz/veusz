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

"""Base class for all Datasets."""

from __future__ import division

from ..compat import cbasestr
from .commonfn import _

class DatasetException(Exception):
    """Raised with dataset errors."""
    pass

class DatasetExpressionException(DatasetException):
    """Raised if there is an error evaluating a dataset expression."""
    pass

class DatasetBase(object):
    """Base class for all datasets."""

class DatasetConcreteBase(DatasetBase):
    """A base dataset class for datasets which are real, and not proxies,
    etc."""

    # number of dimensions the dataset holds
    dimensions = 0

    # datatype is fundamental type of data
    # displaytype is formatting suggestion for data
    datatype = displaytype = 'numeric'

    # dataset type to show to user
    dstype = 'Dataset'

    # list of columns in dataset (if any)
    columns = ()
    # use descriptions for columns
    column_descriptions = ()

    # can values be edited
    editable = False

    # class for representing part of this dataset
    subsetclass = None

    def __init__(self, linked=None):
        """Initialise commonfn members."""
        # document member set when this dataset is set in document
        self.document = None

        # file this dataset is linked to
        self.linked = linked

        # tags applied to dataset
        self.tags = set()

    def saveLinksToSavedDoc(self, fileobj, savedlinks, relpath=None):
        '''Save the link to the saved document, if this dataset is linked.

        savedlinks is a dict containing any linked files which have
        already been written

        relpath is a directory to save linked files relative to
        '''

        # links should only be saved once
        if self.linked is not None and self.linked not in savedlinks:
            savedlinks[self.linked] = True
            self.linked.saveToFile(fileobj, relpath=relpath)

    def saveToFile(self, fileobj, name, mode='text', hdfgroup=None):
        """Save dataset to file."""
        self.saveDataRelationToText(fileobj, name)
        if self.linked is None:
            if mode == 'text':
                self.saveDataDumpToText(fileobj, name)
            elif mode == 'hdf5':
                self.saveDataDumpToHDF5(hdfgroup, name)

    def saveDataRelationToText(self, fileobj, name):
        """Save a dataset relation to a text stream fileobj.
        Not for datasets which are raw data."""

    def saveDataDumpToText(self, fileobj, name):
        """Save dataset to file if file is text and data
        is actually a set of data and not a relation
        """

    def saveDataDumpToHDF5(self, group, name):
        """Save dumped dataset to HDF5.
        group is the group to save it in (h5py group)
        """

    def userSize(self):
        """Return dimensions of dataset for user."""
        return ""

    def userPreview(self):
        """Return a small preview of the dataset for the user, e.g.
        1, 2, 3, ..., 4, 5, 6."""
        return None

    def description(self):
        """Get description of dataset."""
        return ""

    def uiConvertToDataItem(self, val):
        """Return a value cast to this dataset data type.
        We assume here it is a float, so override if not
        """
        from .. import setting
        if isinstance(val, cbasestr):
            val, ok = setting.uilocale.toDouble(val)
            if ok: return val
            raise ValueError("Invalid floating point number")
        return float(val)

    def uiDataItemToData(self, val):
        """Return val converted to data."""
        return float(val)

    def _getItemHelper(self, key):
        """Help get arguments to constructor."""
        args = {}
        for col in self.columns:
            array = getattr(self, col)
            if array is not None:
                args[col] = array[key]
        return args

    def __getitem__(self, key):
        """Return a dataset based on this dataset

        e.g. dataset[5:100] - make a dataset based on items 5 to 99 inclusive
        """
        return self.returnCopyWithNewData(**self._getItemHelper(key))

    def __len__(self):
        """Return length of dataset."""
        return len(self.data)

    def deleteRows(self, row, numrows):
        """Delete numrows rows starting from row.
        Returns deleted rows as a dict of {column:data, ...}
        """
        pass

    def insertRows(self, row, numrows, rowdata):
        """Insert numrows rows starting from row.
        rowdata is a dict of {column: data}.
        """
        pass

    def canUnlink(self):
        """Can dataset be unlinked?"""
        return self.linked is not None

    def linkedInformation(self):
        """Return information about any linking for the user."""
        if self.linked is None:
            return _('Linked file: None')
        else:
            return _('Linked file: %s') % self.linked.filename

    def returnCopy(self):
        """Return an unlinked copy of self."""
        pass

    def returnCopyWithNewData(self, **args):
        """Return copy with new data given."""
        pass

    def renameable(self):
        """Is it possible to rename this dataset?"""
        return self.linked is None

    def datasetAsText(self, fmt='%g', join='\t'):
        """Return dataset as text (for use by user)."""
        return ''
