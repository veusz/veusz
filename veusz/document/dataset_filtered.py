#    Copyright (C) 2015 Jeremy S. Sanders
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

from __future__ import division, print_function
import numpy as N

from ..compat import citems, czip
from .. import qtall as qt4
from .datasets import Dataset, DatasetBase, evalDatasetExpression, _

class DatasetFilterGenerator(object):
    """This object is shared by all DatasetFiltered datasets, to calculate
    the filter expression."""

    def __init__(self, inexpr, indatasets,
                 prefix="", suffix="",
                 invert=False, replacenans=False):
        """
        inexpr = filter expression
        indatasets = list of input datasets
        prefix = output prefix
        suffix = output suffix
        invert = invert filter
        replacenans = replace filtered values by nans
        """

        self.changeset = -1
        self.inexpr = inexpr
        self.indatasets = indatasets
        self.prefix = prefix
        self.suffix = suffix
        self.invert = invert
        self.replacenans = replacenans

        self.outdatasets = {}

    def filterNumeric(self, ds, filterarr):
        """Filter a numeric dataset."""
        outdata = {}
        minlen = len(filterarr)
        for attr in ds.columns:
            data = getattr(ds, attr)
            if data is None:
                filtered = None
            else:
                filtered = N.array(data[:minlen])
                if self.replacenans:
                    filtered[N.logical_not(filterarr)] = N.nan
                else:
                    filtered = filtered[filterarr]
            outdata[attr] = filtered
        return ds.returnCopyWithNewData(**outdata)

    def filterText(self, ds, filterarr):
        """Filter a text dataset."""
        if self.replacenans:
            filtered = [(d if f else "")
                        for f, d in czip(filterarr, data)]
        else:
            filtered = [d for f, d in czip(filterarr, data) if f]
        return ds.returnCopyWithNewData(data=filtered)

    def checkUpdate(self, doc):
        """Update filtering calculation if doc changed."""

        if doc.changeset == self.changeset:
            return
        self.changeset = doc.changeset

        # this is populated by output
        self.outdatasets = {}

        # evaluate filter expression
        d = evalDatasetExpression(doc, self.inexpr)
        if d is None:
            return
        if d.dimensions != 1:
            doc.log(
                _("Invalid number of dimensions in filter expression '%s'\n") %
                self.inexpr)
            return
        if d.datatype != "numeric":
            doc.log(
                _("Input filter expression non-numeric: '%s'\n") % self.inexpr)
            return

        filterarr = d.data.astype(N.bool)
        if self.invert:
            filterarr = N.logical_not(filterarr)

        # do filtering of datasets
        for name in self.indatasets:
            ds = doc.data.get(name)
            if ds is None or ds.dimensions != 1:
                doc.log(_("Filtered dataset '%s' has more than 1 dimension\n") %
                        name)
                continue
            minlen = min(len(ds.data), len(filterarr))
            filterarrchop = filterarr[:minlen]

            if ds.datatype == "numeric":
                filtered = self.filterNumeric(ds, filterarrchop)
            elif ds.datatype == "text":
                filtered = self.filterText(ds, filterarrchop)
            else:
                doc.log(_("Could not filter dataset '%s'\n") % name)
                continue

            self.outdatasets[name] = filtered

    def saveToFile(self, doc, fileobj):
        """Save datasets to file."""

        # find current datasets in document which use this generator
        # (some may have been deleted)
        names = []
        for name, ds in sorted(doc.data.items()):
            if getattr(ds, "generator", None) is self:
                names.append(ds.namein)

        args = [
            repr(self.inexpr),
            repr(names),
        ]
        if self.prefix:
            args.append("prefix="+repr(self.prefix))
        if self.suffix:
            args.append("suffix="+repr(self.suffix))
        if self.invert:
            args.append("invert=True")
        if self.replacenans:
            args.append("replacenans=True")

        fileobj.write("FilterDatasets(%s)\n" % ", ".join(args))

class DatasetFiltered(DatasetBase):

    dstype = 'Filtered'
    dimensions = 1
    editable = False

    def __init__(self, gen, name, doc):
        DatasetBase.__init__(self)
        self.generator = gen
        self.namein = name
        self.document = doc
        self.changeset = -1
        self._internalds = Dataset(data=[])

    def name(self):
        """Lookup name."""
        for name, ds in citems(self.document.data):
            if ds is self:
                return name
        raise ValueError('Could not find self in document.data')

    def _checkUpdate(self):
        """Recalculate if document has changed."""
        if self.document.changeset != self.changeset:
            self.generator.checkUpdate(self.document)
            self.changeset = self.document.changeset

            ds = self.generator.outdatasets.get(self.namein)
            self._internalds = Dataset(data=[]) if ds is None else ds

    def saveDataRelationToText(self, fileobj, name):
        """Save plugin to file, if this is the first one."""

        # Am I the first dataset in the document with this generator?
        am1st = False
        for ds in sorted(self.document.data):
            data = self.document.data[ds]
            if data is self:
                am1st = True
                break
            elif getattr(data, "generator", None) is self.generator:
                # not 1st
                break
        if am1st:
            self.generator.saveToFile(self.document, fileobj)

    def __getattr__(self, attr):
        """Lookup attribute from internal dataset."""
        return getattr(self._internalds, attr)

# class _DatasetFiltered(object):
#     """Shared methods for filtered datasets."""

#     dstype = _("Filtered")

#     def __init__(self, gen, name):
#         self.generator = gen
#         self.namein = name

#     def getFilteredData(self, attr):


#     def linkedInformation(self):
#         return _("Dataset '%s' filtered using '%s'") % (
#             self.namein, self.generator.inexpr)

#     def canUnlink(self):
#         """Can relationship be unlinked?"""
#         return True

#     def deleteRows(self, row, numrows):
#         pass

#     def insertRows(self, row, numrows, rowdata):
#         pass

#     def saveDataRelationToText(self, fileobj, name):
#         """Save plugin to file, if this is the first one."""

#         # Am I the first dataset in the document with this generator?
#         am1st = False
#         for ds in sorted(self.document.data):
#             data = self.document.data[ds]
#             if data is self:
#                 am1st = True
#                 break
#             elif getattr(data, "generator", None) is self.generator:
#                 # not 1st
#                 break
#         if am1st:
#             self.generator.saveToFile(self.document, fileobj)

#     def saveDataDumpToText(self, fileobj, name):
#         """Save data to text: not used."""

#     def saveDataDumpToHDF5(self, group, name):
#         """Save data to HDF5: not used."""

# class Dataset1DFiltered(_DatasetFiltered, Dataset1DBase):
#     """Return 1D dataset from a plugin."""

#     def __init__(self, gen, name):
#         _DatasetFiltered.__init__(self, gen, name)
#         Dataset1DBase.__init__(self)

#     def userSize(self):
#         """Size of dataset."""
#         return str( self.data.shape[0] )

#     def __getitem__(self, key):
#         """Return a dataset based on this dataset

#         We override this from DatasetBase as it would return a
#         DatsetExpression otherwise, not chopped sets of data.
#         """
#         return Dataset(**self._getItemHelper(key))

#     # parent class sets these attributes, so override setattr to do nothing
#     data = property( lambda self: self.getFilteredData('data'),
#                      lambda self, val: None )
#     serr = property( lambda self: self.getFilteredData('serr'),
#                      lambda self, val: None )
#     nerr = property( lambda self: self.getFilteredData('nerr'),
#                      lambda self, val: None )
#     perr = property( lambda self: self.getFilteredData('perr'),
#                      lambda self, val: None )

# class DatasetTextFiltered(_DatasetFiltered, DatasetText):
#     """Return text dataset from a plugin."""

#     def __init__(self, gen, name):
#         _DatasetFiltered.__init__(self, gen, name)
#         DatasetText.__init__(self, [])

#     def __getitem__(self, key):
#         return DatasetText(self.data[key])

#     data = property( lambda self: self.getFilteredData('data'),
#                      lambda self, val: None )

# class DatasetDateTimeFiltered(_DatasetFiltered, DatasetDateTimeBase):
#     """Return date dataset from plugin."""

#     def __init__(self, gen, name):
#         _DatasetFiltered.__init__(self, gen, name)
#         DatasetDateTimeBase.__init__(self)
#         self.serr = self.perr = self.nerr = None

#     def __getitem__(self, key):
#         return DatasetDateTime(self.data[key])

#     data = property( lambda self: self.getFilteredData('data'),
#                      lambda self, val: None )



# class DatasetFiltered(DatasetBase):
#     """A dataset filtered by an expression."""

#     dimensions = 1

#     def __init__(self, generator, namein, doc):
#         DatasetBase.__init__(self)
#         self.generator = generator
#         self.document = doc
#         self.namein = namein
#         self.linked = None
#         self.changeset = -1

#         self._data = self._datatype = self._displaytype = None
#         self._columns = self._column_descriptions = None

#     def _checkUpdate(self):
#         """Recalculate if document has changed."""
#         if self.document.changeset != self.changeset:
#             self.generator.checkUpdate(self.document)
#             self.changeset = self.document.changeset

#             data = self.generator.outdatasets.get(self.namein)
#             if data is None:
#                 self._data = {"data": N.array([], dtype=N.float64)}
#                 self._datatype = Dataset1DBase.datatype
#                 self._displaytype = Dataset1DBase.displaytype
#                 self._columns = Dataset1DBase.columns
#                 self._column_descriptions = Dataset1DBase.column_descriptions
#             else:
#                 (self._datatype, self._displaytype,
#                  self._columns, self._column_descriptions,
#                  self._data) = data

#     def linkedInformation(self):
#         return _("Dataset '%s' filtered using '%s'") % (
#             self.namein, self.generator.inexpr)

#     def userSize(self):
#         return str(len(self.data))

#     def saveDataRelationToText(self, fileobj, name):
#         """Save in output file."""

#         # Am I the first dataset in the document with this generator?
#         am1st = False
#         for ds in sorted(self.document.data):
#             data = self.document.data[ds]
#             if data is self:
#                 am1st = True
#                 break
#             elif getattr(data, "generator", None) is self.generator:
#                 # not 1st
#                 break
#         if am1st:
#             self.generator.saveToFile(self.document, fileobj)

#     # descriptors
#     @property
#     def datatype(self):
#         self._checkUpdate()
#         return self._datatype
#     @property
#     def displaytype(self):
#         self._checkUpdate()
#         return self._displaytype
#     @property
#     def columns(self):
#         self._checkUpdate()
#         return self._columns
#     @property
#     def column_descriptions(self):
#         self._checkUpdate()
#         return self._column_descriptions

#     # data parts
#     @property
#     def data(self):
#         self._checkUpdate()
#         return self._data.get("data")
#     @property
#     def serr(self):
#         self._checkUpdate()
#         return self._data.get("serr")
#     @property
#     def perr(self):
#         self._checkUpdate()
#         return self._data.get("perr")
#     @property
#     def nerr(self):
#         self._checkUpdate()
#         return self._data.get("nerr")

class OperationDatasetFilter(object):
    """Operation to filter datasets."""

    descr = _("filter datasets")

    def __init__(self, inexpr, indatasets,
                 prefix="", suffix="",
                 invert=False, replacenans=False):
        assert prefix != "" or suffix != ""
        self.inexpr = inexpr
        self.indatasets = indatasets
        self.prefix = prefix
        self.suffix = suffix
        self.invert = invert
        self.replacenans = replacenans

    def do(self, doc):
        """Do the operation."""

        gen = DatasetFilterGenerator(
            self.inexpr, self.indatasets,
            prefix=self.prefix, suffix=self.suffix,
            invert=self.invert, replacenans=self.replacenans)

        self.olddatasets = {}
        gen.checkUpdate(doc)

        for name in self.indatasets:
            outname = self.prefix + name + self.suffix
            self.olddatasets[outname] = doc.data.get(outname)
            doc.setData(outname, DatasetFiltered(gen, name, doc))

        return gen.outdatasets

    def undo(self, doc):
        """Undo operation."""

        for name, val in citems(self.olddatasets):
            if val is None:
                doc.deleteData(name)
            else:
                doc.setData(name, val)
