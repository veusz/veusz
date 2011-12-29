# -*- coding: utf-8 -*-
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

"""Classes for linked files"""

from itertools import izip

import veusz.utils as utils

import operations
import simpleread

class LinkedFileBase(object):
    """A base class for linked files containing common routines."""

    # operation to recreate dataset
    create_operation = None

    def __init__(self, params):
        """Save parameters."""
        self.params = params

    @property
    def filename(self):
        """Get filename."""
        return self.params.filename

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""
        pass

    def _getSaveFilename(self, relpath):
        """Get filename to write to save file.
        If relpath is a string, write relative to path given
        """
        if relpath:
            return utils.relpath(self.params.filename, relpath)
        else:
            return self.filename

    def reloadLinks(self, document):
        """Reload datasets linked to this file."""
        pass

    def _deleteLinkedDatasets(self, document):
        """Delete linked datasets from document linking to self."""

        for name, ds in document.data.items():
            if ds.linked == self:
                document.deleteData(name)

    def _moveReadDatasets(self, tempdoc, document):
        """Move datasets from tempdoc to document if they do not exist
        in the destination."""

        read = []
        for name, ds in tempdoc.data.items():
            if name not in document.data:
                read.append(name)
                document.setData(name, ds)
                ds.document = document
                ds.linked = self
        return read

    def reloadLinks(self, document):
        """Reload links using an operation"""

        # get the operation for reloading
        op = self.create_operation(self.params)

        # load data into a temporary document
        tempdoc = document.__class__()

        try:
            tempdoc.applyOperation(op)
        except Exception, ex:
            # if something breaks, record an error and return nothing
            document.log(unicode(ex))

            # find datasets which are linked using this link object
            # return errors for them
            errors = dict([(name, 1) for name, ds in document.data.iteritems()
                           if ds.linked is self])
            return ([], errors)

        # delete datasets which are linked and imported here
        self._deleteLinkedDatasets(document)
        # move datasets into document
        read = self._moveReadDatasets(tempdoc, document)

        # return errors (if any)
        errors = op.outinvalids

        return (read, errors)

class LinkedFile(LinkedFileBase):
    """Instead of reading data from a string, data can be read from
    a "linked file". This means the same document can be reloaded, and
    the data would be reread from the file.

    This class is used to store a link filename with the descriptor
    """

    create_operation = operations.OperationDataImport

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file.
        If relpath is set, save links relative to path given
        """

        p = self.params
        params = [ repr(self._getSaveFilename(relpath)),
                   repr(p.descriptor),
                   "linked=True",
                   "ignoretext=" + repr(p.ignoretext) ]

        if p.encoding != "utf_8":
            params.append("encoding=" + repr(p.encoding))
        if p.useblocks:
            params.append("useblocks=True")
        if p.prefix:
            params.append("prefix=" + repr(p.prefix))
        if p.suffix:
            params.append("suffix=" + repr(p.suffix))

        fileobj.write("ImportFile(%s)\n" % (", ".join(params)))

class LinkedFile2D(LinkedFileBase):
    """Class representing a file linked to a 2d dataset."""

    create_operation = operations.OperationDataImport2D

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""

        args = [ repr(self._getSaveFilename(relpath)),
                 repr(self.params.datasetnames) ]
        for par in ("xrange", "yrange", "invertrows", "invertcols", "transpose",
                    "prefix", "suffix", "encoding"):
            v = getattr(self.params, par)
            if v is not None and v != "" and v != self.params.defaults[par]:
                args.append( "%s=%s" % (par, repr(v)) )
        args.append("linked=True")

        fileobj.write("ImportFile2D(%s)\n" % ", ".join(args))

class LinkedFileFITS(LinkedFileBase):
    """Links a FITS file to the data."""

    create_operation = operations.OperationDataImportFITS

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""

        p = self.params
        args = [p.dsname, self._getSaveFilename(relpath), p.hdu]
        args = [repr(i) for i in args]
        for param, column in ( ("datacol", p.datacol),
                               ("symerrcol", p.symerrcol),
                               ("poserrcol", p.poserrcol),
                               ("negerrcol", p.negerrcol) ):
            if column is not None:
                args.append("%s=%s" % (param, repr(column)))
        args.append("linked=True")

        fileobj.write("ImportFITSFile(%s)\n" % ", ".join(args))

class LinkedFileCSV(LinkedFileBase):
    """A CSV file linked to datasets."""

    create_operation = operations.OperationDataImportCSV

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""

        paramsout = [ repr(self._getSaveFilename(relpath)) ]

        # add parameters which aren"t defaults
        for param, default in sorted(self.params.defaults.items()):
            v = getattr(self.params, param)
            if param == 'prefix' or param == 'suffix':
                param = 'ds' + param
            if param != 'filename' and param != 'tags' and v != default:
                paramsout.append("%s=%s" % (param, repr(v)))

        fileobj.write("ImportFileCSV(%s)\n" % (", ".join(paramsout)))

class LinkedFilePlugin(LinkedFileBase):
    """Represent a file linked using an import plugin."""

    create_operation = operations.OperationDataImportPlugin

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the vsz document file."""

        p = self.params
        params = [repr(p.plugin),
                  repr(self._getSaveFilename(relpath)),
                  "linked=True"]
        if p.encoding != "utf_8":
            params.append("encoding=" + repr(p.encoding))
        if p.prefix:
            params.append("prefix=" + repr(p.prefix))
        if p.suffix:
            params.append("suffix=" + repr(p.suffix))
        for name, val in p.pluginpars.iteritems():
            params.append("%s=%s" % (name, repr(val)))

        fileobj.write("ImportFilePlugin(%s)\n" % (", ".join(params)))
