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

    def _reloadViaOperation(self, document, op):
        """Reload links using a supplied operation, op."""
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

        # return zero errors
        errors = dict( [(ds, 0) for ds in read] )

        return (read, errors)

class LinkedFile(LinkedFileBase):
    """Instead of reading data from a string, data can be read from
    a "linked file". This means the same document can be reloaded, and
    the data would be reread from the file.

    This class is used to store a link filename with the descriptor
    """

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file.
        If relpath is set, save links relative to path given
        """

        p = self.params
        params = [ repr(self._getSaveFilename(relpath)),
                   repr(p.descriptor),
                   "linked=True",
                   "ignoretext=" + repr(self.ignoretext) ]

        if p.encoding != "utf_8":
            params.append("encoding=" + repr(p.encoding))
        if p.useblocks:
            params.append("useblocks=True")
        if p.prefix:
            params.append("prefix=" + repr(p.prefix))
        if p.suffix:
            params.append("suffix=" + repr(p.suffix))

        fileobj.write("ImportFile(%s)\n" % (", ".join(params)))

    def reloadLinks(self, document):
        """Reload datasets linked to this file.

        Returns a tuple of
        - List of datasets read
        - Dict of tuples containing dataset names and number of errors
        """

        # a bit clumsy, but we need to load this into a separate document
        # to make sure we do not overwrited non-linked data (which may
        # be specified in the descriptor)

        tempdoc = document.__class__()
        sr = simpleread.SimpleRead(self.descriptor)

        stream = simpleread.FileStream(
            utils.openEncoding(self.filename, self.encoding))

        sr.readData(stream,
                    useblocks=self.useblocks,
                    ignoretext=self.ignoretext)
        sr.setInDocument(tempdoc, linkedfile=self,
                         prefix=self.prefix, suffix=self.suffix)

        errors = sr.getInvalidConversions()

        self._deleteLinkedDatasets(document)
        read = self._moveReadDatasets(tempdoc, document)

        # returns list of datasets read, and a dict of variables with number
        # of errors
        return (read, errors)

class LinkedFile2D(LinkedFileBase):
    """Class representing a file linked to a 2d dataset."""

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""

        args = [ repr(self._getSaveFilename(relpath)),
                 repr(self.params.datasets) ]
        for p in ("xrange", "yrange", "invertrows", "invertcols", "transpose",
                  "prefix", "suffix", "encoding"):
            v = getattr(self.params, p)
            if (v is not None) and (v != ""):
                args.append( "%s=%s" % (p, repr(v)) )
        args.append("linked=True")

        fileobj.write("ImportFile2D(%s)\n" % ", ".join(args))

    def reloadLinks(self, document):
        """Reload datasets linked to this file."""

        op = operations.OperationDataImport2D(self.params)
        return self._reloadViaOperation(document, op)

class LinkedFileFITS(LinkedFileBase):
    """Links a FITS file to the data."""

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""

        args = [self.dsname, self._getSaveFilename(relpath), self.hdu]
        args = [repr(i) for i in args]
        for c, a in izip(self.columns,
                         ("datacol", "symerrcol",
                          "poserrcol", "negerrcol")):
            if c is not None:
                args.append("%s=%s" % (a, repr(c)))
        args.append("linked=True")

        fileobj.write("ImportFITSFile(%s)\n" % ", ".join(args))

    def reloadLinks(self, document):
        """Reload any linked data from the CSV file."""

        op = operations.OperationDataImportFITS(self.params)
        return self._reloadViaOperation(document, op)

class LinkedFileCSV(LinkedFileBase):
    """A CSV file linked to datasets."""

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""

        paramsout = [repr(self._getSaveFilename(relpath)),
                     "linked=True"]

        # add parameters which aren"t defaults
        for param, default in sorted(self.params.defaults.items()):
            v = getattr(self.params, param)
            if v != default:
                paramsout.append("%s=%s" % (param, repr(v)))

        fileobj.write("ImportFileCSV(%s)\n" % (", ".join(paramsout)))

    def reloadLinks(self, document):
        """Reload any linked data from the CSV file."""

        op = operations.OperationDataImportCSV(self.params)
        return self._reloadViaOperation(document, op)

class LinkedFilePlugin(LinkedFileBase):
    """Represent a file linked using an import plugin."""

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
        for name, val in p.pluginparams.iteritems():
            params.append("%s=%s" % (name, repr(val)))

        fileobj.write("ImportFilePlugin(%s)\n" % (", ".join(params)))

    def reloadLinks(self, document):
        """Reload data from file."""

        op = operations.OperationDataImportPlugin(self.params)
        return self._reloadViaOperation(document, op)
