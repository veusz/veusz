#    Copyright (C) 2013 Jeremy S. Sanders
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

from __future__ import division, print_function
from ..compat import citems, cstr, crepr, cbasestr
from .. import qtall as qt4
from .. import utils
from .. import document
from . import simpleread
from . import base

def _(text, disambiguation=None, context="Import_2D"):
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class ImportParams2D(base.ImportParamsBase):
    """2D import parameters.

    additional parameters:
     datastr: text to read from instead of file
     xrange: tuple with range of x data coordinates
     yrange: tuple with range of y data coordinates
     invertrows: invert rows when reading
     invertcols: invert columns when reading
     transpose: swap rows and columns
    """

    defaults = {
        'datasetnames': None,
        'datastr': None,
        'xrange': None,
        'yrange': None,
        'invertrows': False,
        'invertcols': False,
        'transpose': False,
        'gridatedge': False,
        }
    defaults.update(base.ImportParamsBase.defaults)

class LinkedFile2D(base.LinkedFileBase):
    """Class representing a file linked to a 2d dataset."""

    def createOperation(self):
        """Return operation to recreate self."""
        return OperationDataImport2D

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""

        args = [ crepr(self._getSaveFilename(relpath)),
                 crepr(self.params.datasetnames) ]
        for par in ("xrange", "yrange", "invertrows", "invertcols", "transpose",
                    "prefix", "suffix", "encoding"):
            v = getattr(self.params, par)
            if v is not None and v != "" and v != self.params.defaults[par]:
                args.append( "%s=%s" % (par, crepr(v)) )

        args.append("linked=True")

        fileobj.write("ImportFile2D(%s)\n" % ", ".join(args))

class OperationDataImport2D(base.OperationDataImportBase):
    """Import a 2D matrix from a file."""
    
    descr = _('import 2d data')

    def doImport(self, document):
        """Import data."""

        p = self.params

        # get stream
        if p.filename is not None:
            stream = simpleread.FileStream(
                utils.openEncoding(p.filename, p.encoding) )
        elif p.datastr is not None:
            stream = simpleread.StringStream(p.datastr)
        else:
            assert False

        # linked file
        LF = None
        if p.linked:
            assert p.filename
            LF = LinkedFile2D(p)

        for name in p.datasetnames:
            sr = simpleread.SimpleRead2D(name, p)
            sr.readData(stream)
            self.outdatasets += sr.setInDocument(document, linkedfile=LF)

def ImportFile2D(comm, filename, datasetnames, xrange=None, yrange=None,
                 invertrows=None, invertcols=None, transpose=None,
                 prefix="", suffix="", encoding='utf_8',
                 linked=False):
    """Import two-dimensional data from a file.
    filename is the name of the file to read
    datasetnames is a list of datasets to read from the file, or a single
    dataset name

    xrange is a tuple containing the range of data in x coordinates
    yrange is a tuple containing the range of data in y coordinates
    if invertrows=True, then rows are inverted when read
    if invertcols=True, then cols are inverted when read
    if transpose=True, then rows and columns are swapped

    prefix and suffix are prepended and appended to dataset names

    encoding is encoding character set

    if linked=True then the dataset is linked to the file
    """

    # look up filename on path
    realfilename = comm.findFileOnImportPath(filename)

    if isinstance(datasetnames, cbasestr):
        datasetnames = [datasetnames]

    params = ImportParams2D(
        datasetnames=datasetnames, 
        filename=realfilename, xrange=xrange,
        yrange=yrange, invertrows=invertrows,
        invertcols=invertcols, transpose=transpose,
        prefix=prefix, suffix=suffix,
        linked=linked)
    op = OperationDataImport2D(params)
    comm.document.applyOperation(op)
    if comm.verbose:
        print("Imported datasets %s" % (', '.join(datasetnames)))

def ImportString2D(comm, datasetnames, dstring, xrange=None, yrange=None,
                   invertrows=None, invertcols=None, transpose=None):
    """Read two dimensional data from the string specified.
    datasetnames is a list of datasets to read from the string or a single
    dataset name

    xrange is a tuple containing the range of data in x coordinates
    yrange is a tuple containing the range of data in y coordinates
    if invertrows=True, then rows are inverted when read
    if invertcols=True, then cols are inverted when read
    if transpose=True, then rows and columns are swapped
    """

    if isinstance(datasetnames, cbasestr):
        datasetnames = [datasetnames]

    params = ImportParams2D(
        datasetnames=datasetnames,
        datastr=dstring, xrange=xrange,
        yrange=yrange, invertrows=invertrows,
        invertcols=invertcols, transpose=transpose)
    op = OperationDataImport2D(params)
    comm.document.applyOperation(op)
    if comm.verbose:
        print("Imported datasets %s" % (', '.join(datasetnames)))

document.registerImportCommand('ImportFile2D', ImportFile2D)
document.registerImportCommand('ImportString2D', ImportString2D)
