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
##############################################################################

from __future__ import division, print_function
from .. import qtall as qt
from .. import utils
from .. import document
from . import simpleread
from . import base

def _(text, disambiguation=None, context="Import_ND"):
    return qt.QCoreApplication.translate(context, text, disambiguation)

class ImportParamsND(base.ImportParamsBase):
    """nD import parameters.

     transpose: transpose array
     mode: text or csv
     csvdelimiter/csvtextdelimiter: csv text delimiters
     csvlocale: locale when importing csv
    """

    defaults = {
        'dataset': None,
        'datastr': None,
        'shape': None,
        'transpose': False,
        'mode': 'text',
        'csvdelimiter': ',',
        'csvtextdelimiter': '"',
        'csvlocale': 'en_US',
        }
    defaults.update(base.ImportParamsBase.defaults)

class LinkedFileND(base.LinkedFileBase):
    """Class representing a file linked to an nD dataset."""

    def createOperation(self):
        """Return operation to recreate self."""
        return OperationDataImportND

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""
        self._saveHelper(
            fileobj,
            'ImportFileND',
            ('filename', 'dataset'),
            relpath=relpath)

class OperationDataImportND(base.OperationDataImportBase):
    """Import an n-D matrix from a file."""

    descr = _('import nD data')

    def doImport(self):
        """Import data."""

        p = self.params

        # get stream
        if p.mode == 'csv':
            stream = simpleread.CSVStream(
                p.filename, p.csvdelimiter, p.csvtextdelimiter,
                p.csvlocale, p.encoding)
        elif p.filename is not None:
            stream = simpleread.FileStream(
                utils.openEncoding(p.filename, p.encoding) )
        elif p.datastr is not None:
            stream = simpleread.StringStream(p.datastr)
        else:
            raise RuntimeError("Invalid combination of parameters")

        # linked file
        LF = None
        if p.linked:
            assert p.filename
            LF = LinkedFileND(p)

        sr = simpleread.SimpleReadND(p.dataset, p)
        sr.readData(stream)
        sr.setOutput(self.outdatasets, linkedfile=LF)

def ImportFileND(
        comm, filename, dataset,
        shape=None,
        transpose=False,
        mode='text', csvdelimiter=',', csvtextdelimiter='"',
        csvlocale='en_US',
        prefix="", suffix="", encoding='utf_8',
        linked=False):

    """Import n-dimensional data from a file.
    filename is the name of the file to read
    dataset is the dataset to read

    if shape is set, the dataset is reshaped to these dimensions after loading
    if transpose=True, then rows and columns, etc, are swapped

    mode is either 'text' or 'csv'
    csvdelimiter is the csv delimiter for csv
    csvtextdelimiter is the csv text delimiter for csv
    csvlocale is locale to use when reading csv data

    prefix and suffix are prepended and appended to dataset names

    encoding is encoding character set

    if linked=True then the dataset is linked to the file

    Returns: list of imported datasets
    """

    # look up filename on path
    realfilename = comm.findFileOnImportPath(filename)

    params = ImportParamsND(
        dataset=dataset,
        filename=realfilename,
        transpose=transpose,
        mode=mode,
        csvdelimiter=csvdelimiter,
        csvtextdelimiter=csvtextdelimiter,
        csvlocale=csvlocale,
        prefix=prefix, suffix=suffix,
        linked=linked)
    op = OperationDataImportND(params)
    comm.document.applyOperation(op)

    if comm.verbose:
        print("Imported datasets %s" % ', '.join(op.outnames))
    return op.outnames

def ImportStringND(comm, dataset, dstring, shape=None, transpose=False):
    """Read n-dimensional data from the string specified.
    dataset is a dataset to read from the string

    if shape is set, then the array is reshaped to these dimensions
    if transpose=True, then rows and columns, etc, are swapped

    Returns: list of imported datasets
    """

    params = ImportParamsND(
        dataset=dataset,
        datastr=dstring,
        shape=shape,
        transpose=transpose)
    op = OperationDataImportND(params)
    comm.document.applyOperation(op)

    if comm.verbose:
        print("Imported datasets %s" % ', '.join(op.outnames))
    return op.outnames

document.registerImportCommand('ImportFileND', ImportFileND)
document.registerImportCommand(
    'ImportStringND', ImportStringND, filenamearg=-1)
