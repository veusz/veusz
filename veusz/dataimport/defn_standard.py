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
from ..compat import citems
from .. import qtall as qt4
from .. import utils
from .. import document
from . import simpleread
from . import base

def _(text, disambiguation=None, context="Import_Standard"):
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class ImportParamsSimple(base.ImportParamsBase):
    """simpleread import parameters.

    additional parameters:
     descriptor: data descriptor
     useblocks: read datasets as blocks
     datastr: text to read from instead of file
     ignoretext: whether to ignore lines of text
    """

    defaults = {
        'descriptor': '',
        'useblocks': False,
        'datastr': None,
        'ignoretext': False,
        }
    defaults.update(base.ImportParamsBase.defaults)

class LinkedFile(base.LinkedFileBase):
    """Instead of reading data from a string, data can be read from
    a "linked file". This means the same document can be reloaded, and
    the data would be reread from the file.

    This class is used to store a link filename with the descriptor
    """

    def createOperation(self):
        """Return operation to recreate self."""
        return OperationDataImport

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file.
        If relpath is set, save links relative to path given
        """
        self._saveHelper(
            fileobj,
            'ImportFile',
            ('filename', 'descriptor'),
            relpath=relpath)

class OperationDataImport(base.OperationDataImportBase):
    """Import 1D data from text files."""

    descr = _('import data')

    def __init__(self, params):
        """Setup operation.
        """

        base.OperationDataImportBase.__init__(self, params)
        self.simpleread = simpleread.SimpleRead(params.descriptor)

    def doImport(self):
        """Import data.

        Returns a list of datasets which were imported.
        """

        p = self.params
        # open stream to import data from
        if p.filename is not None:
            stream = simpleread.FileStream(
                utils.openEncoding(p.filename, p.encoding))
        elif p.datastr is not None:
            stream = simpleread.StringStream(p.datastr)
        else:
            raise RuntimeError("No filename or string")

        # do the import
        self.simpleread.clearState()
        self.simpleread.readData(stream, useblocks=p.useblocks,
                                 ignoretext=p.ignoretext)

        # associate linked file
        LF = None
        if p.linked:
            assert p.filename
            LF = LinkedFile(p)

        # actually set the data in the document
        self.simpleread.setOutput(
            self.outdatasets,
            linkedfile=LF, prefix=p.prefix, suffix=p.suffix)
        self.outinvalids = self.simpleread.getInvalidConversions()

def ImportFile(comm, filename, descriptor, useblocks=False, linked=False,
               prefix='', suffix='', ignoretext=False, encoding='utf_8',
               renames=None):
    """Read data from file with filename using descriptor.
    If linked is True, the data won't be saved in a saved document,
    the data will be reread from the file.

    If useblocks is set, then blank lines or the word 'no' are used
    to split the data into blocks. Dataset names are appended with an
    underscore and the block number (starting from 1).

    If prefix is set, prefix is prepended to each dataset name
    Suffix is added to each dataset name
    ignoretext ignores lines of text in the file

    encoding is name of text file encoding
    renames is a dict mapping existing to new names after import

    Returned is a tuple (datasets, errors)
     where datasets is a list of datasets read
     errors is a dict of the datasets with the number of errors while
     converting the data
    """

    realfilename = comm.findFileOnImportPath(filename)

    params = ImportParamsSimple(
        descriptor=descriptor, filename=realfilename,
        useblocks=useblocks, linked=linked,
        prefix=prefix, suffix=suffix,
        ignoretext=ignoretext,
        encoding=encoding,
        renames=renames)
    op = OperationDataImport(params)
    comm.document.applyOperation(op)

    if comm.verbose:
        print("Imported datasets %s" % ' '.join(op.outnames))
        for name, num in citems(op.outinvalids):
            print("%i errors encountered reading dataset %s" % (num, name))

    return (op.outnames, op.outinvalids)

def ImportString(comm, descriptor, dstring, useblocks=False):
    """Read data from the string using a descriptor.

    If useblocks is set, then blank lines or the word 'no' are used
    to split the data into blocks. Dataset names are appended with an
    underscore and the block number (starting from 1).

    Returned is a tuple (datasets, errors)
     where datasets is a list of datasets read
     errors is a dict of the datasets with the number of errors while
     converting the data
    """

    params = ImportParamsSimple(
        descriptor=descriptor,
        datastr=dstring,
        useblocks=useblocks)
    op = OperationDataImport(params)
    comm.document.applyOperation(op)

    if comm.verbose:
        print("Imported datasets %s" % ' '.join(op.outnames))
        for name, num in citems(op.outinvalids):
            print("%i errors encountered reading dataset %s" % (num, name))

    return (op.outnames, op.outinvalids)

document.registerImportCommand('ImportFile', ImportFile)
document.registerImportCommand('ImportString', ImportString, filenamearg=-1)
