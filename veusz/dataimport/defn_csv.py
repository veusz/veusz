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
import re

from .. import qtall as qt4
from .. import document
from . import readcsv
from . import base

def _(text, disambiguation=None, context="Import_CSV"):
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class ImportParamsCSV(base.ImportParamsBase):
    """CSV import parameters.

    additional parameters:
     readrows: read data in rows
     delimiter: CSV delimiter
     skipwhitespace: whether to ignore white space following CSV delimiter
     textdelimiter: delimiter for text
     headerignore: number of lines to ignore after headers
     rowsignore: number of lines to ignore at top of file
     blanksaredata: treat blank entries as nans
     numericlocale: name of local for numbers
     dateformat: date format string
     headermode: 'multi', '1st' or 'none'
    """

    defaults = {
        'readrows': False,
        'delimiter': ',',
        'skipwhitespace' : False,
        'textdelimiter': '"',
        'headerignore': 0,
        'rowsignore': 0,
        'blanksaredata': False,
        'numericlocale': 'en_US',
        'dateformat': 'YYYY-MM-DD|T|hh:mm:ss',
        'headermode': 'multi',
        }
    defaults.update(base.ImportParamsBase.defaults)

    def __init__(self, **argsv):
        base.ImportParamsBase.__init__(self, **argsv)
        if self.headermode not in ('multi', '1st', 'none'):
            raise ValueError("Invalid headermode")

class OperationDataImportCSV(base.OperationDataImportBase):
    """Import data from a CSV file."""

    descr = _('import CSV data')

    def doImport(self):
        """Do the data import."""

        try:
            csvr = readcsv.ReadCSV(self.params)
        except re.error:
            # invalid date RE
            raise base.ImportingError(_('Invalid date regular expression'))

        csvr.readData()

        LF = None
        if self.params.linked:
            LF = LinkedFileCSV(self.params)

        # set the data in the output structure
        csvr.setData(self.outdatasets, linkedfile=LF)

class LinkedFileCSV(base.LinkedFileBase):
    """A CSV file linked to datasets."""

    def createOperation(self):
        """Return operation to recreate self."""
        return OperationDataImportCSV

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""
        self._saveHelper(
            fileobj,
            'ImportFileCSV',
            ('filename',),
            renameparams={'prefix': 'dsprefix', 'suffix': 'dssuffix'},
            relpath=relpath)

def ImportFileCSV(comm, filename,
                  readrows=False,
                  delimiter=',', skipwhitespace=False, textdelimiter='"',
                  encoding='utf_8',
                  headerignore=0, rowsignore=0,
                  blanksaredata=False,
                  numericlocale='en_US',
                  dateformat='YYYY-MM-DD|T|hh:mm:ss',
                  headermode='multi',
                  dsprefix='', dssuffix='', prefix=None,
                  renames=None,
                  linked=False):
    """Read data from a comma separated file (CSV).

    Data are read from filename

    readrows: if true, data are read across rather than down
    delimiter: character for delimiting data (usually ',')
    skipwhitespace: if true, white space following delimiter is ignored
    textdelimiter: character surrounding text (usually '"')
    encoding: encoding used in file
    headerignore: number of lines to ignore after header text
    rowsignore: number of rows to ignore at top of file
    blanksaredata: treats blank lines in csv files as blank data values
    numericlocale: format to use for reading numbers
    dateformat: format for interpreting dates
    headermode: 'multi': multiple headers allowed in file
                '1st': first text found are headers
                'none': no headers, guess data and use default names

    Dataset names are prepended and appended, by dsprefix and dssuffix,
    respectively
     (prefix is backware compatibility only, it adds an underscore
      relative to dsprefix)

    renames is a map of old names to new names to rename on import

    If linked is True the data are linked with the file.

    Returns: list of imported datasets
    """

    # backward compatibility
    if prefix:
        dsprefix = prefix + '_'

    # lookup filename
    realfilename = comm.findFileOnImportPath(filename)

    params = ImportParamsCSV(
        filename=realfilename, readrows=readrows,
        delimiter=delimiter, skipwhitespace=skipwhitespace, 
        textdelimiter=textdelimiter,
        encoding=encoding,
        headerignore=headerignore, rowsignore=rowsignore,
        blanksaredata=blanksaredata,
        numericlocale=numericlocale, dateformat=dateformat,
        headermode=headermode,
        prefix=dsprefix, suffix=dssuffix,
        renames=renames,
        linked=linked,
        )
    op = OperationDataImportCSV(params)
    comm.document.applyOperation(op)

    if comm.verbose:
        print("Imported datasets %s" % ' '.join(op.outnames))
    return op.outnames

document.registerImportCommand('ImportFileCSV', ImportFileCSV)
