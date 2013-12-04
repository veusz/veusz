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
from ...compat import citems, cstr, crepr
from ...document import (LinkedFileBase, OperationDataImportBase, Dataset,
                         Dataset2D, ImportParamsBase,
                         registerImportCommand)
from . import readcsv

class ImportParamsCSV(ImportParamsBase):
    """CSV import parameters.

    additional parameters:
     readrows: readdata in rows
     delimiter: CSV delimiter
     textdelimiter: delimiter for text
     headerignore: number of lines to ignore after headers
     rowsignore: number of lines to ignore at top fo file
     blanksaredata: treat blank entries as nans
     numericlocale: name of local for numbers
     dateformat: date format string
     headermode: 'multi', '1st' or 'none'
    """

    defaults = {
        'readrows': False,
        'delimiter': ',',
        'textdelimiter': '"',
        'headerignore': 0,
        'rowsignore': 0,
        'blanksaredata': False,
        'numericlocale': 'en_US',
        'dateformat': 'YYYY-MM-DD|T|hh:mm:ss',
        'headermode': 'multi',
        }
    defaults.update(ImportParamsBase.defaults)

    def __init__(self, **argsv):
        ImportParamsBase.__init__(self, **argsv)
        if self.headermode not in ('multi', '1st', 'none'):
            raise ValueError("Invalid headermode")

class OperationDataImportCSV(OperationDataImportBase):
    """Import data from a CSV file."""

    descr = _('import CSV data')

    def doImport(self, document):
        """Do the data import."""
        
        csvr = readcsv.ReadCSV(self.params)
        csvr.readData()

        LF = None
        if self.params.linked:
            LF = linked.LinkedFileCSV(self.params)
        
        # set the data
        self.outdatasets = csvr.setData(document, linkedfile=LF)

class LinkedFileCSV(LinkedFileBase):
    """A CSV file linked to datasets."""

    def createOperation(self):
        """Return operation to recreate self."""
        return OperationDataImportCSV

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""

        paramsout = [ crepr(self._getSaveFilename(relpath)) ]

        # add parameters which aren"t defaults
        for param, default in sorted(citems(self.params.defaults)):
            v = getattr(self.params, param)
            if param == 'prefix' or param == 'suffix':
                param = 'ds' + param
            if param != 'filename' and param != 'tags' and v != default:
                paramsout.append("%s=%s" % (param, crepr(v)))

        fileobj.write("ImportFileCSV(%s)\n" % (", ".join(paramsout)))

def ImportFileCSV(comm, filename,
                  readrows=False,
                  delimiter=',', textdelimiter='"',
                  encoding='utf_8',
                  headerignore=0, rowsignore=0,
                  blanksaredata=False,
                  numericlocale='en_US',
                  dateformat='YYYY-MM-DD|T|hh:mm:ss',
                  headermode='multi',
                  dsprefix='', dssuffix='', prefix=None,
                  linked=False):
    """Read data from a comma separated file (CSV).

    Data are read from filename

    readrows: if true, data are read across rather than down
    delimiter: character for delimiting data (usually ',')
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

    If linked is True the data are linked with the file."""

    # backward compatibility
    if prefix:
        dsprefix = prefix + '_'

    # lookup filename
    realfilename = comm.findFileOnImportPath(filename)

    params = ImportParamsCSV(
        filename=realfilename, readrows=readrows,
        delimiter=delimiter, textdelimiter=textdelimiter,
        encoding=encoding,
        headerignore=headerignore, rowsignore=rowsignore,
        blanksaredata=blanksaredata,
        numericlocale=numericlocale, dateformat=dateformat,
        headermode=headermode,
        prefix=dsprefix, suffix=dssuffix,
        linked=linked,
        )
    op = OperationDataImportCSV(params)
    comm.document.applyOperation(op)

    if comm.verbose:
        print("Imported datasets %s" % (' '.join(op.outdatasets),))

    return op.outdatasets

registerImportCommand('ImportFileCSV', ImportFileCSV)
