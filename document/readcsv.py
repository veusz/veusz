#    Copyright (C) 2006 Jeremy S. Sanders
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

"""This module contains routines for importing CSV data files
in an easy-to-use manner."""

from collections import defaultdict
import numpy as N

import datasets
import veusz.utils as utils

class _FileReaderCols(object):
    """Read a CSV file in rows. This acts as an iterator.

    This is a very simple wrapper around the csv module
    """

    def __init__(self, csvreader):
        self.csvreader = csvreader

    def next(self):
        """Return next row."""
        return self.csvreader.next()

class _FileReaderRows(object):
    """Read a CSV file in columns. This acts as an iterator.

    This means we have to read the whole file in, then return cols :-(
    """

    def __init__(self, csvreader):
        self.data = []
        self.maxlength = 0

        for line in csvreader:
            self.maxlength = max(self.maxlength, len(line))
            self.data.append(line)

        self.counter = 0

    def next(self):
        """Return the next column."""

        if self.counter == self.maxlength:
            raise StopIteration

        # probably is a better way to do this
        retn = []
        for row in self.data:
            if self.counter >= len(row):
                retn.append('')
            else:
                retn.append(row[self.counter])

        self.counter += 1
        return retn

# list of codes which can be added to column descriptors
typecodes = (
    ('(string)', 'string'),
    ('(text)', 'string'),
    ('(date)', 'date'),
    ('(time)', 'date'),
    ('(float)', 'float'),
    ('(numeric)', 'float'),
    ('(number)', 'float'),
    )

class ReadCSV(object):
    """A class to import data from CSV files."""

    def __init__(self, filename, readrows=False, 
                 delimiter=',', textdelimiter='"',
                 encoding='utf_8',
                 headerignore=0, blanksaredata=False,
                 prefix='', suffix=''):
        """Initialise the reader to import data from filename.

        If readrows is True, then data are read from columns, rather than
        rows

        headerignore is number of lines to ignore after headers
        if blanksaredata is true, treat blank entries as nans

        prefix is a prefix to prepend to the name of datasets from this file
        """

        self.filename = filename
        self.readrows = readrows
        self.delimiter = delimiter
        self.textdelimiter = textdelimiter
        self.encoding = encoding
        self.headerignore = headerignore
        self.blanksaredata = blanksaredata
        self.prefix = prefix
        self.suffix = suffix

        # datasets. Each name is associated with a list
        self.data = {}

    def _generateName(self, column):
        """Generate a name for a column."""
        if self.readrows:
            prefix = 'row'
        else:
            prefix = 'col'

        name = '%s%s%i%s' % (self.prefix, prefix, column+1, self.suffix)
        return name

    def _getNameAndColType(self, colnum, colval):
        """Get column name and type."""

        name = colval.strip()
        if name in ('+', '-', '+-'):
            # loop to find previous valid column
            prevcol = colnum - 1
            while prevcol >= 0:
                n = self.colnames[prevcol]
                if len(n) > 0 and n[-1] not in "+-":
                    # we add a \0 here so that there's no chance of the user
                    # using this as a column name
                    name = n + '\0' + name
                    return self.coltypes[prevcol], name
                prevcol -= 1
            else:
                # did not find anything
                name = self._generateName(colnum)

        # examine whether object type is at end of name
        # convert, and remove, if is
        type = 'float'
        for codename, codetype in typecodes:
            if name[-len(codename):] == codename:
                type = codetype
                name = name[:-len(codename)].strip()
                break

        return type, self.prefix + name + self.suffix

    def _setNameAndType(self, colnum, colname, coltype):
        """Set a name for column number given column name and type."""
        while colnum >= len(self.coltypes):
            self.coltypes.append('')
        self.coltypes[colnum] = coltype
        self.nametypes[colname] = coltype
        self.colnames[colnum] = colname
        self.colignore[colnum] = self.headerignore
        if colname not in self.data:
            self.data[colname] = []

    def readData(self):
        """Read the data into the document."""

        # open the csv file
        csvf = utils.UnicodeCSVReader( self.filename,
                                       delimiter=self.delimiter,
                                       quotechar=self.textdelimiter,
                                       encoding=self.encoding )

        # make in iterator for the file
        if self.readrows:
            it = _FileReaderRows(csvf)
        else:
            it = _FileReaderCols(csvf)

        # dataset names for each column
        self.colnames = {}
        # type of column (float, string or date)
        self.coltypes = []
        # type of names of columns
        self.nametypes = {}
        # ignore lines after headers
        self.colignore = defaultdict(lambda: int(self.headerignore))

        # iterate over each line (or column)
        while True:
            try:
                line = it.next()
            except StopIteration:
                break

            # iterate over items on line
            for colnum, col in enumerate(line):

                if colnum >= len(self.coltypes) or self.coltypes[colnum] == '':
                    ctype = 'float'
                else:
                    ctype = self.coltypes[colnum]

                # ignore lines after headers
                if colnum < len(self.coltypes) and self.colignore[colnum] > 0:
                    self.colignore[colnum] -= 1
                    continue

                try:
                    # do any necessary conversion
                    if ctype == 'float':
                        v = float(col)
                    elif ctype == 'date':
                        v = utils.dateStringToDate(col)
                    elif ctype == 'string':
                        v = col
                    else:
                        raise RuntimeError, "Invalid type in CSV reader"

                except ValueError:
                    if col.strip() == '':
                        # skip blanks unless blanksaredata is set
                        if self.blanksaredata and colnum in self.colnames:
                            # assumes a numeric data type
                            self.data[self.colnames[colnum]].append(N.nan)
                    elif ( colnum in self.colnames and
                         len(self.data[self.colnames[colnum]]) == 0 ):
                        # if dataset is empty, convert to a string dataset
                        self._setNameAndType(colnum, self.colnames[colnum], 'string')
                        self.data[self.colnames[colnum]].append(col)
                    else:
                        # start a new dataset if conversion failed
                        coltype, name = self._getNameAndColType(colnum, col)
                        self._setNameAndType(colnum, name.strip(), coltype)

                else:
                    # generate a name if required
                    if colnum not in self.colnames:
                        self._setNameAndType(colnum, self._generateName(colnum),
                                             'float')

                    # conversion okay
                    # append number to data
                    coldata = self.data[self.colnames[colnum]]
                    coldata.append(v)

    def setData(self, document, linkedfile=None):
        """Set the read-in datasets in the document."""

        # iterate over each read-in dataset
        dsnames = []
        for name in self.data.iterkeys():

            # skip error data here, they are used below
            # error data name contains \0
            if name.find('\0') >= 0:
                continue

            dsnames.append(name)

            # get data and errors (if any)
            data = []
            for k in (name, name+'\0+-', name+'\0+', name+'\0-'):
                data.append( self.data.get(k, None) )

            # make them have a maximum length by adding NaNs
            maxlen = max([len(x) for x in data if x is not None])
            for i in range(len(data)):
                if data[i] is not None and len(data[i]) < maxlen:
                    data[i] = N.concatenate(
                        ( data[i], N.zeros(maxlen-len(data[i]))*N.nan ) )

            # create dataset
            dstype = self.nametypes[name]
            if dstype == 'string':
                ds = datasets.DatasetText(data=data[0], linked=linkedfile)
            elif dstype == 'date':
                ds = datasets.DatasetDateTime(data=data[0], linked=linkedfile)
            else:
                ds = datasets.Dataset(data=data[0], serr=data[1],
                                      perr=data[2], nerr=data[3],
                                      linked=linkedfile)

            document.setData(name, ds)

        dsnames.sort()
        return dsnames
