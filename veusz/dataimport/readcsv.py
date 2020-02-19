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

from __future__ import division
import re
import csv
import numpy as N

from .base import ImportingError
from ..compat import crange, cnext, cstr, CIterator
from .. import datasets
from .. import utils
from .. import qtall as qt

class _FileReaderCols(CIterator):
    """Read a CSV file in rows. This acts as an iterator.

    This is a very simple wrapper around the csv module
    """

    def __init__(self, csvreader):
        self.csvreader = csvreader
        self.maxlen = 0
        self.line = 1

    def __iter__(self):
        return self

    def __next__(self):
        """Return next row."""
        try:
            row = cnext(self.csvreader)
        except csv.Error as e:
            raise ImportingError("Error in line %i: %s" % (self.line, cstr(e)))

        self.line += 1

        # add blank columns up to maximum previously read
        self.maxlen = max(self.maxlen, len(row))
        row = row + ['']*(self.maxlen - len(row))

        return row

class _FileReaderRows(CIterator):
    """Read a CSV file in columns. This acts as an iterator.

    This means we have to read the whole file in, then return cols :-(
    """

    def __init__(self, csvreader):
        self.data = []
        self.maxlength = 0

        lineno = 1
        try:
            for line in csvreader:
                self.maxlength = max(self.maxlength, len(line))
                self.data.append(line)
                lineno += 1
        except csv.Error as e:
            raise ImportingError("Error in line %i: %s" % (lineno, cstr(e)))

        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
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

class _NextValue(Exception):
    """A class to be raised to move to next value."""

class ReadCSV(object):
    """A class to import data from CSV files."""

    def __init__(self, params):
        """Initialise the reader.
        params is a ParamsCSV object
        """

        self.params = params
        self.numericlocale = qt.QLocale(params.numericlocale)
        self.datere = re.compile(
            utils.dateStrToRegularExpression(params.dateformat))

        # created datasets. Each name is associated with a list
        self.data = {}

    def _generateName(self, column):
        """Generate a name for a column."""
        if self.params.readrows:
            prefix = 'row'
        else:
            prefix = 'col'

        name = '%s%s%i%s' % (
            self.params.prefix, prefix, column+1, self.params.suffix)
        return name

    def _getNameAndColType(self, colnum, colval):
        """Get column name and type."""

        name = colval.strip()

        if name in ('+', '-', '+-'):
            # loop to find previous valid column
            prevcol = colnum - 1
            while prevcol >= 0:
                if prevcol not in self.colnames:
                    break
                n = self.colnames[prevcol]
                if len(n) > 0 and n[-1] not in "+-":
                    # we add a \0 here so that there's no chance of the user
                    # using this as a column name
                    name = n + '\0' + name

                    if self.coltypes[prevcol] == 'unknown':
                        # force previous column to float if this
                        # column is an error bar
                        self.coltypes[prevcol] = 'float'
                    elif self.coltypes[prevcol] != 'float':
                        # we can't treat this as an error bar if the
                        # previous column is not numeric
                        break

                    return 'float', name
                prevcol -= 1

            # did not find anything
            name = self._generateName(colnum)

        # examine whether object type is at end of name
        # convert, and remove, if is
        dtype = 'unknown'
        for codename, codetype in typecodes:
            if name[-len(codename):] == codename:
                dtype = codetype
                name = name[:-len(codename)].strip()
                break
        return dtype, self.params.prefix + name + self.params.suffix

    def _setNameAndType(self, colnum, colname, coltype):
        """Set a name for column number given column name and type."""
        while colnum >= len(self.coltypes):
            self.coltypes.append('')

        if colname in self.nametypes:
            # if there is an existing dataset with the same name,
            # ensure there is consistency of type
            coltype = self.nametypes[colname]
        else:
            self.nametypes[colname] = coltype

        self.coltypes[colnum] = coltype
        self.colnames[colnum] = colname
        self.colignore[colnum] = self.params.headerignore
        self.colblanks[colnum] = 0
        if colname not in self.data:
            self.data[colname] = []

    def _guessType(self, val):
        """Guess type for new dataset."""
        v, ok = self.numericlocale.toDouble(val)
        if ok:
            return 'float'
        m = self.datere.match(val)
        try:
            utils.dateREMatchToDate(m)
            return 'date'
        except ValueError:
            return 'string'

    def _newValueInBlankColumn(self, colnum, col):
        """Handle occurance of new data value in previously blank column.
        """

        if self.params.headermode == '1st':
            # just use name of column as title in 1st header mode
            coltype, name = self._getNameAndColType(colnum, col)
            self._setNameAndType(colnum, name.strip(), coltype)
            raise _NextValue()
        elif self.params.headermode == 'none':
            # no header, so just start a new data set
            dtype = self._guessType(col)
            self._setNameAndType(
                colnum, self._generateName(colnum), dtype)
        else:
            # see whether it looks like data, not a header
            dtype = self._guessType(col)
            if dtype == 'string':
                # use text as dataset name
                coltype, name = self._getNameAndColType(colnum, col)
                self._setNameAndType(colnum, name.strip(), coltype)
                raise _NextValue()
            else:
                # use guessed data type and generated name
                self._setNameAndType(
                    colnum, self._generateName(colnum), dtype)

    def _newUnknownDataValue(self, colnum, col):
        """Process data value if data type is unknown.
        """

        # blank value
        if col.strip() == '':
            if self.params.blanksaredata:
                # keep track of blanks above autodetected data
                self.colblanks[colnum] += 1
            # skip back to next value
            raise _NextValue()

        # guess type from data value
        dtype = self._guessType(col)
        self.nametypes[self.colnames[colnum]] = dtype
        self.coltypes[colnum] = dtype

        # add back on blanks if necessary with correct format
        for i in crange(self.colblanks[colnum]):
            d = (N.nan, '')[dtype == 'string']
            self.data[self.colnames[colnum]].append(d)
        self.colblanks[colnum] = 0

    def _handleFailedConversion(self, colnum, col):
        """If conversion from text to data type fails."""
        if col.strip() == '':
            # skip blanks unless blanksaredata is set
            if self.params.blanksaredata:
                # assumes a numeric data type
                self.data[self.colnames[colnum]].append(N.nan)
        else:
            if self.params.headermode == '1st':
                # no more headers, so fill with invalid number
                self.data[self.colnames[colnum]].append(N.nan)
            else:
                # start a new dataset if conversion failed
                coltype, name = self._getNameAndColType(colnum, col)
                self._setNameAndType(colnum, name.strip(), coltype)

    def _handleVal(self, colnum, col):
        """Handle a value from the file.
        colnum: number of column
        col: data value
        """

        if colnum not in self.colnames:
            # ignore blanks
            if col.strip() == '':
                return
            # process value
            self._newValueInBlankColumn(colnum, col)

        # ignore lines after headers
        if self.colignore[colnum] > 0:
            self.colignore[colnum] -= 1
            return

        # process value if data type unknown
        if self.coltypes[colnum] == 'unknown':
            self._newUnknownDataValue(colnum, col)

        ctype = self.coltypes[colnum]
        try:
            # convert text to data type of column
            if ctype == 'float':
                v, ok = self.numericlocale.toDouble(col)
                if not ok:
                    raise ValueError
            elif ctype == 'date':
                m = self.datere.match(col)
                v = utils.dateREMatchToDate(m)
            elif ctype == 'string':
                v = col
            else:
                raise RuntimeError("Invalid type in CSV reader")

        except ValueError:
            self._handleFailedConversion(colnum, col)

        else:
            # conversion succeeded - append number to data
            self.data[self.colnames[colnum]].append(v)

    def readData(self):
        """Read the data into the document."""

        par = self.params

        # open the csv file
        csvf = utils.get_unicode_csv_reader(
            par.filename,
            delimiter=par.delimiter,
            quotechar=par.textdelimiter,
            skipinitialspace=par.skipwhitespace,
            encoding=par.encoding )

        # make in iterator for the file
        if par.readrows:
            it = _FileReaderRows(csvf)
        else:
            it = _FileReaderCols(csvf)

        # ignore rows (at top), if requested
        for i in crange(par.rowsignore):
            try:
                cnext(it)
            except StopIteration:
                return

        # dataset names for each column
        self.colnames = {}
        # type of column (float, string or date)
        self.coltypes = []
        # type of names of columns
        self.nametypes = {}
        # ignore lines after headers
        self.colignore = {}
        # keep track of how many blank values before 1st data for auto
        # type detection
        self.colblanks = {}

        # iterate over each line (or column)
        while True:
            try:
                line = cnext(it)
            except StopIteration:
                break

            # iterate over items on line
            for colnum, col in enumerate(line):
                try:
                    self._handleVal(colnum, col)
                except _NextValue:
                    pass

    def setData(self, outmap, linkedfile=None):
        """Set the read-in datasets in the dict outmap."""

        for name in self.data:
            # skip error data here, they are used below
            # error data name contains \0
            if name.find('\0') >= 0:
                continue

            # get data and errors (if any)
            data = []
            for k in (name, name+'\0+-', name+'\0+', name+'\0-'):
                data.append( self.data.get(k, None) )

            # make them have a maximum length by adding NaNs
            maxlen = max([len(x) for x in data if x is not None])
            for i in crange(len(data)):
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
                ds = datasets.Dataset(
                    data=data[0], serr=data[1], perr=data[2], nerr=data[3],
                    linked=linkedfile)

            outmap[name] = ds

        return sorted(outmap)
