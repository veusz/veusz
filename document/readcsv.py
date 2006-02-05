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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

"""This module contains routines for importing CSV data files
in an easy-to-use manner."""

import csv
import datasets

class _FileReaderCols:
    """Read a CSV file in rows. This acts as an iterator.

    This is a very simple wrapper around the csv module
    """

    def __init__(self, csvreader):
        self.csvreader = csvreader

    def next(self):
        """Return next row."""
        return self.csvreader.next()

class _FileReaderRows:
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

class ReadCSV:
    """A class to import data from CSV files."""

    def __init__(self, filename, readrows=False, prefix=None):
        """Initialise the reader to import data from filename.

        If readrows is True, then data are read from columns, rather than
        rows

        prefix is a prefix to prepend to the name of datasets from this file
        """

        self.filename = filename
        self.readrows = readrows
        self.prefix = prefix

        # datasets. Each name is associated with a list
        self.data = {}

    def _generateName(self, column):
        """Generate a name for a column."""
        if self.readrows:
            prefix = 'row'
        else:
            prefix = 'col'

        name = '%s%i' % (prefix, column+1)
        if self.prefix != None:
            name = '%s_%s' % (self.prefix, name)
        return name

    def readData(self):
        """Read the data into the document."""

        # open the csv file
        f = open(self.filename)
        csvf = csv.reader(f)

        # make in iterator for the file
        if self.readrows:
            it = _FileReaderRows(csvf)
        else:
            it = _FileReaderCols(csvf)

        # dataset names for each column
        colnames = {}

        # iterate over each line (or column)
        while True:
            try:
                line = it.next()
            except StopIteration:
                break

            # iterate over items on line
            for colnum, col in enumerate(line):
                try:
                    # try to convert to a float
                    f = float(col)

                except ValueError:
                    # not a number, so generate a dataset name

                    # skip blanks
                    if col == '':
                        continue

                    # append on suffix to previous name if an error
                    name = col
                    if col in ('+', '-', '+-'):
                        if colnum > 0:
                            name = colnames[colnum-1] + col
                        else:
                            name = self._generateName(colnum)

                    else:
                        # add on prefix if reqd
                        if self.prefix != None:
                            name = '%s_%s' % (self.prefix, name)

                    # add on dataset
                    colnames[colnum] = name
                    if name not in self.data:
                        self.data[name] = []
                    
                else:
                    # it was a number

                    # add on more columns if necessary
                    if colnum not in colnames:
                        name = self._generateName(colnum)
                        colnames[colnum] = name
                        if name not in self.data:
                            self.data[name] = []

                    # append number to data
                    coldata = self.data[colnames[colnum]]
                    coldata.append(f)

    def setData(self, document, linkedfile=None):
        """Set the read-in datasets in the document."""

        # iterate over each read-in dataset
        dsnames = []
        for name in self.data.iterkeys():

            # skip error data here, they are used below
            if name[-1] in '+-':
                continue

            dsnames.append(name)

            # get data and errors (if any)
            data = self.data[name]
            serr = self.data.get(name+'+-', None)
            perr = self.data.get(name+'+', None)
            nerr = self.data.get(name+'-', None)
                                
            ds = datasets.Dataset(data=data, serr=serr, perr=perr,
                                  nerr=nerr, linked=linkedfile)
            document.setData(name, ds)

        dsnames.sort()
        return dsnames

#r = ReadCSV('../test2.csv', prefix='foo', readrows=True)
#r.readData()
#print r.data
