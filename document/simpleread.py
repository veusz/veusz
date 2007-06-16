#    Copyright (C) 2004 Jeremy S. Sanders
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

"""SimpleRead: a class for the reading of data formatted in a simple way

To read the data it takes a descriptor which takes the form of

varname<error specifiers><[repeater]>
where <> marks optional arguments, e.g.

x+- y+,- z+-[1:5]

+- means symmetric error bars
+,- means asymmetric error bars
, is a separator

z+-[1:5] means read z_1+- z_2+- ... z_5+-

Indicices can be unspecified, [], [:], [1:], [:5]
The first 3 mean the same thing, the last means read from 1 to 5
"""

import re
import sys
import itertools
import StringIO

import numarray as N
import numarray.ieeespecial as NIE

import datasets

class DescriptorError(ValueError):
    """Used to indicate an error with the descriptor."""
    pass

class _DescriptorPart:

    # used to split the descriptor expression
    partsplitter = re.compile( r'(\+-|\+|-|[\.a-zA-Z0-9_]+'
                               r'|\[[\-0-9]*[:][\-0-9]*\]|,)' )
    # check a variable name
    #checkvar = re.compile(r'^[A-Za-z][A-za-z0-9]*$')
    
    def __init__(self, text):
        """Initialise descriptor for 1 variable plus errors."""

        self.name = None
        self.errorcount = 0
        self.columns = []

        self.startindex = None
        self.stopindex = None

        # split up the expression into its components (removing blank parts)
        parts = _DescriptorPart.partsplitter.split(text)
        parts = [i for i in parts if i != ""]

        for count, part in itertools.izip(itertools.count(), parts):
            if part == '+-':
                self.columns.append('SYM')
            elif part == '+':
                self.columns.append('POS')
            elif part == '-':
                self.columns.append('NEG')

            # there's no match on this part
            elif _DescriptorPart.partsplitter.match(part) == None:
                raise DescriptorError, ( 'Cannot understand descriptor '
                                         'syntax "%s"' % part )

            # column indicator
            elif part[0] == '[':
                # remove brackets
                part = part[1:-1]

                # retrieve start and stop values if specified
                colon = part.find(':')
                if colon >= 0:
                    startindex = part[:colon]
                    stopindex = part[colon+1:]

                    if len(startindex) > 0:
                        self.startindex = int(startindex)
                    else:
                        self.startindex = 1
                    if len(stopindex) > 0:
                        self.stopindex = int(stopindex)
                    else:
                        self.stopindex = 999999

            elif part == ',':
                # multiple commas skip columns
                if count == len(parts)-1 or parts[count+1] == ',':
                    self.columns.append('SKIP')

            # must be variable name
            else:
                self.columns.append('DATA')
                self.name = part

        if self.name == None:
            raise DescriptorError, 'Value name missing in "%s"' % text

        # Calculate indicies for looping over values
        self.single = self.startindex == None and self.stopindex == None
        if self.single:
            # one value only
            self.startindex = self.stopindex = 1

    def readFromStream(self, stream, thedatasets, block=None):
        """Read data from stream, and write to thedatasets."""

        # loop over column range
        for index in xrange(self.startindex, self.stopindex+1):
            # name for variable
            if self.single:
                name = self.name
            else:
                name = '%s_%i' % (self.name, index)

            # if we're reading multiple blocks
            if block != None:
                name += '_%i' % block

            # loop over columns until we run out, or we don't need any
            for col in self.columns:
                # get next column and return if we run out of data
                val = stream.nextColumn()
                if val == None:
                    return

                # do conversion
                try:
                    val = float(val)
                except ValueError:
                    val = NIE.nan
                    self.errorcount += 1

                # append a suffix to specify whether error or value
                # \0 is used as the user cannot enter it
                fullname = '%s\0%s' % (name, col)

                # add data into dataset
                if fullname not in thedatasets:
                    thedatasets[fullname] = []
                thedatasets[fullname].append(val)

    def setInDocument(self, thedatasets, document, block=None, linkedfile=None):
        """Set the read-in data in the document."""

        names = []
        for index in xrange(self.startindex, self.stopindex+1):
            # name for variable
            if self.single:
                name = '%s' % (self.name,)
            else:
                name = '%s_%i' % (self.name, index)
            if block != None:
                name += '_%i' % block

            if name+'\0DATA' in thedatasets:
                vals = thedatasets[name+'\0DATA']
                pos = neg = sym = None

                # retrieve the data for this dataset
                if name+'\0POS' in thedatasets:
                    pos = thedatasets[name+'\0POS']
                if name+'\0NEG' in thedatasets:
                    neg = thedatasets[name+'\0NEG']
                if name+'\0SYM' in thedatasets:
                    sym = thedatasets[name+'\0SYM']

                # create the dataset
                ds = datasets.Dataset(data = vals, serr = sym,
                                      nerr = neg, perr = pos)
                ds.linked = linkedfile

                document.setData( name, ds )
                names.append(name)
            else:
                break

        return names

class FileStream:
    """Class to read in the data from the file-like object."""

    def __init__(self, file):
        """File can be any iterator-like object."""
        
        self.remainingline = []
        self.file = file

    def nextColumn(self):
        """Return value of next column of line."""
        try:
            return self.remainingline.pop(0)
        except IndexError:
            return None

    def allColumns(self):
        """Get all columns of current line (none are discarded)."""
        return self.remainingline

    def flushLine(self):
        """Forget the rest of the line."""
        self.remainingline = []

    def newLine(self):
        """Read in, and split the next line."""

        while True:
            # get next line from data source
            try:
                line = self.file.next()
            except StopIteration:
                # end of file
                return False

            # remove comment symbols
            for sym in "#;!%":
                comment = line.find(sym)
                if comment >= 0:
                    line = line[:comment]

            # check for continuation character
            line = line.strip()
            if len(line) == 0:
                return True

            continuation = line[-1] == '\\'
            if continuation:
                line = line[:-1]

            # break up and append to buffer
            self.remainingline += line.split()

            if not continuation:
                return True

class StringStream(FileStream):
    '''For reading data from a string.'''
    
    def __init__(self, text):
        """This works by using a StringIO class to iterate over the text."""
        
        FileStream.__init__( self, StringIO.StringIO(text) )

class SimpleRead:
    '''Class to read in datasets from a stream.

    The descriptor specifies the format of data to read from the stream
    Read the docstring for this module for information
    '''
    
    def __init__(self, descriptor):
        self._parseDescriptor(descriptor)
        self.clearState()

    def clearState(self):
        """Start reading from scratch."""
        self.datasets = {}
        self.blocks = None
        
    def _parseDescriptor(self, descriptor):
        """Take a descriptor, and parse it into its individual parts."""

        self.parts = [_DescriptorPart(i) for i in descriptor.split()]

    def readData(self, stream, useblocks=False):
        """Read in the data from the stream.

        If useblocks is True, data are read as separate blocks.
        Dataset names are appending with an underscore and a block
        number if set.
        """

        if useblocks:
            self._readDataBlocked(stream)
        else:
            self._readDataUnblocked(stream)

    def _readDataUnblocked(self, stream):
        """Read in that data from the stream."""

        allparts = self.parts

        # loop over lines
        while stream.newLine():

            if stream.remainingline[:1] == ['descriptor']:
                # a change descriptor statement
                descriptor =  ' '.join(stream.remainingline[1:])
                self._parseDescriptor(descriptor)
                allparts += self.parts
            else:
                # normal text
                for i in self.parts:
                    i.readFromStream(stream, self.datasets)
            stream.flushLine()

        self.parts = allparts
        self.blocks = None

    def _readDataBlocked(self, stream):
        """Read in the data, using blocks."""

        blocks = {}
        block = 1
        while stream.newLine():
            l = stream.remainingline

            # if this is a blank line, separating data then advance to a new
            # block
            if len(l) == 0 or l[0].lower() == 'no':
                # blank lines separate blocks
                if block in blocks:
                    block += 1
            else:
                # read in data
                for i in self.parts:
                    i.readFromStream(stream, self.datasets, block=block)
                blocks[block] = True

            # lose remaining data
            stream.flushLine()

        self.blocks = blocks.keys()

    def getInvalidConversions(self):
        """Return the number of invalid conversions after reading data.

        Returns a dict of dataset, number values."""

        out = {}
        for i in self.parts:
            out[i.name] = i.errorcount
        return out

    def setInDocument(self, document, linkedfile=None):
        """Set the data in the document.

        Returns list of variable names read.
        """

        # iterate over blocks used
        if self.blocks == None:
            blocks = [None]
        else:
            blocks = self.blocks

        names = []
        for b in blocks:
            for i in self.parts:
                names += i.setInDocument(self.datasets, document,
                                         block=b, linkedfile=linkedfile)

        return names

#####################################################################
# 2D data reading

class Read2DError(ValueError):
    pass

class SimpleRead2D:
    def __init__(self, name):
        """Read dataset with name given."""
        self.name = name
        self.xrange = None
        self.yrange = None
        self.invertrows = False
        self.invertcols = False
        self.transpose = False

    ####################################################################
    # Follow functions are for setting parameters during reading of data

    def _paramXRange(self, cols):
        try:
            self.xrange = ( float(cols[1]), float(cols[2]) )
        except ValueError:
            raise Read2DError, "Could not interpret xrange"
        
    def _paramYRange(self, cols):
        try:
            self.yrange = ( float(cols[1]), float(cols[2]) )
        except ValueError:
            raise Read2DError, "Could not interpret yrange"

    def _paramInvertRows(self, cols):
        self.invertrows = True
        
    def _paramInvertCols(self, cols):
        self.invertcols = True

    def _paramTranspose(self, cols):
        self.transpose = True

    ####################################################################

    def readData(self, stream):
        """Read data from stream given

        stream consists of:
        optional:
         xrange A B   - set the range of x from A to B
         yrange A B   - set the range of y from A to B
         invertrows   - invert order of the rows
         invertcols   - invert order of the columns
         transpose    - swap rows and columns
        then:
         matrix of columns and rows, separated by line endings
         the rows are in reverse-y order (highest y first)
         blank line stops reading for further datasets
        """

        settings = {
            'xrange': self._paramXRange,
            'yrange': self._paramYRange,
            'invertrows': self._paramInvertRows,
            'invertcols': self._paramInvertCols,
            'transpose': self._paramTranspose
            }

        rows = []
        # loop over lines
        while stream.newLine():
            cols = stream.allColumns()

            if len(cols) > 0:
                # check to see whether parameter is set
                c = cols[0].lower()
                if c in settings:
                    settings[c](cols)
                    stream.flushLine()
                    continue
            else:
                # if there's data and we get to a blank line, finish
                if len(rows) != 0:
                    break

            # read columns
            line = []
            while True:
                v = stream.nextColumn()
                if v == None:
                    break
                try:
                    line.append( float(v) )
                except ValueError:
                    raise Read2DError, "Could not interpret number '%s'" % v

            # add row to dataset
            if len(line) != 0:
                if self.invertcols:
                    line.reverse()
                rows.insert(0, line)

        # swap rows if requested
        if self.invertrows:
            rows.reverse()

        # dodgy formatting probably...
        if len(rows) == 0:
            raise Read2DError, "No data could be imported for dataset"

        # convert the data to a numarray
        try:
            self.data = N.array(rows)
        except ValueError:
            raise Read2DError, "Could not convert data to 2D matrix"

        # obvious check
        if len(self.data.shape) != 2:
            raise Read2DError, "Dataset was not 2D"

        # transpose matrix if requested
        if self.transpose:
            self.data = N.transpose(self.data).copy()

    def setInDocument(self, document, linkedfile=None):
        """Set the data in the document.

        Returns list containing name of dataset read
        """

        ds = datasets.Dataset2D(self.data, xrange=self.xrange,
                                yrange=self.yrange)
        ds.linked = linkedfile

        document.setData(self.name, ds)
        
        return [self.name]

