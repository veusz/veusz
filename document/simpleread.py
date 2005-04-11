# simpleread.py
# A simple data reading routine

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
import StringIO

import numarray.ieeespecial

import doc

class _DescriptorPart:

    # used to split the descriptor expression
    partsplitter = re.compile( r'(\+-|\+|-|[a-zA-Z][a-zA-Z0-9_]*'
                               r'|\[[\-0-9]*[:][\-0-9]*\]|,)' )
    # check a variable name
    #checkvar = re.compile(r'^[A-Za-z][A-za-z0-9]*$')
    
    def __init__(self, text):
        """Initialise descriptor for 1 variable plus errors."""

        self.symerr = -1
        self.poserr = -1
        self.negerr = -1
        self.val = -1
        self.name = None
        self.errorcount = 0

        self.startindex = None
        self.stopindex = None

        # split up the expression into its components
        parts = _DescriptorPart.partsplitter.split(text)

        index = 0
        for p in parts:
            if p == '':
                pass
            elif p == '+-':
                self.symerr = index
                index += 1
            elif p == '+':
                self.poserr = index
                index += 1
            elif p == '-':
                self.negerr = index
                index += 1

            # there's no match on this part
            elif _DescriptorPart.partsplitter.match(p) == None:
                raise ValueError, ( 'Cannot understand descriptor '
                                    'syntax "%s"' % p )

            # column indicator
            elif p[0] == '[':
                # remove brackets
                p = p[1:-1]

                # retrieve start and stop values if specified
                colon = p.find(':')
                if colon >= 0:
                    startindex = p[:colon]
                    stopindex = p[colon+1:]

                    if len(startindex) > 0:
                        self.startindex = int(startindex)
                    else:
                        self.startindex = 1
                    if len(stopindex) > 0:
                        self.stopindex = int(stopindex)
                    else:
                        self.stopindex = 999999

            elif p == ',':
                # skip
                pass

            # must be variable name
            else:
                self.val = index
                index += 1
                self.name = p

        if self.name == None:
            raise ValueError, 'Value name missing in "%s"' % text

        # Calculate indicies for looping over values
        self.single = self.startindex == None and self.stopindex == None
        if self.single:
            # one value only
            self.startindex = self.stopindex = 1

    def readFromStream(self, stream, datasets):
        """Read data from stream, and write to datasets."""

        valindexes = (self.val, self.symerr, self.poserr, self.negerr)
        # loop over column range
        for index in xrange(self.startindex, self.stopindex+1):
            # name for variable
            if self.single:
                name = '%s' % (self.name,)
            else:
                name = '%s_%i' % (self.name, index)

            # loop over columns until we run out, or we don't need any
            count = 0
            while count in valindexes:
                val = stream.nextColumn()
                if val == None:
                    return

                # do conversion
                try:
                    val = float(val)
                except ValueError:
                    val = numarray.ieeespecial.nan
                    self.errorcount += 1

                # append a suffix to specfiy whether error or value
                if count == self.symerr:
                    fullname = '%s_sym' % name
                elif count == self.poserr:
                    fullname = '%s_pos' % name
                elif count == self.negerr:
                    fullname = '%s_neg' % name
                else:
                    fullname = name

                # add data into dataset
                if fullname not in datasets:
                    datasets[fullname] = []
                datasets[fullname].append(val)

                count += 1

    def setInDocument(self, datasets, document, linkedfile=None):
        """Set the read-in data in the document."""

        names = []
        for index in xrange(self.startindex, self.stopindex+1):
            # name for variable
            if self.single:
                name = '%s' % (self.name,)
            else:
                name = '%s_%i' % (self.name, index)

            if name in datasets:
                vals = datasets[name]
                pos = neg = sym = None

                if name + '_pos' in datasets:
                    pos = datasets[ name + '_pos' ]
                if name + '_neg' in datasets:
                    neg = datasets[ name + '_neg' ]
                if name + '_sym' in datasets:
                    sym = datasets[ name + '_sym' ]

                ds = doc.Dataset( data = vals, serr = sym,
                                  nerr = neg, perr = pos )

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

    def newLine(self):
        """Read in, and split the next line."""

        while 1:
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
        self.parseDescriptor(descriptor)
        self.datasets = {}

    def parseDescriptor(self, descriptor):
        """Take a descriptor, and parse it into its individual parts."""

        self.parts = [_DescriptorPart(i) for i in descriptor.split()]

    def readData(self, stream):
        """Read in that data from the stream."""

        # loop over lines
        while stream.newLine():
            for i in self.parts:
                i.readFromStream(stream, self.datasets)

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

        names = []
        for i in self.parts:
            names += i.setInDocument(self.datasets, document,
                                     linkedfile)

        return names
