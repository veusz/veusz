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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

"""SimpleRead: a class for the reading of data formatted in a simple way

To read the data it takes a descriptor which takes the form of

varname<error specifiers><[repeater]>
where <> marks optional arguments, e.g.

x+- y+,- z+-[1:5]
x(text) y(date) z(number),+-

+- means symmetric error bars
+,- means asymmetric error bars
, is a separator
(text) or (date) specifies datatype

z+-[1:5] means read z_1+- z_2+- ... z_5+-

Indicices can be unspecified, [:], [1:], [:5]
The first 3 mean the same thing, the last means read from 1 to 5

Commas are now optional in 1.6, so descriptors can look like
x +- y + -
"""

from __future__ import division
import re
import ast

import numpy as N

from ..compat import crange, cnext, CStringIO
from .. import utils
from .. import datasets
from .. import qtall as qt4
from .import base

# a regular expression for splitting descriptor into tokens
descrtokens_split_re = re.compile(r'''
(
 `[^`]*`       |  # quoted name
 [ ,]          |  # comma or space
 \([a-z]+?\)   |  # data type
 \+- | \+ | -  |  # error bars
 \[.*?\]          # indices
)
''', re.VERBOSE)

range_re = re.compile(r'''^
 \[
 (-?[0-9]+)? 
 :
 (-?[0-9]+)? 
 \]
$''', re.VERBOSE)

def interpretDescriptor(descr):
    """Get a descriptor and create a set of descriptor objects."""

    parts = []

    split = descrtokens_split_re.split(descr.strip())
    tokens = [x for x in split if x != '']
    # make sure that last dataset is added
    tokens += ['DUMMY']

    name = datatype = idxrange = None
    columns = []
    for tokenindex, token in enumerate(tokens):
        # skip spaces
        if token == ' ':
            if tokenindex > 0 and tokens[tokenindex-1] == ',':
                columns.append(',')
            continue

        # ignore column
        if token == ',':
            if tokenindex == 0 or ( tokens[tokenindex-1] == ',' or
                                    tokens[tokenindex-1] == ' ' ):
                columns.append(',')
            continue

        # does token match datatype name?
        if ( token[0] == '(' and token[-1] == ')' and
             token[1:-1] in datatype_name_convert ):
            datatype = datatype_name_convert[token[1:-1]]
            continue

        # match error bars
        if token in ('+', '-', '+-'):
            columns.append(token)
            continue

        # does token match a range?
        m = range_re.match(token)
        if m:
            if m.group(1):
                startindex = int(m.group(1))
            else:
                startindex = 1
            if m.group(2):
                stopindex = int(m.group(2))
            else:
                stopindex = 999999999
            idxrange = (startindex, stopindex)
            continue

        # quoted dataset name, so remove quotes
        if token[0] == '`' and token[-1] == '`':
            token = token[1:-1]

        # add previous entry
        if name is not None:
            parts.append( DescriptorPart(name, datatype, columns, idxrange) )
            name = datatype = idxrange = None
            columns = []

        columns.append('D')
        name = token

    return parts

class DescriptorError(ValueError):
    """Used to indicate an error with the descriptor."""
    pass

# this is a regular expression to match properly quoted strings
# hopefully a matching expression can be passed to eval
string_re = re.compile( r'''
^
u?"" |            # match empty double-quoted string
u?".*?[^\\]" |    # match double-quoted string, ignoring escaped quotes
u?'' |            # match empty single-quoted string
u?'.*?[^\\]'      # match single-quoted string, ignoring escaped quotes
$
''', re.VERBOSE )

# a line starting with text
text_start_re = re.compile( r'^[A-Za-z]' )

# convert data type strings in descriptor to internal datatype
datatype_name_convert = {
    'float': 'float',
    'numeric': 'float',
    'number': 'float',
    'text': 'string',
    'string': 'string',
    'date': 'date',
    'time': 'date'
    }

def guessDataType(val):
    """Try to work out data type from sample value (val)

    Return values are one of
    float, string, or date
    """

    # if the dataset type is specified
    # check for identifiers in dataset name
    # guess the type:
    # obvious float
    try:
        float(val)
        return 'float'
    except ValueError:
        pass

    # do all libcs check for these?
    if val.lower() in ('inf', '+inf', '-inf', 'nan'):
        return 'float'

    # obvious string
    if string_re.match(val):
        return 'string'

    # date
    if utils.isDateTime(val):
        return 'date'

    # assume string otherwise
    return 'string'

class DescriptorPart(object):
    """Represents part of a descriptor."""

    def __init__(self, name, datatype, columns, idxrange):
        """Construct DescriptorPart
        name is dataset name
        datatype is None or one of the possible options
        columns is a list of the columns '+', '-', '+-', ',' or 'D'
         for errors, ignoring a column or a data column
        idxrange is None or a tuple (minidx, maxidx)
        """
        self.name = name
        self.datatype = datatype
        self.columns = tuple(columns)

        self.errorcount = 0
        
        self.single = idxrange is None
        if self.single:
            self.startindex = self.stopindex = 1
        else:
            self.startindex, self.stopindex = idxrange

    def readFromStream(self, stream, thedatasets, block=None):
        """Read data from stream, and write to thedatasets."""

        # loop over column range
        for index in crange(self.startindex, self.stopindex+1):
            # name for variable
            if self.single:
                name = self.name
            else:
                name = '%s_%i' % (self.name, index)

            # if we're reading multiple blocks
            if block is not None:
                name += '_%i' % block

            # loop over columns until we run out, or we don't need any
            for col in self.columns:
                # get next column and return if we run out of data
                val = stream.nextColumn()
                if val is None:
                    return
                # append a suffix to specify whether error or value
                # \0 is used as the user cannot enter it
                fullname = '%s\0%s' % (name, col)

                # get dataset (or get new one)
                try:
                    dataset = thedatasets[fullname]
                except KeyError:
                    dataset = thedatasets[fullname] = []

                if not self.datatype:
                    # try to guess type of data
                    self.datatype = guessDataType(val)

                # convert according to datatype
                if self.datatype == 'float':
                    try:
                        # do conversion
                        dat = float(val)
                    except ValueError:
                        dat = N.nan
                        self.errorcount += 1

                elif self.datatype == 'string':
                    if string_re.match(val):
                        conv = val
                        if conv[0:1] != 'u':
                            # a hack for python2
                            conv = 'u' + val
                        try:
                            dat = ast.literal_eval(conv)
                        except:
                            dat = val
                    else:
                        dat = val

                elif self.datatype == 'date':
                    dat = utils.dateStringToDate(val)

                # add data into dataset
                dataset.append(dat)

    def setOutput(self, thedatasets, outmap, block=None,
                  linkedfile=None,
                  prefix="", suffix="", tail=None):
        """Set the read-in data in the document."""

        # we didn't read any data
        if self.datatype is None:
            return

        for index in crange(self.startindex, self.stopindex+1):
            # name for variable
            if self.single:
                name = '%s' % (self.name,)
            else:
                name = '%s_%i' % (self.name, index)
            if block is not None:
                name += '_%i' % block

            # does the dataset exist?
            if name+'\0D' in thedatasets:
                vals = thedatasets[name+'\0D']
                pos = neg = sym = None

                # retrieve the data for this dataset
                if name+'\0+' in thedatasets: pos = thedatasets[name+'\0+']
                if name+'\0-' in thedatasets: neg = thedatasets[name+'\0-']
                if name+'\0+-' in thedatasets: sym = thedatasets[name+'\0+-']

                # make sure components are the same length
                minlength = 99999999999999
                for ds in vals, pos, neg, sym:
                    if ds is not None and len(ds) < minlength:
                        minlength = len(ds)
                for ds in vals, pos, neg, sym:
                    if ds is not None and len(ds) != minlength:
                        del ds[minlength:]

                # only remember last N values
                if tail is not None:
                    vals = vals[-tail:]
                    if sym is not None: sym = sym[-tail:]
                    if pos is not None: pos = pos[-tail:]
                    if neg is not None: neg = neg[-tail:]

                # create the dataset
                if self.datatype == 'float':
                    ds = datasets.Dataset( data = vals, serr = sym,
                                           nerr = neg, perr = pos,
                                           linked = linkedfile )
                elif self.datatype == 'date':
                    ds = datasets.DatasetDateTime( data=vals,
                                                   linked=linkedfile )
                elif self.datatype == 'string':
                    ds = datasets.DatasetText( data=vals,
                                               linked = linkedfile )
                else:
                    raise RuntimeError("Invalid data type")

                finalname = prefix + name + suffix
                outmap[finalname] = ds
            else:
                break

class Stream(object):
    """This object reads through an input data source (override
    readLine) and interprets data from the source."""
                                    
    # this is a regular expression for finding data items in data stream
    # I'll try to explain this bit-by-bit (these are ORd, and matched in order)
    find_re = re.compile( r'''
    `.+?`[^ \t\n\r#!%;]* | # match dataset name quoted in back-ticks
                           # we also need to match following characters to catch
                           # corner cases in the descriptor
    u?"" |          # match empty double-quoted string
    u?".*?[^\\]" |  # match double-quoted string, ignoring escaped quotes
    u?'' |          # match empty single-quoted string
    u?'.*?[^\\]' |  # match single-quoted string, ignoring escaped quotes
    [#!%;](?=descriptor) | # match separately comment char before descriptor
    [#!%;].* |      # match comment to end of line
    [^ \t\n\r#!%;]+ # match normal space/tab separated items
    ''', re.VERBOSE )

    def __init__(self):
        """Initialise stream object."""
        self.remainingline = []

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

    def readLine(self):
        """Read the next line of the data source.
        StopIteration is raised if there is no more data."""
        pass

    def newLine(self):
        """Read in, and split the next line."""

        while True:
            # get next line from data source
            try:
                line = self.readLine()
            except StopIteration:
                # end of file
                return False

            # break up and append to buffer (removing comments)
            cmpts = self.find_re.findall(line)
            self.remainingline += [ x for x in cmpts if x[0] not in '#!%;']

            if self.remainingline and self.remainingline[-1] == '\\':
                # this is a continuation: drop this item and read next line
                self.remainingline.pop()
            else:
                return True

class FileStream(Stream):
    """A stream based on a python-style file (or iterable)."""

    def __init__(self, file):
        """File can be any iterator-like object."""
        Stream.__init__(self)
        self.file = file

    def readLine(self):
        """Read the next line of the data source.
        StopIteration is raised if there is no more data."""
        return cnext(self.file)

class StringStream(FileStream):
    '''For reading data from a string.'''
    
    def __init__(self, text):
        """A stream which reads in from a text string."""
        
        FileStream.__init__( self, CStringIO(text) )

class CSVStream(Stream):
    """Read text from csv file."""

    def __init__(self, filename, delim, textdelim, locale, encoding):
        Stream.__init__(self)

        self.csvfile = utils.get_unicode_csv_reader(
            filename,
            delimiter=delim,
            quotechar=textdelim,
            encoding=encoding )
        self.localename = locale
        self.locale = qt4.QLocale(locale)

    def newLine(self):
        """Get next line from CSV file."""
        try:
            line = cnext(self.csvfile)
        except StopIteration:
            return False

        # delete empty cells on left, to make compatible with normal
        # text stream
        i = 0
        while i < len(line) and not line[i]:
            i += 1
        line = line[i:]

        if self.localename == 'en_US':
            # no conversion
            self.remainingline += line
        else:
            for t in line:
                v, ok = self.locale.toDouble(t)
                if ok:
                    # add on converted text - yuck - double conversion
                    self.remainingline.append('%e' % v)
                else:
                    self.remainingline.append(t)
        return True

class SimpleRead(object):
    '''Class to read in datasets from a stream.

    The descriptor specifies the format of data to read from the stream
    Read the docstring for this module for information

    tail attribute if set says to only use last tail data points when setting
    '''

    def __init__(self, descriptor):
        # convert descriptor to part objects
        descriptor = descriptor.strip()
        self._parseDescriptor(descriptor)

        # construct data names automatically
        self.autodescr = (descriptor == '')

        # get read for reading data
        self.clearState()

    def clearState(self):
        """Start reading from scratch."""
        self.datasets = {}
        self.blocks = None
        self.tail = None

    def _parseDescriptor(self, descriptor):
        """Take a descriptor, and parse it into its individual parts."""
        self.parts = interpretDescriptor(descriptor)

    def readData(self, stream, useblocks=False, ignoretext=False):
        """Read in the data from the stream.

        If useblocks is True, data are read as separate blocks.
        Dataset names are appending with an underscore and a block
        number if set.
        """

        self.ignoretext = ignoretext
        if useblocks:
            self._readDataBlocked(stream, ignoretext)
        else:
            self._readDataUnblocked(stream, ignoretext)

    def _readDataUnblocked(self, stream, ignoretext):
        """Read in that data from the stream."""

        allparts = list(self.parts)

        # loop over lines
        while stream.newLine():
            if stream.remainingline[:1] == ['descriptor']:
                # a change descriptor statement
                descriptor =  ' '.join(stream.remainingline[1:])
                self._parseDescriptor(descriptor)
                allparts += self.parts
                self.autodescr = False
            elif ( self.ignoretext and len(stream.remainingline) > 0 and 
                   text_start_re.match(stream.remainingline[0]) and
                   len(self.parts) > 0 and
                   self.parts[0].datatype != 'string' and
                   stream.remainingline[0] not in ('inf', 'nan') ):
                # ignore the line if it is text and ignore text is on
                # and first column is not text
                pass
            else:
                # normal text
                for p in self.parts:
                    p.readFromStream(stream, self.datasets)

                # automatically create parts if data are remaining
                if self.autodescr:
                    while len(stream.remainingline) > 0:
                        p = DescriptorPart(
                            str(len(self.parts)+1), None, 'D', None )
                        p.readFromStream(stream, self.datasets)
                        self.parts.append(p)
                        allparts.append(p)

            stream.flushLine()

        self.parts = allparts
        self.blocks = None

    def _readDataBlocked(self, stream, ignoretext):
        """Read in the data, using blocks."""

        allparts = list(self.parts)

        blocks = {}
        block = 1
        while stream.newLine():
            line = stream.remainingline

            # if this is a blank line, separating data then advance to a new
            # block
            if len(line) == 0 or line[0].lower() == 'no':
                # blank lines separate blocks
                if block in blocks:
                    block += 1
            else:
                # read in data
                for p in self.parts:
                    p.readFromStream(stream, self.datasets, block=block)

                # automatically create parts if data are remaining
                if self.autodescr:
                    while len(stream.remainingline) > 0:
                        p = DescriptorPart(
                            str(len(self.parts)+1), None, 'D', None )
                        p.readFromStream(stream, self.datasets, block=block)
                        self.parts.append(p)
                        allparts.append(p)

                blocks[block] = True

            # lose remaining data
            stream.flushLine()

        self.parts = allparts
        self.blocks = list(blocks.keys())

    def getInvalidConversions(self):
        """Return the number of invalid conversions after reading data.

        Returns a dict of dataset, number values."""

        out = {}
        for p in self.parts:
            out[p.name] = p.errorcount
        return out

    def getDatasetCounts(self):
        """Get a dict of the datasets read (main data part) and number
        of entries read."""
        out = {}
        for name in self.datasets:
            if name[-2:] == '\0D':
                out[name[:-2]] = len(self.datasets[name])
        return out

    def setOutput(self, out, linkedfile=None,
                  prefix='', suffix=''):
        """Set the data in the out dict.
        """

        # iterate over blocks used
        if self.blocks is None:
            blocks = [None]
        else:
            blocks = self.blocks

        # if automatically making parts, use a prefix/suffix if not set
        if self.autodescr and prefix == '' and suffix == '':
            prefix = 'col'

        for block in blocks:
            for part in self.parts:
                part.setOutput(
                    self.datasets, out,
                    block=block,
                    linkedfile=linkedfile,
                    prefix=prefix, suffix=suffix,
                    tail=self.tail)

#####################################################################
# 2D data reading

class Read2DError(base.ImportingError):
    pass

class SimpleRead2D(object):
    def __init__(self, name, params):
        """Read dataset with name given.
        params is a ImportParams2D object
        """
        self.name = name
        self.params = params.copy()

        # not present in ImportParams2D
        self.xedge = self.yedge = self.xcent = self.ycent = None

    ####################################################################
    # Follow functions are for setting parameters during reading of data

    def _paramXRange(self, cols):
        try:
            self.params.xrange = ( float(cols[1]), float(cols[2]) )
        except ValueError:
            raise Read2DError("xrange is not two numerical values")

    def _paramYRange(self, cols):
        try:
            self.params.yrange = ( float(cols[1]), float(cols[2]) )
        except ValueError:
            raise Read2DError("yrange is not two numerical values")

    def _getNumList(self, attr, cols):
        """Generic conversion of a list of numbers."""
        try:
            g = [float(v) for v in cols[1:]]
        except ValueError:
            raise Read2DError("%s is not a list of numerical values" % attr)
        if utils.checkOrder(g) == 0:
            raise Read2DError("%s are not ascending or descending" % attr)
        setattr(self, attr, g)

    def _paramInvertRows(self, cols):
        self.params.invertrows = True

    def _paramInvertCols(self, cols):
        self.params.invertcols = True

    def _paramTranspose(self, cols):
        self.params.transpose = True

    def _paramGridAtEdge(self, cols):
        self.params.gridatedge = True

    ####################################################################

    def readData(self, stream):
        """Read data from stream given

        stream consists of:
        optional:
         xrange A B   - set the range of x from A to B
         yrange A B   - set the range of y from A to B
         xedge A B... - list of x values (instead of xrange)
         yedge A B... - list of y values (instead of yrange)
         xcent A B... - list of x centres (instead of xrange)
         ycent A B... - list of y centres (instead of yrange)
         invertrows   - invert order of the rows
         invertcols   - invert order of the columns
         transpose    - swap rows and columns
         gridatedge   - positions of pixels are given at top and left
        then:
         matrix of columns and rows, separated by line endings
         the rows are in reverse-y order (highest y first)
         blank line stops reading for further datasets
        """

        settings = {
            'xrange': self._paramXRange,
            'yrange': self._paramYRange,
            'xedge': lambda cols: self._getNumList('xedge', cols),
            'yedge': lambda cols: self._getNumList('yedge', cols),
            'xcent': lambda cols: self._getNumList('xcent', cols),
            'ycent': lambda cols: self._getNumList('ycent', cols),
            'invertrows': self._paramInvertRows,
            'invertcols': self._paramInvertCols,
            'transpose': self._paramTranspose,
            'gridatedge': self._paramGridAtEdge,
            }

        rows = []
        # loop over lines
        while stream.newLine():
            cols = stream.allColumns()

            if len(cols) == 0:
                if len(rows) != 0:
                    # end of data
                    break
                continue

            # check to see whether parameter is set
            c = cols[0].lower()
            if c in settings:
                settings[c](cols)
                stream.flushLine()
                continue

            # read columns
            line = []
            while True:
                v = stream.nextColumn()
                if v is None:
                    break
                try:
                    line.append( float(v) )
                except ValueError:
                    raise Read2DError("Could not interpret number '%s'" % v)

            rows.insert(0, line)

        if self.params.gridatedge:

            if any( [getattr(self, x) is not None
                     for x in ("xedge", "yedge", "xcent", "ycent")] ):
                raise Read2DError(
                    "x|y grid|cent are incompatible with gridatedge")

            if len(rows) > 0:
                self.xcent = N.array(rows[-1])
            else:
                self.xcent = N.array([])
            self.ycent = N.array([r[0] for r in rows[:-1]])

            # chop out grid
            rows = [ r[1:] for r in rows[:-1] ]

        # dodgy formatting probably...
        if len(rows) == 0:
            raise Read2DError("No data could be imported for dataset")

        try:
            self.data = N.array(rows, dtype=N.float64)
        except ValueError:
            raise Read2DError("Could not convert data to 2D matrix")

        # obvious check
        if len(self.data.shape) != 2:
            raise Read2DError("Dataset was not 2D")

        if self.params.invertcols:
            self.data = self.data[:,::-1]
        if self.params.invertrows:
            self.data = self.data[::-1,:]

        # transpose matrix if requested
        if self.params.transpose:
            self.data = N.transpose(self.data).copy()
            self.xedge, self.yedge = self.xedge, self.yedge

        # check orders of coords - flip if wrong
        for attr in 'xedge', 'xcent', 'yedge', 'ycent':
            v = getattr(self, attr)
            if v is not None:
                order = utils.checkOrder(v)
                if order == 0:
                    raise Read2DError(
                        '%s must be ascending or descending' % attr)
                elif order == -1:
                    # flip direction of coord and data
                    setattr(self, attr, v[::-1])
                    if attr[0] == 'x':
                        self.data = self.data[:,::-1]
                    else:
                        self.data = self.data[::-1,:]

        # more sanity checks
        if ( (self.xedge is not None and
              len(self.xedge) != self.data.shape[1]+1) or
             (self.yedge is not None and
              len(self.yedge) != self.data.shape[0]+1) ):
            raise Read2DError("xedge and yedge lengths must be data shape+1")

        if ( (self.xcent is not None and
              len(self.xcent) != self.data.shape[1]) or
             (self.ycent is not None and
              len(self.ycent) != self.data.shape[0]) ):
            raise Read2DError("xcent and ycent lengths must be data shape")

    def setOutput(self, out, linkedfile=None):
        """Set the data in the output dict out
        """

        ds = datasets.Dataset2D(self.data,
                                xrange=self.params.xrange,
                                yrange=self.params.yrange,
                                xedge=self.xedge, yedge=self.yedge,
                                xcent=self.xcent, ycent=self.ycent)

        ds.linked = linkedfile

        fullname = self.params.prefix + self.name + self.params.suffix
        out[fullname] = ds

#####################################################################
# n-dimensional data reading

class ReadNDError(base.ImportingError):
    pass

class SimpleReadND(object):
    def __init__(self, name, params):
        """Read dataset with name given.
        params is a ImportParamsND object
        """
        self.name = name
        self.params = params.copy()

    ####################################################################

    def _paramTranspose(self, cols):
        self.params.transpose = True

    def _paramShape(self, cols):
        sizes = [int(x) for x in cols[1:]]
        self.params.shape = tuple(sizes)

    ####################################################################

    def readData(self, stream):
        """Read data from stream given

        stream consists of:
        optional:
         transpose    - swap rows and columns
         shape size1 size2 ... - give dimensions of dataset
        then:
         Matrix of columns and rows, separated by line endings.
         A single line is a single dimension
         Higher orders are given by using increasing numbers of
         separating newlines
        """

        settings = {
            'transpose': self._paramTranspose,
            'shape': self._paramShape,
            }

        vals = []

        # keep track of where we are in terms of index
        dimstack = []
        dimidx = 0

        # loop over lines
        while stream.newLine():
            cols = stream.allColumns()

            # check to see whether parameter is set
            if len(cols) > 0:
                c = cols[0].lower()
                if c in settings:
                    settings[c](cols)
                    stream.flushLine()
                    continue

            # read columns
            line = []
            while True:
                v = stream.nextColumn()
                if v is None:
                    break
                try:
                    line.append(float(v))
                except ValueError:
                    raise ReadNDError("Could not interpret number '%s'" % v)

            if len(line) > 0:
                # previous blank lines
                if dimidx != len(dimstack):
                    # insert new dimensions, if required
                    while dimidx < 0:
                        dimstack.insert(0, 0)
                        dimidx += 1
                        vals = [vals]

                    # move to next value at current level
                    dimstack[dimidx] += 1
                    for i in crange(dimidx+1, len(dimstack)):
                        dimstack[i] = 0
                    dimidx = len(dimstack)

                # lookup correct place in hierarchy and append
                v = vals
                for s in dimstack:
                    while s >= len(v):
                        v.append([])
                    v = v[s]
                v.append(N.array(line, dtype=N.float64))
            else:
                if len(vals) > 0:
                    dimidx -= 1

        if self.params.shape is not None:
            # flatten so we can reshape properly later (this is to
            # allow free form input with the shape option)
            fdata = []
            def flatten(d):
                for x in d:
                    if isinstance(x, list):
                        flatten(x)
                    else:
                        fdata.append(x)
            flatten(vals)
            vals = N.hstack((fdata))

        try:
            self.data = N.array(vals, dtype=N.float64)
        except ValueError:
            raise ReadNDError("Could not convert data to n-D matrix")

        # obvious check
        if self.data.ndim < 1:
            raise ReadNDError("Needs at least a 1D dataset")

        # for 1d data, an extra dimension is sometimes added
        if self.data.ndim == 2 and self.data.shape[0] == 1:
            self.data = self.data.reshape(self.data.shape[-1])

        # reshape if requested
        if self.params.shape is not None:
            self.data = self.data.reshape(self.params.shape)

        # transpose matrix if requested
        if self.params.transpose:
            self.data = N.transpose(self.data).copy()

    def setOutput(self, out, linkedfile=None):
        """Set the data in the output dict out
        """

        ds = datasets.DatasetND(self.data)
        ds.linked = linkedfile
        fullname = self.params.prefix + self.name + self.params.suffix
        out[fullname] = ds
