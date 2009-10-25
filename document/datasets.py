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
###############################################################################

# $Id$

"""Classes to represent datasets."""

import itertools
import re

import numpy as N

import doc
import simpleread
import operations
import readcsv

import veusz.utils as utils
import veusz.setting as setting

def _convertNumpy(a):
    """Convert to a numpy double if possible."""
    if a is None:
        # leave as None
        return None
    elif not isinstance(a, N.ndarray):
        # convert to numpy array
        return N.array(a, dtype='float64')
    else:
        # make conversion if numpy type is not correct
        if a.dtype != N.dtype('float64'):
            return a.astype('float64')
        else:
            return a

def generateValidDatasetParts(*datasets):
    """Generator to return array of valid parts of datasets.

    Yields new datasets between rows which are invalid
    """

    # find NaNs and INFs in input dataset
    invalid = datasets[0].invalidDataPoints()
    minlen = invalid.shape[0]
    for ds in datasets[1:]:
        try:
            nextinvalid = ds.invalidDataPoints()
            minlen = min(nextinvalid.shape[0], minlen)
            invalid = N.logical_or(invalid[:minlen], nextinvalid[:minlen])
        except AttributeError:
            # if not a dataset
            pass

    # get indexes of invalid pounts
    indexes = invalid.nonzero()[0].tolist()

    # no bad points: optimisation
    if not indexes:
        yield datasets
        return

    # add on shortest length of datasets
    indexes.append( minlen )

    lastindex = 0
    for index in indexes:
        if index != lastindex:
            retn = []
            for ds in datasets:
                if ds is not None:
                    retn.append( ds[lastindex:index] )
                else:
                    retn.append( None )
            yield retn
        lastindex = index+1


class LinkedFileBase(object):
    """A base class for linked files containing common routines."""

    # filename is member

    def saveToFile(self, file, relpath=None):
        '''Save the link to the document file.'''
        pass

    def _getSaveFilename(self, relpath):
        """Get filename to write to save file.
        If relpath is a string, write relative to path given
        """
        if relpath:
            return utils.relpath(self.filename, relpath)
        else:
            return self.filename

    def reloadLinks(self, document):
        '''Reload datasets linked to this file.'''
        pass

    def _deleteLinkedDatasets(self, document):
        """Delete linked datasets from document linking to self."""

        for name, ds in document.data.items():
            if ds.linked == self:
                del document.data[name]

    def _moveReadDatasets(self, tempdoc, document):
        """Move datasets from tempdoc to document if they do not exist
        in the destination."""

        read = []
        for name, ds in tempdoc.data.items():
            if name not in document.data:
                read.append(name)
                document.data[name] = ds
                ds.document = document
                ds.linked = self
        document.setModified(True)
        return read

class LinkedFile(LinkedFileBase):
    '''Instead of reading data from a string, data can be read from
    a "linked file". This means the same document can be reloaded, and
    the data would be reread from the file.

    This class is used to store a link filename with the descriptor
    '''

    def __init__(self, filename, descriptor, useblocks=False,
                 prefix='', suffix='', ignoretext=False,
                 encoding='utf_8'):
        '''Set up the linked file with the descriptor given.'''
        self.filename = filename
        self.descriptor = descriptor
        self.useblocks = useblocks
        self.prefix = prefix
        self.suffix = suffix
        self.ignoretext = ignoretext
        self.encoding = encoding

    def saveToFile(self, fileobj, relpath=None):
        '''Save the link to the document file.
        If relpath is set, save links relative to path given
        '''

        params = [ repr(self._getSaveFilename(relpath)),
                   repr(self.descriptor),
                   'linked=True',
                   'ignoretext=' + repr(self.ignoretext) ]

        if self.encoding != 'utf_8':
            params.append('encoding=' + repr(self.encoding))
        if self.useblocks:
            params.append('useblocks=True')
        if self.prefix:
            params.append('prefix=' + repr(self.prefix))
        if self.suffix:
            params.append('suffix=' + repr(self.suffix))

        fileobj.write('ImportFile(%s)\n' % (', '.join(params)))

    def reloadLinks(self, document):
        '''Reload datasets linked to this file.

        Returns a tuple of
        - List of datasets read
        - Dict of tuples containing dataset names and number of errors
        '''

        # a bit clumsy, but we need to load this into a separate document
        # to make sure we do not overwrited non-linked data (which may
        # be specified in the descriptor)
        
        tempdoc = doc.Document()
        sr = simpleread.SimpleRead(self.descriptor)
        
        stream = simpleread.FileStream(
            utils.openEncoding(self.filename, self.encoding))

        sr.readData(stream,
                    useblocks=self.useblocks,
                    ignoretext=self.ignoretext)
        sr.setInDocument(tempdoc, linkedfile=self,
                         prefix=self.prefix, suffix=self.suffix)

        errors = sr.getInvalidConversions()

        self._deleteLinkedDatasets(document)
        read = self._moveReadDatasets(tempdoc, document)

        # returns list of datasets read, and a dict of variables with number
        # of errors
        return (read, errors)

class Linked2DFile(LinkedFileBase):
    '''Class representing a file linked to a 2d dataset.'''

    def __init__(self, filename, datasets):
        self.filename = filename
        self.datasets = datasets

        self.xrange = None
        self.yrange = None
        self.invertrows = None
        self.invertcols = None
        self.transpose = None
        self.prefix = ''
        self.suffix = ''
        self.encoding = 'utf_8'

    def saveToFile(self, fileobj, relpath=None):
        '''Save the link to the document file.'''

        args = [repr(self._getSaveFilename(relpath)), repr(self.datasets)]
        for p in ('xrange', 'yrange', 'invertrows', 'invertcols', 'transpose',
                  'prefix', 'suffix', 'encoding'):
            v = getattr(self, p)
            if (v is not None) and (v != ''):
                args.append( '%s=%s' % (p, repr(v)) )
        args.append('linked=True')

        fileobj.write('ImportFile2D(%s)\n' % ', '.join(args))

    def reloadLinks(self, document):
        '''Reload datasets linked to this file.
        '''

        # put data in a temporary document as above
        tempdoc = doc.Document()
        try:
            op = operations.OperationDataImport2D(self.datasets,
                                                  filename=self.filename,
                                                  xrange=self.xrange,
                                                  yrange=self.yrange,
                                                  invertrows=self.invertrows,
                                                  invertcols=self.invertcols,
                                                  transpose=self.transpose,
                                                  prefix=self.prefix,
                                                  suffix=self.suffix,
                                                  encoding=self.encoding)
            tempdoc.applyOperation(op)
        except simpleread.Read2DError:
            errors = {}
            for i in self.datasets:
                errors[i] = 1
            return ([], errors)

        self._deleteLinkedDatasets(document)
        read = self._moveReadDatasets(tempdoc, document)

        # zero errors
        errors = dict( [(ds, 0) for ds in read] )

        return (read, errors)

class LinkedFITSFile(LinkedFileBase):
    """Links a FITS file to the data."""
    
    def __init__(self, dsname, filename, hdu, columns):
        '''Initialise the linked file object

        dsname is name of dataset
        filename is filename to load data from
        hdu is the hdu to load the data from
        columns is a list of columns for data, sym, pos and neg data
        '''
        
        self.dsname = dsname
        self.filename = filename
        self.hdu = hdu
        self.columns = columns

    def saveToFile(self, file, relpath=None):
        '''Save the link to the document file.'''

        args = [self.dsname, self._getSaveFilename(relpath), self.hdu]
        args = [repr(i) for i in args]
        for c, a in itertools.izip(self.columns,
                                   ('datacol', 'symerrcol',
                                    'poserrcol', 'negerrcol')):
            if c is not None:
                args.append('%s=%s' % (a, repr(c)))
        args.append('linked=True')

        file.write('ImportFITSFile(%s)\n' % ', '.join(args))

    def reloadLinks(self, document):
        '''Reload datasets linked to this file.'''

        op = operations.OperationDataImportFITS(self.dsname, self.filename,
                                                self.hdu,
                                                datacol = self.columns[0],
                                                symerrcol = self.columns[1],
                                                poserrcol = self.columns[2],
                                                negerrcol = self.columns[3],
                                                linked=True)

        # don't use applyoperation interface as we don't want this to be undoable
        op.do(document)
        
        return ([self.dsname], {self.dsname: 0})

class LinkedCSVFile(LinkedFileBase):
    """A CSV file linked to datasets."""

    def __init__(self, filename, readrows=False,
                 delimiter=',', textdelimiter='"',
                 encoding='utf_8',
                 prefix='', suffix=''):
        """Read CSV data from filename

        Read across rather than down if readrows
        Prepend prefix to dataset names if set.
        """

        self.filename = filename
        self.readrows = readrows
        self.delimiter = delimiter
        self.textdelimiter = textdelimiter
        self.encoding = encoding
        self.prefix = prefix
        self.suffix = suffix

    def saveToFile(self, file, relpath=None):
        """Save the link to the document file."""

        params = [repr(self._getSaveFilename(relpath)),
                  'linked=True']
        if self.prefix:
            params.append('dsprefix=' + repr(self.prefix))
        if self.suffix:
            params.append('dssuffix=' + repr(self.suffix))
        if self.readrows:
            params.append('readrows=True')
        if self.encoding != 'utf_8':
            params.append('encoding=' + repr(self.encoding))
        if self.delimiter != ',':
            params.append('delimiter=' + repr(self.delimiter))
        if self.textdelimiter != '"':
            params.append('textdelimiter=' + repr(self.textdelimiter))

        file.write('ImportFileCSV(%s)\n' % (', '.join(params)))
        
    def reloadLinks(self, document):
        """Reload any linked data from the CSV file."""

        # again, this is messy as we have to make sure we don't
        # overwrite any non-linked data

        tempdoc = doc.Document()
        csv = readcsv.ReadCSV(self.filename, readrows=self.readrows,
                              delimiter=self.delimiter,
                              textdelimiter=self.textdelimiter,
                              encoding=self.encoding,
                              prefix=self.prefix, suffix=self.suffix)
        csv.readData()
        csv.setData(tempdoc, linkedfile=self)

        self._deleteLinkedDatasets(document)
        read = self._moveReadDatasets(tempdoc, document)

        # zero errors
        errors = dict( [(ds, 0) for ds in read] )

        return (read, errors)
        
class DatasetException(Exception):
    """Raised with dataset errors."""
    pass

class DatasetBase(object):
    """A base dataset class."""

    # number of dimensions the dataset holds
    dimensions = 0
    datatype = 'numeric'
    columns = ()
    column_descriptions = ()
    recreatable_dataset = False  # can recreate in create dialog

    def saveLinksToSavedDoc(self, file, savedlinks, relpath=None):
        '''Save the link to the saved document, if this dataset is linked.

        savedlinks is a dict containing any linked files which have
        already been written

        relpath is a directory to save linked files relative to
        '''

        # links should only be saved once
        if self.linked is not None and self.linked not in savedlinks:
            savedlinks[self.linked] = True
            self.linked.saveToFile(file, relpath=relpath)

    def name(self):
        """Get dataset name."""
        for name, ds in self.document.data.iteritems():
            if ds == self:
                break
        else:
            raise ValueError('Could not find self in document.data')
        return name

    def description(self, showlinked=True):
        """Get description of database."""
        return ""

    def convertToDataItem(self, val):
        """Return a value cast to this dataset data type."""
        return None

    def __getitem__(self, key):
        """Return a dataset based on this dataset

        e.g. dataset[5:100] - make a dataset based on items 5 to 99 inclusive
        """

        args = {}
        for col in self.columns:
            array = getattr(self, col)
            if array is not None:
                args[col] = array[key]
        
        return type(self)(**args)

    def __len__(self):
        """Return length of dataset."""

        return len(self.data)
    
    def deleteRows(self, row, numrows):
        """Delete numrows rows starting from row.
        Returns deleted rows as a dict of {column:data, ...}
        """
        pass

    def insertRows(self, row, numrows, rowdata):
        """Insert numrows rows starting from row.
        rowdata is a dict of {column: data}.
        """
        pass

class Dataset2D(DatasetBase):
    '''Represents a two-dimensional dataset.'''

    # number of dimensions the dataset holds
    dimensions = 2

    def __init__(self, data, xrange=None, yrange=None):
        '''Create a two dimensional dataset based on data.

        data: 2d numpy of imaging data
        xrange: a tuple of (start, end) coordinates for x
        yrange: a tuple of (start, end) coordinates for y
        '''

        self.document = None
        self.linked = None
        self.data = _convertNumpy(data)

        self.xrange = xrange
        self.yrange = yrange

        if not self.xrange:
            self.xrange = (0, data.shape[1])
        if not self.yrange:
            self.yrange = (0, data.shape[0])

    def getDataRanges(self):
        return self.xrange, self.yrange

    def saveToFile(self, file, name):
        """Write the 2d dataset to the file given."""

        # return if there is a link
        if self.linked is not None:
            return

        file.write("ImportString2D(%s, '''\n" % repr(name))
        file.write("xrange %e %e\n" % self.xrange)
        file.write("yrange %e %e\n" % self.yrange)

        # write rows backwards, so lowest y comes first
        for row in self.data[::-1]:
            s = ('%e ' * len(row)) % tuple(row)
            file.write("%s\n" % (s[:-1],))

        file.write("''')\n")

    def description(self, showlinked=True):
        """Get description of dataset."""
        text = self.name()
        text += ' (%ix%i)' % self.data.shape
        text += ', x=%g->%g' % tuple(self.xrange)
        text += ', y=%g->%g' % tuple(self.yrange)
        if self.linked and showlinked:
            text += ', linked to %s' % self.linked.filename
        return text

    def convertToDataItem(self, val):
        """Return a value cast to this dataset data type."""
        return float(val)

class Dataset(DatasetBase):
    '''Represents a dataset.'''

    # number of dimensions the dataset holds
    dimensions = 1
    columns = ('data', 'serr', 'nerr', 'perr')
    column_descriptions = ('Data', 'Sym. errors', 'Neg. errors', 'Pos. errors')

    def __init__(self, data = None, serr = None, nerr = None, perr = None,
                 linked = None):
        '''Initialise dataset with the sets of values given.

        The values can be given as numpy 1d arrays or lists of numbers
        linked optionally specifies a LinkedFile to link the dataset to
        '''
        
        self.document = None
        self._invalidpoints = None
        self.data = _convertNumpy(data)
        self.serr = _convertNumpy(serr)
        self.perr = _convertNumpy(perr)
        self.nerr = _convertNumpy(nerr)
        self.linked = linked

        # check the sizes of things match up
        s = self.data.shape
        for i in (self.serr, self.nerr, self.perr):
            if i is not None and i.shape != s:
                raise DatasetException('Lengths of error data do not match data')

    def description(self, showlinked=True):
        """Get description of dataset."""

        text = self.name()
        if self.serr is not None:
            text += ',+-'
        if self.perr is not None:
            text += ',+'
        if self.nerr is not None:
            text += ',-'
        text += ' (length %i)' % len(self.data)

        if self.linked and showlinked:
            text += ' linked to %s' % self.linked.filename
        return text

    def duplicate(self):
        """Return new dataset based on this one."""
        attrs = {}
        for attr in ('data', 'serr', 'nerr', 'perr'):
            data = getattr(self, attr)
            if data is not None:
                attrs[attr] = data.copy()
        
        return Dataset(**attrs)

    def invalidDataPoints(self):
        """Return a numpy bool detailing which datapoints are invalid."""
        if self._invalidpoints is None:
            # recalculate valid points
            self._invalidpoints = N.logical_not(N.isfinite(self.data))
            for error in self.serr, self.perr, self.nerr:
                if error is not None:
                    self._invalidpoints = N.logical_or(self._invalidpoints,
                                                       N.logical_not(N.isfinite(error)))

        return self._invalidpoints
    
    def hasErrors(self):
        '''Whether errors on dataset'''
        return (self.serr is not None or self.nerr is not None or
                self.perr is not None)

    def getPointRanges(self):
        '''Get range of coordinates for each point in the form
        (minima, maxima).'''

        minvals = self.data.copy()
        maxvals = self.data.copy()

        if self.serr is not None:
            minvals -= self.serr
            maxvals += self.serr

        if self.nerr is not None:
            minvals += self.nerr

        if self.perr is not None:
            maxvals += self.perr

        return ( minvals[N.isfinite(minvals)],
                 maxvals[N.isfinite(maxvals)] )

    def getRange(self):
        '''Get total range of coordinates. Returns None if empty.'''
        minvals, maxvals = self.getPointRanges()
        if len(minvals) > 0 and len(maxvals) > 0:
            return ( minvals.min(), maxvals.max() )
        else:
            return None

    def empty(self):
        '''Is the data defined?'''
        return self.data is None or len(self.data) == 0

    def changeValues(self, thetype, vals):
        """Change the requested part of the dataset to vals.

        thetype == data | serr | perr | nerr
        """
        self._invalidpoints = None
        if thetype in self.columns:
            setattr(self, thetype, vals)
        else:
            raise ValueError, 'thetype does not contain an allowed value'

        # just a check...
        s = self.data.shape
        for x in (self.serr, self.nerr, self.perr):
            assert x is None or x.shape == s

        self.document.setModified(True)

    def saveToFile(self, file, name):
        '''Save data to file.
        '''

        # return if there is a link
        if self.linked is not None:
            return

        # build up descriptor
        datasets = [self.data]
        descriptor = "%s(numeric)" % name
        if self.serr is not None:
            descriptor += ',+-'
            datasets.append(self.serr)
        if self.perr is not None:
            descriptor += ',+'
            datasets.append(self.perr)
        if self.nerr is not None:
            descriptor += ',-'
            datasets.append(self.nerr)

        file.write( "ImportString(%s,'''\n" % repr(descriptor) )

        # write line line-by-line
        format = '%e ' * len(datasets)
        format = format[:-1] + '\n'
        for line in itertools.izip( *datasets ):
            file.write( format % line )

        file.write( "''')\n" )

    def convertToDataItem(self, val):
        """Return a value cast to this dataset data type."""
        return float(val)

    def deleteRows(self, row, numrows):
        """Delete numrows rows starting from row.
        Returns deleted rows as a dict of {column:data, ...}
        """
        retn = {}
        for col in self.columns:
            coldata = getattr(self, col)
            if coldata is not None:
                retn[col] = coldata[row:row+numrows]
                setattr(self, col, N.delete( coldata, N.s_[row:row+numrows] ))
        
        return retn

    def insertRows(self, row, numrows, rowdata):
        """Insert numrows rows starting from row.
        rowdata is a dict of {column: data}.
        """
        for col in self.columns:
            coldata = getattr(self, col)
            data = N.zeros(numrows)
            if col in rowdata:
                data[:len(rowdata[col])] = N.array(rowdata[col])
            if coldata is not None:
                setattr(self, col, N.insert(coldata, [row]*numrows, data))

class DatasetText(DatasetBase):
    """Represents a text dataset: holding an array of strings."""

    dimensions = 1
    datatype = 'text'
    columns = ('data',)
    column_descriptions = ('Data',)

    def __init__(self, data=None, linked=None):
        """Initialise dataset with data given. Data are a list of strings."""

        self.data = list(data)
        self.linked = linked

    def description(self, showlinked=True):
        text = '%s (%i items)' % (self.name(), len(self.data))
        if self.linked and showlinked:
            text += ', linked to %s' % self.linked.filename
        return text

    def duplicate(self):
        return DatasetText(data=self.data) # data is copied by this constructor

    def changeValues(self, type, vals):
        if type == 'data':
            self.data = list(vals)
        else:
            raise ValueError, 'type does not contain an allowed value'

        self.document.setModified(True)
    
    def convertToDataItem(self, val):
        """Return a value cast to this dataset data type."""
        return unicode(val)

    def saveToFile(self, file, name):
        '''Save data to file.
        '''

        # don't save if a link
        if self.linked is not None:
            return

        descriptor = '%s(text)' % name
        file.write( "ImportString(%s,r'''\n" % repr(descriptor) )
        for line in self.data:
            # need to "escape" ''' marks in text
            r = repr(line).replace("'''", "''' \"'''\" r'''") + '\n'
            file.write(r)
        file.write( "''')\n" )

    def deleteRows(self, row, numrows):
        """Delete numrows rows starting from row.
        Returns deleted rows as a dict of {column:data, ...}
        """
        retn = {'data': self.data[row:row+numrows]}
        del self.data[row:row+numrows]
        return retn

    def insertRows(self, row, numrows, rowdata):
        """Insert numrows rows starting from row.
        rowdata is a dict of {column: data}.
        """
        data = rowdata.get('data', [])

        insdata = data + (['']*(numrows-len(data)))
        for d in insdata[::-1]:
            self.data.insert(row, d)

class DatasetExpressionException(DatasetException):
    """Raised if there is an error evaluating a dataset expression."""
    pass

# This code requires Python 2.4
# class _ExprDict(dict):
#     """A subclass of a dict which help evaluate expressions on the fly.

#     We do this because there's not an easy way to work out what order
#     the expressions should be evaluated in if there is interdependence

#     This is a subclass of dict so we can grab values from the document
#     if they match dataset names

#     part should be set to the part currently being evaluated
#     """

#     # allowed extensions to dataset names
#     _validextn = {'data': True, 'serr': True, 'nerr': True, 'perr': True}

#     def __init__(self, oldenvironment, document, part):
#         """Evaluate expressions in environment of document.

#         document is the document to look for data in
#         oldenvironment is a globals environment or suchlike
#         part is the part of the dataset to use by default (e.g. 'serr')
#         """

#         # copy environment from existing environment
#         self.update(oldenvironment)
#         self.document = document
#         self.part = part

#     def __getitem__(self, item):
#         """Return the value corresponding to key item from the dict

#         This works by checking for the dataset in the document first, and
#         returning that if it exists
#         """

#         # make a copy as we might change it if it contains an extn
#         i = item

#         # look for a valid extension to the dataset name
#         part = self.part
#         p = item.split('_')
#         if len(p) > 1:
#             if p[-1] in _ExprDict._validextn:
#                 part = p[-1]
#             i = '_'.join(p[:-1])

#         print "**", i, "**"
#         if i in self.document.data:
#             return getattr(self.document.data[i], part)
#         else:
#             # revert to old behaviour
#             return dict.__getitem__(self, item)

dataexpr_split_re = re.compile(r'([\.+\-*/\(\)\[\],<>=!|%^~& ])')
dataexpr_columns = {'data':True, 'serr':True, 'perr':True, 'nerr':True}

def _substituteDatasets(datasets, expression, thispart):
    """Subsitiute the names of datasets with calls to a function which will
    evaluate them.

    This is horribly hacky, but python-2.3 can't use eval with dict subclass
    """

    # split apart the expression to look for dataset names
    # re could be compiled if this gets slow
    bits = dataexpr_split_re.split(expression)

    for i, bit in enumerate(bits):
        # test whether there's an _data, _serr or such at the end of the name
        part = thispart
        bitbits = bit.split('_')
        if len(bitbits) > 1:
            if bitbits[-1] in dataexpr_columns:
                part = bitbits.pop(-1)
            bit = '_'.join(bitbits)

        if bit in datasets:
            # replace name with a function to call
            bits[i] = "_DS_(%s, %s)" % (repr(bit), repr(part))

    return ''.join(bits)

def _evaluateDataset(datasets, dsname, dspart):
    """Return the dataset given.

    dsname is the name of the dataset
    dspart is the part to get (e.g. data, serr)
    """
    if dspart in dataexpr_columns:
        val = getattr(datasets[dsname], dspart)
        if val is None:
            raise DatasetExpressionException("Dataset '%s' does not have part '%s'" % (dsname, dspart))
        return val
    else:
        raise DatasetExpressionException('Internal error - invalid dataset part')

class DatasetExpression(Dataset):
    """A dataset which is linked to another dataset by an expression."""

    recreatable_dataset = True

    def __init__(self, data=None, serr=None, nerr=None, perr=None):
        """Initialise the dataset with the expressions given."""

        self.document = None
        self.linked = None
        self._invalidpoints = None

        # store the expressions to use to generate the dataset
        self.expr = {}
        self.expr['data'] = data
        self.expr['serr'] = serr
        self.expr['nerr'] = nerr
        self.expr['perr'] = perr

        self.cachedexpr = {}

        self.docchangeset = { 'data': None, 'serr': None,
                              'perr': None, 'nerr': None }
        self.evaluated = {}

    def evaluateDataset(self, dsname, dspart):
        """Return the dataset given.
        
        dsname is the name of the dataset
        dspart is the part to get (e.g. data, serr)
        """
        return _evaluateDataset(self.document.data, dsname, dspart)
                    
    def _evaluatePart(self, expr, part):
        """Evaluate expression expr for part part."""
        # replace dataset names with calls (ugly hack)
        # but necessary for Python 2.3 as we can't replace
        # dict in eval by subclass
        expr = _substituteDatasets(self.document.data, expr, part)

        # check expression for nasties if it has changed
        if self.cachedexpr.get(part) != expr:
            if ( not setting.transient_settings['unsafe_mode'] and
                 utils.checkCode(expr, securityonly=True) ):
                raise DatasetExpressionException(
                    "Unsafe expression '%s' in %s part of dataset" % (
                        self.expr[part], part))
            self.cachedexpr[part] = expr

        # set up environment to evaluate expressions in
        environment = self.document.eval_context.copy()

        # this fn gets called to return the value of a dataset
        environment['_DS_'] = self.evaluateDataset

        # actually evaluate the expression
        try:
            e = self.evaluated[part] = N.array(
                eval(expr, environment),
                N.float64)
        except Exception, ex:
            raise DatasetExpressionException(
                "Error evaluating expession: %s\n"
                "Error: %s" % (self.expr[part], str(ex)) )

        # make evaluated error expression have same shape as data
        if part != 'data':
            data = getattr(self, 'data')
            if data.shape != e.shape:
                try:
                    # 1-dimensional - make it right size and trim
                    oldsize = len(e)
                    e = N.resize(e, data.shape)
                    e[oldsize:] = N.nan
                except TypeError:
                    # 0-dimensional - just make it repeat
                    e = N.resize(e, data.shape)
                self.evaluated[part] = e

    def _propValues(self, part):
        """Check whether expressions need reevaluating,
        and recalculate if necessary."""

        # if document has been modified since the last invocation
        if self.docchangeset[part] != self.document.changeset:
            # avoid infinite recursion!
            self.docchangeset[part] = self.document.changeset
            expr = self.expr[part]
            self.evaluated[part] = None

            if expr is not None and expr != '':
                self._evaluatePart(expr, part)

        # return the evaluated form of the expression
        return self.evaluated[part]

    def saveToFile(self, file, name):
        '''Save data to file.
        '''

        parts = [repr(name), repr(self.expr['data'])]
        if self.expr['serr']:
            parts.append('symerr=%s' % repr(self.expr['serr']))
        if self.expr['nerr']:
            parts.append('negerr=%s' % repr(self.expr['nerr']))
        if self.expr['perr']:
            parts.append('poserr=%s' % repr(self.expr['perr']))
        parts.append('linked=True')

        s = 'SetDataExpression(%s)\n' % ', '.join(parts)
        file.write(s)
        
    # expose evaluated data as properties
    # this allows us to recalculate the expressions on the fly
    data = property(lambda self: self._propValues('data'))
    serr = property(lambda self: self._propValues('serr'))
    perr = property(lambda self: self._propValues('perr'))
    nerr = property(lambda self: self._propValues('nerr'))

    def __getitem__(self, key):
        """Return a dataset based on this dataset

        We override this from DatasetBase as it would return a
        DatsetExpression otherwise, not chopped sets of data.
        """

        args = {}
        for col in self.columns:
            array = getattr(self, col)
            if array is not None:
                args[col] = array[key]
        
        return Dataset(**args)

    def deleteRows(self, row, numrows):
        pass

    def insertRows(self, row, numrows, rowdata):
        pass

def getSpacing(data):
    """Given a set of values, get minimum, maximum, step size
    and number of steps.
    
    Function allows that values may be missing

    Function assumes that at least one of the steps is the minimum step size
    (i.e. steps are not all multiples of some mininimum)
    """

    uniquesorted = N.unique1d(data)
    sigfactor = (uniquesorted[-1]-uniquesorted[0])*1e-13

    # differences between elements
    deltas = N.unique1d( N.ediff1d(uniquesorted) )

    mindelta = None
    for delta in deltas:
        if delta > sigfactor:
            if mindelta is None:
                # first delta
                mindelta = delta
            elif N.fabs(mindelta-delta) > sigfactor:
                # new delta - check is multiple of old delta
                ratio = delta/mindelta
                if N.fabs(int(ratio)-ratio) > 1e-3:
                    raise DatasetExpressionException('Variable spacings not yet supported in constructing 2D datasets')
    return (uniquesorted[0], uniquesorted[-1], mindelta,
            int((uniquesorted[-1]-uniquesorted[0])/mindelta)+1)

class Dataset2DXYZExpression(DatasetBase):
    '''A 2d dataset with expressions for x, y and z.'''

    # number of dimensions the dataset holds
    dimensions = 2

    def __init__(self, exprx, expry, exprz):
        """Initialise dataset.

        Parameters are mathematical expressions based on datasets."""

        self.document = None
        self.linked = None
        self._invalidpoints = None
        self.lastchangeset = -1
        self.cacheddata = None
        
        for expr in exprx, expry, exprz:
            if utils.checkCode(expr, securityonly=True) is not None:
                raise DatasetExpressionException("Unsafe expression '%s'" % expr)

        # copy parameters
        self.exprx = exprx
        self.expry = expry
        self.exprz = exprz

    def evaluateDataset(self, dsname, dspart):
        """Return the dataset given.
        
        dsname is the name of the dataset
        dspart is the part to get (e.g. data, serr)
        """
        return _evaluateDataset(self.document.data, dsname, dspart)
                    
    def evalDataset(self):
        """Return the evaluated dataset."""
        # return cached data if document unchanged
        if self.document.changeset == self.lastchangeset:
            return self.cacheddata

        # update changeset
        self.lastchangeset = self.document.changeset

        evaluated = {}

        environment = self.document.eval_context.copy()
        environment['_DS_'] = self.evaluateDataset

        # evaluate the x, y and z expressions
        for name in ('exprx', 'expry', 'exprz'):
            expr = _substituteDatasets(self.document.data, getattr(self, name),
                                       'data')

            try:
                evaluated[name] = eval(expr, environment)
            except Exception, e:
                raise DatasetExpressionException("Error evaluating expession: %s\n"
                                                 "Error: %s" % (expr, str(e)) )

        minx, maxx, stepx, stepsx = getSpacing(evaluated['exprx'])
        miny, maxy, stepy, stepsy = getSpacing(evaluated['expry'])

        # update cached x and y ranges
        self._xrange = (minx-stepx*0.5, maxx+stepx*0.5)
        self._yrange = (miny-stepy*0.5, maxy+stepy*0.5)
        
        self.cacheddata = N.empty( (stepsy, stepsx) )
        self.cacheddata[:,:] = N.nan
        xpts = ((1./stepx)*(evaluated['exprx']-minx)).astype('int32')
        ypts = ((1./stepy)*(evaluated['expry']-miny)).astype('int32')

        # this is ugly - is this really the way to do it?
        self.cacheddata.flat [ xpts + ypts*stepsx ] = evaluated['exprz']
        return self.cacheddata

    def getXrange(self):
        """Get x range of data as a tuple (min, max)."""
        self.evalDataset()
        return self._xrange
    
    def getYrange(self):
        """Get y range of data as a tuple (min, max)."""
        self.evalDataset()
        return self._yrange
        
    def getDataRanges(self):
        self.evalDataset()
        return (self._xrange, self._yrange)

    def description(self, showlinked=True):
        # FIXME: dataeditdialog descriptions should be taken from here somewhere
        text = self.name()
        text += ' (%ix%i)' % self.data.shape
        text += ', x=%g->%g' % tuple(self.xrange)
        text += ', y=%g->%g' % tuple(self.yrange)

    def saveToFile(self, file, name):
        '''Save expressions to file.
        '''

        s = 'SetData2DExpressionXYZ(%s, %s, %s, %s, linked=True)\n' % (
            repr(name), repr(self.exprx), repr(self.expry), repr(self.exprz) )
        file.write(s)

    data = property(evalDataset)
    xrange = property(getXrange)
    yrange = property(getYrange)

class Dataset2DXYFunc(DatasetBase):
    """Given a range of x and y, this is a dataset which is a function of
    this.
    """

    # number of dimensions the dataset holds
    dimensions = 2

    def __init__(self, xstep, ystep, expr):
        """Create 2d dataset:

        xstep: tuple(xmin, xmax, step)
        ystep: tuple(ymin, ymax, step)
        expr: expression of x and y
        """

        self.document = None
        self.linked = None
        self._invalidpoints = None

        self.xstep = xstep
        self.ystep = ystep
        self.expr = expr

        if utils.checkCode(expr, securityonly=True) is not None:
            raise DatasetExpressionException("Unsafe expression '%s'" % expr)
        
        self.xrange = (self.xstep[0] - self.xstep[2]*0.5,
                       self.xstep[1] + self.xstep[2]*0.5)
        self.yrange = (self.ystep[0] - self.ystep[2]*0.5,
                       self.ystep[1] + self.ystep[2]*0.5)

    @property
    def data(self):
        """Make the 2d dataset."""

        env = self.document.eval_context.copy()

        xarange = N.arange(self.xstep[0], self.xstep[1]+self.xstep[2],
                           self.xstep[2])
        yarange = N.arange(self.ystep[0], self.ystep[1]+self.ystep[2],
                           self.ystep[2])
        ystep, xstep = N.indices( (len(yarange), len(xarange)) )
        xstep = xarange[xstep]
        ystep = yarange[ystep]

        env['x'] = xstep
        env['y'] = ystep
        try:
            data = eval(self.expr, env)
        except Exception, e:
            raise DatasetExpressionException("Error evaluating expession: %s\n"
                                             "Error: %s" % (self.expr, str(e)) )
        return data

    def getDataRanges(self):
        return (self.xrange, self.yrange)

    def saveToFile(self, file, name):
        '''Save expressions to file.
        '''
        s = 'SetData2DXYFunc(%s, %s, %s, %s, linked=True)\n' % (
            repr(name), repr(self.xstep), repr(self.ystep), repr(self.expr) )
        file.write(s)
