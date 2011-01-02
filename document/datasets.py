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

import re
from itertools import izip

import numpy as N

import simpleread
import operations
import readcsv

import veusz.qtall as qt4
import veusz.utils as utils
import veusz.setting as setting
import veusz.plugins as plugins

def _convertNumpy(a):
    """Convert to a numpy double if possible."""
    if a is None:
        # leave as None
        return None
    elif not isinstance(a, N.ndarray):
        # convert to numpy array
        return N.array(a, dtype=N.float64)
    else:
        # make conversion if numpy type is not correct
        if a.dtype != N.float64:
            return a.astype(N.float64)
        else:
            return a

def _convertNumpyAbs(a):
    """Convert to numpy 64 bit positive values, if possible."""
    if a is None:
        return None
    if not isinstance(a, N.ndarray):
        a = N.array(a, dtype=N.float64)
    elif a.dtype != N.float64:
        a = a.astype(N.float64)
    return N.abs(a)

def _convertNumpyNegAbs(a):
    """Convert to numpy 64 bit negative values, if possible."""
    if a is None:
        return None
    if not isinstance(a, N.ndarray):
        a = N.array(a, dtype=N.float64)
    elif a.dtype != N.float64:
        a = a.astype(N.float64)
    return -N.abs(a)

def _copyOrNone(a):
    """Return a copy if not None, or None."""
    if a is None:
        return None
    elif isinstance(a, N.ndarray):
        return N.array(a)
    elif isinstance(a, list):
        return list(a)

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

def datasetNameToDescriptorName(name):
    """Return descriptor name for dataset."""
    if re.match('^[0-9A-Za-z_]+$', name):
        return name
    else:
        return '`%s`' % name

class LinkedFileBase(object):
    """A base class for linked files containing common routines."""

    # filename is member

    def saveToFile(self, fileobj, relpath=None):
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

    def _reloadViaOperation(self, document, op):
        """Reload links using a supplied operation, op."""
        tempdoc = document.__class__()

        try:
            tempdoc.applyOperation(op)
        except Exception, ex:
            # if something breaks, record an error and return nothing
            document.log(unicode(ex))

            # find datasets which are linked using this link object
            # return errors for them
            errors = dict([(name, 1) for name, ds in document.data.iteritems()
                           if ds.linked is self])
            return ([], errors)            
            
        # delete datasets which are linked and imported here
        self._deleteLinkedDatasets(document)
        # move datasets into document
        read = self._moveReadDatasets(tempdoc, document)

        # return zero errors
        errors = dict( [(ds, 0) for ds in read] )

        return (read, errors)

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
        
        tempdoc = document.__class__()
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
        """Reload datasets linked to this file."""

        op = operations.OperationDataImport2D(
            self.datasets, filename=self.filename,
            xrange=self.xrange, yrange=self.yrange,
            transpose=self.transpose,
            invertrows=self.invertrows, invertcols=self.invertcols,
            prefix=self.prefix, suffix=self.suffix, encoding=self.encoding)
        return self._reloadViaOperation(document, op)

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

    def saveToFile(self, fileobj, relpath=None):
        '''Save the link to the document file.'''

        args = [self.dsname, self._getSaveFilename(relpath), self.hdu]
        args = [repr(i) for i in args]
        for c, a in izip(self.columns,
                         ('datacol', 'symerrcol',
                          'poserrcol', 'negerrcol')):
            if c is not None:
                args.append('%s=%s' % (a, repr(c)))
        args.append('linked=True')

        fileobj.write('ImportFITSFile(%s)\n' % ', '.join(args))

    def reloadLinks(self, document):
        '''Reload datasets linked to this file.'''

        op = operations.OperationDataImportFITS(
            self.dsname, self.filename, self.hdu,
            datacol = self.columns[0], symerrcol = self.columns[1],
            poserrcol = self.columns[2], negerrcol = self.columns[3],
            linked=True )

        # don't use applyoperation interface as we don't want to be undoable
        op.do(document)
        
        return ([self.dsname], {self.dsname: 0})

class LinkedCSVFile(LinkedFileBase):
    """A CSV file linked to datasets."""

    def __init__(self, filename, readrows=False,
                 delimiter=',', textdelimiter='"',
                 encoding='utf_8',
                 headerignore=0, blanksaredata=False,
                 prefix='', suffix=''):
        """Read CSV data from filename

        Read across rather than down if readrows
        headerignore is number of lines to ignore after each header
        blanksaredata treats blank cells as NaN values or empty strings
        Prepend prefix to dataset names if set.
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

    def saveToFile(self, fileobj, relpath=None):
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
        if self.headerignore > 0:
            params.append('headerignore=' + repr(self.headerignore))
        if self.blanksaredata:
            params.append('blanksaredata=True')

        fileobj.write('ImportFileCSV(%s)\n' % (', '.join(params)))
        
    def reloadLinks(self, document):
        """Reload any linked data from the CSV file."""

        # again, this is messy as we have to make sure we don't
        # overwrite any non-linked data

        op = operations.OperationDataImportCSV(
            self.filename, readrows=self.readrows,
            delimiter=self.delimiter,
            textdelimiter=self.textdelimiter,
            encoding=self.encoding,
            headerignore=self.headerignore,
            blanksaredata=self.blanksaredata,
            prefix=self.prefix, suffix=self.suffix )
        return self._reloadViaOperation(document, op)

class LinkedFilePlugin(LinkedFileBase):
    """Represent a file linked using an import plugin."""

    def __init__(self, plugin, filename, pluginparams, encoding='utf_8',
                 prefix='', suffix=''):
        """Setup the link with the plugin-read file."""
        self.plugin = plugin
        self.filename = filename
        self.encoding = encoding
        self.prefix = prefix
        self.suffix = suffix
        self.pluginparams = pluginparams

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the vsz document file."""

        params = [repr(self.plugin),
                  repr(self._getSaveFilename(relpath)),
                  'linked=True']
        if self.encoding != 'utf_8':
            params.append('encoding=' + repr(self.encoding))
        if self.prefix:
            params.append('prefix=' + repr(self.prefix))
        if self.suffix:
            params.append('suffix=' + repr(self.suffix))
        for name, val in self.pluginparams.iteritems():
            params.append('%s=%s' % (name, repr(val)))

        fileobj.write('ImportFilePlugin(%s)\n' % (', '.join(params)))

    def reloadLinks(self, document):
        """Reload data from file."""

        op = operations.OperationDataImportPlugin(
            self.plugin, self.filename, 
            encoding=self.encoding, prefix=self.prefix, suffix=self.suffix,
            **self.pluginparams)
        return self._reloadViaOperation(document, op)
        
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

    # class for representing part of this dataset
    subsetclass = None

    def saveLinksToSavedDoc(self, fileobj, savedlinks, relpath=None):
        '''Save the link to the saved document, if this dataset is linked.

        savedlinks is a dict containing any linked files which have
        already been written

        relpath is a directory to save linked files relative to
        '''

        # links should only be saved once
        if self.linked is not None and self.linked not in savedlinks:
            savedlinks[self.linked] = True
            self.linked.saveToFile(fileobj, relpath=relpath)

    def name(self):
        """Get dataset name."""
        for name, ds in self.document.data.iteritems():
            if ds == self:
                return name
        raise ValueError('Could not find self in document.data')

    def description(self, showlinked=True):
        """Get description of database."""
        return ""

    def uiConvertToDataItem(self, val):
        """Return a value cast to this dataset data type."""
        return None
    
    def _getItemHelper(self, key):
        """Help get arguments to constructor."""
        args = {}
        for col in self.columns:
            array = getattr(self, col)
            if array is not None:
                args[col] = array[key]
        return args

    def __getitem__(self, key):
        """Return a dataset based on this dataset

        e.g. dataset[5:100] - make a dataset based on items 5 to 99 inclusive
        """
        return type(self)(**self._getItemHelper(key))

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

    def canUnlink(self):
        """Can dataset be unlinked?"""
        return self.linked is not None

    def linkedInformation(self):
        """Return information about any linking for the user."""
        if self.linked is not None:
            name = self.linked.filename
        else:
            name = 'None'
        return 'Linked file: %s' % name

    def returnCopy(self):
        """Return an unlinked copy of self."""
        pass

    def renameable(self):
        """Is it possible to rename this dataset?"""
        return self.linked is None

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

        if not xrange:
            self.xrange = (0, data.shape[1])
        if not yrange:
            self.yrange = (0, data.shape[0])

    def indexToPoint(self, xidx, yidx):
        """Convert a set of indices to pixels in integers to
        floating point vals using xrange and yrange."""

        xfracpix = (xidx+0.5) * (1./self.data.shape[1])
        xfloat = xfracpix * (self.xrange[1] - self.xrange[0]) + self.xrange[0]
        yfracpix = (yidx+0.5) * (1./self.data.shape[0])
        yfloat = yfracpix * (self.yrange[1] - self.yrange[0]) + self.yrange[0]
        return xfloat, yfloat

    def getDataRanges(self):
        return self.xrange, self.yrange

    def saveToFile(self, fileobj, name):
        """Write the 2d dataset to the file given."""

        # return if there is a link
        if self.linked is not None:
            return

        fileobj.write("ImportString2D(%s, '''\n" % repr(name))
        fileobj.write("xrange %e %e\n" % self.xrange)
        fileobj.write("yrange %e %e\n" % self.yrange)

        # write rows backwards, so lowest y comes first
        for row in self.data[::-1]:
            s = ('%e ' * len(row)) % tuple(row)
            fileobj.write("%s\n" % (s[:-1],))

        fileobj.write("''')\n")

    def description(self, showlinked=True):
        """Get description of dataset."""
        text = self.name()
        text += ' (%ix%i)' % self.data.shape
        text += ', x=%g->%g' % tuple(self.xrange)
        text += ', y=%g->%g' % tuple(self.yrange)
        if self.linked and showlinked:
            text += ', linked to %s' % self.linked.filename
        return text

    def uiConvertToDataItem(self, val):
        """Return a value cast to this dataset data type."""
        if isinstance(val, basestring) or isinstance(val, qt4.QString):
            val, ok = setting.uilocale.toDouble(val)
            if ok: return val
            raise ValueError, "Invalid floating point number"
        return float(val)

    def returnCopy(self):
        return Dataset2D( N.array(self.data), self.xrange, self.yrange)

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
        
        # convert data to numpy arrays
        data = _convertNumpy(data)
        serr = _convertNumpyAbs(serr)
        perr = _convertNumpyAbs(perr)
        nerr = _convertNumpyNegAbs(nerr)

        # check the sizes of things match up
        s = data.shape
        for x in (serr, nerr, perr):
            if x is not None and x.shape != s:
                raise DatasetException('Lengths of error data do not match data')

        # finally assign data
        self.document = None
        self._invalidpoints = None
        self.linked = linked
        self.data = data
        self.serr = serr
        self.perr = perr
        self.nerr = nerr

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

    def saveToFile(self, fileobj, name):
        '''Save data to file.
        '''

        # return if there is a link
        if self.linked is not None:
            return

        # build up descriptor
        datasets = [self.data]

        descriptor = datasetNameToDescriptorName(name) + '(numeric)'
        if self.serr is not None:
            descriptor += ',+-'
            datasets.append(self.serr)
        if self.perr is not None:
            descriptor += ',+'
            datasets.append(self.perr)
        if self.nerr is not None:
            descriptor += ',-'
            datasets.append(self.nerr)

        fileobj.write( "ImportString(%s,'''\n" % repr(descriptor) )

        # write line line-by-line
        format = '%e ' * len(datasets)
        format = format[:-1] + '\n'
        for line in izip( *datasets ):
            fileobj.write( format % line )

        fileobj.write( "''')\n" )

    def uiConvertToDataItem(self, val):
        """Return a value cast to this dataset data type."""
        if isinstance(val, basestring) or isinstance(val, qt4.QString):
            val, ok = setting.uilocale.toDouble(val)
            if ok: return val
            raise ValueError, "Invalid floating point number"
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

    def returnCopy(self):
        """Return version of dataset with no linking."""
        return Dataset(data = _copyOrNone(self.data),
                       serr = _copyOrNone(self.serr),
                       perr = _copyOrNone(self.perr),
                       nerr = _copyOrNone(self.nerr))

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

    def changeValues(self, type, vals):
        if type == 'data':
            self.data = list(vals)
        else:
            raise ValueError, 'type does not contain an allowed value'

        self.document.setModified(True)
    
    def uiConvertToDataItem(self, val):
        """Return a value cast to this dataset data type."""
        return unicode(val)

    def saveToFile(self, fileobj, name):
        '''Save data to file.
        '''

        # don't save if a link
        if self.linked is not None:
            return

        descriptor = datasetNameToDescriptorName(name) + '(text)'
        fileobj.write( "ImportString(%s,r'''\n" % repr(descriptor) )
        for line in self.data:
            # need to "escape" ''' marks in text
            r = repr(line).replace("'''", "''' \"'''\" r'''") + '\n'
            fileobj.write(r)
        fileobj.write( "''')\n" )

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

    def returnCopy(self):
        """Returns version of dataset with no linking."""
        return DatasetText(self.data)

class DatasetExpressionException(DatasetException):
    """Raised if there is an error evaluating a dataset expression."""
    pass

# split expression on python operators or quoted `DATASET`
dataexpr_split_re = re.compile(r'(`.*?`|[\.+\-*/\(\)\[\],<>=!|%^~& ])')
# identify whether string is a quoted identifier
dataexpr_quote_re = re.compile(r'^`.*`$')
dataexpr_columns = {'data':True, 'serr':True, 'perr':True, 'nerr':True}

def _substituteDatasets(datasets, expression, thispart):
    """Substitute the names of datasets with calls to a function which will
    evaluate them.

    Returns (new expression, list of substituted datasets)

    This is horribly hacky, but python-2.3 can't use eval with dict subclass
    """

    # split apart the expression to look for dataset names
    # re could be compiled if this gets slow
    bits = dataexpr_split_re.split(expression)

    dslist = []
    for i, bit in enumerate(bits):
        # test whether there's an _data, _serr or such at the end of the name
        part = thispart

        if dataexpr_quote_re.match(bit):
            # quoted text, so remove backtick-"quotes"
            bit = bit[1:-1]

        bitbits = bit.split('_')
        if len(bitbits) > 1:
            if bitbits[-1] in dataexpr_columns:
                part = bitbits.pop(-1)
            bit = '_'.join(bitbits)

        if bit in datasets:
            # replace name with a function to call
            bits[i] = "_DS_(%s, %s)" % (repr(bit), repr(part))
            dslist.append(bit)

    return ''.join(bits), dslist

def _evaluateDataset(datasets, dsname, dspart):
    """Return the dataset given.

    dsname is the name of the dataset
    dspart is the part to get (e.g. data, serr)
    """
    if dspart in dataexpr_columns:
        val = getattr(datasets[dsname], dspart)
        if val is None:
            raise DatasetExpressionException(
                "Dataset '%s' does not have part '%s'" % (dsname, dspart))
        return val
    else:
        raise DatasetExpressionException(
            'Internal error - invalid dataset part')

_safeexpr = set()
def simpleEvalExpression(doc, expr, part='data'):
    """Evaluate expression and return data.

    part is 'data', 'serr', 'perr' or 'nerr' - these are the
    dataset parts which are evaluated by the expression
    """

    expr = _substituteDatasets(doc.data, expr, part)[0]

    if expr not in _safeexpr:
        if ( not setting.transient_settings['unsafe_mode'] and
             utils.checkCode(expr, securityonly=True) ):
            doc.log("Unsafe expression: %s\n" % expr)
            return N.array([])

    # for speed, track safe expressions
    _safeexpr.add(expr)

    env = doc.eval_context.copy()
    def evaluateDataset(dsname, dspart):
        return _evaluateDataset(doc.data, dsname, dspart)

    env['_DS_'] = evaluateDataset
    try:
        evalout = eval(expr, env)
    except Exception, ex:
        doc.log(unicode(ex))
        return N.array([])
    return evalout

class DatasetExpression(Dataset):
    """A dataset which is linked to another dataset by an expression."""

    def __init__(self, data=None, serr=None, nerr=None, perr=None,
                 parametric=None):
        """Initialise the dataset with the expressions given.

        parametric is option and can be (minval, maxval, steps) or None
        """

        self.document = None
        self.linked = None
        self._invalidpoints = None

        # store the expressions to use to generate the dataset
        self.expr = {}
        self.expr['data'] = data
        self.expr['serr'] = serr
        self.expr['nerr'] = nerr
        self.expr['perr'] = perr
        self.parametric = parametric

        self.cachedexpr = {}

        self.docchangeset = -1
        self.evaluated = {}

    def evaluateDataset(self, dsname, dspart):
        """Return the dataset given.
        
        dsname is the name of the dataset
        dspart is the part to get (e.g. data, serr)
        """
        return _evaluateDataset(self.document.data, dsname, dspart)
                    
    def _evaluatePart(self, expr, part):
        """Evaluate expression expr for part part."""
        # replace dataset names with calls
        expr = _substituteDatasets(self.document.data, expr, part)[0]

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

        # create dataset using parametric expression
        if self.parametric:
            p = self.parametric
            if p[2] >= 2:
                deltat = (p[1]-p[0]) / (p[2]-1)
                t = N.arange(p[2])*deltat + p[0]
            else:
                t = N.array([p[0]])
            environment['t'] = t

        # this fn gets called to return the value of a dataset
        environment['_DS_'] = self.evaluateDataset

        # actually evaluate the expression
        try:
            result = eval(expr, environment)
            evalout = N.array(result, N.float64)
        except Exception, ex:
            raise DatasetExpressionException(
                "Error evaluating expression: %s\n"
                "Error: %s" % (self.expr[part], unicode(ex)) )

        # make evaluated error expression have same shape as data
        if part != 'data':
            data = self.evaluated['data']
            if evalout.shape == ():
                # zero dimensional - expand to data shape
                evalout = N.resize(evalout, data.shape)
            else:
                # 1-dimensional - make it right size and trim
                oldsize = evalout.shape[0]
                evalout = N.resize(evalout, data.shape)
                evalout[oldsize:] = N.nan
        else:
            if evalout.shape == ():
                # zero dimensional - make a single point
                evalout = N.resize(evalout, 1)

        self.evaluated[part] = evalout

    def updateEvaluation(self):
        """Update evaluation of parts of dataset.
        Throws DatasetExpressionException if error
        """
        if self.docchangeset != self.document.changeset:
            # avoid infinite recursion!
            self.docchangeset = self.document.changeset

            # zero out previous values
            for part in self.columns:
                self.evaluated[part] = None

            # update all parts
            for part in self.columns:
                expr = self.expr[part]
                if expr is not None and expr.strip() != '':
                    self._evaluatePart(expr, part)

    def _propValues(self, part):
        """Check whether expressions need reevaluating,
        and recalculate if necessary."""
        try:
            self.updateEvaluation()
        except DatasetExpressionException, ex:
            self.document.log(unicode(ex))

        # catch case where error in setting data, need to return "real" data
        if self.evaluated['data'] is None:
            self.evaluated['data'] = N.array([])
        return self.evaluated[part]

    # expose evaluated data as properties
    # this allows us to recalculate the expressions on the fly
    data = property(lambda self: self._propValues('data'))
    serr = property(lambda self: self._propValues('serr'))
    perr = property(lambda self: self._propValues('perr'))
    nerr = property(lambda self: self._propValues('nerr'))

    def saveToFile(self, fileobj, name):
        '''Save data to file.
        '''

        parts = [repr(name), repr(self.expr['data'])]
        if self.expr['serr']:
            parts.append('symerr=%s' % repr(self.expr['serr']))
        if self.expr['nerr']:
            parts.append('negerr=%s' % repr(self.expr['nerr']))
        if self.expr['perr']:
            parts.append('poserr=%s' % repr(self.expr['perr']))
        if self.parametric is not None:
            parts.append('parametric=%s' % repr(self.parametric))

        parts.append('linked=True')

        s = 'SetDataExpression(%s)\n' % ', '.join(parts)
        fileobj.write(s)
        
    def __getitem__(self, key):
        """Return a dataset based on this dataset

        We override this from DatasetBase as it would return a
        DatsetExpression otherwise, not chopped sets of data.
        """
        return Dataset(**self._getItemHelper(key))

    def deleteRows(self, row, numrows):
        pass

    def insertRows(self, row, numrows, rowdata):
        pass

    def canUnlink(self):
        """Whether dataset can be unlinked."""
        return True

    def linkedInformation(self):
        """Return information about linking."""
        text = []
        if self.parametric:
            text.append('Linked parametric dataset')
        else:
            text.append('Linked expression dataset')
        for label, part in izip(self.column_descriptions,
                                self.columns):
            if self.expr[part]:
                text.append('%s: %s' % (label, self.expr[part]))

        if self.parametric:
            text.append("where t goes from %g:%g in %i steps" % self.parametric)

        return '\n'.join(text)

class DatasetRange(Dataset):
    """Dataset consisting of a range of values e.g. 1 to 10 in 10 steps."""

    def __init__(self, numsteps, data, serr=None, perr=None, nerr=None):
        """Construct dataset.

        numsteps: number of steps in range
        data, serr, perr and nerr are tuples containing (start, stop) values."""
        self.range_data = data
        self.range_serr = serr
        self.range_perr = perr
        self.range_nerr = nerr
        self.numsteps = numsteps

        for name in ('data', 'serr', 'perr', 'nerr'):
            val = getattr(self, 'range_%s' % name)
            if val is not None:
                minval, maxval = val
                if numsteps == 1:
                    vals = N.array( [minval] )
                else:
                    delta = (maxval - minval) / (numsteps-1)
                    vals = N.arange(numsteps)*delta + minval
            else:
                vals = None
            setattr(self, name, vals)

        self.document = None
        self.linked = None
        self._invalidpoints = None

    def saveToFile(self, fileobj, name):
        """Save dataset to file."""

        parts = [repr(name), repr(self.numsteps), repr(self.range_data)]
        if self.range_serr is not None:
            parts.append('symerr=%s' % repr(self.range_serr))
        if self.range_perr is not None:
            parts.append('poserr=%s' % repr(self.range_perr))
        if self.range_nerr is not None:
            parts.append('negerr=%s' % repr(self.range_nerr))
        parts.append('linked=True')

        s = 'SetDataRange(%s)\n' % ', '.join(parts)
        fileobj.write(s)

    def canUnlink(self):
        return True

    def linkedInformation(self):
        """Return information about linking."""
        text = ['Linked range dataset']
        for label, part in izip(self.column_descriptions,
                                self.columns):
            val = getattr(self, 'range_%s' % part)
            if val:
                text.append('%s: %g:%g' % (label, val[0], val[1]))
        return '\n'.join(text)

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
                    raise DatasetExpressionException(
                        'Variable spacings not yet supported '
                        'in constructing 2D datasets')
    return (uniquesorted[0], uniquesorted[-1], mindelta,
            int((uniquesorted[-1]-uniquesorted[0])/mindelta)+1)

class Dataset2DXYZExpression(Dataset2D):
    '''A 2d dataset with expressions for x, y and z.'''

    def __init__(self, exprx, expry, exprz):
        """Initialise dataset.

        Parameters are mathematical expressions based on datasets."""

        self.document = None
        self.linked = None
        self._invalidpoints = None
        self.lastchangeset = -1
        self.cacheddata = None
        
        # copy parameters
        self.exprx = exprx
        self.expry = expry
        self.exprz = exprz

        # cache x y and z expressions
        self.cachedexpr = {}

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

        evaluated = {}

        environment = self.document.eval_context.copy()
        environment['_DS_'] = self.evaluateDataset

        # evaluate the x, y and z expressions
        for name in ('exprx', 'expry', 'exprz'):
            expr = _substituteDatasets(self.document.data, getattr(self, name),
                                       'data')[0]

            # check expression if not checked before
            if self.cachedexpr.get(name) != expr:
                if ( not setting.transient_settings['unsafe_mode'] and
                     utils.checkCode(expr, securityonly=True) ):
                    raise DatasetExpressionException(
                        "Unsafe expression '%s'" % (
                            expr))
                self.cachedexpr[name] = expr

            try:
                evaluated[name] = eval(expr, environment)
            except Exception, e:
                raise DatasetExpressionException(
                    "Error evaluating expression: %s\n"
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

        # update changeset
        self.lastchangeset = self.document.changeset

        return self.cacheddata

    @property
    def xrange(self):
        """Get x range of data as a tuple (min, max)."""
        return self.getDataRanges()[0]
    
    @property
    def yrange(self):
        """Get y range of data as a tuple (min, max)."""
        return self.getDataRanges()[1]
        
    def getDataRanges(self):
        """Get both ranges of axis."""
        try:
            self.evalDataset()
            return (self._xrange, self._yrange)
        except DatasetExpressionException:
            return ( (0., 1.), (0., 1.) )
        
    @property
    def data(self):
        """Get data, or none if error."""
        try:
            return self.evalDataset()
        except DatasetExpressionException, ex:
            self.document.log(unicode(ex))
            return N.array( [[]] )

    def description(self, showlinked=True):
        # FIXME: dataeditdialog descriptions should be taken from here somewhere
        text = self.name()
        text += ' (%ix%i)' % self.data.shape
        text += ', x=%g->%g' % tuple(self.xrange)
        text += ', y=%g->%g' % tuple(self.yrange)

    def saveToFile(self, fileobj, name):
        '''Save expressions to file.
        '''

        s = 'SetData2DExpressionXYZ(%s, %s, %s, %s, linked=True)\n' % (
            repr(name), repr(self.exprx), repr(self.expry), repr(self.exprz) )
        fileobj.write(s)

    def canUnlink(self):
        """Can relationship be unlinked?"""
        return True

    def linkedInformation(self):
        """Return linking information."""
        return 'Linked 2D function: x=%s, y=%s, z=%s' % (
            self.exprx, self.expry, self.exprz)

class Dataset2DExpression(Dataset2D):
    """Evaluate an expression of 2d datasets."""

    def __init__(self, expr):
        """Create 2d expression dataset."""
        self.document = None
        self.linked = None
        self.expr = expr
        self.lastchangeset = -1
        self.cachedexpr = None

        if utils.checkCode(expr, securityonly=True) is not None:
            raise DatasetExpressionException("Unsafe expression '%s'" % expr)
        
    @property
    def data(self):
        """Return data, or empty array if error."""
        try:
            return self.evalDataset()[0]
        except DatasetExpressionException, ex:
            self.document.log(unicode(ex))
            return N.array([[]])

    @property
    def xrange(self):
        """Return x range."""
        try:
            return self.evalDataset()[1]
        except DatasetExpressionException, ex:
            self.document.log(unicode(ex))
            return [0., 1.]

    @property
    def yrange(self):
        """Return y range."""
        try:
            return self.evalDataset()[2]
        except DatasetExpressionException, ex:
            self.document.log(unicode(ex))
            return [0., 1.]

    def evalDataset(self):
        """Do actual evaluation."""

        if self.document.changeset == self.lastchangeset:
            return self._cacheddata

        environment = self.document.eval_context.copy()

        def getdataset(dsname, dspart):
            """Return the dataset given in document."""
            return _evaluateDataset(self.document.data, dsname, dspart)
        environment['_DS_'] = getdataset

        # substituted expression
        expr, datasets = _substituteDatasets(self.document.data, self.expr,
                                             'data')

        # check expression if not checked before
        if self.cachedexpr != expr:
            if ( not setting.transient_settings['unsafe_mode'] and
                 utils.checkCode(expr, securityonly=True) ):
                raise DatasetExpressionException(
                    "Unsafe expression '%s'" % (
                        expr))
            self.cachedexpr = expr

        # do evaluation
        try:
            evaluated = eval(expr, environment)
        except Exception, e:
            raise DatasetExpressionException(
                "Error evaluating expression: %s\n"
                "Error: %s" % (expr, str(e)) )

        # find 2d dataset dimensions
        dsdim = None
        for ds in datasets:
            d = self.document.data[ds]
            if d.dimensions == 2:
                dsdim = d
                break

        if dsdim is None:
            # use default if none found
            rangex = rangey = [0., 1.]
        else:
            # use 1st dataset otherwise
            rangex = dsdim.xrange
            rangey = dsdim.yrange

        self.lastchangeset = self.document.changeset
        self._cacheddata = evaluated, rangex, rangey
        return self._cacheddata

    def saveToFile(self, fileobj, name):
        '''Save expression to file.'''
        s = 'SetData2DExpression(%s, %s, linked=True)\n' % (
            repr(name), repr(self.expr) )
        fileobj.write(s)

    def canUnlink(self):
        """Can relationship be unlinked?"""
        return True

    def linkedInformation(self):
        """Return linking information."""
        return 'Linked 2D expression: %s' % self.expr


class Dataset2DXYFunc(Dataset2D):
    """Given a range of x and y, this is a dataset which is a function of
    this.
    """

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

        self.lastchangeset = -1

    @property
    def data(self):
        """Return data, or empty array if error."""
        try:
            return self.evalDataset()
        except DatasetExpressionException, ex:
            self.document.log(unicode(ex))
            return N.array([[]])

    def evalDataset(self):
        """Evaluate the 2d dataset."""

        if self.document.changeset == self.lastchangeset:
            return self.cacheddata

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
            raise DatasetExpressionException("Error evaluating expression: %s\n"
                                             "Error: %s" % (self.expr, str(e)) )

        # ensure we get an array out of this (in case expr is scalar)
        data = data + xstep*0

        self.cacheddata = data
        self.lastchangeset = self.document.changeset
        return data

    def saveToFile(self, fileobj, name):
        '''Save expressions to file.
        '''
        s = 'SetData2DXYFunc(%s, %s, %s, %s, linked=True)\n' % (
            repr(name), repr(self.xstep), repr(self.ystep), repr(self.expr) )
        fileobj.write(s)

    def canUnlink(self):
        """Can relationship be unlinked?"""
        return True

    def linkedInformation(self):
        """Return linking information."""
        return 'Linked 2D function: x=%g:%g:%g, y=%g:%g:%g, z=%s' % tuple(
            list(self.xstep) + list(self.ystep) + [self.expr])

class _DatasetPlugin(object):
    """Shared methods for dataset plugins."""

    def __init__(self, manager, ds):
        self.pluginmanager = manager
        self.pluginds = ds

    def getPluginData(self, attr):
        self.pluginmanager.update()
        return getattr(self.pluginds, attr)

    def linkedInformation(self):
        """Return information about how this dataset was created."""

        fields = []
        for name, val in self.pluginmanager.fields.iteritems():
            fields.append('%s: %s' % (unicode(name), unicode(val)))

        return '%s plugin dataset (fields %s), size %i' % (
            self.pluginmanager.plugin.name,
            ', '.join(fields),
            self.data.shape[0])

    def canUnlink(self):
        """Can relationship be unlinked?"""
        return True

    def deleteRows(self, row, numrows):
        pass

    def insertRows(self, row, numrows, rowdata):
        pass

    def saveToFile(self, fileobj, name):
        """Save plugin to file, if this is the first one."""

        # only try to save if this is the 1st dataset of this plugin
        # manager in the document, so that we don't save more than once
        docdatasets = set( self.document.data.values() )

        for ds in self.pluginmanager.veuszdatasets:
            if ds in docdatasets:
                if ds is self:
                    # is 1st dataset
                    self.pluginmanager.saveToFile(fileobj)
                return

class Dataset1DPlugin(_DatasetPlugin, Dataset):
    """Return 1D dataset from a plugin."""

    def __init__(self, manager, ds):
        _DatasetPlugin.__init__(self, manager, ds)
        Dataset.__init__(self, data=[])

    def __getitem__(self, key):
        return Dataset(**self._getItemHelper(key))

    # parent class sets these attributes, so override setattr to do nothing
    data = property( lambda self: self.getPluginData('data'),
                     lambda self, val: None )
    serr = property( lambda self: self.getPluginData('serr'),
                     lambda self, val: None )
    nerr = property( lambda self: self.getPluginData('nerr'),
                     lambda self, val: None )
    perr = property( lambda self: self.getPluginData('perr'),
                     lambda self, val: None )

class Dataset2DPlugin(_DatasetPlugin, Dataset2D):
    """Return 2D dataset from a plugin."""

    def __init__(self, manager, ds):
        _DatasetPlugin.__init__(self, manager, ds)
        Dataset2D.__init__(self, [[]])

    def __getitem__(self, key):
        return Dataset2D(self.data[key], xrange=self.xrange, yrange=self.yrange)
        
    data   = property( lambda self: self.getPluginData('data'),
                       lambda self, val: None )
    xrange = property( lambda self: self.getPluginData('rangex'),
                       lambda self, val: None )
    yrange = property( lambda self: self.getPluginData('rangey'),
                       lambda self, val: None )

class DatasetTextPlugin(_DatasetPlugin, DatasetText):
    """Return text dataset from a plugin."""

    def __init__(self, manager, ds):
        _DatasetPlugin.__init__(self, manager, ds)
        DatasetText.__init__(self, [])

    def __getitem__(self, key):
        return DatasetText(self.data[key])

    data = property( lambda self: self.getPluginData('data'),
                     lambda self, val: None )
