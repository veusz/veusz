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
###############################################################################

# $Id$

"""Classes to represent datasets."""

import itertools
import re

import numarray as N

import doc
import simpleread
import operations
import readcsv

def _convertNumarray(a):
    """Convert to a numarray if possible (doing copy)."""
    if a == None:
        return None
    elif type(a) != type(N.arange(1, type=N.Float64)):
        return N.array(a, type=N.Float64)
    else:
        return a.astype(N.Float64)

class LinkedFileBase(object):
    """A base class for linked files containing common routines."""

    def saveToFile(self, file):
        '''Save the link to the document file.'''
        pass

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

    def __init__(self, filename, descriptor, useblocks=False):
        '''Set up the linked file with the descriptor given.'''
        self.filename = filename
        self.descriptor = descriptor
        self.useblocks = useblocks

    def saveToFile(self, file):
        '''Save the link to the document file.'''

        params = [ repr(self.filename),
                   repr(self.descriptor),
                   'linked=True' ]

        if self.useblocks:
            params.append('useblocks=True')

        file.write('ImportFile(%s)\n' % (', '.join(params)))

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
        sr.readData( simpleread.FileStream(open(self.filename)),
                     useblocks=self.useblocks )
        sr.setInDocument(tempdoc, linkedfile=self)

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

    def saveToFile(self, file):
        '''Save the link to the document file.'''

        args = [repr(self.filename), repr(self.datasets)]
        for p in ('xrange', 'yrange', 'invertrows', 'invertcols', 'transpose'):
            v = getattr(self, p)
            if v != None:
                args.append( '%s=%s' % (p, repr(v)) )
        args.append('linked=True')

        file.write('ImportFile2D(%s)\n' % ', '.join(args))

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
                                                  transpose=self.transpose)
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

    def saveToFile(self, file):
        '''Save the link to the document file.'''

        args = [self.dsname, self.filename, self.hdu]
        args = [repr(i) for i in args]
        for c, a in itertools.izip(self.columns,
                                   ('datacol', 'symerrcol',
                                    'poserrcol', 'negerrcol')):
            if c != None:
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
                                                negerrcol = self.columns[3])

        # don't use applyoperation interface as we don't want this to be undoable
        op.do(document)
        
        return ([self.dsname], {self.dsname: 0})

class LinkedCSVFile(LinkedFileBase):
    """A CSV file linked to datasets."""

    def __init__(self, filename, readrows=False, prefix=None):
        """Read CSV data from filename

        Read across rather than down if readrows
        Prepend prefix to dataset names if set.
        """

        self.filename = filename
        self.readrows = readrows
        self.prefix = prefix

    def saveToFile(self, file):
        """Save the link to the document file."""

        params = [repr(self.filename),
                  'linked=True']
        if self.prefix:
            params.append('prefix=%s' % repr(self.prefix))
        if self.readrows:
            params.append('readrows=True')

        file.write('ImportFileCSV(%s)\n' % (', '.join(params)))
        
    def reloadLinks(self, document):
        """Reload any linked data from the CSV file."""

        # again, this is messy as we have to make sure we don't
        # overwrite any non-linked data

        tempdoc = doc.Document()
        csv = readcsv.ReadCSV(self.filename, readrows=self.readrows,
                              prefix=self.prefix)
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

    def saveLinksToSavedDoc(self, file, savedlinks):
        '''Save the link to the saved document, if this dataset is linked.

        savedlinks is a dict containing any linked files which have
        already been written
        '''

        # links should only be saved once
        if self.linked != None and self.linked not in savedlinks:
            savedlinks[self.linked] = True
            self.linked.saveToFile(file)

class Dataset2D(DatasetBase):
    '''Represents a two-dimensional dataset.'''

    # number of dimensions the dataset holds
    dimensions = 2

    def __init__(self, data, xrange=None, yrange=None):
        '''Create a two dimensional dataset based on data.

        data: 2d numarray of imaging data
        xrange: a tuple of (start, end) coordinates for x
        yrange: a tuple of (start, end) coordinates for y
        '''

        self.document = None
        self.linked = None
        self.data = _convertNumarray(data)

        self.xrange = xrange
        self.yrange = yrange

        if not self.xrange:
            self.xrange = (0, data.shape[0])
        if not self.yrange:
            self.yrange = (0, data.shape[1])

    def getDataRanges(self):
        return self.xrange, self.yrange

    def saveToFile(self, file, name):
        """Write the 2d dataset to the file given."""

        # return if there is a link
        if self.linked != None:
            return

        file.write("ImportString2D(%s, '''\n" % repr(name))
        file.write("xrange %e %e\n" % self.xrange)
        file.write("yrange %e %e\n" % self.yrange)

        # write rows backwards, so lowest y comes first
        for row in self.data[::-1]:
            s = ('%e ' * len(row)) % tuple(row)
            file.write("%s\n" % (s[:-1],))

        file.write("''')\n")

class Dataset(DatasetBase):
    '''Represents a dataset.'''

    # number of dimensions the dataset holds
    dimensions = 1

    def __init__(self, data = None, serr = None, nerr = None, perr = None,
                 linked = None):
        '''Initialise dataset with the sets of values given.

        The values can be given as numarray 1d arrays or lists of numbers
        linked optionally specifies a LinkedFile to link the dataset to
        '''
        
        self.document = None
        self.data = _convertNumarray(data)
        self.serr = _convertNumarray(serr)
        self.perr = _convertNumarray(perr)
        self.nerr = _convertNumarray(nerr)
        self.linked = linked

        # check the sizes of things match up
        s = self.data.shape
        for i in (self.serr, self.nerr, self.perr):
            if i != None and i.shape != s:
                raise DatasetException('Lengths of error data do not match data')

    def duplicate(self):
        """Return new dataset based on this one."""
        return Dataset(self.data, self.serr, self.nerr, self.perr, None)

    def hasErrors(self):
        '''Whether errors on dataset'''
        return self.serr != None or self.nerr != None or self.perr != None

    def getPointRanges(self):
        '''Get range of coordinates for each point in the form
        (minima, maxima).'''

        minvals = self.data.copy()
        maxvals = self.data.copy()

        if self.serr != None:
            minvals -= self.serr
            maxvals += self.serr

        if self.nerr != None:
            minvals += self.nerr

        if self.perr != None:
            maxvals += self.perr

        return (minvals, maxvals)

    def getRange(self):
        '''Get total range of coordinates.'''
        minvals, maxvals = self.getPointRanges()
        return ( N.minimum.reduce(minvals),
                 N.maximum.reduce(maxvals) )

    def empty(self):
        '''Is the data defined?'''
        return self.data == None or len(self.data) == 0

    def changeValues(self, type, vals):
        """Change the requested part of the dataset to vals.

        type == vals | serr | perr | nerr
        """
        if type == 'vals':
            self.data = vals
        elif type == 'serr':
            self.serr = vals
        elif type == 'nerr':
            self.nerr = vals
        elif type == 'perr':
            self.perr = vals
        else:
            raise ValueError, 'type does not contain an allowed value'

        # just a check...
        s = self.data.shape
        for i in (self.serr, self.nerr, self.perr):
            assert i == None or i.shape == s

        self.document.setModified(True)

    def saveToFile(self, file, name):
        '''Save data to file.
        '''

        # return if there is a link
        if self.linked != None:
            return

        # build up descriptor
        datasets = [self.data]
        descriptor = name
        if self.serr != None:
            descriptor += ',+-'
            datasets.append(self.serr)
        if self.perr != None:
            descriptor += ',+'
            datasets.append(self.perr)
        if self.nerr != None:
            descriptor += ',-'
            datasets.append(self.nerr)

        file.write( "ImportString(%s,'''\n" % repr(descriptor) )

        # write line line-by-line
        format = '%e ' * len(datasets)
        format = format[:-1] + '\n'
        for line in itertools.izip( *datasets ):
            file.write( format % line )

        file.write( "''')\n" )

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

class DatasetExpression(Dataset):
    """A dataset which is linked to another dataset by an expression."""

    def __init__(self, data=None, serr=None, nerr=None, perr=None):
        """Initialise the dataset with the expressions given."""

        self.document = None
        self.linked = None

        # store the expressions to use to generate the dataset
        self.expr = {}
        self.expr['data'] = data
        self.expr['serr'] = serr
        self.expr['nerr'] = nerr
        self.expr['perr'] = perr

        self.docchangeset = { 'data': None, 'serr': None,
                              'perr': None, 'nerr': None }
        self.evaluated = {}

        # set up a default environment
        self.environment = globals().copy()
        exec 'from numarray import *' in self.environment

        # this fn gets called to return the value of a dataset
        self.environment['_DS_'] = self._evaluateDataset

        # used to break the dataset expression into parts
        # to look for dataset names
        # basically this is most non-alphanumeric chars (except _)
        self.splitre = re.compile(r'([+\-*/\(\)\[\],<>=!|%^~&])')

    def _substituteDatasets(self, expression, thispart):
        """Subsitiute the names of datasets with calls to a function which will evaluate
        them.

        This is horribly hacky, but python-2.3 can't use eval with dict subclass
        """

        # split apart the expression to look for dataset names
        # re could be compiled if this gets slow
        bits = self.splitre.split(expression)

        datasets = self.document.data

        for i, bit in enumerate(bits):
            # test whether there's an _data, _serr or such at the end of the name
            part = thispart
            bitbits = bit.split('_')
            if len(bitbits) > 1:
                if bitbits[-1] in self.expr:
                    part = bitbits.pop(-1)
                bit = '_'.join(bitbits)

            if bit in datasets:
                # replace name with a function to call
                bits[i] = "_DS_(%s, %s)" % (repr(bit), repr(part))

        return ''.join(bits)

    def _evaluateDataset(self, dsname, dspart):
        """Return the dataset given.
        
        dsname is the name of the dataset
        dspart is the part to get (e.g. data, serr)
        """
        return getattr(self.document.data[dsname], dspart)
                    
    def _propValues(self, part):
        """Check whether expressions need reevaluating, and recalculate if necessary."""

        assert self.document != None

        # if document has been modified since the last invocation
        if self.docchangeset[part] != self.document.changeset:
            # avoid infinite recursion!
            self.docchangeset[part] = self.document.changeset
            expr = self.expr[part]
            self.evaluated[part] = None

            if expr != None and expr != '':
                # replace dataset names with calls (ugly hack)
                # but necessary for Python 2.3 as we can't replace
                # dict in eval by subclass
                expr = self._substituteDatasets(expr, part)
                env = self.environment.copy()
                
                try:
                    self.evaluated[part] = eval(expr, env)
                except Exception, e:
                    raise DatasetExpressionException("Error evaluating expession: %s\n"
                                                     "Error: %s" % (self.expr[part], str(e)) )

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

