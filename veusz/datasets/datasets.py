# -*- coding: utf-8 -*-
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

"""Classes to represent datasets."""

from __future__ import division
import re

import numpy as N

from ..compat import czip, crange, citems, cbasestr, cstr, crepr
from .. import qtall as qt4
from .. import utils
from .. import setting

def _(text, disambiguation=None, context="Datasets"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

def convertNumpy(a, dims=1):
    """Convert to a numpy double if possible.

    dims is number of dimensions to check for
    """
    if a is None:
        # leave as None
        return None
    elif isinstance(a, N.ndarray):
        # make conversion if numpy type is not correct
        if a.dtype != N.float64:
            a = a.astype(N.float64)
    else:
        # convert to numpy array
        a = N.array(a, dtype=N.float64)

    if a.ndim != dims:
        if a.ndim == 0:
            if dims == 1:
                a = a.reshape((1,))
            elif dims == 2:
                a = a.reshape((1,1))
            else:
                raise RuntimeError()
        else:
            raise ValueError("Only %i-dimensional arrays or lists allowed" % dims)
    return a

def convertNumpyAbs(a):
    """Convert to numpy 64 bit positive values, if possible."""
    if a is None:
        return None
    else:
        return N.abs( convertNumpy(a) )

def convertNumpyNegAbs(a):
    """Convert to numpy 64 bit negative values, if possible."""
    if a is None:
        return None
    else:
        return -N.abs( convertNumpy(a) )

def _copyOrNone(a):
    """Return a copy if not None, or None."""
    if a is None:
        return None
    elif isinstance(a, N.ndarray):
        return N.array(a)
    elif isinstance(a, list):
        return list(a)

def generateValidDatasetParts(datasets, breakds=True):
    """Generator to return array of valid parts of datasets.

    if breakds is True:
      Yields new datasets between rows which are invalid
    else:
      Yields single, filtered dataset
    """

    # find NaNs and INFs in input dataset
    invalid = datasets[0].invalidDataPoints()
    minlen = invalid.shape[0]
    for ds in datasets[1:]:
        if isinstance(ds, DatasetBase) and not ds.empty():
            nextinvalid = ds.invalidDataPoints()
            minlen = min(nextinvalid.shape[0], minlen)
            invalid = N.logical_or(invalid[:minlen], nextinvalid[:minlen])

    if breakds:
        # return multiple datasets, breaking at invalid values

        # get indexes of invalid points
        indexes = invalid.nonzero()[0].tolist()

        # no bad points: optimisation
        if not indexes:
            yield datasets
            return

        # add on shortest length of datasets
        indexes.append(minlen)

        lastindex = 0
        for index in indexes:
            if index != lastindex:
                retn = []
                for ds in datasets:
                    if ds is not None and (
                            not isinstance(ds, DatasetBase) or
                            not ds.empty()):
                        retn.append(ds[lastindex:index])
                    else:
                        retn.append(None)
                yield retn
            lastindex = index+1

    else:
        # in this mode we return single datasets where the invalid
        # values are masked out

        if not N.any(invalid):
            yield datasets
            return

        valid = N.logical_not(invalid)
        retn = []
        for ds in datasets:
            if ds is None:
                retn.append(None)
            else:
                retn.append(ds[valid])
        yield retn

def datasetNameToDescriptorName(name):
    """Return descriptor name for dataset."""
    if re.match('^[0-9A-Za-z_]+$', name):
        return name
    else:
        return '`%s`' % name

class DatasetException(Exception):
    """Raised with dataset errors."""
    pass

class DatasetBase(object):
    """Base class for all datasets."""

class DatasetConcreteBase(DatasetBase):
    """A base dataset class for datasets which are real, and not proxies,
    etc."""

    # number of dimensions the dataset holds
    dimensions = 0

    # datatype is fundamental type of data
    # displaytype is formatting suggestion for data
    datatype = displaytype = 'numeric'

    # dataset type to show to user
    dstype = 'Dataset'

    # list of columns in dataset (if any)
    columns = ()
    # use descriptions for columns
    column_descriptions = ()

    # can values be edited
    editable = False

    # class for representing part of this dataset
    subsetclass = None

    def __init__(self, linked=None):
        """Initialise common members."""
        # document member set when this dataset is set in document
        self.document = None

        # file this dataset is linked to
        self.linked = linked

        # tags applied to dataset
        self.tags = set()

        # A name for the dataset to show to the user.  This is
        # modified by the document on SetData, etc, but can be set for
        # temporary datasets.
        self.username = None

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

    def saveToFile(self, fileobj, name, mode='text', hdfgroup=None):
        """Save dataset to file."""
        self.saveDataRelationToText(fileobj, name)
        if self.linked is None:
            if mode == 'text':
                self.saveDataDumpToText(fileobj, name)
            elif mode == 'hdf5':
                self.saveDataDumpToHDF5(hdfgroup, name)

    def saveDataRelationToText(self, fileobj, name):
        """Save a dataset relation to a text stream fileobj.
        Not for datasets which are raw data."""

    def saveDataDumpToText(self, fileobj, name):
        """Save dataset to file if file is text and data
        is actually a set of data and not a relation
        """

    def saveDataDumpToHDF5(self, group, name):
        """Save dumped dataset to HDF5.
        group is the group to save it in (h5py group)
        """

    def userSize(self):
        """Return dimensions of dataset for user."""
        return ""

    def userPreview(self):
        """Return a small preview of the dataset for the user, e.g.
        1, 2, 3, ..., 4, 5, 6."""
        return None

    def description(self):
        """Get description of dataset."""
        return ""

    def uiConvertToDataItem(self, val):
        """Return a value cast to this dataset data type.
        We assume here it is a float, so override if not
        """
        if isinstance(val, cbasestr):
            val, ok = setting.uilocale.toDouble(val)
            if ok: return val
            raise ValueError("Invalid floating point number")
        return float(val)

    def uiDataItemToData(self, val):
        """Return val converted to data."""
        return float(val)

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
        return self.returnCopyWithNewData(**self._getItemHelper(key))

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
        if self.linked is None:
            return _('Linked file: None')
        else:
            return _('Linked file: %s') % self.linked.filename

    def returnCopy(self):
        """Return an unlinked copy of self."""
        pass

    def returnCopyWithNewData(self, **args):
        """Return copy with new data given."""
        pass

    def renameable(self):
        """Is it possible to rename this dataset?"""
        return self.linked is None

    def datasetAsText(self, fmt='%g', join='\t'):
        """Return dataset as text (for use by user)."""
        return ''

def regularGrid(vals):
    '''Are the values equally spaced?'''
    if len(vals) < 2:
        return False
    vals = N.array(vals)
    deltas = vals[1:] - vals[:-1]
    return N.all(N.abs(deltas - deltas[0]) < (deltas[0]*1e-5))

class Dataset2DBase(DatasetConcreteBase):
    """Ancestor for 2D datasets."""

    # number of dimensions the dataset holds
    dimensions = 2

    # dataset type
    dstype = _('2D')

    # subclasses must define data, x/yrange, x/yedge, x/ycent as
    # attributes or properties

    def isLinearImage(self):
        """Are these simple linear pixels?"""
        return ( self.xedge is None and self.yedge is None and
                 self.xcent is None and self.ycent is None )

    def getPixelEdges(self, scalefnx=None, scalefny=None):
        """Return edges for x and y pixels.

        scalefnx/y: function to convert values to plotted pixel scale
                    (used to calculate edges from centres on screen)
        """

        def fromcentres(vals, scalefn):
            """Calculate edges from centres."""
            if scalefn:
                vals = scalefn(vals)

            if len(vals) == 0:
                e = []
            elif len(vals) == 1:
                if vals[0] != 0:
                    e = [0, vals[0]*2]
                else:
                    e = [0, 1]
            else:
                e = N.concatenate((
                    [vals[0] - 0.5*(vals[1]-vals[0])],
                    0.5*(vals[:-1] + vals[1:]),
                    [vals[-1] + 0.5*(vals[-1]-vals[-2])]
                ))
            return N.array(e)

        if self.xedge is not None:
            xg = self.xedge
            if scalefnx:
                xg = scalefnx(xg)
        elif self.xcent is not None:
            xg = fromcentres(self.xcent, scalefnx)
        else:
            xg = N.linspace(self.xrange[0], self.xrange[1],
                            self.data.shape[1]+1)
            if scalefnx:
                xg = scalefnx(xg)

        if self.yedge is not None:
            yg = self.yedge
            if scalefny:
                yg = scalefny(yg)
        elif self.ycent is not None:
            yg = fromcentres(self.ycent, scalefny)
        else:
            yg = N.linspace(self.yrange[0], self.yrange[1],
                            self.data.shape[0]+1)
            if scalefny:
                yg = scalefny(yg)

        return xg, yg

    def getPixelCentres(self):
        """Return lists of pixel centres in x and y."""

        yw, xw = self.data.shape

        if self.xcent is not None:
            xc = self.xcent
        elif self.xedge is not None:
            xc = 0.5*(self.xedge[:-1]+self.xedge[1:])
        else:
            xc = (N.arange(xw) + 0.5) * (
                (self.xrange[1]-self.xrange[0])/xw) + self.xrange[0]

        if self.ycent is not None:
            yc = self.ycent
        elif self.yedge is not None:
            yc = 0.5*(self.yedge[:-1]+self.yedge[1:])
        else:
            yc = (N.arange(yw) + 0.5) * (
                (self.yrange[1]-self.yrange[0])/yw) + self.yrange[0]

        return xc, yc

    def getDataRanges(self):
        """Return ranges of x and y data (as tuples)."""
        xe, ye = self.getPixelEdges()
        return (xe[0], xe[-1]), (ye[0], ye[-1])

    def datasetAsText(self, fmt='%g', join='\t'):
        """Return dataset as text.
        fmt is the format specifier to use
        join is the string to separate the items
        """
        format = ((fmt+join) * (self.data.shape[1]-1)) + fmt + '\n'

        # write rows backwards, so lowest y comes first
        lines = []
        for row in self.data[::-1]:
            line = format % tuple(row)
            lines.append(line)
        return ''.join(lines)

    def userSize(self):
        """Return dimensions of dataset for user."""
        return u'%i×%i' % self.data.shape

    def userPreview(self):
        """Return preview of data."""
        return dsPreviewHelper(self.data.flatten())

    def description(self):
        """Get description of dataset."""

        xr, yr = self.getDataRanges()
        text = _(u"2D (%i×%i), numeric, x=%.4g->%.4g, y=%.4g->%.4g") % (
            self.data.shape[0], self.data.shape[1],
            xr[0], xr[1], yr[0], yr[1])
        return text

    def returnCopy(self):
        return Dataset2D( N.array(self.data),
                          xrange=self.xrange, yrange=self.yrange,
                          xedge=self.xedge, yedge=self.yedge,
                          xcent=self.xcent, ycent=self.ycent )

    def returnCopyWithNewData(self, **args):
        return Dataset2D(**args)

class Dataset2D(Dataset2DBase):
    '''Represents a two-dimensional dataset.'''

    editable = True

    def __init__(self, data=None, xrange=None, yrange=None,
                 xedge=None, yedge=None,
                 xcent=None, ycent=None):
        '''Create a two dimensional dataset based on data.

        data: 2d numpy of imaging data

        Range specfied by:
         xrange: a tuple of (start, end) coordinates for x
         yrange: a tuple of (start, end) coordinates for y
        _or_
         xedge: list of values start..end (npix+1 values)
         yedge: list of values start..end (npix+1 values)
        _or_
         xcent: list of values (npix values)
         ycent: list of values (npix values)
        '''

        Dataset2DBase.__init__(self)

        self.data = convertNumpy(data, dims=2)

        # try to regularise data if possible
        # by converting regular grids to ranges
        if xedge is not None and regularGrid(xedge):
            xrange = (xedge[0], xedge[-1])
            xedge = None
        if yedge is not None and regularGrid(yedge):
            yrange = (yedge[0], yedge[-1])
            yedge = None
        if xcent is not None and regularGrid(xcent):
            delta = 0.5*(xcent[1]-xcent[0])
            xrange = (xcent[0]-delta, xcent[-1]+delta)
            xcent = None
        if ycent is not None and regularGrid(ycent):
            delta = 0.5*(ycent[1]-ycent[0])
            yrange = (ycent[0]-delta, ycent[-1]+delta)
            ycent = None

        self.xrange = self.yrange = None
        self.xedge = self.yedge = self.xcent = self.ycent = None

        if xrange is not None:
            self.xrange = tuple(xrange)
        elif xedge is not None:
            self.xedge = N.array(xedge)
        elif xcent is not None:
            self.xcent = N.array(xcent)
        elif self.data is not None:
            self.xrange = (0, self.data.shape[1])
        else:
            self.xrange = (0., 1.)

        if yrange is not None:
            self.yrange = tuple(yrange)
        elif yedge is not None:
            self.yedge = N.array(yedge)
        elif ycent is not None:
            self.ycent = N.array(ycent)
        elif self.data is not None:
            self.yrange = (0, self.data.shape[0])
        else:
            self.yrange = (0., 1.)

    def saveDataDumpToText(self, fileobj, name):
        """Write the 2d dataset to the file given."""

        fileobj.write("ImportString2D(%s, '''\n" % crepr(name))
        if self.xcent is not None:
            fileobj.write("xcent %s\n" %
                          " ".join(("%e" % v for v in self.xcent)) )
        elif self.xedge is not None:
            fileobj.write("xedge %s\n" %
                          " ".join(("%e" % v for v in self.xedge)) )
        else:
            fileobj.write("xrange %e %e\n" % tuple(self.xrange))

        if self.ycent is not None:
            fileobj.write("ycent %s\n" %
                          " ".join(("%e" % v for v in self.ycent)) )
        elif self.yedge is not None:
            fileobj.write("yedge %s\n" %
                          " ".join(("%e" % v for v in self.yedge)) )
        else:
            fileobj.write("yrange %e %e\n" % tuple(self.yrange))

        fileobj.write(self.datasetAsText(fmt='%e', join=' '))
        fileobj.write("''')\n")

    def saveDataDumpToHDF5(self, group, name):
        """Save 2D data in hdf5 file."""

        tdgrp = group.create_group(utils.escapeHDFDataName(name))
        tdgrp.attrs['vsz_datatype'] = '2d'

        for v in ('data', 'xcent', 'xedge', 'ycent',
                  'yedge', 'xrange', 'yrange'):
            if getattr(self, v) is not None:
                tdgrp[v] = getattr(self, v)

                # map attributes for importing
                if v != 'data':
                    tdgrp['data'].attrs['vsz_' + v] = tdgrp[v].ref

        # unicode text not stored properly unless encoded
        tdgrp['data'].attrs['vsz_name'] = name.encode('utf-8')

def dsPreviewHelper(d):
    """Get preview of numpy data d."""
    if d.shape[0] <= 6:
        line1 = ', '.join( ['%.3g' % x for x in d] )
    else:
        line1 = ', '.join( ['%.3g' % x for x in d[:3]] +
                           [ '...' ] +
                           ['%.3g' % x for x in d[-3:]] )

    try:
        line2 = _('mean: %.3g, min: %.3g, max: %.3g') % (
            N.nansum(d) / N.isfinite(d).sum(),
            N.nanmin(d),
            N.nanmax(d))
    except (ValueError, ZeroDivisionError):
        # nanXXX returns error if no valid data points
        return line1
    return line1 + '\n' + line2

class Dataset1DBase(DatasetConcreteBase):
    """Base for 1D datasets."""

    # number of dimensions the dataset holds
    dimensions = 1
    columns = ('data', 'serr', 'nerr', 'perr')
    column_descriptions = (_('Data'), _('Sym. errors'), _('Neg. errors'),
                           _('Pos. errors') )
    dstype = _('1D')

    # subclasses must define .data, .serr, .perr, .nerr

    def userSize(self):
        """Size of dataset."""
        return str( self.data.shape[0] )

    def userPreview(self):
        """Preview of data."""
        return dsPreviewHelper(self.data)

    def description(self):
        """Get description of dataset."""

        if self.serr is not None:
            templ = _("1D (length %i, symmetric errors)")
        elif self.perr is not None or self.nerr is not None:
            templ = _("1D (length %i, asymmetric errors)")
        else:
            templ = _("1D (length %i)")
        return templ % len(self.data)

    def invalidDataPoints(self):
        """Return a numpy bool detailing which datapoints are invalid."""

        valid = N.isfinite(self.data)
        for error in self.serr, self.perr, self.nerr:
            if error is not None:
                valid = N.logical_and(valid, N.isfinite(error))
        return N.logical_not(valid)

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

    def rangeVisit(self, fn):
        '''Call fn on data points and error values, in order to get range.'''
        fn(self.data)
        if self.serr is not None:
            fn(self.data - self.serr)
            fn(self.data + self.serr)
        if self.nerr is not None:
            fn(self.data + self.nerr)
        if self.perr is not None:
            fn(self.data + self.perr)

    def empty(self):
        '''Is the data defined?'''
        return self.data is None or len(self.data) == 0

    def datasetAsText(self, fmt='%g', join='\t'):
        """Return data as text."""

        # work out which columns to write
        cols = []
        for c in (self.data, self.serr, self.perr, self.nerr):
            if c is not None:
                cols.append(c)

        # format statement
        format = (fmt + join) * (len(cols)-1) + fmt + '\n'

        # do the conversion
        lines = []
        for line in czip(*cols):
            lines.append( format % line )
        return ''.join(lines)

    def returnCopy(self):
        """Return version of dataset with no linking."""
        return Dataset(data = _copyOrNone(self.data),
                       serr = _copyOrNone(self.serr),
                       perr = _copyOrNone(self.perr),
                       nerr = _copyOrNone(self.nerr))

    def returnCopyWithNewData(self, **args):
        """Return dataset of same type using the column data given."""
        return Dataset(**args)

class Dataset(Dataset1DBase):
    '''Represents a dataset.'''

    editable = True

    def __init__(self, data = None, serr = None, nerr = None, perr = None,
                 linked = None):
        '''Initialise dataset with the sets of values given.

        The values can be given as numpy 1d arrays or lists of numbers
        linked optionally specifies a LinkedFile to link the dataset to
        '''

        Dataset1DBase.__init__(self, linked=linked)

        # convert data to numpy arrays
        self.data = convertNumpy(data)
        self.serr = convertNumpyAbs(serr)
        self.perr = convertNumpyAbs(perr)
        self.nerr = convertNumpyNegAbs(nerr)

        # check the sizes of things match up
        s = self.data.shape
        for x in self.serr, self.nerr, self.perr:
            if x is not None and x.shape != s:
                raise DatasetException('Lengths of error data do not match data')

    def changeValues(self, thetype, vals):
        """Change the requested part of the dataset to vals.

        thetype == data | serr | perr | nerr
        """
        if thetype in self.columns:
            setattr(self, thetype, vals)
        else:
            raise ValueError('thetype does not contain an allowed value')

        # just a check...
        s = self.data.shape
        for x in (self.serr, self.nerr, self.perr):
            assert x is None or x.shape == s

        # tell the document that we've changed
        self.document.modifiedData(self)

    def saveDataDumpToText(self, fileobj, name):
        '''Save data to file.
        '''

        # build up descriptor
        descriptor = datasetNameToDescriptorName(name) + '(numeric)'
        if self.serr is not None:
            descriptor += ',+-'
        if self.perr is not None:
            descriptor += ',+'
        if self.nerr is not None:
            descriptor += ',-'

        fileobj.write( "ImportString(%s,'''\n" % crepr(descriptor) )
        fileobj.write( self.datasetAsText(fmt='%e', join=' ') )
        fileobj.write( "''')\n" )

    def saveDataDumpToHDF5(self, group, name):
        """Save dataset to HDF5."""

        # store as a group to simplify things
        odgrp = group.create_group(utils.escapeHDFDataName(name))
        odgrp.attrs['vsz_datatype'] = '1d'

        for key, suffix in (
                ('data', ''), ('serr', ' (+-)'),
                ('perr', ' (+)'), ('nerr', ' (-)')):
            if getattr(self, key) is not None:
                odgrp[key] = getattr(self, key)
                odgrp[key].attrs['vsz_name'] = (name + suffix).encode('utf-8')

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

        self.document.modifiedData(self)
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
                newdata = N.insert(coldata, [row]*numrows, data)
                setattr(self, col, newdata)

        self.document.modifiedData(self)

class DatasetDateTimeBase(Dataset1DBase):
    """Dataset holding dates and times."""

    columns = ('data',)
    column_descriptions = (_('Data'),)

    dstype = _('Date')
    displaytype = 'date'

    def description(self):
        return _('Date/time (length %i)') % len(self.data)

    def returnCopy(self):
        """Returns version of dataset with no linking."""
        return DatasetDateTime(data=N.array(self.data))

    def returnCopyWithNewData(self, **args):
        """Return dataset of same type using the column data given."""
        return DatasetDateTime(**args)

    def uiConvertToDataItem(self, val):
        """Return a value cast to this dataset data type."""
        if isinstance(val, cbasestr):
            v = utils.dateStringToDate( cstr(val) )
            if not N.isfinite(v):
                try:
                    v = float(val)
                except ValueError:
                    pass
            return v
        else:
            return N.nan

    def uiDataItemToData(self, val):
        """Return val converted to data."""
        return utils.dateFloatToString(val)

    def datasetAsText(self, fmt=None, join=None):
        """Return data as text."""
        lines = [ utils.dateFloatToString(val) for val in self.data ]
        lines.append('')
        return '\n'.join(lines)

class DatasetDateTime(DatasetDateTimeBase):
    """Standard date/time class for use by humans."""

    editable = True

    def __init__(self, data=None, linked=None):
        DatasetDateTimeBase.__init__(self, linked=linked)

        self.data = convertNumpy(data)
        self.perr = self.nerr = self.serr = None

    def saveDataDumpToText(self, fileobj, name):
        '''Save data to file.
        '''
        descriptor = datasetNameToDescriptorName(name) + '(date)'
        fileobj.write( "ImportString(%s,'''\n" % crepr(descriptor) )
        fileobj.write( self.datasetAsText() )
        fileobj.write( "''')\n" )

    def saveDataDumpToHDF5(self, group, name):
        """Save date data to hdf5 file."""
        dgrp = group.create_group(utils.escapeHDFDataName(name))
        dgrp.attrs['vsz_datatype'] = 'date'
        dgrp['data'] = self.data
        data = dgrp['data']
        data.attrs['vsz_convert_datetime'] = 1
        data.attrs['vsz_name'] = name.encode('utf-8')

    def deleteRows(self, row, numrows):
        """Delete numrows rows starting from row.
        Returns deleted rows as a dict of {column:data, ...}
        """
        retn = {
            'data': self.data[row:row+numrows],
        }
        self.data = N.delete(self.data, N.s_[row:row+numrows])
        self.document.modifiedData(self)
        return retn

    def insertRows(self, row, numrows, rowdata):
        """Insert numrows rows starting from row.
        rowdata is a dict of {column: data}.
        """
        data = N.zeros(numrows)
        if 'data' in rowdata:
            data[:len(rowdata['data'])] = N.array(rowdata['data'])
        self.data =  N.insert(self.data, [row]*numrows, data)
        self.document.modifiedData(self)

    def changeValues(self, thetype, vals):
        """Change the requested part of the dataset to vals.

        thetype == data
        """
        if thetype != 'data':
            raise ValueError('invalid column %s' % thetype)

        self.data = N.array(vals)

        # tell the document that we've changed
        self.document.modifiedData(self)

class DatasetText(DatasetConcreteBase):
    """Represents a text dataset: holding an array of strings."""

    dimensions = 1
    datatype = displaytype = 'text'
    columns = ('data',)
    column_descriptions = (_('Data'),)
    dstype = _('Text')
    editable = True

    def __init__(self, data=None, linked=None):
        """Initialise dataset with data given. Data are a list of strings."""

        DatasetConcreteBase.__init__(self, linked=linked)
        self.data = list(data)

    def description(self):
        return _('Text (length %i)') % len(self.data)

    def userSize(self):
        """Size of dataset."""
        return str( len(self.data) )

    def changeValues(self, type, vals):
        if type == 'data':
            self.data = list(vals)
        else:
            raise ValueError('type does not contain an allowed value')

        self.document.modifiedData(self)

    def uiConvertToDataItem(self, val):
        """Return a value cast to this dataset data type."""
        return cstr(val)

    def uiDataItemToData(self, val):
        """Return val converted to data."""
        return val

    def saveDataDumpToText(self, fileobj, name):
        '''Save data to file.
        '''
        fileobj.write("SetDataText(%s, [\n" % repr(name))
        for line in self.data:
            fileobj.write("    %s,\n" % crepr(line))
        fileobj.write("])\n")

    def saveDataDumpToHDF5(self, group, name):
        """Save text data to hdf5 file."""
        tgrp = group.create_group(utils.escapeHDFDataName(name))
        tgrp.attrs['vsz_datatype'] = 'text'
        # make sure data are encoded
        encdata = [x.encode('utf-8') for x in self.data]
        tgrp['data'] = encdata
        tgrp['data'].attrs['vsz_name'] = name.encode('utf-8')

    def datasetAsText(self, fmt=None, join=None):
        """Return data as text."""
        lines = list(self.data)
        lines.append('')
        return '\n'.join(lines)

    def deleteRows(self, row, numrows):
        """Delete numrows rows starting from row.
        Returns deleted rows as a dict of {column:data, ...}
        """
        retn = {'data': self.data[row:row+numrows]}
        del self.data[row:row+numrows]

        self.document.modifiedData(self)
        return retn

    def insertRows(self, row, numrows, rowdata):
        """Insert numrows rows starting from row.
        rowdata is a dict of {column: data}.
        """
        data = rowdata.get('data', [])

        insdata = data + (['']*(numrows-len(data)))
        for d in insdata[::-1]:
            self.data.insert(row, d)

        self.document.modifiedData(self)

    def returnCopy(self):
        """Returns version of dataset with no linking."""
        return DatasetText(self.data)

    def returnCopyWithNewData(self, **args):
        """Return dataset of same type using the column data given."""
        return DatasetText(**args)

class DatasetExpressionException(DatasetException):
    """Raised if there is an error evaluating a dataset expression."""
    pass

# split expression on python operators or quoted `DATASET`
dataexpr_split_re = re.compile(r'(`.*?`|[\.+\-*/\(\)\[\],<>=!|%^~& ])')
# identify whether string is a quoted identifier
dataexpr_quote_re = re.compile(r'^`.*`$')
dataexpr_columns = {'data':True, 'serr':True, 'perr':True, 'nerr':True}

def substituteDatasets(datasets, expression, thispart):
    """Substitute the names of datasets with calls to a function which will
    evaluate them.

    Returns (new expression, list of substituted datasets)
    """

    # split apart the expression to look for dataset names
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
            bits[i] = "_DS_(%s, %s)" % (crepr(bit), crepr(part))
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
                _("Dataset '%s' does not have part '%s'") % (dsname, dspart))
        return val
    else:
        raise DatasetExpressionException(
            'Internal error - invalid dataset part')

def _returnNumericDataset(doc, vals, dimensions, subdatasets):
    """Used internally to convert a set of values (which needs to be
    numeric) into a Dataset.

    subdatasets is list of datasets substituted into expression
    """

    err = None

    # try to convert array to a numpy array
    try:
        vals = N.array(vals, dtype=N.float64)
    except ValueError:
        err = _('Could not convert to array')

    # if error on first time, try to sanitize input arrays
    if err and dimensions == 1:
        try:
            vals = list(vals)
            vals[0] = N.array(vals[0])
            minlen = len(vals[0])
            if len(vals) in (2,3):
                # expand/convert error bars
                for i in crange(1, len(vals)):
                    if N.isscalar(vals[i]):
                        # convert to scalar
                        vals[i] = N.zeros(minlen) + vals[i]
                    else:
                        # convert to array
                        vals[i] = N.array(vals[i])
                        if vals[i].ndim != 1:
                            raise ValueError
                    minlen = min(minlen, len(vals[i]))

                # chop to minimum length
                for i in crange(len(vals)):
                    vals[i] = vals[i][:minlen]
            vals = N.array(vals, dtype=N.float64)
            err = None
        except (ValueError, IndexError, TypeError):
            pass

    if not err:
        if dimensions == 1:
            # see whether data values suitable for a 1d dataset
            if vals.ndim == 1:
                # 1d, so ok
                return Dataset(data=vals)
            elif vals.ndim == 2:
                # 2d, see whether data are error bars
                if vals.shape[0] == 2:
                    return Dataset(
                        data=vals[0,:], serr=vals[1,:])
                elif vals.shape[0] == 3:
                    return Dataset(
                        data=vals[0,:], perr=vals[1,:],
                        nerr=vals[2,:])
                else:
                    err = _('Expression has wrong dimensions')
        elif dimensions == 2 and vals.ndim == 2:
            # try to use dimensions of first-substituted dataset
            dsrange = {}
            for ds in subdatasets:
                d = doc.data[ds]
                if d.dimensions == 2:
                    for p in ('xrange', 'yrange', 'xedge', 'yedge',
                              'xcent', 'ycent'):
                        dsrange[p] = getattr(d, p)
                    break

            return Dataset2D(vals, **dsrange)
        else:
            err = _('Expression has wrong dimensions')

    raise DatasetExpressionException(err)

def evalDatasetExpression(doc, origexpr, datatype='numeric',
                          dimensions=1, part='data'):
    """Evaluate expression and return an appropriate Dataset.

    part is 'data', 'serr', 'perr' or 'nerr' - these are the
    dataset parts which are evaluated by the expression

    Returns None if error
    """

    d = doc.data.get(origexpr)
    if ( d is not None and
         d.datatype == datatype and
         d.dimensions == dimensions ):
        return d

    if utils.id_re.match(origexpr):
        # if name is a python identifier, then it has to be a dataset
        # name. As it wasn't there, just return with nothing rather
        # than print error message
        return None

    if not origexpr:
        # ignore blank names
        return None

    # replace dataset names by calls to _DS_(name,part)
    expr, subdatasets = substituteDatasets(doc.data, origexpr, part)

    comp = doc.compileCheckedExpression(expr, origexpr=origexpr)
    if comp is None:
        return

    # set up environment for evaluation
    env = doc.eval_context.copy()
    def doeval(dsname, dspart):
        return _evaluateDataset(doc.data, dsname, dspart)
    env['_DS_'] = doeval

    # do evaluation
    try:
        evalout = eval(comp, env)
    except Exception as ex:
        doc.log(_("Error evaluating '%s': '%s'" % (origexpr, cstr(ex))))
        return None

    # return correct dataset for data type
    try:
        if datatype == 'numeric':
            return _returnNumericDataset(doc, evalout, dimensions, subdatasets)
        elif datatype == 'text':
            return DatasetText([cstr(x) for x in evalout])
        else:
            raise RuntimeError('Invalid data type')
    except DatasetExpressionException as ex:
        doc.log(_("Error evaluating '%s': %s\n") % (origexpr, cstr(ex)))

    return None

class DatasetExpression(Dataset1DBase):
    """A dataset which is linked to another dataset by an expression."""

    dstype = _('Expression')

    def __init__(self, data=None, serr=None, nerr=None, perr=None,
                 parametric=None):
        """Initialise the dataset with the expressions given.

        parametric is option and can be (minval, maxval, steps) or None
        """

        Dataset1DBase.__init__(self)

        # store the expressions to use to generate the dataset
        self.expr = {}
        self.expr['data'] = data
        self.expr['serr'] = serr
        self.expr['nerr'] = nerr
        self.expr['perr'] = perr
        self.parametric = parametric

        self.docchangeset = -1
        self.evaluated = {}

    def evaluateDataset(self, dsname, dspart):
        """Return the dataset given.

        dsname is the name of the dataset
        dspart is the part to get (e.g. data, serr)
        """
        return _evaluateDataset(self.document.data, dsname, dspart)

    def _evaluatePart(self, expr, part):
        """Evaluate expression expr for part part.

        Returns True if succeeded
        """
        # replace dataset names with calls
        newexpr = substituteDatasets(self.document.data, expr, part)[0]

        comp = self.document.compileCheckedExpression(newexpr, origexpr=expr)
        if comp is None:
            return False

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
            result = eval(comp, environment)
            evalout = N.array(result, N.float64)

            if len(evalout.shape) > 1:
                raise RuntimeError("Number of dimensions is not 1")
        except Exception as ex:
            self.document.log(
                _("Error evaluating expression: %s\n"
                  "Error: %s") % (self.expr[part], cstr(ex)) )
            return False

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
        return True

    def updateEvaluation(self):
        """Update evaluation of parts of dataset.

        Returns False if problem with any evaluation
        """
        ok = True
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
                    ok = ok and self._evaluatePart(expr, part)

        return ok

    def _propValues(self, part):
        """Check whether expressions need reevaluating,
        and recalculate if necessary."""

        self.updateEvaluation()

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

    def saveDataRelationToText(self, fileobj, name):
        '''Save data to file.
        '''

        parts = [crepr(name), crepr(self.expr['data'])]
        if self.expr['serr']:
            parts.append('symerr=%s' % crepr(self.expr['serr']))
        if self.expr['nerr']:
            parts.append('negerr=%s' % crepr(self.expr['nerr']))
        if self.expr['perr']:
            parts.append('poserr=%s' % crepr(self.expr['perr']))
        if self.parametric is not None:
            parts.append('parametric=%s' % crepr(self.parametric))

        parts.append('linked=True')

        s = 'SetDataExpression(%s)\n' % ', '.join(parts)
        fileobj.write(s)

    def __getitem__(self, key):
        """Return a dataset based on this dataset

        We override this from DatasetConcreteBase as it would return a
        DatsetExpression otherwise, not chopped sets of data.
        """
        return Dataset(**self._getItemHelper(key))

    def canUnlink(self):
        """Whether dataset can be unlinked."""
        return True

    def linkedInformation(self):
        """Return information about linking."""
        text = []
        if self.parametric:
            text.append(_('Linked parametric dataset'))
        else:
            text.append(_('Linked expression dataset'))
        for label, part in czip(self.column_descriptions,
                                self.columns):
            if self.expr[part]:
                text.append('%s: %s' % (label, self.expr[part]))

        if self.parametric:
            text.append(_("where t goes from %g:%g in %i steps") % self.parametric)

        return '\n'.join(text)

class DatasetRange(Dataset1DBase):
    """Dataset consisting of a range of values e.g. 1 to 10 in 10 steps."""

    dstype = _('Range')

    def __init__(self, numsteps, data, serr=None, perr=None, nerr=None):
        """Construct dataset.

        numsteps: number of steps in range
        data, serr, perr and nerr are tuples containing (start, stop) values."""

        Dataset1DBase.__init__(self)

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

    def __getitem__(self, key):
        """Return a dataset based on this dataset

        We override this from DatasetConcreteBase as it would return a
        DatsetExpression otherwise, not chopped sets of data.
        """
        return Dataset(**self._getItemHelper(key))

    def userSize(self):
        """Size of dataset."""
        return str( self.numsteps )

    def saveDataRelationToText(self, fileobj, name):
        """Save dataset to file."""

        parts = [crepr(name), crepr(self.numsteps), crepr(self.range_data)]
        if self.range_serr is not None:
            parts.append('symerr=%s' % crepr(self.range_serr))
        if self.range_perr is not None:
            parts.append('poserr=%s' % crepr(self.range_perr))
        if self.range_nerr is not None:
            parts.append('negerr=%s' % crepr(self.range_nerr))
        parts.append('linked=True')

        s = 'SetDataRange(%s)\n' % ', '.join(parts)
        fileobj.write(s)

    def canUnlink(self):
        return True

    def linkedInformation(self):
        """Return information about linking."""
        text = [_('Linked range dataset')]
        for label, part in czip(self.column_descriptions,
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

    try:
        data = N.array(data) + 0
    except ValueError:
        raise DatasetExpressionException('Expression is not an array')

    if len(data.shape) != 1:
        raise DatasetExpressionException('Array is not 1D')
    if len(data) < 2:
        raise DatasetExpressionException('Two values required to convert to 2D')

    uniquesorted = N.unique(data)

    sigfactor = (uniquesorted[-1]-uniquesorted[0])*1e-13

    # differences between elements
    deltas = N.unique( N.ediff1d(uniquesorted) )

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

    if mindelta is None or mindelta == 0:
        raise DatasetExpressionException('Could not identify delta')

    return (uniquesorted[0], uniquesorted[-1], mindelta,
            int((uniquesorted[-1]-uniquesorted[0])/mindelta)+1)

class Dataset2DXYZExpression(Dataset2DBase):
    '''A 2d dataset with expressions for x, y and z.'''

    dstype = _('2D XYZ')

    def __init__(self, exprx, expry, exprz):
        """Initialise dataset.

        Parameters are mathematical expressions based on datasets."""
        Dataset2DBase.__init__(self)

        self.lastchangeset = -1
        self.cacheddata = None
        self.xedge = self.yedge = self.xcent = self.ycent = None

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

        # FIXME: handle irregular grids
        # return cached data if document unchanged
        if self.document.changeset == self.lastchangeset:
            return self.cacheddata
        self.lastchangeset = self.document.changeset
        self.cacheddata = None

        evaluated = {}

        environment = self.document.eval_context.copy()
        environment['_DS_'] = self.evaluateDataset

        # evaluate the x, y and z expressions
        for name in ('exprx', 'expry', 'exprz'):
            origexpr = getattr(self, name)
            expr = substituteDatasets(self.document.data, origexpr, 'data')[0]

            comp = self.document.compileCheckedExpression(
                expr, origexpr=origexpr)
            if comp is None:
                return None

            try:
                evaluated[name] = eval(comp, environment)
            except Exception as e:
                self.document.log(_("Error evaluating expression: %s\n"
                                    "Error: %s") % (expr, cstr(e)) )
                return None

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
        try:
            self.cacheddata.flat [ xpts + ypts*stepsx ] = evaluated['exprz']
        except Exception as e:
            self.document.log(_("Shape mismatch when constructing dataset\n"
                                "Error: %s") % cstr(e) )
            return None

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
        ds = self.evalDataset()
        if ds is None:
            return ( (0., 1.), (0., 1.) )
        return (self._xrange, self._yrange)

    @property
    def data(self):
        """Get data, or none if error."""
        ds = self.evalDataset()
        if ds is None:
            return N.array( [[]] )
        return ds

    def saveDataRelationToText(self, fileobj, name):
        '''Save expressions to file.
        '''

        s = 'SetData2DExpressionXYZ(%s, %s, %s, %s, linked=True)\n' % (
            crepr(name), crepr(self.exprx), crepr(self.expry), crepr(self.exprz) )
        fileobj.write(s)

    def canUnlink(self):
        """Can relationship be unlinked?"""
        return True

    def linkedInformation(self):
        """Return linking information."""
        return _('Linked 2D function: x=%s, y=%s, z=%s') % (
            self.exprx, self.expry, self.exprz)

class Dataset2DExpression(Dataset2DBase):
    """Evaluate an expression of 2d datasets."""

    dstype = _('2D Expr')

    def __init__(self, expr):
        """Create 2d expression dataset."""

        Dataset2DBase.__init__(self)

        self.expr = expr
        self.lastchangeset = -1

    @property
    def data(self):
        """Return data, or empty array if error."""
        ds = self.evalDataset()
        return ds.data if ds is not None else N.array([[]])

    @property
    def xrange(self):
        """Return x range."""
        ds = self.evalDataset()
        return ds.xrange if ds is not None else [0., 1.]

    @property
    def yrange(self):
        """Return y range."""
        ds = self.evalDataset()
        return ds.yrange if ds is not None else [0., 1.]

    @property
    def xedge(self):
        """Return x grid points."""
        ds = self.evalDataset()
        return ds.xedge if ds is not None else None

    @property
    def yedge(self):
        """Return y grid points."""
        ds = self.evalDataset()
        return ds.yedge if ds is not None else None

    @property
    def xcent(self):
        """Return x cent points."""
        ds = self.evalDataset()
        return ds.xcent if ds is not None else None

    @property
    def ycent(self):
        """Return y cent points."""
        ds = self.evalDataset()
        return ds.ycent if ds is not None else None

    def evalDataset(self):
        """Do actual evaluation."""
        return self.document.evalDatasetExpression(self.expr, dimensions=2)

    def saveDataRelationToText(self, fileobj, name):
        '''Save expression to file.'''
        s = 'SetData2DExpression(%s, %s, linked=True)\n' % (
            crepr(name), crepr(self.expr) )
        fileobj.write(s)

    def canUnlink(self):
        """Can relationship be unlinked?"""
        return True

    def linkedInformation(self):
        """Return linking information."""
        return _('Linked 2D expression: %s') % self.expr

class Dataset2DXYFunc(Dataset2DBase):
    """Given a range of x and y, this is a dataset which is a function of
    this.
    """

    dstype = _('2D f(x,y)')

    def __init__(self, xstep, ystep, expr):
        """Create 2d dataset:

        xstep: tuple(xmin, xmax, step)
        ystep: tuple(ymin, ymax, step)
        expr: expression of x and y
        """

        Dataset2DBase.__init__(self)

        if xstep is None or ystep is None:
            raise DatasetException('Steps are not set')

        self.xstep = xstep
        self.ystep = ystep
        self.expr = expr

        self.xrange = (
            self.xstep[0] - self.xstep[2]*0.5,
            self.xstep[1] + self.xstep[2]*0.5)
        self.yrange = (
            self.ystep[0] - self.ystep[2]*0.5,
            self.ystep[1] + self.ystep[2]*0.5)
        self.xedge = self.yedge = self.xcent = self.ycent = None

        self.cacheddata = None
        self.lastchangeset = -1

    @property
    def data(self):
        """Return data, or empty array if error."""
        try:
            return self.evalDataset()
        except DatasetExpressionException as ex:
            self.document.log(cstr(ex))
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
        except Exception as e:
            raise DatasetExpressionException(
                _("Error evaluating expression: %s\n"
                  "Error: %s") % (self.expr, str(e)) )

        # ensure we get an array out of this (in case expr is scalar)
        data = data + xstep*0

        self.cacheddata = data
        self.lastchangeset = self.document.changeset
        return data

    def saveDataRelationToText(self, fileobj, name):
        '''Save expressions to file.
        '''
        s = 'SetData2DXYFunc(%s, %s, %s, %s, linked=True)\n' % (
            crepr(name), crepr(self.xstep), crepr(self.ystep), crepr(self.expr) )
        fileobj.write(s)

    def canUnlink(self):
        """Can relationship be unlinked?"""
        return True

    def linkedInformation(self):
        """Return linking information."""
        return _('Linked 2D function: x=%g:%g:%g, y=%g:%g:%g, z=%s') % tuple(
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
        for name, val in citems(self.pluginmanager.fields):
            fields.append('%s: %s' % (cstr(name), cstr(val)))

        try:
            shape = [str(x) for x in self.data.shape]
        except AttributeError:
            shape = [str(len(self.data))]
        shape = u'\u00d7'.join(shape)

        return '%s plugin dataset (fields %s), size %s' % (
            self.pluginmanager.plugin.name,
            ', '.join(fields),
            shape)

    def canUnlink(self):
        """Can relationship be unlinked?"""
        return True

    def deleteRows(self, row, numrows):
        pass

    def insertRows(self, row, numrows, rowdata):
        pass

    def saveDataRelationToText(self, fileobj, name):
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

    def saveDataDumpToText(self, fileobj, name):
        """Save data to text: not used."""

    def saveDataDumpToHDF5(self, group, name):
        """Save data to HDF5: not used."""

    @property
    def dstype(self):
        """Return type of plugin."""
        return self.pluginmanager.plugin.name

class Dataset1DPlugin(_DatasetPlugin, Dataset1DBase):
    """Return 1D dataset from a plugin."""

    def __init__(self, manager, ds):
        _DatasetPlugin.__init__(self, manager, ds)
        Dataset1DBase.__init__(self)

    def userSize(self):
        """Size of dataset."""
        return str( self.data.shape[0] )

    def __getitem__(self, key):
        """Return a dataset based on this dataset

        We override this from DatasetConcreteBase as it would return a
        DatsetExpression otherwise, not chopped sets of data.
        """
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

class Dataset2DPlugin(_DatasetPlugin, Dataset2DBase):
    """Return 2D dataset from a plugin."""

    def __init__(self, manager, ds):
        _DatasetPlugin.__init__(self, manager, ds)
        Dataset2DBase.__init__(self)

    def __getitem__(self, key):
        return Dataset2D(self.data[key], xrange=self.xrange, yrange=self.yrange,
                         xedge=self.xedge, yedge=self.yedge,
                         xcent=self.xcent, ycent=self.ycent)

    data   = property( lambda self: self.getPluginData('data'),
                       lambda self, val: None )
    xrange = property( lambda self: self.getPluginData('rangex'),
                       lambda self, val: None )
    yrange = property( lambda self: self.getPluginData('rangey'),
                       lambda self, val: None )
    xedge  = property( lambda self: self.getPluginData('xedge'),
                       lambda self, val: None )
    yedge  = property( lambda self: self.getPluginData('yedge'),
                       lambda self, val: None )
    xcent  = property( lambda self: self.getPluginData('xcent'),
                       lambda self, val: None )
    ycent  = property( lambda self: self.getPluginData('ycent'),
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

class DatasetDateTimePlugin(_DatasetPlugin, DatasetDateTimeBase):
    """Return date dataset from plugin."""

    def __init__(self, manager, ds):
        _DatasetPlugin.__init__(self, manager, ds)
        DatasetDateTimeBase.__init__(self)
        self.serr = self.perr = self.nerr = None

    def __getitem__(self, key):
        return DatasetDateTime(self.data[key])

    data = property( lambda self: self.getPluginData('data'),
                     lambda self, val: None )
