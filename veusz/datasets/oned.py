#    Copyright (C) 2016 Jeremy S. Sanders
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

"""One dimensional datasets."""

import numpy as N

from .commonfn import (
    _, dsPreviewHelper, copyOrNone, convertNumpy,
    convertNumpyAbs, convertNumpyNegAbs, datasetNameToDescriptorName)
from .base import DatasetConcreteBase, DatasetException

from ..compat import czip,  crepr
from .. import utils

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
        return Dataset(data = copyOrNone(self.data),
                       serr = copyOrNone(self.serr),
                       perr = copyOrNone(self.perr),
                       nerr = copyOrNone(self.nerr))

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
