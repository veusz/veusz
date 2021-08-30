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

from .. import utils

class Dataset1DBase(DatasetConcreteBase):
    """Base for 1D datasets."""

    # number of dimensions the dataset holds
    dimensions = 1
    # includes data values, symmetric errors, negative+positive errors, flags for fine-tuned control of datapoints
    columns = ('data', 'serr', 'nerr', 'perr', 'flags') 
    column_descriptions = (
        _('Data'),
        _('Sym. errors'),
        _('Neg. errors'),
        _('Pos. errors'),
        _('Flags')  
    )
    dstype = _('1D')

    # subclasses must define .data, .serr, .perr, .nerr, .flags 

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

    def flagExcludeUnset(self):
        """Return a numpy bool detailing which datapoints do not have the 'exclude' flag set."""

        #on-the-fly conversion to int16, alternative is to enforce int16 datatype at data import/entry stage
        if self.flags is not None:
            return ((self.flags.astype(N.int16) & N.int16(2))==0)
        else:
            return N.ones(self.data.shape, dtype=N.bool)

    def flagDontProcessUnset(self):
        """Return a numpy bool detailing which datapoints do not have the 'do not process' flag set."""
        
        #on-the-fly conversion to int16, alternative is to enforce int16 datatype at data import/entry stage
        if self.flags is not None:
            return ((self.flags.astype(N.int16) & N.int16(1))==0)
        else:
            return N.ones(self.data.shape, dtype=N.bool)

    def validDataPoints(self):
        """Return a numpy bool detailing which datapoints have values, errors and flags (only if defined) that are are real numbers, and the exclude flag is not set."""

        valid = N.isfinite(self.data)
        for error in self.serr, self.perr, self.nerr, self.flags:  
            if error is not None:
                valid = N.logical_and(valid, N.isfinite(error))
        if self.flags is not None:   
            valid = N.logical_and(valid, self.flagExcludeUnset()) 
        return valid

    def invalidDataPoints(self):
        """Return a numpy bool detailing which datapoints are invalid."""
        
        return N.logical_not(self.validDataPoints())

    def validatedData(self):
        """Return a numpy array of the data for which the values, errors and flags (only if defined) are real numbers, and the exclude flag is not set."""
        
        return self.data[self.validDataPoints()]

    def hasErrors(self):
        '''Whether errors on dataset'''
        return (self.serr is not None or self.nerr is not None or
                self.perr is not None)

    def getPointRanges(self):
        '''Get range of coordinates for each point in the form
        (minima, maxima), excluding invalid data points.'''

        # boolean array for indexing valid data
        valid = self.validDataPoints()
        
        # implicitly creates copy
        minvals = self.data[valid]
        maxvals = self.data[valid]

        # note currently if multiple errors are defined these are superimposed on one another
        if self.serr is not None:
            minvals -= self.serr[valid]
            maxvals += self.serr[valid]

        if self.nerr is not None:
            minvals += self.nerr[valid]

        if self.perr is not None:
            maxvals += self.perr[valid]

        # only contains finite values
        return (
            minvals, maxvals
        )

    def getRange(self):
        '''Get total range of coordinates. Returns None if empty.'''
        minvals, maxvals = self.getPointRanges()
        if len(minvals) > 0 and len(maxvals) > 0:
            return ( minvals.min(), maxvals.max() )
        else:
            return None

    def updateRangeAuto(self, axrange, noneg):
    
        # boolean array for indexing valid data
        valid = self.validDataPoints()

        val = pos = neg = self.data[valid]
        if self.serr is not None:
            pos = pos + self.serr[valid]
            neg = neg - self.serr[valid]
        if self.perr is not None:
            pos = pos + self.perr[valid]
        if self.nerr is not None:
            neg = neg + self.nerr[valid]

        for v in val, pos, neg:
            if noneg:
                v = v[v>0]
            if len(v) > 0:
                axrange[0] = min(axrange[0], N.nanmin(v))
                axrange[1] = max(axrange[1], N.nanmax(v))

    def rangeVisit(self, fn):
        '''Call fn on data points and error values, in order to get range.'''
        
        # boolean array for indexing valid data
        valid = self.validDataPoints()

        val = self.data[valid]
        fn(val)
        if self.serr is not None:
            fn(val - self.serr[valid])
            fn(val + self.serr[valid])
        if self.nerr is not None:
            fn(val + self.nerr[valid])
        if self.perr is not None:
            fn(val + self.perr[valid])

    def empty(self):
        '''Is the data defined?'''
        return self.data is None or len(self.data) == 0

    def datasetAsText(self, fmt='%g', join='\t'):
        """Return data as text."""

        # work out which columns to write
        cols = []
        for c in (self.data, self.serr, self.perr, self.nerr, self.flags):  
            if c is not None:
                cols.append(c)

        # format statement
        format = (fmt + join) * (len(cols)-1) + fmt + '\n'

        # do the conversion
        lines = []
        for line in zip(*cols):
            lines.append( format % line )
        return ''.join(lines)

    def returnCopy(self):
        """Return version of dataset with no linking."""
        return Dataset(
            data=copyOrNone(self.data),
            serr=copyOrNone(self.serr),
            perr=copyOrNone(self.perr),
            nerr=copyOrNone(self.nerr),
            flags=copyOrNone(self.flags) 
        )

    def returnCopyWithNewData(self, **args):
        """Return dataset of same type using the column data given."""
        return Dataset(**args)

class Dataset(Dataset1DBase):
    '''Represents a dataset.'''

    editable = True

    def __init__(self, data = None, serr = None, nerr = None, perr = None, flags = None, 
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
        #ideally add function to create integer
        self.flags = convertNumpyAbs(flags)

        # check the sizes of things match up
        s = self.data.shape
        for x in self.serr, self.nerr, self.perr, self.flags: 
            if x is not None and x.shape != s:
                raise DatasetException('Lengths of error data do not match data')

    def changeValues(self, thetype, vals):
        """Change the requested part of the dataset to vals.

        thetype == data | serr | perr | nerr | flags
        """
        if thetype in self.columns:
            setattr(self, thetype, vals)
        else:
            raise ValueError('thetype does not contain an allowed value')

        # just a check...
        s = self.data.shape
        for x in (self.serr, self.nerr, self.perr, self.flags): 
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
        #flags denoted by = symbol
        if self.flags is not None:  
            descriptor += ',='     

        fileobj.write( "ImportString(%s,'''\n" % repr(descriptor) )
        fileobj.write( self.datasetAsText(fmt='%e', join=' ') )
        fileobj.write( "''')\n" )

    def saveDataDumpToHDF5(self, group, name):
        """Save dataset to HDF5."""

        # store as a group to simplify things
        odgrp = group.create_group(utils.escapeHDFDataName(name))
        odgrp.attrs['vsz_datatype'] = '1d'

        for key, suffix in (
                ('data', ''), ('serr', ' (+-)'),
                ('perr', ' (+)'), ('nerr', ' (-)'), ('flags', ' (=)')): 
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

    def __init__(self, numsteps, data, serr=None, perr=None, nerr=None, flags=None): 
        """Construct dataset.

        numsteps: number of steps in range
        data, serr, perr and nerr are tuples containing (start, stop) values."""

        Dataset1DBase.__init__(self)

        self.range_data = data
        self.range_serr = serr
        self.range_perr = perr
        self.range_nerr = nerr
        self.range_flags = flags 
        self.numsteps = numsteps

        for name in ('data', 'serr', 'perr', 'nerr', 'flags'):  
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

        parts = [repr(name), repr(self.numsteps), repr(self.range_data)]
        if self.range_serr is not None:
            parts.append('symerr=%s' % repr(self.range_serr))
        if self.range_perr is not None:
            parts.append('poserr=%s' % repr(self.range_perr))
        if self.range_nerr is not None:
            parts.append('negerr=%s' % repr(self.range_nerr))
        if self.range_flags is not None:                      
            parts.append('flags=%s' % repr(self.range_flags))  
        parts.append('linked=True')

        s = 'SetDataRange(%s)\n' % ', '.join(parts)
        fileobj.write(s)

    def canUnlink(self):
        return True

    def linkedInformation(self):
        """Return information about linking."""
        text = [_('Linked range dataset')]
        for label, part in zip(self.column_descriptions, self.columns):
            val = getattr(self, 'range_%s' % part)
            if val:
                text.append('%s: %g:%g' % (label, val[0], val[1]))
        return '\n'.join(text)
