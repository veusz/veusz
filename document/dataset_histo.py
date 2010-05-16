#    Copyright (C) 2010 Jeremy S. Sanders
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

import numpy as N
from datasets import Dataset, simpleEvalExpression

class DatasetHistoGenerator(object):
    def __init__(self, document, inexpr,
                 binmanual = None, binparams = None,
                 method = 'counts'):
        """
        inexpr = ds expression
        binmanual = None / [1,2,3,4,5]
        binparams = None / (num, minval, maxval, islog)
        method = ('counts', 'density', or 'fractions')
        """

        self.changeset = -1

        self.document = document
        self.inexpr = inexpr
        self.binmanual = binmanual
        self.binparams = binparams
        self.method = method

    def getData(self):
        """Get data from input expression, caching result."""
        if self.document.changeset != self.changeset:
            self._cacheddata = simpleEvalExpression(self.document, self.inexpr)
            self.changeset = self.document.changeset
        return self._cacheddata

    def binLocations(self):
        """Compute locations of bins edges, giving N+1 items."""
        if self.binmanual:
            return N.array(self.binmanual)
        else:
            numbins, minval, maxval, islog = self.binparams

            if minval == 'Auto' or maxval == 'Auto':
                data = self.getData()
                if len(data) == 0:
                    return N.array([])
                if minval == 'Auto':
                    minval = N.nanmin(data)
                if maxval == 'Auto':
                    maxval = N.nanmax(data)

            if not islog:
                delta = (maxval - minval) / numbins
                return N.arange(numbins+1)*delta + minval
            else:
                if minval <= 0:
                    minval = 1e-99
                if maxval <= 0:
                    maxval = 1e99
                lmin, lmax = N.log(minval), N.log(maxval)
                delta = (lmax - lmin) / numbins
                return N.exp( N.arange(numbins+1)*delta + lmin )

    def getBinLocations(self):
        """Return bin centre, -ve bin width, +ve bin width."""
        binlocs = self.binLocations()

        if self.binparams and self.binparams[3]:
            # log bins
            lbin = N.log(binlocs)
            data = N.exp( 0.5*(lbin[:-1] + lbin[1:]) )
        else:
            # otherwise linear bins
            data = 0.5*(binlocs[:-1] + binlocs[1:])

        # error bars
        nerr = binlocs[:-1] - data
        perr = binlocs[1:] - data
        return data, nerr, perr

    def getBinVals(self):
        """Return results for each bin."""
        binlocs = self.binLocations()
        if len(binlocs) == 0:
            return N.array([])
        data = self.getData()

        normed = self.method == 'density'
        hist, edges = N.histogram(data, bins=binlocs, normed=normed)
        if self.method == 'fractions':
            hist = hist * (1./len(data))
        return hist

    def getBinDataset(self):
        self.bindataset = DatasetHistoBins(self, self.document)
        return self.bindataset
    def getValueDataset(self):
        self.valuedataset = DatasetHistoValues(self, self.document)
        return self.valuedataset

    def saveToFile(self, fileobj):
        """Save two datasets to file."""
        try:
            binname = self.bindataset.name()
        except ValueError:
            binname = ''
        try:
            valname = self.valuedataset.name()
        except ValueError:
            valname = ''

        fileobj.write( ("CreateHistogram(%s, %s, %s, binparams=%s, "
                        "binmanual=%s, method=%s)\n") %
                       (repr(self.inexpr), repr(binname), repr(valname),
                        repr(self.binparams), repr(self.binmanual),
                        repr(self.method)) )

class DatasetHistoBins(Dataset):
    """A dataset for getting the bin positions for the histogram."""

    def __init__(self, generator, document):
        self.generator = generator
        self.document = document
        self.linked = None
        self._invalidpoints = None
        self.changeset = -1

    def getData(self):
        """Get bin positions, caching results."""
        if self.changeset != self.generator.document.changeset:
            self.datacache = self.generator.getBinLocations()
            self.changeset = self.generator.document.changeset
        return self.datacache

    def saveToFile(self, fileobj, name):
        """Save dataset (counterpart does this)."""
        pass

    data = property(lambda self: self.getData()[0])
    nerr = property(lambda self: self.getData()[1])
    perr = property(lambda self: self.getData()[2])
    serr = None

class DatasetHistoValues(Dataset):
    """A dataset for getting the height of the bins in a histogram."""

    def __init__(self, generator, document):
        self.generator = generator
        self.document = document
        self.linked = None
        self._invalidpoints = None
        self.changeset = -1

    def getData(self):
        """Get bin heights, caching results."""
        if self.changeset != self.generator.document.changeset:
            self.datacache = self.generator.getBinVals()
            self.changeset = self.generator.document.changeset
        return self.datacache

    def saveToFile(self, fileobj, name):
        """Save dataset and its counterpart to a file."""
        self.generator.saveToFile(fileobj)

    data = property(lambda self: self.getData())
    serr = perr = nerr = None

class OperationDatasetHistogram(object):
    """Operation to make histogram from data."""

    descr = 'make histogram'

    def __init__(self, expr, outposns, outvalues,
                 binparams=None, binmanual=None, method='counts'):
        """
        inexpr = input dataset expression
        outposns = name of dataset for bin positions
        outvalues = name of dataset for bin values
        binparams = None / (num, minval, maxval, islog)
        binmanual = None / [1,2,3,4,5]
        method = ('counts', 'density', or 'fractions')
        """

        self.expr = expr
        self.outposns = outposns
        self.outvalues = outvalues
        self.binparams = binparams
        self.binmanual = binmanual
        self.method = method

    def do(self, document):
        """Create histogram datasets."""

        gen = DatasetHistoGenerator(
            document, self.expr, binparams=self.binparams,
            binmanual=self.binmanual,
            method=self.method )

        if self.outvalues != '':
            self.oldvaluesds = document.data.get(self.outvalues, None)
            document.data[self.outvalues] =  gen.getValueDataset()

        if self.outposns != '':
            self.oldposnsds = document.data.get(self.outposns, None)
            document.data[self.outposns] = gen.getBinDataset()

    def undo(self, document):
        """Undo creation of datasets."""

        if self.oldposnsds is not None:
            if self.outposns != '':
                document.data[self.outposns] = self.oldposnsds
        else:
            del document.data[self.outposns]

        if self.oldvaluesds is not None:
            if self.outvalues != '':
                document.data[self.outvalues] = self.oldvaluesds
        else:
            del document.data[self.outvalues]
