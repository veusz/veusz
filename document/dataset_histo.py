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
from datasets import DatasetBase, simpleEvalExpression

class DatasetHistoGenerator(object):
    def __init__(self, document, inexpr,
                 binmanual = None, binexpr = None,
                 method = 'counts'):
        """
        inexpr = ds expression
        binmanual = None / [1,2,3,4,5]
        binexpr = None / (num, minval, maxval, islog)
        method = ('counts', 'density', or 'fractions')
        """

        self.changeset = -1

        self.document = document
        self.inexpr = inexpr
        self.binmanual = binmanual
        self.binexpr = binexpr
        self.method = method

    def getData(self):
        """Get data from input expression, caching result."""
        if self.document.changeset != self.changeset:
            self._cacheddata = simpleEvalExpression(self.document, self.inexpr)
            self.changeset = self.document.changeset
        return self._cacheddata

    def binLocations(self):
        """Compute locations of bins edges, giving N+1 items."""
        if self.binmanual is not None:
            return N.array(self.binmanual)
        else:
            numbins, minval, maxval, islog = self.binexpr

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
                return N.arange(minval, maxval+delta, delta)
            else:
                if minval < 0. or maxval < 0.:
                    minval, maxval = 0.01, 100.
                lmin, lmax = N.log(minval), N.log(maxval)
                delta = (lmax - lmin) / numbins
                return N.exp( N.arange(lmin, lmax+delta, delta) )

    def getBinLocations(self):
        """Return bin centre, -ve bin width, +ve bin width."""
        binlocs = self.binLocations()

        if self.binexpr and self.binexpr[3]:
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
            hist *= (1./len(data));
        return hist

    def getBinDataset(self):
        return DatasetHistoBins(self)
    def getValueDataset(self):
        return DatasetHistoValues(self)

class DatasetHistoBins(DatasetBase):
    """A dataset for getting the bin positions for the histogram."""

    def __init__(self, generator):
        self.generator = generator
        self.changeset = -1

    def getData(self):
        """Get bin positions, caching results."""
        if self.changeset != self.generator.document.changeset:
            self.datacache = self.generator.getBinLocations()
            self.changeset = self.generator.document.changeset
        return self.datacache

    data = property(lambda self: self.getData[0])
    nerr = property(lambda self: self.getData[1])
    perr = property(lambda self: self.getData[2])
    serr = None

class DatasetHistoValues(DatasetBase):
    """A dataset for getting the height of the bins in a histogram."""

    def __init__(self, generator):
        self.generator = generator
        self.changeset = -1

    def getData(self):
        """Get bin heights, caching results."""
        if self.changeset != self.generator.document.changeset:
            self.datacache = self.generator.getBinVals()
            self.changeset = self.generator.document.changeset
        return self.datacache

    data = property(lambda self: self.getData())
    serr = perr = nerr = None
