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

from __future__ import division
import numpy as N

from ..compat import citems

from .commonfn import _
from .oned import Dataset1DBase
from .expression import evalDatasetExpression

class DatasetHistoGenerator(object):
    def __init__(self, document, inexpr,
                 binmanual = None, binparams = None,
                 method = 'counts',
                 cumulative = 'none',
                 errors=False):
        """
        inexpr = ds expression
        binmanual = None / [1,2,3,4,5]
        binparams = None / (num, minval, maxval, islog)
        method = ('counts', 'density', or 'fractions')
        cumulative = ('none', 'smalltolarge', 'largetosmall')
        errors = True/False
        """

        self.changeset = -1

        self.document = document
        self.inexpr = inexpr
        self.binmanual = binmanual
        if binparams is None:
            self.binparams = (10, 'Auto', 'Auto', False)
        else:
            self.binparams = binparams
        self.method = method
        self.cumulative = cumulative
        self.errors = errors
        self.bindataset = self.valuedataset = None

    def getData(self):
        """Get data from input expression, caching result."""
        if self.document.changeset != self.changeset:
            d = evalDatasetExpression(self.document, self.inexpr)
            if d is not None:
                d = d.data
                # only use finite data
                d = d[N.isfinite(d)]
                if len(d) == 0:
                    d = None

            self._cacheddata = d
            self.changeset = self.document.changeset
        return self._cacheddata

    def binLocations(self):
        """Compute locations of bins edges, giving N+1 items."""
        if self.binmanual is not None:
            return N.array(self.binmanual)
        else:
            numbins, minval, maxval, islog = self.binparams

            if minval == 'Auto' or maxval == 'Auto':
                data = self.getData()
                if data is None:
                    return N.array([])
                if minval == 'Auto':
                    minval = N.min(data)
                if maxval == 'Auto':
                    maxval = N.max(data)

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

        if self.getData() is None:
            return (N.array([]), None, None)

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

    def getErrors(self, data, binlocs):
        """Compute error bars if requried."""

        hist, edges = N.histogram(data, bins=binlocs)
        hist = hist.astype(N.float64)  # integers can break plots (github#49)

        # calculate scaling values for error bars
        if self.method == 'density':
            ratio = 1. / (hist.size*(edges[1]-edges[0]))
        elif self.method == 'fractions':
            ratio = 1. / data.size
        else:
            ratio = 1.

        # compute cumulative values (errors correlated)
        if self.cumulative == 'smalltolarge':
            hist = N.cumsum(hist)
        elif self.cumulative == 'largetosmall':
            hist = N.cumsum(hist[::-1])[::-1]

        # Gehrels 1986 ApJ 303 336
        perr = 1. + N.sqrt(hist + 0.75)
        nerr = N.where(hist > 0, N.sqrt(hist - 0.25), 0.)

        return -nerr*ratio, perr*ratio

    def getBinVals(self):
        """Return results for each bin."""

        data = self.getData()
        if data is None:
            return (N.array([]), None, None)

        density = self.method == 'density'
        binlocs = self.binLocations()
        hist, edges = N.histogram(data, bins=binlocs, density=density)
        hist = hist.astype(N.float64)  # integers can break plots (github#49)
        
        if self.method == 'fractions':
            hist = hist * (1./data.size)

        # if cumulative wanted
        if self.cumulative == 'smalltolarge':
            hist = N.cumsum(hist)
        elif self.cumulative == 'largetosmall':
            hist = N.cumsum(hist[::-1])[::-1]

        if self.errors:
            nerr, perr = self.getErrors(data, binlocs)
        else:
            nerr, perr = None, None

        return hist, nerr, perr

    def generateBinDataset(self):
        self.bindataset = DatasetHistoBins(self, self.document)
        return self.bindataset
    def generateValueDataset(self):
        self.valuedataset = DatasetHistoValues(self, self.document)
        return self.valuedataset

    def saveToFile(self, fileobj):
        """Save two datasets to file."""

        # lookup names of datasets in document
        bindsname = valuedsname = ''
        for name, ds in citems(self.document.data):
            if ds is self.bindataset:
                bindsname = name
            elif ds is self.valuedataset:
                valuedsname = name

        fileobj.write( ("CreateHistogram(%s, %s, %s, binparams=%s, "
                        "binmanual=%s, method=%s, "
                        "cumulative=%s, errors=%s)\n") %
                       (repr(self.inexpr), repr(bindsname), repr(valuedsname),
                        repr(self.binparams), repr(self.binmanual),
                        repr(self.method), repr(self.cumulative),
                        repr(self.errors)) )

    def linkedInformation(self):
        """Informating about linking."""

        if self.binmanual is not None:
            bins = _('manual bins')
        else:
            bins = _('%i bins from %s to %s') % (self.binparams[0],
                                                 self.binparams[1],
                                                 self.binparams[2])

        return _("Histogram of '%s' with %s") % (self.inexpr, bins)

class DatasetHistoBins(Dataset1DBase):
    """A dataset for getting the bin positions for the histogram."""

    dstype = _('Histogram')

    def __init__(self, generator, document):
        Dataset1DBase.__init__(self)
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

    def linkedInformation(self):
        """Informating about linking."""
        return self.generator.linkedInformation() + _(" (bin positions)")

    def saveDataDumpToText(self, fileobj, name):
        pass

    def saveDataDumpToHDF5(self, group, name):
        pass

    data = property(lambda self: self.getData()[0])
    nerr = property(lambda self: self.getData()[1])
    perr = property(lambda self: self.getData()[2])
    serr = None

class DatasetHistoValues(Dataset1DBase):
    """A dataset for getting the height of the bins in a histogram."""

    dstype = _('Histogram')

    def __init__(self, generator, document):
        Dataset1DBase.__init__(self)
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

    def saveDataRelationToText(self, fileobj, name):
        """Save dataset and its counterpart to a file."""
        self.generator.saveToFile(fileobj)

    def saveDataDumpToText(self, fileobj, name):
        pass

    def saveDataDumpToHDF5(self, group, name):
        pass

    def linkedInformation(self):
        """Informating about linking."""
        return self.generator.linkedInformation() + _(" (bin values)")

    data = property(lambda self: self.getData()[0])
    nerr = property(lambda self: self.getData()[1])
    perr = property(lambda self: self.getData()[2])
    serr = None
