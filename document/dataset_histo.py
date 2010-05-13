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
        if self.document.changeset != self.changeset:
            self._cacheddata = simpleEvalExpression(self.document, self.inexpr)
            self.changeset = self.document.changeset
        return self._cacheddata

    def getBinLocations(self):
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
                    
            delta = (maxval - minval) / (numbins-1)
            return N.arange(minval, maxval+delta, delta)

    def getBinVals(self):
        binlocs = self.getBinLocations()
        if len(binlocs) == 0:
            return N.array([])
        data = self.getData()

        normed = self.method == 'density'
        hist, edges = N.histogram(data, bins=binlocs, normed=normed)
        if self.method == 'fractions':
            hist *= (1./len(data));
        return hist

class DatasetHistoBins(DatasetBase):

    def __init__(self, generator):
        self.generator = generator

    data = property(lambda self: return self.generator.getBinLocations()[:-1])
    serr = perr = nerr = N.array([])

class DatasetHistoValues(DatasetBase):

    def __init__(self, generator):
        self.generator = generator

    data = property(lambda self: return self.generator.getBinVals())
    serr = perr = nerr = N.array([])
