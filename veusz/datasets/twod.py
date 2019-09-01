# -*- coding: utf-8 -*-
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

"""Two dimensional datasets."""

from __future__ import division
import numpy as N

from ..compat import crepr, cstr
from .. import utils

from .commonfn import _, dsPreviewHelper, convertNumpy
from .base import (
    DatasetConcreteBase, DatasetException, DatasetExpressionException)

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
            try:
                xc = (N.arange(xw) + 0.5) * (
                    (self.xrange[1]-self.xrange[0])/xw) + self.xrange[0]
            except ZeroDivisionError:
                xc = 0.

        if self.ycent is not None:
            yc = self.ycent
        elif self.yedge is not None:
            yc = 0.5*(self.yedge[:-1]+self.yedge[1:])
        else:
            try:
                yc = (N.arange(yw) + 0.5) * (
                    (self.yrange[1]-self.yrange[0])/yw) + self.yrange[0]
            except ZeroDivisionError:
                yc = 0.

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

        env = self.document.evaluate.context.copy()

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
