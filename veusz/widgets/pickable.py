#    pickable.py
#    stuff related to the Picker (aka Read Data) tool

#    Copyright (C) 2011 Benjamin K. Stuhl
#    Email: Benjamin K. Stuhl <bks24@cornell.edu>
#
#    This file is part of Veusz.
#
#    Veusz is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    Veusz is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################

import numpy as N

class PickInfo:
    """Encapsulates the results of a Pick operation. graphpos and coords are
       numeric (x,y) tuples, labels are the textual labels for the x and y
       datasets, and index is some object that the picker can use to figure out
       what the 'next' and 'previous' points are. index must implement __str__();
       return '' if it has no user-visible meaning."""
    def __init__(self, widget=None, graphpos=None, labels=None, coords=None, index=None):
        self.widget = widget
        self.graphpos = graphpos
        self.labels = labels
        self.coords = coords
        self.index = index
        self.distance = float('inf')
        self.displaytype = ('numeric', 'numeric')

    def __bool__(self):
        return bool(
            self.widget and self.graphpos and self.labels and self.coords)

class Index:
    """A class containing all the state a GenericPickable needs to find the
       next or previous point"""
    def __init__(self, ivar, index, sign):
        self.ivar = ivar
        self.index = index
        self.sign = sign

        # default to not trusting the actual index to be meaningful
        self.useindex = False

    def __str__(self):
        if not self.useindex:
            return ''
        else:
            # 1-based index
            return str(self.index+1)

def _chooseOrderingSign(m, c, p):
    """Figures out whether p or m is visually right of c"""
    assert c is not None

    if p is not None and m is not None:
        if p[0] > m[0] or (p[0] == m[0] and p[1] < m[1]):
            # p is visually to the right of or above m
            return 1
        else:
            return -1
    elif p is not None:
        if p[0] > c[0]:
            # p is visually right of c
            return 1
        else:
            return -1
    elif m is not None:
        if m[0] < c[0]:
            # m is visually left of c
            return 1
        else:
            return -1
    else:
        assert m is not None or p is not None

class GenericPickable:
    """Utility class which abstracts the math of picking the closest point out
       of a list of points"""

    def __init__(self, widget, labels, vals, graphvals):
        self.widget = widget
        self.labels = labels
        self.xvals, self.yvals = vals
        self.xgraph, self.ygraph = graphvals

    def _pickSign(self, i):
        if len(self.xgraph) <= 1:
            # we only have one element, so it doesn't matter anyways
            return 1

        # go backwards to get previous finite point
        mi = i-1
        while mi >= 0 and not N.isfinite(self.xgraph[mi]+self.ygraph[mi]):
            mi -= 1
        if mi < 0:
            m = None
        else:
            m = self.xgraph[mi], self.ygraph[mi]

        # point in centre
        c = self.xgraph[i], self.ygraph[i]

        # find next finite point
        pi = i+1
        while pi < len(self.xgraph) and not N.isfinite(self.xgraph[pi]+self.ygraph[pi]):
            pi += 1
        if pi == len(self.xgraph):
            p = None
        else:
            p = self.xgraph[pi], self.ygraph[pi]

        if p is None and m is None:
            # only 1 point after removing non-finite points
            return 1

        return _chooseOrderingSign(m, c, p)

    def pickPoint(self, x0, y0, bounds, distance_direction):
        info = PickInfo(self.widget, labels=self.labels)

        if self.widget.settings.hide:
            return info

        if self.xvals is None or self.yvals is None:
            return info
        if len(self.xgraph) == 0 or len(self.ygraph) == 0:
            return info

        # calculate distances
        if distance_direction == 'vertical':
            # measure distance along y
            dist = N.abs(self.ygraph - y0)
        elif distance_direction == 'horizontal':
            # measure distance along x
            dist = N.abs(self.xgraph - x0)
        elif distance_direction == 'radial':
            # measure radial distance
            dist = N.sqrt((self.xgraph - x0)**2 + (self.ygraph - y0)**2)
        else:
            # programming error
            assert (distance_direction == 'radial' or
                    distance_direction == 'vertical' or
                    distance_direction == 'horizontal')

        # ignore points which are offgraph or not finite
        with N.errstate(invalid='ignore'):
            outofbounds = (
                (self.xgraph < bounds[0]) | (self.xgraph > bounds[2]) |
                (self.ygraph < bounds[1]) | (self.ygraph > bounds[3]) |
                ~N.isfinite(self.xgraph) | ~N.isfinite(self.ygraph) )
        dist[outofbounds] = N.inf

        m = N.min(dist)
        # if there are multiple equidistant points, arbitrarily take
        # the first one

        try:
            i = N.nonzero(dist == m)[0][0]
        except IndexError:
            return info

        info.graphpos = self.xgraph[i], self.ygraph[i]
        info.coords = self.xvals[i], self.yvals[i]
        info.distance = m
        info.index = Index(self.xvals[i], i, self._pickSign(i))

        return info

    def pickIndex(self, oldindex, direction, bounds):
        info = PickInfo(self.widget, labels=self.labels)

        if self.widget.settings.hide:
            return info

        if self.xvals is None or self.yvals is None:
            return info

        if oldindex.index is None:
            # no explicit index, so find the closest location to the previous
            # independent variable value
            i = N.logical_not( N.logical_or(
                    self.xvals < oldindex.ivar, self.xvals > oldindex.ivar) )

            # and pick the next
            if oldindex.sign == 1:
                i = max(N.nonzero(i)[0])
            else:
                i = min(N.nonzero(i)[0])
        else:
            i = oldindex.index

        if direction == 'right':
            incr = oldindex.sign
        elif direction == 'left':
            incr = -oldindex.sign
        else:
            assert direction == 'right' or direction == 'left'

        i += incr

        # skip points that are outside of the bounds or are not finite
        while (i >= 0 and i < len(self.xgraph) and
                ( not N.isfinite(self.xgraph[i]) or
                  not N.isfinite(self.ygraph[i]) or
                  (self.xgraph[i] < bounds[0] or self.xgraph[i] > bounds[2] or
                   self.ygraph[i] < bounds[1] or self.ygraph[i] > bounds[3]) )):
            i += incr

        if i < 0 or i >= len(self.xgraph):
            return info

        info.graphpos = self.xgraph[i], self.ygraph[i]
        info.coords = self.xvals[i], self.yvals[i]
        info.index = Index(self.xvals[i], i, oldindex.sign)

        return info

class DiscretePickable(GenericPickable):
    """A specialization of GenericPickable that knows how to deal with widgets
       with axes and data sets"""
    def __init__(self, widget, xdata_propname, ydata_propname, mapdata_fn):
        s = widget.settings
        doc = widget.document
        self.xdata = xdata = s.get(xdata_propname).getData(doc)
        self.ydata = ydata = s.get(ydata_propname).getData(doc)
        labels = s.__getattr__(xdata_propname), s.__getattr__(ydata_propname)

        if not xdata or not ydata or not mapdata_fn:
            GenericPickable.__init__( self, widget, labels, (None, None), (None, None) )
            return

        # take data, ensure same size and map it
        x, y = xdata.data, ydata.data
        minlen = min(len(x), len(y))
        x, y = x[:minlen], y[:minlen]
        xs, ys = mapdata_fn(x, y)

        # and set us up with the mapped data
        GenericPickable.__init__( self, widget, labels, (x, y), (xs, ys) )

    def pickPoint(self, x0, y0, bounds, distance_direction):
        info = GenericPickable.pickPoint(self, x0, y0, bounds, distance_direction)
        if not info:
            return None

        info.displaytype = (self.xdata.displaytype, self.ydata.displaytype)

        # indicies are persistent
        info.index.useindex = True
        return info

    def pickIndex(self, oldindex, direction, bounds):
        info = GenericPickable.pickIndex(self, oldindex, direction, bounds)
        info.displaytype = (self.xdata.displaytype, self.ydata.displaytype)

        if not info:
            return info

        # indicies are persistent
        info.index.useindex = True
        return info
