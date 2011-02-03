#    pickable.py
#    stuff related to the Picker (aka Read Data) tool

#    Copyright (C) 2011 Benjamin K. Stuhl
#    Email: Benjamin K. Stuhl <bks24@cornell.edu>
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

# $Id: $

import numpy as N

import veusz.document as document

class PickInfo:
    """Encapsulates the results of a Pick operation. screenpos and coords are
       numeric (x,y) tuples, labels are the textual labels for the x and y
       datasets, and index is some object that the picker can use to figure out
       what the 'next' and 'previous' points are. index must implement __str__();
       return '' if it has no user-visible meaning."""
    def __init__(self, widget=None, screenpos=None, labels=None, coords=None, index=None):
        self.widget = widget
        self.screenpos = screenpos
        self.labels = labels
        self.coords = coords
        self.index = index
        self.distance = float('inf')

    def __nonzero__(self):
        if self.widget and self.screenpos and self.labels and self.coords:
            return True
        return False

class DiscreteIndex:
    def __init__(self, value, sign=0):
        self.value = value
        self.sign = sign

    def __str__(self):
        return str(self.value)

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

class DiscretePickable:
    """Utility class which abstracts the math of picking the closest point out
       of a widget which has a list of points"""
    def __init__(self, widget, xdata_propname, ydata_propname, mapdata_fn):
        self.widget = widget
        self.map_fn = mapdata_fn

        s = widget.settings
        doc = widget.document
        self.xdata = s.get(xdata_propname).getData(doc)
        self.ydata = s.get(ydata_propname).getData(doc)
        self.labels = s.__getattr__(xdata_propname), s.__getattr__(ydata_propname)

    def pickPoint(self, x0, y0, bounds, distance_direction):
        info = PickInfo(self.widget, labels=self.labels)

        if self.widget.settings.hide:
            return info

        if not self.xdata or not self.ydata or not self.map_fn:
            return info

        # loop over all valid data and look for the closest point
        offset = 0
        import document
        for xvals, yvals in document.generateValidDatasetParts(self.xdata, self.ydata):
            xplotter, yplotter = self.map_fn(xvals.data, yvals.data, bounds)

            if len(xplotter) != len(yplotter):
                l = min(len(xplotter), len(yplotter))
                xplotter = xplotter[0:l]
                yplotter = yplotter[0:l]

            if distance_direction == 'vertical':
                # measure distance along y
                dist = N.abs(yplotter - y0)
            elif distance_direction == 'horizontal':
                # measure distance along x
                dist = N.abs(xplotter - x0)
            elif distance_direction == 'radial':
                # measure radial distance
                dist = N.sqrt((xplotter - x0)**2 + (yplotter - y0)**2)
            else:
                # programming error
                assert (distance_direction == 'radial' or
                        distance_direction == 'vertical' or
                        distance_direction == 'horizontal')

            # ignore points which are offscreen
            outofbounds = ( (xplotter < bounds[0]) | (xplotter > bounds[2]) |
                            (yplotter < bounds[1]) | (yplotter > bounds[3]) )
            dist[outofbounds] = float('inf')

            m = N.min(dist)
            if m < info.distance:
                # if there are multiple equidistant points, arbitrarily take
                # the first one
                i = N.nonzero(dist == m)[0][0]

                info.screenpos = xplotter[i], yplotter[i]
                info.coords = xvals.data[i], yvals.data[i]
                info.distance = m
                info.index = DiscreteIndex(i + offset)

            offset += len(xplotter)

        return info

    def pickIndex(self, oldindex, direction, bounds):
        info = PickInfo(self.widget, labels=self.labels)

        if self.widget.settings.hide:
            return info

        if not self.xdata or not self.ydata or not self.map_fn:
            return info

        centerindex = oldindex.value
        # try to map one to each side of the current position
        mapped = { centerindex - 1 : [None], centerindex: [None], centerindex + 1: [None] }

        # iterate over only valid data to do the subscripting
        offset = 0
        for xvals, yvals in document.generateValidDatasetParts(self.xdata, self.ydata):
            chunklen = min(len(xvals.data), len(yvals.data))

            if oldindex < offset:
                break

            index = centerindex - offset
            for i in index-1, index, index+1:
                if i < 0:
                    continue

                if i < chunklen:
                    x, y = xvals.data[i], yvals.data[i]
                    xs, ys = self.map_fn(N.array(x), N.array(y), bounds)

                    # don't go outside the current plot window
                    if ( (xs < bounds[0]) or (xs > bounds[2]) or
                         (ys < bounds[1]) or (ys > bounds[3]) ):
                        continue

                    mapped[i + offset] = ((xs, ys), (x, y))

            offset += chunklen

        if ( mapped[centerindex][0] is None or
             (mapped[centerindex-1][0] is None and mapped[centerindex+1][0] is None) ):
            return info

        if oldindex.sign == 0:
            sign = _chooseOrderingSign(mapped[centerindex-1][0],
                                       mapped[centerindex][0],
                                       mapped[centerindex+1][0])
        else:
            sign = oldindex.sign

        # we've now know which way is left and which way is right
        if direction == 'right':
            index = oldindex.value + sign
        elif direction == 'left':
            index = oldindex.value - sign
        else:
            # programming error
            assert direction == 'left' or direction == 'right'

        info.index = DiscreteIndex(index, sign)
        if mapped[index][0] is not None:
            info.screenpos = mapped[index][0]
            info.coords = mapped[index][1]

        return info
