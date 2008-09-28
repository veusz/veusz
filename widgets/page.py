#    Copyright (C) 2004 Jeremy S. Sanders
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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

"""Widget that represents a page in the document."""

import veusz.qtall as qt4

import veusz.document as document

import widget
import root

# x, y, fplot, xyplot
# x-> xyplot(x)
# y-> xyplot(y)
# fplot(y) -> x
# y -> fplot(y)

# x -> xyplot(x)
# y -> (xyplot(y), fplot(y) -> x)

defaultrange = [1e99, -1e99]

class _AxisDependHelper(object):
    """A class to work out the dependency of widgets on axes and vice
    versa, in terms of ranges of the axes.

    It then works out the ranges for each of the axes from the plotters.
    """

    def __init__(self, root):
        self.root = root
        self.nodes = {}
        self.axes = []

    def recursivePlotterSearch(self, widget):
        """Find a list of plotters below widget.

        Builds up a dict of "nodes" representing each widget: plotter/axis
        Each node is a list of tuples saying which widgets need evaling first
        The tuples are (widget, depname), where depname is a name for the 
        part of the plotter, e.g. "sx" or "sy" for x or y.
        """
        if hasattr(widget, 'isplotter'):
            nodes = self.nodes

            # if the widget is a plotter, find which axes the plotter can
            # provide range information about
            for axname, depname in widget.providesAxesDependency():
                axis = widget.lookupAxis(axname)
                axdep = (axis, None)
                if axdep not in nodes:
                    nodes[axdep] = []
                nodes[axdep].append( (widget, depname) )
            # find which axes the plotter needs information from
            for depname, axname in widget.requiresAxesDependency():
                axis = widget.lookupAxis(axname)
                widdep = (widget, depname)
                if widdep not in nodes:
                    nodes[widdep] = []
                nodes[widdep].append( (axis, None) )

        elif hasattr(widget, 'isaxis'):
            # it is an axis, so keep track of it
            if hasattr(widget, 'isaxis'):
                self.axes.append(widget)

        else:
            # otherwise search children
            for c in widget.children:
                self.recursivePlotterSearch(c)

    def findPlotters(self):
        self.recursivePlotterSearch(self.root)
        self.ranges = dict( [(a, list(defaultrange)) for a in self.axes] )

    def processDepends(self, widget, depends):
        """Go through dependencies of widget.
        If the dependency has no dependency itself, then update the
        axis with the widget or vice versa
        """
        modified = False
        i = 0
        while i < len(depends):
            dep = depends[i]
            if dep not in self.nodes:
                dwidget, dwidget_dep = dep
                if hasattr(dwidget, 'isplotter'):
                    # update range of axis with (dwidget, dwidget_dep)
                    dwidget.updateAxisRange(dwidget_dep,
                                            self.ranges[widget])
                elif hasattr(dwidget, 'isaxis'):
                    # set actual range on axis, as axis no longer has a
                    # dependency
                    if dwidget in self.ranges:
                        axrange = self.ranges[dwidget]
                        if axrange == defaultrange:
                            axrange = None
                        dwidget.setAutoRange(axrange)
                        del self.ranges[dwidget]
                del depends[i]
                modified = True
                continue
            i += 1
        return modified

    def findAxisRanges(self):
        """Find the ranges from the plotters and set the axis ranges.

        Follows the dependencies calculated above.
        """

        # probaby horribly inefficient
        nodes = self.nodes
        while nodes:
            # iterate over dependencies for each widget
            inloop = True

            for (widget, widget_depname), depends in nodes.iteritems():
                # go through dependencies of widget
                if self.processDepends(widget, depends):
                    # if modified, we keep looping
                    inloop = False

                # delete dependencies for widget if none remaining
                if not depends:
                    del nodes[(widget, widget_depname)]
                    break

            # prevent infinite loops, break out if we do nothing in an
            # iteration
            if inloop:
                break

        for axis, axrange in self.ranges.iteritems():
            if axrange == defaultrange:
                axrange = None
            axis.setAutoRange(axrange)

class Page(widget.Widget):
    """A class for representing a page of plotting."""

    typename='page'
    allowusercreation = True
    allowedparenttypes = [root.Root]
    description='Blank page'

    def __init__(self, parent, name=None):
        """Initialise object."""

        widget.Widget.__init__(self, parent, name=name)
        #self.axisrangehelper = _AxisRangeHelper(self)

    def draw(self, parentposn, painter, outerbounds=None):
        """Draw the plotter. Clip graph inside bounds."""

        # document should pass us the page bounds
        x1, y1, x2, y2 = parentposn

        #self.axisrangehelper.findPlotters()
        axisdependhelper = _AxisDependHelper(self)
        axisdependhelper.findPlotters()
        axisdependhelper.findAxisRanges()

        if self.settings.hide:
            bounds = self.computeBounds(parentposn, painter)
            return bounds

        painter.beginPaintingWidget(self, parentposn)
        painter.save()
        painter.veusz_page_size = (x2-x1, y2-y1)
        painter.setClipRect( qt4.QRectF(x1, y1, x2-x1, y2-y1) )
        bounds = widget.Widget.draw(self, parentposn, painter,
                                    parentposn)
        painter.restore()
        painter.endPaintingWidget()

        return bounds

# allow the factory to instantiate this
document.thefactory.register( Page )
    
