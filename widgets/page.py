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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

"""Widget that represents a page in the document."""

import collections

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import widget
import root
import controlgraph
import axisuser

def _(text, disambiguation=None, context='Page'):
    """Translate text."""
    return unicode( 
        qt4.QCoreApplication.translate(context, text, disambiguation))

defaultrange = [1e99, -1e99]

class AxisDependHelper(object):
    """A class to work out the dependency of widgets on axes and vice
    versa, in terms of ranges of the axes.

    Note: Here a widget is really (widget, depname), as each widget
    can have a different dependency (e.g. sx and sy dependencies for
    plotters).

    It then works out the ranges for each of the axes from the plotters.

    connection types:
      plotter->axis : axis needs to know data range
      axis->plotter : plotter needs to know axis range
      axis<->axis   : axes are mutually dependent

    aim: calculate ranges of axes given plotters

    problem: cycles in the graph

    f1<-x: function f1 depends on axis x
    f2<-y: function f2 depends on axis y
    y<-f1: axis y depends on function f1
    x<-f2: axis x depends on function f2

    solution: break dependency cycle: choose somewhere - probably better
    to choose where widget depends on axis

    however, axis<->axis cycle can't be broken

    additional solution: convert all dependencies on axis1 or axis2 to
    axiscomb
    """

    def __init__(self, root):
        self.root = root

        # map widgets to widgets it depends on
        self.deps = collections.defaultdict(list)
        # list of axes
        self.axes = []
        # list of plotters associated with each axis
        self.axis_plotter_map = collections.defaultdict(list)
        # ranges for each axis
        self.ranges = {}
        # pairs of dependent widgets
        self.pairs = []

    def recursivePlotterSearch(self, widget):
        """Find a list of plotters below widget.

        Builds up a dict of "nodes" representing each widget: plotter/axis
        Each node is a list of tuples saying which widgets need evaling first
        The tuples are (widget, depname), where depname is a name for the 
        part of the plotter, e.g. "sx" or "sy" for x or y.
        """

        if isinstance(widget, axisuser.AxisUser):

            # keep track of which widgets depend on which axes
            widgetaxes = {}
            for axname in widget.getAxesNames():
                axis = widget.lookupAxis(axname)
                widgetaxes[axname] = axis
                self.axis_plotter_map[axis].append(widget)

            # if the widget is a user of an axis, find which axes the
            # widget can provide range information about
            for axname, depname in widget.affectsAxisRange():
                axis = widgetaxes[axname]
                if axis.usesAutoRange():
                    # only add dependency if axis has an automatic range
                    p1, p2 = (axis, None), (widget, depname)
                    self.deps[p1].append(p2)
                    self.pairs.append( (p2, p1) )

            # find which axes the axis-user needs information from
            for depname, axname in widget.requiresAxisRange():
                axis = widgetaxes[axname]
                if axis.usesAutoRange():
                    p2, p1 = (axis, None), (widget, depname)
                    self.deps[p1].append(p2)
                    self.pairs.append( (p2, p1) )

        if hasattr(widget, 'isaxis'):
            # keep track of all axis widgets
            self.axes.append(widget)
            self.ranges[widget] = list(defaultrange)

        for c in widget.children:
            self.recursivePlotterSearch(c)

    def breakCycles(self, origcyclic):
        """Remove cycles if possible."""

        numcyclic = len(origcyclic)
        best = -1

        for i in xrange(len(self.pairs)):
            if not hasattr(self.pairs[i][0][0], 'isaxis'):
                p = self.pairs[:i] + self.pairs[i+1:]
                ordered, cyclic = utils.topological_sort(p)
                if len(cyclic) <= numcyclic:
                    numcyclic = len(cyclic)
                    best = i

        # delete best, or last one if none better found
        p = self.pairs[best]
        del self.pairs[best]

        idx = self.deps[p[1]].index(p[0])
        del self.deps[p[1]][idx]

    def findPlotters(self):
        """Construct a list of plotters associated with each axis.
        Returns nodes:

        {axisobject: [plotterobject, plotter2...]),
         ...}
        """

        self.recursivePlotterSearch(self.root)

    def processDepends(self):
        """Go through dependencies of widget.
        If the dependency has no dependency itself, then update the
        axis with the widget or vice versa

        Algorithm:
          Iterate over dependencies for widget.

          If the widget has a dependency on a widget which doesn't
          have a dependency itself, update range from that
          widget. Then delete that depency from the dependency list.
        """

        # get ordered list, breaking cycles
        while True:
            ordered, cyclic = utils.topological_sort(self.pairs)
            if not cyclic:
                break
            self.breakCycles(cyclic)

        # iterate over widgets in order
        for dep in ordered:
            widget, widget_dep = dep

            # iterate over dependent widgets
            for widgetd, widgetd_dep in self.deps[dep]:

                if ( isinstance(widgetd, axisuser.AxisUser) and
                     (not widgetd.settings.isSetting('hide') or
                      not widgetd.settings.hide) ):
                    # update range of axis with (widgetd, widgetd_dep)
                    # do not do this if the widget is hidden
                    widgetd.getRange(widget, widgetd_dep, self.ranges[widget])

                if hasattr(widgetd, 'isaxis') and widgetd in self.ranges:
                    # set actual range on axis, as axis no longer has a
                    # dependency
                    axrange = self.ranges[widgetd]
                    if axrange == defaultrange:
                        axrange = None
                    widgetd.setAutoRange(axrange)
                    del self.ranges[widgetd]

    def findAxisRanges(self):
        """Find the ranges from the plotters and set the axis ranges.

        Follows the dependencies calculated above.
        """

        self.processDepends()

        # set any remaining ranges
        for axis, axrange in self.ranges.iteritems():
            if axrange == defaultrange:
                axrange = None
            axis.setAutoRange(axrange)

class Page(widget.Widget):
    """A class for representing a page of plotting."""

    typename='page'
    allowusercreation = True
    allowedparenttypes = [root.Root]
    description=_('Blank page')

    def __init__(self, parent, name=None):
        """Initialise object."""
        widget.Widget.__init__(self, parent, name=name)
        if type(self) == Page:
            self.readDefaults()
 
    @classmethod
    def addSettings(klass, s):
        widget.Widget.addSettings(s)
        
        # page sizes are initially linked to the document page size
        s.add( setting.DistancePhysical(
                'width',
                setting.Reference('/width'),
                descr=_('Width of page'),
                usertext=_('Page width'),
                formatting=True) )
        s.add( setting.DistancePhysical(
                'height',
                setting.Reference('/height'),
                descr=_('Height of page'),
                usertext=_('Page height'),
                formatting=True) )
        
    def draw(self, parentposn, painthelper, outerbounds=None):
        """Draw the plotter. Clip graph inside bounds."""

        # document should pass us the page bounds
        x1, y1, x2, y2 = parentposn

        # find ranges of axes
        axisdependhelper = AxisDependHelper(self)
        axisdependhelper.findPlotters()
        axisdependhelper.findAxisRanges()

        # store axis->plotter mappings in painthelper
        painthelper.axisplottermap.update(axisdependhelper.axis_plotter_map)
        # reverse mapping
        pamap = collections.defaultdict(list)
        for axis, plotters in painthelper.axisplottermap.iteritems():
            for plot in plotters:
                pamap[plot].append(axis)
        painthelper.plotteraxismap.update(pamap)

        if self.settings.hide:
            bounds = self.computeBounds(parentposn, painthelper)
            return bounds

        # clip to page
        painter = painthelper.painter(self, parentposn)
        with painter:
            # w and h are non integer
            w = self.settings.get('width').convert(painter)
            h = self.settings.get('height').convert(painter)

        painthelper.setControlGraph(self, [
                controlgraph.ControlMarginBox(self, [0, 0, w, h],
                                              [-10000, -10000,
                                                10000,  10000],
                                              painthelper,
                                              ismovable = False)
                ] )

        bounds = widget.Widget.draw(self, parentposn, painthelper,
                                    parentposn)
        return bounds

    def updateControlItem(self, cgi):
        """Call helper to set page size."""
        cgi.setPageSize()

# allow the factory to instantiate this
document.thefactory.register( Page )
    
