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

from __future__ import division, print_function
import collections
import textwrap
import numpy as N

from ..compat import crange, citems
from .. import qtall as qt
from .. import document
from .. import setting
from .. import utils

from . import widget
from . import controlgraph

def _(text, disambiguation=None, context='Page'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

defaultrange = [1e99, -1e99]

def _resolveLinkedAxis(axis):
    """Follow a chain of axis function dependencies."""
    loopcheck = set()
    while axis is not None and axis.isLinked():
        loopcheck.add(axis)
        axis = axis.getLinkedAxis()
        if axis in loopcheck:
            # fail if loop
            return None
    return axis

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

    x <-> axis1 <-> axis2

    For linked axes (e.g. AxisFunction):
      * Don't keep track of range separately -> propagate to real axis
      * For dependency order resolution, use real axis
      * In self.deps, use axisfunction axis so we know which axis to use
    """

    def __init__(self):
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

        # track axes which map from one axis to another
        self.axis_to_axislinked = {}
        self.axislinked_to_axis = {}

    def recursivePlotterSearch(self, widget):
        """Find a list of plotters below widget.

        Builds up a dict of "nodes" representing each widget: plotter/axis
        Each node is a list of tuples saying which widgets need evaling first
        The tuples are (widget, depname), where depname is a name for the 
        part of the plotter, e.g. "sx" or "sy" for x or y.
        """

        if widget.isplotter:
            # keep track of which widgets depend on which axes
            widgetaxes = {}
            for axname in widget.getAxesNames():
                axis = widget.lookupAxis(axname)
                widgetaxes[axname] = axis
                self.axis_plotter_map[axis].append(widget)

            # if the widget is a plotter, find which axes the widget
            # can provide range information about
            for axname, depname in widget.affectsAxisRange():
                origaxis = widgetaxes[axname]
                resolvedaxis = _resolveLinkedAxis(origaxis)
                if resolvedaxis is not None and resolvedaxis.usesAutoRange():
                    # only add dependency if axis has an automatic range
                    self.deps[(origaxis, None)].append((widget, depname))
                    self.pairs.append( ((widget, depname),
                                        (resolvedaxis, None)) )

            # find which axes the plotter needs information from
            for depname, axname in widget.requiresAxisRange():
                origaxis = widgetaxes[axname]
                resolvedaxis = _resolveLinkedAxis(origaxis)
                if resolvedaxis is not None and resolvedaxis.usesAutoRange():
                    self.deps[(widget, depname)].append((origaxis, None))
                    self.pairs.append( ((resolvedaxis, None),
                                        (widget, depname)) )

        elif widget.isaxis:
            if widget.isaxis and widget.isLinked():
                # function of another axis
                linked = widget.getLinkedAxis()
                if linked is not None:
                    self.axis_to_axislinked[linked] = widget
                    self.axislinked_to_axis[widget] = linked
            else:
                # make a range for a normal axis
                self.axes.append(widget)
                self.ranges[widget] = list(defaultrange)

        for c in widget.children:
            self.recursivePlotterSearch(c)

    def breakCycles(self, origcyclic):
        """Remove cycles if possible."""

        numcyclic = len(origcyclic)
        best = -1

        for i in crange(len(self.pairs)):
            if not self.pairs[i][0][0].isaxis:
                p = self.pairs[:i] + self.pairs[i+1:]
                ordered, cyclic = utils.topological_sort(p)
                if len(cyclic) <= numcyclic:
                    numcyclic = len(cyclic)
                    best = i

        # delete best, or last one if none better found
        p = self.pairs[best]
        del self.pairs[best]

        try:
            idx = self.deps[p[1]].index(p[0])
            del self.deps[p[1]][idx]
        except ValueError:
            pass

    def _updateAxisAutoRange(self, axis):
        """Update auto range for axis."""
        # set actual range on axis, as axis no longer has a
        # dependency
        axrange = self.ranges[axis]
        if axrange == defaultrange:
            axrange = None
        axis.setAutoRange(axrange)
        del self.ranges[axis]

    def _updateRangeFromPlotter(self, axis, plotter, plotterdep):
        """Update the range for axis from the plotter."""

        if axis.isLinked():
            # take range and map back to real axis
            therange = list(defaultrange)
            plotter.getRange(axis, plotterdep, therange)

            if therange != defaultrange:
                # follow up chain
                loopcheck = set()
                while axis.isLinked():
                    loopcheck.add(axis)
                    therange = axis.invertFunctionVals(therange)
                    axis = axis.getLinkedAxis()
                    if axis in loopcheck:
                        axis = None
                if axis is not None and therange is not None:
                    self.ranges[axis] = [
                        N.nanmin((self.ranges[axis][0], therange[0])),
                        N.nanmax((self.ranges[axis][1], therange[1]))
                        ]
        else:
            plotter.getRange(axis, plotterdep, self.ranges[axis])

    def processWidgetDeps(self, dep):
        """Process dependencies for a single widget."""
        widget, widget_dep = dep

        # iterate over dependent widgets
        for widgetd, widgetd_dep in self.deps[dep]:

            if ( widgetd.isplotter and
                 (not widgetd.settings.isSetting('hide') or
                  not widgetd.settings.hide) ):
                self._updateRangeFromPlotter(widget, widgetd, widgetd_dep)

            elif widgetd.isaxis:
                axis = _resolveLinkedAxis(widgetd)
                if axis in self.ranges:
                    self._updateAxisAutoRange(axis)

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
            self.processWidgetDeps(dep)

            # process deps for any axis functions
            while dep[0] in self.axis_to_axislinked:
                dep = (self.axis_to_axislinked[dep[0]], None)
                self.processWidgetDeps(dep)

    def findAxisRanges(self):
        """Find the ranges from the plotters and set the axis ranges.

        Follows the dependencies calculated above.
        """

        self.processDepends()

        # set any remaining ranges
        for axis in list(self.ranges.keys()):
            self._updateAxisAutoRange(axis)

class Page(widget.Widget):
    """A class for representing a page of plotting."""

    typename='page'
    allowusercreation = True
    description=_('Blank page')
 
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

        s.add( setting.Notes(
                'notes', '',
                descr=_('User-defined notes'),
                usertext=_('Notes')
                ) )

    @classmethod
    def allowedParentTypes(klass):
        from . import root
        return (root.Root,)

    @property
    def userdescription(self):
        """Return user-friendly description."""
        return textwrap.fill(self.settings.notes, 60)

    def draw(self, parentposn, painthelper, outerbounds=None):
        """Draw the plotter. Clip graph inside bounds."""

        # document should pass us the page bounds
        x1, y1, x2, y2 = parentposn

        # find ranges of axes
        axisdependhelper = AxisDependHelper()
        axisdependhelper.recursivePlotterSearch(self)
        axisdependhelper.findAxisRanges()

        # store axis->plotter mappings in painthelper
        painthelper.axisplottermap.update(axisdependhelper.axis_plotter_map)
        # reverse mapping
        pamap = collections.defaultdict(list)
        for axis, plotters in citems(painthelper.axisplottermap):
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
    
