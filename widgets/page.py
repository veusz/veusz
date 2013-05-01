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

import widget
import root
import controlgraph
import axisuser

# x, y, fplot, xyplot
# x-> xyplot(x)
# y-> xyplot(y)
# fplot(y) -> x
# y -> fplot(y)

# x -> xyplot(x)
# y -> (xyplot(y), fplot(y) -> x)

def _(text, disambiguation=None, context='Page'):
    """Translate text."""
    return unicode( 
        qt4.QCoreApplication.translate(context, text, disambiguation))

defaultrange = [1e99, -1e99]

class _AxisDependHelper(object):
    """A class to work out the dependency of widgets on axes and vice
    versa, in terms of ranges of the axes.

    It then works out the ranges for each of the axes from the plotters.
    """

    def __init__(self, root):
        self.root = root
        self.nodes = collections.defaultdict(list)
        self.axes = []
        self.axis_plotter_map = collections.defaultdict(list)

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
                axdep = (axis, None)
                self.nodes[axdep].append( (widget, depname) )

            # find which axes the axis-user needs information from
            for depname, axname in widget.requiresAxisRange():
                axis = widgetaxes[axname]
                widdep = (widget, depname)
                self.nodes[widdep].append( (axis, None) )

        if hasattr(widget, 'isaxis'):
            # keep track of all axis widgets
            self.axes.append(widget)

        for c in widget.children:
            self.recursivePlotterSearch(c)

    def findPlotters(self):
        """Construct a list of plotters associated with each axis.
        Returns nodes:

        {axisobject: [plotterobject, plotter2...]),
         ...}
        """

        self.recursivePlotterSearch(self.root)
        print self.nodes
        self.ranges = dict( [(a, list(defaultrange)) for a in self.axes] )

    def processDepends(self, widget, depends):
        """Go through dependencies of widget.
        If the dependency has no dependency itself, then update the
        axis with the widget or vice versa

        Algorithm:
          Iterate over dependencies for widget.

          If the widget has a dependency on a widget which doesn't
          have a dependency itself, update range from that
          widget. Then delete that depency from the dependency list.
        """

        modified = False
        i = 0
        while i < len(depends):
            dep = depends[i]

            print "  ", dep[0]
            if dep not in self.nodes:
                dwidget, dwidget_dep = dep

                if ( isinstance(dwidget, axisuser.AxisUser) and
                    (not dwidget.settings.isSetting('hide') or
                     not dwidget.settings.hide) ):
                    # update range of axis with (dwidget, dwidget_dep)
                    # do not do this if the widget is hidden

                    print "getRange", dwidget, widget, dwidget_dep
                    dwidget.getRange(widget, dwidget_dep,
                                     self.ranges[widget])

                if hasattr(dwidget, 'isaxis') and dwidget in self.ranges:
                    # set actual range on axis, as axis no longer has a
                    # dependency
                    axrange = self.ranges[dwidget]
                    if axrange == defaultrange:
                        axrange = None
                    print "setAutoRange", dwidget, axrange
                    dwidget.setAutoRange(axrange)
                    del self.ranges[dwidget]

                del depends[i]
                modified = True
                continue
            i += 1
        return modified

    def breakBidirectional(self):



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
                if widget and self.processDepends(widget, depends):
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


        print "nodes"
        for key, val in self.nodes.iteritems():
            print key, key[0].name
            for i in val:
                print " ", i, i[0].name
        print


        # set any remaining ranges
        for axis, axrange in self.ranges.iteritems():
            if axrange == defaultrange:
                axrange = None
            print "setAutoRange", axis, axrange
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
        axisdependhelper = _AxisDependHelper(self)
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
    
