# graph widget for containing other sorts of widget

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

from __future__ import division
import textwrap

from .. import qtall as qt
from .. import setting
from .. import utils
from .. import document

from . import widget
from . import controlgraph

def _(text, disambiguation=None, context='Graph'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class Graph(widget.Widget):
    """Graph for containing other sorts of widgets"""

    typename='graph'
    allowusercreation = True
    description = _('Base graph')

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        s.add( setting.Distance(
            'leftMargin',
            '1.7cm',
            descr=_('Distance from left of graph to edge'),
            usertext=_('Left margin'),
            formatting=True,
        ) )
        s.add( setting.Distance(
            'rightMargin',
            '0.2cm',
            descr=_('Distance from right of graph to edge'),
            usertext=_('Right margin'),
            formatting=True,
        ) )
        s.add( setting.Distance(
            'topMargin',
            '0.2cm',
            descr=_('Distance from top of graph to edge'),
            usertext=_('Top margin'),
            formatting=True,
        ) )
        s.add( setting.Distance(
            'bottomMargin',
            '1.7cm',
            descr=_('Distance from bottom of graph to edge'),
            usertext=_('Bottom margin'),
            formatting=True,
        ) )
        s.add( setting.FloatOrAuto(
            'aspect',
            'Auto',
            descr=_('Fix aspect ratio of graph to this value'),
            usertext=_('Aspect ratio'),
            minval = 0.01,
            maxval = 100.,
            formatting=True,
        ) )

        s.add( setting.Notes(
            'notes', '',
            descr=_('User-defined notes'),
            usertext=_('Notes')
        ) )

        s.add(
            setting.GraphBrush(
                'Background',
                descr = _('Background plot fill'),
                usertext=_('Background')),
            pixmap='settings_bgfill',
        )
        s.add(
            setting.Line(
                'Border',
                descr = _('Graph border line'),
                usertext=_('Border')),
            pixmap='settings_border',
        )

    @classmethod
    def allowedParentTypes(klass):
        from . import page, grid
        return (page.Page, grid.Grid)

    @property
    def userdescription(self):
        """Return user-friendly description."""
        return textwrap.fill(self.settings.notes, 60)

    def addDefaultSubWidgets(self):
        """Add axes automatically."""

        from . import axis
        if self.parent.getChild('x') is None:
            ax = axis.Axis(self, name='x')
            ax.linkToStylesheet()
        if self.parent.getChild('y') is None:
            ay = axis.Axis(self, name='y')
            ay.linkToStylesheet()

    def getAxesDict(self, axesnames, ignoremissing=False):
        """Get the axes for widgets to plot against.
        axesnames is a list/set of names to find.
        Returns a dict of objects
        """

        axes = {}
        # recursively go back up the tree to find axes
        w = self
        while w is not None and len(axes) < len(axesnames):
            for c in w.children:
                name = c.name
                if ( name in axesnames and name not in axes and
                     c.isaxis ):
                    axes[name] = c
            w = w.parent

        # didn't find everything...
        if w is None and not ignoremissing:
            for name in axesnames:
                if name not in axes:
                    axes[name] = None

        # return list of found axes
        return axes

    def getAxes(self, axesnames):
        """Return a list of axes widgets given a list of names."""
        ad = self.getAxesDict(axesnames)
        return [ad[n] for n in axesnames]

    def adjustBoundsForAspect(self, bounds):
        s = self.settings
        if s.aspect != 'Auto':
            saspect = s.aspect
            width = bounds[2]-bounds[0]
            height = bounds[3]-bounds[1]
            gaspect = width/height

            bounds = list(bounds)
            if saspect > gaspect:
                # want a graph which is wider than the current size
                # => add space to top/bottom
                newheight = width / saspect
                delta = (height-newheight) / 2
                bounds[1] += delta
                bounds[3] -= delta
            else:
                # want a graph which is narrower than the current size
                # => add space to left/right
                newwidth = height * saspect
                delta = (width-newwidth) / 2
                bounds[0] += delta
                bounds[2] -= delta

        return bounds

    def getMargins(self, painthelper):
        """Use settings to compute margins."""
        s = self.settings
        return (
            s.get('leftMargin').convert(painthelper),
            s.get('topMargin').convert(painthelper),
            s.get('rightMargin').convert(painthelper),
            s.get('bottomMargin').convert(painthelper),
        )

    def draw(self, parentposn, painthelper, outerbounds = None):
        '''Update the margins before drawing.'''

        # yuck, avoid circular imports
        from . import axisbroken

        s = self.settings

        bounds = self.computeBounds(parentposn, painthelper)
        maxbounds = self.computeBounds(
            parentposn, painthelper, withmargin=False)

        # do no painting if hidden
        if s.hide:
            return bounds

        # controls for adjusting graph margins
        painter = painthelper.painter(self, bounds)
        painthelper.setControlGraph(self, [
            controlgraph.ControlMarginBox(self, bounds, maxbounds, painthelper)
        ])

        bounds = self.adjustBoundsForAspect(bounds)

        with painter:
            # set graph rectangle attributes
            path = qt.QPainterPath()
            path.addRect(qt.QRectF(
                qt.QPointF(bounds[0], bounds[1]),
                qt.QPointF(bounds[2], bounds[3]) ))
            utils.brushExtFillPath(
                painter, s.Background, path,
                stroke=s.Border.makeQPenWHide(painter))

            # debugging positions (uncomment)
            # painter.drawRect( qt.QRectF(
            #         qt.QPointF(parentposn[0], parentposn[1]),
            #         qt.QPointF(parentposn[2], parentposn[3]) ))

            # if outerbounds:
            #     painter.drawRect( qt.QRectF(
            #             qt.QPointF(outerbounds[0], outerbounds[1]),
            #             qt.QPointF(outerbounds[2], outerbounds[3]) ))

        # child drawing algorithm is a bit complex due to axes
        # being shared between graphs and broken axes

        # this is a map of axis names to plot to axis widgets
        axestodraw = {}
        # axes widgets for each plotter (precalculated by Page)
        axesofwidget = painthelper.plotteraxismap
        for c in self.children:
            try:
                for a in axesofwidget[c]:
                    axestodraw[a.name] = a
            except (KeyError, AttributeError):
                if c.isaxis:
                    axestodraw[c.name] = c

        # grid lines are normally plotted before other child widgets
        axisdrawlist = sorted(axestodraw.items(), reverse=True)
        for aname, awidget in axisdrawlist:
            awidget.updateAxisLocation(bounds)
            awidget.computePlottedRange()
            awidget.drawGrid(
                bounds, painthelper, outerbounds=outerbounds, ontop=False)

        # broken axis handling
        brokenaxes = set()
        for axis in axestodraw.values():
            if isinstance(axis, axisbroken.AxisBroken):
                brokenaxes.add(axis)

        # don't duplicate drawing axes
        axesdrawn = set()

        # reset counter and compute automatic colors
        painthelper.autoplottercount = 0
        for c in self.children:
            c.setupAutoColor(painter)

        # do normal drawing of children
        # iterate over children in reverse order
        for c in reversed(self.children):

            if c.isaxis:
                axesdrawn.add(c)

            axes = axesofwidget.get(c, None)
            if axes is not None and any((a in brokenaxes for a in axes)):
                # handle broken axes
                childbrokenaxes = sorted(
                    [(a.name, a) for a in axes if a in brokenaxes],
                    key=lambda x: x[0])
                def iteratebrokenaxes(b):
                    """Recursively iterate over each broken axis and redraw
                    child for each.
                    We might have more than one broken axis per child, so
                    hence this rather strange iteration.
                    """
                    ax = b[0][1]
                    for i in range(ax.breakvnum):
                        ax.switchBreak(i, bounds)
                        if len(b) == 1:
                            c.draw(bounds, painthelper, outerbounds=outerbounds)
                        else:
                            iteratebrokenaxes(b[1:])
                    ax.switchBreak(None, bounds)
                iteratebrokenaxes(childbrokenaxes)

            else:

                # standard non broken axis drawing
                c.draw(bounds, painthelper, outerbounds=outerbounds)

        # then for grid lines on top
        axiswidgets = [axis for name, axis in axisdrawlist]
        for awidget in axiswidgets:
            awidget.drawGrid(
                bounds, painthelper, outerbounds=outerbounds, ontop=True)
            awidget.drawAutoMirror(bounds, painthelper, axiswidgets)

        # draw remaining axes
        for awidget in axiswidgets:
            if awidget not in axesdrawn:
                awidget.draw(bounds, painthelper, outerbounds=outerbounds)

        return bounds

    def updateControlItem(self, cgi):
        """Graph resized or moved - call helper routine to move self."""
        cgi.setWidgetMargins()

# allow users to make Graph objects
document.thefactory.register( Graph )
