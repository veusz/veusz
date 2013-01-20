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

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.utils as utils
import veusz.document as document

import widget
import axis
import page
import grid
import controlgraph

def _(text, disambiguation=None, context='Graph'):
    """Translate text."""
    return unicode( 
        qt4.QCoreApplication.translate(context, text, disambiguation))

class Graph(widget.Widget):
    """Graph for containing other sorts of widgets"""
    
    typename='graph'
    allowedparenttypes = [page.Page, grid.Grid]
    allowusercreation = True
    description = _('Base graph')

    def __init__(self, parent, name=None):
        """Initialise object and create axes."""

        widget.Widget.__init__(self, parent, name=name)
        self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        s.add( setting.Distance( 'leftMargin',
                                 '1.7cm',
                                 descr=_('Distance from left of graph to edge'),
                                 usertext=_('Left margin'),
                                 formatting=True) )
        s.add( setting.Distance( 'rightMargin',
                                 '0.2cm',
                                 descr=_('Distance from right of graph to edge'),
                                 usertext=_('Right margin'),
                                 formatting=True) )
        s.add( setting.Distance( 'topMargin',
                                 '0.2cm',
                                 descr=_('Distance from top of graph to edge'),
                                 usertext=_('Top margin'),
                                 formatting=True) )
        s.add( setting.Distance( 'bottomMargin',
                                 '1.7cm',
                                 descr=_('Distance from bottom of graph to edge'),
                                 usertext=_('Bottom margin'),
                                 formatting=True) )
        s.add( setting.GraphBrush( 'Background',
                                   descr = _('Background plot fill'),
                                   usertext=_('Background')),
               pixmap='settings_bgfill' )
        s.add( setting.Line('Border', descr = _('Graph border line'),
                            usertext=_('Border')),
               pixmap='settings_border')
        
    def addDefaultSubWidgets(self):
        """Add axes automatically."""

        if self.parent.getChild('x') is None:
            axis.Axis(self, name='x')
        if self.parent.getChild('y') is None:
            axis.Axis(self, name='y')

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
                     isinstance(c, axis.Axis) ):
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

    def draw(self, parentposn, painthelper, outerbounds = None):
        '''Update the margins before drawing.'''

        s = self.settings

        margins = ( s.get('leftMargin').convert(painthelper),
                    s.get('topMargin').convert(painthelper),
                    s.get('rightMargin').convert(painthelper),
                    s.get('bottomMargin').convert(painthelper) )

        bounds = self.computeBounds(parentposn, painthelper, margins=margins)
        maxbounds = self.computeBounds(parentposn, painthelper)

        # controls for adjusting graph margins
        painter = painthelper.painter(self, bounds)
        with painter:
            painthelper.setControlGraph(self, [
                    controlgraph.ControlMarginBox(self, bounds, maxbounds,
                                                  painthelper) ])

            # do no painting if hidden
            if s.hide:
                return bounds

            # set graph rectangle attributes
            path = qt4.QPainterPath()
            path.addRect( qt4.QRectF(qt4.QPointF(bounds[0], bounds[1]),
                                     qt4.QPointF(bounds[2], bounds[3])) )
            utils.brushExtFillPath(painter, s.Background, path,
                                   stroke=s.Border.makeQPenWHide(painter))

            # child drawing algorithm is a bit complex due to axes
            # being shared between graphs and broken axes

            # get list of axes to draw
            axestodraw = set()
            for c in self.children:
                try:
                    for axis in c.getAxesNames():
                        axestodraw.add(axis)
                except AttributeError:
                    if hasattr(c, 'isaxis'):
                        axestodraw.add(c.name)

            axes = self.getAxesDict(axestodraw, ignoremissing=True)
            axeswidgets = set(axes.values())

            # grid lines are normally plotted before other child widgets
            for axis in axeswidgets:
                axis.drawGrid(bounds, painthelper, outerbounds=outerbounds,
                              ontop=False)

            # do normal drawing of children
            # iterate over children in reverse order
            for c in reversed(self.children):
                if c not in axeswidgets:
                    c.draw(bounds, painthelper, outerbounds=outerbounds)

            # then for grid lines on top
            for axis in axeswidgets:
                axis.drawGrid(bounds, painthelper, outerbounds=outerbounds,
                              ontop=True)

            # draw axes on top of grid lines
            for aname, awidget in sorted(axes.items()):
                awidget.draw(bounds, painthelper, outerbounds=outerbounds)

        return bounds

    def updateControlItem(self, cgi):
        """Graph resized or moved - call helper routine to move self."""
        cgi.setWidgetMargins()

# allow users to make Graph objects
document.thefactory.register( Graph )

