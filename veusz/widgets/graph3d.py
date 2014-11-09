#    Copyright (C) 2013 Jeremy S. Sanders
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

from .. import qtall as qt4
from .. import document
from .. import setting
from .. import threed
from . import widget
from ..helpers import threed

def _(text, disambiguation=None, context='Graph3D'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class Graph3D(widget.Widget):
    """3D graph (orthogonal) containing other widgets."""
    
    typename='graph3d'
    allowusercreation = True
    description = _('3d graph')

    def __init__(self, parent, name=None):
        """Initialise object and create axes."""

        widget.Widget.__init__(self, parent, name=name)
        self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        s.add( setting.Distance(
                'leftMargin',
                '0.2cm',
                descr=_('Distance from left of graph to edge'),
                usertext=_('Left margin'),
                formatting=True) )
        s.add( setting.Distance(
                'rightMargin',
                '0.2cm',
                descr=_('Distance from right of graph to edge'),
                usertext=_('Right margin'),
                formatting=True) )
        s.add( setting.Distance(
                'topMargin',
                '0.2cm',
                descr=_('Distance from top of graph to edge'),
                usertext=_('Top margin'),
                formatting=True) )
        s.add( setting.Distance(
                'bottomMargin',
                '0.2cm',
                descr=_('Distance from bottom of graph to edge'),
                usertext=_('Bottom margin'),
                formatting=True) )

    @classmethod
    def allowedParentTypes(self):
        from . import page, grid
        return (page.Page, grid.Grid)

    def addDefaultSubWidgets(self):
        """Add axes automatically."""
        from . import axis3d
        for n in ('x', 'y', 'z'):
            if self.parent.getChild(n) is None:
                axis3d.Axis3D(self, name=n)

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
                     hasattr(c, 'isaxis3d') and c.isaxis3d ):
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

    def getMargins(self, painthelper):
        """Use settings to compute margins."""
        s = self.settings
        return ( s.get('leftMargin').convert(painthelper),
                 s.get('topMargin').convert(painthelper),
                 s.get('rightMargin').convert(painthelper),
                 s.get('bottomMargin').convert(painthelper) )

    def draw(self, parentposn, painthelper, outerbounds=None):
        '''Update the margins before drawing.'''
        bounds = self.computeBounds(parentposn, painthelper)
        maxbounds = self.computeBounds(
            parentposn, painthelper, withmargin=False)

        s = self.settings

        # do no painting if hidden
        if s.hide:
            return bounds

        axestodraw = {}
        axesofwidget = painthelper.plotteraxismap

        for c in self.children:
            try:
                for a in axesofwidget[c]:
                    axestodraw[a.name] = a
            except (KeyError, AttributeError):
                if c.isaxis:
                    axestodraw[c.name] = c

        scene = threed.Scene()
        for c in self.children:
            obj = c.drawToObject()
            if obj:
                scene.root.addObject(obj)

        def printM(m):
            for i in xrange(4):
                s=''
                for j in xrange(4):
                    s += '%8.3f ' % m.get(i, j)
                print(s)

        borderlineprop = threed.LineProp()
        camera = threed.Camera()
        camera.setPointing(
            threed.Vec3(-8,-8,-8), threed.Vec3(0.,0.,0),
            threed.Vec3(0,-1,0))

        painter = painthelper.painter(self, bounds)
        with painter:
            scene.render(
                painter, camera,
                bounds[0], bounds[1], bounds[2], bounds[3])

document.thefactory.register(Graph3D)
