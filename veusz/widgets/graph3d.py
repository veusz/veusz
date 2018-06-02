# -*- coding: utf-8 -*-

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

from __future__ import division, print_function

import math
import numpy as N
from ..compat import cvalues
from .. import qtall as qt
from .. import document
from .. import setting
from . import widget
from ..helpers import threed

def _(text, disambiguation=None, context='Graph3D'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

# non-reflective back of cube
class BackSurface(setting.Surface3D):
    def __init__(self, name, **args):
        setting.Surface3D.__init__(self, name, **args)
        self.get('color').newDefault('white')
        self.get('reflectivity').newDefault(0)
        self.get('hide').newDefault(True)

class Graph3D(widget.Widget):
    """3D graph (orthogonal axes) containing other plotting widgets."""

    typename='graph3d'
    allowusercreation = True
    description = _('3d graph')

    # start and end points of edges of cube
    _borderedges = (
        ((0,0,0), (0,0,1)),
        ((0,0,0), (0,1,0)),
        ((0,0,0), (1,0,0)),
        ((0,0,1), (0,1,1)),
        ((0,0,1), (1,0,1)),
        ((0,1,0), (0,1,1)),
        ((0,1,0), (1,1,0)),
        ((0,1,1), (1,1,1)),
        ((1,0,0), (1,0,1)),
        ((1,0,0), (1,1,0)),
        ((1,0,1), (1,1,1)),
        ((1,1,0), (1,1,1)),
    )

    # centres of each face
    _facecentres = (
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0.5, 1.0),
        (0.5, 1.0, 0.5),
        (1.0, 0.5, 0.5),
    )

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        s.add( setting.Float(
            'xSize', 1.,
            minval=0.01, maxval=100,
            descr=_('X size'),
            usertext=_('X size') ))
        s.add( setting.Float(
            'ySize', 1.,
            minval=0.01, maxval=100,
            descr=_('Y size'),
            usertext=_('Y size') ))
        s.add( setting.Float(
            'zSize', 1.,
            minval=0.01, maxval=100,
            descr=_('Z size'),
            usertext=_('Z size') ))

        s.add( setting.Float(
            'xPos', 0.,
            minval=-100, maxval=100,
            descr=_('X position'),
            usertext=_('X position') ))
        s.add( setting.Float(
            'yPos', 0.,
            minval=-100, maxval=100,
            descr=_('Y position'),
            usertext=_('Y position') ))
        s.add( setting.Float(
            'zPos', 0.,
            minval=-100, maxval=100,
            descr=_('Z position'),
            usertext=_('Z position') ))

        s.add(setting.Line3D(
            'Border',
            descr = _('Graph border'),
            usertext = _('Border')),
               pixmap = 'settings_border' )
        s.add(BackSurface(
            'Back',
            descr = _('Graph back'),
            usertext = _('Back')),
               pixmap = 'settings_bgfill' )

    @classmethod
    def allowedParentTypes(self):
        from . import scene3d
        return (scene3d.Scene3D,)

    def addDefaultSubWidgets(self):
        """Add axes automatically."""
        from . import axis3d
        for n in ('x', 'y', 'z'):
            if self.parent.getChild(n) is None:
                ax = axis3d.Axis3D(self, name=n)
                ax.linkToStylesheet()

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

    def addBorder(self, painter, root):
        s = self.settings
        if s.Border.hide:
            return
        lineprop = s.Border.makeLineProp(painter)
        edges = N.array(self._borderedges)
        ls = threed.LineSegments(
            threed.ValVector(N.ravel(edges[:,0,:])),
            threed.ValVector(N.ravel(edges[:,1,:])),
            lineprop)
        root.addObject(ls)

    def addBackSurface(self, painter, root):
        back = self.settings.Back
        if back.hide:
            return
        prop = back.makeSurfaceProp(painter)

        # triangles with the correct orientation of the norm vector
        # not to draw the surface if it is pointing towards the viewer
        for p1, p2, p3 in (
                ((0,0,0), (0,0,1), (1,0,0)),
                ((0,0,1), (0,0,0), (0,1,0)),
                ((0,1,0), (0,1,1), (0,0,1)),
                ((0,1,0), (1,1,0), (0,1,1)),
                ((0,1,0), (0,0,0), (1,0,0)),
                ((0,1,1), (1,0,1), (0,0,1)),
                ((0,1,1), (1,1,1), (1,0,1)),
                ((1,0,0), (1,1,0), (0,1,0)),
                ((1,0,1), (1,0,0), (0,0,1)),
                ((1,0,1), (1,1,0), (1,0,0)),
                ((1,0,1), (1,1,1), (1,1,0)),
                ((1,1,0), (1,1,1), (0,1,1)),
        ):
            root.addObject(threed.TriangleFacing(
                threed.Vec3(*p1), threed.Vec3(*p2), threed.Vec3(*p3), prop))

    def drawToObject(self, painter, painthelper):
        """Make objects, returning root"""

        s = self.settings

        # do no painting if hidden
        if s.hide:
            return

        # do axis min-max computation
        axestodraw = {}
        axesofwidget = painthelper.plotteraxismap

        for c in self.children:
            try:
                for a in axesofwidget[c]:
                    axestodraw[a.name] = a
            except (KeyError, AttributeError):
                if c.isaxis:
                    axestodraw[c.name] = c

        for axis in cvalues(axestodraw):
            axis.computePlottedRange()

        cont = threed.ObjectContainer()
        cont.objM = (
            # graph position
            threed.translationM4(threed.Vec3(
                s.xPos - 0.5*s.xSize,
                s.yPos - 0.5*s.ySize,
                s.zPos - 0.5*s.zSize)) *
            # graph size
            threed.scaleM4(threed.Vec3(s.xSize, s.ySize, s.zSize))
        )

        # add graph box
        self.addBorder(painter, cont)
        # add graph behind fill
        self.addBackSurface(painter, cont)
        # make clickable
        cont.assignWidgetId(id(self))

        # reset counter and compute automatic colors
        painthelper.autoplottercount = 0
        for c in self.children:
            c.setupAutoColor(painter)

        # build 3d scene from children
        for c in self.children:
           obj = c.drawToObject(painter, painthelper)
           if obj:
               cont.addObject(obj)

        return cont

document.thefactory.register(Graph3D)
