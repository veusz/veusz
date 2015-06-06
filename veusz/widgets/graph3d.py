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

import sys
import math
import numpy as N
from ..compat import cvalues
from .. import qtall as qt4
from .. import document
from .. import setting
from . import widget

try:
    from ..helpers import threed
except ImportError:
    sys.stderr.write('Cannot import threed helper modules\n')
    threed = None

def _(text, disambiguation=None, context='Graph3D'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class BackSurface(setting.Surface3D):
    def __init__(self, name, **args):
        setting.Surface3D.__init__(self, name, **args)
        self.get('color').newDefault('white')

class Graph3D(widget.Widget):
    """3D graph (orthogonal) containing other widgets."""
    
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

    def __init__(self, parent, name=None):
        """Initialise object and create axes."""

        widget.Widget.__init__(self, parent, name=name)
        self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        widget.Widget.addSettings(s)

        s.add( setting.FloatSlider(
            'xRotation',
            0,
            minval=-180, maxval=180, step=15, tick=45,
            descr=_(u'Rotation around x axis (°)'),
            usertext=_('X rotation') ))
        s.add( setting.FloatSlider(
            'yRotation',
            0,
            minval=-180, maxval=180, step=15, tick=45,
            descr=_(u'Rotation around y axis (°)'),
            usertext=_('Y rotation') ))
        s.add( setting.FloatSlider(
            'zRotation',
            0,
            minval=-180, maxval=180, step=15, tick=45,
            descr=_(u'Rotation around z axis (°)'),
            usertext=_('Z rotation') ))

        s.add( setting.FloatSlider(
            'distance',
            5,
            minval=1, maxval=50, step=0.5, tick=5, scale=0.1,
            descr=_(u'Viewing distance'),
            usertext=_('Distance') ))

        s.add( setting.Choice(
            'rendermode',
            ('painters', 'bsp'),
            'painters',
            uilist=("Fast (Painter's)",
                    "Accurate (BSP)"),
            usertext=_('Render method'),
            descr=_('Method used to draw 3D plot') ))

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

    def addBorder(self, root):
        s = self.settings
        if s.Border.hide:
            return
        lineprop = s.Border.makeLineProp()
        edges = N.array(self._borderedges)
        ls = threed.LineSegments(
            threed.ValVector(N.ravel(edges[:,0,:])),
            threed.ValVector(N.ravel(edges[:,1,:])),
            lineprop)
        root.addObject(ls)

    def addBackSurface(self, root):
        back = self.settings.Back
        if back.hide:
            return
        prop = back.makeSurfaceProp()

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

    def draw(self, parentposn, painthelper, outerbounds=None):
        '''Update the margins before drawing.'''
        bounds = self.computeBounds(parentposn, painthelper)
        maxbounds = self.computeBounds(
            parentposn, painthelper, withmargin=False)

        if threed is None:
            return bounds

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

        for axis in cvalues(axestodraw):
            axis.computePlottedRange()

        root = threed.ObjectContainer()
        root.objM = (
            threed.rotateM4(s.zRotation/180.*math.pi, threed.Vec3(0,0,1)) *
            threed.rotateM4(s.yRotation/180.*math.pi, threed.Vec3(0,1,0)) *
            threed.rotateM4(s.xRotation/180.*math.pi, threed.Vec3(1,0,0)) *
            threed.translationM4(threed.Vec3(-0.5,-0.5,-0.5)) )

        for c in self.children:
           obj = c.drawToObject()
           if obj:
               root.addObject(obj)

        self.addBorder(root)
        self.addBackSurface(root)

        camera = threed.Camera()
        camera.setPointing(
            threed.Vec3(s.distance, 0., 0.), threed.Vec3(0.,0.,0.),
            threed.Vec3(0,0,1))
        camera.setPerspective(45, 1, 4, 6)

        mode = {
            'painters': threed.Scene.RENDER_PAINTERS,
            'bsp': threed.Scene.RENDER_BSP,
        }[s.rendermode]
        scene = threed.Scene(mode)

        painter = painthelper.painter(self, bounds)
        with painter:
            scene.render(
                root,
                painter, camera,
                bounds[0], bounds[1], bounds[2], bounds[3])

document.thefactory.register(Graph3D)
