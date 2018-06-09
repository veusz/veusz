# -*- coding: utf-8 -*-

#    Copyright (C) 2018 Jeremy S. Sanders
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

from .. import qtall as qt
from .. import document
from .. import setting
from . import widget
from . import controlgraph
from ..helpers import threed

def _(text, disambiguation=None, context='Scene3D'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

# define 2nd and 3rd lighting classes
class Lighting3D_2(setting.Lighting3D):
    def __init__(self, name, **args):
        setting.Lighting3D.__init__(self, name, **args)
        self.get('enable').newDefault(False)
        self.get('color').newDefault('red')
        self.get('x').newDefault(2)
class Lighting3D_3(setting.Lighting3D):
    def __init__(self, name, **args):
        setting.Lighting3D.__init__(self, name, **args)
        self.get('enable').newDefault(False)
        self.get('color').newDefault('blue')
        self.get('x').newDefault(-2)

class Scene3D(widget.Widget):
    """3D scene containing other widgets."""

    typename='scene3d'
    allowusercreation = True
    description = _('3d scene')

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
            35.,
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

        s.add( setting.FloatOrAuto(
            'size',
            'Auto',
            minval=0,
            descr=_('Automatic or fixed graph size scaling value'),
            usertext=_('Size'),
            formatting=True ))

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
                '1cm',
                descr=_('Distance from left of graph to edge'),
                usertext=_('Left margin'),
                formatting=True) )
        s.add( setting.Distance(
                'rightMargin',
                '1cm',
                descr=_('Distance from right of graph to edge'),
                usertext=_('Right margin'),
                formatting=True) )
        s.add( setting.Distance(
                'topMargin',
                '1cm',
                descr=_('Distance from top of graph to edge'),
                usertext=_('Top margin'),
                formatting=True) )
        s.add( setting.Distance(
                'bottomMargin',
                '1cm',
                descr=_('Distance from bottom of graph to edge'),
                usertext=_('Bottom margin'),
                formatting=True) )

        s.add(setting.Lighting3D(
            'Lighting1',
            descr=_('Lighting (1)'),
            usertext=_('Lighting (1)')),
               pixmap = 'settings_lighting' )
        s.add(Lighting3D_2(
            'Lighting2',
            descr=_('Lighting (2)'),
            usertext=_('Lighting (2)')),
               pixmap = 'settings_lighting' )
        s.add(Lighting3D_3(
            'Lighting3',
            descr=_('Lighting (3)'),
            usertext=_('Lighting (3)')),
               pixmap = 'settings_lighting' )

    @classmethod
    def allowedParentTypes(self):
        from . import page, grid
        return (page.Page, grid.Grid)

    def getMargins(self, painthelper):
        """Use settings to compute margins."""
        s = self.settings
        return ( s.get('leftMargin').convert(painthelper),
                 s.get('topMargin').convert(painthelper),
                 s.get('rightMargin').convert(painthelper),
                 s.get('bottomMargin').convert(painthelper) )

    def makeObjects(self, painter, bounds, painthelper):
        """Make objects, returning root"""

        s = self.settings

        # do no painting if hidden
        if s.hide:
            return

        root = threed.ObjectContainer()
        root.objM = threed.rotate3M4(
            s.xRotation/180.*math.pi,
            s.yRotation/180.*math.pi,
            s.zRotation/180.*math.pi)

        # build 3d scene from children
        for c in self.children:
           obj = c.drawToObject(painter, painthelper)
           if obj:
               root.addObject(obj)

        return root

    def makeScene(self, painter):
        """Make Scene and Camera objects."""

        s = self.settings

        camera = threed.Camera()
        # camera necessary to make x,y,z coordinates point in the
        # right direction, with the origin in the lower left towards
        # the viewer
        camera.setPointing(
            threed.Vec3(0,  0, -s.distance),
            threed.Vec3(0,  0,  0),
            threed.Vec3(0, -1,  0))
        camera.setPerspective(90, 1, 100)

        mode = {
            'painters': threed.Scene.RENDER_PAINTERS,
            'bsp': threed.Scene.RENDER_BSP,
        }[s.rendermode]
        scene = threed.Scene(mode)

        # add lighting if enabled
        for light in s.Lighting1, s.Lighting2, s.Lighting3:
            if light.enable:
                scene.addLight(
                    # FIXME: z and y negative here to make direction
                    # correct relative to camera origin
                    threed.Vec3(light.x, -light.y, -light.z),
                    light.get('color').color(painter),
                    light.intensity*0.01)

        return scene, camera

    def draw(self, parentposn, painthelper, outerbounds=None):
        '''Update the margins before drawing.'''

        bounds = self.computeBounds(parentposn, painthelper)

        painter = painthelper.painter(self, bounds)
        painthelper.setControlGraph(self, [
            controlgraph.ControlMarginBox(
                self, bounds, outerbounds, painthelper)])

        root = self.makeObjects(painter, bounds, painthelper)
        if root is None:
            return bounds

        scene, camera = self.makeScene(painter)

        # def ptToScreen(pt):
        #     pt_v = (camera.viewM*root.objM)*threed.Vec4(pt[0],pt[1],pt[2],1)
        #     pt_proj = threed.calcProjVec(camera.perspM, pt_v)
        #     pt_screen = threed.projVecToScreen(scene.screenM, pt_proj)
        #     return pt_screen,qt.QPointF(pt_screen.get(0), pt_screen.get(1))

        # finally render the scene
        scale = self.settings.size
        if scale == 'Auto':
            scale = -1
        with painter:
            scene.render(
                root,
                painter, camera,
                bounds[0], bounds[1], bounds[2], bounds[3], scale)

        #     painter.setPen(qt.QPen(qt.Qt.red))
        #     origin = ptToScreen((0,0,0))[1]
        #     for axpt in ((0.5,0,0),(0,0.5,0),(0,0,0.5)):
        #         painter.drawLine(origin, ptToScreen(axpt)[1])

        # axx = threed.Vec4(0,0.5,0,1)
        # threed.solveInverseRotation(camera.viewM, camera.perspM, scene.screenM, axx, ptToScreen((0,0.5,0))[0])

    def identifyWidgetAtPoint(self, painthelper, bounds, scaling, x, y):
        painter = document.PainterRoot()
        painter.updateMetaData(painthelper)

        root = self.makeObjects(painter, bounds, painthelper)
        if root is None:
            return self
        scene, camera = self.makeScene(painter)

        sizescale = self.settings.size
        if sizescale == 'Auto':
            sizescale = -1
        widgetid = scene.idPixel(
            root, painter, camera,
            bounds[0], bounds[1], bounds[2], bounds[3], sizescale,
            scaling, x, y)

        # recursive check id of children against returned value
        widget = [self]
        def checkwidget(r):
            for c in r.children:
                if id(c) == widgetid:
                    widget[0] = c
                checkwidget(c)

        checkwidget(self)
        return widget[0]

    def updateControlItem(self, cgi):
        """Area moved or resized - call helper routine to move self."""
        cgi.setWidgetMargins()

document.thefactory.register(Scene3D)
