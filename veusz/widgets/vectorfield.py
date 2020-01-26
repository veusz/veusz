# -*- coding: utf-8 -*-

#    Copyright (C) 2010 Jeremy S. Sanders
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
###############################################################################

from __future__ import division
import numpy as N

from ..compat import czip
from .. import setting
from .. import document
from .. import utils
from .. import qtall as qt

from . import plotters

def _(text, disambiguation=None, context='VectorField'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class VectorField(plotters.GenericPlotter):
    '''A plotter for plotting a vector field.'''

    typename = 'vectorfield'
    allowusercreation = True
    description = _('Plot a vector field')

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        plotters.GenericPlotter.addSettings(s)

        # datasets
        s.add( setting.DatasetExtended(
                'data1', '',
                dimensions = 2,
                descr = _('X coordinate length or vector magnitude'),
                usertext = _('dx or r')),
               0 )
        s.add( setting.DatasetExtended(
                'data2', '',
                dimensions = 2,
                descr = _('Y coordinate length or vector angle'),
                usertext = _('dy or theta')),
               1 )
        s.add( setting.Choice('mode',
                              ['cartesian', 'polar'],
                              'cartesian',
                              descr = _('Cartesian (dx,dy) or polar (r,theta)'),
                              usertext = _('Mode')),
               2 )
        s.add( setting.FloatChoice(
            'rotate',
            [0., 45., 90., 135., 180., -135., -90., -45.],
            0.,
            descr = _('Rotate vector clockwise by this angle in degrees'),
            usertext = _('Rotate')),
               3 )
        s.add( setting.Bool(
            'reflectx', False,
            descr = _('Reflect vector in X direction'),
            usertext = _('Reflect X')),
               4 )
        s.add( setting.Bool(
            'reflecty', False,
            descr = _('Reflect vector in Y direction'),
            usertext = _('Reflect Y')),
               5 )

        # formatting
        s.add( setting.DistancePt('baselength', '10pt',
                                  descr = _('Base length of unit vector'),
                                  usertext = _('Base length'),
                                  formatting=True),
               0 )
        s.add( setting.DistancePt('arrowsize', '2pt',
                                  descr = _('Size of any arrows'),
                                  usertext = _('Arrow size'),
                                  formatting=True),
               1 )
        s.add( setting.Bool('scalearrow', True,
                            descr = _('Scale arrow head by length'),
                            usertext = _('Scale arrow'),
                            formatting=True),
               2 )
        s.add( setting.Arrow('arrowfront', 'none',
                             descr = _('Arrow in front direction'),
                             usertext=_('Arrow front'), formatting=True),
               3)
        s.add( setting.Arrow('arrowback', 'none',
                             descr = _('Arrow in back direction'),
                             usertext=_('Arrow back'), formatting=True),
               4)

        s.add( setting.Line('Line',
                            descr = _('Line style'),
                            usertext = _('Line')),
               pixmap = 'settings_plotline' )
        s.add( setting.ArrowFill('Fill',
                                 descr = _('Arrow fill settings'),
                                 usertext = _('Arrow fill')),
               pixmap = 'settings_plotmarkerfill' )

    def affectsAxisRange(self):
        """Range information provided by widget."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def getRange(self, axis, depname, axrange):
        """Automatically determine the ranges of variable on the axes."""

        for name in ('data1', 'data2'):
            data = self.settings.get(name).getData(self.document)
            if data is None:
                continue

            if data.dimensions == 2:
                xr, yr = data.getDataRanges()
                if depname == 'sx':
                    dxrange = xr
                    axrange[0] = min( axrange[0], dxrange[0] )
                    axrange[1] = max( axrange[1], dxrange[1] )
                elif depname == 'sy':
                    dyrange = yr
                    axrange[0] = min( axrange[0], dyrange[0] )
                    axrange[1] = max( axrange[1], dyrange[1] )

    def drawKeySymbol(self, number, painter, x, y, width, height):
        """Draw the plot symbol and/or line."""
        painter.save()

        s = self.settings
        painter.setPen( s.Line.makeQPenWHide(painter) )
        painter.setBrush( s.get('Fill').makeQBrushWHide(painter) )
        utils.plotLineArrow(
            painter, x+width, y+height*0.5,
            width, 180, height*0.25,
            arrowleft=s.arrowfront,
            arrowright=s.arrowback)

        painter.restore()

    def dataDraw(self, painter, axes, posn, cliprect):
        """Draw the widget."""

        s = self.settings
        d = self.document

        # ignore non existing datasets
        data1 = s.get('data1').getData(d)
        data2 = s.get('data2').getData(d)
        if data1 is None or data2 is None:
            return

        # require 2d datasets
        if data1.dimensions != 2 or data2.dimensions != 2:
            return

        # get base length (ensure > 0)
        baselength = max(s.get('baselength').convert(painter), 1e-6)

        # try to be nice if the datasets don't match
        data1st, data2nd = data1.data, data2.data
        xw = min(data1st.shape[1], data2nd.shape[1])
        yw = min(data1st.shape[0], data2nd.shape[0])

        # get pixel coordinates
        xc, yc = data1.getPixelCentres()
        xc, yc = xc[:xw], yc[:yw]
        xdsvals = N.reshape(N.tile(xc, yw), xw*yw)
        ydsvals = N.reshape(N.tile(yc[:, N.newaxis], xw), xw*yw)

        # convert using axes to plotter values
        xplotter = axes[0].dataToPlotterCoords(posn, xdsvals)
        yplotter = axes[1].dataToPlotterCoords(posn, ydsvals)

        pen = s.Line.makeQPenWHide(painter)
        painter.setPen(pen)

        if s.mode == 'cartesian':
            dx = (data1st[:yw, :xw] * baselength).ravel()
            dy = (data2nd[:yw, :xw] * baselength).ravel()

        elif s.mode == 'polar':
            r = data1st[:yw, :xw].ravel() * baselength
            theta = data2nd[:yw, :xw].ravel()
            dx = r * N.cos(theta)
            dy = r * N.sin(theta)

        if s.rotate != 0.:
            angle = -s.rotate / 180 * N.pi
            rotx = dx*N.cos(angle) - dy*N.sin(angle)
            roty = dx*N.sin(angle) + dy*N.cos(angle)
            dx, dy = rotx, roty

        if s.reflectx:
            dx = -dx
        if s.reflecty:
            dy = -dy

        x1, x2 = xplotter-dx, xplotter+dx
        y1, y2 = yplotter+dy, yplotter-dy

        if s.arrowfront == 'none' and s.arrowback == 'none':
            utils.plotLinesToPainter(painter, x1, y1, x2, y2,
                                     cliprect)
        else:
            arrowsize = s.get('arrowsize').convert(painter)
            painter.setBrush( s.get('Fill').makeQBrushWHide(painter) )

            # this is backward - have to convert from dx, dy to angle, length
            angles = 180 - N.arctan2(dy, dx) * (180./N.pi)
            lengths = N.sqrt(dx**2+dy**2) * 2
            
            # scale arrow heads by arrow length if requested
            if s.scalearrow:
                arrowsizes = (arrowsize/baselength/2) * lengths
            else:
                arrowsizes = N.zeros(lengths.shape) + arrowsize

            for x, y, l, a, asize in czip(x2, y2, lengths, angles, arrowsizes):
                if l != 0.:
                    utils.plotLineArrow(painter, x, y, l, a, asize,
                                        arrowleft=s.arrowfront,
                                        arrowright=s.arrowback)
                
# allow the factory to instantiate a vector field
document.thefactory.register( VectorField )
