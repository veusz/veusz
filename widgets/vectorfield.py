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

import itertools
import numpy as N

import veusz.setting as setting
import veusz.document as document
import veusz.utils as utils
import veusz.qtall as qt4

import plotters

class VectorField(plotters.GenericPlotter):
    '''A plotter for plotting a vector field.'''

    typename = 'vectorfield'
    allowusercreation = True
    description = 'Plot a vector field'

    def __init__(self, parent, name=None):
        """Initialse vector field plotter."""

        plotters.GenericPlotter.__init__(self, parent, name=name)

        if type(self) == VectorField:
            self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        plotters.GenericPlotter.addSettings(s)

        # datasets
        s.add( setting.Dataset('data1', '',
                               dimensions = 2,
                               descr = 'X coordinate length or vector magnitude',
                               usertext = u'δx or r'),
               0 )
        s.add( setting.Dataset('data2', '',
                               dimensions = 2,
                               descr = 'Y coordinate length or vector angle',
                               usertext = u'δy or θ'),
               1 )
        s.add( setting.Choice('mode',
                              ['cartesian', 'polar'],
                              'cartesian',
                              descr = u'Cartesian (δx,δy) or polar (r,θ)',
                              usertext = 'Mode'),
               2 )

        # formatting
        s.add( setting.DistancePt('baselength', '10pt',
                                  descr = "Base length of unit vector",
                                  usertext = "Base length",
                                  formatting=True),
               0 )
        s.add( setting.DistancePt('arrowsize', '2pt',
                                  descr = "Size of any arrows",
                                  usertext = "Arrow size",
                                  formatting=True),
               1 )
        s.add( setting.Bool('scalearrow', True,
                            descr = 'Scale arrow head by length',
                            usertext = 'Scale arrow',
                            formatting=True),
               2 )
        s.add( setting.Arrow('arrowfront', 'none',
                             descr = 'Arrow in front direction',
                             usertext='Arrow front', formatting=True),
               3)
        s.add( setting.Arrow('arrowback', 'none',
                             descr = 'Arrow in back direction',
                             usertext='Arrow back', formatting=True),
               4)

        s.add( setting.Line('Line',
                            descr = 'Line style',
                            usertext = 'Line'),
               pixmap = 'settings_plotline' )
        s.add( setting.ArrowFill('Fill',
                                 descr = 'Arrow fill settings',
                                 usertext = 'Arrow fill'),
               pixmap = 'settings_plotmarkerfill' )

    def dataHasChanged(self):
        s = self.settings
        return self.dsmonitor.hasChanged(s.get('data1'), s.get('data2'))

    def providesAxesDependency(self):
        """Range information provided by widget."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )
        
    def updateAxisRange(self, axis, depname, axrange):
        """Automatically determine the ranges of variable on the axes."""

        for name in (self.settings.data1, self.settings.data2):
            try:
                data = self.document.data[name]
            except KeyError:
                continue

            if data.dimensions == 2:
                if depname == 'sx':
                    dxrange = data.xrange
                    axrange[0] = min( axrange[0], dxrange[0] )
                    axrange[1] = max( axrange[1], dxrange[1] )
                elif depname == 'sy':
                    dyrange = data.yrange
                    axrange[0] = min( axrange[0], dyrange[0] )
                    axrange[1] = max( axrange[1], dyrange[1] )

    def draw(self, parentposn, phelper, outerbounds = None):
        """Draw the widget."""

        posn = plotters.GenericPlotter.draw(self, parentposn, phelper,
                                            outerbounds = outerbounds)
        x1, y1, x2, y2 = posn
        s = self.settings
        d = self.document

        # hide if hidden!
        if s.hide:
            return

        # get axes widgets
        axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

        # return if there's no proper axes
        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return

        # ignore non existing datasets
        try:
            data1 = d.data[s.data1]
            data2 = d.data[s.data2]
        except KeyError:
            return

        # require 2d datasets
        if data1.dimensions != 2 or data2.dimensions != 2:
            return

        # clip data within bounds of plotter
        cliprect = self.clipAxesBounds(axes, posn)
        painter = phelper.painter(self, posn, clip=cliprect)

        baselength = s.get('baselength').convert(painter)

        # try to be nice if the datasets don't match
        data1st, data2nd = data1.data, data2.data
        xw = min(data1st.shape[1], data2nd.shape[1])
        yw = min(data1st.shape[0], data2nd.shape[0])

        # construct indices into datasets
        yvals, xvals = N.mgrid[0:yw, 0:xw]
        # convert using 1st dataset to axes values
        xdsvals, ydsvals = data1.indexToPoint(xvals.ravel(), yvals.ravel())

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

        x1, x2 = xplotter-dx, xplotter+dx
        y1, y2 = yplotter+dy, yplotter-dy

        if s.arrowfront == 'none' and s.arrowback == 'none':
            utils.plotLinesToPainter(painter, x1, y1, x2, y2,
                                     cliprect)
        else:
            arrowsize = s.get('arrowsize').convert(painter)
            painter.setBrush( s.get('Fill').makeQBrushWHide() )

            # this is backward - have to convert from dx, dy to angle, length
            angles = 180 - N.arctan2(dy, dx) * (180./N.pi)
            lengths = N.sqrt(dx**2+dy**2) * 2
            
            # scale arrow heads by arrow length if requested
            if s.scalearrow:
                arrowsizes = (arrowsize/baselength/2) * lengths
            else:
                arrowsizes = N.zeros(lengths.shape) + arrowsize

            for x, y, l, a, asize in itertools.izip(x2, y2, lengths, angles,
                                                    arrowsizes):
                utils.plotLineArrow(painter, x, y, l, a, asize,
                                    arrowleft=s.arrowfront,
                                    arrowright=s.arrowback)
                
# allow the factory to instantiate a vector field
document.thefactory.register( VectorField )

