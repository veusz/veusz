#    Copyright (C) 2014 Jeremy S. Sanders
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

"""Generic 3D plotting widet."""

from __future__ import division
import itertools
import numpy as N

from ..compat import czip, crange
from .. import qtall as qt4
from .. import setting
from .. import document
from .. import threed

from . import plotters3d

def _(text, disambiguation=None, context='Function3D'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

def constructPolyline(outobj, lineprop, lx, ly, lz):
    """Construct polyline object from coordinates.

    Split up line into finite sections
    """

    stack = N.column_stack(( lx, ly, lz, N.ones(len(lx)) ))
    notfinite = N.logical_not(N.isfinite(lx+ly+lz))
    badidxs = list(N.where(notfinite)[0])
    if badidxs:
        last = 0
        for i in badidxs + [len(lx)]:
            if i-last > 2:
                outobj.append(threed.Polyline(stack[last:i], lineprop))
            last = i
    else:
        outobj.append(threed.Polyline(stack, lineprop))

def constructSurface(outobj, surfprop, lx, ly, lz):
    """Split up gridded surface into triangles."""
    w, h = lx.shape
    for i in crange(w-1):
        for j in crange(h-1):
            p0 = (lx[i,j], ly[i,j], lz[i,j], 1)
            p1 = (lx[i+1,j], ly[i+1,j], lz[i+1,j], 1)
            p2 = (lx[i,j+1], ly[i,j+1], lz[i,j+1], 1)
            p3 = (lx[i+1,j+1], ly[i+1,j+1], lz[i+1,j+1], 1)
            outobj.append(threed.Triangle((p0,p1,p2), surfprop))
            outobj.append(threed.Triangle((p3,p1,p2), surfprop))

class Function3D(plotters3d.GenericPlotter3D):
    """Plotting functions in 3D."""

    typename='function3d'
    description=_('Plot a 3D function')

    @classmethod
    def addSettings(klass, s):
        plotters3d.GenericPlotter3D.addSettings(s)

        s.add(setting.Int(
            'linesteps',
            50,
            minval = 3,
            descr = _('Number of steps to evaluate the function over for lines'),
            usertext=_('Line steps'),
            formatting=True ))
        s.add(setting.Int(
            'surfacesteps',
            20,
            minval = 3,
            descr = _('Number of steps to evaluate the function over for surfaces'
                      ' in each direction'),
            usertext=_('Surface steps'),
            formatting=True ))
        s.add(setting.Choice(
            'mode',
            ['x=fn(y,z)', 'y=fn(x,z)', 'z=fn(x,y)',
             'x,y,z=fns(t)', 'x,y=fns(z)', 'y,z=fns(x)', 'x,z=fns(y)'],
            'x,y,z=fns(t)',
            descr=_('Type of function to plot'),
            usertext=_('Mode') ), 0)

        s.add(setting.Str(
            'fnx', '',
            descr=_('Function for x coordinate'),
            usertext=_('X function') ), 1)
        s.add(setting.Str(
            'fny', '',
            descr=_('Function for y coordinate'),
            usertext=_('Y function') ), 2)
        s.add(setting.Str(
            'fnz', '',
            descr=_('Function for z coordinate'),
            usertext=_('Z function') ), 3)

        s.add(setting.Line3D(
            'Line',
            descr = _('Line settings'),
            usertext = _('Plot line')),
               pixmap = 'settings_plotline' )
        s.add(setting.Surface3D(
            'Surface',
            descr = _('Surface fill settings'),
            usertext=_('Surface')),
              pixmap='settings_bgfill' )

    def affectsAxisRange(self):
        """Which axes this widget affects."""
        s = self.settings
        mode = s.mode
        if mode == 'z=fn(x,y)':
            return ((s.zAxis, 'sz'),)
        elif mode == 'x=fn(y,z)':
            return ((s.xAxis, 'sx'),)
        elif mode == 'y=fn(x,z)':
            return ((s.yAxis, 'sy'),)
        elif mode == 'x,y,z=fns(t)':
            return ((s.xAxis, 'sx'), (s.yAxis, 'sy'), (s.zAxis, 'sz'))

        # FIXME: more

    def requiresAxisRange(self):
        """Which axes this widget depends on."""
        s = self.settings
        mode = s.mode
        if mode == 'z=fn(x,y)':
            return (('sx', s.xAxis), ('sy', s.yAxis))
        elif mode == 'x=fn(y,z)':
            return (('sy', s.yAxis), ('sz', s.zAxis))
        elif mode == 'y=fn(x,z)':
            return (('sx', s.xAxis), ('sz', s.zAxis))
        elif mode == 'x,y,z=fns(t)':
            return ()

    def getLineVals(self):
        """Get vals for line plot."""
        s = self.settings
        mode = s.mode

        if mode == 'x,y,z=fns(t)':
            if not s.fnx or not s.fny or not s.fnz:
                return None

            xcomp = self.document.compileCheckedExpression(s.fnx)
            ycomp = self.document.compileCheckedExpression(s.fny)
            zcomp = self.document.compileCheckedExpression(s.fnz)
            if xcomp is None or ycomp is None or zcomp is None:
                return None

            env = self.document.eval_context.copy()
            env['t'] = N.linspace(0, 1, s.linesteps)
            zeros = N.zeros(s.linesteps, dtype=N.float64)
            try:
                valsx = eval(xcomp, env) + zeros
                valsy = eval(ycomp, env) + zeros
                valsz = eval(zcomp, env) + zeros
            except:
                # something wrong in the evaluation
                return None

        return valsx, valsy, valsz

    def getGridVals(self):
        """Get values for 2D grid.

        Return xgrid, ygrid, zgrid, depvariable
        """

        s = self.settings
        mode = s.mode

        var, ovar1, ovar2, ax1, ax2 = {
            'z=fn(x,y)': ('z', 'x', 'y', 0, 1),
            'x=fn(y,z)': ('x', 'y', 'z', 1, 2),
            'y=fn(x,z)': ('y', 'x', 'z', 0, 2),
        }[mode]
        axes = self.fetchAxes()

        # range of other axes
        pr1 = axes[ax1].getPlottedRange()
        pr2 = axes[ax2].getPlottedRange()
        steps = s.surfacesteps

        # set variables in environment
        grid1, grid2 = N.indices((steps, steps))
        grid1 = grid1 * ((pr1[1]-pr1[0])/(steps-1.)) + pr1[0]
        grid2 = grid2 * ((pr2[1]-pr2[0])/(steps-1.)) + pr2[0]
        env = self.document.eval_context.copy()
        env[ovar1] = grid1
        env[ovar2] = grid2

        comp = self.document.compileCheckedExpression(
            getattr(s, 'fn%s' % var))
        if comp is None:
            return None

        try:
            height = eval(comp, env) + N.zeros((steps, steps), dtype=N.float64)
        except:
            # something wrong in the evaluation
            return None

        if var == 'x':
            return height, grid1, grid2, var
        elif var == 'y':
            return grid1, height, grid2, var
        else:
            return grid1, grid2, height, var

    def getRange(self, axis, depname, axrange):
        mode = self.settings.mode
        if mode == 'x,y,z=fns(t)':
            retn = self.getLineVals()
            if not retn:
                return
            valsx, valsy, valsz = retn

            coord = {'sx': valsx, 'sy': valsy, 'sz': valsz}[depname]
            finite = coord[N.isfinite(coord)]
            if len(finite) > 0:
                axrange[0] = min(axrange[0], finite.min())
                axrange[1] = max(axrange[1], finite.max())

        elif mode in ('z=fn(x,y)', 'x=fn(y,z)', 'y=fn(x,z)'):
            retn = self.getGridVals()
            if not retn:
                return
            xgrid, ygrid, zgrid, var = retn

            v = {'x': xgrid, 'y': ygrid, 'z': zgrid}[var].ravel()
            finite = v[N.isfinite(v)]
            if len(finite) > 0:
                axrange[0] = min(axrange[0], finite.min())
                axrange[1] = max(axrange[1], finite.max())

    def dataDrawToObject(self, axes):

        s = self.settings
        mode = s.mode

        axes = self.fetchAxes()
        if axes is None:
            return

        lineprop = s.Line.makeLineProp()

        outobj = []
        if mode == 'x,y,z=fns(t)':
            retn = self.getLineVals()
            if not retn:
                return
            valsx, valsy, valsz = retn
            lx = axes[0].dataToLogicalCoords(valsx)
            ly = axes[1].dataToLogicalCoords(valsy)
            lz = axes[2].dataToLogicalCoords(valsz)

            if not s.Line.hide:
                constructPolyline(outobj, lineprop, lx, ly, lz)

        elif mode in ('z=fn(x,y)', 'x=fn(y,z)', 'y=fn(x,z)'):
            retn = self.getGridVals()
            if not retn:
                return
            valsx, valsy, valsz, var = retn
            lx = axes[0].dataToLogicalCoords(valsx)
            ly = axes[1].dataToLogicalCoords(valsy)
            lz = axes[2].dataToLogicalCoords(valsz)

            # draw grid over each axis
            if not s.Surface.hide:
                surfprop = s.Surface.makeSurfaceProp()
                constructSurface(outobj, surfprop, lx, ly, lz)

            if not s.Line.hide:
                for i in crange(lx.shape[0]):
                    constructPolyline(
                        outobj, lineprop, lx[i, :], ly[i, :], lz[i, :])
                for i in crange(lx.shape[1]):
                    constructPolyline(
                        outobj, lineprop, lx[:, i], ly[:, i], lz[:, i])

        if len(outobj) == 0:
            return None
        elif len(outobj) == 1:
            return outobj[0]
        else:
            return threed.Compound(outobj)

document.thefactory.register(Function3D)
