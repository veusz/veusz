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

"""3D function plotting widget."""

from __future__ import division, print_function
import numpy as N

from .. import qtall as qt
from .. import setting
from .. import document
from .. import utils
from ..helpers import threed

from . import plotters3d

def _(text, disambiguation=None, context='Function3D'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class FunctionSurface(setting.Surface3DWColorMap):
    def __init__(self, *args, **argsv):
        setting.Surface3DWColorMap.__init__(self, *args, **argsv)
        self.get('color').newDefault(setting.Reference('../color'))

class FunctionLine(setting.Line3DWColorMap):
    def __init__(self, *args, **argsv):
        setting.Line3DWColorMap.__init__(self, *args, **argsv)
        self.get('color').newDefault(setting.Reference('../color'))
        self.get('reflectivity').newDefault(20)

class Function3D(plotters3d.GenericPlotter3D):
    """Plotting functions in 3D."""

    typename='function3d'
    description=_('3D function')
    allowusercreation=True

    # list of the supported modes
    _modes = [
        'x=fn(y,z)', 'y=fn(x,z)', 'z=fn(x,y)',
        'x,y,z=fns(t)',
        'x,y=fns(z)', 'y,z=fns(x)', 'x,z=fns(y)'
    ]

    # which axes are affected by which modes
    _affects = {
        'z=fn(x,y)': (('zAxis', 'both'),),
        'x=fn(y,z)': (('xAxis', 'both'),),
        'y=fn(x,z)': (('yAxis', 'both'),),
        'x,y,z=fns(t)': (('xAxis', 'sx'), ('yAxis', 'sy'), ('zAxis', 'sz')),
        'x,y=fns(z)': (('xAxis', 'both'), ('yAxis', 'both')),
        'y,z=fns(x)': (('yAxis', 'both'), ('zAxis', 'both')),
        'x,z=fns(y)': (('xAxis', 'both'), ('zAxis', 'both')),
    }

    # which modes require which axes as inputs
    _requires = {
        'z=fn(x,y)': (('both', 'xAxis'), ('both', 'yAxis')),
        'x=fn(y,z)': (('both', 'yAxis'), ('both', 'zAxis')),
        'y=fn(x,z)': (('both', 'xAxis'), ('both', 'zAxis')),
        'x,y,z=fns(t)': (),
        'x,y=fns(z)': (('both', 'zAxis'),),
        'y,z=fns(x)': (('both', 'xAxis'),),
        'x,z=fns(y)': (('both', 'yAxis'),),
    }

    # which modes require which variables
    _varmap = {
        'x,y=fns(z)': ('x', 'y', 'z'),
        'y,z=fns(x)': ('y', 'z', 'x'),
        'x,z=fns(y)': ('x', 'z', 'y'),
    }

    @staticmethod
    def _fnsetnshowhide(v):
        """Return which function settings to show or hide depending on
        mode."""
        return {
            'z=fn(x,y)': (('fnz',), ('fnx', 'fny')),
            'x=fn(y,z)': (('fnx',), ('fny', 'fnz')),
            'y=fn(x,z)': (('fny',), ('fnx', 'fnz')),
            'x,y,z=fns(t)': (('fnx', 'fny', 'fnz'), ()),
            'x,y=fns(z)': (('fnx', 'fny'), ('fnz',)),
            'y,z=fns(x)': (('fny', 'fnz'), ('fnx',)),
            'x,z=fns(y)': (('fnx', 'fnz'), ('fny',)),
            }[v]

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
        s.add(setting.ChoiceSwitch(
            'mode', klass._modes,
            'x,y,z=fns(t)',
            descr=_('Type of function to plot'),
            usertext=_('Mode'),
            showfn=klass._fnsetnshowhide), 0)

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
        s.add(setting.Str(
            'fncolor', '',
            descr=_('Function to give color (0-1)'),
            usertext=_('Color function') ), 4)

        s.add( setting.Color(
            'color',
            'auto',
            descr = _('Master color'),
            usertext = _('Color'),
            formatting=True), 0 )
        s.add(FunctionLine(
            'Line',
            descr = _('Line settings'),
            usertext = _('Plot line')),
               pixmap = 'settings_plotline' )
        s.add(setting.LineGrid3D(
            'GridLine',
            descr = _('Grid line settings'),
            usertext = _('Grid line')),
               pixmap = 'settings_gridline' )
        s.add(FunctionSurface(
            'Surface',
            descr = _('Surface fill settings'),
            usertext=_('Surface')),
              pixmap='settings_bgfill' )

    def affectsAxisRange(self):
        """Which axes this widget affects."""
        s = self.settings
        affects = self._affects[s.mode]
        return [(getattr(s, v[0]), v[1]) for v in affects]

    def requiresAxisRange(self):
        """Which axes this widget depends on."""
        s = self.settings
        requires = self._requires[s.mode]
        return [(v[0], getattr(s, v[1])) for v in requires]

    def getLineVals(self):
        """Get vals for line plot by evaluating function."""
        s = self.settings
        mode = s.mode

        if mode == 'x,y,z=fns(t)':
            if not s.fnx or not s.fny or not s.fnz:
                return None

            xcomp = self.document.evaluate.compileCheckedExpression(s.fnx)
            ycomp = self.document.evaluate.compileCheckedExpression(s.fny)
            zcomp = self.document.evaluate.compileCheckedExpression(s.fnz)
            if xcomp is None or ycomp is None or zcomp is None:
                return None

            # evaluate each expression
            env = self.document.evaluate.context.copy()
            env['t'] = N.linspace(0, 1, s.linesteps)
            zeros = N.zeros(s.linesteps, dtype=N.float64)
            try:
                valsx = eval(xcomp, env) + zeros
                valsy = eval(ycomp, env) + zeros
                valsz = eval(zcomp, env) + zeros
            except:
                # something wrong in the evaluation
                return None

            fncolor = s.fncolor.strip()
            if fncolor:
                fncolor = self.document.evaluate.compileCheckedExpression(
                    fncolor)
                try:
                    valscolor = eval(fncolor, env) + zeros
                except:
                    return None
            else:
                valscolor = None

            retn = (valsx, valsy, valsz, valscolor)

        else:
            # lookup variables to go with function
            var = self._varmap[mode]
            fns = [getattr(s, 'fn'+var[0]), getattr(s, 'fn'+var[1])]
            if not fns[0] or not fns[1]:
                return None

            # get points to evaluate functions over
            axis = self.fetchAxis(var[2])
            if not axis:
                return
            arange = axis.getPlottedRange()
            if axis.settings.log:
                evalpts = N.logspace(
                    N.log10(arange[0]), N.log10(arange[1]), s.linesteps)
            else:
                evalpts = N.linspace(arange[0], arange[1], s.linesteps)

            # evaluate expressions
            env = self.document.evaluate.context.copy()
            env[var[2]] = evalpts
            zeros = N.zeros(s.linesteps, dtype=N.float64)
            try:
                vals1 = eval(fns[0], env) + zeros
                vals2 = eval(fns[1], env) + zeros
            except:
                # something wrong in the evaluation
                return None

            fncolor = s.fncolor.strip()
            if fncolor:
                fncolor = self.document.evaluate.compileCheckedExpression(
                    fncolor)
                try:
                    valscolor = eval(fncolor, env) + zeros
                except:
                    return None
            else:
                valscolor = None

            # assign correct output points
            retn = [None]*4
            idxs = ('x', 'y', 'z')
            retn[idxs.index(var[0])] = vals1
            retn[idxs.index(var[1])] = vals2
            retn[idxs.index(var[2])] = evalpts
            retn[3] = valscolor

        return retn

    def getGridVals(self):
        """Get values for 2D grid.

        Return steps1, steps2, height, axidx, depvariable
        axidx are the indices into the axes for height, step1, step2
        """

        s = self.settings
        mode = s.mode

        var, ovar1, ovar2, axidx = {
            'x=fn(y,z)': ('x', 'y', 'z', (0, 1, 2)),
            'y=fn(x,z)': ('y', 'z', 'x', (1, 2, 0)),
            'z=fn(x,y)': ('z', 'x', 'y', (2, 0, 1)),
        }[mode]

        axes = self.fetchAxes()
        if axes is None:
            return None

        # range of other axes
        ax1, ax2 = axes[axidx[1]], axes[axidx[2]]
        pr1 = ax1.getPlottedRange()
        pr2 = ax2.getPlottedRange()
        steps = s.surfacesteps
        logax1, logax2 = ax1.settings.log, ax2.settings.log

        # convert log ranges to linear temporarily
        if logax1:
            pr1 = N.log(pr1)
        if logax2:
            pr2 = N.log(pr2)

        # set variables in environment
        grid1, grid2 = N.indices((steps, steps))
        del1 = (pr1[1]-pr1[0])/(steps-1.)
        steps1 = N.arange(steps)*del1 + pr1[0]
        grid1 = grid1*del1 + pr1[0]
        del2 = (pr2[1]-pr2[0])/(steps-1.)
        steps2 = N.arange(steps)*del2 + pr2[0]
        grid2 = grid2*del2 + pr2[0]

        fncolor = s.fncolor.strip()
        if fncolor:
            colgrid1 = 0.5*(grid1[1:,1:]+grid1[:-1,:-1])
            colgrid2 = 0.5*(grid2[1:,1:]+grid2[:-1,:-1])
            if logax1:
                colgrid1 = N.exp(colgrid1)
            if logax2:
                colgrid2 = N.exp(colgrid2)

        # convert back to log
        if logax1:
            grid1 = N.exp(grid1)
        if logax2:
            grid2 = N.exp(grid2)

        env = self.document.evaluate.context.copy()
        env[ovar1] = grid1
        env[ovar2] = grid2

        fn = getattr(s, 'fn%s' % var)  # get function from user
        if not fn:
            return
        comp = self.document.evaluate.compileCheckedExpression(fn)
        if comp is None:
            return None

        try:
            height = eval(comp, env) + N.zeros(grid1.shape, dtype=N.float64)
        except Exception:
            # something wrong in the evaluation
            return None

        if fncolor:
            compcolor = self.document.evaluate.compileCheckedExpression(
                fncolor)
            if not compcolor:
                return
            env[ovar1] = colgrid1
            env[ovar2] = colgrid2

            try:
                colors = eval(compcolor, env) + N.zeros(
                    colgrid1.shape, dtype=N.float64)
            except Exception:
                # something wrong in the evaluation
                return None
            colors = N.clip(colors, 0, 1)
        else:
            colors = None

        return height, steps1, steps2, axidx, var, colors

    def getRange(self, axis, depname, axrange):
        """Get range of axis."""
        mode = self.settings.mode
        if mode == 'x,y,z=fns(t)':
            # get range of each variable
            retn = self.getLineVals()
            if not retn:
                return
            valsx, valsy, valsz, valscolor = retn
            coord = {'sx': valsx, 'sy': valsy, 'sz': valsz}[depname]

        elif mode in ('x,y=fns(z)', 'y,z=fns(x)', 'x,z=fns(y)'):
            # is this axis one of the ones we affect?
            var = self._varmap[mode]
            if self.fetchAxis(var[0]) is axis:
                v = var[0]
            elif self.fetchAxis(var[1]) is axis:
                v = var[1]
            else:
                return

            retn = self.getLineVals()
            if not retn:
                return
            coord = retn[('x', 'y', 'z').index(v)]

        elif mode in ('z=fn(x,y)', 'x=fn(y,z)', 'y=fn(x,z)'):
            retn = self.getGridVals()
            if not retn:
                return
            height, steps1, steps2, axidx, var, color = retn
            if axis is not self.fetchAxis(var):
                return
            coord = height

        finite = coord[N.isfinite(coord)]
        if len(finite) > 0:
            axrange[0] = min(axrange[0], finite.min())
            axrange[1] = max(axrange[1], finite.max())

    def updatePropColorMap(self, prop, setn, colorvals):
        """Update line/surface properties given color map values.

        prop is updated to use the data values colorvars (0-1) to apply
        a color map from the setting setn given."""

        cmap = self.document.evaluate.getColormap(
            setn.colorMap, setn.colorMapInvert)
        color2d = colorvals.reshape((1, colorvals.size))
        colorimg = utils.applyColorMap(
            cmap, 'linear', color2d, 0., 1., setn.transparency)
        prop.setRGBs(colorimg)

    def dataDrawSurface(self, painter, axes, container):
        """Draw a surface plot."""
        retn = self.getGridVals()
        if not retn:
            return
        height, steps1, steps2, axidx, depvar, colors = retn
        lheight = axes[axidx[0]].dataToLogicalCoords(height)
        lsteps1 = axes[axidx[1]].dataToLogicalCoords(steps1)
        lsteps2 = axes[axidx[2]].dataToLogicalCoords(steps2)

        # draw grid over each axis
        surfprop = lineprop = None
        s = self.settings
        if not s.Surface.hide:
            surfprop = s.Surface.makeSurfaceProp(painter)
            if colors is not None:
                self.updatePropColorMap(surfprop, s.Surface, colors)

        if not s.GridLine.hide:
            lineprop = s.GridLine.makeLineProp(painter)

        dirn = {'x': threed.Mesh.X_DIRN,
                'y': threed.Mesh.Y_DIRN,
                'z': threed.Mesh.Z_DIRN}[depvar]

        mesh = threed.Mesh(
            threed.ValVector(lsteps1), threed.ValVector(lsteps2),
            threed.ValVector(N.ravel(lheight)),
            dirn, lineprop, surfprop,
            s.GridLine.hidehorz, s.GridLine.hidevert)
        container.addObject(mesh)

    def dataDrawLine(self, painter, axes, clipcontainer):
        """Draw a line function."""

        s = self.settings
        if s.Line.hide:
            return

        retn = self.getLineVals()
        if not retn:
            return

        valsx, valsy, valsz, valscolor = retn
        lineprop = s.Line.makeLineProp(painter)
        if valscolor is not None:
            self.updatePropColorMap(lineprop, s.Line, valscolor)

        lx = axes[0].dataToLogicalCoords(valsx)
        ly = axes[1].dataToLogicalCoords(valsy)
        lz = axes[2].dataToLogicalCoords(valsz)

        line = threed.PolyLine(lineprop)
        line.addPoints(
            threed.ValVector(lx), threed.ValVector(ly),
            threed.ValVector(lz))

        clipcontainer.addObject(line)

    def dataDrawToObject(self, painter, axes):
        """Do actual drawing of function."""

        s = self.settings
        mode = s.mode

        axes = self.fetchAxes()
        if axes is None:
            return

        s = self.settings

        clipcontainer = self.makeClipContainer(axes)
        if mode in ('x,y,z=fns(t)', 'x,y=fns(z)', 'y,z=fns(x)', 'x,z=fns(y)'):
            self.dataDrawLine(painter, axes, clipcontainer)
        elif mode in ('z=fn(x,y)', 'x=fn(y,z)', 'y=fn(x,z)'):
            self.dataDrawSurface(painter, axes, clipcontainer)

        clipcontainer.assignWidgetId(id(self))
        return clipcontainer

document.thefactory.register(Function3D)
