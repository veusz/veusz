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
import numpy as N

from ..compat import czip
from .. import qtall as qt4
from .. import setting
from .. import document
from .. import threed

from . import plotters3d

def _(text, disambiguation=None, context='Function3D'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class Function3D(plotters3d.GenericPlotter3D):
    """Plotting functions in 3D."""

    typename='function3d'
    description=_('Plot a 3D function')

    @classmethod
    def addSettings(klass, s):
        plotters3d.GenericPlotter3D.addSettings(s)

        s.add(setting.Int(
            'steps',
            50,
            minval = 3,
            descr = _('Number of steps to evaluate the function over'),
            usertext=_('Steps'),
            formatting=True ))
        s.add(setting.Choice(
            'mode',
            ['z=fn(x,y)', 'x=fn(y,z)', 'y=fn(x,z)',
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

    def affectsAxisRange(self):
        """Which axes this widget affects."""
        s = self.settings
        mode = s.mode
        if mode == 'z=fn(x,y)':
            return ((s.zAxis, 'sz'),)
        elif mode == 'x,y,z=fns(t)':
            return ((s.xAxis, 'sx'), (s.yAxis, 'sy'), (s.zAxis, 'sz'))
        # FIXME: more

    def requiresAxisRange(self):
        """Which axes this widget depends on."""
        s = self.settings
        mode = s.mode
        if mode == 'z=fn(x,y)':
            return ((s.xAxis, 'sx'), (s.yAxis, 'sy'))
        elif mode == 'x,y,z=fns(t)':
            return ()

    def getLineVals(self):
        """Get vals for line plot."""
        s = self.settings
        mode = s.mode

        err = None, None, None

        if mode == 'x,y,z=fns(t)':
            if not s.fnx or not s.fny or not s.fnz:
                return err

            xcomp = self.document.compileCheckedExpression(s.fnx)
            ycomp = self.document.compileCheckedExpression(s.fny)
            zcomp = self.document.compileCheckedExpression(s.fnz)
            if xcomp is None or ycomp is None or zcomp is None:
                return err

            env = self.document.eval_context.copy()
            env['t'] = N.linspace(0, 1, s.steps)
            zeros = N.zeros(s.steps, dtype=N.float64)
            try:
                valsx = eval(xcomp, env) + zeros
                valsy = eval(ycomp, env) + zeros
                valsz = eval(zcomp, env) + zeros
            except:
                # something wrong in the evaluation
                return err

        return valsx, valsy, valsz

    def getRange(self, axis, depname, axrange):
        mode = self.settings.mode
        if mode == 'x,y,z=fns(t)':
            valsx, valsy, valsz = self.getLineVals()
            if valsx is None:
                return
            coord = {'sx': valsx, 'sy': valsy, 'sz': valsz}[depname]
            finite = coord[N.isfinite(coord)]
            if len(finite) == 0:
                return
            axrange[0] = min(axrange[0], finite.min())
            axrange[1] = max(axrange[0], finite.max())

    def dataDrawToObject(self, axes):

        s = self.settings
        mode = s.mode
        outobj = []
        if mode == 'x,y,z=fns(t)':
            valsx, valsy, valsz = self.getLineVals()
            if valsx is None:
                return

            axes = self.fetchAxes()
            if axes is None:
                return

            lx = axes[0].dataToLogicalCoords(valsx)
            ly = axes[1].dataToLogicalCoords(valsy)
            lz = axes[2].dataToLogicalCoords(valsz)
            l1 = N.ones(len(lx))

            lineprop = threed.LineProp()
            pts = []
            last = None
            for coord in czip(lx, ly, lz, l1):
                if N.isfinite(sum(coord)):
                    pts.append(coord)
                else:
                    if pts:
                        outobj.append(threed.Polyline(pts, lineprop))
                        pts = []
            if pts:
                outobj.append(threed.Polyline(pts, lineprop))

        if len(outobj) == 0:
            return None
        elif len(outobj) == 1:
            return outobj[0]
        else:
            return threed.Compound(outobj)

document.thefactory.register(Function3D)
