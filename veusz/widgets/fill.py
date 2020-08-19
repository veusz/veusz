#    Copyright (C) 2020 Jeremy S. Sanders
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

"""For filling regions between xy points."""

from __future__ import division
import numpy as N

from ..compat import czip
from .. import qtall as qt
from .. import datasets
from .. import document
from .. import setting
from .. import utils

from .plotters import GenericPlotter

def _(text, disambiguation=None, context='Fill'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

def _mode_show_hide(val):
    return {
        'data-data': (
            ('datax1', 'datay1', 'datax2', 'datay2'),
            ('func1', 'func2', 'funcvar'),
        ),
        'data-func': (
            ('datax1', 'datay1', 'func1', 'funcvar'),
            ('datax2', 'datay2', 'func2'),
        ),
        'func-func': (
            ('func1', 'func2', 'funcvar'),
            ('datax1', 'datay1', 'datax2', 'datay2'),
        ),
    }[val]

class Fill(GenericPlotter):
    """A class for filling between points and functions."""

    typename='fill'
    allowusercreation=True
    description=_('Fill a region between two datasets')

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        GenericPlotter.addSettings(s)

        s.add( setting.ChoiceSwitch(
            'mode',
            ('data-data', 'data-func', 'func-func'),
            'data-data',
            showfn=_mode_show_hide,
            descr=_('Mode'),
            usertext=_('Mode'),
            )
        )

        s.add( setting.DatasetExtended(
            'datax1', 'x1',
            descr=_('X dataset 1'),
            usertext=_('X data 1')), 0 )
        s.add( setting.DatasetExtended(
            'datay1', 'y1',
            descr=_('Y dataset 1'),
            usertext=_('Y data 1')), 0 )
        s.add( setting.DatasetExtended(
            'datax2', 'x2',
            descr=_('X dataset 2'),
            usertext=_('X data 2')), 0 )
        s.add( setting.DatasetExtended(
            'datay2', 'y2',
            descr=_('Y dataset 2'),
            usertext=_('Y data 2')), 0 )
        s.add( setting.Str(
            'func1', 'x',
            descr=_('Function expression 1'),
            usertext=_('Function 1')), 0 )
        s.add( setting.Str(
            'func2', 'x**2',
            descr=_('Function expression 2'),
            usertext=_('Function 2')), 0 )
        s.add( setting.Choice(
            'funcvar', ['x', 'y'], 'x',
            descr=_('Variable the function is a function of'),
            usertext=_('Variable')),
               0 )

        s.add( setting.Choice(
            'fillmode',
            ('horizontal', 'vertical', 'polygon'),
            'vertical',
            descr=_('Fill mode'),
            usertext=_('Fill mode')),
               0 )

        s.add( setting.XYPlotLine(
            'EdgeLine',
            descr = _('Edge line settings'),
            usertext = _('Edge line')),
               pixmap = 'settings_plotline' )

    @property
    def userdescription(self):
        """User-friendly description."""
        return "%(datax1)s,%(datay1)s to %(datax2)s,%(datay2)s" % (
            self.settings)

    def affectsAxisRange(self):
        s = self.settings
        mode = s.mode
        if mode == 'data-data' or mode == 'data-func':
            return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )
        else:  # func-func
            if s.funcvar == 'x':
                return ((s.yAxis, 'sy'),)
            else:
                return ((s.xAxis, 'sx'),)

    def requiresAxisRange(self):
        s = self.settings
        mode = s.mode
        if mode == 'func-func':
            if s.variable == 'x':
                return (('sx', s.xAxis),)
            else:
                return (('sy', s.yAxis),)
        else:
            return ()

    def getRange(self, axis, depname, axrange):
        """Adjust the range of the axis depending on the values plotted."""
        s = self.settings

        mode = s.mode

        if depname == 'sx':
            if mode in ('data-data', 'data-func'):
                data1 = s.get('datax1').getData(self.document)
            if mode == 'data-data':
                data2 = s.get('datax2').getData(self.document)

        else:
            if mode in ('data-data', 'data-func'):
                data1 = s.get('datay1').getData(self.document)
            if mode == 'data-data':
                data2 = s.get('datay2').getData(self.document)

        if data1:
            data1.updateRangeAuto(
                axrange, log=axis.settings.log, errors=False)
        if data2:
            data2.updateRangeAuto(
                axrange, log=axis.settings.log, errors=False)

    def dataDraw(self, painter, axes, posn, cliprect):
        """Plot the data on a plotter."""

        s = self.settings
        doc = self.document

        datax1 = s.get('datax1').getData(self.document)
        datay1 = s.get('datay1').getData(self.document)
        datax2 = s.get('datax2').getData(self.document)
        datay2 = s.get('datay2').getData(self.document)

        datax1 = s.datax1.getData(doc)
        datax2 = s.datax2.getData(doc)
        datay1 = s.datay1.getData(doc)
        datay2 = s.datay2.getData(doc)

        if datax1 is None or datax2 is None or datax3 is None or datax4 is None:
            return

        print('here')

document.thefactory.register(Fill)
