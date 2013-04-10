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

'''An axis based on a function of another axis.'''

import numpy as N

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.document as document
import veusz.utils as utils

import axis
import axisuser

def _(text, disambiguation=None, context='FunctionAxis'):
    '''Translate text.'''
    return unicode( 
        qt4.QCoreApplication.translate(context, text, disambiguation))

class AxisFunction(axis.Axis, axisuser.AxisUser):
    '''An axis using an function of another axis.'''

    typename = 'axis-function'
    description = 'An axis based on a function of the values of another axis'

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        axis.Axis.addSettings(s)

        s.add( setting.Str('function', 't',
                           descr=_('Monatonic function (use t as variable)'),
                           usertext=_('Function')), 1 )
        s.add( setting.Axis('otheraxis', '', 'both',
                            descr =
                            _('Axis for which this axis is based on'),
                            usertext=_('Other axis')), 2 )

        s.get('min').hidden = True
        s.get('max').hidden = True

    def getAxesNames(self):
        '''Axes used by widget.'''
        return (self.settings.otheraxis,)

    def providesAxesDependency(self):
        return ((self.settings.otheraxis, 'both'),)

    def requiresAxesDependency(self):
        return (('both', self.settings.otheraxis),)

    def setAutoRange(self, autorange):
        print "AR",autorange
        axis.Axis.setAutoRange(self, autorange)

    def updateAxisRange(self, axis, depname, range):
        """Update range variable for axis with dependency name given."""
        print "uAR", axis, depname, range

# allow the factory to instantiate the widget
document.thefactory.register( AxisFunction )
