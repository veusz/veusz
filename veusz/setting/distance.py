#    Copyright (C) 2016 Jeremy S. Sanders
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

import re

from .settingbase import Setting
from .settingdb import uilocale, ui_floattostring, ui_stringtofloat
from . import controls

from .. import utils

# these are functions used by the distance setting below.
# they don't work as class methods

def _distPhys(match, painter, mult):
    """Convert a physical unit measure in multiples of points."""
    return (painter.pixperpt * mult *
            float(match.group(1)) * painter.scaling)

def _idistval(val, unit):
    """Convert value to text, dropping zeros and . points on right."""
    return ("%.3f" % val).rstrip('0').rstrip('.') + unit

def _distInvPhys(pixdist, painter, mult, unit):
    """Convert number of pixels into physical distance."""
    return _idistval( pixdist / (mult * painter.pixperpt * painter.scaling),
                      unit )

def _distPerc(match, painter):
    """Convert from a percentage of maxsize."""
    return painter.maxsize * 0.01 * float(match.group(1))

def _distInvPerc(pixdist, painter):
    """Convert pixel distance into percentage."""
    return _idistval(pixdist * 100. / painter.maxsize, '%')

def _distFrac(match, painter):
    """Convert from a fraction a/b of maxsize."""
    try:
        return painter.maxsize * float(match.group(1))/float(match.group(4))
    except ZeroDivisionError:
        return 0.

def _distRatio(match, painter):
    """Convert from a simple 0.xx ratio of maxsize."""

    # if it's greater than 1 then assume it's a point measurement
    if float(match.group(1)) > 1.:
        return _distPhys(match, painter, 1)

    return painter.maxsize * float(match.group(1))

# regular expression to match distances
distre_expr = r'''^
 [ ]*                                # optional whitespace

 (\.?[0-9]+|[0-9]+\.[0-9]*)          # a floating point number

 [ ]*                                # whitespace

 (cm|pt|mm|inch|in|"|%||             # ( unit, no unit,
  (?P<slash>/) )                     # or / )

 (?(slash)[ ]*                       # if it was a slash, match any whitespace
  (\.?[0-9]+|[0-9]+\.[0-9]*))        # and match following fp number

 [ ]*                                # optional whitespace
$'''

class Distance(Setting):
    """A veusz distance measure, e.g. 1pt or 3%."""

    typename = 'distance'

    # match a distance
    distre = re.compile(distre_expr, re.VERBOSE)

    # functions to convert from unit values to points
    unit_func = {
        'cm': lambda match, painter:
            _distPhys(match, painter, 720/25.4),
        'pt': lambda match, painter:
            _distPhys(match, painter, 1.),
        'mm': lambda match, painter:
            _distPhys(match, painter, 72/25.4),
        'in': lambda match, painter:
            _distPhys(match, painter, 72.),
        'inch': lambda match, painter:
            _distPhys(match, painter, 72.),
        '"': lambda match, painter:
            _distPhys(match, painter, 72.),
        '%': _distPerc,
        '/': _distFrac,
        '': _distRatio
        }

    # inverse functions for converting points to units
    inv_unit_func = {
        'cm': lambda match, painter:
            _distInvPhys(match, painter, 720/25.4, 'cm'),
        'pt': lambda match, painter:
            _distInvPhys(match, painter, 1., 'pt'),
        'mm': lambda match, painter:
            _distInvPhys(match, painter, 72/25.4, 'mm'),
        'in': lambda match, painter:
            _distInvPhys(match, painter, 72., 'in'),
        'inch': lambda match, painter:
            _distInvPhys(match, painter, 72., 'in'),
        '"': lambda match, painter:
            _distInvPhys(match, painter, 72., 'in'),
        '%': _distInvPerc,
        '/': _distInvPerc,
        '': _distInvPerc
        }

    @classmethod
    def isDist(kls, dist):
        """Is the text a valid distance measure?"""

        return kls.distre.match(dist) is not None

    def convertTo(self, val):
        if self.distre.match(val) is not None:
            return val
        else:
            raise utils.InvalidType

    def toTextUI(self):
        # convert decimal point to display locale
        return self.val.replace('.', uilocale.decimalPoint())

    def fromTextUI(self, text):
        # convert decimal point from display locale
        text = text.replace(uilocale.decimalPoint(), '.')

        if self.isDist(text):
            return text
        else:
            raise utils.InvalidType

    def makeControl(self, *args):
        return controls.Distance(self, *args)

    @classmethod
    def convertDistance(kls, painter, dist):
        '''Convert a distance to plotter units.

        dist: eg 0.1 (fraction), 10% (percentage), 1/10 (fraction),
                 10pt, 1cm, 20mm, 1inch, 1in, 1" (size)
        maxsize: size fractions are relative to
        painter: painter to get metrics to convert physical sizes
        '''

        # match distance against expression
        m = kls.distre.match(dist)
        if m is not None:
            # lookup function to call to do conversion
            func = kls.unit_func[m.group(2)]
            return func(m, painter)

        # none of the regexps match
        raise ValueError( "Cannot convert distance in form '%s'" %
                          dist )

    def convert(self, painter):
        """Convert this setting's distance as above"""
        return self.convertDistance(painter, self.val)

    def convertPts(self, painter):
        """Get the distance in points."""
        return self.convert(painter) / painter.pixperpt

    def convertInverse(self, distpix, painter):
        """Convert distance in pixels into units of this distance.
        """

        m = self.distre.match(self.val)
        if m is not None:
            # if it matches convert back
            inversefn = self.inv_unit_func[m.group(2)]
        else:
            # otherwise force unit
            inversefn = self.inv_unit_func['cm']

        # do inverse mapping
        return inversefn(distpix, painter)

class DistancePt(Distance):
    """For a distance in points."""

    def makeControl(self, *args):
        return controls.DistancePt(self, *args)

class DistancePhysical(Distance):
    """For physical distances (no fractional)."""

    def isDist(self, val):
        m = self.distre.match(val)
        if m:
            # disallow non-physical distances
            if m.group(2) not in ('/', '', '%'):
                return True
        return False

    def makeControl(self, *args):
        return controls.Distance(self, *args, physical=True)

class DistanceOrAuto(Distance):
    """A distance or the value Auto"""

    typename = 'distance-or-auto'

    distre = re.compile( distre_expr + r'|^Auto$', re.VERBOSE )

    def isAuto(self):
        return self.val == 'Auto'

    def makeControl(self, *args):
        return controls.Distance(self, allowauto=True, *args)
