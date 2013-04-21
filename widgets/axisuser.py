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

"""Inherited by users of axis information."""

class AxisUser(object):
    """Inherited by objects which use or provide information to an axis."""

    def getAxesNames(self):
        """Returns names of axes used."""
        return ()

    def lookupAxis(self, axisname):
        """Find widget associated with axisname."""
        w = self.parent
        while w:
            for c in w.children:
                if c.name == axisname and hasattr(c, 'isaxis'):
                    return c
            w = w.parent
        return None

    def providesAxesDependency(self):
        """Returns information on the following axes.
        format is ( ('x', 'sx'), ('y', 'sy') )
        where key is the axis and value is a provided bound
        """
        return ()

    def requiresAxesDependency(self):
        """Requires information about the axis given before providing
        information.
        Format (('sx', 'x'), ('sy', 'y'))
        """
        return ()

    def getRange(self, axis, depname, therange):
        """Update range variable for axis with dependency name given."""
        pass
