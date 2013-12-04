#    Copyright (C) 2011 Jeremy S. Sanders
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

"""Parameters for import routines."""

from __future__ import division
from ..compat import citems

class ImportParamsBase(object):
    """Import parameters for the various imports.

    Parameters:
     filename: filename to import from
     linked: whether to link to file
     encoding: encoding for file
     prefix: prefix for output dataset names
     suffix: suffix for output dataset names
     tags: list of tags to apply to output datasets
    """

    defaults = {
        'filename': None,
        'linked': False,
        'encoding': 'utf_8',
        'prefix': '',
        'suffix': '',
        'tags': None,
        }

    def __init__(self, **argsv):
        """Initialise the reader to import data from filename.
        """

        #  set defaults
        for k, v in citems(self.defaults):
            setattr(self, k, v)

        # set parameters
        for k, v in citems(argsv):
            if k not in self.defaults:
                raise ValueError("Invalid parameter %s" % k)
            setattr(self, k, v)

        # extra parameters to copy besides defaults
        self._extras = []

    def copy(self):
        """Make a copy of the parameters object."""

        newp = {}
        for k in list(self.defaults.keys()) + self._extras:
            newp[k] = getattr(self, k)
        return self.__class__(**newp)
