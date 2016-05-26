# version.py
# return the version number

#    Copyright (C) 2004 Jeremy S. Sanders
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

"""
Return Veusz' version number
"""

from __future__ import division
import os.path
import sys

from . import utilfuncs

_errmsg = """Failed to find VERSION file.

This is probably because the resource files are not installed in the
python module directory. You may need to set the environment variable
VEUSZ_RESOURCE_DIR or add a "resources" symlink in the main veusz
module directory pointing to the directory where resources are
located. See INSTALL for details.
"""

def version():
    """Return the version number as a string."""

    filename = os.path.join(utilfuncs.resourceDirectory, "VERSION")
    try:
        with open(filename) as f:
            return f.readline().strip()
    except EnvironmentError:
        sys.stderr.write(_errmsg)
        sys.exit(1)
