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
import datetime

from . import utilfuncs
from ..compat import curlrequest
from .. import qtall as qt

_errmsg = """Failed to find VERSION file.

This is probably because the resource files are not installed in the
python module directory. You may need to set the environment variable
VEUSZ_RESOURCE_DIR or add a "resources" symlink in the main veusz
module directory pointing to the directory where resources are
located. See INSTALL for details.
"""

_ver = None
def version():
    """Return the version number as a string."""

    global _ver
    if _ver:
        return _ver

    filename = os.path.join(utilfuncs.resourceDirectory, "VERSION")
    try:
        with open(filename) as f:
            _ver = f.readline().strip()
        return _ver
    except EnvironmentError:
        sys.stderr.write(_errmsg)
        sys.exit(1)

def versionToTuple(ver):
    """Convert version to tuple, e.g. '2.1.1' -> (2,1,1)."""
    return tuple([int(x) for x in ver.split('.')])

def latestVersion():
    """Get latest version of Veusz from website as string.

    Returns None if error
    """

    try:
        f = curlrequest.urlopen(
            'http://veusz.github.io/download/newest-version.html')
        p = f.read()
        f.close()

        latest = p.decode('ascii').strip()
        # check format
        intver = versionToTuple(latest)
    except Exception:
        return None

    return latest

class VersionCheckThread(qt.QThread):
    """Asynchronously check for new version, emitting signal if found."""

    newversion = qt.pyqtSignal(str)

    # minimum number of days to wait between checks
    mininterval = 7

    def run(self):
        # can't import setting above because of loops
        from .. import setting

        if ( disableVersionChecks or
             setting.settingdb['vercheck_disabled'] or
             not setting.settingdb['vercheck_asked_user']):
            return

        today = datetime.date.today()
        dayssincecheck = (
            today -
            datetime.date(*setting.settingdb['vercheck_last_done'])).days

        if dayssincecheck >= self.mininterval or dayssincecheck < 0:
            setting.settingdb['vercheck_last_done'] = (
                today.year, today.month, today.day)
            #print("doing check")
            latestver = latestVersion()
            if latestver:
                setting.settingdb['vercheck_latest'] = latestver

        thisver = version()
        latestver = setting.settingdb['vercheck_latest']
        #print('latest ver', latestver, thisver)

        # is newer version available?
        if ( latestver and
             versionToTuple(latestver) > versionToTuple(thisver)):
            self.newversion.emit(latestver)

# patch this to be True if you are packaging Veusz and want to disable
# version checks
disableVersionChecks=False
