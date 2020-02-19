#    Copyright (C) 2009 Jeremy S. Sanders
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

from __future__ import division
import os.path

from .. import qtall as qt
from ..compat import cstr
from .. import setting

def removeBadRecents(itemlist):
    """Remove duplicates from list and bad entries."""
    previous = set()
    i = 0
    while i < len(itemlist):
        if itemlist[i] in previous:
            del itemlist[i]
        elif not os.path.exists(itemlist[i]):
            del itemlist[i]
        else:
            previous.add(itemlist[i])
            i += 1

    # trim list
    del itemlist[10:]

class RecentFilesButton(qt.QPushButton):
    """A button for remembering recent files.

    emits filechosen(filename) if a file is chosen
    """

    filechosen = qt.pyqtSignal(cstr)

    def __init__(self, *args):
        qt.QPushButton.__init__(self, *args)

        self.menu = qt.QMenu()
        self.setMenu(self.menu)
        self.settingname = None

    def setSetting(self, name):
        """Specify settings to use when loading menu.
        Should be called before use."""
        self.settingname = name
        self.fillMenu()

    def fillMenu(self):
        """Add filenames to menu."""
        self.menu.clear()
        recent = setting.settingdb.get(self.settingname, [])
        removeBadRecents(recent)
        setting.settingdb[self.settingname] = recent

        for filename in recent:
            if os.path.exists(filename):
                act = self.menu.addAction( os.path.basename(filename) )
                def loadrecentfile(f):
                    return lambda: self.filechosen.emit(f)
                act.triggered.connect(loadrecentfile(filename))

    def addFile(self, filename):
        """Add filename to list of recent files."""
        recent = setting.settingdb.get(self.settingname, [])
        recent.insert(0, os.path.abspath(filename))
        setting.settingdb[self.settingname] = recent
        self.fillMenu()
