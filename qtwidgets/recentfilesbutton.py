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

import os.path

import veusz.qtall as qt4
import veusz.setting as setting

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

class RecentFilesButton(qt4.QPushButton):
    """A button for remembering recent files.

    emits filechosen(filename) if a file is chosen
    """

    def __init__(self, *args):
        qt4.QPushButton.__init__(self, *args)

        self.menu = qt4.QMenu()
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
                def loadRecentFile(filename=filename):
                    self.emit(qt4.SIGNAL('filechosen'), filename)
                self.connect( act, qt4.SIGNAL('triggered()'),
                              loadRecentFile )

    def addFile(self, filename):
        """Add filename to list of recent files."""
        recent = setting.settingdb.get(self.settingname, [])
        recent.insert(0, os.path.abspath(filename))
        setting.settingdb[self.settingname] = recent
        self.fillMenu()

                      
