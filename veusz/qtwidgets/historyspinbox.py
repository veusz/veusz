#    Copyright (C) 2010 Jeremy S. Sanders
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
from .. import qtall as qt4
from .. import setting

class HistorySpinBox(qt4.QSpinBox):
    """A SpinBox which remembers its setting between calls."""

    def __init__(self, *args):
        qt4.QSpinBox.__init__(self, *args)
        self.default = 0

    def getSettingName(self):
        """Get name for saving in settings."""
        # get dialog for widget
        dialog = self.parent()
        while not isinstance(dialog, qt4.QDialog):
            dialog = dialog.parent()

        # combine dialog and object names to make setting
        return "%s_%s_HistorySpinBox" % ( dialog.objectName(),
                                          self.objectName() )

    def loadHistory(self):
        """Load contents of HistorySpinBox from settings."""
        num = setting.settingdb.get(self.getSettingName(), self.default)
        self.setValue(num)

    def saveHistory(self):
        """Save contents of HistorySpinBox to settings."""
        setting.settingdb[self.getSettingName()] = self.value()

    def showEvent(self, event):
        """Show HistorySpinBox and load history."""
        qt4.QSpinBox.showEvent(self, event)
        self.loadHistory()

    def hideEvent(self, event):
        """Save history as widget is hidden."""
        qt4.QSpinBox.hideEvent(self, event)
        self.saveHistory()
