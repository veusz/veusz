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
from .. import qtall as qt
from .. import setting

class HistoryCheck(qt.QCheckBox):
    """Checkbox remembers its setting between calls
    """

    def __init__(self, *args):
        qt.QCheckBox.__init__(self, *args)
        self.default = False

    def getSettingName(self):
        """Get name for saving in settings."""
        # get dialog for widget
        dialog = self.parent()
        while not isinstance(dialog, qt.QDialog):
            dialog = dialog.parent()

        # combine dialog and object names to make setting
        return '%s_%s_HistoryCheck'  % ( dialog.objectName(),
                                         self.objectName() )

    def loadHistory(self):
        """Load contents of HistoryCheck from settings."""
        checked = setting.settingdb.get(self.getSettingName(), self.default)
        # this is to ensure toggled() signals get sent
        self.setChecked(not checked)
        self.setChecked(checked)

    def saveHistory(self):
        """Save contents of HistoryCheck to settings."""
        setting.settingdb[self.getSettingName()] = self.isChecked()

    def showEvent(self, event):
        """Show HistoryCheck and load history."""
        qt.QCheckBox.showEvent(self, event)
        # we do this now rather than in __init__ because the widget
        # has no name set at __init__
        self.loadHistory()

    def hideEvent(self, event):
        """Save history as widget is hidden."""
        qt.QCheckBox.hideEvent(self, event)
        self.saveHistory()

