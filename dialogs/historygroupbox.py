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

# $Id$

import veusz.qtall as qt4
import veusz.setting as setting

class HistoryGroupBox(qt4.QGroupBox):
    """Group box remembers settings of radio buttons inside it."""

    def getSettingName(self):
        """Get name for saving in settings."""
        # get dialog for widget
        dialog = self.parent()
        while not isinstance(dialog, qt4.QDialog):
            dialog = dialog.parent()

        # combine dialog and object names to make setting
        return '%s_%s_HistoryGroup'  % ( dialog.objectName(),
                                         self.objectName() )

    def loadHistory(self):
        """Load contents of HistoryCheck from settings."""
        checked = setting.settingdb.get(self.getSettingName(), "")

        # set item to be checked
        for w in self.children():
            if w.objectName() == checked:
                w.setChecked(True)
                return

    def getRadioChecked(self):
        """Get name of radio button checked."""
        for w in self.children():
            if isinstance(w, qt4.QRadioButton) and w.isChecked():
                return unicode( w.objectName() )
        return None

    def saveHistory(self):
        """Save contents of HistoryCheck to settings."""
        setting.settingdb[self.getSettingName()] = self.getRadioChecked()

    def showEvent(self, event):
        """Show HistoryCheck and load history."""
        qt4.QGroupBox.showEvent(self, event)
        self.loadHistory()

    def hideEvent(self, event):
        """Save history as widget is hidden."""
        qt4.QGroupBox.hideEvent(self, event)
        self.saveHistory()
