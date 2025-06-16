#    Copyright (C) 2010 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This file is part of Veusz.
#
#    Veusz is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    Veusz is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################

from .. import qtall as qt
from .. import setting

class HistoryGroupBox(qt.QGroupBox):
    """Group box remembers settings of radio buttons inside it.

    emits radioClicked(radiowidget) when clicked
    """

    radioClicked = qt.pyqtSignal(qt.QObject)

    def getSettingName(self):
        """Get name for saving in settings."""
        # get dialog for widget
        dialog = self.parent()
        while not isinstance(dialog, qt.QDialog):
            dialog = dialog.parent()

        # combine dialog and object names to make setting
        return '%s_%s_HistoryGroup'  % (
            dialog.objectName(), self.objectName() )

    def loadHistory(self):
        """Load from settings."""
        # connect up radio buttons to emit clicked signal
        for w in self.children():
            if isinstance(w, qt.QRadioButton):
                def doemit(widget):
                    return lambda: self.radioClicked.emit(widget)
                w.clicked.connect(doemit(w))

        # set item to be checked
        checked = setting.settingdb.get(self.getSettingName(), "")
        for w in self.children():
            if isinstance(w, qt.QRadioButton) and (
                w.objectName() == checked or checked == ""):
                w.click()
                return

    def getRadioChecked(self):
        """Get name of radio button checked."""
        for w in self.children():
            if isinstance(w, qt.QRadioButton) and w.isChecked():
                return w
        return None

    def saveHistory(self):
        """Save to settings."""
        name = str(self.getRadioChecked().objectName())
        setting.settingdb[self.getSettingName()] = name

    def showEvent(self, event):
        """Show and load history."""
        qt.QGroupBox.showEvent(self, event)
        self.loadHistory()

    def hideEvent(self, event):
        """Save history as widget is hidden."""
        qt.QGroupBox.hideEvent(self, event)
        self.saveHistory()
