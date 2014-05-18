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

from ..compat import cstr
from .. import qtall as qt4
from .. import setting

class HistoryGroupBox(qt4.QGroupBox):
    """Group box remembers settings of radio buttons inside it.

    emits radioClicked(radiowidget) when clicked
    """

    radioClicked = qt4.pyqtSignal(qt4.QObject)

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
        """Load from settings."""
        # connect up radio buttons to emit clicked signal
        for w in self.children():
            if isinstance(w, qt4.QRadioButton):
                def doemit(widget):
                    return lambda: self.radioClicked.emit(widget)
                w.clicked.connect(doemit(w))

        # set item to be checked
        checked = setting.settingdb.get(self.getSettingName(), "")
        for w in self.children():
            if isinstance(w, qt4.QRadioButton) and (
                w.objectName() == checked or checked == ""):
                w.click()
                return

    def getRadioChecked(self):
        """Get name of radio button checked."""
        for w in self.children():
            if isinstance(w, qt4.QRadioButton) and w.isChecked():
                return w
        return None

    def saveHistory(self):
        """Save to settings."""
        name = cstr(self.getRadioChecked().objectName())
        setting.settingdb[self.getSettingName()] = name

    def showEvent(self, event):
        """Show and load history."""
        qt4.QGroupBox.showEvent(self, event)
        self.loadHistory()

    def hideEvent(self, event):
        """Save history as widget is hidden."""
        qt4.QGroupBox.hideEvent(self, event)
        self.saveHistory()
