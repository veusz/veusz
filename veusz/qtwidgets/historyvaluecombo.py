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

"""A combobox which remembers previous setting
"""

from __future__ import division
from ..compat import crange
from .. import qtall as qt
from .. import setting

class HistoryValueCombo(qt.QComboBox):
    """This combobox records what value was previously saved
    """

    def __init__(self, *args):
        qt.QComboBox.__init__(self, *args)
        self.defaultlist = []
        self.defaultval = None
        self.hasshown = False

    def getSettingName(self):
        """Get name for saving in settings."""

        # get dialog for widget
        dialog = self.parent()
        while not isinstance(dialog, qt.QDialog):
            dialog = dialog.parent()

        # combine dialog and object names to make setting
        return '%s_%s_HistoryValueCombo'  % ( dialog.objectName(),
                                              self.objectName() )

    def saveHistory(self):
        """Save contents of history combo to settings."""

        # only save history if it has been loaded
        if not self.hasshown:
            return

        # collect current items
        history = [ self.itemText(i) for i in crange(self.count()) ]
        history.insert(0, self.currentText())

        # remove dups
        histout = []
        histset = set()
        for item in history:
            if item not in histset:
                histout.append(item)
                histset.add(item)

        # save the history
        setting.settingdb[self.getSettingName()] = histout

    def showEvent(self, event):
        """Show HistoryCombo and load history."""
        qt.QComboBox.showEvent(self, event)
        if self.hasshown:
            return

        self.clear()
        self.addItems(self.defaultlist)
        text = setting.settingdb.get(self.getSettingName(), self.defaultval)
        if text is not None:
            indx = self.findText(text)
            if indx < 0:
                if self.isEditable():
                    self.insertItem(0, text)
                indx = 0
            self.setCurrentIndex(indx)
        self.hasshown = True

    def hideEvent(self, event):
        """Save history as widget is hidden."""
        qt.QComboBox.hideEvent(self, event)

        if self.hasshown:
            text = self.currentText()
            setting.settingdb[self.getSettingName()] = text

