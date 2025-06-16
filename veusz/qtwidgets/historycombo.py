#    Copyright (C) 2009 Jeremy S. Sanders
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

"""A combobox which remembers its history.

The history is stored in the Veusz settings database.
"""

from .. import qtall as qt
from .. import setting

class HistoryCombo(qt.QComboBox):
    """This combobox records what items have been entered into it so the
    user can choose them again.

    Duplicates and blanks are ignored.
    """

    def __init__(self, *args, **argsv):
        qt.QComboBox.__init__(self, *args, **argsv)

        # sane defaults
        self.setEditable(True)
        self.setMaxCount(50)
        self.setInsertPolicy(qt.QComboBox.InsertPolicy.InsertAtTop)
        self.setDuplicatesEnabled(False)
        self.setSizePolicy( qt.QSizePolicy(
            qt.QSizePolicy.Policy.MinimumExpanding, qt.QSizePolicy.Policy.Fixed) )
        self.completer().setCaseSensitivity(qt.Qt.CaseSensitivity.CaseSensitive)

        # stops combobox readjusting in size to fit contents
        self.setSizeAdjustPolicy(
            qt.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)

        self.default = []
        self.hasshown = False

    def text(self):
        """Get text in combobox
        - this gives it the same interface as QLineEdit."""
        return self.currentText()

    def setText(self, text):
        """Set text in combobox
        - gives same interface as QLineEdit."""
        self.lineEdit().setText(text)

    def hasAcceptableInput(self):
        """Input valid?
        - gives same interface as QLineEdit."""
        return self.lineEdit().hasAcceptableInput()

    def replaceAndAddHistory(self, item):
        """Replace the text and place item at top of history."""

        self.lineEdit().setText(item)
        index = self.findText(item)  # lookup for existing item (if any)
        if index != -1:
            # remove any old items matching this
            self.removeItem(index)

        # put new item in
        self.insertItem(0, item)
        # set selected item in drop down list match current item
        self.setCurrentIndex(0)

    def getSettingName(self):
        """Get name for saving in settings."""

        # get dialog for widget
        dialog = self.parent()
        while not isinstance(dialog, qt.QDialog):
            dialog = dialog.parent()

        # combine dialog and object names to make setting
        return '%s_%s_HistoryCombo'  % (
            dialog.objectName(), self.objectName() )

    def loadHistory(self):
        """Load contents of history combo from settings."""
        self.clear()
        history = setting.settingdb.get(self.getSettingName(), self.default)
        self.insertItems(0, history)

        self.hasshown = True

    def saveHistory(self):
        """Save contents of history combo to settings."""

        # only save history if it has been loaded
        if not self.hasshown:
            return

        # collect current items
        history = [ self.itemText(i) for i in range(self.count()) ]
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
        # we do this now rather than in __init__ because the widget
        # has no name set at __init__
        if not self.hasshown:
            self.loadHistory()

    def hideEvent(self, event):
        """Save history as widget is hidden."""
        qt.QComboBox.hideEvent(self, event)
        self.saveHistory()
