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

"""A combobox which remembers its history.

The history is stored in the Veusz settings database.
"""

from __future__ import division
from ..compat import crange
from .. import qtall as qt4
from .. import setting

class HistoryCombo(qt4.QComboBox):
    """This combobox records what items have been entered into it so the
    user can choose them again.

    Duplicates and blanks are ignored.
    """

    def __init__(self, *args):
        qt4.QComboBox.__init__(self, *args)

        # sane defaults
        self.setEditable(True)
        self.setMaxCount(50)
        self.setInsertPolicy(qt4.QComboBox.InsertAtTop)
        self.setDuplicatesEnabled(False)
        self.setSizePolicy( qt4.QSizePolicy(qt4.QSizePolicy.MinimumExpanding,
                                            qt4.QSizePolicy.Fixed) )

        # stops combobox readjusting in size to fit contents
        self.setSizeAdjustPolicy(
            qt4.QComboBox.AdjustToMinimumContentsLengthWithIcon)

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
        while not isinstance(dialog, qt4.QDialog):
            dialog = dialog.parent()

        # combine dialog and object names to make setting
        return '%s_%s_HistoryCombo'  % ( dialog.objectName(),
                                         self.objectName() )

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
        qt4.QComboBox.showEvent(self, event)
        # we do this now rather than in __init__ because the widget
        # has no name set at __init__
        if not self.hasshown:
            self.loadHistory()

    def hideEvent(self, event):
        """Save history as widget is hidden."""
        qt4.QComboBox.hideEvent(self, event)
        self.saveHistory()
