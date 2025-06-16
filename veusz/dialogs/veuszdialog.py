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

"""Define a base dialog class cleans up self after being hidden."""

import os.path

from .. import qtall as qt
from .. import utils

# register functions to open up dialogs to recreate a dataset
recreate_register = {}

class VeuszDialog(qt.QDialog):
    """Base dialog class.
    - Loads self from ui file.
    - Deletes self on closing.
    - Emits dialogFinished when dialog is done
    """

    dialogFinished = qt.pyqtSignal(qt.QDialog)

    def __init__(self, mainwindow, uifile, modal=False):
        """Initialise dialog given Veusz mainwindow and uifile for dialog.
        If modal is False, base on a top level window instead
        """

        flag = qt.Qt.WindowType.Dialog
        if not modal:
            flag |= (
                qt.Qt.WindowType.CustomizeWindowHint |
                qt.Qt.WindowType.WindowMinimizeButtonHint |
                qt.Qt.WindowType.WindowMaximizeButtonHint |
                qt.Qt.WindowType.WindowCloseButtonHint |
                qt.Qt.WindowType.WindowTitleHint |
                qt.Qt.WindowType.WindowSystemMenuHint
            )

        qt.QDialog.__init__(self, mainwindow, flag)
        self.setAttribute(qt.Qt.WidgetAttribute.WA_DeleteOnClose)

        qt.loadUi(os.path.join(utils.resourceDirectory, 'ui', uifile), self)

        self.mainwindow = mainwindow

    def hideEvent(self, event):
        """Emits dialogFinished if hidden."""
        if not event.spontaneous():
            self.dialogFinished.emit(self)
        return qt.QDialog.hideEvent(self, event)
