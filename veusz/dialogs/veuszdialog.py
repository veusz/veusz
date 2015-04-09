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

"""Define a base dialog class cleans up self after being hidden."""

from __future__ import division
import os.path

from .. import qtall as qt4
from .. import utils

# register functions to open up dialogs to recreate a dataset
recreate_register = {}

class VeuszDialog(qt4.QDialog):
    """Base dialog class.
    - Loads self from ui file.
    - Deletes self on closing.
    - Emits dialogFinished when dialog is done
    """

    dialogFinished = qt4.pyqtSignal(qt4.QDialog)

    def __init__(self, mainwindow, uifile, modal=False):
        """Initialise dialog given Veusz mainwindow and uifile for dialog.
        If modal is False, base on a top level window instead
        """

        flag = qt4.Qt.Dialog
        if not modal:
            flag |= (
                qt4.Qt.CustomizeWindowHint | qt4.Qt.WindowMinimizeButtonHint |
                qt4.Qt.WindowMaximizeButtonHint | qt4.Qt.WindowCloseButtonHint |
                qt4.Qt.WindowTitleHint | qt4.Qt.WindowSystemMenuHint )

        qt4.QDialog.__init__(self, mainwindow, flag)
        self.setAttribute(qt4.Qt.WA_DeleteOnClose)

        qt4.loadUi(os.path.join(utils.resourceDirectory, 'ui', uifile), self)

        self.mainwindow = mainwindow

    def hideEvent(self, event):
        """Emits dialogFinished if hidden."""
        if not event.spontaneous():
            self.dialogFinished.emit(self)
        return qt4.QDialog.hideEvent(self, event)
