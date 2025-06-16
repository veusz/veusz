#    Copyright (C) 2006 Jeremy S. Sanders
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

"""Dialog to show if there is an error loading."""

from .. import qtall as qt
from .veuszdialog import VeuszDialog

class ErrorLoadingDialog(VeuszDialog):
    """Dialog when error loading."""

    def __init__(self, parent, filename, error, traceback):
        VeuszDialog.__init__(self, parent, 'errorloading.ui')

        # insert filename into label
        text = self.errorlabel.text()
        text = text % filename
        self.errorlabel.setText(text)
        self.errormessagelabel.setText(error)

        # put backtrace into error edit box
        self.errortextedit.setPlainText(traceback)

        # set warning pixmap to left of dialog
        icon = qt.QCoreApplication.instance().style().standardIcon(
            qt.QStyle.StandardPixmap.SP_MessageBoxWarning, None, self)
        self.iconlabel.setPixmap(icon.pixmap(32))
