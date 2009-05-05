#    Copyright (C) 2006 Jeremy S. Sanders
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

"""Dialog to show if there is an error loading."""

import os.path

import veusz.qtall as qt4
import veusz.utils as utils

class ErrorLoadingDialog(qt4.QDialog):
    """Dialog when error loading."""

    def __init__(self, parent, filename, error, traceback):
        qt4.QDialog.__init__(self, parent)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'errorloading.ui'),
                   self)

        # insert filename into label
        text = unicode(self.errorlabel.text())
        text = text % filename
        self.errorlabel.setText(text)
        self.errormessagelabel.setText(error)

        # put backtrace into error edit box
        self.errortextedit.setPlainText(traceback)

        # set warning pixmap to left of dialog
        icon = qt4.qApp.style().standardIcon(qt4.QStyle.SP_MessageBoxWarning,
                                             None, self)
        self.iconlabel.setPixmap(icon.pixmap(32))
