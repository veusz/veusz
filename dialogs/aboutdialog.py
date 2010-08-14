# about dialog box
# aboutdialog.py

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

"""About dialog module."""

import os.path

import veusz.qtall as qt4
import veusz.utils as utils
from veuszdialog import VeuszDialog

class AboutDialog(VeuszDialog):
    """About dialog."""

    def __init__(self, mainwindow):
        VeuszDialog.__init__(self, mainwindow, 'about.ui')

        # draw logo in dialog
        self.frame.setBackgroundRole(qt4.QPalette.Base)
        self.frame.setAutoFillBackground(True)
        self.logolabel.setPixmap( utils.getPixmap('logo.png') )

        # add version to copyright text
        copyrighttext = unicode(self.copyrightlabel.text())
        copyrighttext = copyrighttext % {'version': utils.version()}
        self.copyrightlabel.setText(copyrighttext)

        self.connect(self.licenseButton, qt4.SIGNAL('clicked()'),
                     self.licenseClicked)

    def licenseClicked(self):
        """Show the license."""
        LicenseDialog(self).exec_()
        
class LicenseDialog(VeuszDialog):
    """About license dialog."""

    def __init__(self, parent):
        VeuszDialog.__init__(self, parent, 'license.ui')

        try:
            f = open(os.path.join(utils.veuszDirectory, 'COPYING'), 'rU')
            text = f.read()
        except IOError:
            text = 'Could not open the license file.'

        self.licenseEdit.setPlainText(text)
