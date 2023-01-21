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

"""About dialog module."""

import os.path
from .. import qtall as qt
from .. import utils
from .veuszdialog import VeuszDialog
from .exceptiondialog import versionHeader

class AboutDialog(VeuszDialog):
    """About dialog."""

    def __init__(self, mainwindow):
        VeuszDialog.__init__(self, mainwindow, 'about.ui', modal=True)

        # draw logo in dialog
        logo = utils.SvgWidgetFixedAspect(os.path.join(utils.imagedir, 'logo.svg'))
        self.logolayout.addWidget(logo)
        self.logoframe.setBackgroundRole(qt.QPalette.Base)
        self.logoframe.setAutoFillBackground(True)

        # add version to copyright text
        copyrighttext = self.copyrightlabel.text()
        copyrighttext = copyrighttext % {'version': utils.version()}
        self.copyrightlabel.setText(copyrighttext)

        self.licenseButton.clicked.connect(self.licenseClicked)
        self.versionsButton.clicked.connect(self.versionsClicked)

    def licenseClicked(self):
        """Show the license."""
        LicenseDialog(self).exec_()

    def versionsClicked(self):
        """Show software versions."""
        SoftwareVersionsDialog(self).exec_()

class LicenseDialog(VeuszDialog):
    """About license dialog."""

    def __init__(self, parent):
        VeuszDialog.__init__(self, parent, 'license.ui')
        self.licenseEdit.setPlainText(utils.getLicense())

class SoftwareVersionsDialog(VeuszDialog):
    """Software versions lists."""

    def __init__(self, parent):
        VeuszDialog.__init__(self, parent, 'softwareversions.ui')
        self.versionInfoText.setPlainText(versionHeader())
