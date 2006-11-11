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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id: $

import os.path

import veusz.qtall as qt4
import veusz.setting as setting

class PreferencesDialog(qt4.QDialog):
    """Preferences dialog."""

    def __init__(self, *args):
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'preferences.ui'),
                   self)

        # set export dpi
        self.exportDPI.setValidator( qt4.QIntValidator(10, 10000, self) )
        self.exportDPI.setEditText( str(setting.settingdb['export_DPI']) )

        # set antialias
        self.exportAntialias.setChecked( setting.settingdb['export_antialias'])

        # set color setting
        self.exportColor.setCurrentIndex(
            {True:0, False:1}[setting.settingdb['export_color']])

    def accept(self):
        """Keep settings if okay pressed."""
        
        qt4.QDialog.accept(self)

        # update dpi if possible
        try:
            setting.settingdb['export_DPI'] = int(self.exportDPI.currentText())
        except ValueError:
            pass

        # other settings
        setting.settingdb['export_antialias'] = self.exportAntialias.isChecked()

        setting.settingdb['export_color'] = {0: True, 1: False}[self.exportColor.currentIndex()]
        
