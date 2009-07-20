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

# $Id: $

import os.path

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.utils as utils

class PreferencesDialog(qt4.QDialog):
    """Preferences dialog."""

    def __init__(self, *args):
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'preferences.ui'),
                   self)

        # set export dpi
        self.exportDPI.setValidator( qt4.QIntValidator(10, 10000, self) )
        self.exportDPI.setEditText( str(setting.settingdb['export_DPI']) )

        # set antialias
        self.exportAntialias.setChecked( setting.settingdb['export_antialias'])

        # quality of jpeg export
        self.exportQuality.setValue( setting.settingdb['export_quality'] )

        # set color setting
        self.exportColor.setCurrentIndex(
            {True:0, False:1}[setting.settingdb['export_color']])

        # default stylesheet
        self.styleLineEdit.setText(setting.settingdb['stylesheet_default'])

        self.connect( self.styleBrowseButton, qt4.SIGNAL('clicked()'),
                      self.styleBrowseClicked )

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
        setting.settingdb['export_quality'] = self.exportQuality.value()

        setting.settingdb['export_color'] = {0: True, 1: False}[self.exportColor.currentIndex()]
        
        setting.settingdb['stylesheet_default'] = unicode(self.styleLineEdit.text())

    def styleBrowseClicked(self):
        """Browse for a stylesheet."""
        filename = self.parent()._fileOpenDialog(
            'vst', 'Veusz stylesheet', 'Import stylesheet')
        if filename:
            self.styleLineEdit.setText(filename)

