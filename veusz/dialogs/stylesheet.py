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
###############################################################################

from __future__ import division

from ..compat import cstrerror
from .. import utils
from .. import qtall as qt
from .. import document
from ..windows.treeeditwindow import TabbedFormatting, PropertyList, \
    SettingsProxySingle
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="StylesheetDialog"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class StylesheetDialog(VeuszDialog):
    """This is a dialog box to edit stylesheets.
    Most of the work is done elsewhere, so this doesn't do a great deal
    """

    def __init__(self, parent, document):
        VeuszDialog.__init__(self, parent, 'stylesheet.ui')
        self.document = document
        self.stylesheet = document.basewidget.settings.StyleSheet

        self.stylesListWidget.setMinimumWidth(100)

        # initial properties widget
        self.tabformat = None
        self.properties = None

        self.fillStyleList()

        self.stylesListWidget.currentItemChanged.connect(
            self.slotStyleItemChanged)

        self.stylesListWidget.setCurrentRow(0)

        # we disable default buttons as they keep grabbing the enter key
        close = self.buttonBox.button(qt.QDialogButtonBox.Close)
        close.setDefault(False)
        close.setAutoDefault(False)

        self.saveButton.clicked.connect(self.slotSaveStyleSheet)
        self.loadButton.clicked.connect(self.slotLoadStyleSheet)

        # recent button shows list of recently used files for loading
        self.recentButton.filechosen.connect(self.loadStyleSheet)
        self.recentButton.setSetting('stylesheetdialog_recent')

    def loadStyleSheet(self, filename):
        """Load the given stylesheet."""
        self.document.applyOperation(
            document.OperationLoadStyleSheet(filename) )

    def fillStyleList(self):
        """Fill list of styles."""
        for stns in self.stylesheet.getSettingsList():
            item = qt.QListWidgetItem(
                utils.getIcon(stns.pixmap), stns.usertext)
            item.VZsettings = stns
            self.stylesListWidget.addItem(item)

    def slotStyleItemChanged(self, current, previous):
        """Item changed in list of styles."""
        if current is None:
            return

        if self.tabformat:
            self.tabformat.deleteLater()
        if self.properties:
            self.properties.deleteLater()

        settings = current.VZsettings

        # update formatting properties
        setnsproxy = SettingsProxySingle(self.document, settings)
        self.tabformat = TabbedFormatting(self.document, setnsproxy)
        self.formattingGroup.layout().addWidget(self.tabformat)

        # update properties
        self.properties = PropertyList(self.document, showformatsettings=False)
        self.properties.updateProperties(setnsproxy, showformatting=False)
        self.propertiesScrollArea.setWidget(self.properties)

    def slotSaveStyleSheet(self):
        """Save stylesheet as a file."""
    
        filename = self.parent().fileSaveDialog(
            [_('Veusz stylesheet (*.vst)')], _('Save stylesheet'))
        if filename:
            try:
                f = open(filename, 'w')
                self.document.exportStyleSheet(f)
                f.close()
                self.recentButton.addFile(filename)

            except EnvironmentError as e:
                qt.QMessageBox.critical(
                    self, _("Error - Veusz"),
                    _("Unable to save '%s'\n\n%s") % (
                        filename, cstrerror(e)))

    def slotLoadStyleSheet(self):
        """Load a style sheet."""
        filename = self.parent().fileOpenDialog(
            [_('Veusz stylesheet (*.vst)')], _('Load stylesheet'))
        if filename:
            try:
                self.loadStyleSheet(filename)
            except EnvironmentError as e:
                qt.QMessageBox.critical(
                    self, _("Error - Veusz"),
                    _("Unable to load '%s'\n\n%s") % (
                        filename, cstrerror(e)))
            else:
                # add to recent file list
                self.recentButton.addFile(filename)
