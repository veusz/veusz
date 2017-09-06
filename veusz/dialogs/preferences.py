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

from __future__ import division
from .. import qtall as qt4
from .. import setting
from .. import utils
from .. import document
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="PrefsDialog"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

# names for display of colors and a longer description
color_names = {
    'page': (_('Page'),
             _('Page background color')),
    'error': (_('Error'),
              _('Color for errors')),
    'command': (_('Console command'),
                _('Commands in the console window color')),
    'cntrlline': (_('Control line'),
                  _('Color of lines controlling widgets')),
    'cntrlcorner': (_('Control corner'),
                    _('Color of corners controlling widgets')),
    }

class PreferencesDialog(VeuszDialog):
    """Preferences dialog."""

    def __init__(self, mainwindow):
        """Setup dialog."""
        VeuszDialog.__init__(self, mainwindow, 'preferences.ui', modal=True)

        self.plotwindow = mainwindow.plot

        # for ease of use
        setdb = setting.settingdb

        # view settings
        self.antialiasCheck.setChecked( setdb['plot_antialias'] )
        self.englishCheck.setChecked( setdb['ui_english'] )
        for intv in self.plotwindow.updateintervals:
            self.intervalCombo.addItem(intv[1])
        index = [i[0] for i in self.plotwindow.updateintervals].index(
            setdb['plot_updatepolicy'])
        self.intervalCombo.setCurrentIndex(index)
        self.threadSpinBox.setValue( setdb['plot_numthreads'] )
        self.translationEdit.setText( setdb['translation_file'] )
        self.translationBrowseButton.clicked.connect(
            self.translationBrowseClicked)

        # disable thread option if not supported
        if not qt4.QFontDatabase.supportsThreadedFontRendering():
            self.threadSpinBox.setEnabled(False)
            self.threadSpinBox.setToolTip(_("Disabled because of lack of "
                                            "threaded drawing support"))

        # use cwd for file dialogs
        (self.dirDocCWDRadio if setdb['dirname_usecwd'] else self.dirDocPrevRadio).click()

        # exporting documents
        {
            'doc': self.dirExportDocRadio,
            'cwd': self.dirExportCWDRadio,
            'prev': self.dirExportPrevRadio,
        }[setdb.get('dirname_export_location')].click()

        # set icon size
        self.iconSizeCombo.setCurrentIndex(
            self.iconSizeCombo.findText(
                str(setdb['toolbar_size'])))

        # default stylesheet
        self.styleLineEdit.setText(setdb['stylesheet_default'])
        self.styleBrowseButton.clicked.connect(self.styleBrowseClicked)

        # default custom settings
        self.customLineEdit.setText(setdb['custom_default'])
        self.customBrowseButton.clicked.connect(self.customBrowseClicked)

        # for plugins
        plugins = list( setdb.get('plugins', []) )
        self.pluginmodel = qt4.QStringListModel(plugins)
        self.pluginList.setModel(self.pluginmodel)
        self.pluginAddButton.clicked.connect(self.pluginAddClicked)
        self.pluginRemoveButton.clicked.connect(self.pluginRemoveClicked)

        # specifics for color tab
        self.setupColorTab()

        # for point picker
        self.pickerToConsoleCheck.setChecked( setdb['picker_to_console'] )
        self.pickerToClipboardCheck.setChecked( setdb['picker_to_clipboard'] )

        # python path
        self.externalPythonPath.setText(setdb['external_pythonpath'])
        self.externalGhostscript.setText(setdb['external_ghostscript'])
        self.externalGhostscriptBrowse.clicked.connect(
            self.externalGhostscriptBrowseClicked)

    def setupColorTab(self):
        """Initialise color tab
        this makes a grid of controls for each color
        consisting of label, isdefault check and change color button."""

        setdb = setting.settingdb

        # theme
        themes = sorted(list(document.colors.colorthemes))
        self.colorThemeDefCombo.addItems(themes)
        self.colorThemeDefCombo.setCurrentIndex(
            themes.index(setdb['colortheme_default']))

        # UI colors
        self.chosencolors = {}
        self.colorbutton = {}
        self.colordefaultcheck = {}
        layout = qt4.QGridLayout()
        for row, colname in enumerate(setdb.colors):
            isdefault, colval = setting.settingdb['color_%s' % colname]
            self.chosencolors[colname] = qt4.QColor(colval)

            # label
            name, tooltip = color_names[colname]
            label = qt4.QLabel(name)
            label.setToolTip(tooltip)
            layout.addWidget(label, row, 0)

            # is default check
            defcheck = qt4.QCheckBox(_("Default"))
            defcheck.setToolTip(
                _("Use the default color instead of the one chosen here"))
            layout.addWidget(defcheck, row, 1)
            self.colordefaultcheck[colname] = defcheck
            defcheck.setChecked(isdefault)

            # connect button to method to change color
            button = self.colorbutton[colname] = qt4.QPushButton()
            def getcolclick(cname):
                # double function to get around colname changing
                return lambda: self.colorButtonClicked(cname)
            button.clicked.connect(getcolclick(colname))
            layout.addWidget(button, row, 2)

        self.colorGroup.setLayout(layout)

        self.updateButtonColors()

    def colorButtonClicked(self, cname):
        """Open color dialog if color button clicked."""
        retcolor = qt4.QColorDialog.getColor( self.chosencolors[cname], self )
        if retcolor.isValid():
            self.chosencolors[cname] = retcolor
            self.updateButtonColors()

    def updateButtonColors(self):
        """Update color icons on color buttons."""
        for name, val in self.chosencolors.items():
            pixmap = qt4.QPixmap(16, 16)
            pixmap.fill(val)
            self.colorbutton[name].setIcon( qt4.QIcon(pixmap) )

    def accept(self):
        """Keep settings if okay pressed."""

        qt4.QDialog.accept(self)

        # view settings
        setdb = setting.settingdb
        setdb['plot_updatepolicy'] = (
            self.plotwindow.updateintervals[self.intervalCombo.currentIndex()][0] )
        setdb['plot_antialias'] = self.antialiasCheck.isChecked()
        setdb['ui_english'] = self.englishCheck.isChecked()
        setdb['plot_numthreads'] = self.threadSpinBox.value()
        setdb['translation_file'] = self.translationEdit.text()

        # use cwd
        setdb['dirname_usecwd'] = self.dirDocCWDRadio.isChecked()

        for radio, val in (
                (self.dirExportDocRadio, 'doc'),
                (self.dirExportCWDRadio, 'cwd'),
                (self.dirExportPrevRadio, 'prev'),
                ):
            if radio.isChecked():
                setdb['dirname_export_location'] = val

        # update icon size if necessary
        iconsize = int( self.iconSizeCombo.currentText() )
        if iconsize != setdb['toolbar_size']:
            setdb['toolbar_size'] = iconsize
            for widget in self.parent().children(): # find toolbars
                if isinstance(widget, qt4.QToolBar):
                    widget.setIconSize( qt4.QSize(iconsize, iconsize) )

        # new document settings
        setdb['stylesheet_default'] = self.styleLineEdit.text()
        setdb['custom_default'] = self.customLineEdit.text()

        # color theme
        setdb['colortheme_default'] = self.colorThemeDefCombo.currentText()

        # UI colors
        for name, color in self.chosencolors.items():
            isdefault = self.colordefaultcheck[name].isChecked()
            colorname = color.name()
            setdb['color_' + name] = (isdefault, colorname)

        # plugins
        plugins = self.pluginmodel.stringList()
        setdb['plugins'] = plugins

        # picker
        setdb['picker_to_clipboard'] = self.pickerToClipboardCheck.isChecked()
        setdb['picker_to_console'] = self.pickerToConsoleCheck.isChecked()

        # python path
        setdb['external_pythonpath'] = self.externalPythonPath.text()
        setdb['external_ghostscript'] = self.externalGhostscript.text()

        self.plotwindow.updatePlotSettings()

        # write settings out now, rather than wait until the end
        setdb.writeSettings()

    def translationBrowseClicked(self):
        """Browse for a translation."""
        filename = self.parent().fileOpenDialog(
            [_('Translation file (*.qm)')], _('Choose translation file'))
        if filename:
            self.translationEdit.setText(filename)

    def styleBrowseClicked(self):
        """Browse for a stylesheet."""
        filename = self.parent().fileOpenDialog(
            [_('Veusz stylesheet (*.vst)')], _('Choose stylesheet'))
        if filename:
            self.styleLineEdit.setText(filename)

    def customBrowseClicked(self):
        """Browse for a custom definitons."""
        filename = self.parent().fileOpenDialog(
            [_('Veusz document (*.vsz)')], _('Choose custom definitons'))
        if filename:
            self.customLineEdit.setText(filename)

    def pluginAddClicked(self):
        """Add a new plugin."""
        filename = self.parent().fileOpenDialog(
            [_('Python scripts (*.py)')], _('Choose plugin'))
        if filename:
            self.pluginmodel.insertRows(0, 1)
            self.pluginmodel.setData( self.pluginmodel.index(0),
                                      filename )

    def pluginRemoveClicked(self):
        """Remove selected plugin."""
        sel = self.pluginList.selectionModel().currentIndex()
        if sel.isValid():
            self.pluginmodel.removeRow( sel.row() )

    def externalGhostscriptBrowseClicked(self):
        """Choose a ghostscript executable."""

        filename = self.parent().fileOpenDialog(
            [_('All files (*)')], _('Choose ghostscript executable'))
        if filename:
            self.externalGhostscript.setText(filename)
