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
from ..compat import citems
from .. import qtall as qt4
from .. import setting
from .. import utils
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

        # disable thread option if not supported
        if not qt4.QFontDatabase.supportsThreadedFontRendering():
            self.threadSpinBox.setEnabled(False)
            self.threadSpinBox.setToolTip(_("Disabled because of lack of "
                                            "threaded drawing support"))

        # use cwd for file dialogs
        self.cwdCheck.setChecked( setdb['dirname_usecwd'] )

        # set icon size
        self.iconSizeCombo.setCurrentIndex(
            self.iconSizeCombo.findText(
                str(setdb['toolbar_size'])))

        # set export dpi
        dpis = ('75', '90', '100', '150', '200', '300')
        self.exportDPI.addItems(dpis)
        self.exportDPIPDF.addItems(dpis)

        self.exportDPI.setValidator( qt4.QIntValidator(10, 10000, self) )
        self.exportDPI.setEditText( str(setdb['export_DPI']) )
        self.exportDPIPDF.setValidator( qt4.QIntValidator(10, 10000, self) )
        self.exportDPIPDF.setEditText( str(setdb['export_DPI_PDF']) )
        self.exportSVGTextAsText.setChecked( setdb['export_SVG_text_as_text'] )

        # set export antialias
        self.exportAntialias.setChecked( setdb['export_antialias'])

        # quality of jpeg export
        self.exportQuality.setValue( setdb['export_quality'] )

        # changing background color of bitmaps
        self.exportBackgroundButton.clicked.connect(
            self.slotExportBackgroundChanged )
        self.updateExportBackground(setdb['export_background'])

        # set color setting
        self.exportColor.setCurrentIndex(
            {True:0, False:1}[setdb['export_color']])

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

    def setupColorTab(self):
        """Initialise color tab
        this makes a grid of controls for each color
        consisting of label, isdefault check and change color button."""

        self.chosencolors = {}
        self.colorbutton = {}
        self.colordefaultcheck = {}
        layout = qt4.QGridLayout()
        setdb = setting.settingdb
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
        for name, val in citems(self.chosencolors):
            pixmap = qt4.QPixmap(16, 16)
            pixmap.fill(val)
            self.colorbutton[name].setIcon( qt4.QIcon(pixmap) )

    def updateExportBackground(self, colorname):
        """Update color on export background."""
        pixmap = qt4.QPixmap(16, 16)
        col = utils.extendedColorToQColor(colorname)
        pixmap.fill(col)

        # update button (storing color in button itself - what fun!)
        self.exportBackgroundButton.setIcon(qt4.QIcon(pixmap))
        self.exportBackgroundButton.iconcolor = colorname

    def slotExportBackgroundChanged(self):
        """Button clicked to change background."""

        color = qt4.QColorDialog.getColor(
            utils.extendedColorToQColor(self.exportBackgroundButton.iconcolor),
            self,
            "Choose color",
            qt4.QColorDialog.ShowAlphaChannel )
        if color.isValid():
            self.updateExportBackground( utils.extendedColorFromQColor(color) )

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

        # use cwd
        setdb['dirname_usecwd'] = self.cwdCheck.isChecked()

        # update icon size if necessary
        iconsize = int( self.iconSizeCombo.currentText() )
        if iconsize != setdb['toolbar_size']:
            setdb['toolbar_size'] = iconsize
            for widget in self.parent().children(): # find toolbars
                if isinstance(widget, qt4.QToolBar):
                    widget.setIconSize( qt4.QSize(iconsize, iconsize) )

        # update dpi if possible
        # FIXME: requires some sort of visual notification of validator
        for cntrl, setn in ((self.exportDPI, 'export_DPI'),
                            (self.exportDPIPDF, 'export_DPI_PDF')):
            try:
                text = cntrl.currentText()
                valid = cntrl.validator().validate(text, 0)[0]
                if valid == qt4.QValidator.Acceptable:
                    setdb[setn] = int(text)
            except ValueError:
                pass

        # export settings
        setdb['export_antialias'] = self.exportAntialias.isChecked()
        setdb['export_quality'] = self.exportQuality.value()

        setdb['export_color'] = {0: True, 1: False}[
            self.exportColor.currentIndex()]
        setdb['export_background'] = self.exportBackgroundButton.iconcolor
        setdb['export_SVG_text_as_text'] = self.exportSVGTextAsText.isChecked()

        # new document settings
        setdb['stylesheet_default'] = self.styleLineEdit.text()
        setdb['custom_default'] = self.customLineEdit.text()

        # colors
        for name, color in citems(self.chosencolors):
            isdefault = self.colordefaultcheck[name].isChecked()
            colorname = color.name()
            setdb['color_' + name] = (isdefault, colorname)

        # picker
        setdb['picker_to_clipboard'] = self.pickerToClipboardCheck.isChecked()
        setdb['picker_to_console'] = self.pickerToConsoleCheck.isChecked()

        # plugins
        plugins = self.pluginmodel.stringList()
        setdb['plugins'] = plugins

        self.plotwindow.updatePlotSettings()

        # write settings out now, rather than wait until the end
        setdb.writeSettings()

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
