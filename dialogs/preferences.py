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

import os.path

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.utils as utils

class PreferencesDialog(qt4.QDialog):
    """Preferences dialog."""

    def __init__(self, *args):
        """Setup dialog."""
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'preferences.ui'),
                   self)

        win = self
        while not hasattr(win, 'plot'):
            win = win.parent()
        self.plotwindow = win.plot

        # view settings
        self.antialiasCheck.setChecked( setting.settingdb['plot_antialias'] )
        self.intervalCombo.addItem('Disabled')
        for intv in self.plotwindow.intervals[1:]:
            self.intervalCombo.addItem('%gs' % (intv * 0.001))
        index = self.plotwindow.intervals.index(
            setting.settingdb['plot_updateinterval'])
        self.intervalCombo.setCurrentIndex(index)

        # set icon size
        self.iconSizeCombo.setCurrentIndex(
            self.iconSizeCombo.findText(
                str(setting.settingdb['toolbar_size'])))

        # set export dpi
        self.exportDPI.setValidator( qt4.QIntValidator(10, 10000, self) )
        self.exportDPI.setEditText( str(setting.settingdb['export_DPI']) )

        # set export antialias
        self.exportAntialias.setChecked( setting.settingdb['export_antialias'])

        # quality of jpeg export
        self.exportQuality.setValue( setting.settingdb['export_quality'] )

        # changing background color of bitmaps
        self.connect( self.exportBackgroundButton, qt4.SIGNAL('clicked()'),
                      self.slotExportBackgroundChanged )
        self.updateExportBackground(setting.settingdb['export_background'])

        # set color setting
        self.exportColor.setCurrentIndex(
            {True:0, False:1}[setting.settingdb['export_color']])

        # default stylesheet
        self.styleLineEdit.setText(setting.settingdb['stylesheet_default'])
        self.connect( self.styleBrowseButton, qt4.SIGNAL('clicked()'),
                      self.styleBrowseClicked )

        # default custom settings
        self.customLineEdit.setText(setting.settingdb['custom_default'])
        self.connect( self.customBrowseButton, qt4.SIGNAL('clicked()'),
                      self.customBrowseClicked )

        # for plugins
        plugins = list( setting.settingdb.get('plugins', []) )
        self.pluginmodel = qt4.QStringListModel(plugins)
        self.pluginList.setModel(self.pluginmodel)
        self.connect( self.pluginAddButton, qt4.SIGNAL('clicked()'),
                      self.pluginAddClicked )
        self.connect( self.pluginRemoveButton, qt4.SIGNAL('clicked()'),
                      self.pluginRemoveClicked )

        # specifics for color tab
        self.setupColorTab()

    def setupColorTab(self):
        """Initialise color tab
        this makes a grid of controls for each color
        consisting of label, isdefault check and change color button."""

        self.chosencolors = {}
        self.colorbutton = {}
        self.colordefaultcheck = {}
        layout = qt4.QGridLayout()
        for row, colname in enumerate(setting.settingdb.colors):
            isdefault, colval = setting.settingdb['color_%s' %
                                                  colname]
            self.chosencolors[colname] = qt4.QColor(colval)

            # label
            name, tooltip = setting.settingdb.color_names[colname]
            label = qt4.QLabel(name)
            label.setToolTip(tooltip)
            layout.addWidget(label, row, 0)

            # is default check
            defcheck = qt4.QCheckBox("Default")
            defcheck.setToolTip(
                "Use the default color instead of the one chosen here")
            layout.addWidget(defcheck, row, 1)
            self.colordefaultcheck[colname] = defcheck
            defcheck.setChecked(isdefault)

            # button
            button = qt4.QPushButton()

            # connect button to method to change color
            def clicked(color=colname):
                self.colorButtonClickedSlot(color)
            self.connect(button, qt4.SIGNAL('clicked()'),
                         clicked)
            layout.addWidget(button, row, 2)
            self.colorbutton[colname] = button

        self.colorGroup.setLayout(layout)

        self.updateButtonColors()

    def colorButtonClickedSlot(self, color):
        """Open color dialog if color button clicked."""
        retcolor = qt4.QColorDialog.getColor( self.chosencolors[color], self )
        if retcolor.isValid():
            self.chosencolors[color] = retcolor
            self.updateButtonColors()

    def updateButtonColors(self):
        """Update color icons on color buttons."""
        for name, val in self.chosencolors.iteritems():
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
        setting.settingdb['plot_updateinterval'] = (
            self.plotwindow.intervals[ self.intervalCombo.currentIndex() ] )
        setting.settingdb['plot_antialias'] = self.antialiasCheck.isChecked()

        # update icon size if necessary
        iconsize = int( self.iconSizeCombo.currentText() )
        if iconsize != setting.settingdb['toolbar_size']:
            setting.settingdb['toolbar_size'] = iconsize
            for widget in self.parent().children(): # find toolbars
                if isinstance(widget, qt4.QToolBar):
                    widget.setIconSize( qt4.QSize(iconsize, iconsize) )

        # update dpi if possible
        try:
            setting.settingdb['export_DPI'] = int(self.exportDPI.currentText())
        except ValueError:
            pass

        # export settings
        setting.settingdb['export_antialias'] = self.exportAntialias.isChecked()
        setting.settingdb['export_quality'] = self.exportQuality.value()

        setting.settingdb['export_color'] = {0: True, 1: False}[self.exportColor.currentIndex()]
        setting.settingdb['export_background'] = self.exportBackgroundButton.iconcolor

        # new document settings
        setting.settingdb['stylesheet_default'] = unicode(self.styleLineEdit.text())
        setting.settingdb['custom_default'] = unicode(self.customLineEdit.text())

        # colors
        for name, color in self.chosencolors.iteritems():
            isdefault = self.colordefaultcheck[name].isChecked()
            colorname = unicode(color.name())
            setting.settingdb['color_' + name] = (isdefault, colorname)

        # plugins
        plugins = [unicode(x) for x in self.pluginmodel.stringList()]
        setting.settingdb['plugins'] = plugins

        self.plotwindow.updatePlotSettings()

    def styleBrowseClicked(self):
        """Browse for a stylesheet."""
        filename = self.parent()._fileOpenDialog(
            'vst', 'Veusz stylesheet', 'Choose stylesheet')
        if filename:
            self.styleLineEdit.setText(filename)

    def customBrowseClicked(self):
        """Browse for a custom definitons."""
        filename = self.parent()._fileOpenDialog(
            'vsz', 'Veusz documents', 'Choose custom definitons')
        if filename:
            self.customLineEdit.setText(filename)

    def pluginAddClicked(self):
        """Add a new plugin."""
        filename = self.parent()._fileOpenDialog(
            'py', 'Python scripts', 'Choose plugin')
        if filename:
            self.pluginmodel.insertRows(0, 1)
            self.pluginmodel.setData( self.pluginmodel.index(0),
                                      qt4.QVariant(filename) )

    def pluginRemoveClicked(self):
        """Remove selected plugin."""
        sel = self.pluginList.selectionModel().currentIndex()
        if sel.isValid():
            self.pluginmodel.removeRow( sel.row() )
