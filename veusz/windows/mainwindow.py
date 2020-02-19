# -*- coding: utf-8 -*-
#    Copyright (C) 2003 Jeremy S. Sanders
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

"""Implements the main window of the application."""

from __future__ import division, print_function
import os
import os.path
import sys
import glob
import re
import datetime

try:
    import h5py
except ImportError:
    h5py = None

from ..compat import cstr, cstrerror, cgetcwd, cbytes
from .. import qtall as qt

from .. import document
from .. import utils
from ..utils import vzdbus
from .. import setting
from .. import plugins

from . import consolewindow
from . import plotwindow
from . import treeeditwindow
from .datanavigator import DataNavigatorWindow

def _(text, disambiguation=None, context='MainWindow'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

# shortcut to this
setdb = setting.settingdb

class DBusWinInterface(vzdbus.Object):
    """Simple DBus interface to window for triggering actions."""

    interface = 'org.veusz.actions'

    def __init__(self, actions, index):
        prefix = '/Windows/%i/Actions' % index
        # possible exception in dbus means we have to check sessionbus
        if vzdbus.sessionbus is not None:
            vzdbus.Object.__init__(self, vzdbus.sessionbus, prefix)
        self.actions = actions

    @vzdbus.method(dbus_interface=interface, out_signature='as')
    def GetActions(self):
        """Get list of actions which can be activated."""
        return sorted(self.actions)

    @vzdbus.method(dbus_interface=interface, in_signature='s')
    def TriggerAction(self, action):
        """Activate action given."""
        self.actions[action].trigger()

class MainWindow(qt.QMainWindow):
    """ The main window class for the application."""

    # this is emitted when a dialog is opened by the main window
    dialogShown = qt.pyqtSignal(qt.QWidget)
    # emitted when a document is opened
    documentOpened = qt.pyqtSignal()

    windows = []
    @classmethod
    def CreateWindow(cls, filename=None, mode='graph'):
        """Window factory function.

        If filename is given then that file is loaded into the window.
        Returns window created
        """

        # create the window, and optionally load a saved file
        win = cls()
        win.show()
        if filename:
            # load document
            win.openFileInWindow(filename)
        else:
            win.setupDefaultDoc(mode)

        # try to select first graph of first page
        win.treeedit.doInitialWidgetSelect()

        cls.windows.append(win)

        # check if tutorial wanted (only for graph mode)
        if not setting.settingdb['ask_tutorial'] and mode=='graph':
            win.askTutorial()
            # don't ask again
            setting.settingdb['ask_tutorial'] = True

        # check if version check is ok
        win.askVersionCheck()
        # periodically do the check
        win.doVersionCheck()

        # is it ok to do feedback?
        win.askFeedbackCheck()
        # periodically send feedback
        win.doFeedback()

        return win

    def __init__(self, *args):
        qt.QMainWindow.__init__(self, *args)
        self.setAcceptDrops(True)

        # icon and different size variations
        self.setWindowIcon( utils.getIcon('veusz') )

        # master documenent
        self.document = document.Document()

        # filename for document and update titlebar
        self.filename = ''
        self.updateTitlebar()

        # keep a list of references to dialogs
        self.dialogs = []

        # construct menus and toolbars
        self._defineMenus()

        # make plot window
        self.plot = plotwindow.PlotWindow(self.document, self,
                                          menu = self.menus['view'])
        self.setCentralWidget(self.plot)
        self.plot.showToolbar()

        # likewise with the tree-editing window
        self.treeedit = treeeditwindow.TreeEditDock(self.document, self)
        self.addDockWidget(qt.Qt.LeftDockWidgetArea, self.treeedit)
        self.propdock = treeeditwindow.PropertiesDock(self.document,
                                                      self.treeedit, self)
        self.addDockWidget(qt.Qt.LeftDockWidgetArea, self.propdock)
        self.formatdock = treeeditwindow.FormatDock(self.document,
                                                    self.treeedit, self)
        self.addDockWidget(qt.Qt.LeftDockWidgetArea, self.formatdock)
        self.datadock = DataNavigatorWindow(self.document, self, self)
        self.addDockWidget(qt.Qt.RightDockWidgetArea, self.datadock)

        # make the console window a dock
        self.console = consolewindow.ConsoleWindow(self.document,
                                                   self)
        self.console.hide()
        self.interpreter = self.console.interpreter
        self.addDockWidget(qt.Qt.BottomDockWidgetArea, self.console)

        # assemble the statusbar
        statusbar = self.statusbar = qt.QStatusBar(self)
        self.setStatusBar(statusbar)
        self.updateStatusbar(_('Ready'))

        # a label for the picker readout
        self.pickerlabel = qt.QLabel(statusbar)
        self._setPickerFont(self.pickerlabel)
        statusbar.addPermanentWidget(self.pickerlabel)
        self.pickerlabel.hide()

        # plot queue - how many plots are currently being drawn
        self.plotqueuecount = 0
        self.plot.sigQueueChange.connect(self.plotQueueChanged)
        self.plotqueuelabel = qt.QLabel()
        self.plotqueuelabel.setToolTip(_("Number of rendering jobs remaining"))
        statusbar.addWidget(self.plotqueuelabel)
        self.plotqueuelabel.show()

        # a label for the cursor position readout
        self.axisvalueslabel = qt.QLabel(statusbar)
        statusbar.addPermanentWidget(self.axisvalueslabel)
        self.axisvalueslabel.show()
        self.slotUpdateAxisValues(None)

        # a label for the page number readout
        self.pagelabel = qt.QLabel(statusbar)
        statusbar.addPermanentWidget(self.pagelabel)
        self.pagelabel.show()

        # working directory - use previous one
        self.dirname = setdb.get('dirname', qt.QDir.homePath())
        if setdb['dirname_usecwd']:
            self.dirname = cgetcwd()

        # connect plot signals to main window
        self.plot.sigUpdatePage.connect(self.slotUpdatePage)
        self.plot.sigAxisValuesFromMouse.connect(self.slotUpdateAxisValues)
        self.plot.sigPickerEnabled.connect(self.slotPickerEnabled)
        self.plot.sigPointPicked.connect(self.slotUpdatePickerLabel)

        # disable save if already saved
        self.document.signalModified.connect(self.slotModifiedDoc)
        # if the treeeditwindow changes the page, change the plot window
        self.treeedit.sigPageChanged.connect(self.plot.setPageNumber)

        # if a widget in the plot window is clicked by the user
        self.plot.sigWidgetClicked.connect(self.treeedit.selectWidget)
        self.treeedit.widgetsSelected.connect(self.plot.selectedWidgets)

        # enable/disable undo/redo
        self.menus['edit'].aboutToShow.connect(self.slotAboutToShowEdit)

        #Get the list of recently opened files
        self.populateRecentFiles()
        self.setupWindowGeometry()
        self.defineViewWindowMenu()

        # if document requests it, ask whether an allowed import
        self.document.sigAllowedImports.connect(self.slotAllowedImportsDoc)

        # add on dbus interface
        self.dbusdocinterface = document.DBusInterface(self.document)
        self.dbuswininterface = DBusWinInterface(
            self.vzactions, self.dbusdocinterface.index)

        # has the document already been setup
        self.documentsetup = False

    def updateStatusbar(self, text):
        '''Display text for a set period.'''
        self.statusBar().showMessage(text, 2000)

    def dragEnterEvent(self, event):
        """Check whether event is valid to be dropped."""
        if (event.mimeData().hasUrls() and
            self._getVeuszDropFiles(event)):
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Respond to a drop event on the current window"""
        if event.mimeData().hasUrls():
            files = self._getVeuszDropFiles(event)
            if files:
                if self.document.isBlank():
                    self.openFileInWindow(files[0])
                else:
                    self.CreateWindow(files[0])
                for filename in files[1:]:
                    self.CreateWindow(filename)

    def _getVeuszDropFiles(self, event):
        """Return a list of veusz files from a drag/drop event containing a
        text/uri-list"""

        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        else:
            # get list of vsz files dropped
            urls = [u.toLocalFile() for u in mime.urls()]
            urls = [u for u in urls if os.path.splitext(u)[1] == '.vsz']
            return urls

    def setupDefaultDoc(self, mode):
        """Setup default document."""

        if not self.documentsetup:
            # add page and default graph
            self.document.makeDefaultDoc(mode)

            # set color theme
            self.document.basewidget.settings.get(
                'colorTheme').set(setting.settingdb['colortheme_default'])

            # load defaults if set
            self.loadDefaultStylesheet()
            self.loadDefaultCustomDefinitions()

            # done setup
            self.documentsetup = True

    def loadDefaultStylesheet(self):
        """Loads the default stylesheet for the new document."""
        filename = setdb['stylesheet_default']
        if filename:
            try:
                self.document.applyOperation(
                    document.OperationLoadStyleSheet(filename) )
            except EnvironmentError as e:
                qt.QMessageBox.warning(
                    self, _("Error - Veusz"),
                    _("Unable to load default stylesheet '%s'\n\n%s") %
                    (filename, cstrerror(e)))
            else:
                # reset any modified flag
                self.document.setModified(False)
                self.document.changeset = 0

    def loadDefaultCustomDefinitions(self):
        """Loads the custom definitions for the new document."""
        filename = setdb['custom_default']
        if filename:
            try:
                self.document.applyOperation(
                    document.OperationLoadCustom(filename) )
            except EnvironmentError as e:
                qt.QMessageBox.warning(
                    self, _("Error - Veusz"),
                    _("Unable to load custom definitions '%s'\n\n%s") %
                    (filename, cstrerror(e)))
            else:
                # reset any modified flag
                self.document.setModified(False)
                self.document.changeset = 0

    def slotAboutToShowEdit(self):
        """Enable/disable undo/redo menu items."""

        # enable distable, and add appropriate text to describe
        # the operation being undone/redone
        canundo = self.document.canUndo()
        undotext = _('Undo')
        if canundo:
            undotext = "%s %s" % (undotext, self.document.historyundo[-1].descr)
        self.vzactions['edit.undo'].setText(undotext)
        self.vzactions['edit.undo'].setEnabled(canundo)

        canredo = self.document.canRedo()
        redotext = _('Redo')
        if canredo:
            redotext = "%s %s" % (redotext, self.document.historyredo[-1].descr)
        self.vzactions['edit.redo'].setText(redotext)
        self.vzactions['edit.redo'].setEnabled(canredo)

    def slotEditUndo(self):
        """Undo the previous operation"""
        if self.document.canUndo():
            self.document.undoOperation()
        self.treeedit.checkWidgetSelected()

    def slotEditRedo(self):
        """Redo the previous operation"""
        if self.document.canRedo():
            self.document.redoOperation()

    def slotEditPreferences(self):
        from ..dialogs.preferences import PreferencesDialog
        dialog = PreferencesDialog(self)
        dialog.exec_()

    def slotEditStylesheet(self):
        from ..dialogs.stylesheet import StylesheetDialog
        dialog = StylesheetDialog(self, self.document)
        self.showDialog(dialog)
        return dialog

    def slotEditCustom(self):
        from ..dialogs.custom import CustomDialog
        dialog = CustomDialog(self, self.document)
        self.showDialog(dialog)
        return dialog

    def definePlugins(self, pluginlist, actions, menuname):
        """Create menu items and actions for plugins.

        pluginlist: list of plugin classes
        actions: dict of actions to add new actions to
        menuname: string giving prefix for new menu entries (inside actions)
        """

        def getLoadDialog(pluginkls):
            def _loadPlugin():
                from ..dialogs.plugin import handlePlugin
                handlePlugin(self, self.document, pluginkls)
            return _loadPlugin

        menu = []
        for pluginkls in pluginlist:
            actname = menuname + '.' + '.'.join(pluginkls.menu)
            text = pluginkls.menu[-1]
            if pluginkls.has_parameters:
                text += '...'
            actions[actname] = utils.makeAction(
                self,
                pluginkls.description_short,
                text,
                getLoadDialog(pluginkls))

            # build up menu from tuple of names
            menulook = menu
            namebuild = [menuname]
            for cmpt in pluginkls.menu[:-1]:
                namebuild.append(cmpt)
                name = '.'.join(namebuild)

                for c in menulook:
                    if c[0] == name:
                        menulook = c[2]
                        break
                else:
                    menulook.append( [name, cmpt, []] )
                    menulook = menulook[-1][2]

            menulook.append(actname)

        return menu

    def _defineMenus(self):
        """Initialise the menus and toolbar."""

        # these are actions for main menu toolbars and menus
        a = utils.makeAction
        self.vzactions = {
            'file.new.menu':
                a(self, _('New document'), _('New'),
                  None,
                  icon='kde-document-new'),
            'file.new.graph':
                a(self,
                  _('New graph document'),
                  _('&New graph document'),
                  self.slotFileNewGraph,
                  icon='kde-document-new-graph', key='Ctrl+N'),
            'file.new.polar':
                a(self,
                  _('New polar plot document'),
                  _('New polar document'),
                  self.slotFileNewPolar,
                  icon='kde-document-new-polar'),
            'file.new.ternary':
                a(self,
                  _('New ternary plot document'),
                  _('New ternary document'),
                  self.slotFileNewTernary,
                  icon='kde-document-new-ternary'),
            'file.new.graph3d':
                a(self,
                  _('New 3D plot document'),
                  _('New 3D document'),
                  self.slotFileNewGraph3D,
                  icon='kde-document-new-graph3d'),

            'file.open':
                a(self, _('Open a document'), _('&Open...'),
                  self.slotFileOpen,
                  icon='kde-document-open', key='Ctrl+O'),
            'file.reload':
                a(self, _('Reload document from saved version'),
                  _('Reload...'), self.slotFileReload),
            'file.save':
                a(self, _('Save the document'), _('&Save'),
                  self.slotFileSave,
                  icon='kde-document-save', key='Ctrl+S'),
            'file.saveas':
                a(self, _('Save the current document under a new name'),
                  _('Save &As...'), self.slotFileSaveAs,
                  icon='kde-document-save-as'),
            'file.print':
                a(self, _('Print the document'), _('&Print...'),
                  self.slotFilePrint,
                  icon='kde-document-print', key='Ctrl+P'),
            'file.export':
                a(self, _('Export to graphics formats'), _('&Export...'),
                  self.slotFileExport,
                  icon='kde-document-export'),
            'file.close':
                a(self, _('Close current window'), _('Close Window'),
                  self.slotFileClose,
                  icon='kde-window-close', key='Ctrl+W'),
            'file.quit':
                a(self, _('Exit the program'), _('&Quit'),
                  self.slotFileQuit,
                  icon='kde-application-exit', key='Ctrl+Q'),

            'edit.undo':
                a(self, _('Undo the previous operation'), _('Undo'),
                  self.slotEditUndo,
                  icon='kde-edit-undo',  key='Ctrl+Z'),
            'edit.redo':
                a(self, _('Redo the previous operation'), _('Redo'),
                  self.slotEditRedo,
                  icon='kde-edit-redo', key='Ctrl+Shift+Z'),
            'edit.prefs':
                a(self, _('Edit preferences'), _('Preferences...'),
                  self.slotEditPreferences,
                  icon='veusz-edit-prefs'),
            'edit.custom':
                a(self,
                  _('Edit custom functions, constants, colors and colormaps'),
                  _('Custom definitions...'),
                  self.slotEditCustom,
                  icon='veusz-edit-custom'),

            'edit.stylesheet':
                a(self,
                  _('Edit stylesheet to change default widget settings'),
                  _('Default styles...'),
                  self.slotEditStylesheet, icon='settings_stylesheet'),

            'view.edit':
                a(self, _('Show or hide edit window'), _('Edit window'),
                  None, checkable=True),
            'view.props':
                a(self, _('Show or hide property window'), _('Properties window'),
                  None, checkable=True),
            'view.format':
                a(self, _('Show or hide formatting window'), _('Formatting window'),
                  None, checkable=True),
            'view.console':
                a(self, _('Show or hide console window'), _('Console window'),
                  None, checkable=True),
            'view.datanav':
                a(self, _('Show or hide data navigator window'), _('Data navigator window'),
                  None, checkable=True),

            'view.maintool':
                a(self, _('Show or hide main toolbar'), _('Main toolbar'),
                  None, checkable=True),
            'view.datatool':
                a(self, _('Show or hide data toolbar'), _('Data toolbar'),
                  None, checkable=True),
            'view.viewtool':
                a(self, _('Show or hide view toolbar'), _('View toolbar'),
                  None, checkable=True),
            'view.edittool':
                a(self, _('Show or hide editing toolbar'), _('Editing toolbar'),
                  None, checkable=True),
            'view.addtool':
                a(self, _('Show or hide insert toolbar'), _('Insert toolbar'),
                  None, checkable=True),

            'data.import':
                a(self, _('Import data into Veusz'), _('&Import...'),
                  self.slotDataImport, icon='kde-vzdata-import'),
            'data.edit':
                a(self, _('Edit and enter new datasets'), _('&Editor...'),
                  lambda: self.slotDataEdit(), icon='kde-edit-veuszedit'),
            'data.create':
                a(self, _('Create new datasets using ranges, parametrically or as functions of existing datasets'), _('&Create...'),
                  self.slotDataCreate, icon='kde-dataset-new-veuszedit'),
            'data.create2d':
                a(self, _('Create new 2D datasets from existing datasets, or as a function of x and y'), _('Create &2D...'),
                  self.slotDataCreate2D, icon='kde-dataset2d-new-veuszedit'),
            'data.capture':
                a(self, _('Capture remote data'), _('Ca&pture...'),
                  self.slotDataCapture, icon='veusz-capture-data'),
            'data.filter':
                a(self, _('Filter data'), _('&Filter...'),
                  self.slotDataFilter, icon='kde-filter'),
            'data.histogram':
                a(self, _('Histogram data'), _('&Histogram...'),
                  self.slotDataHistogram, icon='button_bar'),
            'data.reload':
                a(self, _('Reload linked datasets'), _('&Reload'),
                  self.slotDataReload, icon='kde-view-refresh'),

            'help.home':
                a(self, _('Go to the Veusz home page on the internet'),
                  _('Home page'), self.slotHelpHomepage),
            'help.bug':
                a(self, _('Report a bug on the internet'),
                  _('Suggestions and bugs'), self.slotHelpBug),
            'help.update':
                a(self, _('Download latest version'),
                  _('Download latest version'), self.slotHelpUpdate),

            'help.tutorial':
                a(self, _('An interactive Veusz tutorial'),
                  _('Tutorial'), self.slotHelpTutorial),
            'help.about':
                a(self, _('Displays information about the program'), _('About...'),
                  self.slotHelpAbout, icon='veusz')
            }

        # create main toolbar
        tb = self.maintoolbar = qt.QToolBar(_("Main toolbar - Veusz"), self)
        iconsize = setdb['toolbar_size']
        tb.setIconSize(qt.QSize(iconsize, iconsize))
        tb.setObjectName('veuszmaintoolbar')
        self.addToolBar(qt.Qt.TopToolBarArea, tb)

        utils.makeMenuGroupSaved(
            'file.new.menu', self, self.vzactions, (
                    'file.new.graph', 'file.new.graph3d',
                    'file.new.polar', 'file.new.ternary',
            )
        )

        utils.addToolbarActions(
            tb, self.vzactions,
            ('file.new.menu', 'file.open', 'file.save',
             'file.print', 'file.export'))

        # data toolbar
        tb = self.datatoolbar = qt.QToolBar(_("Data toolbar - Veusz"), self)
        tb.setIconSize(qt.QSize(iconsize, iconsize))
        tb.setObjectName('veuszdatatoolbar')
        self.addToolBar(qt.Qt.TopToolBarArea, tb)
        utils.addToolbarActions(
            tb, self.vzactions,
            ('data.import', 'data.edit',
             'data.create', 'data.capture',
             'data.filter', 'data.reload'))

        # menu structure
        filemenu = [
            ['file.new', _('New'),
             ['file.new.graph', 'file.new.graph3d', 'file.new.polar',
              'file.new.ternary']],
            'file.open',
            ['file.filerecent', _('Open &Recent'), []],
            'file.reload',
            '',
            'file.save', 'file.saveas',
            '',
            'file.print', 'file.export',
            '',
            'file.close', 'file.quit'
            ]
        editmenu = [
            'edit.undo', 'edit.redo',
            '',
            ['edit.select', _('&Select'), []],
            '',
            'edit.prefs', 'edit.stylesheet', 'edit.custom',
            ''
            ]
        viewwindowsmenu = [
            'view.edit', 'view.props', 'view.format',
            'view.console', 'view.datanav',
            '',
            'view.maintool', 'view.viewtool',
            'view.addtool', 'view.edittool'
            ]
        viewmenu = [
            ['view.viewwindows', _('&Windows'), viewwindowsmenu],
            ''
            ]
        insertmenu = [
            ]

        # load dataset plugins and create menu
        datapluginsmenu = self.definePlugins( plugins.datasetpluginregistry,
                                              self.vzactions, 'data.ops' )

        datamenu = [
            ['data.ops', _('&Operations'), datapluginsmenu],
            'data.import', 'data.edit', 'data.create',
            'data.create2d', 'data.capture', 'data.filter', 'data.histogram',
            'data.reload',
            ]
        helpmenu = [
            'help.home', 'help.bug', 'help.update',
            '',
            'help.tutorial',
            '',
            ['help.examples', _('&Example documents'), []],
            '',
            'help.about'
            ]

        # load tools plugins and create menu
        toolsmenu = self.definePlugins( plugins.toolspluginregistry,
                                        self.vzactions, 'tools' )

        menus = [
            ['file', _('&File'), filemenu],
            ['edit', _('&Edit'), editmenu],
            ['view', _('&View'), viewmenu],
            ['insert', _('&Insert'), insertmenu],
            ['data', _('&Data'), datamenu],
            ['tools', _('&Tools'), toolsmenu],
            ['help', _('&Help'), helpmenu],
            ]

        self.menus = {}
        utils.constructMenus(self.menuBar(), self.menus, menus, self.vzactions)

        # set icon for File->New
        self.menus['file.new'].setIcon(utils.getIcon('kde-document-new'))

        self.populateExamplesMenu()

    def _setPickerFont(self, label):
        f = label.font()
        f.setBold(True)
        f.setPointSizeF(f.pointSizeF() * 1.2)
        label.setFont(f)

    def populateExamplesMenu(self):
        """Add examples to help menu."""

        # not cstr here forces to unicode for Python 2, getting
        # filenames in unicode
        examples = [ os.path.join(utils.exampleDirectory, f)
                     for f in os.listdir(cstr(utils.exampleDirectory))
                     if os.path.splitext(f)[1] == ".vsz" ]

        menu = self.menus["help.examples"]
        for ex in sorted(examples):
            name = os.path.splitext(os.path.basename(ex))[0]

            def _openexample(ex=ex):
                MainWindow.CreateWindow(ex)

            a = menu.addAction(name, _openexample)
            a.setStatusTip(_("Open %s example document") % name)

    def defineViewWindowMenu(self):
        """Setup View -> Window menu."""

        def viewHideWindow(window):
            """Toggle window visibility."""
            w = window
            def f():
                w.setVisible(not w.isVisible())
            return f

        # set whether windows are visible and connect up to toggle windows
        self.viewwinfns = []
        for win, act in ((self.treeedit, 'view.edit'),
                         (self.propdock, 'view.props'),
                         (self.formatdock, 'view.format'),
                         (self.console, 'view.console'),
                         (self.datadock, 'view.datanav'),
                         (self.maintoolbar, 'view.maintool'),
                         (self.datatoolbar, 'view.datatool'),
                         (self.treeedit.edittoolbar, 'view.edittool'),
                         (self.treeedit.addtoolbar, 'view.addtool'),
                         (self.plot.viewtoolbar, 'view.viewtool')):

            a = self.vzactions[act]
            fn = viewHideWindow(win)
            self.viewwinfns.append( (win, a, fn) )
            a.triggered.connect(fn)

        # needs to update state every time menu is shown
        self.menus['view.viewwindows'].aboutToShow.connect(
            self.slotAboutToShowViewWindow)

    def slotAboutToShowViewWindow(self):
        """Enable/disable View->Window item check boxes."""

        for win, act, fn in self.viewwinfns:
            act.setChecked(not win.isHidden())

    def showDialog(self, dialog):
        """Show dialog given."""
        dialog.dialogFinished.connect(self.deleteDialog)
        self.dialogs.append(dialog)
        dialog.show()
        self.dialogShown.emit(dialog)

    def deleteDialog(self, dialog):
        """Remove dialog from list of dialogs."""
        try:
            idx = self.dialogs.index(dialog)
            del self.dialogs[idx]
        except ValueError:
            pass

    def slotDataImport(self):
        """Display the import data dialog."""
        from ..dialogs import importdialog
        dialog = importdialog.ImportDialog(self, self.document)
        self.showDialog(dialog)
        return dialog

    def slotDataEdit(self, editdataset=None):
        """Edit existing datasets.

        If editdataset is set to a dataset name, edit this dataset
        """
        from ..dialogs import dataeditdialog
        dialog = dataeditdialog.DataEditDialog(self, self.document)
        self.showDialog(dialog)
        if editdataset is not None:
            dialog.selectDataset(editdataset)
        return dialog

    def slotDataCreate(self):
        """Create new datasets."""
        from ..dialogs.datacreate import DataCreateDialog
        dialog = DataCreateDialog(self, self.document)
        self.showDialog(dialog)
        return dialog

    def slotDataCreate2D(self):
        """Create new datasets."""
        from ..dialogs.datacreate2d import DataCreate2DDialog
        dialog = DataCreate2DDialog(self, self.document)
        self.showDialog(dialog)
        return dialog

    def slotDataCapture(self):
        """Capture remote data."""
        from ..dialogs.capturedialog import CaptureDialog
        dialog = CaptureDialog(self.document, self)
        self.showDialog(dialog)
        return dialog

    def slotDataFilter(self):
        """Filter datasets."""
        from ..dialogs.filterdialog import FilterDialog
        dialog = FilterDialog(self, self.document)
        self.showDialog(dialog)
        return dialog

    def slotDataHistogram(self):
        """Histogram data."""
        from ..dialogs.histodata import HistoDataDialog
        dialog = HistoDataDialog(self, self.document)
        self.showDialog(dialog)
        return dialog

    def slotDataReload(self):
        """Reload linked datasets."""
        from ..dialogs.reloaddata import ReloadData
        dialog = ReloadData(self.document, self)
        self.showDialog(dialog)
        return dialog

    def slotHelpHomepage(self):
        """Go to the veusz homepage."""
        qt.QDesktopServices.openUrl(qt.QUrl('https://veusz.github.io/'))

    def slotHelpBug(self):
        """Go to the veusz bug page."""
        qt.QDesktopServices.openUrl(
            qt.QUrl('https://github.com/veusz/veusz/issues') )

    def askTutorial(self):
        """Ask if tutorial wanted."""
        retn = qt.QMessageBox.question(
            self, _("Veusz Tutorial"),
            _("Veusz includes a tutorial to help get you started.\n"
              "Would you like to start the tutorial now?\n"
              "If not, you can access it later through the Help menu."),
            qt.QMessageBox.Yes | qt.QMessageBox.No
            )

        if retn == qt.QMessageBox.Yes:
            self.slotHelpTutorial()

    def slotHelpTutorial(self):
        """Show a Veusz tutorial."""
        if self.document.isBlank():
            # run the tutorial
            from .tutorial import TutorialDock
            tutdock = TutorialDock(self.document, self, self)
            self.addDockWidget(qt.Qt.RightDockWidgetArea, tutdock)
            tutdock.show()
        else:
            # open up a blank window for tutorial
            win = self.CreateWindow()
            win.slotHelpTutorial()

    def slotHelpAbout(self):
        """Show about dialog."""
        from ..dialogs.aboutdialog import AboutDialog
        AboutDialog(self).exec_()

    def askVersionCheck(self, mininterval=2):
        """Check with user whether to do version checks.

        This is only done after the user has been using the program
        for mininterval days

        """

        dayssinceinstall = (
            datetime.date.today() -
            datetime.date(*setting.settingdb['install_date'])).days
        if (dayssinceinstall<mininterval or
            setting.settingdb['vercheck_asked_user'] or
            setting.settingdb['vercheck_disabled'] or
            utils.disableVersionChecks):
            return

        retn = qt.QMessageBox.question(
            self, _("Version check"),
            _("Veusz will periodically check for new Veusz versions and\n"
              "let you know if there is a new one available.\n\n"
              "Is this ok? This choice can be changed in Preferences."),
            qt.QMessageBox.Yes | qt.QMessageBox.No,
            qt.QMessageBox.Yes
            )

        setting.settingdb['vercheck_disabled'] = retn==qt.QMessageBox.No
        setting.settingdb['vercheck_asked_user'] = True

    def doVersionCheck(self):
        """Check whether there is a new version.
        """
        self.vzactions['help.update'].setVisible(False)

        # check is done asynchronously
        thread = utils.VersionCheckThread(self)
        thread.newversion.connect(self.slotNewVersion)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def askFeedbackCheck(self, mininterval=3):
        """Check with user whether to do feedback.

        This is only done after the user has been using the program
        for mininterval days

        """

        dayssinceinstall = (
            datetime.date.today() -
            datetime.date(*setting.settingdb['install_date'])).days
        if (dayssinceinstall<mininterval or
            setting.settingdb['feedback_asked_user'] or
            setting.settingdb['feedback_disabled'] or
            utils.disableFeedback):
            return

        retn = qt.QMessageBox.question(
            self, _("Send automatic anonymous feedback"),
            _("Veusz can automatically send anonymous feedback "
              "to the developers, with information about the version "
              "of software dependencies, the computer language and how "
              "often features are used.\n\n"
              "Is this ok? This choice can be changed in Preferences."),
            qt.QMessageBox.Yes | qt.QMessageBox.No,
            qt.QMessageBox.Yes
            )

        setting.settingdb['feedback_disabled'] = retn==qt.QMessageBox.No
        setting.settingdb['feedback_asked_user'] = True

    def doFeedback(self):
        """Give feedback."""
        thread = utils.FeedbackCheckThread(self)
        thread.start()

    def slotNewVersion(self, ver):
        """Called when there is a new version."""
        msg = _('Veusz %s is available for download - see Help menu') % ver
        self.statusBar().showMessage(msg, 5000)
        self.vzactions['help.update'].setText(
            _('Download new Veusz %s') % ver)
        self.vzactions['help.update'].setVisible(True)

    def slotHelpUpdate(self):
        """Open web page to update."""
        qt.QDesktopServices.openUrl(qt.QUrl(
            'https://veusz.github.io/download/'))

    def queryOverwrite(self):
        """Do you want to overwrite the current document.

        Returns qt.QMessageBox.(Yes,No,Cancel)."""

        # include filename in mesage box if we can
        filetext = ''
        if self.filename:
            filetext = " '%s'" % os.path.basename(self.filename)

        return qt.QMessageBox.warning(
            self,
            _("Save file?"),
            _("Document%s was modified. Save first?") % filetext,
            qt.QMessageBox.Save | qt.QMessageBox.Discard |
            qt.QMessageBox.Cancel)

    def closeEvent(self, event):
        """Before closing, check whether we need to save first."""

        # if the document has been modified then query user for saving
        if self.document.isModified():
            v = self.queryOverwrite()
            if v == qt.QMessageBox.Cancel:
                event.ignore()
                return
            elif v == qt.QMessageBox.Save:
                self.slotFileSave()

        # store working directory
        setdb['dirname'] = self.dirname

        # store the current geometry in the settings database
        geometry = ( self.x(), self.y(), self.width(), self.height() )
        setdb['geometry_mainwindow'] = geometry

        # store docked windows
        data = self.saveState().data()
        setdb['geometry_mainwindowstate'] = cbytes(data)

        # save current setting db
        setdb.writeSettings()

        event.accept()

    def setupWindowGeometry(self):
        """Restoring window geometry if possible."""

        # count number of main windows shown
        nummain = 0
        for w in qt.qApp.topLevelWidgets():
            if isinstance(w, qt.QMainWindow):
                nummain += 1

        # if we can restore the geometry, do so
        if 'geometry_mainwindow' in setdb:
            geometry = setdb['geometry_mainwindow']
            self.resize( qt.QSize(geometry[2], geometry[3]) )
            if nummain <= 1:
                geomrect = qt.QApplication.desktop().availableGeometry()
                newpos = qt.QPoint(geometry[0], geometry[1])
                if geomrect.contains(newpos):
                    self.move(newpos)

        # restore docked window geometry
        if 'geometry_mainwindowstate' in setdb:
            try:
                self.restoreState(setdb['geometry_mainwindowstate'])
            except Exception:
                # type can be wrong if switching between Py2/3 PyQ4/5
                pass

    def slotFileNewGraph(self):
        """New file (graph)."""
        self.CreateWindow()

    def slotFileNewPolar(self):
        """New file (polar)."""
        self.CreateWindow(mode='polar')

    def slotFileNewTernary(self):
        """New file (ternary)."""
        self.CreateWindow(mode='ternary')

    def slotFileNewGraph3D(self):
        """New file (graph3d)."""
        self.CreateWindow(mode='graph3d')

    def slotFileSave(self):
        """Save file."""

        if self.filename == '':
            self.slotFileSaveAs()
        else:
            # show busy cursor
            qt.QApplication.setOverrideCursor( qt.QCursor(qt.Qt.WaitCursor) )
            try:
                ext = os.path.splitext(self.filename)[1]
                mode = 'hdf5' if ext == '.vszh5' else 'vsz'
                self.document.save(self.filename, mode)
                self.updateStatusbar(_("Saved to %s") % self.filename)
            except EnvironmentError as e:
                qt.QApplication.restoreOverrideCursor()
                qt.QMessageBox.critical(
                    self, _("Error - Veusz"),
                    _("Unable to save document as '%s'\n\n%s") %
                    (self.filename, cstrerror(e)))
            else:
                # restore the cursor
                qt.QApplication.restoreOverrideCursor()

    def updateTitlebar(self):
        """Put the filename into the title bar."""
        if self.filename == '':
            self.setWindowTitle(_('Untitled - Veusz'))
        else:
            self.setWindowTitle( _("%s - Veusz") %
                                 os.path.basename(self.filename) )

    def plotQueueChanged(self, incr):
        self.plotqueuecount += incr
        text = u'•' * self.plotqueuecount
        self.plotqueuelabel.setText(text)

    def fileSaveDialog(self, filters, dialogtitle):
        """A generic file save dialog for exporting / saving.

        filters: list of filters
        """

        fd = qt.QFileDialog(self, dialogtitle)
        fd.setDirectory(self.dirname)
        fd.setFileMode(qt.QFileDialog.AnyFile)
        fd.setAcceptMode(qt.QFileDialog.AcceptSave)
        fd.setNameFilters(filters)

        # selected filetype is saved under a key constructed here
        filetype_re = re.compile(r'.*\(\*\.([a-z0-9]+)\)')
        filtertypes = [filetype_re.match(f).group(1) for f in filters]
        filterkey = '_'.join(['filterdefault'] + filtertypes)
        if filterkey in setting.settingdb:
            filter = setting.settingdb[filterkey]
            if filter in filters:
                fd.selectNameFilter(filter)

        # okay was selected (and is okay to overwrite if it exists)
        if fd.exec_() == qt.QDialog.Accepted:
            # save directory for next time
            self.dirname = fd.directory().absolutePath()
            # update the edit box
            filename = fd.selectedFiles()[0]
            filetype = filetype_re.match(fd.selectedNameFilter()).group(1)
            if os.path.splitext(filename)[1][1:] != filetype:
                filename += '.' + filetype
            setting.settingdb[filterkey] = fd.selectedNameFilter()
            return filename

        return None

    def fileOpenDialog(self, filters, dialogtitle):
        """Display an open dialog and return a filename.

        filters: list of filters in format "Filetype (*.vsz)"
        """

        fd = qt.QFileDialog(self, dialogtitle)
        fd.setDirectory(self.dirname)
        fd.setFileMode( qt.QFileDialog.ExistingFile )
        fd.setAcceptMode( qt.QFileDialog.AcceptOpen )
        fd.setNameFilters(filters)

        # if the user chooses a file
        if fd.exec_() == qt.QDialog.Accepted:
            # save directory for next time
            self.dirname = fd.directory().absolutePath()

            filename = fd.selectedFiles()[0]
            try:
                with open(filename):
                    pass
            except EnvironmentError as e:
                qt.QMessageBox.critical(
                    self, _("Error - Veusz"),
                    _("Unable to open '%s'\n\n%s") %
                    (filename, cstrerror(e)))
                return None
            return filename
        return None

    def slotFileSaveAs(self):
        """Save As file."""

        filters = [_('Veusz document files (*.vsz)')]
        if h5py is not None:
            filters += [_('Veusz HDF5 document files (*.vszh5)')]
        filename = self.fileSaveDialog(filters, _('Save as'))
        if filename:
            self.filename = filename
            self.updateTitlebar()

            self.slotFileSave()

    def openFile(self, filename):
        """Select whether to load the file in the
        current window or in a blank window and calls the appropriate loader"""

        if self.document.isBlank():
            # If the file is new and there are no modifications,
            # reuse the current window
            self.openFileInWindow(filename)
        else:
            # create a new window
            self.CreateWindow(filename)

    def loadDocument(self, filename):
        """Load a Veusz document.
        Return True if loaded ok
        """

        def _callbackunsafe():
            """Callback when loading document to ask whether ok to continue loading
            if unsafe commands are found."""
            qt.QApplication.restoreOverrideCursor()
            msgbox = qt.QMessageBox(
                qt.QMessageBox.Warning,
                _("Unsafe code in document"),
                _("The document '%s' contains potentially unsafe code "
                  "which may damage your computer or data. Please check "
                  "that the file comes from a trusted source.") % filename,
                qt.QMessageBox.NoButton,
                self)
            cont = msgbox.addButton(_("C&ontinue anyway"), qt.QMessageBox.AcceptRole)
            stop = msgbox.addButton(_("&Stop loading"), qt.QMessageBox.RejectRole)
            msgbox.setDefaultButton(stop)
            msgbox.exec_()
            qt.QApplication.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            return msgbox.clickedButton() is cont

        def _callbackimporterror(filename, error):
            """Ask user if they want to give a new filename in case of import
            error.
            """
            qt.QApplication.restoreOverrideCursor()
            msgbox = qt.QMessageBox(self)
            msgbox.setWindowTitle(_("Import error"))
            msgbox.setText(
                _("Could not import data from file '%s':\n\n %s") % (
                    filename, error))
            msgbox.setInformativeText(_("Do you want to look for another file?"))
            msgbox.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.Cancel)
            filename = None
            if msgbox.exec_() == qt.QMessageBox.Yes:
                filename = qt.QFileDialog.getOpenFileName(self, "Choose data file")
                filename = filename[0] if filename else None
            qt.QApplication.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            return filename

        # save stdout and stderr, then redirect to console
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = self.console.con_stdout
        sys.stderr = self.console.con_stderr

        qt.QApplication.setOverrideCursor( qt.QCursor(qt.Qt.WaitCursor) )

        try:
            # get loading mode
            ext = os.path.splitext(filename)[1].lower()
            if ext in ('.vsz', '.py'):
                mode = 'vsz'
            elif ext in ('.h5', '.hdf5', '.he5', '.vszh5'):
                mode = 'hdf5'
            else:
                raise document.LoadError(
                    _("Did not recognise file type '%s'") % ext)

            # do the actual loading
            self.document.load(
                filename,
                mode=mode,
                callbackunsafe=_callbackunsafe,
                callbackimporterror=_callbackimporterror)

        except document.LoadError as e:
            from ..dialogs.errorloading import ErrorLoadingDialog
            qt.QApplication.restoreOverrideCursor()
            if e.backtrace:
                d = ErrorLoadingDialog(self, filename, cstr(e), e.backtrace)
                d.exec_()
            else:
                qt.QMessageBox.critical(
                    self, _("Error opening %s - Veusz") % filename,
                    cstr(e))
            return False

        qt.QApplication.restoreOverrideCursor()

        # need to remember to restore stdout, stderr
        sys.stdout, sys.stderr = stdout, stderr

        self.documentsetup = True
        return True

    def openFileInWindow(self, filename):
        """Actually do the work of loading a new document.
        """

        ok = self.loadDocument(filename)
        if not ok:
            return

        # remember file for recent list
        self.addRecentFile(filename)

        # let the main window know
        self.filename = filename
        self.updateTitlebar()
        self.updateStatusbar(_("Opened %s") % filename)

        # use current directory of file if not using cwd mode
        if not setdb['dirname_usecwd']:
            self.dirname = os.path.dirname( os.path.abspath(filename) )

        # notify cmpts which need notification that doc has finished opening
        self.documentOpened.emit()

    def addRecentFile(self, filename):
        """Add a file to the recent files list."""

        recent = setdb['main_recentfiles']
        filename = os.path.abspath(filename)

        if filename in recent:
            del recent[recent.index(filename)]
        recent.insert(0, filename)
        setdb['main_recentfiles'] = recent[:10]
        self.populateRecentFiles()

    def slotFileOpen(self):
        """Open an existing file in a new window."""

        filters = ['*.vsz']
        if h5py is not None:
            filters.append('*.vszh5')

        filename = self.fileOpenDialog(
            [_('Veusz document files (%s)') % ' '.join(filters)],
            _('Open'))
        if filename:
            self.openFile(filename)

    def populateRecentFiles(self):
        """Populate the recently opened files menu with a list of
        recently opened files"""

        def opener(path):
            def _fileOpener():
                self.openFile(path)
            return _fileOpener

        menu = self.menus["file.filerecent"]
        menu.clear()

        if setdb['main_recentfiles']:
            files = [f for f in setdb['main_recentfiles']
                     if os.path.isfile(f)]

            # add each recent file to menu
            newmenuitems = []
            for i, path in enumerate(files):
                newmenuitems.append(
                    ('filerecent%i' % i,_('Open File %s') % path,
                     os.path.basename(path),
                     'file.filerecent', opener(path),
                     '', False, ''))

            menu.setEnabled(True)
            self.recentFileActions = utils.populateMenuToolbars(
                newmenuitems, self.maintoolbar, self.menus)
        else:
            menu.setEnabled(False)

    def slotFileReload(self):
        """Reload document from saved version."""

        retn = qt.QMessageBox.warning(
            self,
            _("Reload file"),
            _("Reload document from file, losing any changes?"),
            qt.QMessageBox.Yes | qt.QMessageBox.Cancel,
            qt.QMessageBox.Cancel)
        if retn == qt.QMessageBox.Yes:
            if not os.path.exists(self.filename):
                qt.QMessageBox.critical(
                    self,
                    _("Reload file"),
                    _("File %s no longer exists") % self.filename)
            else:
                self.openFileInWindow(self.filename)

    def slotFileExport(self):
        """Export the graph."""
        from ..dialogs.export import ExportDialog
        dialog = ExportDialog(self, self.document, self.filename)
        self.showDialog(dialog)
        return dialog

    def slotFilePrint(self):
        """Print the document."""
        document.printDialog(self, self.document, filename=self.filename)

    def slotModifiedDoc(self, ismodified):
        """Disable certain actions if document is not modified."""

        # enable/disable file, save menu item
        self.vzactions['file.save'].setEnabled(ismodified)

        # enable/disable reloading from saved document
        self.vzactions['file.reload'].setEnabled(
            bool(self.filename) and ismodified)

    def slotFileClose(self):
        """File close window chosen."""
        self.close()

    def slotFileQuit(self):
        """File quit chosen."""
        qt.qApp.closeAllWindows()

    def slotUpdatePage(self, number):
        """Update page number when the plot window says so."""

        np = self.document.getNumberPages()
        if np == 0:
            self.pagelabel.setText(_("No pages"))
        else:
            self.pagelabel.setText(_("Page %i/%i") % (number+1, np))

    def slotUpdateAxisValues(self, values):
        """Update the position where the mouse is relative to the axes."""

        if values:
            # construct comma separated text representing axis values
            valitems = [
                '%s=%#.4g' % (name, values[name])
                for name in sorted(values) ]
            self.axisvalueslabel.setText(', '.join(valitems))
        else:
            self.axisvalueslabel.setText(_('No position'))

    def slotPickerEnabled(self, enabled):
        if enabled:
            self.pickerlabel.setText(_('No point selected'))
            self.pickerlabel.show()
        else:
            self.pickerlabel.hide()

    def slotUpdatePickerLabel(self, info):
        """Display the picked point"""
        xv, yv = info.coords
        xn, yn = info.labels
        xt, yt = info.displaytype
        ix = str(info.index)
        if ix:
            ix = '[' + ix + ']'

        # format values for display
        def fmt(val, dtype):
            if dtype == 'date':
                return utils.dateFloatToString(val)
            elif dtype == 'numeric':
                return '%0.5g' % val
            elif dtype == 'text':
                return val
            else:
                raise RuntimeError

        xtext = fmt(xv, xt)
        ytext = fmt(yv, yt)

        t = '%s: %s%s = %s, %s%s = %s' % (
            info.widget.name, xn, ix, xtext, yn, ix, ytext)
        self.pickerlabel.setText(t)
        if setdb['picker_to_console']:
            self.console.appendOutput(t + "\n", 'error')
        if setdb['picker_to_clipboard']:
            clipboard = qt.QApplication.clipboard()
            if clipboard.mimeData().hasText():
                clipboard.setText(clipboard.text()+"\n"+t)
            else:
                qt.QApplication.clipboard().setText(t)

    def slotAllowedImportsDoc(self, module, names):
        """Are allowed imports?"""
        from ..dialogs.safetyimport import SafetyImportDialog
        d = SafetyImportDialog(self, module, names)
        d.exec_()
