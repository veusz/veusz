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

import os.path
import sys
import traceback

import veusz.qtall as qt4

import veusz.document as document
import veusz.utils as utils
import veusz.setting as setting
import veusz.plugins as plugins

import consolewindow
import plotwindow
import treeeditwindow
from datanavigator import DataNavigatorWindow

from veusz.dialogs.aboutdialog import AboutDialog
from veusz.dialogs.reloaddata import ReloadData
from veusz.dialogs.datacreate import DataCreateDialog
from veusz.dialogs.datacreate2d import DataCreate2DDialog
from veusz.dialogs.preferences import PreferencesDialog
from veusz.dialogs.errorloading import ErrorLoadingDialog
from veusz.dialogs.capturedialog import CaptureDialog
from veusz.dialogs.stylesheet import StylesheetDialog
from veusz.dialogs.custom import CustomDialog
from veusz.dialogs.safetyimport import SafetyImportDialog
from veusz.dialogs.histodata import HistoDataDialog
from veusz.dialogs.plugin import handlePlugin
import veusz.dialogs.importdialog as importdialog
import veusz.dialogs.dataeditdialog as dataeditdialog

# shortcut to this
setdb = setting.settingdb

class MainWindow(qt4.QMainWindow):
    """ The main window class for the application."""

    windows = []
    @classmethod
    def CreateWindow(cls, filename=None):
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
            # add page and default graph
            win.document.makeDefaultDoc()

            # load defaults if set
            win.loadDefaultStylesheet()
            win.loadDefaultCustomDefinitions()

        # try to select first graph of first page
        win.treeedit.doInitialWidgetSelect()
            
        cls.windows.append(win)
        return win

    def __init__(self, *args):
        qt4.QMainWindow.__init__(self, *args)
        self.setAcceptDrops(True)

        # icon and different size variations
        d = utils.imagedir
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
        self.addDockWidget(qt4.Qt.LeftDockWidgetArea, self.treeedit)
        self.propdock = treeeditwindow.PropertiesDock(self.document,
                                                      self.treeedit, self)
        self.addDockWidget(qt4.Qt.LeftDockWidgetArea, self.propdock)
        self.formatdock = treeeditwindow.FormatDock(self.document,
                                                    self.treeedit, self)
        self.addDockWidget(qt4.Qt.LeftDockWidgetArea, self.formatdock)
        self.datadock = DataNavigatorWindow(self.document, self, self)
        self.addDockWidget(qt4.Qt.RightDockWidgetArea, self.datadock)

        # make the console window a dock
        self.console = consolewindow.ConsoleWindow(self.document,
                                                   self)
        self.console.hide()
        self.interpreter = self.console.interpreter
        self.addDockWidget(qt4.Qt.BottomDockWidgetArea, self.console)

        # assemble the statusbar
        statusbar = self.statusbar = qt4.QStatusBar(self)
        self.setStatusBar(statusbar)
        self.updateStatusbar('Ready')

        # a label for the picker readout
        self.pickerlabel = qt4.QLabel(statusbar)
        self._setPickerFont(self.pickerlabel)
        statusbar.addPermanentWidget(self.pickerlabel)
        self.pickerlabel.hide()

        # plot queue - how many plots are currently being drawn
        self.plotqueuecount = 0
        self.connect( self.plot, qt4.SIGNAL("queuechange"),
                      self.plotQueueChanged )
        self.plotqueuelabel = qt4.QLabel()
        self.plotqueuelabel.setToolTip("Number of rendering jobs remaining")
        statusbar.addWidget(self.plotqueuelabel)
        self.plotqueuelabel.show()

        # a label for the cursor position readout
        self.axisvalueslabel = qt4.QLabel(statusbar)
        statusbar.addPermanentWidget(self.axisvalueslabel)
        self.axisvalueslabel.show()
        self.slotUpdateAxisValues(None)

        # a label for the page number readout
        self.pagelabel = qt4.QLabel(statusbar)
        statusbar.addPermanentWidget(self.pagelabel)
        self.pagelabel.show()

        # working directory - use previous one
        self.dirname = setdb.get('dirname', qt4.QDir.homePath())
        self.dirname_export = setdb.get('dirname_export', self.dirname)
        if setdb['dirname_usecwd']:
            self.dirname = self.dirname_export = os.getcwd()

        # connect plot signals to main window
        self.connect( self.plot, qt4.SIGNAL("sigUpdatePage"),
                      self.slotUpdatePage )
        self.connect( self.plot, qt4.SIGNAL("sigAxisValuesFromMouse"),
                      self.slotUpdateAxisValues )
        self.connect( self.plot, qt4.SIGNAL("sigPickerEnabled"),
                      self.slotPickerEnabled )
        self.connect( self.plot, qt4.SIGNAL("sigPointPicked"),
                      self.slotUpdatePickerLabel )

        # disable save if already saved
        self.connect( self.document, qt4.SIGNAL("sigModified"),
                      self.slotModifiedDoc )
        # if the treeeditwindow changes the page, change the plot window
        self.connect( self.treeedit, qt4.SIGNAL("sigPageChanged"),
                      self.plot.setPageNumber )

        # if a widget in the plot window is clicked by the user
        self.connect( self.plot, qt4.SIGNAL("sigWidgetClicked"),
                      self.treeedit.selectWidget )
        self.connect( self.treeedit, qt4.SIGNAL("widgetsSelected"),
                      self.plot.selectedWidgets )

        # enable/disable undo/redo
        self.connect(self.menus['edit'], qt4.SIGNAL('aboutToShow()'),
                     self.slotAboutToShowEdit)

        #Get the list of recently opened files
        self.populateRecentFiles()
        self.setupWindowGeometry()
        self.defineViewWindowMenu()

        # if document requests it, ask whether an allowed import
        self.connect(self.document, qt4.SIGNAL('check_allowed_imports'),
                     self.slotAllowedImportsDoc)

    def updateStatusbar(self, text):
        '''Display text for a set period.'''
        self.statusBar().showMessage(text, 2000)

    def dragEnterEvent(self, event):
        """Check whether event is valid to be dropped."""
        if (event.provides("text/uri-list") and
            self._getVeuszDropFiles(event)):
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Respond to a drop event on the current window"""
        if event.provides("text/uri-list"):
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
            urls = [unicode(u.path()) for u in mime.urls()]
            urls = [u for u in urls if os.path.splitext(u)[1] == '.vsz']
            return urls

    def loadDefaultStylesheet(self):
        """Loads the default stylesheet for the new document."""
        filename = setdb['stylesheet_default']
        if filename:
            try:
                self.document.applyOperation(
                    document.OperationLoadStyleSheet(filename) )
            except IOError:
                qt4.QMessageBox.warning(self, "Veusz",
                                        "Unable to load default "
                                        "stylesheet '%s'" % filename)
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
            except IOError:
                qt4.QMessageBox.warning(self, "Veusz",
                                        "Unable to load default custom "
                                        "definitons '%s'" % filename)
            else:
                # reset any modified flag
                self.document.setModified(False)
                self.document.changeset = 0

    def slotAboutToShowEdit(self):
        """Enable/disable undo/redo menu items."""
        
        # enable distable, and add appropriate text to describe
        # the operation being undone/redone
        canundo = self.document.canUndo()
        undotext = 'Undo'
        if canundo:
            undotext = "%s %s" % (undotext, self.document.historyundo[-1].descr)
        self.vzactions['edit.undo'].setText(undotext)
        self.vzactions['edit.undo'].setEnabled(canundo)
        
        canredo = self.document.canRedo()
        redotext = 'Redo'
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
        dialog = PreferencesDialog(self)
        dialog.exec_()

    def slotEditStylesheet(self):
        dialog = StylesheetDialog(self, self.document)
        self.showDialog(dialog)
        return dialog
        
    def slotEditCustom(self):
        dialog = CustomDialog(self, self.document)
        self.showDialog(dialog)
        return dialog

    def definePlugins(self, pluginlist, actions, menuname):
        """Create menu items and actions for plugins.

        pluginlist: list of plugin classes
        actions: dict of actions to add new actions to
        menuname: string giving prefix for new menu entries (inside actions)
        """

        menu = []
        for pluginkls in pluginlist:
            def loaddialog(pluginkls=pluginkls):
                """Load plugin dialog"""
                handlePlugin(self, self.document, pluginkls)

            actname = menuname + '.' + '.'.join(pluginkls.name)
            text = pluginkls.menu[-1]
            if pluginkls.has_parameters:
                text += '...'
            actions[actname] = utils.makeAction(
                self,
                pluginkls.description_short,
                text,
                loaddialog)

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
            'file.new':
                a(self, 'New document', '&New',
                  self.slotFileNew,
                  icon='kde-document-new', key='Ctrl+N'),
            'file.open':
                a(self, 'Open a document', '&Open...',
                  self.slotFileOpen,
                  icon='kde-document-open', key='Ctrl+O'),
            'file.save':
                a(self, 'Save the document', '&Save',
                  self.slotFileSave,
                  icon='kde-document-save', key='Ctrl+S'),
            'file.saveas':
                a(self, 'Save the current graph under a new name',
                  'Save &As...', self.slotFileSaveAs,
                  icon='kde-document-save-as'),
            'file.print':
                a(self, 'Print the document', '&Print...',
                  self.slotFilePrint,
                  icon='kde-document-print', key='Ctrl+P'),
            'file.export':
                a(self, 'Export the current page', '&Export...',
                  self.slotFileExport,
                  icon='kde-document-export'),
            'file.close':
                a(self, 'Close current window', 'Close Window',
                  self.slotFileClose,
                  icon='kde-window-close', key='Ctrl+W'),
            'file.quit':
                a(self, 'Exit the program', '&Quit',
                  self.slotFileQuit,
                  icon='kde-application-exit', key='Ctrl+Q'),
            
            'edit.undo':
                a(self, 'Undo the previous operation', 'Undo',
                  self.slotEditUndo,
                  icon='kde-edit-undo',  key='Ctrl+Z'),
            'edit.redo':
                a(self, 'Redo the previous operation', 'Redo',
                  self.slotEditRedo,
                  icon='kde-edit-redo', key='Ctrl+Shift+Z'),
            'edit.prefs':
                a(self, 'Edit preferences', 'Preferences...',
                  self.slotEditPreferences,
                  icon='veusz-edit-prefs'),
            'edit.custom':
                a(self, 'Edit custom functions and constants',
                  'Custom definitions...',
                  self.slotEditCustom,
                  icon='veusz-edit-custom'),

            'edit.stylesheet':
                a(self,
                  'Edit stylesheet to change default widget settings',
                  'Default styles...',
                  self.slotEditStylesheet, icon='settings_stylesheet'),

            'view.edit':
                a(self, 'Show or hide edit window', 'Edit window',
                  None, checkable=True),
            'view.props':
                a(self, 'Show or hide property window', 'Properties window',
                  None, checkable=True),
            'view.format':
                a(self, 'Show or hide formatting window', 'Formatting window',
                  None, checkable=True),
            'view.console':
                a(self, 'Show or hide console window', 'Console window',
                  None, checkable=True),
            'view.datanav':
                a(self, 'Show or hide data navigator window', 'Data navigator window',
                  None, checkable=True),

            'view.maintool':
                a(self, 'Show or hide main toolbar', 'Main toolbar',
                  None, checkable=True),
            'view.datatool':
                a(self, 'Show or hide data toolbar', 'Data toolbar',
                  None, checkable=True),
            'view.viewtool':
                a(self, 'Show or hide view toolbar', 'View toolbar',
                  None, checkable=True),
            'view.edittool':
                a(self, 'Show or hide editing toolbar', 'Editing toolbar',
                  None, checkable=True),
            'view.addtool':
                a(self, 'Show or hide insert toolbar', 'Insert toolbar',
                  None, checkable=True),
            
            'data.import':
                a(self, 'Import data into Veusz', '&Import...',
                  self.slotDataImport, icon='kde-vzdata-import'),
            'data.edit':
                a(self, 'Edit existing datasets', '&Edit...',
                  self.slotDataEdit, icon='kde-edit-veuszedit'),
            'data.create':
                a(self, 'Create new datasets', '&Create...',
                  self.slotDataCreate, icon='kde-dataset-new-veuszedit'),
            'data.create2d':
                a(self, 'Create new 2D datasets', 'Create &2D...',
                  self.slotDataCreate2D, icon='kde-dataset2d-new-veuszedit'),
            'data.capture':
                a(self, 'Capture remote data', 'Ca&pture...',
                  self.slotDataCapture, icon='veusz-capture-data'),
            'data.histogram':
                a(self, 'Histogram data', '&Histogram...',
                  self.slotDataHistogram, icon='button_bar'),
            'data.reload':
                a(self, 'Reload linked datasets', '&Reload',
                  self.slotDataReload, icon='kde-view-refresh'),

            'help.home':
                a(self, 'Go to the Veusz home page on the internet',
                  'Home page', self.slotHelpHomepage),
            'help.project':
                a(self, 'Go to the Veusz project page on the internet',
                  'GNA Project page', self.slotHelpProjectPage),
            'help.bug':
                a(self, 'Report a bug on the internet',
                  'Suggestions and bugs', self.slotHelpBug),
            'help.about':
                a(self, 'Displays information about the program', 'About...',
                  self.slotHelpAbout, icon='veusz')
            }

        # create main toolbar
        tb = self.maintoolbar = qt4.QToolBar("Main toolbar - Veusz", self)
        iconsize = setdb['toolbar_size']
        tb.setIconSize(qt4.QSize(iconsize, iconsize))
        tb.setObjectName('veuszmaintoolbar')
        self.addToolBar(qt4.Qt.TopToolBarArea, tb)
        utils.addToolbarActions(tb, self.vzactions, 
                                ('file.new', 'file.open', 'file.save',
                                 'file.print', 'file.export'))

        # data toolbar
        tb = self.datatoolbar = qt4.QToolBar("Data toolbar - Veusz", self)
        tb.setIconSize(qt4.QSize(iconsize, iconsize))
        tb.setObjectName('veuszdatatoolbar')
        self.addToolBar(qt4.Qt.TopToolBarArea, tb)
        utils.addToolbarActions(tb, self.vzactions,
                                ('data.import', 'data.edit',
                                 'data.create', 'data.capture',
                                 'data.reload'))

        # menu structure
        filemenu = [
            'file.new', 'file.open',
            ['file.filerecent', 'Open &Recent', []],
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
            ['edit.select', '&Select', []],
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
            ['view.viewwindows', '&Windows', viewwindowsmenu],
            ''
            ]
        insertmenu = [
            ]

        # load dataset plugins and create menu
        datapluginsmenu = self.definePlugins( plugins.datasetpluginregistry,
                                              self.vzactions, 'data.ops' )

        datamenu = [
            ['data.ops', '&Operations', datapluginsmenu],
            'data.import', 'data.edit', 'data.create',
            'data.create2d', 'data.capture', 'data.histogram',
            'data.reload',
            ]
        helpmenu = [
            'help.home', 'help.project', 'help.bug',
            '',
            'help.about'
            ]

        # load tools plugins and create menu
        toolsmenu = self.definePlugins( plugins.toolspluginregistry,
                                        self.vzactions, 'tools' )

        menus = [
            ['file', '&File', filemenu],
            ['edit', '&Edit', editmenu],
            ['view', '&View', viewmenu],
            ['insert', '&Insert', insertmenu],
            ['data', '&Data', datamenu],
            ['tools', '&Tools', toolsmenu],
            ['help', '&Help', helpmenu],
            ]

        self.menus = {}
        utils.constructMenus(self.menuBar(), self.menus, menus, self.vzactions)

    def _setPickerFont(self, label):
        f = label.font()
        f.setBold(True)
        f.setPointSizeF(f.pointSizeF() * 1.2)
        label.setFont(f)

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
            self.connect(a, qt4.SIGNAL('triggered()'), fn)

        # needs to update state every time menu is shown
        self.connect(self.menus['view.viewwindows'],
                     qt4.SIGNAL('aboutToShow()'),
                     self.slotAboutToShowViewWindow)

    def slotAboutToShowViewWindow(self):
        """Enable/disable View->Window item check boxes."""

        for win, act, fn in self.viewwinfns:
            act.setChecked(not win.isHidden())

    def showDialog(self, dialog):
        """Show dialog given."""
        self.connect(dialog, qt4.SIGNAL('dialogFinished'), self.deleteDialog)
        self.dialogs.append(dialog)
        dialog.show()

    def deleteDialog(self, dialog):
        """Remove dialog from list of dialogs."""
        try:
            idx = self.dialogs.index(dialog)
            del self.dialogs[idx]
        except ValueError:
            pass

    def slotDataImport(self):
        """Display the import data dialog."""
        dialog = importdialog.ImportDialog(self, self.document)
        self.showDialog(dialog)
        return dialog

    def slotDataEdit(self, editdataset=None):
        """Edit existing datasets.

        If editdataset is set to a dataset name, edit this dataset
        """
        dialog = dataeditdialog.DataEditDialog(self, self.document)
        self.showDialog(dialog)
        if editdataset is not None:
            dialog.selectDataset(editdataset)
        return dialog

    def slotDataCreate(self):
        """Create new datasets."""
        dialog = DataCreateDialog(self, self.document)
        self.showDialog(dialog)
        return dialog

    def slotDataCreate2D(self):
        """Create new datasets."""
        dialog = DataCreate2DDialog(self, self.document)
        self.showDialog(dialog)
        return dialog

    def slotDataCapture(self):
        """Capture remote data."""
        dialog = CaptureDialog(self.document, self)
        self.showDialog(dialog)
        return dialog

    def slotDataHistogram(self):
        """Histogram data."""
        dialog = HistoDataDialog(self, self.document)
        self.showDialog(dialog)
        return dialog

    def slotDataReload(self):
        """Reload linked datasets."""
        dialog = ReloadData(self.document, self)
        self.showDialog(dialog)
        return dialog

    def slotHelpHomepage(self):
        """Go to the veusz homepage."""
        qt4.QDesktopServices.openUrl(qt4.QUrl('http://home.gna.org/veusz/'))

    def slotHelpProjectPage(self):
        """Go to the veusz project page."""
        qt4.QDesktopServices.openUrl(qt4.QUrl('http://gna.org/projects/veusz/'))

    def slotHelpBug(self):
        """Go to the veusz bug page."""
        qt4.QDesktopServices.openUrl(
            qt4.QUrl('https://gna.org/bugs/?group=veusz') )

    def slotHelpAbout(self):
        """Show about dialog."""
        AboutDialog(self).exec_()

    def queryOverwrite(self):
        """Do you want to overwrite the current document.

        Returns qt4.QMessageBox.(Yes,No,Cancel)."""

        # include filename in mesage box if we can
        filetext = ''
        if self.filename:
            filetext = " '%s'" % os.path.basename(self.filename)

        # show message box
        mb = qt4.QMessageBox("Save file?",
                             "Document%s was modified. Save first?" % filetext,
                             qt4.QMessageBox.Warning,
                             qt4.QMessageBox.Yes | qt4.QMessageBox.Default,
                             qt4.QMessageBox.No,
                             qt4.QMessageBox.Cancel | qt4.QMessageBox.Escape,
                             self)
        mb.setButtonText(qt4.QMessageBox.Yes, "&Save")
        mb.setButtonText(qt4.QMessageBox.No, "&Discard")
        mb.setButtonText(qt4.QMessageBox.Cancel, "&Cancel")
        return mb.exec_()

    def closeEvent(self, event):
        """Before closing, check whether we need to save first."""

        # if the document has been modified then query user for saving
        if self.document.isModified():
            v = self.queryOverwrite()
            if v == qt4.QMessageBox.Cancel:
                event.ignore()
                return
            elif v == qt4.QMessageBox.Yes:
                self.slotFileSave()

        # store working directory
        setdb['dirname'] = self.dirname
        setdb['dirname_export'] = self.dirname_export

        # store the current geometry in the settings database
        geometry = ( self.x(), self.y(), self.width(), self.height() )
        setdb['geometry_mainwindow'] = geometry

        # store docked windows
        data = str(self.saveState())
        setdb['geometry_mainwindowstate'] = data

        # save current setting db
        setdb.writeSettings()

        event.accept()

    def setupWindowGeometry(self):
        """Restoring window geometry if possible."""

        # count number of main windows shown
        nummain = 0
        for w in qt4.qApp.topLevelWidgets():
            if isinstance(w, qt4.QMainWindow):
                nummain += 1

        # if we can restore the geometry, do so
        if 'geometry_mainwindow' in setdb:
            geometry = setdb['geometry_mainwindow']
            self.resize( qt4.QSize(geometry[2], geometry[3]) )
            if nummain <= 1:
                self.move( qt4.QPoint(geometry[0], geometry[1]) )

        # restore docked window geometry
        if 'geometry_mainwindowstate' in setdb:
            b = qt4.QByteArray(setdb['geometry_mainwindowstate'])
            self.restoreState(b)

    def slotFileNew(self):
        """New file."""
        self.CreateWindow()

    def slotFileSave(self):
        """Save file."""

        if self.filename == '':
            self.slotFileSaveAs()
        else:
            # show busy cursor
            qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )
            try:
                ofile = open(self.filename, 'w')
                self.document.saveToFile(ofile)
                self.updateStatusbar("Saved to %s" % self.filename)
            except IOError, e:
                qt4.QApplication.restoreOverrideCursor()
                qt4.QMessageBox.critical(self, "Cannot save document",
                                         "Cannot save document as '%s'\n"
                                         "\n%s (error %i)" %
                                         (self.filename,  e.strerror, e.errno))
            else:
                # restore the cursor
                qt4.QApplication.restoreOverrideCursor()
                
    def updateTitlebar(self):
        """Put the filename into the title bar."""
        if self.filename == '':
            self.setWindowTitle('Untitled - Veusz')
        else:
            self.setWindowTitle( "%s - Veusz" %
                                 os.path.basename(self.filename) )

    def plotQueueChanged(self, incr):
        self.plotqueuecount += incr
        text = u'•' * self.plotqueuecount
        self.plotqueuelabel.setText(text)

    def _fileSaveDialog(self, filetype, filedescr, dialogtitle):
        """A generic file save dialog for exporting / saving."""
        
        fd = qt4.QFileDialog(self, dialogtitle)
        fd.setDirectory(self.dirname)
        fd.setFileMode( qt4.QFileDialog.AnyFile )
        fd.setAcceptMode( qt4.QFileDialog.AcceptSave )
        fd.setFilter( "%s (*.%s)" % (filedescr, filetype) )

        # okay was selected (and is okay to overwrite if it exists)
        if fd.exec_() == qt4.QDialog.Accepted:
            # save directory for next time
            self.dirname = fd.directory().absolutePath()
            # update the edit box
            filename = unicode( fd.selectedFiles()[0] )
            if os.path.splitext(filename)[1] == '':
                filename += '.' + filetype

            return filename
        return None

    def _fileOpenDialog(self, filetype, filedescr, dialogtitle):
        """Display an open dialog and return a filename."""
        
        fd = qt4.QFileDialog(self, dialogtitle)
        fd.setDirectory(self.dirname)
        fd.setFileMode( qt4.QFileDialog.ExistingFile )
        fd.setAcceptMode( qt4.QFileDialog.AcceptOpen )
        fd.setFilter( "%s (*.%s)" % (filedescr, filetype) )
        
        # if the user chooses a file
        if fd.exec_() == qt4.QDialog.Accepted:
            # save directory for next time
            self.dirname = fd.directory().absolutePath()

            filename = unicode( fd.selectedFiles()[0] )
            try:
                open(filename)
            except IOError, e:
                qt4.QMessageBox.critical(self, "Unable to open file",
                                         "Unable to open file '%s'\n'%s'" %
                                         (filename, unicode(e)))
                return None
            return filename
        return None

    def slotFileSaveAs(self):
        """Save As file."""

        filename = self._fileSaveDialog('vsz', 'Veusz script files', 'Save as')
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

    class _unsafeCmdMsgBox(qt4.QMessageBox):
        """Show document is unsafe."""
        def __init__(self, window, filename):
            qt4.QMessageBox.__init__(self, "Unsafe code in document",
                                     "The document '%s' contains potentially "
                                     "unsafe code which may damage your "
                                     "computer or data. Please check that the "
                                     "file comes from a "
                                     "trusted source." % filename,
                                     qt4.QMessageBox.Warning,
                                     qt4.QMessageBox.Yes,
                                     qt4.QMessageBox.No | qt4.QMessageBox.Default,
                                     qt4.QMessageBox.NoButton,
                                     window)
            self.setButtonText(qt4.QMessageBox.Yes, "C&ontinue anyway")
            self.setButtonText(qt4.QMessageBox.No, "&Stop loading")
 
    class _unsafeVeuszCmdMsgBox(qt4.QMessageBox):
        """Show document has unsafe Veusz commands."""
        def __init__(self, window):
            qt4.QMessageBox.__init__(self, 'Unsafe Veusz commands',
                                     'This Veusz document contains potentially'
                                     ' unsafe Veusz commands for Saving, '
                                     'Exporting or Printing. Please check that the'
                                     ' file comes from a trusted source.',
                                     qt4.QMessageBox.Warning,
                                     qt4.QMessageBox.Yes,
                                     qt4.QMessageBox.No | qt4.QMessageBox.Default,
                                     qt4.QMessageBox.NoButton,
                                     window)
            self.setButtonText(qt4.QMessageBox.Yes, "C&ontinue anyway")
            self.setButtonText(qt4.QMessageBox.No, "&Ignore command")

    def openFileInWindow(self, filename):
        """Actually do the work of loading a new document.
        """

        # FIXME: This function suffers from spaghetti code
        # it needs splitting up into bits to make it clearer
        # the steps are fairly well documented below, however
        #####################################################

        qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )

        # read script
        try:
            script = open(filename, 'rU').read()
        except IOError, e:
            qt4.QApplication.restoreOverrideCursor()
            qt4.QMessageBox.warning(self, "Cannot open document",
                                    "Cannot open the document '%s'\n"
                                    "\n%s (error %i)" % (filename,
                                                         e.strerror, e.errno))
            return

        # check code for any security issues
        ignore_unsafe = setting.transient_settings['unsafe_mode']
        if not ignore_unsafe:
            errors = utils.checkCode(script, securityonly=True)
            if errors:
                qt4.QApplication.restoreOverrideCursor()
                if ( self._unsafeCmdMsgBox(self, filename).exec_() ==
                     qt4.QMessageBox.No ):
                    return
                ignore_unsafe = True # allow unsafe veusz commands below

        # set up environment to run script
        env = self.document.eval_context.copy()
        interface = document.CommandInterface(self.document)

        # allow safe commands as-is
        for cmd in interface.safe_commands:
            env[cmd] = getattr(interface, cmd)

        # define root node
        env['Root'] = interface.Root

        # wrap "unsafe" commands with a message box to check the user
        safenow = [ignore_unsafe]
        def _unsafeCaller(func):
            def wrapped(*args, **argsk):
                if not safenow[0]:
                    qt4.QApplication.restoreOverrideCursor()
                    if ( self._unsafeVeuszCmdMsgBox(self).exec_() ==
                         qt4.QMessageBox.No ):
                        return
                safenow[0] = True
                func(*args, **argsk)
            return wrapped
        for name in interface.unsafe_commands:
            env[name] = _unsafeCaller(getattr(interface, name))
                               
        # save stdout and stderr, then redirect to console
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = self.console.con_stdout
        sys.stderr = self.console.con_stderr

        # get ready to load document
        env['__file__'] = os.path.abspath(filename)
        self.document.wipe()
        self.document.suspendUpdates()

        # allow import to happen relative to loaded file
        interface.AddImportPath( os.path.dirname(os.path.abspath(filename)) )

        try:
            # actually run script text
            exec script in env
        except Exception, e:
            # need to remember to restore stdout, stderr
            sys.stdout, sys.stderr = stdout, stderr
            
            # display error dialog if there is an error loading
            qt4.QApplication.restoreOverrideCursor()
            self.document.enableUpdates()
            i = sys.exc_info()
            backtrace = traceback.format_exception( *i )
            d = ErrorLoadingDialog(self, filename, str(e), ''.join(backtrace))
            d.exec_()
            return

        # need to remember to restore stdout, stderr
        sys.stdout, sys.stderr = stdout, stderr

        # document is loaded
        self.document.enableUpdates()
        self.document.setModified(False)
        self.document.clearHistory()

        # remember file for recent list
        self.addRecentFile(filename)

        # let the main window know
        self.filename = filename
        self.updateTitlebar()
        self.updateStatusbar("Opened %s" % filename)

        # use current directory of file if not using cwd mode
        if not setdb['dirname_usecwd']:
            self.dirname = os.path.dirname( os.path.abspath(filename) )
            self.dirname_export = self.dirname

        # notify cmpts which need notification that doc has finished opening
        self.emit(qt4.SIGNAL("documentopened"))
        qt4.QApplication.restoreOverrideCursor()

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

        filename = self._fileOpenDialog('vsz', 'Veusz script files', 'Open')
        if filename:
            self.openFile(filename)
        
    def populateRecentFiles(self):
        """Populate the recently opened files menu with a list of
        recently opened files"""

        menu = self.menus["file.filerecent"]
        menu.clear()

        newMenuItems = []
        if setdb['main_recentfiles']:
            files = [f for f in setdb['main_recentfiles']
                     if os.path.isfile(f)]
            self._openRecentFunctions = []

            # add each recent file to menu
            for i, path in enumerate(files):

                def fileOpener(filename=path):
                    self.openFile(filename)

                self._openRecentFunctions.append(fileOpener)
                newMenuItems.append(('filerecent%i' % i, 'Open File %s' % path,
                                     os.path.basename(path),
                                     'file.filerecent', fileOpener,
                                     '', False, ''))

            menu.setEnabled(True)
            self.recentFileActions = utils.populateMenuToolbars(
                newMenuItems, self.maintoolbar, self.menus)
        else:
            menu.setEnabled(False)
    
    def slotFileExport(self):
        """Export the graph."""

        # check there is a page
        if self.document.getNumberPages() == 0:
            qt4.QMessageBox.warning(self, "Veusz",
                                    "No pages to export")
            return

        # File types we can export to in the form ([extensions], Name)
        fd = qt4.QFileDialog(self, 'Export page')
        fd.setDirectory( self.dirname_export )

        fd.setFileMode( qt4.QFileDialog.AnyFile )
        fd.setAcceptMode( qt4.QFileDialog.AcceptSave )

        # Create a mapping between a format string and extensions
        filtertoext = {}
        # convert extensions to filter
        exttofilter = {}
        filters = []
        # a list of extensions which are allowed
        validextns = []
        formats = document.Export.formats
        for extns, name in formats:
            extensions = " ".join(["*." + item for item in extns])
            # join eveything together to make a filter string
            filterstr = '%s (%s)' % (name, extensions)
            filtertoext[filterstr] = extns
            for e in extns:
                exttofilter[e] = filterstr
            filters.append(filterstr)
            validextns += extns
        fd.setNameFilters(filters)

        # restore last format if possible
        try:
            filt = setdb['export_lastformat']
            fd.selectNameFilter(filt)
            extn = formats[filters.index(filt)][0][0]
        except (KeyError, IndexError, ValueError):
            extn = 'pdf'
            fd.selectNameFilter( exttofilter[extn] )

        if self.filename:
            # try to convert current filename to export name
            filename = os.path.basename(self.filename)
            filename = os.path.splitext(filename)[0] + '.' + extn
            fd.selectFile(filename)
        
        if fd.exec_() == qt4.QDialog.Accepted:
            # save directory for next time
            self.dirname_export = fd.directory().absolutePath()

            filterused = str(fd.selectedFilter())
            setdb['export_lastformat'] = filterused

            chosenextns = filtertoext[filterused]
            
            # show busy cursor
            qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )

            filename = unicode( fd.selectedFiles()[0] )
            
            # Add a default extension if one isn't supplied
            # this is the extension without the dot
            ext = os.path.splitext(filename)[1][1:]
            if (ext not in validextns) and (ext not in chosenextns):
                filename += "." + chosenextns[0]

            e = document.Export( self.document,
                                 filename,
                                 self.plot.getPageNumber(),
                                 bitmapdpi=setdb['export_DPI'],
                                 pdfdpi=setdb['export_DPI_PDF'],
                                 antialias=setdb['export_antialias'],
                                 color=setdb['export_color'],
                                 quality=setdb['export_quality'],
                                 backcolor=setdb['export_background'] )
            try:
                e.export()
            except (IOError, RuntimeError), inst:
                qt4.QMessageBox.critical(self, "Veusz",
                                         "Error exporting file:\n%s" % inst)

            # restore the cursor
            qt4.QApplication.restoreOverrideCursor()

    def slotFilePrint(self):
        """Print the document."""

        if self.document.getNumberPages() == 0:
            qt4.QMessageBox.warning(self, "Veusz",
                                    "No pages to print")
            return

        prnt = qt4.QPrinter(qt4.QPrinter.HighResolution)
        prnt.setColorMode(qt4.QPrinter.Color)
        prnt.setCreator('Veusz %s' % utils.version())
        prnt.setDocName(self.filename)

        dialog = qt4.QPrintDialog(prnt, self)
        dialog.setMinMax(1, self.document.getNumberPages())
        if dialog.exec_():
            # get page range
            if dialog.printRange() == qt4.QAbstractPrintDialog.PageRange:
                # page range
                minval, maxval = dialog.fromPage(), dialog.toPage()
            else:
                # all pages
                minval, maxval = 1, self.document.getNumberPages()

            # pages are relative to zero
            minval -= 1
            maxval -= 1

            # reverse or forward order
            if prnt.pageOrder() == qt4.QPrinter.FirstPageFirst:
                pages = range(minval, maxval+1)
            else:
                pages = range(maxval, minval-1, -1)

            # if more copies are requested
            pages *= prnt.numCopies()

            # do the printing
            self.document.printTo( prnt, pages )

    def slotModifiedDoc(self, ismodified):
        """Disable certain actions if document is not modified."""

        # enable/disable file, save menu item
        self.vzactions['file.save'].setEnabled(ismodified)

    def slotFileClose(self):
        """File close window chosen."""
        self.close()

    def slotFileQuit(self):
        """File quit chosen."""
        qt4.qApp.closeAllWindows()
        
    def slotUpdatePage(self, number):
        """Update page number when the plot window says so."""

        np = self.document.getNumberPages()
        if np == 0:
            self.pagelabel.setText("No pages")
        else:
            self.pagelabel.setText("Page %i/%i" % (number+1, np))

    def slotUpdateAxisValues(self, values):
        """Update the position where the mouse is relative to the axes."""

        if values:
            # construct comma separated text representing axis values
            valitems = []
            for name, val in values.iteritems():
                valitems.append('%s=%#.4g' % (name, val))
            valitems.sort()
            self.axisvalueslabel.setText(', '.join(valitems))
        else:
            self.axisvalueslabel.setText('No position')

    def slotPickerEnabled(self, enabled):
        if enabled:
            self.pickerlabel.setText('No point selected')
            self.pickerlabel.show()
        else:
            self.pickerlabel.hide()

    def slotUpdatePickerLabel(self, info):
        """Display the picked point"""
        xv, yv = info.coords
        xn, yn = info.labels
        ix = str(info.index)
        if ix:
            ix = '[' + ix + ']'
        t = '%s: %s%s = %0.5g, %s%s = %0.5g' % (
                info.widget.name, xn, ix, xv, yn, ix, yv)
        self.pickerlabel.setText(t)

    def slotAllowedImportsDoc(self, module, names):
        """Are allowed imports?"""

        d = SafetyImportDialog(self, module, names)
        d.exec_()
