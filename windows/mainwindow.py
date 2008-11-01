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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

"""Implements the main window of the application."""

import os.path
import sys
import traceback

import veusz.qtall as qt4

import veusz.document as document
import veusz.utils as utils
import veusz.setting as setting

import consolewindow
import plotwindow
import treeeditwindow

from veusz.dialogs.aboutdialog import AboutDialog
from veusz.dialogs.reloaddata import ReloadData
from veusz.dialogs.datacreate import DataCreateDialog
from veusz.dialogs.datacreate2d import DataCreate2DDialog
from veusz.dialogs.preferences import PreferencesDialog
from veusz.dialogs.errorloading import ErrorLoadingDialog
import veusz.dialogs.importdialog as importdialog
import veusz.dialogs.dataeditdialog as dataeditdialog

class MainWindow(qt4.QMainWindow):
    """ The main window class for the application."""

    windows = []
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

        # try to select first graph of first page
        win.treeedit.doInitialWidgetSelect()
            
        cls.windows.append(win)
        return win

    CreateWindow = classmethod(CreateWindow)

    def __init__(self, *args):
        qt4.QMainWindow.__init__(self, *args)

        self.setWindowIcon( utils.getIcon('veusz.png') )

        self.document = document.Document()

        self.setAcceptDrops(True)

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

        # make the console window a dock
        self.console = consolewindow.ConsoleWindow(self.document,
                                                   self)
        self.console.hide()
        self.interpreter = self.console.interpreter
        self.addDockWidget(qt4.Qt.BottomDockWidgetArea, self.console)

        # keep page number up to date
        statusbar = self.statusbar = qt4.QStatusBar(self)
        self.setStatusBar(statusbar)
        self.updateStatusbar('Ready')
        self.pagelabel = qt4.QLabel(statusbar)
        statusbar.addWidget(self.pagelabel)
        self.axisvalueslabel = qt4.QLabel(statusbar)
        statusbar.addWidget(self.axisvalueslabel)
        self.axisvalueslabel.show()

        self.dirname = os.getcwd()
        self.exportDir = os.getcwd()
        
        self.connect( self.plot, qt4.SIGNAL("sigUpdatePage"),
                      self.slotUpdatePage )
        self.connect( self.plot, qt4.SIGNAL("sigAxisValuesFromMouse"),
                      self.slotUpdateAxisValues )
        
        # disable save if already saved
        self.connect( self.document, qt4.SIGNAL("sigModified"),
                      self.slotModifiedDoc )
        # if the treeeditwindow changes the page, change the plot window
        self.connect( self.treeedit, qt4.SIGNAL("sigPageChanged"),
                      self.plot.setPageNumber )

        # if a widget in the plot window is clicked by the user
        self.connect( self.plot, qt4.SIGNAL("sigWidgetClicked"),
                      self.treeedit.selectWidget )
        self.connect( self.treeedit, qt4.SIGNAL("widgetSelected"),
                      self.plot.selectedWidget )

        # enable/disable undo/redo
        self.connect(self.menus['edit'], qt4.SIGNAL('aboutToShow()'),
                     self.slotAboutToShowEdit)

        #Get the list of recently opened files
        self.populateRecentFiles()
        self.setupWindowGeometry()
        self.defineViewWindowMenu()

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

    def slotAboutToShowEdit(self):
        """Enable/disable undo/redo menu items."""
        
        # enable distable, and add appropriate text to describe
        # the operation being undone/redone
        canundo = self.document.canUndo()
        undotext = 'Undo'
        if canundo:
            undotext = "%s %s" % (undotext, self.document.historyundo[-1].descr)
        self.actions['editundo'].setText(undotext)
        self.actions['editundo'].setEnabled(canundo)
        
        canredo = self.document.canRedo()
        redotext = 'Redo'
        if canredo:
            redotext = "%s %s" % (redotext, self.document.historyredo[-1].descr)
        self.actions['editredo'].setText(redotext)
        self.actions['editredo'].setEnabled(canredo)
        
    def slotEditUndo(self):
        """Undo the previous operation"""
        if self.document.canUndo():
            self.document.undoOperation()
        
    def slotEditRedo(self):
        """Redo the previous operation"""
        if self.document.canRedo():
            self.document.redoOperation()

    def slotEditPreferences(self):
        dialog = PreferencesDialog(self)
        dialog.exec_()
        
    def _defineMenus(self):
        """Initialise the menus and toolbar."""

        # create toolbar
        self.maintoolbar = qt4.QToolBar("Main toolbar - Veusz", self)
        self.maintoolbar.setObjectName('veuszmaintoolbar')
        self.addToolBar(qt4.Qt.TopToolBarArea, self.maintoolbar)

        # add main menus
        menus = [
            ('file', '&File'),
            ('edit', '&Edit'),
            ('view', '&View'),
            ('insert', '&Insert'),
            ('data', '&Data'),
            ('help', '&Help')
            ]

        self.menus = {}
        for menuid, text in menus:
            menu = self.menuBar().addMenu(text)
            self.menus[menuid] = menu

        # items for main menus
        # Items are: Lookup id, description, menu text, which menu,
        #  Slot, Icon (or ''), whether to add to toolbar,
        #  Keyboard shortcut (or '')
        # For menus wih submenus slot should be replaced by a list of
        # submenus items of the dame form where the menu will be of the form
        # menuid.itemid
        items = (
            ('filenew', 'New document', '&New', 'file',
             self.slotFileNew, 'stock-new.png', True, 'Ctrl+N'),
            ('fileopen', 'Open a document', '&Open...', 'file',
             self.slotFileOpen, 'stock-open.png', True, 'Ctrl+O'),
            #If we were looking for HIG goodness, there wouldn't be a submenu here
            ('filerecent', 'Open a recently edited document',
             'Open &Recent', 'file', [], '', False, ''),
            ('file', ),
            ('filesave', 'Save the document', '&Save', 'file',
             self.slotFileSave, 'stock-save.png', True, 'Ctrl+S'),
            ('filesaveas', 'Save the current graph under a new name',
             'Save &As...', 'file', self.slotFileSaveAs, 'stock-save-as.png',
             False, ''),
            ('file', ),
            ('fileprint', 'Print the document', '&Print...', 'file',
             self.slotFilePrint, 'stock-print.png', True, 'Ctrl+P'),
            ('fileexport', 'Export the current page', '&Export...', 'file',
             self.slotFileExport, 'stock-export.png', True, ''),
            ('fileexportstylesheet', 'Export stylesheet to file', 'Export stylesheet...', 'file',
             self.slotFileExportStyleSheet, '', False, ''), 
            ('fileimportstylesheet', 'Import stylesheet from file', 'Import stylesheet...', 'file',
             self.slotFileImportStyleSheet, '', False, ''), 
 
            ('file', ),
            ('fileclose', 'Close current window', 'Close Window', 'file',
             self.slotFileClose, '', False, 'Ctrl+W'),
            ('filequit', 'Exit the program', '&Quit', 'file',
             self.slotFileQuit, 'stock-quit.png', False, 'Ctrl+Q'),

            ('editundo', 'Undo the previous operation', 'Undo', 'edit',
             self.slotEditUndo, '', False,  'Ctrl+Z'),
            ('editredo', 'Redo the previous operation', 'Redo', 'edit',
             self.slotEditRedo, '', False, 'Ctrl+Shift+Z'),
            ('edit', ),
            ('editprefs', 'Edit preferences', 'Preferences...', 'edit',
             self.slotEditPreferences, '', False, ''),
            ('edit', ),

            ('viewwindows', 'Show or hide windows or toolbars',
             'Windows', 'view', [], '', False, ''),
            ('viewedit', 'Show or hide edit window', 'Edit window',
             'view.viewwindows', None, '', False, ''),
            ('viewprops', 'Show or hide property window', 'Properties window',
             'view.viewwindows', None, '', False, ''),
            ('viewformat', 'Show or hide formatting window', 'Formatting window',
             'view.viewwindows', None, '', False, ''),
            ('viewconsole', 'Show or hide console window', 'Console window',
             'view.viewwindows', None, '', False, ''),
            ('view.viewwindows', ),
            ('viewmaintool', 'Show or hide main toolbar', 'Main toolbar',
             'view.viewwindows', None, '', False, ''),
            ('viewviewtool', 'Show or hide view toolbar', 'View toolbar',
             'view.viewwindows', None, '', False, ''),
            ('viewedittool', 'Show or hide editing toolbar', 'Editing toolbar',
             'view.viewwindows', None, '', False, ''),
            ('view', ),
            
            ('dataimport', 'Import data into Veusz', '&Import...', 'data',
             self.slotDataImport, 'stock-import.png', False, ''),
            ('dataedit', 'Edit existing datasets', '&Edit...', 'data',
             self.slotDataEdit, 'stock-edit.png', False, ''),
            ('datacreate', 'Create new datasets', '&Create...', 'data',
             self.slotDataCreate, 'stock-new.png', False, ''),
            ('datacreate2d', 'Create new 2D datasets', 'Create &2D...', 'data',
             self.slotDataCreate2D, 'stock-new.png', False, ''),
            ('datareload', 'Reload linked datasets', '&Reload', 'data',
             self.slotDataReload, 'stock-refresh.png', False, ''),

            ('helphome', 'Go to the Veusz home page on the internet',
             'Home page', 'help', self.slotHelpHomepage, '', False, ''),
            ('helpproject', 'Go to the Veusz project page on the internet',
             'GNA Project page', 'help', self.slotHelpProjectPage, '',
             False, ''),
            ('helpbug', 'Report a bug on the internet',
             'Suggestions and bugs', 'help', self.slotHelpBug, '', False, ''),
            ('help', ),
            ('helpabout', 'Displays information about the program', 'About...',
             'help', self.slotHelpAbout, '', False, '')
            )
            
        self.actions = utils.populateMenuToolbars(items, self.maintoolbar,
                                                  self.menus)
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
        for win, act in ((self.treeedit, 'viewedit'),
                         (self.propdock, 'viewprops'),
                         (self.formatdock, 'viewformat'),
                         (self.console, 'viewconsole'),
                         (self.maintoolbar, 'viewmaintool'),
                         (self.treeedit.toolbar, 'viewedittool'),
                         (self.plot.viewtoolbar, 'viewviewtool')):

            a = self.actions[act]
            a.setCheckable(True)
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

    def slotDataImport(self):
        """Display the import data dialog."""
        dialog = importdialog.ImportDialog2(self, self.document)
        self.dialogs.append(dialog)
        dialog.show()

    def slotDataEdit(self):
        """Edit existing datasets."""
        dialog = dataeditdialog.DataEditDialog(self, self.document)
        self.dialogs.append(dialog)
        dialog.show()

    def slotDataCreate(self):
        """Create new datasets."""
        dialog = DataCreateDialog(self, self.document)
        self.dialogs.append(dialog)
        dialog.show()

    def slotDataCreate2D(self):
        """Create new datasets."""
        dialog = DataCreate2DDialog(self, self.document)
        self.dialogs.append(dialog)
        dialog.show()

    def slotDataReload(self):
        """Reload linked datasets."""
        d = ReloadData(self.document, self)
        d.exec_()

    def slotHelpHomepage(self):
        """Go to the veusz homepage."""
        import webbrowser
        webbrowser.open('http://home.gna.org/veusz/')

    def slotHelpProjectPage(self):
        """Go to the veusz project page."""
        import webbrowser
        webbrowser.open('http://gna.org/projects/veusz/')

    def slotHelpBug(self):
        """Go to the veusz bug page."""
        import webbrowser
        webbrowser.open('https://gna.org/bugs/?group=veusz')

    def slotHelpAbout(self):
        """Show about dialog."""
        d = AboutDialog(self)
        d.exec_()

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

        # store the current geometry in the settings database
        geometry = ( self.x(), self.y(), self.width(), self.height() )
        setting.settingdb['geometry_mainwindow'] = geometry

        # store docked windows
        data = str(self.saveState())
        setting.settingdb['geometry_mainwindowstate'] = data

        event.accept()

    def setupWindowGeometry(self):
        """Restoring window geometry if possible."""

        # count number of main windows shown
        nummain = 0
        for w in qt4.qApp.topLevelWidgets():
            if isinstance(w, qt4.QMainWindow):
                nummain += 1

        # if we can restore the geometry, do so
        if 'geometry_mainwindow' in setting.settingdb:
            geometry = setting.settingdb['geometry_mainwindow']
            self.resize( qt4.QSize(geometry[2], geometry[3]) )
            if nummain <= 1:
                self.move( qt4.QPoint(geometry[0], geometry[1]) )

        # restore docked window geometry
        if 'geometry_mainwindowstate' in setting.settingdb:
            b = qt4.QByteArray(setting.settingdb['geometry_mainwindowstate'])
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
                qt4.QMessageBox("Cannot save document",
                                "Cannot save document as '%s'\n"
                                "\n%s (error %i)" % (self.filename,
                                                     e.strerror, e.errno),
                                qt4.QMessageBox.Critical,
                                qt4.QMessageBox.Ok,
                                qt4.QMessageBox.NoButton,
                                qt4.QMessageBox.NoButton,
                                self).exec_()
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
            self.dirname = fd.directory()
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
            self.dirname = fd.directory()

            filename = unicode( fd.selectedFiles()[0] )
            try:
                open(filename)
            except IOError, e:
                qt4.QMessageBox("Unable to open file",
                                "Unable to open file '%s'\n'%s'" % (filename, str(e)),
                                qt4.QMessageBox.Critical,
                                qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                                qt4.QMessageBox.NoButton,
                                qt4.QMessageBox.NoButton,
                                self).exec_()
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
            script = open(filename, 'r').read()
        except IOError, e:
            qt4.QApplication.restoreOverrideCursor()
            qt4.QMessageBox("Cannot open document",
                            "Cannot open the document '%s'\n"
                            "\n%s (error %i)" % (filename,
                                                 e.strerror, e.errno),
                            qt4.QMessageBox.Warning,
                            qt4.QMessageBox.Ok,
                            qt4.QMessageBox.NoButton,
                            qt4.QMessageBox.NoButton,
                            self).exec_()
            return

        # check code for any security issues
        ignore_unsafe = setting.transient_settings['unsafe_mode']
        if not ignore_unsafe:
            errors = utils.checkCode(script)
            if errors is not None:
                qt4.QApplication.restoreOverrideCursor()
                if self._unsafeCmdMsgBox(self, filename).exec_() == \
                   qt4.QMessageBox.No:
                    return
                ignore_unsafe = True # allow unsafe veusz commands below

        # set up environment to run script
        env = utils.veusz_eval_context.copy()
        interface = document.CommandInterface(self.document)

        # allow safe commands as-is
        for cmd in interface.safe_commands:
            env[cmd] = getattr(interface, cmd)

        # wrap "unsafe" commands with a message box to check the user
        safenow = [ignore_unsafe]
        def _unsafeCaller(func):
            def wrapped(*args, **argsk):
                if not safenow[0]:
                    qt4.QApplication.restoreOverrideCursor()
                    if self._unsafeVeuszCmdMsgBox(self).exec_() == \
                           qt4.QMessageBox.No:
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

        # let the main window know
        self.filename = filename
        self.updateTitlebar()
        self.updateStatusbar("Opened %s" % filename)
        qt4.QApplication.restoreOverrideCursor()

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
        if setting.settingdb['main_recentfiles']:
            files = setting.settingdb['main_recentfiles']
            files = files[:10]
            self._openRecentFunctions = []
            for i, path in enumerate(files):

                #Surely there is an easier way to do this?
                def fileOpenerFunction(filename):
                    path=filename
                    def f():
                        self.openFile(path)
                    return f
                f = fileOpenerFunction(path)
                self._openRecentFunctions.append(f)
                
                newMenuItems.append(('filerecent%i'%i, 'Open File %s'%path,
                                     os.path.basename(path),
                                     'file.filerecent', f,
                                     '', False, ''))

            menu.setEnabled(True)
            self.recentFileActions = utils.populateMenuToolbars(newMenuItems,
                                                                self.maintoolbar,
                                                                 self.menus)
        else:
            menu.setEnabled(False)
                
    
    def slotFileExport(self):
        """Export the graph."""

        # check there is a page
        if self.document.getNumberPages() == 0:
                qt4.QMessageBox("Veusz",
                                "No pages to export",
                                qt4.QMessageBox.Warning,
                                qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                                qt4.QMessageBox.NoButton,
                                qt4.QMessageBox.NoButton,
                                self).exec_()
                return

        # File types we can export to in the form ([extensions], Name)
        formats = [(["eps"], "Encapsulated Postscript"),
                   (["png"], "Portable Network Graphics"),
                   (["jpg"], "Jpeg bitmap format"),
                   (["bmp"], "Windows bitmap format"),
                   (["pdf"], "Portable Document Format"),
                   (["svg"], "Scalable Vector Graphics")]

        fd = qt4.QFileDialog(self, 'Export page')
        if not self.exportDir:
            fd.setDirectory( self.dirname )
        else:
            fd.setDirectory( self.exportDir )
            
        fd.setFileMode( qt4.QFileDialog.AnyFile )
        fd.setAcceptMode( qt4.QFileDialog.AcceptSave )

        # Create a mapping between a format string and extensions
        filtertoext = {}
        filters = []
        # a list of extensions which are allowed
        validextns = []
        for extns, name in formats:
            extensions = " ".join(["*." + item for item in extns])
            # join eveything together to make a filter string
            filterstr = '%s (%s)' % (name, extensions)
            filtertoext[filterstr] = extns
            filters.append(filterstr)
            validextns += extns

        fd.setFilters(filters)
        # restore last format if possible
        try:
            filt = setting.settingdb['export_lastformat']
            fd.selectFilter(filt)
            extn = formats[filters.index(filt)][0][0]
        except KeyError:
            extn = 'eps'

        if self.filename:
            # try to convert current filename to export name
            filename = os.path.basename(self.filename)
            filename = os.path.splitext(filename)[0] + '.' + extn
            fd.selectFile(filename)
        
        if fd.exec_() == qt4.QDialog.Accepted:
            # save directory for next time
            self.exportDir = unicode(fd.directory().absolutePath())

            filterused = str(fd.selectedFilter())
            setting.settingdb['export_lastformat'] = filterused

            chosenextns = filtertoext[filterused]
            
            # show busy cursor
            qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )

            filename = unicode( fd.selectedFiles()[0] )
            
            # Add a default extension if one isn't supplied
            # this is the extension without the dot
            ext = os.path.splitext(filename)[1][1:]
            if (ext not in validextns) and (ext not in chosenextns):
                filename = filename + "." + chosenextns[0]

            try:
                self.document.export(filename, self.plot.getPageNumber(),
                                     dpi=setting.settingdb['export_DPI'],
                                     antialias=setting.settingdb['export_antialias'],
                                     color=setting.settingdb['export_color'],
                                     quality=setting.settingdb['export_quality'])
            except (IOError, RuntimeError), inst:
                qt4.QMessageBox("Veusz",
                                "Error exporting file:\n%s" % inst,
                                qt4.QMessageBox.Critical,
                                qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                                qt4.QMessageBox.NoButton,
                                qt4.QMessageBox.NoButton,
                                self).exec_()
                
            # restore the cursor
            qt4.QApplication.restoreOverrideCursor()

    def slotFilePrint(self):
        """Print the document."""

        if self.document.getNumberPages() == 0:
                qt4.QMessageBox("Veusz",
                                "No pages to print",
                                qt4.QMessageBox.Warning,
                                qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                                qt4.QMessageBox.NoButton,
                                qt4.QMessageBox.NoButton,
                                self).exec_()
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
        self.actions['filesave'].setEnabled(ismodified)

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

    def slotFileExportStyleSheet(self):
        """Export stylesheet as a file."""
    
        filename = self._fileSaveDialog('vst', 'Veusz stylesheet', 'Export stylesheet')
        if filename:
            try:
                f = open(filename, 'w')
            except IOError:
                qt4.QMessageBox("Veusz",
                                "Cannot export stylesheet as '%s'" % filename,
                                qt4.QMessageBox.Critical,
                                qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                                qt4.QMessageBox.NoButton,
                                qt4.QMessageBox.NoButton,
                                self).exec_()
                return
            
            self.document.exportStyleSheet(f)
            
    def slotFileImportStyleSheet(self):
        """Import a style sheet."""
        filename = self._fileOpenDialog('vst', 'Veusz stylesheet', 'Import stylesheet')
        if filename:
            self.document.applyOperation( document.OperationImportStyleSheet(filename) )
            
