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

import veusz.qtall as qt4
import os.path

import veusz.document as document
import veusz.utils as utils
import veusz.setting as setting

import consolewindow
#FIXMEQT4
import plotwindow
#FIXMEQT4
#import treeeditwindow
import action

from veusz.dialogs.aboutdialog import AboutDialog
from veusz.dialogs.reloaddata import ReloadData
from veusz.dialogs.importfits import ImportFITS
import veusz.dialogs.importdialog as importdialog
#import veusz.dialogs.dataeditdialog as dataeditdialog

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
            win.openFileInWindow(filename)
            pass
        cls.windows.append(win)
        return win

    CreateWindow = classmethod(CreateWindow)

    def __init__(self, *args):
        qt4.QMainWindow.__init__(self, *args)

        self.setWindowIcon( action.getIcon('veusz.png') )

        self.document = document.Document()

        self.setAcceptDrops(True)

        self.filename = ''
        self.updateTitlebar()

        # construct menus and toolbars
        self._defineMenus()

        # FIXMEQT4

        self.plot = plotwindow.PlotWindow(self.document, self)
        self.setCentralWidget(self.plot)
        self.plot.createToolbar(self, self.menus['view'])

        # likewise with the tree-editing window
        #self.treeedit = treeeditwindow.TreeEditWindow(self.document, self)
        #self.moveDockWindow( self.treeedit, qt4.Qt.DockLeft, True, 1 )
        self.treeedit = qt4.QWidget()

        # make the console window a dock
        self.console = consolewindow.ConsoleWindow(self.document,
                                                   self)
        self.interpreter = self.console.interpreter
        self.addDockWidget(qt4.Qt.BottomDockWidgetArea, self.console)

        # the plot window is the central window
        self.updateStatusbar('Ready')

        # no dock menu, so we can popup context menus
        #self.setDockMenuEnabled(False)

        # keep page number up to date
        self.pagelabel = qt4.QLabel(self.statusBar())
        self.statusBar().addWidget(self.pagelabel)

        self.dirname = ''
        self.exportDir = ''
        
        self.connect( self.plot, qt4.SIGNAL("sigUpdatePage"),
                      self.slotUpdatePage )

        # disable save if already saved
        self.connect( self.document, qt4.SIGNAL("sigModified"),
                      self.slotModifiedDoc )
        # if the treeeditwindow changes the page, change the plot window
        # FIXMEQT4
        self.connect( self.treeedit, qt4.SIGNAL("sigPageChanged"),
                      self.plot.setPageNumber )

        # if a widget in the plot window is clicked by the user
        #self.connect( self.plot, qt4.SIGNAL("sigWidgetClicked"),
        #              self.treeedit.slotSelectWidget )

        # put the dock windows on the view menu, so they can be shown/hidden
        #self._defineDockViewMenu()

        # enable/disable undo/redo
        self.connect(self.menus['edit'], qt4.SIGNAL('aboutToShow()'),
                     self.slotAboutToShowEdit)

        #Get the list of recently opened files
        self.populateRecentFiles()

    def updateStatusbar(self, text):
        '''Display text for a set period.'''
        self.statusBar().showMessage(text, 2000)

    def _defineDockViewMenu(self):
        """Put the dock windows on the view menu."""

        view = self.menus['view']
        # FIXMEQT4
        # view.setCheckable(True)
        view.insertSeparator(0)
        self.viewdockmenuitems = {}

        for win in self.dockWindows():
            # get name with veusz removed
            # FIXME with something better here
            text = win.caption()
            text.replace(' - Veusz', '')

            item = view.insertItem(text, -1, 0)
            view.connectItem(item, self.slotViewDockWindow)
            self.viewdockmenuitems[item] = win

        self.connect(view, qt4.SIGNAL('aboutToShow()'),
                     self.slotAboutToShowView)

    def dragEnterEvent(self, event):
        if (event.provides("text/uri-list") and self._getVeuszFiles(event)):
            event.accept(True)

    def dropEvent(self, event):
        """Respond to a drop event on the current window"""
        if event.provides("text/uri-list"):
            files = self._getVeuszFiles(event)
            if files:
                if self.document.isBlank():
                    self.openFileInWindow(files[0])
                else:
                    self.CreateWindow(files[0])
                for filename in files[1:]:
                    self.CreateWindow(filename)
            
    def _getVeuszFiles(self, event):
        """Return a list of veusz files from a drag/drop event containing a
        text/uri-list"""
        draggedFiles = qt4.QStringList()
        qt4.QUriDrag.decodeLocalFiles(event, draggedFiles)
        fileList = []
        for i in range(len(draggedFiles)):
            filename=draggedFiles[i]
            if filename[-4:] == ".vsz":
                fileList.append(unicode(filename))
        return fileList

    def slotAboutToShowView(self):
        """Put check marks against dock menu items if appropriate."""

        view = self.menus['view']
        for item, win in self.viewdockmenuitems.items():
            view.setItemChecked(item, win.isVisible())
            view.setItemParameter(item, item)
        
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
        
    def slotViewDockWindow(self, item):
        """Show or hide dock windows as selected."""

        win = self.viewdockmenuitems[item]
        if win.isVisible():
            win.hide()
        else:
            win.show()

    def _defineMenus(self):
        """Initialise the menus and toolbar."""

        # create toolbar
        self.maintoolbar = qt4.QToolBar("Main toolbar - Veusz", self)
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
        items = [
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
            
            ('dataimport', 'Import data into Veusz', '&Import...', 'data',
             self.slotDataImport, 'stock-import.png', False, ''),
            ('dataimport2d', 'Import 2D data into Veusz', 'Import &2D...', 'data',
             self.slotDataImport2D, 'stock-import.png', False, ''),
            ('dataimportfits', 'Import FITS files into Veusz',
             'Import FITS...', 'data', self.slotDataImportFITS, '', False, ''),
            ('dataedit', 'Edit existing datasets', '&Edit...', 'data',
             self.slotDataEdit, 'stock-edit.png', False, ''),
            ('datacreate', 'Create new datasets', '&Create...', 'data',
             self.slotDataCreate, 'stock-new.png', False, ''),
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
            ]
            
        self.actions = action.populateMenuToolbars(items, self.maintoolbar,
                                                   self.menus)
                                                   

    def slotDataImport(self):
        """Display the import data dialog."""
        d = importdialog.ImportDialog(self, self.document)
        d.show()

    def slotDataImport2D(self):
        """Display the 2D import data dialog."""
        d = importdialog.ImportDialog2D(self, self.document)
        d.show()

    def slotDataImportFITS(self):
        """Display the FITS import dialog."""
        d = ImportFITS(self, self.document)
        d.show()

    def slotDataEdit(self):
        """Edit existing datasets."""
        d = dataeditdialog.DataEditDialog(self, self.document)
        d.show()

    def slotDataCreate(self):
        """Create new datasets."""
        d = dataeditdialog.DatasetNewDialog(self.document, self)
        d.show()

    def slotDataReload(self):
        """Reload linked datasets."""
        d = ReloadData(self, self.document)
        d.show()

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
        
        mb = qt4.QMessageBox("Veusz",
                             "Document is modified. Save first?",
                             qt4.QMessageBox.Warning,
                             qt4.QMessageBox.Yes | qt4.QMessageBox.Default,
                             qt4.QMessageBox.No,
                             qt4.QMessageBox.Cancel | qt4.QMessageBox.Escape,
                             self)
        mb.setButtonText(qt4.QMessageBox.Yes, "&Save")
        mb.setButtonText(qt4.QMessageBox.No, "&Discard")
        mb.setButtonText(qt4.QMessageBox.Cancel, "&Cancel")
        return mb.exec_()

    def close(self, alsoDelete):
        """Before closing, check whether we need to save first."""
        if self.document.isModified():
            v = self.queryOverwrite()
            if v == qt4.QMessageBox.Cancel:
                return False
            elif v == qt4.QMessageBox.Yes:
                self.slotFileSave()

        return qt4.QMainWindow.close(self, alsoDelete)

    def closeEvent(self, evt):
        """Called when the window closes."""

        # store the current geometry in the settings database
        geometry = ( self.x(), self.y(), self.width(), self.height() )
        setting.settingdb['geometry_mainwindow'] = geometry

        # store docked windows
        s = qt4.QString()
        stream = qt4.QTextStream(s, qt4.IO_WriteOnly)
        stream << self
        setting.settingdb['geometry_docwindows'] = str(s)

        qt4.QMainWindow.closeEvent(self, evt)

    def showEvent(self, evt):
        """Restoring window geometry if possible."""

        # if we can restore the geometry, do so
        if 'geometry_mainwindow' in setting.settingdb:
            geometry = setting.settingdb['geometry_mainwindow']
            self.resize( qt4.QSize(geometry[2], geometry[3]) )
            self.move( qt4.QPoint(geometry[0], geometry[1]) )

        # restore docked window geometry
        if 'geometry_docwindows' in setting.settingdb:
            s = setting.settingdb['geometry_docwindows']
            s = qt4.QString(s)
            stream = qt4.QTextStream(s, qt4.IO_ReadOnly)
            stream >> self

        qt4.QMainWindow.showEvent(self, evt)

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
            except IOError:
                qt4.QMessageBox("Veusz",
                                "Cannot save file as '%s'" % self.filename,
                                qt4.QMessageBox.Critical,
                                qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                                qt4.QMessageBox.NoButton,
                                qt4.QMessageBox.NoButton,
                                self).exec_()
                
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
        
        fd = qt4.QFileDialog(self, dialogtitle, self.dirname,
                             "%s (*.%s)" % (filedescr, filetype) )
        fd.setFileMode( qt4.QFileDialog.AnyFile )

        # okay was selected
        if fd.exec_() == qt4.QDialog.Accepted:
            # save directory for next time
            self.dirname = fd.directory()
            # update the edit box
            filename = unicode( fd.selectedFiles()[0] )
            if os.path.splitext(filename)[1] == '':
                filename += '.' + filetype

            # test whether file exists and ask whether to overwrite it
            try:
                open(filename)
            except IOError:
                pass
            else:
                v = qt4.QMessageBox("Veusz",
                                    "File exists, overwrite?",
                                    qt4.QMessageBox.Warning,
                                    qt4.QMessageBox.Yes,
                                    qt4.QMessageBox.No | qt4.QMessageBox.Default,
                                    qt4.QMessageBox.NoButton,
                                    self).exec_()
                if v == qt4.QMessageBox.No:
                    return None

            return filename
        return None

    def _fileOpenDialog(self, filetype, filedescr, dialogtitle):
        """Display an open dialog and return a filename."""
        
        fd = qt4.QFileDialog(self, dialogtitle, self.dirname,
                             "%s (*.%s)" % (filedescr, filetype) )
        fd.setFileMode( qt4.QFileDialog.ExistingFile )

        # if the user chooses a file
        if fd.exec_() == qt4.QDialog.Accepted:
            # save directory for next time
            self.dirname = fd.directory()

            filename = unicode( fd.selectedFiles()[0] )
            try:
                open(filename)
            except IOError, e:
                qt4.QMessageBox("Veusz",
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

    def openFileInWindow(self, filename):
        '''Open the given filename in the current window.'''

        # show busy cursor
        qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )
        
        try:
            # load the document in the current window
            self.dirname = os.path.dirname(filename)
            self.interpreter.Load(filename)
            self.document.setModified(False)
            self.filename = filename
            self.updateTitlebar()
            self.updateStatusbar("Opened %s" % filename)

            #Update the list of recently opened files
            fullname = os.path.abspath(filename)
            if 'recent_files' in setting.settingdb:
                filelist = setting.settingdb['recent_files']
                if fullname in filelist:
                    filelist.remove(fullname)
                filelist.insert(0, fullname)
                filelist = filelist[:5]
            else:
                filelist = [fullname]
            setting.settingdb['recent_files'] = filelist
            self.populateRecentFiles()

        except IOError:
            # problem reading file
            qt4.QMessageBox("Veusz",
                            "Cannot open file '%s'" % filename,
                            qt4.QMessageBox.Critical,
                            qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                            qt4.QMessageBox.NoButton,
                            qt4.QMessageBox.NoButton,
                            self).exec_()
        except Exception, e:
            # parsing problem with document
            # FIXME: never used
            qt4.QMessageBox("Veusz",
                            "Error in file '%s'\n'%s'" % (filename, str(e)),
                            qt4.QMessageBox.Critical,
                            qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                            qt4.QMessageBox.NoButton,
                            qt4.QMessageBox.NoButton,
                            self).exec_()
        
        # restore the cursor
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
        if 'recent_files' in setting.settingdb and setting.settingdb['recent_files']:
            files = setting.settingdb['recent_files']
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
            self.recentFileActions = action.populateMenuToolbars(newMenuItems,
                                                                 self.maintoolbar,
                                                                 self.menus)
        else:
            menu.setEnabled(False)
                
    
    def slotFileExport(self):
        """Export the graph."""

        #XXX - This should be disabled if the page count is 0

        #File types we can export to in the form ([extensions], Name)
        formats = [(["eps"], "Encapsulated Postscript"),
                   (["png"], "Portable Network Graphics"),
                   (["svg"], "Scalable Vector Graphics")]

        fd = qt4.QFileDialog(self, 'Export page')
        if not self.exportDir:
            fd.setDirectory( self.dirname )
        else:
            fd.setDirectory( self.exportDir )
            
        fd.setFileMode( qt4.QFileDialog.AnyFile )

        # Create a mapping between a format string and extensions
        filtertoext = {}
        filters = []
        # a list of extensions which are allowed
        validextns = []
        for extns, name in formats:
            extensions = " ".join(["*." + item for item in extns])
            #join eveything together to make a filter string
            filterstr = '%s (%s)' % (name, extensions)
            filtertoext[filterstr] = extns
            filters.append(filterstr)
            validextns += extns

        fd.setFilters(filters)

        if self.filename:
            # try to convert current filename to export name
            filename = os.path.basename(self.filename)
            filename = os.path.splitext(filename)[0] + '.eps'
            fd.selectFile(filename)
        
        if fd.exec_() == qt4.QDialog.Accepted:
            # save directory for next time
            self.exportDir = fd.dir()

            filterused = str(fd.selectedFilter())
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
                self.document.export(filename, self.plot.getPageNumber())
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

        doc = self.document
        prnt = qt4.QPrinter(qt4.QPrinter.HighResolution)
        prnt.setColorMode(qt4.QPrinter.Color)
        prnt.setMinMax( 1, doc.getNumberPages() )
        prnt.setCreator('Veusz %s' % utils.version())
        prnt.setDocName(self.filename)

        if prnt.setup():
            # get page range
            minval, maxval = prnt.fromPage(), prnt.toPage()

            # all pages requested
            if minval == 0 and maxval == 0:
                minval, maxval = 1, doc.getNumberPages()

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
            doc.printTo( prnt, pages )

    def slotModifiedDoc(self, ismodified):
        """Disable certain actions if document is not modified."""

        # enable/disable file, save menu item
        self.actions['filesave'].setEnabled(ismodified)

    def slotFileClose(self):
        """File close window chosen."""
        self.close(True)

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
            
