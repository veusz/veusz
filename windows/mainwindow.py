# mainwindow.py
# the main window of the application

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

import qt
import os.path
from math import sqrt

import consolewindow
import plotwindow
import treeeditwindow
import document
import utils
import setting

import dialogs.aboutdialog
import dialogs.importdialog
import dialogs.dataeditdialog
import dialogs.reloaddata
import dialogs.importfits

_fdirname = os.path.dirname(__file__)

class MainWindow(qt.QMainWindow):
    """ The main window class for the application."""

    def __init__(self, *args):
        qt.QMainWindow.__init__(self, *args)

        self.setIcon( qt.QPixmap(os.path.join(_fdirname, '..',
                                              'images', 'icon.png')) )

        self.document = document.Document()

        self.setAcceptDrops(True)

        self.filename = ''
        self.updateTitlebar()

        # construct menus and toolbars
        self._defineMenus()

        self.plot = plotwindow.PlotWindow(self.document, self)
        self.plotzoom = 0

        # likewise with the tree-editing window
        self.treeedit = treeeditwindow.TreeEditWindow(self.document, self)
        self.moveDockWindow( self.treeedit, qt.Qt.DockLeft, True, 1 )

        # make the console window a dock
        self.console = consolewindow.ConsoleWindow(self.document,
                                                   self)
        self.interpreter = self.console.interpreter
        self.moveDockWindow( self.console, qt.Qt.DockBottom )

        # the plot window is the central window
        self.setCentralWidget( self.plot )
        self.updateStatusbar('Ready')

        # no dock menu, so we can popup context menus
        self.setDockMenuEnabled(False)

        # keep page number up to date
        self.pagelabel = qt.QLabel(self.statusBar())
        self.statusBar().addWidget(self.pagelabel)

        self.dirname = ''
        self.exportDir = ''
        
        self.connect( self.plot, qt.PYSIGNAL("sigUpdatePage"),
                      self.slotUpdatePage )

        # disable save if already saved
        self.connect( self.document, qt.PYSIGNAL("sigModified"),
                      self.slotModifiedDoc )

        # if the treeeditwindow changes the page, change the plot window
        self.connect( self.treeedit, qt.PYSIGNAL("sigPageChanged"),
                      self.plot.setPageNumber )

        # put the dock windows on the view menu, so they can be shown/hidden
        self._defineDockViewMenu()

    def updateStatusbar(self, text):
        '''Display text for a set period.'''
        self.statusBar().message(text, 2000)

    def _defineDockViewMenu(self):
        """Put the dock windows on the view menu."""

        view = self.menus['view']
        view.setCheckable(True)
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

        self.connect(view, qt.SIGNAL('aboutToShow()'),
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
                    self.openFile(files[0])
                else:
                    CreateWindow(files[0])
                for filename in files[1:]:
                    CreateWindow(filename)
            
            
    def _getVeuszFiles(self, event):
        """Return a list of veusz files from a drag/drop event containing a
        text/uri-list"""
        draggedFiles = qt.QStringList()
        qt.QUriDrag.decodeLocalFiles(event, draggedFiles)
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
        self.maintoolbar = qt.QToolBar(self, "maintoolbar")
        self.maintoolbar.setLabel("Main toolbar - Veusz")

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
            menu = qt.QPopupMenu(self)
            self.menuBar().insertItem( text, menu )
            self.menus[menuid] = menu

        # items for main menus
        # Items are: Lookup id, description, menu text, which menu,
        #  Slot, Icon (or ''), whether to add to toolbar,
        #  Keyboard shortcut (or '')
        items = [
            ('filenew', 'New document', '&New', 'file',
             self.slotFileNew, 'stock-new.png', True, 'Ctrl+N'),
            ('fileopen', 'Open a document', '&Open...', 'file',
             self.slotFileOpen, 'stock-open.png', True, 'Ctrl+O'),
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
            ('file', ),
            ('fileclose', 'Close current window', 'Close Window', 'file',
             self.slotFileClose, '', False, 'Ctrl+W'),
            ('filequit', 'Exit the program', '&Quit', 'file',
             self.slotFileQuit, 'stock-quit.png', False, 'Ctrl+Q'),
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
            ('viewzoomin', 'Zoom into the plot', 'Zoom &In', 'view',
             self.slotViewZoomIn, 'stock-zoom-in.png', False, 'Ctrl++'),
            ('viewzoomout', 'Zoom out of the plot', 'Zoom &Out', 'view',
             self.slotViewZoomOut, 'stock-zoom-out.png', False, 'Ctrl+-'),
            ('viewzoom11', 'Restore plot to natural size', 'Zoom 1:1', 'view',
             self.slotViewZoom11, 'stock-zoom-100.png', False, 'Ctrl+1'),
            ('viewzoomwidth', 'Zoom plot to show whole width',
             'Zoom to width', 'view', self.slotViewZoomWidth,
             'stock-zoom-fit.png', False, ''),
            ('viewzoomheight', 'Zoom plot to show whole height',
             'Zoom to height', 'view', self.slotViewZoomHeight,
             'stock-zoom-fit.png', False, ''),
            ('viewzoompage', 'Zoom plot to show whole page',
             'Zoom to page', 'view', self.slotViewZoomPage,
             'stock-zoom-fit.png', False, ''),
            ('view',),
            ('viewprevpage', 'Move to the previous page', '&Previous page',
             'view', self.slotViewPreviousPage, 'stock_previous-page.png',
             True, 'Ctrl+PgUp'),
            ('viewnextpage', 'Move to the next page', '&Next page',
             'view', self.slotViewNextPage, 'stock_next-page.png',
             True, 'Ctrl+PgDown'),
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
            
        # construct the menus and toolbar from the above data
        self.actions = {}
        for i in items:
            # we just want to insert a separator
            if len(i) == 1:
                self.menus[i[0]].insertSeparator()
                continue
            
            menuid, descr, menutext, menu, slot, icon, addtool, key = i
            if key == '':
                ks = qt.QKeySequence()
            else:
                ks = qt.QKeySequence(key)

            action = qt.QAction(descr, menutext, ks, self)

            # load icon if set
            if icon != '':
                f = os.path.join(_fdirname, 'icons', icon)
                action.setIconSet(qt.QIconSet( qt.QPixmap(f) ))

            # connect the action to the slot
            qt.QObject.connect( action, qt.SIGNAL('activated()'), slot )

            # add to menu
            action.addTo( self.menus[menu] )

            # add to toolbar
            if addtool:
                action.addTo(self.maintoolbar)

            # save for later
            self.actions[menuid] = action

        zoomtb = qt.QToolButton(self.maintoolbar)
        f = os.path.join(_fdirname, 'icons', 'zoom-options.png')
        zoomtb.setIconSet(qt.QIconSet( qt.QPixmap(f) ))

        # drop down zoom button on toolbar
        zoompop = qt.QPopupMenu(zoomtb)
        for action in ('viewzoomin', 'viewzoomout', 'viewzoom11',
                       'viewzoomwidth', 'viewzoomheight', 'viewzoompage'):
            self.actions[action].addTo(zoompop)
        zoomtb.setPopup(zoompop)
        zoomtb.setPopupDelay(0)

    def slotDataImport(self):
        """Display the import data dialog."""
        d = dialogs.importdialog.ImportDialog(self, self.document)
        d.show()

    def slotDataImport2D(self):
        """Display the 2D import data dialog."""
        d = dialogs.importdialog.ImportDialog2D(self, self.document)
        d.show()

    def slotDataImportFITS(self):
        """Display the FITS import dialog."""
        d = dialogs.importfits.ImportFITS(self, self.document)
        d.show()

    def slotDataEdit(self):
        """Edit existing datasets."""
        d = dialogs.dataeditdialog.DataEditDialog(self, self.document)
        d.show()

    def slotDataCreate(self):
        """Create new datasets."""
        d = dialogs.dataeditdialog.DatasetNewDialog(self.document, self)
        d.show()

    def slotDataReload(self):
        """Reload linked datasets."""
        d = dialogs.reloaddata.ReloadData(self, self.document)
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
        d = dialogs.aboutdialog.AboutDialog(self)
        d.exec_loop()

    def queryOverwrite(self):
        """Do you want to overwrite the current document.

        Returns qt.QMessageBox.(Yes,No,Cancel)."""
        
        mb = qt.QMessageBox("Veusz",
                            "Document is modified. Save first?",
                            qt.QMessageBox.Warning,
                            qt.QMessageBox.Yes | qt.QMessageBox.Default,
                            qt.QMessageBox.No,
                            qt.QMessageBox.Cancel | qt.QMessageBox.Escape,
                            self)
        mb.setButtonText(qt.QMessageBox.Yes, "&Save")
        mb.setButtonText(qt.QMessageBox.No, "&Discard")
        mb.setButtonText(qt.QMessageBox.Cancel, "&Cancel")
        return mb.exec_loop()

    def close(self, alsoDelete):
        """Before closing, check whether we need to save first."""
        if self.document.isModified():
            v = self.queryOverwrite()
            if v == qt.QMessageBox.Cancel:
                return False
            elif v == qt.QMessageBox.Yes:
                self.slotFileSave()

        return qt.QMainWindow.close(self, alsoDelete)

    def closeEvent(self, evt):
        """Called when the window closes."""

        # store the current geometry in the settings database
        geometry = ( self.x(), self.y(), self.width(), self.height() )
        setting.settingdb['geometry_mainwindow'] = geometry

        # store docked windows
        s = qt.QString()
        stream = qt.QTextStream(s, qt.IO_WriteOnly)
        stream << self
        setting.settingdb['geometry_docwindows'] = str(s)

        qt.QMainWindow.closeEvent(self, evt)

    def showEvent(self, evt):
        """Restoring window geometry if possible."""

        # if we can restore the geometry, do so
        if 'geometry_mainwindow' in setting.settingdb:
            geometry = setting.settingdb['geometry_mainwindow']
            self.resize( qt.QSize(geometry[2], geometry[3]) )
            self.move( qt.QPoint(geometry[0], geometry[1]) )

        # restore docked window geometry
        if 'geometry_docwindows' in setting.settingdb:
            s = setting.settingdb['geometry_docwindows']
            s = qt.QString(s)
            stream = qt.QTextStream(s, qt.IO_ReadOnly)
            stream >> self

        qt.QMainWindow.showEvent(self, evt)

    def slotFileNew(self):
        """New file."""
        CreateWindow()

    def slotFileSave(self):
        """Save file."""

        if self.filename == '':
            self.slotFileSaveAs()
        else:
            # show busy cursor
            qt.QApplication.setOverrideCursor( qt.QCursor(qt.Qt.WaitCursor) )

            try:
                ofile = open(self.filename, 'w')
                self.document.saveToFile(ofile)
                self.updateStatusbar("Saved to %s" % self.filename)
            except IOError:
                qt.QMessageBox("Veusz",
                               "Cannot save file as '%s'" % self.filename,
                               qt.QMessageBox.Critical,
                               qt.QMessageBox.Ok | qt.QMessageBox.Default,
                               qt.QMessageBox.NoButton,
                               qt.QMessageBox.NoButton,
                               self).exec_loop()
                
            # restore the cursor
            qt.QApplication.restoreOverrideCursor()
                
    def updateTitlebar(self):
        """Put the filename into the title bar."""
        if self.filename == '':
            self.setCaption('Untitled - Veusz')
        else:
            self.setCaption( "%s - Veusz" %
                             os.path.basename(self.filename) )

    def slotFileSaveAs(self):
        """Save As file."""

        fd = qt.QFileDialog(self, 'save as dialog', True)
        fd.setDir(self.dirname)
        fd.setMode( qt.QFileDialog.AnyFile )
        fd.setFilter( "Veusz script files (*.vsz)" )
        fd.setCaption('Save as')

        # okay was selected
        if fd.exec_loop() == qt.QDialog.Accepted:
            # save directory for next time
            self.dirname = fd.dir()
            # update the edit box
            filename = unicode( fd.selectedFile() )
            if os.path.splitext(filename)[1] == '':
                filename += '.vsz'

            # test whether file exists and ask whether to overwrite it
            try:
                open(filename, 'r')
            except IOError:
                pass
            else:
                v = qt.QMessageBox("Veusz",
                                   "File exists, overwrite?",
                                   qt.QMessageBox.Warning,
                                   qt.QMessageBox.Yes,
                                   qt.QMessageBox.No | qt.QMessageBox.Default,
                                   qt.QMessageBox.NoButton,
                                   self).exec_loop()
                if v == qt.QMessageBox.No:
                    return

            self.filename = filename
            self.updateTitlebar()

            self.slotFileSave()

    def openFile(self, filename):
        '''Open the given filename in the current window.'''

        # show busy cursor
        qt.QApplication.setOverrideCursor( qt.QCursor(qt.Qt.WaitCursor) )
        
        try:
            # load the document in the current window
            self.dirname = os.path.dirname(filename)
            self.interpreter.Load(filename)
            self.document.setModified(False)
            self.filename = filename
            self.updateTitlebar()
            self.updateStatusbar("Opened %s" % filename)
        except IOError:
            # problem reading file
            qt.QMessageBox("Veusz",
                           "Cannot open file '%s'" % filename,
                           qt.QMessageBox.Critical,
                           qt.QMessageBox.Ok | qt.QMessageBox.Default,
                           qt.QMessageBox.NoButton,
                           qt.QMessageBox.NoButton,
                           self).exec_loop()
        except Exception, e:
            # parsing problem with document
            # FIXME: never used
            qt.QMessageBox("Veusz",
                           "Error in file '%s'\n'%s'" % (filename, str(e)),
                           qt.QMessageBox.Critical,
                           qt.QMessageBox.Ok | qt.QMessageBox.Default,
                           qt.QMessageBox.NoButton,
                           qt.QMessageBox.NoButton,
                           self).exec_loop()

        # restore the cursor
        qt.QApplication.restoreOverrideCursor()

    def slotFileOpen(self):
        """Open an existing file in a new window."""

        fd = qt.QFileDialog(self, 'open dialog', True)
        fd.setDir( self.dirname )
        fd.setMode( qt.QFileDialog.ExistingFile )
        fd.setFilter ( "Veusz script files (*.vsz)" )
        fd.setCaption('Open')

        # if the user chooses a file
        if fd.exec_loop() == qt.QDialog.Accepted:
            # save directory for next time
            self.dirname = fd.dir()

            filename = unicode( fd.selectedFile() )
            if self.document.isBlank():
                # If the file is new and there are no modifications,
                # reuse the current window
                self.openFile(filename)
            else:
                # create a new window
                CreateWindow(filename)
                
    def slotFileExport(self):
        """Export the graph."""

        #XXX - This should be disabled if the page count is 0

        #File types we can export to in the form ([extensions], Name)
        formats = [(["eps"], "Encapsulated Postscript"),
                   (["png"], "Portable Network Graphics")]

        fd = qt.QFileDialog(self, 'export dialog', True)
        if not self.exportDir:
            fd.setDir( self.dirname )
        else:
            fd.setDir( self.exportDir )
            
        fd.setMode( qt.QFileDialog.AnyFile )

        #Create a mapping between a format string and extensions
        filtertoext = {}
        filters = []
        # a list of extensions which are allowed
        validextns = []
        for extns, name in formats:
            extensions = " ".join(["(*." + item + ")"
                                   for item in extns])
            #join eveything together to make a filter string
            filterstr = " ".join([name, extensions])
            filtertoext[filterstr] = extns
            filters.append(filterstr)
            validextns += extns

        fd.setFilters(";;".join(filters))
            
        fd.setCaption('Export')

        if fd.exec_loop() == qt.QDialog.Accepted:
            # save directory for next time
            self.exportDir = fd.dir()

            filterused = str(fd.selectedFilter())
            chosenextns = filtertoext[filterused]
            
            # show busy cursor
            qt.QApplication.setOverrideCursor( qt.QCursor(qt.Qt.WaitCursor) )

            filename = unicode( fd.selectedFile() )
            
            # Add a default extension if one isn't supplied
            # this is the extension without the dot
            ext = os.path.splitext(filename)[1][1:]
            if (ext not in validextns) and (ext not in chosenextns):
                filename = filename + "." + returnedextns[0]

            try:
                self.document.export(filename, self.plot.getPageNumber())
            except (IOError, RuntimeError), inst:
                qt.QMessageBox("Veusz",
                               "Error exporting file:\n%s" % inst,
                               qt.QMessageBox.Critical,
                               qt.QMessageBox.Ok | qt.QMessageBox.Default,
                               qt.QMessageBox.NoButton,
                               qt.QMessageBox.NoButton,
                               self).exec_loop()

            # restore the cursor
            qt.QApplication.restoreOverrideCursor()

    def slotFilePrint(self):
        """Print the document."""

        doc = self.document
        prnt = qt.QPrinter(qt.QPrinter.HighResolution)
        prnt.setColorMode(qt.QPrinter.Color)
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
            if prnt.pageOrder() == qt.QPrinter.FirstPageFirst:
                pages = range(minval, maxval+1)
            else:
                pages = range(maxval, minval-1, -1)

            # if more copies are requested
            pages *= prnt.numCopies()

            # do the printing
            doc.printTo( prnt, pages )

    def slotViewZoomIn(self):
        """Zoom into the plot."""

        if self.plotzoom < 6:
            self.plotzoom += 1
        self.plot.setZoomFactor( sqrt(2) ** self.plotzoom )

    def slotViewZoomOut(self):
        """Zoom out of the plot."""

        if self.plotzoom > -6:
            self.plotzoom -= 1
        self.plot.setZoomFactor( sqrt(2) ** self.plotzoom )

    def slotViewZoom11(self):
        """Restore the zoom to 1:1"""
        
        self.plotzoom = 0
        self.plot.setZoomFactor( sqrt(2) ** self.plotzoom )

    def slotViewZoomWidth(self):
        """Make the plot fit the page width."""
        self.plot.zoomWidth()

    def slotViewZoomHeight(self):
        """Make the plot fit the page height."""
        self.plot.zoomHeight()

    def slotViewZoomPage(self):
        """Make the plot fit the page."""
        self.plot.zoomPage()

    def slotModifiedDoc(self, ismodified):
        """Disable certain actions if document is not modified."""

        # enable/disable file, save menu item
        self.actions['filesave'].setEnabled(ismodified)

    def slotFileClose(self):
        """File close window chosen."""
        self.close(True)

    def slotFileQuit(self):
        """File quit chosen."""
        qt.qApp.closeAllWindows()
        
    def slotViewPreviousPage(self):
        """View the previous page."""
        self.plot.setPageNumber( self.plot.getPageNumber() - 1 )

    def slotViewNextPage(self):
        """View the next page."""
        self.plot.setPageNumber( self.plot.getPageNumber() + 1 )

    def slotUpdatePage(self, number):
        """Update page number when the plot window says so."""

        np = self.document.getNumberPages()
        if np == 0:
            self.pagelabel.setText("No pages")
        else:
            self.pagelabel.setText("Page %i/%i" % (number+1, np))

        # disable previous and next page actions
        self.actions['viewprevpage'].setEnabled( number != 0 )
        self.actions['viewnextpage'].setEnabled( number < np-1 )

def CreateWindow(filename=None):
    """Window factory function"""
    win = MainWindow()
    win.show()
    if filename:
        win.openFile(filename)
    return win
