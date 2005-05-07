# data import dialog

#    Copyright (C) 2004 Jeremy S. Sanders
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

"""Module for implementing dialog box for importing data in Veusz."""

import os.path

import qt
import document
import setting

_importhelp=('The descriptor describes the format of the data '
             'in the file. Each dataset is described separated with spaces.'
             ' Commas separate the dataset names from the description of the '
             'errors\n'
             'eg\tz\t[z with no errors - 1 column for dataset]\n'
             '\tx,+-\t[x with symmetric errors - 2 columns for dataset]\n'
             '\ty,+,-\t[y with asymmetric errors - 3 columns for dataset]\n'
             '\tx[1:5],+,-\t[x_1 to x_5, each with asymmetric errors -'
             ' 15 columns in total]\n'
             '\tx y,+-\t[x with no errors, y with symmetric errors -'
             ' 3 columns in total]')

class ImportDialog(qt.QDialog):
    """Data import dialog."""

    def __init__(self, parent, document):
        """Initialise document."""

        qt.QDialog.__init__(self, parent, 'DataImportDialog', False)
        self.setCaption('Import data - Veusz')
        self.document = document

        spacing = self.fontMetrics().height() / 2

        # change the filename
        fnhbox = qt.QHBox(self)
        fnhbox.setSpacing(spacing)
        l = qt.QLabel('&Filename:', fnhbox)
        self.filenameedit = qt.QLineEdit(fnhbox)
        l.setBuddy(self.filenameedit)
        qt.QToolTip.add(self.filenameedit,
                        'The name of the file to import data from')

        qt.QObject.connect( self.filenameedit,
                            qt.SIGNAL('textChanged(const QString&)'),
                            self.slotFilenameChanged )

        browsebutton = qt.QPushButton("&Browse...", fnhbox)
        qt.QObject.connect( browsebutton, qt.SIGNAL('clicked()'),
                            self.slotBrowse )

        # preview the file
        self.previewedit = qt.QTextEdit(self)
        self.previewedit.setTextFormat(qt.Qt.PlainText)
        self.previewedit.setReadOnly(True)
        self.previewedit.setWordWrap(qt.QTextEdit.NoWrap)
        self.previewedit.setText('<preview>')
        f = qt.QFont('Fixed')
        self.previewedit.setFont(f)
        qt.QToolTip.add(self.previewedit,
                        'A preview of the contents of the file')
        
        # edit the descriptor
        dhbox = qt.QHBox(self)
        dhbox.setSpacing(spacing)
        l = qt.QLabel('&Descriptor:', dhbox)
        self.descriptoredit = qt.QLineEdit(dhbox)
        l.setBuddy(self.descriptoredit)
        qt.QToolTip.add(self.descriptoredit,
                        'Names of columns when importing data, '
                        'e.g. "x y" or "a[:]"')
        
        # allow links from the file, so the data are reread from the file
        # on reloading
        self.linkcheck = qt.QCheckBox('&Link the datasets to the file',
                                      self)
        qt.QToolTip.add(self.linkcheck,
                        'Linked datasets are not saved to a document file,'
                        '\nbut are reloaded from the linked data file.')

        # buttons
        bhbox = qt.QHBox(self)
        bhbox.setSpacing(spacing)
        
        importbutton = qt.QPushButton("&Import", bhbox)
        importbutton.setDefault( True )
        closebutton = qt.QPushButton("&Close", bhbox)
        qt.QObject.connect( closebutton, qt.SIGNAL('clicked()'),
                            self.slotClose )
        qt.QObject.connect( importbutton, qt.SIGNAL('clicked()'),
                            self.slotImport )

        # construct the dialog
        self.layout = qt.QVBoxLayout(self, spacing)
        self.layout.addWidget( fnhbox )
        self.layout.addWidget( self.previewedit )

        l = qt.QLabel(_importhelp, self)
        l.setAlignment( l.alignment() | qt.Qt.WordBreak )
        self.layout.addWidget( l )

        self.layout.addWidget( dhbox )
        self.layout.addWidget( self.linkcheck )
        self.layout.addWidget( bhbox )

    dirname = '.'

    def closeEvent(self, evt):
        """Called when the window closes."""

        # store the current geometry in the settings database
        geometry = ( self.x(), self.y(), self.width(), self.height() )
        setting.settingdb.database['geometry_importdialog'] = geometry

        qt.QDialog.closeEvent(self, evt)

    def showEvent(self, evt):
        """Restoring window geometry if possible."""

        # if we can restore the geometry, do so
        if 'geometry_importdialog' in setting.settingdb.database:
            geometry =  setting.settingdb.database['geometry_importdialog']
            self.resize( qt.QSize(geometry[2], geometry[3]) )
            self.move( qt.QPoint(geometry[0], geometry[1]) )

        qt.QDialog.showEvent(self, evt)

    def sizeHint(self):
        """Returns recommended size of dialog."""
        return qt.QSize(600, 400)

    def slotBrowse(self):
        """Browse button pressed in dialog."""

        fd = qt.QFileDialog(self, 'import dialog', True)
        fd.setDir( ImportDialog.dirname )
        fd.setMode( qt.QFileDialog.ExistingFile )
        fd.setCaption('Browse data file')

        # okay was selected
        if fd.exec_loop() == qt.QDialog.Accepted:
            # save directory for next time
            ImportDialog.dirname = fd.dir()
            # update the edit box
            self.filenameedit.setText( fd.selectedFile() )

    def slotClose(self):
        """Close the dialog."""

        self.close(True)

    def slotFilenameChanged(self, filename):
        """Update preview window when filename changed."""

        try:
            ifile = open(unicode(filename), 'r')
            text = ifile.read(2048)
            self.previewedit.setText(text)

        except IOError:
            self.previewedit.setText('<preview>')
        
    def slotImport(self):
        """Do the import data."""
        
        filename = unicode( self.filenameedit.text() )
        filename = os.path.abspath(filename)
        descriptor = unicode( self.descriptoredit.text() )
        islinked = self.linkcheck.isChecked()
        
        try:
            ifile = open(filename, 'r')
        except IOError:
            mb = qt.QMessageBox("Veusz",
                                "Cannot find file '%s'" % filename,
                                qt.QMessageBox.Warning,
                                qt.QMessageBox.Ok | qt.QMessageBox.Default,
                                qt.QMessageBox.NoButton,
                                qt.QMessageBox.NoButton,
                                self)
            mb.exec_loop()
            return

        # do the import
        stream = document.FileStream(ifile)

        try:
            sr = document.SimpleRead(descriptor)
        except ValueError:
            mb = qt.QMessageBox("Veusz",
                                "Cannot interpret descriptor",
                                qt.QMessageBox.Warning,
                                qt.QMessageBox.Ok | qt.QMessageBox.Default,
                                qt.QMessageBox.NoButton,
                                qt.QMessageBox.NoButton,
                                self)
            mb.exec_loop()
            return

        # show a busy cursor
        qt.QApplication.setOverrideCursor( qt.QCursor(qt.Qt.WaitCursor) )

        # read the data
        sr.readData(stream)

        # restore the cursor
        qt.QApplication.restoreOverrideCursor()

        text = ''
        for var, count in sr.getInvalidConversions().items():
            if count != 0:
                text += '%i conversions failed for variable "%s"\n' % (
                    count, var)
        if text != '':
            text += '\n'

        # link the data to a file, if told to
        if islinked:
            LF = document.LinkedFile(filename, descriptor)
        else:
            LF = None

        names = sr.setInDocument(self.document, linkedfile=LF)

        text += 'Imported data for variables:\n'
        for i in names:
            text += i + '\n'

        if LF != None:
            text += '\nDatasets were linked to file "%s"\n' % filename

        self.previewedit.setText(text)
        
        
