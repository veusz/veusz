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

import qt
import document

class ImportDialog(qt.QDialog):
    """Data import dialog."""

    def __init__(self, parent, document):
        """Initialise document."""

        qt.QDialog.__init__(self, parent, 'DataImportDialog', False)
        self.setCaption('Import data')
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
        self.previewedit.setReadOnly(True)
        self.previewedit.setWordWrap(qt.QTextEdit.NoWrap)
        self.previewedit.setText('<preview>')
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
        self.layout.addWidget( dhbox )
        self.layout.addWidget( bhbox )

    dirname = '.'

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
            ifile = open(str(filename), 'r')
            text = ifile.read(1024)
            self.previewedit.setText(text)

        except IOError:
            self.previewedit.setText('<preview>')
        
    def slotImport(self):
        """Do the import data."""
        
        filename = str( self.filenameedit.text() )
        descriptor = str( self.descriptoredit.text() )
        
        try:
            ifile = open(filename, 'r')
        except IOError:
            mb = qt.QMessageBox("Veusz",
                                "Cannot find file '%s'" % filename,
                                qt.QMessageBox.Error,
                                qt.QMessageBox.Ok | qt.QMessageBox.Default,
                                qt.QMessageBox.NoButton,
                                qt.QMessageBox.NoButton)
            mb.exec_loop()
            return

        # do the import
        stream = document.FileStream(ifile)
        sr = document.SimpleRead(descriptor)
        sr.readData(stream)
        names = sr.setInDocument(self.document)

        text = 'Successfully imported data for variables:\n'
        for i in names:
            text += i + '\n'
        self.previewedit.setText(text)
        
        
