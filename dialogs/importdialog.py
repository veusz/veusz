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

"""Module for implementing dialog boxes for importing data in Veusz."""

import os.path
import re

import qt
import document
import setting

class ImportDialogBase(qt.QDialog):
    """Base class for data importing dialogs.

    A lot of the functionality is replicated otherwise
    """

    dirname = '.'

    def __init__(self, parent, document):
        qt.QDialog.__init__(self, parent, 'DataImportDialog', False)
        self.document = document

        self.spacing = self.fontMetrics().height() / 2

        # layout for dialog
        self.layout = qt.QVBoxLayout(self, self.spacing)
 
        # change the filename
        fnhbox = qt.QHBox(self)
        fnhbox.setSpacing(self.spacing)
        self.layout.addWidget( fnhbox )
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
        self.layout.addWidget( self.previewedit )
        self.previewedit.setTextFormat(qt.Qt.PlainText)
        self.previewedit.setReadOnly(True)
        self.previewedit.setWordWrap(qt.QTextEdit.NoWrap)
        self.previewedit.setText('<preview>')
        f = qt.QFont('Fixed')
        self.previewedit.setFont(f)
        qt.QToolTip.add(self.previewedit,
                        'A preview of the contents of the file')
        
        # space for descendents of this class to add widgets
        self.widgetspace = qt.QVBox(self)
        self.layout.addWidget(self.widgetspace)
        self.widgetspace.setSpacing(self.spacing)

        # allow links from the file, so the data are reread from the file
        # on reloading
        self.linkcheck = qt.QCheckBox('&Link the datasets to the file',
                                      self)
        self.layout.addWidget( self.linkcheck )
        qt.QToolTip.add(self.linkcheck,
                        'Linked datasets are not saved to a document file,'
                        '\nbut are reloaded from the linked data file.')

        # buttons
        w = qt.QWidget(self)
        self.layout.addWidget(w)
        w.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        l = qt.QHBoxLayout(w, 0, self.spacing)
        l.addItem( qt.QSpacerItem(1, 1, qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum) )
        
        b = qt.QPushButton("&Import", w)
        self.importbutton = b
        b.setEnabled(False)
        b.setDefault(True)
        l.addWidget(b)
        self.connect(b, qt.SIGNAL('clicked()'), self.slotImport)
        b = qt.QPushButton("&Close", w)
        l.addWidget(b)
        self.connect(b, qt.SIGNAL('clicked()'), self.slotClose)

    def sizeHint(self):
        """Returns recommended size of dialog."""
        return qt.QSize(600, 400)

    def closeEvent(self, evt):
        """Called when the window closes."""

        # store the current geometry in the settings database
        geometry = ( self.x(), self.y(), self.width(), self.height() )
        setting.settingdb['geometry_importdialog'] = geometry

        qt.QDialog.closeEvent(self, evt)

    def showEvent(self, evt):
        """Restoring window geometry if possible."""

        # if we can restore the geometry, do so
        if 'geometry_importdialog' in setting.settingdb:
            geometry =  setting.settingdb['geometry_importdialog']
            self.resize( qt.QSize(geometry[2], geometry[3]) )
            self.move( qt.QPoint(geometry[0], geometry[1]) )

        qt.QDialog.showEvent(self, evt)

    def slotBrowse(self):
        """Browse button pressed in dialog."""

        fd = qt.QFileDialog(self, 'import dialog', True)
        fd.setMode( qt.QFileDialog.ExistingFile )
        fd.setCaption('Browse data file')

        # use filename to guess a path if possible
        filename = unicode(self.filenameedit.text())
        if os.path.isdir(filename):
            ImportDialogBase.dirname = filename
        elif os.path.isdir( os.path.dirname(filename) ):
            ImportDialogBase.dirname = os.path.dirname(filename)
        
        fd.setDir( ImportDialogBase.dirname )
        
        # okay was selected
        if fd.exec_loop() == qt.QDialog.Accepted:
            # save directory for next time
            ImportDialogBase.dirname = fd.dir()
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
            self.importbutton.setEnabled(True)

        except IOError:
            self.previewedit.setText('<preview>')
            self.importbutton.setEnabled(False)

    def slotImport(self):
        """Import the data."""
        pass

_import1dhelp='''
The descriptor describes the format of the data in the file. Each dataset is described separated with spaces. Commas separate the dataset names from the description of the errors

e.g.\tz\t[z with no errors - 1 column for dataset]
\tx,+-\t[x with symmetric errors - 2 columns for dataset]
\ty,+,-\t[y with asymmetric errors - 3 columns for dataset]
\tx[1:5],+,-\t[x_1 to x_5, each with asymmetric errors - 15 columns in total]
\tx y,+-\t[x with no errors, y with symmetric errors - 3 columns in total]
'''

class ImportDialog(ImportDialogBase):
    """1D data import dialog.

    Dialog allows user to choose a file and specify a descriptor, and
    optionally link to a file.
    """


    def __init__(self, parent, document):
        """Initialise dialog."""

        ImportDialogBase.__init__(self, parent, document)
        self.setCaption('Import data - Veusz')

        # edit the descriptor
        dhbox = qt.QHBox(self.widgetspace)
        dhbox.setSpacing(self.spacing)
        l = qt.QLabel('&Descriptor:', dhbox)
        self.descriptoredit = qt.QLineEdit(dhbox)
        l.setBuddy(self.descriptoredit)
        qt.QToolTip.add(self.descriptoredit,
                        'Names of columns when importing data, '
                        'e.g. "x y" or "a[:]"')
        
        # help for user
        l = qt.QLabel(_import1dhelp.strip(), self.widgetspace)
        l.setAlignment( l.alignment() | qt.Qt.WordBreak )
        
    def slotImport(self):
        """Do the importing"""
        
        # decode the descriptor
        descriptor = unicode( self.descriptoredit.text() )
        try:
            sr = document.SimpleRead(descriptor)
        except document.DescriptorError:
            mb = qt.QMessageBox("Veusz",
                                "Cannot interpret descriptor",
                                qt.QMessageBox.Warning,
                                qt.QMessageBox.Ok | qt.QMessageBox.Default,
                                qt.QMessageBox.NoButton,
                                qt.QMessageBox.NoButton,
                                self)
            mb.exec_loop()
            return

        # open file (shouldn't have got here if the file didn't exist
        filename = unicode( self.filenameedit.text() )
        filename = os.path.abspath(filename)
        ifile = open(filename, 'r')

        # open up an import stream
        stream = document.FileStream(ifile)

        # show a busy cursor
        qt.QApplication.setOverrideCursor( qt.QCursor(qt.Qt.WaitCursor) )

        # read the data
        sr.readData(stream)

        # restore the cursor
        qt.QApplication.restoreOverrideCursor()

        lines = []
        for var, count in sr.getInvalidConversions().items():
            if count != 0:
                lines.append('%i conversions failed for dataset "%s"' %
                             (count, var))
        if len(lines) != 0:
            lines.append('')

        # link the data to a file, if told to
        islinked = self.linkcheck.isChecked()
        if islinked:
            LF = document.LinkedFile(filename, descriptor)
        else:
            LF = None

        names = sr.setInDocument(self.document, linkedfile=LF)

        lines.append('Imported data for datasets:')
        for n in names:
            shape = self.document.getData(n).data.shape
            lines.append(' %s (%i items)' % (n, shape[0]))

        if LF != None:
            lines.append('')
            lines.append('Datasets were linked to file "%s"' % filename)

        self.previewedit.setText( '\n'.join(lines) )

_import2dhelp = '''

Data are read as a 2D matrix from the file.  X is read from left to
right, Y from bottom to top, by default. The default coordinate ranges
of data are 0 to M, 0 to N if the matrix is M*N in size.  These
parameters can be altered by including xrange A B, yrange A B,
invertrows, invertcols and transpose as lines in the data file.
Multiple datasets can be included by separating with blank lines.

'''

class ImportDialog2D(ImportDialogBase):
    """Import 2D datasets"""
    
    def __init__(self, parent, document):
        """Initialise dialog."""
        
        ImportDialogBase.__init__(self, parent, document)
        self.setCaption('Import 2D data - Veusz')

        ws = self.widgetspace

        # edit datasets
        h = qt.QHBox(ws)
        h.setSpacing(self.spacing)
        l = qt.QLabel('&Datasets:', h)
        self.datasetsedit = qt.QLineEdit(h)
        l.setBuddy(self.datasetsedit)
        qt.QToolTip.add(self.datasetsedit,
                        'A space separated list of datasets to import '
                        'from the file')

        grp = qt.QGroupBox("Import parameters", ws)
        grp.setColumns(1)

        # allow range of datasets to be changed
        self.rangeedits = []
        for v in ('X', 'Y'):
            h = qt.QHBox(grp)
            h.setSpacing(self.spacing)
            l = qt.QLabel('Range of &%s: ' % v, h)
            s = qt.QLineEdit(h)
            s.setValidator( qt.QDoubleValidator(self) )
            qt.QToolTip.add(s, 'Optionally specify the inclusive '
                            'range of the %s coordinate' % v)
            l.setBuddy(s)
            qt.QLabel('to', h)
            e = qt.QLineEdit(h)
            e.setValidator( qt.QDoubleValidator(self) )
            self.rangeedits += [s, e]

        self.invertrows = qt.QCheckBox('Invert &rows in file', grp)
        self.invertcols = qt.QCheckBox('Invert colu&mns in file', grp)
        self.transpose = qt.QCheckBox('&Transpose X and Y', grp)

        txt = _import2dhelp.replace('\n', ' ').strip()
        l = qt.QLabel(txt, grp)
        l.setAlignment( l.alignment() | qt.Qt.WordBreak )

    def slotImport(self):
        """Actually import the data."""

        # get datasets and split into a list
        datasets = unicode( self.datasetsedit.text() )
        datasets = re.split('[, ]+', datasets)

        # strip out blank items
        datasets = [i for i in datasets if i != '']

        # an obvious error...
        if len(datasets) == 0:
            self.previewedit.setText('At least one dataset needs to be '
                                     'specified')
            return
        
        # convert range parameters
        ranges = []
        for e in self.rangeedits:
            f = unicode(e.text())
            r = None
            try:
                r = float(f)
            except ValueError:
                pass
            ranges.append(r)

        # propagate settings from dialog to reader
        xrange = None
        yrange = None
        invertrows = None
        invertcols = None
        transpose = None
        linked = self.linkcheck.isChecked()
        
        if ranges[0] != None and ranges[1] != None:
            xrange = (ranges[0], ranges[1])
        if ranges[2] != None and ranges[3] != None:
            yrange = (ranges[2], ranges[3])
        if self.invertrows.isChecked():
            invertrows = True
        if self.invertcols.isChecked():
            invertcols = True
        if self.transpose.isChecked():
            transpose = True

        # get filename
        filename = unicode( self.filenameedit.text() )
        filename = os.path.abspath(filename)

        # show a busy cursor
        qt.QApplication.setOverrideCursor( qt.QCursor(qt.Qt.WaitCursor) )

        # loop over datasets and read...
        try:
            self.document.import2D(filename, datasets, xrange=xrange,
                                   yrange=yrange, invertrows=invertrows,
                                   invertcols=invertcols, transpose=transpose,
                                   linked=linked)
            output = 'Successfully read datasets %s' % (' ,'.join(datasets))
        except document.Read2DError, e:
            output = 'Error importing datasets:\n %s' % str(e)
                
        # restore the cursor
        qt.QApplication.restoreOverrideCursor()

        # show status in preview box
        self.previewedit.setText(output)
 
