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

_importcsvhelp='''
Comma Separated Value (CSV) files are often used to export data from applications such as Excel and OpenOffice. Veusz can read data from these files. At the top of each column of data, a dataset name can be given for the data below. Multiple datasets can be placed below each other if new names are given.

To import error bars, columns with the names "+", "-" or "+-" should be given in columns immediately to the right of the dataset, for positive, negative or symmetric errors.

Veusz can also read data organised in rows rather than columns.
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

        self.methodtab = qt.QTabWidget(self.widgetspace)
        self._addStandardTab()
        self._addCSVTab()

    def _addStandardTab(self):
        """Create tab for standard Veusz import."""

        tabbed = qt.QVBox(self.methodtab)
        tabbed.setSpacing(self.spacing)
        tabbed.setMargin(self.spacing)

        # edit the descriptor
        dhbox = qt.QHBox(tabbed)
        dhbox.setSpacing(self.spacing)
        l = qt.QLabel('&Descriptor:', dhbox)
        self.descriptoredit = qt.QLineEdit(dhbox)
        l.setBuddy(self.descriptoredit)
        qt.QToolTip.add(self.descriptoredit,
                        'Names of columns when importing data, '
                        'e.g. "x y" or "a[:]"')
        
        # help for user
        l = qt.QLabel(_import1dhelp.strip(), tabbed)
        l.setAlignment( l.alignment() | qt.Qt.WordBreak )
        
        self.blockcheck = qt.QCheckBox('Read data in bloc&ks',
                                       tabbed)
        qt.QToolTip.add(self.blockcheck,
                        'If this is selected, blank lines or the word\n'
                        '"no" are used to separate the file into blocks.\n'
                        'An underscore followed by the block number is\n'
                        'added to the dataset names')

        self.methodtab.addTab(tabbed, 'Standard')

    def _addCSVTab(self):
        """Create tab for CSV import."""

        tabbed = qt.QVBox(self.methodtab)
        tabbed.setSpacing(self.spacing)
        tabbed.setMargin(self.spacing)

        grd = qt.QGrid(2, tabbed)
        grd.setSpacing(self.spacing)
        l = qt.QLabel('&Direction:', grd)
        self.dirncombo = qt.QComboBox(False, grd)
        l.setBuddy(self.dirncombo)
        self.dirncombo.insertStrList(['Columns', 'Rows'])
        qt.QToolTip.add(self.dirncombo,
                        'The direction the data are organised in.')

        l = qt.QLabel('&Prefix:', grd)
        self.prefixedit = qt.QLineEdit(grd)
        l.setBuddy(self.prefixedit)
        qt.QToolTip.add(self.prefixedit,
                        'This prefix is prepended to the name of each \n'
                        'dataset imported from the file. This is useful \n'
                        'to make the names unique.')

        # help for user
        l = qt.QLabel(_importcsvhelp.strip(), tabbed)
        l.setAlignment( l.alignment() | qt.Qt.WordBreak )

        self.methodtab.addTab(tabbed, 'CSV')

    def slotImport(self):
        """Do the importing"""

        tabindex = self.methodtab.currentPageIndex()

        if tabindex == 0:
            # standard Veusz import
            self.importStandard()
        elif tabindex == 1:
            self.importCSV()
        else:
            assert False

    def _retnDatasetInfo(self, dsnames):
        """Return a list of information for the dataset names given."""
        
        lines = []
        lines.append('Imported data for datasets:')
        dsnames.sort()
        for name in dsnames:
            ds = self.document.getData(name)
            # build up description
            descr = [name]
            if ds.serr != None:
                descr.append('+-')
            if ds.perr != None:
                descr.append('+')
            if ds.nerr != None:
                descr.append('-')
            descr = ','.join(descr)
            lines.append(' %s (%i items)' % (descr, ds.data.shape[0]))

        linked = self.linkcheck.isChecked()
        filename = unicode( self.filenameedit.text() )
        filename = os.path.abspath(filename)

        # whether the data were linked
        if linked:
            lines.append('')
            lines.append('Datasets were linked to file "%s"' % filename)

        return lines

    def importCSV(self):
        """Import from CSV file."""

        # get various values
        inrows = self.dirncombo.currentItem() == 1
        prefix = unicode( self.prefixedit.text() )
        if len(prefix.strip()) == 0:
            prefix = None
        filename = unicode( self.filenameedit.text() )
        filename = os.path.abspath(filename)
        linked = self.linkcheck.isChecked()

        op = document.OperationDataImportCSV(filename, readrows=inrows,
                                             prefix=prefix, linked=linked)
        
        # show a busy cursor
        qt.QApplication.setOverrideCursor( qt.QCursor(qt.Qt.WaitCursor) )

        # actually import the data
        dsnames = self.document.applyOperation(op)
        
        # restore the cursor
        qt.QApplication.restoreOverrideCursor()

        # what datasets were imported
        lines = self._retnDatasetInfo(dsnames)

        self.previewedit.setText( '\n'.join(lines) )

    def importStandard(self):
        """Standard Veusz importing."""

        # convert controls to values
        descriptor = unicode( self.descriptoredit.text() )
        filename = unicode( self.filenameedit.text() )
        filename = os.path.abspath(filename)
        useblocks = self.blockcheck.isChecked()
        linked = self.linkcheck.isChecked()
        
        try:
            # construct operation. this checks the descriptor.
            op = document.OperationDataImport(descriptor, filename=filename,
                                              useblocks=useblocks, 
                                              linked=linked)

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

        # show a busy cursor
        qt.QApplication.setOverrideCursor( qt.QCursor(qt.Qt.WaitCursor) )

        # actually import the data
        dsnames = self.document.applyOperation(op)
        
        # restore the cursor
        qt.QApplication.restoreOverrideCursor()

        # tell the user what happened
        # failures in conversion
        lines = []
        for var, count in op.simpleread.getInvalidConversions().iteritems():
            if count != 0:
                lines.append('%i conversions failed for dataset "%s"' %
                             (count, var))
        if len(lines) != 0:
            lines.append('')
            
        lines += self._retnDatasetInfo(dsnames)

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
            op = document.OperationDataImport2D(datasets, filename=filename,
                                                xrange=xrange, yrange=yrange,
                                                invertrows=invertrows,
                                                invertcols=invertcols,
                                                transpose=transpose,
                                                linked=linked)
            readds = self.document.applyOperation(op)
            output = 'Successfully read datasets %s' % (' ,'.join(readds))
        except document.Read2DError, e:
            output = 'Error importing datasets:\n %s' % str(e)
                
        # restore the cursor
        qt.QApplication.restoreOverrideCursor()

        # show status in preview box
        self.previewedit.setText(output)
 
