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

import veusz.qtall as qt4

import veusz.document as document
import veusz.setting as setting

class ImportDialog2(qt4.QDialog):

    dirname = '.'

    def __init__(self, parent, document, *args):

        qt4.QDialog.__init__(self, parent, *args)
        qt4.loadUi(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'import.ui'),
                   self)
        self.document = document

        self.connect(self.browsebutton, qt4.SIGNAL('clicked()'),
                     self.slotBrowseClicked)

        self.connect( self.filenameedit,
                      qt4.SIGNAL('textChanged(const QString&)'),
                      self.slotFilenameChanged )

        self.connect( self.importbutton, qt4.SIGNAL('clicked()'),
                      self.slotImport)

    def slotBrowseClicked(self):
        """Browse for a data file."""

        fd = qt4.QFileDialog(self, 'Browse data file')
        fd.setFileMode( qt4.QFileDialog.ExistingFile )

        # use filename to guess a path if possible
        filename = unicode(self.filenameedit.text())
        if os.path.isdir(filename):
            ImportDialog2.dirname = filename
        elif os.path.isdir( os.path.dirname(filename) ):
            ImportDialog2.dirname = os.path.dirname(filename)

        fd.setDirectory(ImportDialog2.dirname)

        # update filename if changed
        if fd.exec_() == qt4.QDialog.Accepted:
            ImportDialog2.dirname = fd.directory().absolutePath()
            self.filenameedit.setText( fd.selectedFiles()[0] )

    def slotFilenameChanged(self, filename):
        """Update preview window when filename changed."""

        try:
            ifile = open(unicode(filename), 'r')
            text = ifile.read(2048)
            self.previewedit.setPlainText(text)
            self.importbutton.setEnabled(True)

        except IOError:
            self.previewedit.setPlainText('<preview>')
            self.importbutton.setEnabled(False)

    def slotImport(self):
        """Do the importing"""

        tabindex = self.methodtab.currentIndex()

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
            if ds.serr is not None:
                descr.append('+-')
            if ds.perr is not None:
                descr.append('+')
            if ds.nerr is not None:
                descr.append('-')
            descr = ','.join(descr)
            lines.append(' %s (%i items)' % (descr, ds.data.shape[0]))

        linked = self.linkcheckbox.isChecked()
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
        inrows = self.directioncombo.currentIndex() == 1
        prefix = unicode( self.prefixedit.text() )
        if len(prefix.strip()) == 0:
            prefix = None
        filename = unicode( self.filenameedit.text() )
        filename = os.path.abspath(filename)
        linked = self.linkcheckbox.isChecked()

        op = document.OperationDataImportCSV(filename, readrows=inrows,
                                             prefix=prefix, linked=linked)
        
        # show a busy cursor
        qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )

        # actually import the data
        dsnames = self.document.applyOperation(op)
        
        # restore the cursor
        qt4.QApplication.restoreOverrideCursor()

        # what datasets were imported
        lines = self._retnDatasetInfo(dsnames)

        self.previewedit.setPlainText( '\n'.join(lines) )

    def importStandard(self):
        """Standard Veusz importing."""

        # convert controls to values
        descriptor = unicode( self.descriptoredit.text() )
        filename = unicode( self.filenameedit.text() )
        filename = os.path.abspath(filename)
        useblocks = self.blockcheckbox.isChecked()
        linked = self.linkcheckbox.isChecked()
        
        try:
            # construct operation. this checks the descriptor.
            op = document.OperationDataImport(descriptor, filename=filename,
                                              useblocks=useblocks, 
                                              linked=linked)

        except document.DescriptorError:
            mb = qt4.QMessageBox("Veusz",
                                "Cannot interpret descriptor",
                                qt4.QMessageBox.Warning,
                                qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                                qt4.QMessageBox.NoButton,
                                qt4.QMessageBox.NoButton,
                                self)
            mb.exec_()
            return

        # show a busy cursor
        qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )

        # actually import the data
        dsnames = self.document.applyOperation(op)
        
        # restore the cursor
        qt4.QApplication.restoreOverrideCursor()

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

        self.previewedit.setPlainText( '\n'.join(lines) )

class ImportDialogBase(qt4.QDialog):
    """Base class for data importing dialogs.

    A lot of the functionality is replicated otherwise
    """

    dirname = '.'

    def __init__(self, parent, document):
        qt4.QDialog.__init__(self, parent)
        self.document = document

        self.spacing = self.fontMetrics().height() / 2

        # layout for dialog
        self.layout = qt4.QVBoxLayout(self)
 
        # change the filename
        fnhbox = qt4.QWidget(self)
        hl = qt4.QHBoxLayout(fnhbox)
        self.layout.addWidget( fnhbox )
        l = qt4.QLabel('&Filename:', fnhbox)
        hl.addWidget(l)
        self.filenameedit = qt4.QLineEdit(fnhbox)
        hl.addWidget(self.filenameedit)
        l.setBuddy(self.filenameedit)
        self.filenameedit.setToolTip('The name of the file to import data from')

        qt4.QObject.connect( self.filenameedit,
                             qt4.SIGNAL('textChanged(const QString&)'),
                             self.slotFilenameChanged )
        
        browsebutton = qt4.QPushButton("&Browse...", fnhbox)
        qt4.QObject.connect( browsebutton, qt4.SIGNAL('clicked()'),
                            self.slotBrowse )

        # preview the file
        self.previewedit = qt4.QTextEdit(self)
        self.layout.addWidget( self.previewedit )
        self.previewedit.setReadOnly(True)
        self.previewedit.setWordWrapMode(qt4.QTextOption.NoWrap)
        self.previewedit.setPlainText('<preview>')
        self.previewedit.setFontFamily('Fixed')
        self.previewedit.setToolTip('A preview of the contents of the file')
        
        # space for descendents of this class to add widgets
        self.widgetspace = qt4.QWidget(self)
        self.widgetspacelayout = qt4.QVBoxLayout(self.widgetspace)
        self.layout.addWidget(self.widgetspace)

        # allow links from the file, so the data are reread from the file
        # on reloading
        self.linkcheck = qt4.QCheckBox('&Link the datasets to the file',
                                      self)
        self.layout.addWidget( self.linkcheck )
        self.linkcheck.setToolTip('Linked datasets are not saved to a document file,'
                                  '\nbut are reloaded from the linked data file.')

        # buttons
        w = qt4.QWidget(self)
        self.layout.addWidget(w)
        w.setSizePolicy(qt4.QSizePolicy.Expanding, qt4.QSizePolicy.Fixed)
        l = qt4.QHBoxLayout(w)
        l.addItem( qt4.QSpacerItem(1, 1, qt4.QSizePolicy.Expanding,
                                  qt4.QSizePolicy.Minimum) )
        
        b = qt4.QPushButton("&Import", w)
        self.importbutton = b
        b.setEnabled(False)
        b.setDefault(True)
        l.addWidget(b)
        self.connect(b, qt4.SIGNAL('clicked()'), self.slotImport)
        b = qt4.QPushButton("&Close", w)
        l.addWidget(b)
        self.connect(b, qt4.SIGNAL('clicked()'), self.slotClose)

    def sizeHint(self):
        """Returns recommended size of dialog."""
        return qt4.QSize(600, 400)

    def closeEvent(self, evt):
        """Called when the window closes."""

        # store the current geometry in the settings database
        geometry = ( self.x(), self.y(), self.width(), self.height() )
        setting.settingdb['geometry_importdialog'] = geometry

        qt4.QDialog.closeEvent(self, evt)

    def showEvent(self, evt):
        """Restoring window geometry if possible."""

        # if we can restore the geometry, do so
        if 'geometry_importdialog' in setting.settingdb:
            geometry =  setting.settingdb['geometry_importdialog']
            self.resize( qt4.QSize(geometry[2], geometry[3]) )
            self.move( qt4.QPoint(geometry[0], geometry[1]) )

        qt4.QDialog.showEvent(self, evt)

    def slotBrowse(self):
        """Browse button pressed in dialog."""

        fd = qt4.QFileDialog(self, 'import dialog', True)
        fd.setMode( qt4.QFileDialog.ExistingFile )
        fd.setWindowTitle('Browse data file')

        # use filename to guess a path if possible
        filename = unicode(self.filenameedit.text())
        if os.path.isdir(filename):
            ImportDialogBase.dirname = filename
        elif os.path.isdir( os.path.dirname(filename) ):
            ImportDialogBase.dirname = os.path.dirname(filename)
        
        fd.setDir( ImportDialogBase.dirname )
        
        # okay was selected
        if fd.exec_() == qt4.QDialog.Accepted:
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
        self.setWindowTitle('Import data - Veusz')

        self.methodtab = qt4.QTabWidget(self.widgetspace)
        self.widgetspacelayout.addWidget(self.methodtab)
        self._addStandardTab()
        self._addCSVTab()

    def _addStandardTab(self):
        """Create tab for standard Veusz import."""

        tabbed = qt4.QWidget(self.methodtab)
        tabbedlayout = qt4.QVBoxLayout(tabbed)

        # edit the descriptor
        dhbox = qt4.QWidget(tabbed)
        dhboxl = qt4.QHBoxLayout(dhbox)
        tabbedlayout.addWidget(dhbox)
        l = qt4.QLabel('&Descriptor:', dhbox)
        dhboxl.addWidget(l)
        self.descriptoredit = qt4.QLineEdit(dhbox)
        dhboxl.addWidget(self.descriptoredit)
        l.setBuddy(self.descriptoredit)
        self.descriptoredit.setToolTip('Names of columns when importing data, '
                                       'e.g. "x y" or "a[:]"')
        
        # help for user
        l = qt4.QLabel(_import1dhelp.strip(), tabbed)
        tabbedlayout.addWidget(l)
        l.setAlignment( l.alignment() | qt4.Qt.WordBreak )
        
        self.blockcheck = qt4.QCheckBox('Read data in bloc&ks',
                                       tabbed)
        tabbedlayout.addWIdget(self.blockcheck)
        self.blockcheck.setToolTip('If this is selected, blank lines or the word\n'
                                   '"no" are used to separate the file into blocks.\n'
                                                        'An underscore followed by the block number is\n'
                                   'added to the dataset names')

        self.methodtab.addTab(tabbed, 'Standard')

    def _addCSVTab(self):
        """Create tab for CSV import."""

        tabbed = qt4.QVBox(self.methodtab)
        tabbed.setSpacing(self.spacing)
        tabbed.setMargin(self.spacing)

        grd = qt4.QGrid(2, tabbed)
        grd.setSpacing(self.spacing)
        l = qt4.QLabel('&Direction:', grd)
        self.dirncombo = qt4.QComboBox(False, grd)
        l.setBuddy(self.dirncombo)
        self.dirncombo.insertStrList(['Columns', 'Rows'])
        qt4.QToolTip.add(self.dirncombo,
                        'The direction the data are organised in.')

        l = qt4.QLabel('&Prefix:', grd)
        self.prefixedit = qt4.QLineEdit(grd)
        l.setBuddy(self.prefixedit)
        qt4.QToolTip.add(self.prefixedit,
                        'This prefix is prepended to the name of each \n'
                        'dataset imported from the file. This is useful \n'
                        'to make the names unique.')

        # help for user
        l = qt4.QLabel(_importcsvhelp.strip(), tabbed)
        l.setAlignment( l.alignment() | qt4.Qt.WordBreak )

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
            if ds.serr is not None:
                descr.append('+-')
            if ds.perr is not None:
                descr.append('+')
            if ds.nerr is not None:
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
        qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )

        # actually import the data
        dsnames = self.document.applyOperation(op)
        
        # restore the cursor
        qt4.QApplication.restoreOverrideCursor()

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
            mb = qt4.QMessageBox("Veusz",
                                "Cannot interpret descriptor",
                                qt4.QMessageBox.Warning,
                                qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                                qt4.QMessageBox.NoButton,
                                qt4.QMessageBox.NoButton,
                                self)
            mb.exec_()
            return

        # show a busy cursor
        qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )

        # actually import the data
        dsnames = self.document.applyOperation(op)
        
        # restore the cursor
        qt4.QApplication.restoreOverrideCursor()

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
        self.setWindowTitle('Import 2D data - Veusz')

        ws = self.widgetspace

        # edit datasets
        h = qt4.QHBox(ws)
        h.setSpacing(self.spacing)
        l = qt4.QLabel('&Datasets:', h)
        self.datasetsedit = qt4.QLineEdit(h)
        l.setBuddy(self.datasetsedit)
        qt4.QToolTip.add(self.datasetsedit,
                        'A space separated list of datasets to import '
                        'from the file')

        grp = qt4.QGroupBox("Import parameters", ws)
        grp.setColumns(1)

        # allow range of datasets to be changed
        self.rangeedits = []
        for v in ('X', 'Y'):
            h = qt4.QHBox(grp)
            h.setSpacing(self.spacing)
            l = qt4.QLabel('Range of &%s: ' % v, h)
            s = qt4.QLineEdit(h)
            s.setValidator( qt4.QDoubleValidator(self) )
            qt4.QToolTip.add(s, 'Optionally specify the inclusive '
                            'range of the %s coordinate' % v)
            l.setBuddy(s)
            qt4.QLabel('to', h)
            e = qt4.QLineEdit(h)
            e.setValidator( qt4.QDoubleValidator(self) )
            self.rangeedits += [s, e]

        self.invertrows = qt4.QCheckBox('Invert &rows in file', grp)
        self.invertcols = qt4.QCheckBox('Invert colu&mns in file', grp)
        self.transpose = qt4.QCheckBox('&Transpose X and Y', grp)

        txt = _import2dhelp.replace('\n', ' ').strip()
        l = qt4.QLabel(txt, grp)
        l.setAlignment( l.alignment() | qt4.Qt.WordBreak )

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
        
        if ranges[0] is not None and ranges[1] is not None:
            xrange = (ranges[0], ranges[1])
        if ranges[2] is not None and ranges[3] is not None:
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
        qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )

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
        qt4.QApplication.restoreOverrideCursor()

        # show status in preview box
        self.previewedit.setText(output)
 
