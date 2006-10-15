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
import csv

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
                      self.slotUpdatePreview )

        self.connect( self.importbutton, qt4.SIGNAL('clicked()'),
                      self.slotImport)

        self.connect( self.methodtab, qt4.SIGNAL('currentChanged(int)'),
                      self.slotUpdatePreview )

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

    def slotUpdatePreview(self, *args):
        """Update preview window when filename or tab changed."""

        filename = unicode(self.filenameedit.text())
        tab = self.methodtab.currentIndex()
        if tab == 0:
            self.doPreviewStandard(filename)
        elif tab == 1:
            self.doPreviewCSV(filename)

    def doPreviewStandard(self, filename):
        """Standard preview - show start of text."""

        try:
            ifile = open(filename, 'r')
        except IOError:
            self.previewedit.setPlainText('')
            self.importbutton.setEnabled(False)
        else:
            text = ifile.read(2048)
            self.previewedit.setPlainText(text)
            self.importbutton.setEnabled(True)

    def doPreviewCSV(self, filename):
        """CSV preview - show first few rows"""

        t = self.previewtablecsv
        t.clear()
        t.setColumnCount(0)
        t.setRowCount(0)
        try:
            ifile = open(filename, 'r')

            # construct list of rows from input file
            reader = csv.reader(ifile)
            rows = []
            numcols = 0
            try:
                for i in xrange(10):
                    row = reader.next()
                    rows.append(row)
                    numcols = max(numcols, len(row))
                rows.append(['...'])
                numcols = max(numcols, 1)
            except StopIteration:
                pass
            numrows = len(rows)

        except IOError:
            self.importbutton.setEnabled(False)
        except csv.Error:
            self.importbutton.setEnabled(False)

        else:
            # fill up table
            t.setColumnCount(numcols)
            t.setRowCount(numrows)
            for r in xrange(numrows):
                for c in xrange(numcols):
                    if c < len(rows[r]):
                        item = qt4.QTableWidgetItem(str(rows[r][c]))
                        t.setItem(r, c, item)

            self.importbutton.setEnabled(True)

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

