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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

# $Id$

"""Module for implementing dialog boxes for importing data in Veusz."""

import os.path
import re
import csv

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

pyfits = None

class ImportDialog2(qt4.QDialog):

    dirname = '.'

    def __init__(self, parent, document, *args):

        qt4.QDialog.__init__(self, parent, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
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

        # notification tab has changed
        self.connect( self.methodtab, qt4.SIGNAL('currentChanged(int)'),
                      self.slotUpdatePreview )

        # if different items are selected in fits tab
        self.connect( self.fitshdulist, qt4.SIGNAL('itemSelectionChanged()'),
                      self.slotFitsUpdateCombos )
        self.connect( self.fitsdatasetname,
                      qt4.SIGNAL('textChanged(const QString&)'),
                      self.slotFitsCheckValid )
        self.connect( self.fitsdatacolumn,
                      qt4.SIGNAL('currentIndexChanged(int)'),
                      self.slotFitsCheckValid )

        # set up some validators for 2d edits
        dval = qt4.QDoubleValidator(self)
        for i in (self.twod_xminedit, self.twod_xmaxedit,
                  self.twod_yminedit, self.twod_ymaxedit):
            i.setValidator(dval)

        # change to tab last used
        self.methodtab.setCurrentIndex(
            setting.settingdb.get('import_lasttab', 0))

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
            okay = self.doPreviewStandard(filename)
        elif tab == 1:
            okay = self.doPreviewCSV(filename)
        elif tab == 2:
            okay = self.doPreviewFITS(filename)
        elif tab == 3:
            okay = self.doPreviewTwoD(filename)
        else:
            assert False

        # enable import button if it looks okay
        if okay is not None:
            self.importbutton.setEnabled(okay)

        # save so we can restore later
        setting.settingdb['import_lasttab'] = tab

    def doPreviewStandard(self, filename):
        """Standard preview - show start of text."""

        try:
            ifile = open(filename, 'r')
        except IOError:
            self.previewedit.setPlainText('')
            return False

        text = ifile.read(2048)+'\n...\n'
        self.previewedit.setPlainText(text)
        return True

    def doPreviewCSV(self, filename):
        """CSV preview - show first few rows"""

        t = self.previewtablecsv
        t.verticalHeader().show() # restore from a previous import
        t.horizontalHeader().show()
        t.horizontalHeader().setStretchLastSection(False)
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
            return False
        except csv.Error:
            return False

        # fill up table
        t.setColumnCount(numcols)
        t.setRowCount(numrows)
        for r in xrange(numrows):
            for c in xrange(numcols):
                if c < len(rows[r]):
                    item = qt4.QTableWidgetItem(str(rows[r][c]))
                    t.setItem(r, c, item)

        return True

    def doPreviewFITS(self, filename):
        """Set up controls for FITS file."""

        # load pyfits if available
        global pyfits
        if pyfits is None:
            try:
                import pyfits as PF
                pyfits = PF
            except ImportError:
                pyfits = None

        # if it isn't
        if pyfits is None:
            self.fitslabel.setText('FITS file support requires that PyFITS is installed.'
                                   ' You can download it from'
                                   ' http://www.stsci.edu/resources/software_hardware/pyfits')
            self.importbutton.setEnabled(False)
            return False
        
        # try to identify fits file
        try:
            ifile = open(filename)
            line = ifile.readline()
            # is this a hack?
            if line.find('SIMPLE  =                    T') == -1:
                raise IOError
            ifile.close()
        except IOError:
            return False

        self.updateFITSView(filename)

        return None

    def doPreviewTwoD(self, filename):
        """Preview 2d dataset files."""
        
        try:
            ifile = open(filename, 'r')
        except IOError:
            self.twod_previewedit.setPlainText('')
            return False

        text = ifile.read(2048) + '\n...\n'
        self.twod_previewedit.setPlainText(text)
        return True

    def updateFITSView(self, filename):
        """Update the fits file details in the import dialog."""
        f = pyfits.open(filename, 'readonly')
        l = self.fitshdulist
        l.clear()

        # this is so we can lookup item attributes later
        self.fitsitemdata = []
        items = []
        for hdunum, hdu in enumerate(f):
            header = hdu.header
            hduitem = qt4.QTreeWidgetItem([str(hdunum), hdu.name])
            data = []
            try:
                # if this fails, show an image
                cols = hdu.get_coldefs()

                # it's a table
                data = ['table', cols]
                rows = header['NAXIS2']
                descr = 'Table (%i rows)' % rows

            except AttributeError:
                # this is an image
                data = ['image']
                naxis = header['NAXIS']
                dims = [str(header['NAXIS%i' % (i+1)]) for i in xrange(naxis)]
                dims = '*'.join(dims)
                descr = '%iD image (%s)' % (naxis, dims)

            hduitem = qt4.QTreeWidgetItem([str(hdunum), hdu.name, descr])
            items.append(hduitem)
            #items.insert(0, hduitem)
            self.fitsitemdata.append(data)

        if items:
            l.addTopLevelItems(items)
            l.setCurrentItem(items[0])

    def slotFitsUpdateCombos(self):
        """Update list of fits columns when new item is selected."""
        
        items = self.fitshdulist.selectedItems()
        if len(items) != 0:
            item = items[0]
            hdunum = int( str(item.text(0)) )
        else:
            item = None
            hdunum = -1

        cols = ['N/A']
        enablecolumns = False
        if hdunum >= 0:
            data = self.fitsitemdata[hdunum]
            if data[0] == 'table':
                enablecolumns = True
                cols = ['None']
                cols += ['%s (%s)' %
                         (i.name, i.format) for i in data[1]]
        
        for c in ('data', 'sym', 'pos', 'neg'):
            cntrl = getattr(self, 'fits%scolumn' % c)
            cntrl.setEnabled(enablecolumns)
            cntrl.clear()
            cntrl.addItems(cols)

        # update enable icon as appropriate
        self.slotFitsCheckValid()

    def slotFitsCheckValid(self, *args):
        """Check validity of Fits import."""

        enableimport = True

        items = self.fitshdulist.selectedItems()
        if len(items) != 0:
            enableimport = True
            item = items[0]
            hdunum = int( str(item.text(0)) )

            # any name for the dataset?
            if unicode(self.fitsdatasetname.text()) == '':
                enableimport = False

            # if a table, need selected item
            data = self.fitsitemdata[hdunum]
            if data[0] == 'table' and self.fitsdatacolumn.currentIndex() == 0:
                enableimport = False

        else:
            enableimport = False

        self.importbutton.setEnabled(enableimport)

    def slotImport(self):
        """Do the importing"""

        tabindex = self.methodtab.currentIndex()
        filename = unicode( self.filenameedit.text() )
        filename = os.path.abspath(filename)
        linked = self.linkcheckbox.isChecked()

        # import according to tab selected
        if tabindex == 0:
            self.importStandard(filename, linked)
        elif tabindex == 1:
            self.importCSV(filename, linked)
        elif tabindex == 2:
            self.importFits(filename, linked)
        elif tabindex == 3:
            self.importTwoD(filename, linked)
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
            lines.append( ' %s' % ds.description(showlinked=False) )

        linked = self.linkcheckbox.isChecked()
        filename = unicode( self.filenameedit.text() )
        filename = os.path.abspath(filename)

        # whether the data were linked
        if linked:
            lines.append('')
            lines.append('Datasets were linked to file "%s"' % filename)

        return lines

    def importStandard(self, filename, linked):
        """Standard Veusz importing."""

        # convert controls to values
        descriptor = unicode( self.descriptoredit.text() )
        useblocks = self.blockcheckbox.isChecked()
        
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

    def importCSV(self, filename, linked):
        """Import from CSV file."""

        # get various values
        inrows = self.directioncombo.currentIndex() == 1
        prefix = unicode( self.prefixedit.text() )
        if len(prefix.strip()) == 0:
            prefix = None

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

        t = self.previewtablecsv
        t.verticalHeader().hide()
        t.horizontalHeader().hide()
        t.horizontalHeader().setStretchLastSection(True)

        t.clear()
        t.setColumnCount(1)
        t.setRowCount(len(lines))
        for i, l in enumerate(lines):
            item = qt4.QTableWidgetItem(l)
            t.setItem(i, 0, item)

    def importFits(self, filename, linked):
        """Import fits file."""
        
        item = self.fitshdulist.selectedItems()[0]
        hdunum = int( str(item.text(0)) )
        data = self.fitsitemdata[hdunum]

        name = unicode(self.fitsdatasetname.text())

        if data[0] == 'table':
            # get list of appropriate columns
            cols = []

            # get data from controls
            for c in ('data', 'sym', 'pos', 'neg'):
                cntrl = getattr(self, 'fits%scolumn' % c)
                
                i = cntrl.currentIndex()
                if i == 0:
                    cols.append(None)
                else:
                    cols.append(data[1][i-1].name)
                    
        else:
            # item is an image, so no columns
            cols = [None]*4

        op = document.OperationDataImportFITS(name, filename, hdunum,
                                              datacol=cols[0],
                                              symerrcol=cols[1],
                                              poserrcol=cols[2],
                                              negerrcol=cols[3],
                                              linked=linked)

        # show a busy cursor
        qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )

        # actually do the import
        self.document.applyOperation(op)
        
        # restore the cursor
        qt4.QApplication.restoreOverrideCursor()

        # inform user
        self.fitsimportstatus.setText("Imported dataset '%s'" % name)
        qt4.QTimer.singleShot(2000, self.fitsimportstatus.clear)

    def importTwoD(self, filename, linked):
        """Import from 2D file."""

        # this really needs improvement...

        # get datasets and split into a list
        datasets = unicode( self.twod_datasetsedit.text() )
        datasets = re.split('[, ]+', datasets)

        # strip out blank items
        datasets = [i for i in datasets if i != '']

        # an obvious error...
        if len(datasets) == 0:
            self.twod_previewedit.setPlainText('At least one dataset needs to '
                                               'be specified')
            return
        
        # convert range parameters
        ranges = []
        for e in (self.twod_xminedit, self.twod_xmaxedit,
                  self.twod_yminedit, self.twod_ymaxedit):
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
        if ranges[0] is not None and ranges[1] is not None:
            xrange = (ranges[0], ranges[1])
        if ranges[2] is not None and ranges[3] is not None:
            yrange = (ranges[2], ranges[3])

        invertrows = self.twod_invertrowscheck.isChecked()
        invertcols = self.twod_invertcolscheck.isChecked()
        transpose = self.twod_transposecheck.isChecked()

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

            output = ['Successfully read datasets:']
            for ds in readds:
                output.append(' %s' % self.document.data[ds].description(showlinked=False))
            
            output = '\n'.join(output)
        except document.Read2DError, e:
            output = 'Error importing datasets:\n %s' % str(e)
                
        # restore the cursor
        qt4.QApplication.restoreOverrideCursor()

        # show status in preview box
        self.twod_previewedit.setPlainText(output)
 
