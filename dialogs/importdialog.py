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
import sys

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils
import exceptiondialog

pyfits = None

class ImportStandardHelpDialog(qt4.QDialog):
    """Class to load help for standard veusz import."""
    def __init__(self, parent, *args):
        qt4.QDialog.__init__(self, parent, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'importhelp.ui'),
                   self)

class ImportDialog(qt4.QDialog):

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
                      qt4.SIGNAL('editTextChanged(const QString&)'),
                      self.slotUpdatePreview )

        self.connect( self.importbutton, qt4.SIGNAL('clicked()'),
                      self.slotImport)

        # user wants help about standard import
        self.connect( self.helpbutton, qt4.SIGNAL('clicked()'),
                      self.slotHelp )

        # notification tab has changed
        self.connect( self.methodtab, qt4.SIGNAL('currentChanged(int)'),
                      self.slotUpdatePreview )

        # if different items are selected in fits tab
        self.connect( self.fitshdulist, qt4.SIGNAL('itemSelectionChanged()'),
                      self.slotFitsUpdateCombos )
        self.connect( self.fitsdatasetname,
                      qt4.SIGNAL('textChanged(const QString&)'),
                      self.enableDisableImport )
        self.connect( self.fitsdatacolumn,
                      qt4.SIGNAL('currentIndexChanged(int)'),
                      self.enableDisableImport )

        # set up some validators for 2d edits
        dval = qt4.QDoubleValidator(self)
        for i in (self.twod_xminedit, self.twod_xmaxedit,
                  self.twod_yminedit, self.twod_ymaxedit):
            i.setValidator(dval)

        # change to tab last used
        self.methodtab.setCurrentIndex(
            setting.settingdb.get('import_lasttab', 0))

        # add completion for filename if there is support in version of qt
        # (requires qt >= 4.3)
        if hasattr(qt4, 'QDirModel'):
            c = self.filenamecompleter = qt4.QCompleter(self)
            model = qt4.QDirModel(c)
            c.setModel(model)
            self.filenameedit.setCompleter(c)

        # whether file import looks likely to work
        self.filepreviewokay = False

        # defaults for prefix and suffix
        self.prefixcombo.default = self.suffixcombo.default = ['', '$FILENAME']

        # default state for check boxes
        self.linkcheckbox.default = True
        self.blockcheckbox.default = False
        self.ignoretextcheckbox.default = True

    def slotBrowseClicked(self):
        """Browse for a data file."""

        fd = qt4.QFileDialog(self, 'Browse data file')
        fd.setFileMode( qt4.QFileDialog.ExistingFile )

        # use filename to guess a path if possible
        filename = unicode(self.filenameedit.text())
        if os.path.isdir(filename):
            ImportDialog.dirname = filename
        elif os.path.isdir( os.path.dirname(filename) ):
            ImportDialog.dirname = os.path.dirname(filename)

        fd.setDirectory(ImportDialog.dirname)

        # update filename if changed
        if fd.exec_() == qt4.QDialog.Accepted:
            ImportDialog.dirname = fd.directory().absolutePath()
            self.filenameedit.replaceAndAddHistory( fd.selectedFiles()[0] )

    def slotHelp(self):
        """Asked for help for standard import."""
        self.helpdialog = ImportStandardHelpDialog(self)
        self.helpdialog.show()

    def slotUpdatePreview(self, *args):
        """Update preview window when filename or tab changed."""

        # save so we can restore later
        tab = self.methodtab.currentIndex()
        setting.settingdb['import_lasttab'] = tab
        filename = unicode(self.filenameedit.text())

        # do correct preview
        self.filepreviewokay = (
            self.doPreviewStandard,
            self.doPreviewCSV,
            self.doPreviewFITS,
            self.doPreviewTwoD)[tab](filename)

        # enable or disable import button
        self.enableDisableImport()

    def doPreviewStandard(self, filename):
        """Standard preview - show start of text."""

        try:
            ifile = open(filename, 'rU')
            text = ifile.read(4096)+'\n'
            if len(ifile.read(1)) != 0:
                # if there is remaining data add ...
                text += '...\n'

            self.previewedit.setPlainText(text)
            return True

        except IOError:
            self.previewedit.setPlainText('')
            return False

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
            ifile = open(filename, 'rU')

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
            self.fitslabel.setText(
                'FITS file support requires that PyFITS is installed.'
                ' You can download it from'
                ' http://www.stsci.edu/resources/software_hardware/pyfits')
            return False
        
        # try to identify fits file
        try:
            ifile = open(filename,  'rU')
            line = ifile.readline()
            # is this a hack?
            if line.find('SIMPLE  =                    T') == -1:
                raise IOError
            ifile.close()
        except IOError:
            return False

        self.updateFITSView(filename)
        return True

    def doPreviewTwoD(self, filename):
        """Preview 2d dataset files."""
        
        try:
            ifile = open(filename, 'rU')
            text = ifile.read(4096)+'\n'
            if len(ifile.read(1)) != 0:
                # if there is remaining data add ...
                text += '...\n'
            self.twod_previewedit.setPlainText(text)
            return True

        except IOError:
            self.twod_previewedit.setPlainText('')
            return False

    def updateFITSView(self, filename):
        """Update the fits file details in the import dialog."""
        f = pyfits.open(str(filename), 'readonly')
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
                naxis = header['NAXIS']
                if naxis ==2:
                    data = ['image']
                else:
                    data = ['invalidimage']
                dims = [str(header['NAXIS%i' % (i+1)]) for i in xrange(naxis)]
                dims = '*'.join(dims)
                if dims:
                    dims = '(%s)' % dims
                descr = '%iD image %s' % (naxis, dims)

            hduitem = qt4.QTreeWidgetItem([str(hdunum), hdu.name, descr])
            items.append(hduitem)
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

        self.enableDisableImport()

    def checkFitsEnable(self):
        """Check validity of Fits import."""

        items = self.fitshdulist.selectedItems()
        if len(items) != 0:
            item = items[0]
            hdunum = int( str(item.text(0)) )

            # any name for the dataset?
            if not unicode(self.fitsdatasetname.text()):
                return False

            # if a table, need selected item
            data = self.fitsitemdata[hdunum]
            if data[0] != 'image' and self.fitsdatacolumn.currentIndex() == 0:
                return False
            
            return True
        return False

    def enableDisableImport(self, *args):
        """Disable or enable import button if allowed."""

        enabled = self.filepreviewokay

        # checks specific to import mode
        if enabled:
            tabindex = self.methodtab.currentIndex()
            if tabindex == 2:  # fits
                enabled = self.checkFitsEnable()

        # actually enable or disable import button
        self.importbutton.setEnabled(enabled)

    def slotImport(self):
        """Do the importing"""

        tabindex = self.methodtab.currentIndex()
        filename = unicode( self.filenameedit.text() )
        filename = os.path.abspath(filename)
        linked = self.linkcheckbox.isChecked()

        # import according to tab selected
        try:
            qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )
            (self.importStandard,
             self.importCSV,
             self.importFits,
             self.importTwoD)[tabindex](filename, linked)
            qt4.QApplication.restoreOverrideCursor()
        except Exception:
            qt4.QApplication.restoreOverrideCursor()

            # show exception dialog
            d = exceptiondialog.ExceptionDialog(sys.exc_info(), self)
            d.exec_()

    def _retnDatasetInfo(self, dsnames):
        """Return a list of information for the dataset names given."""
        
        lines = ['Imported data for datasets:']
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

    def getPrefixSuffix(self, filename):
        """Get prefix and suffix values."""
        f = utils.escapeDatasetName( os.path.basename(filename) )
        prefix = unicode( self.prefixcombo.lineEdit().text() )
        prefix = prefix.replace('$FILENAME', f)
        suffix = unicode( self.suffixcombo.lineEdit().text() )
        suffix = suffix.replace('$FILENAME', f)
        return prefix, suffix

    def importStandard(self, filename, linked):
        """Standard Veusz importing."""

        # convert controls to values
        descriptor = unicode( self.descriptoredit.text() )
        useblocks = self.blockcheckbox.isChecked()
        ignoretext = self.ignoretextcheckbox.isChecked()

        # substitute filename if required
        prefix, suffix = self.getPrefixSuffix(filename)

        try:
            # construct operation. this checks the descriptor.
            op = document.OperationDataImport(descriptor, filename=filename,
                                              useblocks=useblocks, 
                                              linked=linked,
                                              prefix=prefix, suffix=suffix,
                                              ignoretext=ignoretext)

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

        # actually import the data
        dsnames = self.document.applyOperation(op)
        
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
        prefix, suffix = self.getPrefixSuffix(filename)

        op = document.OperationDataImportCSV(filename, readrows=inrows,
                                             prefix=prefix, suffix=suffix,
                                             linked=linked)
        
        # actually import the data
        dsnames = self.document.applyOperation(op)
        
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
        # get values of prefix and suffix set in dialog
        prefix, suffix = self.getPrefixSuffix(filename)
        name = prefix + name + suffix

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

        # construct operation to import fits
        op = document.OperationDataImportFITS(name, filename, hdunum,
                                              datacol=cols[0],
                                              symerrcol=cols[1],
                                              poserrcol=cols[2],
                                              negerrcol=cols[3],
                                              linked=linked)

        # actually do the import
        self.document.applyOperation(op)

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
        datasets = [d for d in datasets if d != '']

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
        rangex = None
        rangey = None
        if ranges[0] is not None and ranges[1] is not None:
            rangex = (ranges[0], ranges[1])
        if ranges[2] is not None and ranges[3] is not None:
            rangey = (ranges[2], ranges[3])

        invertrows = self.twod_invertrowscheck.isChecked()
        invertcols = self.twod_invertcolscheck.isChecked()
        transpose = self.twod_transposecheck.isChecked()

        # substitute filename if required
        prefix, suffix = self.getPrefixSuffix(filename)

        # loop over datasets and read...
        try:
            op = document.OperationDataImport2D(datasets, filename=filename,
                                                xrange=rangex, yrange=rangey,
                                                invertrows=invertrows,
                                                invertcols=invertcols,
                                                transpose=transpose,
                                                prefix=prefix, suffix=suffix,
                                                linked=linked)
            readds = self.document.applyOperation(op)

            output = ['Successfully read datasets:']
            for ds in readds:
                output.append(' %s' % self.document.data[ds].description(showlinked=False))
            
            output = '\n'.join(output)
        except document.Read2DError, e:
            output = 'Error importing datasets:\n %s' % str(e)
                
        # show status in preview box
        self.twod_previewedit.setPlainText(output)
 
