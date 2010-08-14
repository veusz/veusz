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
import veusz.plugins as plugins
import exceptiondialog
from veuszdialog import VeuszDialog

class ImportTab(qt4.QWidget):
    """Tab for a particular import type."""

    resource = ''

    def __init__(self, importdialog, *args):
        """Initialise dialog. importdialog is the import dialog itself."""
        qt4.QWidget.__init__(self, *args)
        self.dialog = importdialog
        self.uiloaded = False

    def loadUi(self):
        """Load up UI file."""
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                self.resource), self)
        self.uiloaded = True

    def reset(self):
        """Reset controls to initial conditions."""
        pass

    def doPreview(self, filename, encoding):
        """Update the preview window, returning whether import
        should be attempted."""
        pass

    def doImport(self, filename, linked, encoding, prefix, suffix):
        """Do the import iteself."""
        pass

    def okToImport(self):
        """Secondary check (after preview) for enabling import button."""
        return True

class ImportTabStandard(ImportTab):
    """Standard import format tab."""

    resource = 'import_standard.ui'

    def loadUi(self):
        """Load widget and setup controls."""
        ImportTab.loadUi(self)
        self.connect( self.helpbutton, qt4.SIGNAL('clicked()'),
                      self.slotHelp )
        self.blockcheckbox.default = False
        self.ignoretextcheckbox.default = True

    def reset(self):
        """Reset controls."""
        self.blockcheckbox.setChecked(False)
        self.ignoretextcheckbox.setChecked(True)
        self.descriptoredit.setEditText("")

    def slotHelp(self):
        """Asked for help."""
        d = VeuszDialog(self.dialog.mainwindow, 'importhelp.ui')
        self.dialog.mainwindow.showDialog(d)

    def doPreview(self, filename, encoding):
        """Standard preview - show start of text."""

        try:
            ifile = utils.openEncoding(filename, encoding)
            text = ifile.read(4096)+'\n'
            if len(ifile.read(1)) != 0:
                # if there is remaining data add ...
                text += '...\n'

            self.previewedit.setPlainText(text)
            return True
        except (UnicodeError, IOError):
            self.previewedit.setPlainText('')
            return False

    def doImport(self, doc, filename, linked, encoding, prefix, suffix):
        """Standard Veusz importing."""

        # convert controls to values
        descriptor = unicode( self.descriptoredit.text() )
        useblocks = self.blockcheckbox.isChecked()
        ignoretext = self.ignoretextcheckbox.isChecked()

        try:
            # construct operation. this checks the descriptor.
            op = document.OperationDataImport(descriptor, filename=filename,
                                              useblocks=useblocks, 
                                              linked=linked,
                                              prefix=prefix, suffix=suffix,
                                              ignoretext=ignoretext,
                                              encoding=encoding)

        except document.DescriptorError:
            qt4.QMessageBox.warning(self, "Veusz",
                                    "Cannot interpret descriptor")
            return

        # actually import the data
        dsnames = doc.applyOperation(op)
        
        # tell the user what happened
        # failures in conversion
        lines = []
        for var, count in op.simpleread.getInvalidConversions().iteritems():
            if count != 0:
                lines.append('%i conversions failed for dataset "%s"' %
                             (count, var))
        if len(lines) != 0:
            lines.append('')
            
        lines += self.dialog.retnDatasetInfo(dsnames)

        self.previewedit.setPlainText( '\n'.join(lines) )

class ImportTabCSV(ImportTab):
    """For importing data from CSV files."""

    resource = 'import_csv.ui'

    def loadUi(self):
        """Load user interface and setup panel."""
        ImportTab.loadUi(self)
        self.connect( self.csvhelpbutton, qt4.SIGNAL('clicked()'),
                      self.slotHelp )
        self.connect( self.csvdelimitercombo,
                      qt4.SIGNAL('editTextChanged(const QString&)'),
                      self.dialog.slotUpdatePreview )
        self.connect( self.csvtextdelimitercombo,
                      qt4.SIGNAL('editTextChanged(const QString&)'),
                      self.dialog.slotUpdatePreview )
        self.csvdelimitercombo.default = [
            ',', '{tab}', '{space}', '|', ':', ';']
        self.csvtextdelimitercombo.default = ['"', "'"]

    def reset(self):
        """Reset controls."""
        self.csvdelimitercombo.setEditText(",")
        self.csvtextdelimitercombo.setEditText('"')
        self.directioncombo.setCurrentIndex(0)

    def slotHelp(self):
        """Asked for help."""
        d = VeuszDialog(self.dialog.mainwindow, 'importhelpcsv.ui')
        self.dialog.mainwindow.showDialog(d)

    def getCSVDelimiter(self):
        """Get CSV delimiter, converting friendly names."""
        delim = str( self.csvdelimitercombo.text() )
        if delim == '{space}':
            delim = ' '
        elif delim == '{tab}':
            delim = '\t'
        return delim

    def doPreview(self, filename, encoding):
        """CSV preview - show first few rows"""

        t = self.previewtablecsv
        t.verticalHeader().show() # restore from a previous import
        t.horizontalHeader().show()
        t.horizontalHeader().setStretchLastSection(False)
        t.clear()
        t.setColumnCount(0)
        t.setRowCount(0)

        try:
            delimiter = self.getCSVDelimiter()
            textdelimiter = str(self.csvtextdelimitercombo.currentText())
        except UnicodeEncodeError:
            # need to be real str not unicode
            return False

        # need to be single character
        if len(delimiter) != 1 or len(textdelimiter) != 1:
            return False

        try:
            reader = utils.UnicodeCSVReader( open(filename),
                                             delimiter=delimiter,
                                             quotechar=textdelimiter,
                                             encoding=encoding )
            # construct list of rows
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

        except (IOError, UnicodeError, csv.Error):
            return False

        # fill up table
        t.setColumnCount(numcols)
        t.setRowCount(numrows)
        for r in xrange(numrows):
            for c in xrange(numcols):
                if c < len(rows[r]):
                    item = qt4.QTableWidgetItem(unicode(rows[r][c]))
                    t.setItem(r, c, item)

        return True

    def doImport(self, doc, filename, linked, encoding, prefix, suffix):
        """Import from CSV file."""

        # get various values
        inrows = self.directioncombo.currentIndex() == 1

        try:
            delimiter = self.getCSVDelimiter()
            textdelimiter = str(self.csvtextdelimitercombo.currentText())
        except UnicodeEncodeError:
            return

        op = document.OperationDataImportCSV(filename, readrows=inrows,
                                             prefix=prefix, suffix=suffix,
                                             linked=linked,
                                             delimiter=delimiter,
                                             textdelimiter=textdelimiter,
                                             encoding=encoding)
        
        # actually import the data
        dsnames = doc.applyOperation(op)
        
        # what datasets were imported
        lines = self.dialog.retnDatasetInfo(dsnames)

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

class ImportTab2D(ImportTab):
    """Tab for importing from a 2D data file."""

    resource = 'import_2d.ui'

    def loadUi(self):
        """Load user interface and set up validators."""
        ImportTab.loadUi(self)
        # set up some validators for 2d edits
        dval = qt4.QDoubleValidator(self)
        for i in (self.twod_xminedit, self.twod_xmaxedit,
                  self.twod_yminedit, self.twod_ymaxedit):
            i.setValidator(dval)

    def reset(self):
        """Reset controls."""
        for combo in (self.twod_xminedit, self.twod_xmaxedit,
                      self.twod_yminedit, self.twod_ymaxedit,
                      self.twod_datasetsedit):
            combo.setEditText("")
        for check in (self.twod_invertrowscheck, self.twod_invertcolscheck,
                      self.twod_transposecheck):
            check.setChecked(False)

    def doPreview(self, filename, encoding):
        """Preview 2d dataset files."""
        
        try:
            ifile = utils.openEncoding(filename, encoding)
            text = ifile.read(4096)+'\n'
            if len(ifile.read(1)) != 0:
                # if there is remaining data add ...
                text += '...\n'
            self.twod_previewedit.setPlainText(text)
            return True

        except (UnicodeError, IOError):
            self.twod_previewedit.setPlainText('')
            return False

    def doImport(self, doc, filename, linked, encoding, prefix, suffix):
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

        # loop over datasets and read...
        try:
            op = document.OperationDataImport2D(datasets, filename=filename,
                                                xrange=rangex, yrange=rangey,
                                                invertrows=invertrows,
                                                invertcols=invertcols,
                                                transpose=transpose,
                                                prefix=prefix, suffix=suffix,
                                                linked=linked,
                                                encoding=encoding)
            readds = doc.applyOperation(op)

            output = ['Successfully read datasets:']
            for ds in readds:
                output.append(' %s' % doc.data[ds].description(
                        showlinked=False))
            
            output = '\n'.join(output)
        except document.Read2DError, e:
            output = 'Error importing datasets:\n %s' % str(e)
                
        # show status in preview box
        self.twod_previewedit.setPlainText(output)

pyfits = None
class ImportTabFITS(ImportTab):
    """Tab for importing from a FITS file."""

    resource = 'import_fits.ui'

    def loadUi(self):
        ImportTab.loadUi(self)
        # if different items are selected in fits tab
        self.connect( self.fitshdulist, qt4.SIGNAL('itemSelectionChanged()'),
                      self.slotFitsUpdateCombos )
        self.connect( self.fitsdatasetname,
                      qt4.SIGNAL('textChanged(const QString&)'),
                      self.dialog.enableDisableImport )
        self.connect( self.fitsdatacolumn,
                      qt4.SIGNAL('currentIndexChanged(int)'),
                      self.dialog.enableDisableImport )

    def reset(self):
        """Reset controls."""
        self.fitsdatasetname.setEditText("")
        for c in ('data', 'sym', 'pos', 'neg'):
            cntrl = getattr(self, 'fits%scolumn' % c)
            cntrl.setCurrentIndex(0)
            
    def doPreview(self, filename, encoding):
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
            self.clearFITSView()
            return False

        self.updateFITSView(filename)
        return True

    def clearFITSView(self):
        """If invalid filename, clear fits preview."""
        self.fitshdulist.clear()
        for c in ('data', 'sym', 'pos', 'neg'):
            cntrl = getattr(self, 'fits%scolumn' % c)
            cntrl.clear()
            cntrl.setEnabled(False)

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
                dims = [ str(header['NAXIS%i' % (i+1)])
                         for i in xrange(naxis) ]
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

        self.dialog.enableDisableImport()

    def okToImport(self):
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

    def doImport(self, doc, filename, linked, encoding, prefix, suffix):
        """Import fits file."""
        
        item = self.fitshdulist.selectedItems()[0]
        hdunum = int( str(item.text(0)) )
        data = self.fitsitemdata[hdunum]

        name = prefix + unicode(self.fitsdatasetname.text()) + suffix

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
        doc.applyOperation(op)

        # inform user
        self.fitsimportstatus.setText("Imported dataset '%s'" % name)
        qt4.QTimer.singleShot(2000, self.fitsimportstatus.clear)

class ImportTabPlugins(ImportTab):
    """Tab for importing using a plugin."""

    resource = 'import_plugins.ui'

    def loadUi(self):
        """Load the user interface."""
        ImportTab.loadUi(self)

        # fill plugin combo
        names = list(sorted([p.name for p in plugins.importpluginregistry]))
        self.pluginType.addItems(names)

        self.connect(self.pluginType, qt4.SIGNAL('currentIndexChanged(int)'),
                     self.pluginChanged)

        self.fields = []

        # load previous plugin
        if 'import_plugin' in setting.settingdb:
            try:
                idx = names.index(setting.settingdb['import_plugin'])
                self.pluginType.setCurrentIndex(idx)
            except ValueError:
                pass

        self.pluginChanged(-1)

    def getPluginFields(self):
        """Return a dict of the fields given."""
        results = {}
        plugin = self.getSelectedPlugin()
        for field, cntrls in zip(plugin.fields, self.fields):
            results[field.name] = field.getControlResults(cntrls)
        return results

    def getSelectedPlugin(self):
        """Get instance selected plugin or none."""
        selname = unicode(self.pluginType.currentText())
        names = [p.name for p in plugins.importpluginregistry]
        try:
            idx = names.index(selname)
        except ValueError:
            return None
        return plugins.importpluginregistry[idx]

    def pluginChanged(self, index):
        """Update controls based on index."""
        plugin = self.getSelectedPlugin()
        setting.settingdb['import_plugin'] = plugin.name

        # delete old controls
        layout = self.pluginParams.layout()
        for line in self.fields:
            for cntrl in line:
                layout.removeWidget(cntrl)
                cntrl.deleteLater()
        del self.fields[:]

        # make new controls
        for row, field in enumerate(plugin.fields):
            cntrls = field.makeControl(None, None)
            layout.addWidget(cntrls[0], row, 0)
            layout.addWidget(cntrls[1], row, 1)
            self.fields.append(cntrls)

        # update label
        self.pluginDescr.setText("%s (%s)\n%s" %
                                 (plugin.name, plugin.author,
                                  plugin.description))

        self.dialog.slotUpdatePreview()

    def doPreview(self, filename, encoding):
        """Preview using plugin."""

        # check file exists
        try:
            f = open(filename, 'r')
            f.close()
        except IOError:
            self.pluginPreview.setPlainText('')
            return False

        # get the plugin selected
        plugin = self.getSelectedPlugin()
        if plugin is None:
            self.pluginPreview.setPlainText('')
            return False

        # ask the plugin for text
        params = plugins.ImportPluginParams(filename, encoding,
                                            self.getPluginFields())
        try:
            text, ok = plugin.getPreview(params)
        except plugins.ImportPluginException, ex:
            text = unicode(ex)
            ok = False
        self.pluginPreview.setPlainText(text)
        return bool(ok)

    def doImport(self, doc, filename, linked, encoding, prefix, suffix):
        """Import using plugin."""
        
        params = self.getPluginFields()
        plugin = unicode(self.pluginType.currentText())
        op = document.OperationDataImportPlugin(
            plugin, filename, linked=linked, encoding=encoding,
            prefix=prefix, suffix=suffix, **params)
        try:
            results = doc.applyOperation(op)
        except plugins.ImportPluginException, ex:
            self.pluginPreview.setPlainText( unicode(ex) )
            return

        out = ['Imported data for datasets:']
        for ds in results:
            out.append( doc.data[ds].description(showlinked=False) )

        self.pluginPreview.setPlainText('\n'.join(out))

class ImportDialog(VeuszDialog):
    """Dialog box for importing data.
    See ImportTab classes above which actually do the work of importing
    """

    dirname = '.'

    def __init__(self, parent, document):

        VeuszDialog.__init__(self, parent, 'import.ui')
        self.document = document

        # whether file import looks likely to work
        self.filepreviewokay = False

        # tabs loaded currently in dialog
        self.tabs = {}
        for tabname, tabclass in (
            ('&Standard', ImportTabStandard),
            ('CS&V', ImportTabCSV),
            ('FI&TS', ImportTabFITS),
            ('&2D', ImportTab2D),
            ('Plugins', ImportTabPlugins),
            ):
            w = tabclass(self)
            self.methodtab.addTab(w, tabname)
        self.connect( self.methodtab, qt4.SIGNAL('currentChanged(int)'),
                      self.slotUpdatePreview )

        self.connect(self.browsebutton, qt4.SIGNAL('clicked()'),
                     self.slotBrowseClicked)

        self.connect( self.filenameedit,
                      qt4.SIGNAL('editTextChanged(const QString&)'),
                      self.slotUpdatePreview )

        self.importbutton = self.buttonBox.addButton("&Import",
                                                     qt4.QDialogButtonBox.ApplyRole)
        self.connect( self.importbutton, qt4.SIGNAL('clicked()'),
                      self.slotImport)

        self.connect( self.buttonBox.button(qt4.QDialogButtonBox.Reset),
                      qt4.SIGNAL('clicked()'), self.slotReset )

        self.connect( self.encodingcombo,
                      qt4.SIGNAL('currentIndexChanged(int)'),
                      self.slotUpdatePreview )

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

        # defaults for prefix and suffix
        self.prefixcombo.default = self.suffixcombo.default = ['', '$FILENAME']

        # default state for check boxes
        self.linkcheckbox.default = True

        # further defaults
        self.encodingcombo.defaultlist = utils.encodings
        self.encodingcombo.defaultval = 'utf_8'

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

    def slotUpdatePreview(self, *args):
        """Update preview window when filename or tab changed."""

        # save so we can restore later
        tab = self.methodtab.currentIndex()
        setting.settingdb['import_lasttab'] = tab
        filename = unicode(self.filenameedit.text())
        encoding = str(self.encodingcombo.currentText())
        importtab = self.methodtab.currentWidget()

        if encoding == '':
            return

        if isinstance(importtab, ImportTab):
            if not importtab.uiloaded:
                importtab.loadUi()
            self.filepreviewokay = importtab.doPreview(
                filename, encoding)

        # enable or disable import button
        self.enableDisableImport()

    def enableDisableImport(self, *args):
        """Disable or enable import button if allowed."""

        importtab = self.methodtab.currentWidget()
        enabled = self.filepreviewokay and importtab.okToImport()

        # actually enable or disable import button
        self.importbutton.setEnabled( enabled )

    def slotImport(self):
        """Do the importing"""

        filename = unicode( self.filenameedit.text() )
        filename = os.path.abspath(filename)
        linked = self.linkcheckbox.isChecked()
        encoding = str(self.encodingcombo.currentText())

        # import according to tab selected
        importtab = self.methodtab.currentWidget()
        prefix, suffix = self.getPrefixSuffix(filename)
        try:
            qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )
            importtab.doImport(self.document, filename, linked, encoding,
                               prefix, suffix)
            qt4.QApplication.restoreOverrideCursor()
        except Exception:
            qt4.QApplication.restoreOverrideCursor()

            # show exception dialog
            d = exceptiondialog.ExceptionDialog(sys.exc_info(), self)
            d.exec_()

    def retnDatasetInfo(self, dsnames):
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
        f = utils.cleanDatasetName( os.path.basename(filename) )
        prefix = unicode( self.prefixcombo.lineEdit().text() )
        prefix = prefix.replace('$FILENAME', f)
        suffix = unicode( self.suffixcombo.lineEdit().text() )
        suffix = suffix.replace('$FILENAME', f)
        return prefix, suffix

    def slotReset(self):
        """Reset input fields."""

        self.filenameedit.setText("")
        self.encodingcombo.setCurrentIndex(
            self.encodingcombo.findText("utf_8"))
        self.linkcheckbox.setChecked(True)
        self.prefixcombo.setEditText("")
        self.suffixcombo.setEditText("")

        importtab = self.methodtab.currentWidget()
        importtab.reset()
