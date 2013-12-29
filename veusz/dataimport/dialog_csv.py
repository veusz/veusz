#    Copyright (C) 2013 Jeremy S. Sanders
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

from __future__ import division, print_function
import csv

from .. import qtall as qt4
from ..dialogs import importdialog, veuszdialog
from ..compat import crange, cnext
from .. import utils
from . import defn_csv

def _(text, disambiguation=None, context="Import_CSV"):
    return qt4.QCoreApplication.translate(context, text, disambiguation)

csv_delimiters = [',', '{tab}', '{space}', '|', ':', ';']
csv_text_delimiters = ['"', "'"]
csv_locales = [_('System'), _('English'), _('European')]

csv_delimiter_map = {
    '{tab}': '\t',
    '{space}': ' '
}

def csvLocaleIndexToLocale(idx):
    """Convert index to text locale."""
    return (qt4.QLocale().name(), 'en_US', 'de_DE')[idx]

class ImportTabCSV(importdialog.ImportTab):
    """For importing data from CSV files."""

    resource = 'import_csv.ui'

    def loadUi(self):
        """Load user interface and setup panel."""
        importdialog.ImportTab.loadUi(self)
        self.csvhelpbutton.clicked.connect(self.slotHelp)
        self.csvdelimitercombo.editTextChanged.connect(
            self.dialog.slotUpdatePreview)
        self.csvtextdelimitercombo.editTextChanged.connect(
            self.dialog.slotUpdatePreview)
        self.csvdelimitercombo.default = csv_delimiters
        self.csvtextdelimitercombo.default = csv_text_delimiters
        self.csvdatefmtcombo.default = [
            'YYYY-MM-DD|T|hh:mm:ss',
            'DD/MM/YY| |hh:mm:ss',
            'M/D/YY| |hh:mm:ss'
            ]
        self.csvnumfmtcombo.defaultlist = csv_locales
        self.csvheadermodecombo.defaultlist = [_('Multiple'), _('1st row'), _('None')]

    def reset(self):
        """Reset controls."""
        self.csvdelimitercombo.setEditText(",")
        self.csvtextdelimitercombo.setEditText('"')
        self.csvdirectioncombo.setCurrentIndex(0)
        self.csvignorehdrspin.setValue(0)
        self.csvignoretopspin.setValue(0)
        self.csvblanksdatacheck.setChecked(False)
        self.csvnumfmtcombo.setCurrentIndex(0)
        self.csvdatefmtcombo.setEditText(
            defn_csv.ImportParamsCSV.defaults['dateformat'])
        self.csvheadermodecombo.setCurrentIndex(0)

    def slotHelp(self):
        """Asked for help."""
        d = veuszdialog.VeuszDialog(self.dialog.mainwindow, 'importhelpcsv.ui')
        self.dialog.mainwindow.showDialog(d)

    def getCSVDelimiter(self):
        """Get CSV delimiter, converting friendly names."""
        delim = str( self.csvdelimitercombo.text() )
        if delim in csv_delimiter_map:
            delim = csv_delimiter_map[delim]
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
            reader = utils.get_unicode_csv_reader(
                filename,
                delimiter=delimiter,
                quotechar=textdelimiter,
                encoding=encoding )

            # construct list of rows
            rows = []
            numcols = 0
            try:
                for i in crange(10):
                    row = cnext(reader)
                    rows.append(row)
                    numcols = max(numcols, len(row))
                rows.append(['...'])
                numcols = max(numcols, 1)
            except StopIteration:
                pass
            numrows = len(rows)

        except (EnvironmentError, UnicodeError, csv.Error):
            return False

        # fill up table
        t.setColumnCount(numcols)
        t.setRowCount(numrows)
        for r in crange(numrows):
            for c in crange(numcols):
                if c < len(rows[r]):
                    item = qt4.QTableWidgetItem(rows[r][c])
                    t.setItem(r, c, item)

        return True

    def doImport(self, doc, filename, linked, encoding, prefix, suffix, tags):
        """Import from CSV file."""

        # get various values
        inrows = self.csvdirectioncombo.currentIndex() == 1

        try:
            delimiter = self.getCSVDelimiter()
            textdelimiter = str(self.csvtextdelimitercombo.currentText())
        except UnicodeEncodeError:
            return

        numericlocale = csvLocaleIndexToLocale(
            self.csvnumfmtcombo.currentIndex() )
        headerignore = self.csvignorehdrspin.value()
        rowsignore = self.csvignoretopspin.value()
        blanksaredata = self.csvblanksdatacheck.isChecked()
        dateformat = self.csvdatefmtcombo.currentText()
        headermode = ('multi', '1st', 'none')[
            self.csvheadermodecombo.currentIndex()]

        # create import parameters and operation objects
        params = defn_csv.ImportParamsCSV(
            filename=filename,
            readrows=inrows,
            encoding=encoding,
            delimiter=delimiter,
            textdelimiter=textdelimiter,
            headerignore=headerignore,
            rowsignore=rowsignore,
            blanksaredata=blanksaredata,
            numericlocale=numericlocale,
            dateformat=dateformat,
            headermode=headermode,
            prefix=prefix, suffix=suffix,
            tags=tags,
            linked=linked,
            )
        op = defn_csv.OperationDataImportCSV(params)

        # actually import the data
        doc.applyOperation(op)

        # update output, showing what datasets were imported
        lines = self.dialog.retnDatasetInfo(op.outdatasets, linked, filename)

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

    def isFiletypeSupported(self, ftype):
        """Is the filetype supported by this tab?"""
        return ftype in ('.tsv', '.csv')

importdialog.registerImportTab(_('CS&V'), ImportTabCSV)
