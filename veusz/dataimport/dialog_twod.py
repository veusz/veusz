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
import re

from .. import qtall as qt
from ..compat import cstr
from .. import utils
from ..dialogs import importdialog
from . import defn_twod
from . import simpleread
from . import dialog_csv

def _(text, disambiguation=None, context="Import_2D"):
    return qt.QCoreApplication.translate(context, text, disambiguation)

class ImportTab2D(importdialog.ImportTab):
    """Tab for importing from a 2D data file."""

    resource = 'import_2d.ui'

    def loadUi(self):
        """Load user interface and set up validators."""
        importdialog.ImportTab.loadUi(self)

        self.rangeedits = [ self.twod_xminedit, self.twod_xmaxedit,
                            self.twod_yminedit, self.twod_ymaxedit ]

        # set up some validators for 2d edits
        dval = qt.QDoubleValidator(self)
        for w in self.rangeedits:
            w.setValidator(dval)

        self.twod_mode.defaultlist = [_('Text'), _('CSV')]
        self.twod_csvdelim.default = dialog_csv.csv_delimiters
        self.twod_csvtextdelim.default = dialog_csv.csv_text_delimiters
        self.twod_csvlocale.defaultlist = dialog_csv.csv_locales

        self.twod_mode.currentIndexChanged.connect(self.slotNewMode)
        self.twod_gridatedge.stateChanged.connect(self.slotGridAtEdgeChanged)

    def slotNewMode(self, index):
        """Change other widgets depending on mode."""
        csv = index == 1
        self.twod_csvdelim.setEnabled(csv)
        self.twod_csvtextdelim.setEnabled(csv)
        self.twod_csvlocale.setEnabled(csv)

    def slotGridAtEdgeChanged(self, state):
        """Enable/disable widgets depending on grid at edge."""
        nogridatedge = state == qt.Qt.Unchecked
        for w in self.rangeedits:
            w.setEnabled(nogridatedge)
            if not nogridatedge:
                w.setEditText("")

    def reset(self):
        """Reset controls."""
        for combo in self.rangeedits + [self.twod_datasetsedit]:
            combo.setEditText("")
        for check in (self.twod_invertrowscheck, self.twod_invertcolscheck,
                      self.twod_transposecheck, self.twod_gridatedge,
                      self.twod_gridatedge):
            check.setChecked(False)
        self.twod_mode.setCurrentIndex(0)
        self.twod_csvdelim.setEditText(dialog_csv.csv_delimiters[0])
        self.twod_csvtextdelim.setEditText(dialog_csv.csv_text_delimiters[0])
        self.twod_csvlocale.setCurrentIndex(0)

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

        except (UnicodeError, EnvironmentError):
            self.twod_previewedit.setPlainText('')
            return False

    def doImport(self, doc, filename, linked, encoding, prefix, suffix, tags):
        """Import from 2D file."""

        # this really needs improvement...

        # get datasets and split into a list
        datasets = self.twod_datasetsedit.text()
        datasets = re.split('[, ]+', datasets)

        # strip out blank items
        datasets = [d for d in datasets if d != '']

        # an obvious error...
        if len(datasets) == 0:
            self.twod_previewedit.setPlainText(
                _('At least one dataset needs to be given'))
            return

        # convert range parameters
        ranges = []
        for e in self.rangeedits:
            f = e.text()
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
        gridatedge = self.twod_gridatedge.isChecked()

        mode = ('text', 'csv')[self.twod_mode.currentIndex()]
        # this needs to be a str for the csv module (py2)
        csvdelimiter = str(self.twod_csvdelim.text())
        if csvdelimiter in dialog_csv.csv_delimiter_map:
            csvdelimiter = dialog_csv.csv_delimiter_map[csvdelimiter]
        csvtextdelimiter = str(self.twod_csvtextdelim.text())
        csvlocale = dialog_csv.csvLocaleIndexToLocale(
            self.twod_csvlocale.currentIndex())

        if len(csvdelimiter) != 1 or len(csvtextdelimiter) != 1:
            self.twod_previewedit.setPlainText(
                _('Delimiters must be single characters'))
            return

        # loop over datasets and read...
        params = defn_twod.ImportParams2D(
            datasetnames=datasets,
            filename=filename,
            xrange=rangex, yrange=rangey,
            invertrows=invertrows,
            invertcols=invertcols,
            transpose=transpose,
            gridatedge=gridatedge,
            mode=mode,
            csvdelimiter=csvdelimiter,
            csvtextdelimiter=csvtextdelimiter,
            csvlocale=csvlocale,
            prefix=prefix, suffix=suffix,
            tags=tags,
            linked=linked,
            encoding=encoding
            )

        try:
            op = defn_twod.OperationDataImport2D(params)
            doc.applyOperation(op)

            output = [_('Successfully read datasets:')]
            for ds in op.outnames:
                output.append('%s: %s' % (
                    ds,
                    doc.data[ds].description())
                )

            output = '\n'.join(output)

            # feature feedback
            utils.feedback.importcts['twod'] += 1

        except (simpleread.Read2DError, csv.Error) as e:
            output = _('Error importing datasets:\n %s') % cstr(e)

        # show status in preview box
        self.twod_previewedit.setPlainText(output)

importdialog.registerImportTab(_('&2D'), ImportTab2D)
