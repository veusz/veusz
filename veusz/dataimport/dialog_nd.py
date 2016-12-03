#    Copyright (C) 2016 Jeremy S. Sanders
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
import re

from .. import qtall as qt4
from ..compat import cstr
from .. import utils
from ..dialogs import importdialog
from . import defn_nd
from . import simpleread
from . import dialog_csv

def _(text, disambiguation=None, context="Import_ND"):
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class ImportTabND(importdialog.ImportTab):
    """Tab for importing from a ND data file."""

    resource = "import_nd.ui"

    def loadUi(self):
        """Load user interface and set up validators."""
        importdialog.ImportTab.loadUi(self)

        self.nd_mode.defaultlist = [_("Text"), _("CSV")]
        self.nd_shapeedit.default = [_("Auto")]
        self.nd_csvdelim.default = dialog_csv.csv_delimiters
        self.nd_csvtextdelim.default = dialog_csv.csv_text_delimiters
        self.nd_csvlocale.defaultlist = dialog_csv.csv_locales

        self.nd_mode.currentIndexChanged.connect(self.slotNewMode)

    def slotNewMode(self, index):
        """Change other widgets depending on mode."""
        csv = index == 1
        self.nd_csvdelim.setEnabled(csv)
        self.nd_csvtextdelim.setEnabled(csv)
        self.nd_csvlocale.setEnabled(csv)

    def reset(self):
        """Reset controls."""
        self.nd_datasetedit.setEditText("")
        self.nd_shapeedit.setEditText(_("Auto"))
        self.nd_transposecheck.setChecked(False)
        self.nd_mode.setCurrentIndex(0)
        self.nd_csvdelim.setEditText(dialog_csv.csv_delimiters[0])
        self.nd_csvtextdelim.setEditText(dialog_csv.csv_text_delimiters[0])
        self.nd_csvlocale.setCurrentIndex(0)

    def doPreview(self, filename, encoding):
        """Preview nD dataset files."""

        try:
            ifile = utils.openEncoding(filename, encoding)
            text = ifile.read(4096) + "\n"
            if len(ifile.read(1)) != 0:
                # if there is remaining data add ...
                text += "...\n"
            self.nd_previewedit.setPlainText(text)
            return True

        except (UnicodeError, EnvironmentError):
            self.nd_previewedit.setPlainText("")
            return False

    def doImport(self, doc, filename, linked, encoding, prefix, suffix, tags):
        """Import from ND file."""

        class error(RuntimeError):
            pass

        try:
            dataset = self.nd_datasetedit.text().strip()
            if not dataset:
                raise error(_("A dataset name should be given"))

            transpose = self.nd_transposecheck.isChecked()

            mode = ("text", "csv")[self.nd_mode.currentIndex()]
            # this needs to be a str for the csv module (py2)
            csvdelimiter = str(self.nd_csvdelim.text())
            if csvdelimiter in dialog_csv.csv_delimiter_map:
                csvdelimiter = dialog_csv.csv_delimiter_map[csvdelimiter]
            csvtextdelimiter = str(self.nd_csvtextdelim.text())
            csvlocale = dialog_csv.csvLocaleIndexToLocale(
                self.nd_csvlocale.currentIndex())

            shapetxt = self.nd_shapeedit.text().strip()
            if shapetxt == _("Auto"):
                shape = None
            else:
                shapesplit = re.split("[,;x* ]+", shapetxt)
                try:
                    shape = tuple([int(x) for x in shapesplit])
                except ValueError:
                    raise error(_("Shape entries should be integers"))
                if len(shape) == 0:
                    raise error(_("No shape entries given"))

            params = defn_nd.ImportParamsND(
                dataset=dataset,
                filename=filename,
                shape=shape,
                transpose=transpose,
                mode=mode,
                csvdelimiter=csvdelimiter,
                csvtextdelimiter=csvtextdelimiter,
                csvlocale=csvlocale,
                prefix=prefix, suffix=suffix,
                tags=tags,
                linked=linked,
                encoding=encoding
                )

            # do the importing
            op = defn_nd.OperationDataImportND(params)
            doc.applyOperation(op)

            # show result
            output = [_("Successfully read:")]
            for ds in op.outnames:
                output.append("%s: %s" % (
                    ds,
                    doc.data[ds].description())
                )
            output = "\n".join(output)

        except error as e:
            output = e.args[0]

        except simpleread.ReadNDError as e:
            output = _("Error importing datasets:\n %s") % cstr(e)

        # show status in preview box
        self.nd_previewedit.setPlainText(output)

importdialog.registerImportTab(_("&ND"), ImportTabND)
