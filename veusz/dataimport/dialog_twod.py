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
import re

from .. import qtall as qt4
from .. import utils
from ..dialogs import importdialog
from . import defn_twod
from . import simpleread

def _(text, disambiguation=None, context="Import_2D"):
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class ImportTab2D(importdialog.ImportTab):
    """Tab for importing from a 2D data file."""

    resource = 'import_2d.ui'

    def loadUi(self):
        """Load user interface and set up validators."""
        importdialog.ImportTab.loadUi(self)
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
                      self.twod_transposecheck, self.twod_gridatedge):
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
            self.twod_previewedit.setPlainText(_('At least one dataset needs to '
                                                 'be specified'))
            return

        # convert range parameters
        ranges = []
        for e in (self.twod_xminedit, self.twod_xmaxedit,
                  self.twod_yminedit, self.twod_ymaxedit):
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

        # loop over datasets and read...
        params = defn_twod.ImportParams2D(
            datasetnames=datasets,
            filename=filename,
            xrange=rangex, yrange=rangey,
            invertrows=invertrows,
            invertcols=invertcols,
            transpose=transpose,
            gridatedge=gridatedge,
            prefix=prefix, suffix=suffix,
            tags=tags,
            linked=linked,
            encoding=encoding
            )

        try:
            op = defn_twod.OperationDataImport2D(params)
            doc.applyOperation(op)

            output = [_('Successfully read datasets:')]
            for ds in op.outdatasets:
                output.append(' %s' % doc.data[ds].description(
                        showlinked=False))

            output = '\n'.join(output)
        except simpleread.Read2DError as e:
            output = _('Error importing datasets:\n %s') % str(e)

        # show status in preview box
        self.twod_previewedit.setPlainText(output)

importdialog.registerImportTab(_('&2D'), ImportTab2D)
