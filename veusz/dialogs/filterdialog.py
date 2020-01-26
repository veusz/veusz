#    Copyright (C) 2015 Jeremy S. Sanders
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

"""Dialog for filtering data."""

from __future__ import division, print_function

from .. import qtall as qt
from .. import document
from ..qtwidgets.datasetbrowser import DatasetBrowser
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="FilterDialog"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class FilterDialog(VeuszDialog):
    """Preferences dialog."""

    def __init__(self, parent, doc):
        """Setup dialog."""
        VeuszDialog.__init__(self, parent, "filter.ui")
        self.document = doc

        self.dsbrowser = DatasetBrowser(doc, parent, None, checkable=True)
        grplayout = qt.QVBoxLayout()
        grplayout.addWidget(self.dsbrowser)
        self.filtergroup.setLayout(grplayout)

        self.buttonBox.button(qt.QDialogButtonBox.Apply).clicked.connect(
            self.applyClicked)
        self.buttonBox.button(qt.QDialogButtonBox.Reset).clicked.connect(
            self.resetClicked)

    def updateStatus(self, text):
        """Show message in dialog."""
        qt.QTimer.singleShot(4000, self.statuslabel.clear)
        self.statuslabel.setText(text)

    def applyClicked(self):
        """Do the filtering."""

        prefix = self.prefixcombo.currentText().strip()
        suffix = self.suffixcombo.currentText().strip()
        if not prefix and not suffix:
            self.updateStatus(_("Prefix and/or suffix must be entered"))
            return

        expr = self.exprcombo.currentText().strip()
        if not expr:
            self.updateStatus(_("Enter a valid filter expression"))
            return

        tofilter = self.dsbrowser.checkedDatasets()
        if not tofilter:
            self.updateStatus(_("Choose at least one dataset to filter"))
            return

        invert = self.invertcheck.isChecked()
        replaceblanks = self.replaceblankscheck.isChecked()

        op = document.OperationDatasetsFilter(
            expr,
            tofilter,
            prefix=prefix, suffix=suffix,
            invert=invert,
            replaceblanks=replaceblanks)

        ok, log = op.check(self.document)
        if not ok:
            self.updateStatus("\n".join(log))
            return

        self.document.applyOperation(op)
        self.updateStatus(_("Filtered %i datasets") % len(tofilter))

    def resetClicked(self):
        """Reset controls to defaults."""
        for cntrl in self.exprcombo, self.prefixcombo, self.suffixcombo:
            cntrl.setEditText("")
        self.dsbrowser.reset()
        self.invertcheck.setChecked(False)
        self.replaceblankscheck.setChecked(False)
        self.updateStatus(_("Dialog reset"))

    def reEditDialog(self, dataset):
        """Load controls with settings from dataset."""
        gen = dataset.generator

        self.exprcombo.setEditText(gen.inexpr)
        self.prefixcombo.setEditText(gen.prefix)
        self.suffixcombo.setEditText(gen.suffix)
        self.invertcheck.setChecked(gen.invert)
        self.replaceblankscheck.setChecked(gen.replaceblanks)

        datasets = [
            d for d in gen.indatasets
            if d in self.document.data
            ]
        self.dsbrowser.setCheckedDatasets(datasets)

def recreateDataset(mainwindow, document, dataset, datasetname):
    """Open dialog to recreate filter."""
    dialog = FilterDialog(mainwindow, document)
    mainwindow.showDialog(dialog)
    dialog.reEditDialog(dataset)
