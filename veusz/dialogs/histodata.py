#    Copyright (C) 2010 Jeremy S. Sanders
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

from __future__ import division

from ..compat import crange, citems, cstr
from .. import qtall as qt
from .. import utils
from .. import datasets
from .. import document

from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="HistogramDialog"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

def checkValidator(combo):
    """Is this validator ok?"""
    valid = combo.validator()
    state, s, x = valid.validate(combo.currentText(), 0)
    return state == qt.QValidator.Acceptable

class ManualBinModel(qt.QAbstractListModel):
    """Model to store a list of floating point values in a list."""
    def __init__(self, thedata):
        qt.QAbstractListModel.__init__(self)
        self.thedata = thedata
    def data(self, index, role):
        if role == qt.Qt.DisplayRole and index.isValid():
            return float(self.thedata[index.row()])
        return None
    def rowCount(self, parent):
        return len(self.thedata)
    def flags(self, index):
        return ( qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled |
                 qt.Qt.ItemIsEditable )
    def setData(self, index, value, role):
        if role == qt.Qt.EditRole:
            try:
                val = float(value)
            except ValueError:
                return False

            self.thedata[ index.row() ] = val
            self.dataChanged.emit(index, index)
            return True
        return False

class HistoDataDialog(VeuszDialog):
    """Preferences dialog."""

    def __init__(self, parent, document):
        """Setup dialog."""
        VeuszDialog.__init__(self, parent, 'histodata.ui')
        self.document = document

        self.minval.default = self.maxval.default = ['Auto']
        regexp = qt.QRegExp("^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?|Auto$")
        validator = qt.QRegExpValidator(regexp, self)
        self.minval.setValidator(validator)
        self.maxval.setValidator(validator)
        self.buttonBox.button(qt.QDialogButtonBox.Apply).clicked.connect(
            self.applyClicked )
        self.buttonBox.button(qt.QDialogButtonBox.Reset).clicked.connect(
            self.resetClicked )
        self.bingenerate.clicked.connect(self.generateManualBins)
        self.binadd.clicked.connect(self.addManualBins)
        self.binremove.clicked.connect(self.removeManualBins)

        self.bindata = []
        self.binmodel = ManualBinModel(self.bindata)
        self.binmanuals.setModel(self.binmodel)

        document.signalModified.connect(self.updateDatasetLists)
        self.updateDatasetLists()

    def escapeDatasets(self, dsnames):
        """Escape dataset names if they are not typical python ones."""

        for i in crange(len(dsnames)):
            if not utils.validPythonIdentifier(dsnames[i]):
                dsnames[i] = '`%s`' % dsnames[i]

    def updateDatasetLists(self):
        """Update list of datasets."""

        datasets = []
        for name, ds in citems(self.document.data):
            if ds.datatype == 'numeric' and ds.dimensions == 1:
                datasets.append(name)
        datasets.sort()

        # make sure names are escaped if they have funny characters
        self.escapeDatasets(datasets)

        # help the user by listing existing datasets
        utils.populateCombo(self.indataset, datasets)

    def datasetExprChanged(self):
        """Validate expression."""
        text = self.indataset.text()
        datasets.evalDatasetExpression(self.document, text)

    class Params(object):
        """Parameters to creation of histogram."""

        def __init__(self, dialog):
            """Initialise parameters from dialog."""
            numbins = dialog.numbins.value()

            if not checkValidator(dialog.minval):
                raise RuntimeError(_("Invalid minimum value"))
            minval = dialog.minval.text()
            if minval != 'Auto':
                minval = float(minval)

            if not checkValidator(dialog.maxval):
                raise RuntimeError(_("Invalid maximum value"))
            maxval = dialog.maxval.text()
            if maxval != 'Auto':
                maxval = float(maxval)

            islog = dialog.logarithmic.isChecked()
            self.binparams = (numbins, minval, maxval, islog)

            self.expr = dialog.indataset.currentText().strip()
            self.outdataset = dialog.outdataset.currentText().strip()
            self.outbins = dialog.outbins.currentText().strip()

            if self.expr == self.outdataset or self.expr == self.outbins:
                raise RuntimeError(_("Output datasets cannot be the same as input datasets"))

            self.method = dialog.methodGroup.getRadioChecked().objectName()
            self.manualbins = list( dialog.bindata )
            self.manualbins.sort()
            if len(self.manualbins) == 0:
                self.manualbins = None

            self.errors = dialog.errorBars.isChecked()
            cuml = dialog.cumlGroup.getRadioChecked().objectName()
            self.cumulative = 'none'
            if cuml == 'cumlStoL':
                self.cumulative = 'smalltolarge'
            elif cuml == 'cumlLtoS':
                self.cumulative = 'largetosmall'

        def getGenerator(self, doc):
            """Return dataset generator."""
            return datasets.DatasetHistoGenerator(
                doc, self.expr, binparams = self.binparams,
                binmanual = self.manualbins, method = self.method,
                cumulative = self.cumulative, errors = self.errors)

        def getOperation(self):
            """Get operation to make histogram."""
            return document.OperationDatasetHistogram(
                self.expr, self.outbins, self.outdataset,
                binparams = self.binparams,
                binmanual = self.manualbins,
                method = self.method,
                cumulative = self.cumulative,
                errors = self.errors)

    def generateManualBins(self):
        """Generate manual bins."""

        try:
            p = HistoDataDialog.Params(self)
        except RuntimeError as ex:
            qt.QMessageBox.warning(self, _("Invalid parameters"), cstr(ex))
            return

        self.binmodel.beginRemoveRows(qt.QModelIndex(), 0, len(self.bindata)-1)
        del self.bindata[:]
        self.binmodel.endRemoveRows()

        if p.expr != '':
            p.manualbins = []
            gen = p.getGenerator(self.document)
            locs = list(gen.binLocations())
            self.binmodel.beginInsertRows(qt.QModelIndex(), 0, len(locs)-1)
            self.bindata += locs
            self.binmodel.endInsertRows()

    def addManualBins(self):
        """Add an extra bin to the manual list."""
        self.binmodel.beginInsertRows(qt.QModelIndex(), 0, 0)
        self.bindata.insert(0, 0.)
        self.binmodel.endInsertRows()

    def removeManualBins(self):
        """Remove selected bins."""
        indexes = self.binmanuals.selectionModel().selectedIndexes()
        if indexes:
            row = indexes[0].row()
            self.binmodel.beginRemoveRows(qt.QModelIndex(), row, row)
            del self.bindata[row]
            self.binmodel.endRemoveRows()

    def resetClicked(self):
        """Reset button clicked."""

        for cntrl in (self.indataset, self.outdataset, self.outbins):
            cntrl.setEditText("")

        self.numbins.setValue(10)
        self.minval.setEditText("Auto")
        self.maxval.setEditText("Auto")
        self.logarithmic.setChecked(False)

        self.binmodel.beginRemoveRows(qt.QModelIndex(), 0, len(self.bindata)-1)
        del self.bindata[:]
        self.binmodel.endRemoveRows()

        self.errorBars.setChecked(False)
        self.counts.click()
        self.cumlOff.click()

    def reEditDataset(self, ds, dsname):
        """Re-edit dataset."""

        gen = ds.generator

        self.indataset.setEditText(gen.inexpr)

        # need to map backwards to get dataset names
        revds = dict( (a,b) for b,a in citems(self.document.data) )
        self.outdataset.setEditText(revds.get(gen.valuedataset, ''))
        self.outbins.setEditText(revds.get(gen.bindataset, ''))

        # if there are parameters
        if gen.binparams:
            p = gen.binparams
            self.numbins.setValue( p[0] )
            self.minval.setEditText( cstr(p[1]) )
            self.maxval.setEditText( cstr(p[2]) )
            self.logarithmic.setChecked( bool(p[3]) )
        else:
            self.numbins.setValue(10)
            self.minval.setEditText("Auto")
            self.maxval.setEditText("Auto")
            self.logarithmic.setChecked(False)

        # if there is a manual list of bins
        if gen.binmanual is not None:
            self.binmodel.beginResetModel()
            self.bindata[:] = list(gen.binmanual)
            self.binmodel.endResetModel()

        # select correct method
        {'counts': self.counts, 'density': self.density,
         'fractions': self.fractions}[gen.method].click()
        
        # select if cumulative
        {'none': self.cumlOff, 'smalltolarge': self.cumlStoL,
         'largetosmall': self.cumlLtoS}[gen.cumulative].click()

        # if error bars
        self.errorBars.setChecked( bool(gen.errors) )

    def applyClicked(self):
        """Create histogram."""

        qt.QTimer.singleShot(4000, self.statuslabel.clear)
        try:
            p = HistoDataDialog.Params(self)
        except RuntimeError as ex:
            self.statuslabel.setText(_("Invalid parameters: %s") % cstr(ex))
            return

        exprresult = datasets.evalDatasetExpression(self.document, p.expr)
        if exprresult is None:
            self.statuslabel.setText(_("Invalid expression"))
            return

        op = p.getOperation()
        self.document.applyOperation(op)

        self.statuslabel.setText(
            _('Created datasets "%s" and "%s"') % (p.outbins, p.outdataset))

def recreateDataset(mainwindow, document, dataset, datasetname):
    """Open dialog to recreate histogram."""
    dialog = HistoDataDialog(mainwindow, document)
    mainwindow.showDialog(dialog)
    dialog.reEditDataset(dataset, datasetname)

