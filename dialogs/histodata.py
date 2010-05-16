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

# $Id$

import os.path

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.utils as utils
import veusz.document as document
from veusz.setting.controls import populateCombo

import numpy as N

class ManualBinModel(qt4.QAbstractListModel):
    """Model to store a list of floating point values in a list."""
    def __init__(self, data):
        qt4.QAbstractListModel.__init__(self)
        self.data = data
    def data(self, index, role):
        if role == qt4.Qt.DisplayRole and index.isValid():
            return qt4.QVariant(float(self.data[index.row()]))
        return qt4.QVariant()
    def rowCount(self, parent):
        return len(self.data)
    def flags(self, index):
        return ( qt4.Qt.ItemIsSelectable | qt4.Qt.ItemIsEnabled |
                 qt4.Qt.ItemIsEditable )
    def setData(self, index, value, role):
        if role == qt4.Qt.EditRole:
            val, ok = value.toDouble()
            if ok:
                self.data[ index.row() ] = val
                self.emit( qt4.SIGNAL("dataChanged(const QModelIndex &,"
                                      " const QModelIndex &)"), index, index)
                return True
        return False

class HistoDataDialog(qt4.QDialog):
    """Preferences dialog."""

    def __init__(self, document, *args):
        """Setup dialog."""
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'histodata.ui'),
                   self)

        self.document = document

        self.minval.default = self.maxval.default = ['Auto']
        regexp = qt4.QRegExp("^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?|Auto$")
        validator = qt4.QRegExpValidator(regexp, self)
        self.minval.setValidator(validator)
        self.maxval.setValidator(validator)
        self.connect( self.buttonBox.button(qt4.QDialogButtonBox.Apply),
                      qt4.SIGNAL("clicked()"), self.applyClicked )
        self.connect( self.bingenerate, qt4.SIGNAL('clicked()'),
                      self.generateManualBins )

        self.bindata = []
        self.binmodel = ManualBinModel(self.bindata)
        self.binmanuals.setModel(self.binmodel)

        self.connect(document, qt4.SIGNAL("sigModified"),
                     self.updateDatasetLists)
        self.updateDatasetLists()

    def escapeDatasets(self, dsnames):
        """Escape dataset names if they are not typical python ones."""

        for i in xrange(len(dsnames)):
            if not utils.validPythonIdentifier(dsnames[i]):
                dsnames[i] = '`%s`' % dsnames[i]

    def updateDatasetLists(self):
        """Update list of datasets."""

        datasets = []
        for name, ds in self.document.data.iteritems():
            if ds.datatype == 'numeric' and ds.dimensions == 1:
                datasets.append(name)
        datasets.sort()

        # make sure names are escaped if they have funny characters
        self.escapeDatasets(datasets)

        # help the user by listing existing datasets
        populateCombo(self.indataset, datasets)

    def datasetExprChanged(self):
        """Validate expression."""
        text = self.indataset.text()
        res = document.simpleEvalExpression(self.document, unicode(text))
        print res

    class Params(object):
        """Parameters to creation of histogram."""
        def __init__(self, dialog):
            """Initialise parameters from dialog."""
            numbins = dialog.numbins.value()
            minval = dialog.minval.text()
            if minval != 'Auto':
                minval = float(minval)
            maxval = dialog.maxval.text()
            if maxval != 'Auto':
                maxval = float(maxval)
            islog = dialog.logarithmic.isChecked()
            self.binparams = (numbins, minval, maxval, islog)

            self.expr = unicode( dialog.indataset.currentText() )
            self.outdataset = unicode( dialog.outdataset.currentText() )
            self.outbins = unicode( dialog.outbins.currentText() )
            self.method = unicode( dialog.methodGroup.getRadioChecked().
                                   objectName() )
            self.manualbins = list( dialog.bindata )
            self.manualbins.sort()
            if len(self.manualbins) == 0:
                self.manualbins = None

        def getGenerator(self, doc):
            """Return dataset generator."""
            return document.DatasetHistoGenerator(
                doc, self.expr, binparams = self.binparams,
                binmanual = self.manualbins, method = self.method )

        def getOperation(self):
            """Get operation to make histogram."""
            return document.OperationDatasetHistogram(
                self.expr, self.outbins, self.outdataset,
                binparams = self.binparams,
                binmanual = self.manualbins,
                method = self.method )

    def generateManualBins(self):
        """Generate manual bins."""
        p = HistoDataDialog.Params(self)
        if p.expr != '':
            p.manualbins = []
            gen = p.getGenerator(self.document)
            self.bindata[:] = list(gen.binLocations())
        else:
            del self.bindata[:]
        self.binmodel.reset()

    def applyClicked(self):
        """Create histogram."""

        p = HistoDataDialog.Params(self)
        op = p.getOperation()
        self.document.applyOperation(op)

        self.statuslabel.setText(
            'Created datasets "%s" and "%s"' % (p.outbins,
                                                p.outdataset) )

        def clearlabel(self=self):
            self.statuslabel.setText("")

        qt4.QTimer.singleShot(2000, clearlabel)
