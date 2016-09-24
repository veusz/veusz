#    Copyright (C) 2007 Jeremy S. Sanders
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

"""Dataset creation dialog for 2d data."""

from __future__ import division
from ..compat import crange, citems, cstr
from .. import qtall as qt4
from .. import utils
from .. import document
from .. import datasets
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="DataCreate2D"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

def checkGetStep(text):
    """Check step syntax is okay.
    Syntax is min:max:stepsize
    Returns None if fails
    """

    parts = text.split(':')
    if len(parts) == 3:
        try:
            return tuple([float(x) for x in parts])
        except ValueError:
            pass
    return None

class DataCreate2DDialog(VeuszDialog):

    def __init__(self, parent, document):
        """Initialise dialog with document."""

        VeuszDialog.__init__(self, parent, 'datacreate2d.ui')
        self.document = document

        self.createbutton = self.buttonBox.addButton(
            _("C&reate"), qt4.QDialogButtonBox.ApplyRole )
        self.createbutton.clicked.connect(self.createButtonClickedSlot)

        self.fromxyfunc.toggled.connect(self.fromxyfuncSlot)
        self.fromxyzexpr.toggled.connect(self.fromxyzexprSlot)
        self.from2dexpr.toggled.connect(self.from2dexprSlot)

        document.signalModified.connect(self.updateDatasetLists)

        for combo in (self.namecombo, self.xexprcombo, self.yexprcombo,
                      self.zexprcombo):
            combo.editTextChanged.connect(self.enableDisableCreate)

        self.fromxyzexpr.toggle()
        self.enableDisableCreate()

    # change mode according to radio pressed
    def fromxyfuncSlot(self, checked):
        self.mode = 'xyfunc'
        if checked: self.updateDatasetLists()
    def fromxyzexprSlot(self, checked):
        self.mode = 'xyzexpr'
        if checked: self.updateDatasetLists()
    def from2dexprSlot(self, checked):
        self.mode = '2dexpr'
        if checked: self.updateDatasetLists()

    def escapeDatasets(self, dsnames):
        """Escape dataset names if they are not typical python ones."""

        for i in crange(len(dsnames)):
            if not utils.validPythonIdentifier(dsnames[i]):
                dsnames[i] = '`%s`' % dsnames[i]

    def updateDatasetLists(self):
        """Update controls depending on selected mode."""

        # get list of 1d and 2d numeric datasets
        datasets = [[],[]]
        for name, ds in citems(self.document.data):
            if ds.datatype == 'numeric':
                datasets[ds.dimensions-1].append(name)
        datasets[0].sort()
        datasets[1].sort()

        # make sure names are escaped if they have funny characters
        self.escapeDatasets(datasets[0])
        self.escapeDatasets(datasets[1])

        # help the user by listing existing datasets
        utils.populateCombo(self.namecombo, datasets[0])

        if self.mode == 'xyzexpr':
            # enable everything
            for combo in self.xexprcombo, self.yexprcombo, self.zexprcombo:
                combo.setDisabled(False)
                utils.populateCombo(combo, datasets[0])
        elif self.mode == '2dexpr':
            # only enable the z expression button
            self.xexprcombo.setDisabled(True)
            self.yexprcombo.setDisabled(True)
            self.zexprcombo.setDisabled(False)
            utils.populateCombo(self.zexprcombo, datasets[1])
        else:
            # enable everything
            for combo in self.xexprcombo, self.yexprcombo, self.zexprcombo:
                combo.setDisabled(False)

            # put in some examples to help the the user
            utils.populateCombo(self.xexprcombo, ['0:10:0.1'])
            utils.populateCombo(self.yexprcombo, ['0:10:0.1'])
            utils.populateCombo(self.zexprcombo, ['x+y'])

    def reEditDataset(self, ds, dsname):
        """Allow dataset to be edited again."""

        self.namecombo.setEditText(dsname)
        self.linkcheckbox.setChecked(True)

        if isinstance(ds, datasets.Dataset2DXYZExpression):
            self.fromxyzexpr.click()
            self.xexprcombo.setEditText(ds.exprx)
            self.yexprcombo.setEditText(ds.expry)
            self.zexprcombo.setEditText(ds.exprz)

        elif isinstance(ds, datasets.Dataset2DExpression):
            self.from2dexpr.click()
            self.xexprcombo.clearEditText()
            self.yexprcombo.clearEditText()
            self.zexprcombo.setEditText(ds.expr)

        elif isinstance(ds, datasets.Dataset2DXYFunc):
            self.fromxyfunc.click()
            self.xexprcombo.setEditText('%g:%g:%g' % tuple(ds.xstep))
            self.yexprcombo.setEditText('%g:%g:%g' % tuple(ds.ystep))
            self.zexprcombo.setEditText(ds.expr)

        else:
            raise RuntimeError('Invalid dataset type')

    def enableDisableCreate(self):
        """Enable or disable create button."""

        # get contents of combo boxes
        text = {}
        for name in ('xexpr', 'yexpr', 'zexpr', 'name'):
            text[name] = getattr(self, name+'combo').currentText().strip()

        disable = False
        # need name and zexpr
        disable = disable or not text['name'] or not text['zexpr']

        if self.mode == 'xyzexpr':
            # need x and yexpr
            disable = disable or not text['xexpr'] or not text['yexpr']

        elif self.mode == '2dexpr':
            # nothing else
            pass

        elif self.mode == 'xyfunc':
            # need x and yexpr in special step format min:max:step
            disable = disable or ( checkGetStep(text['xexpr']) is None or
                                   checkGetStep(text['yexpr']) is None )

        # finally check button
        self.createbutton.setDisabled(disable)

    def createButtonClickedSlot(self):
        """Create button pressed."""

        text = {}
        for name in ('xexpr', 'yexpr', 'zexpr', 'name'):
            text[name] = getattr(self, name+'combo').currentText().strip()

        link = self.linkcheckbox.checkState() == qt4.Qt.Checked

        # create and apply operation, catching evaluation errors
        try:
            if self.mode == 'xyzexpr':
                # build operation
                op = document.OperationDataset2DCreateExpressionXYZ(
                    text['name'],
                    text['xexpr'], text['yexpr'], text['zexpr'],
                    link)

            elif self.mode == '2dexpr':
                op = document.OperationDataset2DCreateExpression(
                    text['name'], text['zexpr'], link)

            elif self.mode == 'xyfunc':
                xstep = checkGetStep(text['xexpr'])
                ystep = checkGetStep(text['yexpr'])

                # build operation
                op = document.OperationDataset2DXYFunc(
                    text['name'],
                    xstep, ystep,
                    text['zexpr'], link)

            # check expression is okay
            op.validateExpression(self.document)

            # try to make dataset
            self.document.applyOperation(op)
            # forces an evaluation
            self.document.data[text['name']].data

        except (document.CreateDatasetException,
                datasets.DatasetException) as e:

            msg = _("Failed to create dataset '%s'") % text['name']
            s = cstr(e)
            if s:
                msg += ' (%s)' % s
        else:
            msg = _("Created dataset '%s'") % text['name']

        self.notifylabel.setText(msg)
        qt4.QTimer.singleShot(4000, self.notifylabel.clear)

def recreateDataset(mainwindow, document, dataset, datasetname):
    """Open dialog to recreate a DatasetExpression / DatasetRange."""
    dialog = DataCreate2DDialog(mainwindow, document)
    mainwindow.showDialog(dialog)
    dialog.reEditDataset(dataset, datasetname)
