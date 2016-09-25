#    Copyright (C) 2006 Jeremy S. Sanders
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

"""Dataset creation dialog."""

from ..compat import cstr
from .. import qtall as qt4
from .. import utils
from .. import document
from .. import datasets
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="DataCreateDialog"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class _DSException(RuntimeError):
    """A class to handle errors while trying to create datasets."""
    pass

class DataCreateDialog(VeuszDialog):
    """Dialog to create datasets.

    They can be created from numerical ranges, parametrically or from
    expressions involving other dataset."""

    def __init__(self, parent, document):
        """Initialise dialog with document."""

        VeuszDialog.__init__(self, parent, 'datacreate.ui')
        self.document = document

        # create button group to get notification of changes
        self.methodGroup.radioClicked.connect(self.slotMethodChanged)

        # connect create button
        self.createbutton = self.buttonBox.addButton(
            _("C&reate"), qt4.QDialogButtonBox.ApplyRole )
        self.replacebutton = self.buttonBox.addButton(
            _("&Replace"), qt4.QDialogButtonBox.ApplyRole )

        self.buttonBox.button(
            qt4.QDialogButtonBox.Reset).clicked.connect(self.resetButtonClicked)

        self.createbutton.clicked.connect(self.createButtonClicked)
        self.replacebutton.clicked.connect(self.createButtonClicked)

        # connect notification of document change
        self.document.signalModified.connect(self.modifiedDocSlot)

        # set validators for edit controls
        self.numstepsedit.setValidator( qt4.QIntValidator(1, 99999999, self) )
        self.tstartedit.setValidator( qt4.QDoubleValidator(self) )
        self.tendedit.setValidator( qt4.QDoubleValidator(self) )
        self.tstepsedit.setValidator( qt4.QIntValidator(1, 99999999, self) )

        # connect up edit control to update create button status
        for edit in (self.numstepsedit, self.tstartedit, self.tendedit,
                     self.tstepsedit, self.nameedit,
                     self.valueedit):
            edit.editTextChanged.connect(self.editsEditSlot)

        self.nameedit.currentIndexChanged[int].connect(self.datasetSelected)

        # edit controls for dataset
        self.dsedits = { 'data': self.valueedit, 'serr': self.symerroredit,
                         'perr': self.poserroredit, 'nerr': self.negerroredit }
        
        # update button state
        self.editsEditSlot('')

    def slotMethodChanged(self, button):
        """Called when a new data creation method is used."""

        # enable and disable correct widgets depending on method
        isvalue = button is self.valueradio
        self.valuehelperlabel.setVisible(isvalue)
        self.numstepsedit.setEnabled(isvalue)

        isparametric = button is self.parametricradio
        self.parametrichelperlabel.setVisible(isparametric)
        self.tstartedit.setEnabled(isparametric)
        self.tendedit.setEnabled(isparametric)
        self.tstepsedit.setEnabled(isparametric)

        isfunction = button is self.expressionradio
        self.expressionhelperlabel.setVisible(isfunction)

        # enable/disable create button
        self.editsEditSlot('')

    def modifiedDocSlot(self):
        """Update create button if document changes."""
        self.editsEditSlot('')

    def datasetSelected(self, index):
        """If dataset is selected from drop down box, reload entries
        for editing."""

        if index >= 0:
            dsname = self.nameedit.text()
            if dsname in self.document.data:
                self.reEditDataset(self.document.data[dsname], dsname)

    def reEditDataset(self, ds, dsname):
        """Given a dataset name, allow it to be edited again
        (if it is editable)."""

        if isinstance(ds, datasets.DatasetExpression): 
            # change selected method
            if ds.parametric is None:
                # standard expression
                self.expressionradio.click()
            else:
                # parametric dataset
                self.parametricradio.click()
                p = ds.parametric
                self.tstartedit.setText( '%g' % p[0] )
                self.tendedit.setText( '%g' % p[1] )
                self.tstepsedit.setText( str(p[2]) )

            # make sure name is set
            self.nameedit.setText(dsname)
            # set expressions
            for part in self.dsedits:
                text = ds.expr[part]
                if text is None:
                    text = ''
                self.dsedits[part].setText(text)

        elif isinstance(ds, datasets.DatasetRange):
            # change selected method
            self.valueradio.click()
            # make sure name is set
            self.nameedit.setText(dsname)
            # set expressions
            for part in self.dsedits:
                data = getattr(ds, 'range_%s' % part)
                if data is None:
                    text = ''
                else:
                    text = '%g:%g' % data
                self.dsedits[part].setText(text)

    def editsEditSlot(self, dummytext):
        """Enable/disable createbutton."""

        # dataset name checks
        dstext = self.nameedit.text()
        dsvalid = utils.validateDatasetName(dstext)
        dsexists = dstext in self.document.data

        # check other edit controls
        method = self.methodGroup.getRadioChecked()
        if method is self.valueradio:
            # value
            editsokay = self.numstepsedit.hasAcceptableInput()
        elif method is self.parametricradio:
            # parametric
            editsokay = (self.tstartedit.hasAcceptableInput() and
                         self.tendedit.hasAcceptableInput() and
                         self.tstepsedit.hasAcceptableInput())
        else:
            # function
            editsokay = True

        # we needs some input on the value
        if not self.valueedit.text():
            editsokay = False

        # hide / show create button depending whether dataset exists
        self.createbutton.setVisible(not dsexists)
        self.replacebutton.setVisible(dsexists)
        
        # enable buttons if expressions valid
        enabled = dsvalid and editsokay
        self.createbutton.setEnabled(enabled)
        self.replacebutton.setEnabled(enabled)

    def resetButtonClicked(self):
        """Reset button clicked - reset dialog."""

        for cntrl in (self.valueedit, self.symerroredit, self.poserroredit,
                      self.negerroredit, self.numstepsedit,
                      self.tstartedit, self.tendedit, self.tstepsedit,
                      self.nameedit):
            cntrl.setEditText("")
                      
        self.linkcheckbox.setChecked(True)
        self.valueradio.click()

    def createButtonClicked(self):
        """Create button pressed."""
        
        dsname = self.nameedit.text()
        dsexists = dsname in self.document.data

        try:
            # select function to create dataset with
            createfn = {
                self.valueradio: self.createFromRange,
                self.parametricradio: self.createParametric,
                self.expressionradio: self.createFromExpression }[
                self.methodGroup.getRadioChecked()]

            # make a new dataset using method
            op = createfn(dsname)
            self.document.applyOperation(op)

            if dsexists:
                status = _("Replaced dataset '%s'") % dsname
            else:
                status = _("Created dataset '%s'") % dsname
            self.statuslabel.setText(status)

        except (document.CreateDatasetException,
                datasets.DatasetException, _DSException) as e:

            # all bad roads lead here - take exception string and tell user
            if dsexists:
                status = _("Replacement failed")
            else:
                status = _("Creation failed")

            if cstr(e) != '':
                status += ': %s' % cstr(e)

            self.statuslabel.setText(status)
            
    def createFromRange(self, name):
        """Make dataset from a range or constant.
        name is the name of the dataset
        
        Raises _DSException if error
        """

        numsteps = int(self.numstepsedit.text())

        # go over each of the ranges / values
        vals = {}
        for key, cntrl in self.dsedits.items():
            text = cntrl.text().strip()

            if not text:
                continue
                
            if text.find(':') != -1:
                # an actual range
                parts = text.split(':')
                
                if len(parts) != 2:
                    raise _DSException(_("Incorrect range format, use form 1:10"))
                try:
                    minval, maxval = float(parts[0]), float(parts[1])
                except ValueError:
                    raise _DSException(_("Invalid number in range"))

            else:
                try:
                    minval = float(text)
                except ValueError:
                    raise _DSException(_("Invalid number"))
                maxval = minval
                
            vals[key] = (minval, maxval)
            
        linked = self.linkcheckbox.checkState() == qt4.Qt.Checked
        return document.OperationDatasetCreateRange(
            name, numsteps, vals, linked=linked)

    def createParametric(self, name):
        """Use a parametric form to create the dataset.

        Raises _DSException if error
        """
        t0 = float(self.tstartedit.text())
        t1 = float(self.tendedit.text())
        numsteps = int(self.tstepsedit.text())

        # get expressions
        vals = {}
        for key, cntrl in self.dsedits.items():
            text = cntrl.text().strip()
            if text:
                vals[key] = text

        linked = self.linkcheckbox.checkState() == qt4.Qt.Checked
        return document.OperationDatasetCreateParameteric(
            name, t0, t1, numsteps, vals, linked=linked)

    def createFromExpression(self, name):
        """Create a dataset based on the expressions given."""

        # get expression for each part of the dataset
        vals = {}
        for key, cntrl in self.dsedits.items():
            text = cntrl.text().strip()
            if text:
                vals[key] = text

        link = self.linkcheckbox.checkState() == qt4.Qt.Checked
        op = document.OperationDatasetCreateExpression(name, vals, link)
        if not op.validateExpression(self.document):
            raise _DSException()
        return op

def recreateDataset(mainwindow, document, dataset, datasetname):
    """Open dialog to recreate a DatasetExpression / DatasetRange."""
    dialog = DataCreateDialog(mainwindow, document)
    mainwindow.showDialog(dialog)
    dialog.reEditDataset(dataset, datasetname)
