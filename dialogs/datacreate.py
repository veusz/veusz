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

# $Id$

"""Dataset creation dialog."""

import os.path
import veusz.qtall as qt4
import veusz.utils as utils
import veusz.document as document
import veusz.setting as setting

class _DSException(RuntimeError):
    """A class to handle errors while trying to create datasets."""
    pass

class DataCreateDialog(qt4.QDialog):

    def __init__(self, parent, document, *args):
        """Initialise dialog with document."""

        qt4.QDialog.__init__(self, parent, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'datacreate.ui'),
                   self)
        self.document = document

        # create button group to get notification of changes
        self.methodBG = qt4.QButtonGroup(self)
        self.methodBG.addButton(self.valueradio, 0)
        self.methodBG.addButton(self.parametricradio, 1)
        self.methodBG.addButton(self.expressionradio, 2)
        self.connect(self.methodBG, qt4.SIGNAL('buttonClicked(int)'),
                     self.slotMethodChanged)

        # connect create button
        self.connect( self.createbutton, qt4.SIGNAL('clicked()'),
                      self.createButtonClickedSlot )

        # connect notification of document change
        self.connect( self.document, qt4.SIGNAL("sigModified"),
                      self.modifiedDocSlot )

        # set validators for edit controls
        self.numstepsedit.setValidator( qt4.QIntValidator(1, 99999999, self) )
        self.tstartedit.setValidator( qt4.QDoubleValidator(self) )
        self.tendedit.setValidator( qt4.QDoubleValidator(self) )
        self.tstepsedit.setValidator( qt4.QIntValidator(1, 99999999, self) )

        # connect up edit control to update create button status
        for i in (self.numstepsedit, self.tstartedit, self.tendedit,
                  self.tstepsedit, self.nameedit,
                  self.valueedit):
            self.connect( i, qt4.SIGNAL('editTextChanged(const QString &)'),
                          self.editsEditSlot )

        # edit controls for dataset
        self.dsedits = { 'data': self.valueedit, 'serr': self.symerroredit,
                         'perr': self.poserroredit, 'nerr': self.negerroredit }
        
        # set initial state
        self.methodBG.button( setting.settingdb.get('DataCreateDialog_method',
                                                    0) ).click()
        self.editsEditSlot('')

    def done(self, r):
        """Dialog is closed."""
        qt4.QDialog.done(self, r)

        # record values for next time dialog is opened
        d = setting.settingdb
        d['DataCreateDialog_method'] = self.methodBG.checkedId()

    def slotMethodChanged(self, buttonid):
        """Called when a new data creation method is used."""

        # enable and disable correct widgets depending on method
        isvalue = buttonid == 0
        self.valuehelperlabel.setVisible(isvalue)
        self.numstepsedit.setEnabled(isvalue)

        isparametric = buttonid == 1
        self.parametrichelperlabel.setVisible(isparametric)
        self.tstartedit.setEnabled(isparametric)
        self.tendedit.setEnabled(isparametric)
        self.tstepsedit.setEnabled(isparametric)

        isfunction = buttonid == 2
        self.expressionhelperlabel.setVisible(isfunction)
        self.linkcheckbox.setEnabled(isfunction)

        # enable/disable create button
        self.editsEditSlot('')

    def modifiedDocSlot(self):
        """Update create button if document changes."""
        self.editsEditSlot('')

    def editsEditSlot(self, dummytext):
        """Enable/disable createbutton."""

        # dataset name checks
        dstext = unicode(self.nameedit.text())
        dsvalid = utils.validateDatasetName(dstext)
        dsexists = dstext in self.document.data

        # check other edit controls
        method = self.methodBG.checkedId()
        if method == 0:
            # value
            editsokay = self.numstepsedit.hasAcceptableInput()
        elif method == 1:
            # parametric
            editsokay = (self.tstartedit.hasAcceptableInput() and
                         self.tendedit.hasAcceptableInput() and
                         self.tstepsedit.hasAcceptableInput())
        else:
            # function
            editsokay = True

        # we needs some input on the value
        if len(unicode(self.valueedit.text())) == 0:
            editsokay = False

        self.createbutton.setEnabled(dsvalid and (not dsexists) and editsokay)

    def createButtonClickedSlot(self):
        """Create button pressed."""
        
        try:
            name = unicode( self.nameedit.text() )
            
            fn = [ self.createFromRange,
                   self.createParametric,
                   self.createFromExpression ][self.methodBG.checkedId()]

            # make a new dataset from the returned data
            fn(name)

            self.statuslabel.setText("Created dataset '%s'" % name)

        except (document.CreateDatasetException,
                document.DatasetException, _DSException), e:
            # all bad roads lead here - take exception string and tell user
            self.statuslabel.setText("Creation failed")
            qt4.QMessageBox("Veusz",
                           unicode(e),
                           qt4.QMessageBox.Warning,
                           qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                           qt4.QMessageBox.NoButton,
                           qt4.QMessageBox.NoButton,
                           self).exec_()

    def createFromRange(self, name):
        """Make dataset from a range or constant.
        name is the name of the dataset
        
        Raises _DSException if error
        """

        numsteps = int( unicode(self.numstepsedit.text()) )

        # go over each of the ranges / values
        vals = {}
        for key, cntrl in self.dsedits.iteritems():
            text = unicode( cntrl.text() ).strip()

            if not text:
                continue
                
            if text.find(':') != -1:
                # an actual range
                parts = text.split(':')
                
                if len(parts) != 2:
                    raise _DSException("Incorrect range format, use form 1:10")
                try:
                    minval, maxval = float(parts[0]), float(parts[1])
                except ValueError:
                    raise _DSException("Invalid number in range")

            else:
                try:
                    minval = float(text)
                except ValueError:
                    raise _DSException("Invalid number")
                maxval = minval
                
            vals[key] = (minval, maxval)
            
        op = document.OperationDatasetCreateRange(name, numsteps, vals)
        self.document.applyOperation(op)

    def createParametric(self, name):
        """Use a parametric form to create the dataset.

        Raises _DSException if error
        """
        t0 = float( unicode(self.tstartedit.text()) )
        t1 = float( unicode(self.tendedit.text()) )
        numsteps = int( unicode(self.tstepsedit.text()) )

        # get expressions
        vals = {}
        for key, cntrl in self.dsedits.iteritems():
            text = unicode( cntrl.text() ).strip()
            if text:
                vals[key] = text
           
        op = document.OperationDatasetCreateParameteric(name, t0, t1, numsteps,
                                                        vals)
        self.document.applyOperation(op)
      
    def createFromExpression(self, name):
        """Create a dataset based on the expressions given."""

        # get expression for each part of the dataset
        vals = {}
        for key, cntrl in self.dsedits.iteritems():
            text = unicode( cntrl.text() ).strip()
            if text:
                vals[key] = text

        link = self.linkcheckbox.checkState() == qt4.Qt.Checked
        op = document.OperationDatasetCreateExpression(name, vals, link)
        op.validateExpression(self.document)
        self.document.applyOperation(op)
