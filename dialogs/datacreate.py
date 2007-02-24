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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id: $

"""Dataset creation dialog."""

import os.path
import veusz.qtall as qt4
import veusz.utils as utils
import veusz.document as document

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

        # connect everything up
        self.connect( self.valueradio, qt4.SIGNAL('toggled(bool)'),
                      self.valueToggledSlot )
        self.connect( self.parametricradio, qt4.SIGNAL('toggled(bool)'),
                      self.parametricToggledSlot )
        self.connect( self.expressionradio, qt4.SIGNAL('toggled(bool)'),
                      self.expressionToggledSlot )

        self.connect( self.createbutton, qt4.SIGNAL('clicked()'),
                      self.createButtonClickedSlot )

        self.connect( self.document, qt4.SIGNAL("sigModified"),
                      self.modifiedDocSlot )

        self.numstepsedit.setValidator( qt4.QIntValidator(1, 99999999, self) )
        self.tstartedit.setValidator( qt4.QDoubleValidator(self) )
        self.tendedit.setValidator( qt4.QDoubleValidator(self) )
        self.tstepsedit.setValidator( qt4.QIntValidator(1, 99999999, self) )

        # connect up edit control to update create button status
        for i in (self.numstepsedit, self.tstartedit, self.tendedit,
                  self.tstepsedit, self.nameedit,
                  self.valueedit):
            self.connect( i, qt4.SIGNAL('textChanged(const QString &)'),
                          self.editsEditSlot )

        # edit controls for dataset
        self.dsedits = { 'data': self.valueedit, 'serr': self.symerroredit,
                         'perr': self.poserroredit, 'nerr': self.negerroredit }
        
        # set initial state
        self.valueradio.toggle()
        self.editsEditSlot('')

    def valueToggledSlot(self, pressed):
        """Enable/disable correct edit controls."""
        self.setRadioState('value',
                           'Enter constant values here, leave blank if appropriate, '
                           'or enter an inclusive range, e.g. 1:10',
                           False)

    def parametricToggledSlot(self, pressed):
        """Enable/disable correct edit controls."""
        self.setRadioState('parametric',
                           'Enter expressions as a function of t, or leave blank',
                           False)

    def expressionToggledSlot(self, pressed):
        """Enable/disable correct edit controls."""
        self.setRadioState('expression',
                           'Enter expressions as a function of other datasets.'
                           ' Append suffixes _data, _serr, _nerr and _perr to '
                           'use different parts of datasets.',
                           True)

    def setRadioState(self, radio, helper, allowlink):
        """Enable/disable edit controls for radio button."""

        # enable/disable radio buttons
        isvalue = (radio == 'value')
        self.numstepsedit.setEnabled(isvalue)

        isparametric = (radio == 'parametric')
        self.tstartedit.setEnabled(isparametric)
        self.tendedit.setEnabled(isparametric)
        self.tstepsedit.setEnabled(isparametric)

        # set some help text
        self.valuehelperlabel.setText(helper)

        self.linkcheckbox.setEnabled(allowlink)

        # keep track of state
        self.radiostate = radio

        # update button
        self.editsEditSlot('')

    def modifiedDocSlot(self):
        self.editsEditSlot('')

    def editsEditSlot(self, dummytext):
        """Enable/disable createbutton."""

        # dataset name checks
        dstext = unicode(self.nameedit.text())
        dsvalid = utils.validateDatasetName(dstext)
        dsexists = dstext in self.document.data

        # check other edit controls
        if self.radiostate == 'value':
            editsokay = self.numstepsedit.hasAcceptableInput()
        elif self.radiostate == 'parametric':
            editsokay = (self.tstartedit.hasAcceptableInput() and
                         self.tendedit.hasAcceptableInput() and
                         self.tstepsedit.hasAcceptableInput())
        else:
            editsokay = True

        # we needs some input on the value
        if len(unicode(self.valueedit.text())) == 0:
            editsokay = False

        self.createbutton.setEnabled(dsvalid and (not dsexists) and editsokay)

    def createButtonClickedSlot(self):
        """Create button pressed."""
        
        try:
            name = unicode( self.nameedit.text() )
            
            fn = { 'value': self.createFromRange,
                   'parametric': self.createParametric,
                   'expression': self.createFromExpression } [self.radiostate]

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
