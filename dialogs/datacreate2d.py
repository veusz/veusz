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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

"""Dataset creation dialog for 2d data."""

import os.path
import re

import veusz.qtall as qt4
import veusz.utils as utils
import veusz.document as document
from veusz.setting.controls import populateCombo

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

class DataCreate2DDialog(qt4.QDialog):

    def __init__(self, parent, document, *args):
        """Initialise dialog with document."""

        qt4.QDialog.__init__(self, parent, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'datacreate2d.ui'),
                   self)
        self.document = document

        self.connect( self.createbutton, qt4.SIGNAL('clicked()'),
                      self.createButtonClickedSlot )

        self.connect( self.fromxyfunc, qt4.SIGNAL('toggled(bool)'),
                      self.fromxyfuncSlot )
        self.connect( self.fromxyzexpr, qt4.SIGNAL('toggled(bool)'),
                      self.fromxyzexprSlot )
        self.connect( self.from2dexpr, qt4.SIGNAL('toggled(bool)'),
                      self.from2dexprSlot )

        self.connect(document, qt4.SIGNAL('sigModified'),
                     self.updateDatasetLists)

        for combo in (self.namecombo, self.xexprcombo, self.yexprcombo,
                      self.zexprcombo):
            self.connect(combo,
                         qt4.SIGNAL('editTextChanged(const QString&)'),
                         self.enableDisableCreate)

        self.fromxyzexpr.toggle()

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

    def updateDatasetLists(self):
        """Update controls depending on selected mode."""

        # get list of 1d and 2d numeric datasets
        datasets = [[],[]]
        for name, ds in self.document.data.iteritems():
            if ds.datatype == 'numeric':
                datasets[ds.dimensions-1].append(name)
        datasets[0].sort()
        datasets[1].sort()        

        # help the user by listing existing datasets
        populateCombo(self.namecombo, datasets[0])

        if self.mode == 'xyzexpr':
            # enable everything
            for combo in self.xexprcombo, self.yexprcombo, self.zexprcombo:
                combo.setDisabled(False)
                populateCombo(combo, datasets[0])
        elif self.mode == '2dexpr':
            # only enable the z expression button
            self.xexprcombo.setDisabled(True)
            self.yexprcombo.setDisabled(True)
            self.zexprcombo.setDisabled(False)
            populateCombo(self.zexprcombo, datasets[1])
        else:
            # enable everything
            for combo in self.xexprcombo, self.yexprcombo, self.zexprcombo:
                combo.setDisabled(False)

            # put in some examples to help the the user
            populateCombo(self.xexprcombo, ['0:10:0.1'])
            populateCombo(self.yexprcombo, ['0:10:0.1'])
            populateCombo(self.zexprcombo, ['x+y'])

    def enableDisableCreate(self):
        """Enable or disable create button."""
        
        text = {}
        for name in ('xexpr', 'yexpr', 'zexpr', 'name'):
            text[name] = unicode(getattr(self, name+'combo').currentText()).strip()

        disable = False
        disable = disable or not text['name'] or not text['zexpr']
        
        if self.mode == 'xyzexpr':
            disable = disable or not text['xexpr'] or not text['yexpr']

        elif self.mode == '2dexpr':
            disable = disable or not text['zexpr']
                
        elif self.mode == 'xyfunc':
            disable = disable or not text['zexpr']
            disable = disable or ( checkGetStep(text['xexpr']) is None or
                                   checkGetStep(text['yexpr']) is None )

        self.createbutton.setDisabled(disable)
        
    def createButtonClickedSlot(self):
        """Create button pressed."""

        text = {}
        for name in ('xexpr', 'yexpr', 'zexpr', 'name'):
            text[name] = unicode(getattr(self, name+'combo').currentText()).strip()

        link = self.linkcheckbox.checkState() == qt4.Qt.Checked
        if self.mode == 'xyzexpr':
            op = document.OperationDataset2DCreateExpressionXYZ(
                text['name'],
                text['xexpr'], text['yexpr'], text['zexpr'],
                link)
            # FIXME need to catch exceptions here
            self.document.applyOperation(op)
            self.document.data[text['name']].data

        elif self.mode == '2dexpr':
            pass

        elif self.mode == 'xyfunc':
            xstep = checkGetStep(text['xexpr'])
            ystep = checkGetStep(text['yexpr'])
            
            op = document.OperationDataset2DXYFunc(
                text['name'],
                xstep, ystep,
                text['zexpr'], link)
            # FIXME need to catch exceptions here
            self.document.applyOperation(op)
            self.document.data[text['name']].data

        self.notifylabel.setText("Created dataset '%s'" % text['name'])
        qt4.QTimer.singleShot(4000, self.clearNotifySlot)

    def clearNotifySlot(self):
        self.notifylabel.setText("")
        
