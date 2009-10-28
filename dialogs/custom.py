#    Copyright (C) 2009 Jeremy S. Sanders
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
import re

import veusz.qtall as qt4
import veusz.utils as utils
import veusz.document as document

class CustomDialog(qt4.QDialog):
    """Class to load help for standard veusz import."""
    def __init__(self, parent, document):
        qt4.QDialog.__init__(self, parent)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'custom.ui'),
                   self)
        self.document = document

        # setup table
        self.definitionTree.setColumnCount(2)
        self.connect(self.definitionTree,
                     qt4.SIGNAL('currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)'),
                     self.slotItemChanged)

        # connect notification of document change
        self.connect( self.document, qt4.SIGNAL('sigModified'),
                      self.updateList )

        # keep track of what's in edit boxes
        self.connect(self.nameEdit,
                     qt4.SIGNAL('textChanged(const QString&)'),
                     self.validateAddRemove)
        self.connect(self.definitionEdit,
                     qt4.SIGNAL('textChanged(const QString&)'),
                     self.validateAddRemove)

        # connect different types of radio
        self.connect(self.functionsRadio, qt4.SIGNAL('clicked()'),
                     self.slotFunctionsClicked)
        self.connect(self.constantsRadio, qt4.SIGNAL('clicked()'),
                     self.slotConstantsClicked)
        # setup radio buttons
        self.functionsRadio.click()

        # connect buttons to slots
        self.connect(self.addButton, qt4.SIGNAL('clicked()'), self.slotAdd)
        self.connect(self.removeButton, qt4.SIGNAL('clicked()'),
                     self.slotRemove)
        self.connect(self.saveButton, qt4.SIGNAL('clicked()'), self.slotSave)
        self.connect(self.loadButton, qt4.SIGNAL('clicked()'), self.slotLoad)

    def updateList(self):
        """Update list of items in list."""

        self.itemlist = dict( getattr(self.document, 'custom_'+self.mode) )
        self.definitionTree.clear()
        self.definitionTree.setHeaderLabels( ['Name', 'Value'] )

        items = self.itemlist
        keys = items.keys()
        keys.sort()

        for name in keys:
            self.definitionTree.addTopLevelItem(
                qt4.QTreeWidgetItem([name, unicode(items[name])]) )

        self.validateAddRemove()

    def slotFunctionsClicked(self):
        """Functions definitions radio clicked."""
        self.mode = 'functions'
        self.updateList()

    def slotConstantsClicked(self):
        """Constant definitions radio clicked."""
        self.mode = 'constants'
        self.updateList()

    def slotItemChanged(self, current, previous):
        """Item clicked on in box."""
        if current is None:
            name, defn = '', ''
        else:
            name, defn = current.text(0), current.text(1)
        self.nameEdit.setText(name)
        self.definitionEdit.setText(defn)

    def validateAddRemove(self, text=None):
        """Text changed in edit boxes."""
        name = unicode(self.nameEdit.text()).strip()

        if self.mode == 'constants':
            # check name is valid before allowing it to be added
            valid = re.match(r'[A-Za-z_][A-Za-z0-9_]*', name) is not None
        elif self.mode == 'functions':
            # check func(a,b,c...)
            valid = re.match(r'[A-Za-z_][A-Za-z0-9_]*'
                             r'\('
                             r'([A-Za-z_][A-Za-z0-9_]*)'
                             r'(,[ ]*[A-Za-z_][A-Za-z0-9_]*)*'
                             r'\)', name) is not None

        self.addButton.setEnabled(valid)

        # only allow existing entries to be removed
        self.removeButton.setEnabled(name in self.itemlist)

    def slotAdd(self):
        """Add an entry."""

        name = unicode(self.nameEdit.text())
        val = unicode(self.definitionEdit.text())

        if self.mode == 'constants':
            try:
                # try to make a float if possible
                val = float(val)
            except ValueError:
                pass

        self.itemlist[name] = val
        self.document.applyOperation( document.OperationSetCustom(
                self.mode, self.itemlist) )
        self.updateList()
    
    def slotRemove(self):
        """Remove an entry."""

        name = unicode(self.nameEdit.text())
        del self.itemlist[name]
        self.document.applyOperation( document.OperationSetCustom(
                self.mode, self.itemlist) )

    def slotSave(self):
        """Save entries."""
        pass

    def slotLoad(self):
        """Load entries."""
        pass
