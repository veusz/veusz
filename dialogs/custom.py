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

class CustomItemModel(qt4.QAbstractTableModel):
    """A model for editing custom items."""

    name_re = re.compile('[A-Za-z_][A-Z-a-z0-9_]*')

    def __init__(self, parent, document):
        qt4.QAbstractTableModel.__init__(self, parent)
        self.document = document

        # connect notification of document change
        self.connect( self.document, qt4.SIGNAL('sigModified'),
                      self.doUpdate )

    def rowCount(self, parent):
        return len(self.document.customs)

    def columnCount(self, parent):
        return 3

    def data(self, index, role):
        if role in (qt4.Qt.DisplayRole, qt4.Qt.EditRole):
            return qt4.QVariant( self.document.customs[index.row()][index.column()] )
        return qt4.QVariant()

    def flags(self, index):
        return ( qt4.Qt.ItemIsSelectable | qt4.Qt.ItemIsEditable |
                 qt4.Qt.ItemIsEnabled )

    def headerData(self, section, orientation, role):
        if role == qt4.Qt.DisplayRole:
            if orientation == qt4.Qt.Horizontal:
                return qt4.QVariant( ('Name', 'Type', 'Definition')[section] )
            else:
                return qt4.QVariant(str(section+1))
        return qt4.QVariant()

    def doUpdate(self):
        """Document changed."""
        self.emit( qt4.SIGNAL('layoutChanged()') )

    def addNewEntry(self):
        """Add a new row to the list of custom items."""
        newcustom = list(self.document.customs)
        newcustom.append( ['name', 'constant', 'None'] )
        self.document.applyOperation( document.OperationSetCustom(newcustom) )

    def deleteEntry(self, num):
        """Delete row num."""
        newcustom = list(self.document.customs)
        del newcustom[num]
        self.document.applyOperation( document.OperationSetCustom(newcustom) )

    def moveUpEntry(self, num):
        """Move up entry."""
        if num == 0 or len(self.document.customs) == 0:
            return
        newcustom = list(self.document.customs)
        row = newcustom[num]
        del newcustom[num]
        newcustom.insert(num-1, row)
        self.document.applyOperation( document.OperationSetCustom(newcustom) )

    def moveDownEntry(self, num):
        """Move down entry."""
        if num >= len(self.document.customs)-1:
            return
        newcustom = list(self.document.customs)
        row = newcustom[num]
        del newcustom[num]
        newcustom.insert(num+1, row)
        self.document.applyOperation( document.OperationSetCustom(newcustom) )

    def setData(self, index, value, role):
        """Edit an item."""
        if index.isValid() and role == qt4.Qt.EditRole:
            col = index.column()
            row = index.row()
            value = unicode(value.toString())

            if col == 0:
                ok = self.name_re.match(value) is not None
            elif col == 1:
                ok = value in ('constant', 'function')
            else:
                ok = True
            if not ok:
                return False

            newcustom = list(self.document.customs)
            newcustom[row][col] = value
            self.document.applyOperation(
                document.OperationSetCustom(newcustom) )

            self.emit(qt4.SIGNAL('dataChanged(const QModelIndex &, const QModelIndex &'),
                      index, index)
            return True
        return False

class ComboTypeDeligate(qt4.QItemDelegate):
    """This class is for choosing between the constant and function
    types in a combo box in a model view."""

    def createEditor(self, parent, option, index):
        """Create combobox for editing type."""
        w = qt4.QComboBox(parent)
        w.addItems(['constant', 'function'])
        w.setFocusPolicy( qt4.Qt.StrongFocus )
        return w

    def setEditorData(self, editor, index):
        """Update data in editor."""
        i = editor.findText( index.data().toString() )
        editor.setCurrentIndex(i)

    def setModelData(self, editor, model, index):
        """Update data in model."""
        model.setData(index, qt4.QVariant(editor.currentText()),
                      qt4.Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        """Update editor geometry."""
        editor.setGeometry(option.rect)

class CustomDialog(qt4.QDialog):
    """Class to load help for standard veusz import."""
    def __init__(self, parent, document):
        qt4.QDialog.__init__(self, parent)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'custom.ui'),
                   self)
        self.document = document

        self.model = CustomItemModel(self, document)
        self.definitionView.setModel(self.model)

        self.combodeligate = ComboTypeDeligate(self)
        self.definitionView.setItemDelegateForColumn(1, self.combodeligate)

        # connect buttons to slots
        self.connect(self.addButton, qt4.SIGNAL('clicked()'), self.slotAdd)
        self.connect(self.removeButton, qt4.SIGNAL('clicked()'),
                     self.slotRemove)
        self.connect(self.upButton, qt4.SIGNAL('clicked()'), self.slotUp)
        self.connect(self.downButton, qt4.SIGNAL('clicked()'), self.slotDown)

        self.connect(self.saveButton, qt4.SIGNAL('clicked()'), self.slotSave)
        self.connect(self.loadButton, qt4.SIGNAL('clicked()'), self.slotLoad)

    def slotAdd(self):
        """Add an entry."""
        self.model.addNewEntry()

    def slotRemove(self):
        """Remove an entry."""
        selrows = self.definitionView.selectionModel().selectedRows()
        if len(selrows) != 0:
            self.model.deleteEntry(selrows[0].row())

    def slotUp(self):
        """Move item up list."""
        selrows = self.definitionView.selectionModel().selectedRows()
        if len(selrows) != 0:
            self.model.moveUpEntry(selrows[0].row())

    def slotDown(self):
        """Move item down list."""
        selrows = self.definitionView.selectionModel().selectedRows()
        if len(selrows) != 0:
            self.model.moveDownEntry(selrows[0].row())

    def slotSave(self):
        """Save entries."""
        pass

    def slotLoad(self):
        """Load entries."""
        pass
