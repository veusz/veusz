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

import veusz.qtall as qt4
import veusz.utils as utils
import veusz.document as document
from veuszdialog import VeuszDialog

class CustomItemModel(qt4.QAbstractTableModel):
    """A model for editing custom items."""

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
        """Lookup data in document customs list."""
        if role in (qt4.Qt.DisplayRole, qt4.Qt.EditRole):
            d = self.document.customs[index.row()][index.column()]
            return qt4.QVariant(d)
        elif role == qt4.Qt.ToolTipRole:
            return ('Constant, function or import',
                    'Name for constant, function(arg1, arg2...) or module name',
                    'Expression defining constant or function, '
                    'or list of symbols to import from module')[index.column()]

        return qt4.QVariant()

    def flags(self, index):
        """Items are editable"""
        return ( qt4.Qt.ItemIsSelectable | qt4.Qt.ItemIsEditable |
                 qt4.Qt.ItemIsEnabled )

    def headerData(self, section, orientation, role):
        """Return the headers at the top of the view."""
        if role == qt4.Qt.DisplayRole:
            if orientation == qt4.Qt.Horizontal:
                return qt4.QVariant( ('Type', 'Name', 'Definition')[section] )
            else:
                return qt4.QVariant(str(section+1))
        return qt4.QVariant()

    def doUpdate(self):
        """Document changed."""
        self.emit( qt4.SIGNAL('layoutChanged()') )

    def addNewEntry(self):
        """Add a new row to the list of custom items."""
        newcustom = list(self.document.customs)
        newcustom.append( ['constant', 'name', 'None'] )
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
                ok = value in ('constant', 'function', 'import')
            elif col == 1:
                dtype = self.document.customs[row][0]
                if dtype == 'constant':
                    ok = document.identifier_re.match(value) is not None
                elif dtype == 'function':
                    ok = document.function_re.match(value) is not None
                elif dtype == 'import':
                    ok = True
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
        w.addItems(['constant', 'function', 'import'])
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

class CustomDialog(VeuszDialog):
    """A dialog to create or edit custom constant and function
    definitions."""

    def __init__(self, parent, document):
        VeuszDialog.__init__(self, parent, 'custom.ui')
        self.document = document

        self.model = CustomItemModel(self, document)
        self.definitionView.setModel(self.model)

        self.combodeligate = ComboTypeDeligate(self)
        self.definitionView.setItemDelegateForColumn(0, self.combodeligate)

        # connect buttons to slots
        self.connect(self.addButton, qt4.SIGNAL('clicked()'), self.slotAdd)
        self.connect(self.removeButton, qt4.SIGNAL('clicked()'),
                     self.slotRemove)
        self.connect(self.upButton, qt4.SIGNAL('clicked()'), self.slotUp)
        self.connect(self.downButton, qt4.SIGNAL('clicked()'), self.slotDown)

        self.connect(self.saveButton, qt4.SIGNAL('clicked()'), self.slotSave)
        self.connect(self.loadButton, qt4.SIGNAL('clicked()'), self.slotLoad)

        # recent button shows list of recently used files for loading
        self.connect(self.recentButton, qt4.SIGNAL('filechosen'),
                     self.loadFile)
        self.recentButton.setSetting('customdialog_recent')

    def loadFile(self, filename):
        """Load the given file."""
        self.document.applyOperation(
            document.OperationLoadCustom(filename) )

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
        filename = self.parent()._fileSaveDialog(
            'vsz', 'Veusz document', 'Save custom definitions')
        if filename:
            try:
                f = open(filename, 'w')
                self.document.saveCustomFile(f)
                f.close()
                self.recentButton.addFile(filename)

            except IOError:
                qt4.QMessageBox.critical(self, "Veusz",
                                         "Cannot save as '%s'" % filename)

    def slotLoad(self):
        """Load entries."""

        filename = self.parent()._fileOpenDialog(
            'vsz', 'Veusz document', 'Load custom definitions')
        if filename:
            try:
                self.loadFile(filename)
            except IOError:
                qt4.QMessageBox.critical(self, "Veusz",
                                         "Cannot load custom definitions '%s'"
                                         % filename)
            else:
                # add to recent file list
                self.recentButton.addFile(filename)
