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

from __future__ import division
import ast

from ..compat import cstrerror
from .. import qtall as qt4
from .. import document
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="CustomDialog"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class CustomItemModel(qt4.QAbstractTableModel):
    """A model for editing custom items."""

    def __init__(self, parent, document):
        qt4.QAbstractTableModel.__init__(self, parent)
        self.document = document

        # connect notification of document change
        document.signalModified.connect(self.doUpdate)

    def rowCount(self, parent):
        if parent.isValid():
            # qt docs say we have to return zero here, and we get a
            # crash if we don't (pressing right arrow)
            return 0
        return len(self.document.evaluate.customs)

    def columnCount(self, parent):
        if parent.isValid():
            return 0
        return 3

    def data(self, index, role):
        """Lookup data in document.evaluate.customs list."""
        if role in (qt4.Qt.DisplayRole, qt4.Qt.EditRole):
            cust = self.document.evaluate.customs[index.row()]
            if cust[0] == 'colormap' and index.column() == 2:
                # values are not a string
                return repr(cust[2])
            else:
                return cust[index.column()]
        elif role == qt4.Qt.ToolTipRole:
            return (_('Constant, function, import or colormap'),
                    _('Constant or colormap: enter name\n'
                      'Function: enter functionname(arg1, arg2...)\n'
                      'Import: enter module name'),
                    _('Constant or function: expression giving value\n'
                      'Import: comma or space separated list of symbols to import\n'
                      'Colormap: (R,G,B[,alpha]) list surrounded by brackets, e.g. ((10,10,10), (20,20,20,128))'),
                    )[index.column()]

        return None

    def flags(self, index):
        """Items are editable"""
        return ( qt4.Qt.ItemIsSelectable | qt4.Qt.ItemIsEditable |
                 qt4.Qt.ItemIsEnabled )

    def headerData(self, section, orientation, role):
        """Return the headers at the top of the view."""
        if role == qt4.Qt.DisplayRole:
            if orientation == qt4.Qt.Horizontal:
                return (_('Type'), _('Name'),
                        _('Definition'))[section]
            else:
                return str(section+1)
        return None

    def doUpdate(self):
        """Document changed."""
        self.layoutChanged.emit()

    def addNewEntry(self):
        """Add a new row to the list of custom items."""
        newcustom = list(self.document.evaluate.customs)
        newcustom.append( ['constant', 'name', 'None'] )
        self.document.applyOperation( document.OperationSetCustom(newcustom) )

    def deleteEntry(self, num):
        """Delete row num."""
        newcustom = list(self.document.evaluate.customs)
        del newcustom[num]
        self.document.applyOperation( document.OperationSetCustom(newcustom) )

    def moveUpEntry(self, num):
        """Move up entry."""
        if num == 0 or len(self.document.evaluate.customs) == 0:
            return
        newcustom = list(self.document.evaluate.customs)
        row = newcustom[num]
        del newcustom[num]
        newcustom.insert(num-1, row)
        self.document.applyOperation( document.OperationSetCustom(newcustom) )

    def moveDownEntry(self, num):
        """Move down entry."""
        if num >= len(self.document.evaluate.customs)-1:
            return
        newcustom = list(self.document.evaluate.customs)
        row = newcustom[num]
        del newcustom[num]
        newcustom.insert(num+1, row)
        self.document.applyOperation( document.OperationSetCustom(newcustom) )

    def setData(self, index, value, role):
        """Edit an item."""
        if index.isValid() and role == qt4.Qt.EditRole:
            col = index.column()
            row = index.row()

            if col == 0:
                ok = value in ('constant', 'function', 'import', 'colormap')
            elif col == 1:
                dtype = self.document.evaluate.customs[row][0]
                if dtype == 'constant':
                    ok = document.identifier_re.match(value) is not None
                elif dtype == 'function':
                    ok = document.function_re.match(value) is not None
                elif dtype == 'import':
                    ok = True
                elif dtype == 'colormap':
                    ok = value.strip() != ''
            elif col == 2:
                dtype = self.document.evaluate.customs[row][0]
                if dtype == 'colormap':
                    try:
                        value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        ok = False
                    else:
                        ok = True
                else:
                    ok = True
            if not ok:
                return False

            newcustom = list(self.document.evaluate.customs)
            newcustom[row][col] = value
            self.document.applyOperation(
                document.OperationSetCustom(newcustom) )

            self.dataChanged.emit(index, index)
            return True
        return False

class ComboTypeDeligate(qt4.QItemDelegate):
    """This class is for choosing between the custom
    types in a combo box in a model view."""

    def createEditor(self, parent, option, index):
        """Create combobox for editing type."""
        w = qt4.QComboBox(parent)
        w.addItems(['constant', 'function', 'import', 'colormap'])
        w.setFocusPolicy( qt4.Qt.StrongFocus )
        return w

    def setEditorData(self, editor, index):
        """Update data in editor."""
        i = editor.findText( index.data() )
        editor.setCurrentIndex(i)

    def setModelData(self, editor, model, index):
        """Update data in model."""
        model.setData(index, editor.currentText(),
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
        self.addButton.clicked.connect(self.slotAdd)
        self.removeButton.clicked.connect(self.slotRemove)
        self.upButton.clicked.connect(self.slotUp)
        self.downButton.clicked.connect(self.slotDown)

        self.saveButton.clicked.connect(self.slotSave)
        self.loadButton.clicked.connect(self.slotLoad)

        # recent button shows list of recently used files for loading
        self.recentButton.filechosen.connect(self.loadFile)
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
        filename = self.parent().fileSaveDialog(
            [_('Veusz document (*.vsz)')], _('Save custom definitions'))
        if filename:
            try:
                f = open(filename, 'w')
                self.document.evaluate.saveCustomFile(f)
                f.close()
                self.recentButton.addFile(filename)
            except EnvironmentError as e:
                qt4.QMessageBox.critical(
                    self, _("Error - Veusz"),
                    _("Unable to save '%s'\n\n%s") % (
                        filename, cstrerror(e)))

    def slotLoad(self):
        """Load entries."""

        filename = self.parent().fileOpenDialog(
            [_('Veusz document (*.vsz)')], _('Load custom definitions'))
        if filename:
            try:
                self.loadFile(filename)
            except EnvironmentError as e:
                qt4.QMessageBox.critical(
                    self, _("Error - Veusz"),
                    _("Unable to load '%s'\n\n%s") % (
                            filename, cstrerror(e)))
            else:
                # add to recent file list
                self.recentButton.addFile(filename)
