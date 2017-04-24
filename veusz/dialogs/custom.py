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
import copy

from ..compat import cstrerror
from .. import qtall as qt4
from .. import document
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="CustomDialog"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class CustomItemModel(qt4.QAbstractTableModel):
    """A model for editing custom items."""

    # headers for type of widget
    headers = {
        'definition': (_('Name'), _('Definition')),
        'import': (_('Module'), _('Symbol list')),
        'color': (_('Name'), _('Definition')),
        'colormap': (_('Name'), _('Definition')),
    }

    # tooltips for columns
    tooltips = {
        'definition': (
            _('Name for constant, or function name and arguments, e.g. "f(x,y)"'),
            _('Python expression defining constant or function, e.g. "x+y"')),
        'import': (
            _('Module to import symbols from, e.g. "scipy.special"'),
            _('Comma-separated list of symbols to import or "*" to import everything')),
        'color': (
            _('Name of color'),
            _('Definition of color ("#RRGGBB", "#RRGGBBAA" or "red")')),
        'colormap': (
            _('Name of colormap'),
            _('Definition of colormap, defined as lists of RGB tuples, e.g. "((0,0,0),(255,255,255))"')),
    }

    def __init__(self, parent, doc, ctype):
        """
        ctype is 'definition', 'import', 'color' or 'colormap'
        """

        qt4.QAbstractTableModel.__init__(self, parent)
        self.doc = doc
        self.ctype = ctype
        self.attr = document.OperationSetCustom.type_to_attr[ctype]

        # connect notification of document change
        doc.signalModified.connect(self.doUpdate)

        # do not inform qt model has changed on document change
        self.moddocupignore = False

    def _getCustoms(self):
        return getattr(self.doc.evaluate, self.attr)

    def _getCustomsCopy(self):
        return copy.deepcopy(self._getCustoms())

    def rowCount(self, parent):
        return 0 if parent.isValid() else len(self._getCustoms())+1

    def columnCount(self, parent):
        return 0 if parent.isValid() else 2

    def data(self, index, role):
        """Lookup data in document.evaluate.customs list."""
        if role in (qt4.Qt.DisplayRole, qt4.Qt.EditRole):
            try:
                defn = self._getCustoms()[index.row()]
            except IndexError:
                # empty row beyond end
                return ''
            col = index.column()
            if self.ctype=='colormap' and col==1:
                return repr(defn[col])
            else:
                return defn[col]
        elif role == qt4.Qt.ToolTipRole:
            # tooltip on row for new entries on last row
            if index.row() == len(self._getCustoms()):
                return self.tooltips[self.ctype][index.column()]

        return None

    def flags(self, index):
        """Items are editable"""
        return (
            qt4.Qt.ItemIsSelectable | qt4.Qt.ItemIsEditable |
            qt4.Qt.ItemIsEnabled )

    def headerData(self, section, orientation, role):
        """Return the headers at the top of the view."""
        if role == qt4.Qt.DisplayRole:
            if orientation == qt4.Qt.Horizontal:
                # columns defined in headers
                return self.headers[self.ctype][section]
            else:
                # number rows
                return str(section+1)
        return None

    def doUpdate(self):
        """Document changed."""
        if not self.moddocupignore:
            self.layoutChanged.emit()

    def validateName(self, val):
        if self.ctype == 'import':
            return document.module_re.match(val)
        elif self.ctype == 'definition':
            return (
                document.identifier_re.match(val) or
                document.function_re.match(val))
        else:
            # color or colormap
            return val.strip() != ''

    def validateDefn(self, value):
        if self.ctype == 'colormap':
            try:
                tmp = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return False
        return value.strip() != ''

    def setData(self, index, value, role):
        """Edit an item."""
        if index.isValid() and role == qt4.Qt.EditRole:
            col = index.column()
            row = index.row()

            if col == 0:
                ok = self.validateName(value)
            elif col == 1:
                ok = self.validateDefn(value)
            if not ok:
                return False

            # extend if necessary
            newcustom = self._getCustomsCopy()
            while len(newcustom) < row+1:
                if self.ctype == 'colormap':
                    newcustom.append(['', ((0,0,0),(255,255,255))])
                else:
                    newcustom.append(['', ''])

            if self.ctype=='colormap' and col==1:
                newcustom[row][col] = eval(value)
            else:
                newcustom[row][col] = value

            self.doc.applyOperation(
                document.OperationSetCustom(self.ctype, newcustom) )

            self.dataChanged.emit(index, index)
            return True
        return False

    def deleteEntry(self, num):
        """Delete row num. True if success."""
        newcustoms = self._getCustomsCopy()
        if num >= len(newcustoms):
            return False
        self.beginRemoveRows(qt4.QModelIndex(), num, num)
        del newcustoms[num]
        self.moddocupignore = True
        self.doc.applyOperation(
            document.OperationSetCustom(self.ctype, newcustoms))
        self.moddocupignore = False
        self.endRemoveRows()
        return True

    def moveUpEntry(self, num):
        """Move up entry."""
        newcustoms = self._getCustomsCopy()
        if num == 0 or num >= len(newcustoms):
            return False
        row = newcustoms[num]
        del newcustoms[num]
        newcustoms.insert(num-1, row)
        self.doc.applyOperation(
            document.OperationSetCustom(self.ctype, newcustoms))
        return True

    def moveDownEntry(self, num):
        """Move down entry."""
        newcustoms = self._getCustomsCopy()
        if num >= len(newcustoms)-1:
            return False
        row = newcustoms[num]
        del newcustoms[num]
        newcustoms.insert(num+1, row)
        self.doc.applyOperation(
            document.OperationSetCustom(self.ctype, newcustoms))
        return True

class CustomDialog(VeuszDialog):
    """A dialog to create or edit custom constant and function
    definitions."""

    def __init__(self, parent, document):
        VeuszDialog.__init__(self, parent, 'custom.ui')
        self.document = document

        # model/view
        self.defnModel = CustomItemModel(self, document, 'definition')
        self.defnView.setModel(self.defnModel)
        self.importModel = CustomItemModel(self, document, 'import')
        self.importView.setModel(self.importModel)
        self.colorModel = CustomItemModel(self, document, 'color')
        self.colorView.setModel(self.colorModel)
        self.colormapModel = CustomItemModel(self, document, 'colormap')
        self.colormapView.setModel(self.colormapModel)

        # buttons
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

    def getTabViewAndModel(self):
        """Get view and model for currently selected tab."""
        return {
            0: (self.defnView, self.defnModel),
            1: (self.importView, self.importModel),
            2: (self.colorView, self.colorModel),
            3: (self.colormapView, self.colormapModel)
            }[self.viewsTab.currentIndex()]

    def slotRemove(self):
        """Remove an entry."""
        view, model = self.getTabViewAndModel()
        selected = view.selectedIndexes()
        if selected:
            model.deleteEntry(selected[0].row())

    def slotUp(self):
        """Move item up list."""
        view, model = self.getTabViewAndModel()
        selected = view.selectedIndexes()
        if selected:
            row = selected[0].row()
            if model.moveUpEntry(row):
                idx = model.index(row-1, selected[0].column())
                view.setCurrentIndex(idx)

    def slotDown(self):
        """Move item down list."""
        view, model = self.getTabViewAndModel()
        selected = view.selectedIndexes()
        if selected:
            row = selected[0].row()
            if model.moveDownEntry(row):
                idx = model.index(row+1, selected[0].column())
                view.setCurrentIndex(idx)

    def slotSave(self):
        """Save entries."""
        filename = self.parent().fileSaveDialog(
            [_('Veusz document (*.vsz)')], _('Save custom definitions'))
        if filename:
            try:
                with open(filename, 'w') as f:
                    self.document.evaluate.saveCustomFile(f)
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
