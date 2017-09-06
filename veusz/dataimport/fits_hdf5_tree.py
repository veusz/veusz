#    Copyright (C) 2017 Jeremy S. Sanders
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

from __future__ import division, print_function, absolute_import

from .. import qtall as qt
from . import fits_hdf5_helpers

def _(text, disambiguation=None, context="ImportTree"):
    return qt.QCoreApplication.translate(context, text, disambiguation)

class GenericTreeModel(qt.QAbstractItemModel):
    """A generic tree model, operating on Node objects."""

    def __init__(self, parent, root, columnheads):
        qt.QAbstractItemModel.__init__(self, parent)
        self.rootnode = root
        self.columnheads = columnheads

    def index(self, row, column, parent):
        if not parent.isValid():
            return self.createIndex(row, column, self.rootnode)
        parentnode = parent.internalPointer()
        return self.createIndex(row, column, parentnode.children[row])

    def parent(self, index):
        if not index.isValid():
            return qt.QModelIndex()
        node = index.internalPointer()
        if node.parent is None:
            return qt.QModelIndex()
        else:
            parent = node.parent
            if parent.parent is None:
                row = 0
            else:
                # find row of parent's parent for parent
                row = parent.parent.children.index(parent)
            return self.createIndex(row, 0, parent)

    def rowCount(self, parent):
        if not parent.isValid():
            return 1
        return len(parent.internalPointer().children)

    def data(self, index, role):
        if not index.isValid():
            return None
        node = index.internalPointer()
        return node.data(index.column(), role)

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        node = index.internalPointer()
        return node.setData(self, index, value, role)

    def flags(self, index):
        defflags = qt.QAbstractItemModel.flags(self, index)
        if not index.isValid():
            return defflags
        else:
            node = index.internalPointer()
            return node.flags(index.column(), defflags)

    def columnCount(self, parent):
        return len(self.columnheads)

    def headerData(self, section, orientation, role):
        if ( orientation == qt.Qt.Horizontal and
             role == qt.Qt.DisplayRole and
             section < len(self.columnheads) ):
            return self.columnheads[section]
        return None

class Node(object):
    """Generic Node used by tree model."""
    def __init__(self, parent):
        self.parent = parent
        self.children = []

    def data(self, column, role):
        return None

    def flags(self, column, defflags):
        return defflags

    def setData(self, model, index, value, role):
        return False

class ErrorNode(Node):
    """Node for showing error messages."""

    def __init__(self, parent, name):
        Node.__init__(self, parent)
        self.name = name

    def data(self, column, role):
        if column == 0 and role == qt.Qt.DisplayRole:
            return self.name
        return None

class ImportNameDeligate(qt.QItemDelegate):
    """This class is for choosing the import name."""

    def __init__(self, parent, datanodes):
        qt.QItemDelegate.__init__(self, parent)
        self.datanodes = datanodes

    def createEditor(self, parent, option, index):
        """Create combobox for editing type."""
        w = qt.QComboBox(parent)
        w.setEditable(True)

        node = index.internalPointer()
        out = []
        for dn in (n for n in self.datanodes if n.toimport):
            name = dn.name
            out.append( (name, '') )
            if ( len(dn.shape) == 1 and node is not dn and
                 dn.shape == node.shape and
                 node.numeric and dn.numeric and
                 name[-4:] != ' (+)' and name[-4:] != ' (-)' and
                 name[-5:] != ' (+-)' ):
                # add error bars for other datasets
                out.append(
                    ('%s (+-)' % name,
                     _("Import as symmetric error bar for '%s'") % name) )
                out.append(
                    ('%s (+)' % name,
                     _("Import as positive error bar for '%s'") % name) )
                out.append(
                    ('%s (-)' % name,
                     _("Import as negative error bar for '%s'") % name) )
        out.sort()

        # remove duplicates
        last = None
        i = 0
        while i < len(out):
            if out[i] == last:
                del out[i]
            else:
                last = out[i]
                i += 1

        w.addItems([x[0] for x in out])
        for v, item in enumerate(out):
            w.setItemData(v, item[1], qt.Qt.ToolTipRole)
        return w

    def setEditorData(self, editor, index):
        """Update data in editor."""
        text = index.data(qt.Qt.EditRole)

        i = editor.findText(text)
        if i != -1:
            editor.setCurrentIndex(i)
        else:
            editor.setEditText(text)

    def setModelData(self, editor, model, index):
        """Update data in model."""
        model.setData(index, editor.currentText(),
                      qt.Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        """Update editor geometry."""
        editor.setGeometry(option.rect)

# name for the columns
_ColName = 0
_ColDataType = 1
_ColShape = 2
_ColToImport = 3
_ColImportName = 4
_ColSlice = 5

class FileNode(Node):
    def grpImport(self):
        """Is this disabled because of a group import?"""
        p = self.parent
        while p is not None:
            if p.grpimport:
                return True
            p = p.parent
        return False

    def _updateRow(self, model, index):
        """This is messy - inform view that this row has changed."""
        par = model.parent(index)
        row = index.row()
        idx1 = model.index(row, 0, par)
        idx2 = model.index(row, model.columnCount(index)-1, par)
        model.dataChanged.emit(idx1, idx2)

class FileGroupNode(FileNode):
    def __init__(self, parent, name, dispname):
        Node.__init__(self, parent)
        self.fullname = name
        self.name = dispname
        self.grpimport = False

    def data(self, column, role):
        if column == _ColName and role == qt.Qt.DisplayRole:
            return self.name
        elif role == qt.Qt.CheckStateRole and column == _ColToImport:
            return ( qt.Qt.Checked
                     if self.grpimport or self.grpImport()
                     else qt.Qt.Unchecked )

        elif role == qt.Qt.ToolTipRole:
            if column == _ColToImport:
                return _("Check to import all datasets under this group")
            elif column == _ColName:
                return self.fullname

        return None

    def setData(self, model, index, value, role):
        """Enable selection of group for importing. This prevents
        importing child items individually."""
        column = index.column()
        if column == _ColToImport and role == qt.Qt.CheckStateRole:
            # import check has changed
            self.grpimport = value == qt.Qt.Checked

            # disable importing of child nodes
            def recursivedisable(node):
                if isinstance(node, FileDataNode):
                    node.toimport = False
                else:
                    if node is not self:
                        node.grpimport = False
                    for c in node.children:
                        recursivedisable(c)
            if self.grpimport:
                recursivedisable(self)

            self._updateRow(model, index)
            return True

        return False

    def flags(self, column, defflags):
        if self.grpImport():
            defflags &= ~qt.Qt.ItemIsEnabled
            return defflags

        if column == _ColToImport:
            defflags |= qt.Qt.ItemIsUserCheckable
        return defflags

class EmptyDataNode(FileNode):
    """Empty dataset."""
    def __init__(self, parent, dsname, dispname):
        Node.__init__(self, parent)
        self.name = dispname
        self.fullname = dsname

    def data(self, column, role):
        """Return data for column"""

        if role == qt.Qt.DisplayRole:
            if column == _ColName:
                return self.name
            elif column == _ColShape:
                return _('Empty')
        elif role == qt.Qt.ToolTipRole:
            if column == _ColName:
                return self.fullname

class FileDataNode(FileNode):
    """Represent an File dataset."""

    def __init__(self, parent, dsname, dsattrs, dtype, rawdtype, shape, dispname):

        """Node arguments:
        parent: parent node
        dsname: dataset name (used as tooltip for node)
        dsattrs: attributes of dataset
        dtype: dtype of datatype ('numeric', 'text', 'invalid')
        rawdtype: internal dtype for tooltip
        shape: shape of dataset
        dispname: display name (text shown for node)
        """

        Node.__init__(self, parent)
        self.name = dispname
        self.fullname = dsname
        self.rawdatatype = str(rawdtype)
        self.shape = shape
        self.toimport = False
        self.importname = ""
        self.slice = None
        self.options = {}
        self.attrs = dsattrs

        self.text = self.numeric = False
        if dtype == 'numeric':
            self.datatype = _('Numeric')
            self.numeric = True
            self.datatypevalid = True
        elif dtype == 'text':
            self.datatype = _('Text')
            self.text = True
            self.datatypevalid = True
        else:
            self.datatype = _('Unsupported')
            self.datatypevalid = False

    def getDims(self):
        """Return dimensions after slice."""
        shape = list(self.shape)

        slice = None
        if "vsz_slice" in self.attrs:
            slice = fits_hdf5_helpers.convertTextToSlice(
                fits_hdf5_helpers.convertFromBytes(self.attrs["vsz_slice"]),
                len(self.shape))
        if self.slice:
            slice = self.slice

        if slice and slice != -1:
            shapei = 0
            for s in slice:
                if isinstance(s, int):
                    del shape[shapei]
                else:
                    shapei += 1
        return len(shape)

    def dimsOkForImport(self):
        """Are dimensions ok to import?
        Need to count dimensions where slice is not fixed
        """
        return self.getDims() >= 1

    def data(self, column, role):
        """Return data for column"""
        if role in (qt.Qt.DisplayRole, qt.Qt.EditRole):
            if column == _ColName:
                return self.name
            elif column == _ColDataType:
                return self.datatype
            elif column == _ColShape:
                return u'\u00d7'.join([str(x) for x in self.shape])

            elif column == _ColImportName:
                if role == qt.Qt.EditRole and not self.importname:
                    return self.name
                else:
                    if self.importname:
                        return self.importname
                    elif "vsz_name" in self.attrs:
                        # needs to be converted to unicode to work!
                        return fits_hdf5_helpers.convertFromBytes(
                            self.attrs["vsz_name"])
                    return None

            elif column == _ColSlice:
                if self.slice:
                    return fits_hdf5_helpers.convertSliceToText(self.slice)
                elif "vsz_slice" in self.attrs:
                    return fits_hdf5_helpers.convertFromBytes(
                        self.attrs["vsz_slice"])
                return None

        elif role == qt.Qt.ToolTipRole:
            if column == _ColName:
                return self.fullname
            elif column == _ColDataType:
                return self.rawdatatype
            elif column == _ColToImport and not self.grpImport():
                return _(
                    'Check to import this dataset')
            elif column == _ColImportName and not self.grpImport():
                return _(
                    'Name to assign after import.\n'
                    'Special suffixes (+), (-) and (+-) can be used to\n'
                    'assign error bars to datasets with the same name.')
            elif column == _ColSlice:
                return _(
                    'Slice data to create a subset to import.\n'
                    'This should be ranges for each dimension\n'
                    'separated by commas.\n'
                    'Ranges can be empty (:), half (:10),\n'
                    ' full (4:10), with steps (1:10:2)\n'
                    ' or negative steps (::-1).\n'
                    'Example syntax: 2:20\n'
                    '   :10,:,2:20\n'
                    '   1:10:5,::5')

        elif role == qt.Qt.CheckStateRole and column == _ColToImport:
            if ( (self.toimport or self.grpImport()) and
                 self.dimsOkForImport() ):
                return qt.Qt.Checked
            return qt.Qt.Unchecked
        return None

    def setData(self, model, index, value, role):
        # enable selection of dataset for importing
        column = index.column()
        if column == _ColToImport and role == qt.Qt.CheckStateRole:
            # import check has changed
            self.toimport = value == qt.Qt.Checked
            if not self.toimport:
                self.importname = ''

            self._updateRow(model, index)
            return True

        elif column == _ColImportName and (self.toimport or self.grpImport()):
            # update name if changed
            self.importname = value
            return True

        elif column == _ColSlice:
            slice = fits_hdf5_helpers.convertTextToSlice(value, len(self.shape))
            if slice != -1:
                self.slice = slice
                self._updateRow(model, index)
                return True

        return False

    def flags(self, column, defflags):

        if ( column == _ColToImport and self.datatypevalid and
             not self.grpImport() and self.dimsOkForImport() ):
            # allow import column to be clicked
            defflags |= qt.Qt.ItemIsUserCheckable
        elif ( column == _ColImportName and (self.toimport or self.grpImport())
               and self.dimsOkForImport() ):
            defflags |= qt.Qt.ItemIsEditable
        elif column == _ColSlice and self.datatypevalid:
            # allow name to be edited
            defflags |= qt.Qt.ItemIsEditable

        return defflags

class FileCompoundNode(FileGroupNode):
    """Node representing a table (Compound data type)."""

    def __init__(self, parent, name, dispname, shape):
        FileGroupNode.__init__(self, parent, name, dispname)
        self.shape = shape

    def data(self, column, role):
        """Return data for column"""
        if role == qt.Qt.DisplayRole:
            if column == _ColDataType:
                return _("Table")
            elif column == _ColShape:
                return u'\u00d7'.join([str(x) for x in self.shape])
        return FileGroupNode.data(self, column, role)

##############################################################################

def setupTreeView(view, rootnode, datanodes):
    """Setup view for nodes."""

    view._importnamedeligate = ImportNameDeligate(
        view, datanodes)
    view.setItemDelegateForColumn(_ColImportName, view._importnamedeligate)

    mod = GenericTreeModel(
        view, rootnode,
        [_('Name'), _('Type'), _('Size'), _('Import'),
         _('Import as'), _('Slice')])

    view.setModel(mod)
    view.expandAll()
    for c in _ColName, _ColDataType, _ColShape:
        view.resizeColumnToContents(c)
