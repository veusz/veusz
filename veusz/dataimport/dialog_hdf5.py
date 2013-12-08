#    Copyright (C) 2013 Jeremy S. Sanders
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

from __future__ import division, print_function

from .. import qtall as qt4
from ..dialogs import importdialog
from ..compat import crange
#from . import defn_fits

def _(text, disambiguation=None, context="Import_HDF5"):
    return qt4.QCoreApplication.translate(context, text, disambiguation)

h5py = None

class Node(object):
    def __init__(self, parent):
        self.parent = parent
        self.children = []

    def data(self, column, role):
        return None

    def flags(self, column, defflags):
        return defflags

    def setData(self, model, index, value, role):
        return False

class GenericTreeModel(qt4.QAbstractItemModel):
    """A generic tree model, operating on Node objects."""

    def __init__(self, parent, root, columnheads):
        qt4.QAbstractItemModel.__init__(self, parent)
        self.rootnode = root
        self.columnheads = columnheads

    def index(self, row, column, parent):
        if not parent.isValid():
            return self.createIndex(row, column, self.rootnode)
        parentnode = parent.internalPointer()
        return self.createIndex(row, column, parentnode.children[row])

    def parent(self, index):
        if not index.isValid():
            return qt4.QModelIndex()
        node = index.internalPointer()
        if node.parent is None:
            return qt4.QModelIndex()
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
        defflags = qt4.QAbstractItemModel.flags(self, index)
        if not index.isValid():
            return defflags
        else:
            node = index.internalPointer()
            return node.flags(index.column(), defflags)

    def columnCount(self, parent):
        return len(self.columnheads)

    def headerData(self, section, orientation, role):
        if ( orientation == qt4.Qt.Horizontal and
             role == qt4.Qt.DisplayRole and
             section < len(self.columnheads) ):
            return self.columnheads[section]
        return None

class ErrorNode(Node):
    def __init__(self, parent, name):
        Node.__init__(self, parent)
        self.name = name

    def data(self, column, role):
        if column == 0 and role == qt4.Qt.DisplayRole:
            return self.name
        return None

class HDFNode(Node):
    def __init__(self, parent, grp):
        Node.__init__(self, parent)
        self.name = grp.name.split("/")[-1]
        if self.name == '':
            self.name = '/'

    def data(self, column, role):
        if column == 0 and role == qt4.Qt.DisplayRole:
            return self.name
        return None

class HDFDataNode(Node):
    ColName = 0
    ColDataType = 1
    ColShape = 2
    ColToImport = 3
    ColImportName = 4
    ColMax = 4

    def __init__(self, parent, ds):
        Node.__init__(self, parent)
        self.name = ds.name.split("/")[-1]
        self.rawdatatype = str(ds.dtype)
        self.shape = ds.shape
        self.toimport = False
        self.importname = ''

        k = ds.dtype.kind
        self.valid = False
        if k in ('b', 'i', 'u', 'f'):
            self.datatype = _('Numeric')
            self.valid = True
        elif k in ('S', 'a'):
            self.datatype = _('Text')
            self.valid = True
        elif k == 'O' and h5py.check_dtype(vlen=ds.dtype):
            self.datatype = _('Text')
            self.valid = True
        else:
            self.datatype = _('Unsupported')

    def data(self, column, role):
        if role in (qt4.Qt.DisplayRole, qt4.Qt.EditRole):
            if column == self.ColName:
                return self.name
            elif column == self.ColDataType:
                return self.datatype
            elif column == self.ColShape:
                return u'\u00d7'.join([str(x) for x in self.shape])
            elif column == self.ColImportName:
                return self.importname

        elif role == qt4.Qt.ToolTipRole:
            if column == self.ColDataType:
                return self.rawdatatype
            elif column == self.ColToImport:
                return _('Check to import this dataset')
            elif column == self.ColImportName:
                return _('Name to assign after import')

        elif role == qt4.Qt.CheckStateRole:
            if column == self.ColToImport:
                return qt4.Qt.Checked if self.toimport else qt4.Qt.Unchecked
        return None

    def setData(self, model, index, value, role):
        # enable selection of dataset for importing
        column = index.column()
        if column == self.ColToImport and role == qt4.Qt.CheckStateRole:
            # import check has changed
            self.toimport = value == qt4.Qt.Checked
            if self.toimport:
                # create default name
                self.importname = self.name
            else:
                self.importname = ''

            # this is messy - inform view that this row has changed
            par = model.parent(index)
            row = index.row()
            idx1 = model.index(row, 0, par)
            idx2 = model.index(row, self.ColMax, par)
            model.dataChanged.emit(idx1, idx2)

            return True

        elif column == self.ColImportName and self.toimport:
            # update name if changed
            self.importname = value
            return True
        return False

    def flags(self, column, defflags):
        if column == self.ColToImport and self.valid:
            # this is the to import column
            defflags |= qt4.Qt.ItemIsUserCheckable
        if column == self.ColImportName and self.toimport:
            # this is the import name column
            defflags |= qt4.Qt.ItemIsEditable
        return defflags

def constructTree(hdf5file):
    """Turn hdf5 file into a tree of nodes."""

    def addsub(parent, grp):
        """To recursively iterate over each parent."""
        for child in sorted(grp.keys()):
            try:
                hchild = grp[child]
            except KeyError:
                continue
            if isinstance(hchild, h5py.Group):
                childnode = HDFNode(parent, hchild)
                addsub(childnode, hchild)
            elif isinstance(hchild, h5py.Dataset):
                childnode = HDFDataNode(parent, hchild)
            parent.children.append(childnode)

    root = HDFNode(None, hdf5file)
    addsub(root, hdf5file)
    return root

class ImportTabHDF5(importdialog.ImportTab):
    """Tab for importing HDF5 file."""

    resource = "import_hdf5.ui"

    def showError(self, err):
        node = ErrorNode(None, err)
        model = GenericTreeModel(self, node, [''])
        self.hdftreeview.setModel(model)

    def loadUi(self):
        importdialog.ImportTab.loadUi(self)

    def doPreview(self, filename, encoding):

        global h5py
        if h5py is None:
            try:
                import h5py
            except ImportError:
                self.showError(_("Cannot load h5py module"))
                return False

        try:
            with h5py.File(filename, "r") as f:
                rootnode = constructTree(f)
        except IOError as e:
            self.showError(_("Cannot open file"))
            return False

        mod = GenericTreeModel(
            self, rootnode,
            [_('Name'), _('Type'), _('Size'), _('Import'),
             _('Import as')])
        self.hdftreeview.setModel(mod)
        self.hdftreeview.expandAll()

        return True

importdialog.registerImportTab(_('HDF&5'), ImportTabHDF5)
