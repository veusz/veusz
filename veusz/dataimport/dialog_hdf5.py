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

    def columnCount(self, parent):
        return len(self.columnheads)

    def headerData(self, section, orientation, role):
        if ( orientation == qt4.Qt.Horizontal and
             role == qt4.Qt.DisplayRole and
             section < len(self.columnheads) ):
            return self.columnheads[section]
        return None

class HDFNode(Node):
    def __init__(self, parent, text):
        Node.__init__(self, parent)
        self.text = text

    def data(self, column, role):
        if column == 0 and role == qt4.Qt.DisplayRole:
            return self.text
        return None

def constructTree(hdf5file):
    """Turn hdf5 file into a tree of nodes."""

    def addsub(parent, grp):
        """To recursively iterate over each parent."""
        for child in sorted(grp.keys()):
            try:
                cgrp = grp[child]
            except KeyError:
                continue
            childnode = HDFNode(parent, cgrp.name.split("/")[-1])
            if isinstance(cgrp, h5py.Group):
                addsub(childnode, cgrp)
            parent.children.append(childnode)

    root = HDFNode(None, "/")
    addsub(root, hdf5file)
    return root

class ImportTabHDF5(importdialog.ImportTab):
    """Tab for importing HDF5 file."""

    resource = "import_hdf5.ui"

    def showError(self, err):
        node = HDFNode(None, err)
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

        mod = GenericTreeModel(self, rootnode, [_('Name')])
        self.hdftreeview.setModel(mod)
        self.hdftreeview.expandAll()

        return True

importdialog.registerImportTab(_('HDF&5'), ImportTabHDF5)
