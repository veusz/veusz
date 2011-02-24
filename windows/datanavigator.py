# -*- coding: utf-8 -*-
#    Copyright (C) 2011 Jeremy S. Sanders
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
###############################################################################

import os.path
from collections import defaultdict

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.document as document

from treemodel import TMNode, TreeModel

import veusz.dialogs.dataeditdialog as dataeditdialog

class DatasetNode(TMNode):
    """Node for a dataset."""

    def __init__(self, doc, data, parent):
        TMNode.__init__(self, data, parent)
        self.doc = doc

    def toolTip(self, column):
        try:
            ds = self.doc.data[self.data[0]]
        except KeyError:
            return qt4.QVariant()
        if column == 0:
            return qt4.QVariant( ds.description() )
        elif column == 1:
            return qt4.QVariant( ds.userPreview() )
        return qt4.QVariant()

    def dataset(self):
        """Get associated dataset."""
        try:
            return self.doc.data[self.data[0]]
        except KeyError:
            return None

    def datasetName(self):
        """Get dataset name."""
        return self.data[0]

    def cloneTo(self, newroot):
        """Make a clone of self at the root given."""
        return self.__class__(self.doc, self.data, newroot)

class FilenameNode(TMNode):
    """A special node for holding filenames of files."""

    def nodeData(self, column):
        """basename of filename for data."""
        if column == 0:
            if self.data[0] == '/':
                return qt4.QVariant('/')
            else:
                return qt4.QVariant(os.path.basename(self.data[0]))
        return qt4.QVariant()

    def toolTip(self, column):
        """Full filename for tooltip."""
        if column == 0:
            return qt4.QVariant(self.data[0])
        return qt4.QVariant()

class DatasetRelationModel(TreeModel):
    """A model to show how the datasets are related to each file."""
    def __init__(self, doc):
        TreeModel.__init__(self, ('Dataset', 'Size', 'Type') )
        self.doc = doc
        self.linkednodes = {}

        self.connect(doc, qt4.SIGNAL('sigModified'), self.refresh)

        self.mode = 'grplinked'

        self.refresh()
        
    def makeGrpTreeLinked(self):
        """Make a tree of datasets grouped by linked file."""
        linknodes = {}
        for name, ds in self.doc.data.iteritems():
            if ds.linked not in linknodes:
                if ds.linked is not None:
                    lname = ds.linked.filename
                else:
                    lname = '/'
                node = FilenameNode( (lname, ''), self.root )
                linknodes[ds.linked] = node
            child = DatasetNode( self.doc, (name, ds.userSize(), ds.dstype), None )
            linknodes[ds.linked].insertChildSorted( child )
        
        # make tree
        tree = TMNode( self.root.data, None )
        for name in sorted(linknodes.keys()):
            tree.insertChildSorted( linknodes[name] )

        return tree

    def flags(self, idx):
        """Return model flags for index."""
        f = TreeModel.flags(self, idx)
        # allow dataset names to be edited
        if idx.isValid() and isinstance(idx.internalPointer(), DatasetNode):
            f |= qt4.Qt.ItemIsEditable
        return f

    def setData(self, idx, data, role):
        """Rename dataset."""
        dsnode = idx.internalPointer()
        newname = unicode(data.toString())
        self.doc.applyOperation(
            document.OperationDatasetRename(dsnode.data[0], newname))
        self.emit( qt4.SIGNAL('dataChanged(const QModelIndex &, const QModelIndex &)'),
                   idx, idx)
        return True

    def refresh(self):
        """Update tree of datasets when document changes."""

        if self.mode == 'grplinked':
            tree = self.makeGrpTreeLinked()
        else:
            raise RuntimeError, "Invalid mode"

        self.syncTree(tree)
        
class DatasetsNavigator(qt4.QTreeView):
    """List of currently opened Misura4 Tests and reference to datasets names"""
    def __init__(self, doc, mainwin, parent=None):
        """Initialise the dataset viewer."""
        qt4.QTreeView.__init__(self, parent)
        self.doc = doc
        self.mainwindow = mainwin
        self.setModel(DatasetRelationModel(doc))
        self.selection = qt4.QItemSelectionModel(self.model())
        self.setSelectionBehavior(qt4.QTreeView.SelectItems)
        self.setUniformRowHeights(True)
        self.setContextMenuPolicy(qt4.Qt.CustomContextMenu)
        self.connect(self, qt4.SIGNAL('customContextMenuRequested(QPoint)'),
                     self.showContextMenu)

    def showContextMenu(self, pt):
        """Context menu for nodes."""
        idx = self.currentIndex()
        if not idx.isValid():
            return

        node = idx.internalPointer()
        menu = None
        if isinstance(node, DatasetNode):
            menu = self.datasetContextMenu(node, pt)
        if menu is not None:
            menu.exec_(self.mapToGlobal(pt))

    def datasetContextMenu(self, dsnode, pt):
        """Return context menu for datasets."""
        dataset = dsnode.dataset()
        dsname = dsnode.datasetName()

        def _edit():
            """Open up dialog box to recreate dataset."""
            dataeditdialog.recreate_register[type(dataset)](
                self.mainwindow, self.doc, dataset, dsname)
        def _edit_data():
            """Open up data edit dialog."""
            dialog = self.mainwindow.slotDataEdit(editdataset=dsname)
        def _delete():
            """Simply delete dataset."""
            self.doc.applyOperation(document.OperationDatasetDelete(dsname))
        def _unlink_file():
            """Unlink dataset from file."""
            self.doc.applyOperation(document.OperationDatasetUnlinkFile(dsname))
        def _unlink_relation():
            """Unlink dataset from relation."""
            self.doc.applyOperation(document.OperationDatasetUnlinkRelation(dsname))

        menu = qt4.QMenu()
        if type(dataset) in dataeditdialog.recreate_register:
            menu.addAction("Edit", _edit)
        else:
            menu.addAction("Edit data", _edit_data)

        menu.addAction("Delete", _delete)
        if dataset.canUnlink():
            if dataset.linked:
                menu.addAction("Unlink file", _unlink_file)
            else:
                menu.addAction("Unlink relation", _unlink_relation)

        useasmenu = menu.addMenu('Use as')
        if dataset is not None:
            self.getMenuUseAs(useasmenu, dataset)
        return menu

    def getMenuUseAs(self, menu, dataset):
        """Build up menu of widget settings to use dataset in."""

        def addifdatasetsetting(path, setn):
            def _setdataset():
                self.doc.applyOperation(
                    document.OperationSettingSet(
                        path, self.doc.datasetName(dataset)) )

            if ( isinstance(setn, setting.Dataset) and
                 setn.dimensions == dataset.dimensions and
                 setn.datatype == dataset.datatype and
                 path[:12] != '/StyleSheet/' ):
                menu.addAction(path, _setdataset)

        self.doc.walkNodes(addifdatasetsetting, nodetypes=('setting',))
        
class DataNavigatorWindow(qt4.QDockWidget):
    def __init__(self, thedocument, mainwin, *args):
        qt4.QDockWidget.__init__(self, *args)
        self.setWindowTitle("Data - Veusz")
        self.setObjectName("veuszdatawindow")

        self.nav = DatasetsNavigator(thedocument, mainwin, parent=self)
        self.setWidget(self.nav)
