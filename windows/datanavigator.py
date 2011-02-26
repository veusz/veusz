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

def treeFromList(nodelist, rootdata):
    """Construct a tree from a list of nodes."""
    tree = TMNode( rootdata, None )
    for node in nodelist:
        tree.insertChildSorted(node)
    return tree

def datasetLinkFile(ds):
    """Get a linked filename from a dataset."""
    if ds.linked is None:
        return '/'
    else:
        return ds.linked.filename

class DatasetRelationModel(TreeModel):
    """A model to show how the datasets are related to each file."""
    def __init__(self, doc, grouping='filename'):
        TreeModel.__init__(self, ('Dataset', 'Size', 'Type'))
        self.doc = doc
        self.linkednodes = {}
        self.grouping = grouping
        self.refresh()

        self.connect(doc, qt4.SIGNAL('sigModified'), self.refresh)

    def makeGrpTreeNone(self):
        """Make tree with no grouping."""
        tree = TMNode( ('Dataset', 'Size', 'Type'), None )
        for name, ds in self.doc.data.iteritems():
            child = DatasetNode( self.doc, (name, ds.userSize(), ds.dstype,
                                            ), None )
            tree.insertChildSorted(child)
        return tree
        
    def makeGrpTreeFilename(self):
        """Make a tree of datasets grouped by linked file."""
        linknodes = {}
        for name, ds in self.doc.data.iteritems():
            filename = datasetLinkFile(ds)
            if filename not in linknodes:
                linknodes[filename] = FilenameNode( (filename, ), None )
            child = DatasetNode( self.doc,
                                 (name, ds.userSize(), ds.dstype), None )
            linknodes[filename].insertChildSorted( child )

        return treeFromList( linknodes.values(), ('Dataset', 'Size', 'Type') )

    def makeGrpTreeSize(self):
        """Make a tree of datasets grouped by dataset size."""
        sizenodes = {}
        for name, ds in self.doc.data.iteritems():
            size = ds.userSize()
            if size not in sizenodes:
                sizenodes[size] = TMNode( (size, ), None )
            child = DatasetNode( self.doc,
                                 (name, ds.dstype, datasetLinkFile(ds)), None )
            sizenodes[size].insertChildSorted( child )
        
        return treeFromList( sizenodes.values(), ('Dataset', 'Type', 'Filename') )

    def makeGrpTreeType(self):
        """Make a tree of datasets grouped by dataset type."""
        typenodes = {}
        for name, ds in self.doc.data.iteritems():
            dstype = ds.dstype
            if dstype not in typenodes:
                typenodes[dstype] = TMNode( (dstype, ), None )
            child = DatasetNode( self.doc,
                                 (name, ds.userSize(), datasetLinkFile(ds)), None )
            typenodes[dstype].insertChildSorted( child )
       
        return treeFromList( typenodes.values(), ('Dataset', 'Size', 'Filename') )

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

        header = self.root.data
        tree = {
            'none': self.makeGrpTreeNone,
            'filename': self.makeGrpTreeFilename,
            'size': self.makeGrpTreeSize,
            'type': self.makeGrpTreeType,
            }[self.grouping]()

        self.syncTree(tree)

class DatasetsNavigatorTree(qt4.QTreeView):
    """List of currently opened Misura4 Tests and reference to datasets names"""
    def __init__(self, doc, mainwin, grouping, parent):
        """Initialise the dataset viewer."""
        qt4.QTreeView.__init__(self, parent)
        self.doc = doc
        self.mainwindow = mainwin
        self.model = DatasetRelationModel(doc, grouping)

        self.setModel(self.model)
        self.setSelectionBehavior(qt4.QTreeView.SelectItems)
        self.setUniformRowHeights(True)
        self.setContextMenuPolicy(qt4.Qt.CustomContextMenu)
        self.connect(self, qt4.SIGNAL('customContextMenuRequested(QPoint)'),
                     self.showContextMenu)
        self.model.refresh()
        self.expandAll()

    def changeGrouping(self, grouping):
        """Change the tree grouping behaviour."""
        self.model.grouping = grouping
        self.model.refresh()
        self.expandAll()

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

class DatasetsNavigatorWidget(qt4.QWidget):
    """Widget which shows the document's datasets."""

    # how datasets can be grouped
    grpnames = ("none", "filename", "type", "size")
    grpentries = {
        "none": "None",
        "filename": "Filename",
        "type": "Type", 
        "size": "Size"
        }

    def __init__(self, thedocument, mainwin, parent):
        """Initialise widget:
        thedocument: document to show
        mainwin: main window of application
        parent: parent of widget."""

        qt4.QWidget.__init__(self, parent)
        self.layout = qt4.QVBoxLayout()
        self.setLayout(self.layout)

        # options for navigator are in this layout
        self.optslayout = qt4.QHBoxLayout()

        # grouping options - use a 
        self.grpbutton = qt4.QPushButton("Group")
        self.grpmenu = qt4.QMenu()
        self.grouping = "filename"
        self.grpact = qt4.QActionGroup(self)
        self.grpact.setExclusive(True)
        for name in self.grpnames:
            a = self.grpmenu.addAction(self.grpentries[name])
            a.grpname = name
            a.setCheckable(True)
            if name == self.grouping:
                a.setChecked(True)
            self.grpact.addAction(a)
        self.connect(self.grpact, qt4.SIGNAL("triggered(QAction*)"), self.slotGrpChanged)
        self.grpbutton.setMenu(self.grpmenu)
        self.optslayout.addWidget(self.grpbutton)

        # filtering
        self.optslayout.addWidget(qt4.QLabel("Filter"))
        self.filteredit = qt4.QLineEdit()
        self.optslayout.addWidget(self.filteredit)

        self.layout.addLayout(self.optslayout)

        # the actual widget tree
        self.navtree = DatasetsNavigatorTree(thedocument, mainwin, self.grouping, None)
        self.layout.addWidget(self.navtree)

    def slotGrpChanged(self, action):
        """Grouping changed by user."""
        self.navtree.changeGrouping(action.grpname)

class DataNavigatorWindow(qt4.QDockWidget):
    def __init__(self, thedocument, mainwin, *args):
        qt4.QDockWidget.__init__(self, *args)
        self.setWindowTitle("Data - Veusz")
        self.setObjectName("veuszdatawindow")

        self.nav = DatasetsNavigatorWidget(thedocument, mainwin, self)
        self.setWidget(self.nav)
