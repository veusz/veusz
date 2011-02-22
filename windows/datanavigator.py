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

class DatasetNode(TMNode):
    """Node for a dataset."""

    def __init__(self, doc, data, parent):
        TMNode.__init__(self, data, parent)
        self.doc = doc

    def getTooltip(self, column):
        try:
            ds = self.doc.data[self.data[0]]
        except KeyError:
            return qt4.QVariant()
        if column == 0:
            return qt4.QVariant( ds.description() )
        elif column == 1:
            return qt4.QVariant( ds.userPreview() )
        return qt4.QVariant()

    def cloneTo(self, newroot):
        """Make a clone of self at the root given."""
        return self.__class__(self.doc, self.data, newroot)

class FilenameNode(TMNode):
    """A special node for holding filenames of files."""

    def getData(self, column):
        """basename of filename for data."""
        if column == 0:
            if self.data[0] == '/':
                return qt4.QVariant('/')
            else:
                return qt4.QVariant(os.path.basename(self.data[0]))
        return qt4.QVariant()

    def getTooltip(self, column):
        """Full filename for tooltip."""
        if column == 0:
            return qt4.QVariant(self.data[0])
        return qt4.QVariant()

class DatasetRelationModel(TreeModel):
    """A model to show how the datasets are related to each file."""
    def __init__(self, doc):
        TreeModel.__init__(self, ('Dataset', 'size') )
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
            child = DatasetNode( self.doc, (name, ds.userSize()), None )
            linknodes[ds.linked].insertChildSorted( child )
        
        # make tree
        tree = TMNode( self.root.data, None )
        for name in sorted(linknodes.keys()):
            tree.insertChildSorted( linknodes[name] )

        return tree

    def refresh(self):
        """Update tree of datasets when document changes."""

        if self.mode == 'grplinked':
            tree = self.makeGrpTreeLinked()
        else:
            raise RuntimeError, "Invalid mode"

        self.syncTree(tree)
        
class DatasetsNavigator(qt4.QTreeView):
    """List of currently opened Misura4 Tests and reference to datasets names"""
    def __init__(self, doc, parent=None):
        qt4.QTreeView.__init__(self, parent)
        self.doc = doc
        self.setModel(DatasetRelationModel(doc))
        self.selection = qt4.QItemSelectionModel(self.model())
        self.setSelectionBehavior(qt4.QTreeView.SelectItems)
        self.setUniformRowHeights(True)
        self.setContextMenuPolicy(qt4.Qt.CustomContextMenu)
        self.connect(self, qt4.SIGNAL('customContextMenuRequested(QPoint)'),
                     self.showContextMenu)

    def showContextMenu(self, pt):
        return
        node=self.model().data(self.currentIndex(), role='data')
        menu = qt4.QMenu(self)
        if node in self.model().heads:
            menu.addAction('Table View', self.viewFile)
            menu.addAction('Reload', self.reload)
            menu.addAction('Close', self.closeFile)
        else:
            menu.addAction('Edit', self.editData)
            menu.addAction('Delete', self.deleteData)
            useasmenu = menu.addMenu('Use as')
            self.getMenuUseAs(useasmenu, node)
        menu.popup(self.mapToGlobal(pt))

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

    #def refresh(self):
    #    self.model().refresh()

    #def viewFile(self):
    #    node=self.model().data(self.currentIndex(), role='data')
    #    # Tabular view of data?

    #def reload(self):
    #    node=self.model().data(self.currentIndex(), role='data')
    #    node.reloadLinks()

    #def closeFile(self):
    #    node=self.model().data(self.currentIndex(), role='data')
    #    for ds in self.model().relations[node]:
    #        self.doc.deleteDataset(self.doc.datasetName(ds))
    #    self.refresh()

    #def editData(self):
    #    pass

    # Redirect to dataset edit window?
    #def deleteData(self):
    #    node=self.model().data(self.currentIndex(), role='data')
    #    self.doc.deleteDataset(self.doc.datasetName(ds))
        
class DataNavigatorWindow(qt4.QDockWidget):
    def __init__(self, thedocument, *args):
        qt4.QDockWidget.__init__(self, *args)
        self.setWindowTitle("Data - Veusz")
        self.setObjectName("veuszdatawindow")

        self.nav = DatasetsNavigator(thedocument, parent=self)
        self.setWidget(self.nav)
