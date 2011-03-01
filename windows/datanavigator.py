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
import numpy as N

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.document as document
import veusz.utils as utils

import veusz.dialogs.dataeditdialog as dataeditdialog
from veusz.qtwidgets.lineeditwithclear import LineEditWithClear

from treemodel import TMNode, TreeModel

def datasetLinkFile(ds):
    """Get a linked filename from a dataset."""
    if ds.linked is None:
        return '/'
    else:
        return ds.linked.filename

def pixmapAsHtml(pix):
    """Get QPixmap as html image text."""
    ba = qt4.QByteArray()
    buf = qt4.QBuffer(ba)
    buf.open(qt4.QIODevice.WriteOnly)
    pix.toImage().save(buf, "PNG")
    b64 = str(buf.data().toBase64())
    return '<img src="data:image/png;base64,%s">' % b64

class DatasetNode(TMNode):
    """Node for a dataset."""

    def __init__(self, doc, dsname, cols, parent):
        ds = doc.data[dsname]
        data = []
        assert cols[0] == 'name'
        for c in cols:
            if c == 'name':
                data.append( dsname )
            elif c == 'size':
                data.append( ds.userSize() )
            elif c == 'type':
                data.append( ds.dstype )
            elif c == 'linkfile':
                data.append( os.path.basename(datasetLinkFile(ds)) )

        TMNode.__init__(self, tuple(data), parent)
        self.doc = doc
        self.cols = cols

    def getPreviewPixmap(self, ds):
        """Get a preview pixmap for a dataset."""
        size = (140, 70)
        if ds.dimensions != 1:
            return None

        pixmap = qt4.QPixmap(*size)
        pixmap.fill(qt4.Qt.transparent)
        p = qt4.QPainter(pixmap)
        p.setRenderHint(qt4.QPainter.Antialiasing)

        # calculate data points
        try:
            if len(ds.data) < size[1]:
                y = ds.data
            else:
                intvl = len(ds.data)/size[1]+1
                y = ds.data[::intvl]
            x = N.arange(len(y))

            # plot data points on image
            minval, maxval = N.nanmin(y), N.nanmax(y)
            y = (y-minval) / (maxval-minval) * size[1]
            finite = N.isfinite(y)
            x, y = x[finite], y[finite]
            x = x * (1./len(x)) * size[0]

            poly = qt4.QPolygonF()
            utils.addNumpyToPolygonF(poly, x, size[1]-y)
            p.setPen( qt4.QPen(qt4.Qt.blue) )
            p.drawPolyline(poly)

            # draw x axis if span 0
            p.setPen( qt4.QPen(qt4.Qt.black) )
            if minval <= 0 and maxval > 0:
                y0 = size[1] - (0-minval)/(maxval-minval)*size[1]
                p.drawLine(x[0], y0, x[-1], y0)
            else:
                p.drawLine(x[0], size[1], x[-1], size[1])
            p.drawLine(x[0], 0, x[0], size[1])

        except (ValueError, ZeroDivisionError):
            # zero sized array after filtering, so return None
            p.end()
            return None

        p.end()
        return pixmap

    def toolTip(self, column):
        """Return tooltip for column."""
        try:
            ds = self.doc.data[self.data[0]]
        except KeyError:
            return qt4.QVariant()

        c = self.cols[column]
        if c == 'name':
            return qt4.QVariant(ds.description())
        elif c == 'size' or c == 'type':
            text = ds.userPreview()
            # add preview of dataset if possible
            pix = self.getPreviewPixmap(ds)
            if pix:
                text = text.replace('\n', '<br>')
                text = '<html>%s<br>%s</html>' % (text, pixmapAsHtml(pix))
            return qt4.QVariant(text)
        elif c == 'linkfile':
            return qt4.QVariant( datasetLinkFile(ds) )
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
        return self.__class__(self.doc, self.data[0], self.cols, newroot)

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

class DatasetRelationModel(TreeModel):
    """A model to show how the datasets are related to each file."""
    def __init__(self, doc, grouping='filename'):
        TreeModel.__init__(self, ('Dataset', 'Size', 'Type'))
        self.doc = doc
        self.linkednodes = {}
        self.grouping = grouping
        self.filter = ''
        self.refresh()

        self.connect(doc, qt4.SIGNAL('sigModified'), self.refresh)

    def makeGrpTreeNone(self):
        """Make tree with no grouping."""
        tree = TMNode( ('Dataset', 'Size', 'Type', 'File'), None )
        for name, ds in self.doc.data.iteritems():
            child = DatasetNode( self.doc, name,
                                 ('name', 'size', 'type', 'linkfile'),
                                 None )
            # filter nodes if requested
            if ( self.filter == '' or
                 any([d.find(self.filter)>=0 for d in child.data]) ):
                tree.insertChildSorted(child)
        return tree
        
    def makeGrpTree(self, coltitles, colitems, grouper, GrpNodeClass):
        """Make a tree grouping with function:
        coltitles: tuple of titles of columns for user
        colitems: tuple of items to lookup in DatasetNode
        grouper: function of dataset to return text for grouping
        GrpNodeClass: class for creating grouping nodes
        """
        grpnodes = {}
        for name, ds in self.doc.data.iteritems():
            child = DatasetNode(self.doc, name, colitems, None)

            # filter if requested
            if ( self.filter != '' and
                 all([d.find(self.filter)<0 for d in child.data]) ):
                continue

            grp = grouper(ds)
            if grp not in grpnodes:
                grpnodes[grp] = GrpNodeClass( (grp,), None )
            grpnodes[grp].insertChildSorted(child)
        return treeFromList(grpnodes.values(), coltitles)

    def makeGrpTreeFilename(self):
        """Make a tree of datasets grouped by linked file."""
        return self.makeGrpTree(
            ('Dataset', 'Size', 'Type'),
            ('name', 'size', 'type'),
            lambda ds: datasetLinkFile(ds),
            FilenameNode
            )

    def makeGrpTreeSize(self):
        """Make a tree of datasets grouped by dataset size."""
        return self.makeGrpTree(
            ('Dataset', 'Type', 'Filename'),
            ('name', 'type', 'linkfile'),
            lambda ds: ds.userSize(),
            TMNode
            )

    def makeGrpTreeType(self):
        """Make a tree of datasets grouped by dataset type."""
        return self.makeGrpTree(
            ('Dataset', 'Size', 'Filename'),
            ('name', 'size', 'linkfile'),
            lambda ds: ds.dstype,
            TMNode
            )

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
        self.emit(
            qt4.SIGNAL('dataChanged(const QModelIndex &, const QModelIndex &)'),
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

    def changeFilter(self, filtertext):
        """Change filtering text."""
        self.model.filter = filtertext
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
        self.connect(self.grpact, qt4.SIGNAL("triggered(QAction*)"),
                     self.slotGrpChanged)
        self.grpbutton.setMenu(self.grpmenu)
        self.grpbutton.setToolTip("Group datasets with property given")
        self.optslayout.addWidget(self.grpbutton)

        # filtering
        self.optslayout.addWidget(qt4.QLabel("Filter"))
        self.filteredit = LineEditWithClear()
        self.filteredit.setToolTip("Enter text here to filter datasets")
        self.connect(self.filteredit, qt4.SIGNAL('textChanged(const QString&)'),
                     self.slotFilterChanged)
        self.optslayout.addWidget(self.filteredit)

        self.layout.addLayout(self.optslayout)

        # the actual widget tree
        self.navtree = DatasetsNavigatorTree(thedocument, mainwin,
                                             self.grouping, None)
        self.layout.addWidget(self.navtree)

    def slotGrpChanged(self, action):
        """Grouping changed by user."""
        self.navtree.changeGrouping(action.grpname)

    def slotFilterChanged(self, filtertext):
        """Filtering changed by user."""
        self.navtree.changeFilter(unicode(filtertext))

class DataNavigatorWindow(qt4.QDockWidget):
    def __init__(self, thedocument, mainwin, *args):
        qt4.QDockWidget.__init__(self, *args)
        self.setWindowTitle("Data - Veusz")
        self.setObjectName("veuszdatawindow")

        self.nav = DatasetsNavigatorWidget(thedocument, mainwin, self)
        self.setWidget(self.nav)
