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

"""A widget for navigating datasets."""

from __future__ import division
import os.path
import numpy as N
import textwrap

from ..compat import crange, citems, czip, cbasestr, cstr
from .. import qtall as qt
from .. import setting
from .. import document
from .. import utils

from .lineeditwithclear import LineEditWithClear
from ..utils.treemodel import TMNode, TreeModel

def _(text, disambiguation=None, context="DatasetBrowser"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

def datasetLinkFile(ds):
    """Get a linked filename from a dataset."""
    return "/" if ds.linked is None else ds.linked.filename

def wrap(text, width):
    """Wrap text at columns. This needs to be split and rejoined."""
    lines = text.split("\n\n")
    out = [textwrap.fill(l, width).strip() for l in lines]
    return "\n\n".join(out)

class DatasetNode(TMNode):
    """Node for a dataset."""

    def __init__(self, model, dsname, cols, parent):
        ds = model.doc.data[dsname]
        data = []
        assert cols[0] == "name"
        for c in cols:
            if c == "name":
                data.append( dsname )
            elif c == "size":
                data.append( ds.userSize() )
            elif c == "type":
                data.append( ds.dstype )
            elif c == "linkfile":
                data.append( os.path.basename(datasetLinkFile(ds)) )
            elif c == "check":
                data.append( dsname in model.checked_datasets )

        TMNode.__init__(self, tuple(data), parent)
        self.model = model
        self.cols = cols
        self.dsname = dsname

    def getPreviewPixmap(self, ds):
        """Get a preview pixmap for a dataset."""
        size = (140, 70)
        if ds.dimensions != 1 or ds.datatype != "numeric":
            return None

        pixmap = qt.QPixmap(*size)
        pixmap.fill(qt.Qt.transparent)
        p = qt.QPainter(pixmap)
        p.setRenderHint(qt.QPainter.Antialiasing)

        # calculate data points
        try:
            if len(ds.data) < size[1]:
                y = ds.data
            else:
                intvl = len(ds.data)//size[1]+1
                y = ds.data[::intvl]
            x = N.arange(len(y))

            # plot data points on image
            minval, maxval = N.nanmin(y), N.nanmax(y)
            y = (y-minval) / (maxval-minval) * size[1]
            finite = N.isfinite(y)
            x, y = x[finite], y[finite]
            x = x * (1./len(x)) * size[0]

            poly = qt.QPolygonF()
            utils.addNumpyToPolygonF(poly, x, size[1]-y)
            p.setPen( qt.QPen(qt.Qt.blue) )
            p.drawPolyline(poly)

            # draw x axis if span 0
            p.setPen( qt.QPen(qt.Qt.black) )
            if minval <= 0 and maxval > 0:
                y0 = size[1] - (0-minval)/(maxval-minval)*size[1]
                p.drawLine(x[0], y0, x[-1], y0)
            else:
                p.drawLine(x[0], size[1], x[-1], size[1])
            p.drawLine(x[0], 0, x[0], size[1])

        except (ValueError, ZeroDivisionError):
            # zero sized array after filtering or min == max, so return None
            p.end()
            return None

        p.end()
        return pixmap

    def toolTip(self, column):
        """Return tooltip for column."""
        try:
            ds = self.model.doc.data[self.data[0]]
        except KeyError:
            return None

        c = self.cols[column]
        if c == "name":
            text = '%s: %s' % (self.data[0], ds.description())
            if ds.linked:
                text += '\n\n' + _('Linked to %s') % ds.linked.filename
            if ds.tags:
                text += '\n\n' + _('Tags: %s') % (' '.join(sorted(ds.tags)))
            return wrap(text, 40)
        elif c == "size" or (c == 'type' and 'size' not in self.cols):
            text = ds.userPreview()
            # add preview of dataset if possible
            pix = self.getPreviewPixmap(ds)
            if pix:
                text = text.replace("\n", "<br>")
                text = "<html>%s<br>%s</html>" % (text, utils.pixmapAsHtml(pix))
            return text
        elif c == "linkfile" or c == "type":
            return wrap(ds.linkedInformation(), 40)
        return None

    def dataset(self):
        """Get associated dataset."""
        try:
            return self.model.doc.data[self.data[0]]
        except KeyError:
            return None

    def datasetName(self):
        """Get dataset name."""
        return self.data[0]

    def cloneTo(self, newroot):
        """Make a clone of self at the root given."""
        return self.__class__(
            self.model, self.dsname, self.cols, newroot)

class FilenameNode(TMNode):
    """A special node for holding filenames of files."""

    def nodeData(self, column):
        """basename of filename for data."""
        if column == 0:
            if self.data[0] == "/":
                return "/"
            else:
                return os.path.basename(self.data[0])
        return None

    def filename(self):
        """Return filename."""
        return self.data[0]

    def toolTip(self, column):
        """Full filename for tooltip."""
        if column == 0:
            return self.data[0]
        return None

def treeFromList(nodelist, rootdata):
    """Construct a tree from a list of nodes."""
    tree = TMNode( rootdata, None )
    for node in nodelist:
        tree.insertChildSorted(node)
    return tree

class DatasetRelationModel(TreeModel):
    """A model to show how the datasets are related to each file."""
    def __init__(self, doc, grouping="filename", readonly=False,
                 filterdims=None, filterdtype=None,
                 checkable=False):
        """Model parameters:
        doc: document
        group: how to group datasets
        readonly: no modification of data
        filterdims/filterdtype: filter dimensions and datatypes.
        checkable: whether datasets are checkable
        """

        TreeModel.__init__(self, (_("Dataset"), _("Size"), _("Type")))
        self.doc = doc
        self.linkednodes = {}
        self.grouping = grouping
        self.filter = ""
        self.readonly = readonly
        self.filterdims = filterdims
        self.filterdtype = filterdtype
        self.checkable = checkable
        self.checked_datasets = set()
        self.refresh()

        doc.signalModified.connect(self.refresh)

    def datasetFilterOut(self, ds, node):
        """Should dataset be filtered out by filter options."""
        filterout = False

        # is filter text not in node text or text
        keep = True
        if self.filter != "":
            keep = False
            if any([t.find(self.filter) >= 0 for t in ds.tags
                    if isinstance(t, cbasestr)]):
                keep = True
            if any([t.find(self.filter) >= 0 for t in node.data
                    if isinstance(t, cbasestr)]):
                keep = True
        # check dimensions haven't been filtered
        if ( self.filterdims is not None and
             ds.dimensions not in self.filterdims ):
            filterout = True
        # check type hasn't been filtered
        if ( self.filterdtype is not None and
             ds.datatype not in self.filterdtype and
             'all' not in self.filterdtype):
            filterout = True

        if filterout:
            return True
        return not keep

    def makeGrpTreeNone(self):
        """Make tree with no grouping."""

        heads = [_("Dataset"), _("Size"), _("Type"), _("File")]
        cols = ["name", "size", "type", "linkfile"]
        if self.checkable:
            heads += [_('Select')]
            cols += [_('check')]

        tree = TMNode(heads , None)
        for name, ds in citems(self.doc.data):
            child = DatasetNode(self, name, cols, None)

            # add if not filtered for filtering
            if not self.datasetFilterOut(ds, child):
                tree.insertChildSorted(child)
        return tree

    def makeGrpTree(self, coltitles, colitems, grouper, GrpNodeClass):
        """Make a tree grouping with function:
        coltitles: tuple of titles of columns for user
        colitems: tuple of items to lookup in DatasetNode
        grouper: function of dataset to return text for grouping
        GrpNodeClass: class for creating grouping nodes
        """

        if self.checkable:
            coltitles = coltitles + [_('Select')]
            colitems = colitems + [_('check')]

        grpnodes = {}
        for name, ds in citems(self.doc.data):
            child = DatasetNode(self, name, colitems, None)

            # check whether filtered out
            if not self.datasetFilterOut(ds, child):
                # get group
                grps = grouper(ds)
                for grp in grps:
                    if grp not in grpnodes:
                        grpnodes[grp] = GrpNodeClass( (grp,), None )
                    # add to group
                    grpnodes[grp].insertChildSorted(child)

        return treeFromList(list(grpnodes.values()), coltitles)

    def makeGrpTreeFilename(self):
        """Make a tree of datasets grouped by linked file."""
        return self.makeGrpTree(
            [_("Dataset"), _("Size"), _("Type")],
            ["name", "size", "type"],
            lambda ds: (datasetLinkFile(ds),),
            FilenameNode
            )

    def makeGrpTreeSize(self):
        """Make a tree of datasets grouped by dataset size."""
        return self.makeGrpTree(
            [_("Dataset"), _("Type"), _("Filename")],
            ["name", "type", "linkfile"],
            lambda ds: (ds.userSize(),),
            TMNode
            )

    def makeGrpTreeType(self):
        """Make a tree of datasets grouped by dataset type."""
        return self.makeGrpTree(
            [_("Dataset"), _("Size"), _("Filename")],
            ["name", "size", "linkfile"],
            lambda ds: (ds.dstype,),
            TMNode
            )

    def makeGrpTreeTags(self):
        """Make a tree of datasets grouped by tags."""

        def getgrp(ds):
            if ds.tags:
                return sorted(ds.tags)
            else:
                return [_("None")]

        return self.makeGrpTree(
            [_("Dataset"), _("Size"), _("Type"), _("Filename")],
            ["name", "size", "type", "linkfile"],
            getgrp,
            TMNode
            )

    def flags(self, idx):
        """Return model flags for index."""
        f = TreeModel.flags(self, idx)
        # allow dataset names to be edited
        obj = self.objFromIndex(idx)
        if idx.isValid() and isinstance(obj, DatasetNode):
            col = idx.column()
            if not self.readonly and col == 0:
                # renameable dataset
                f |= qt.Qt.ItemIsEditable
            elif obj.cols[col] == "check":
                # checkable dataset
                f |= qt.Qt.ItemIsUserCheckable
        return f

    def setData(self, idx, val, role):
        """Rename dataset."""
        dsnode = self.objFromIndex(idx)
        name = dsnode.cols[idx.column()]

        if name == "name":
            if( not utils.validateDatasetName(val) or
                val in self.doc.data ):
                return False

            self.doc.applyOperation(
                document.OperationDatasetRename(dsnode.data[0], val))
            self.dataChanged.emit(idx, idx)
            return True

        elif name == "check":
            # update check box
            name = dsnode.dsname
            if val:
                self.checked_datasets.add(name)
            else:
                self.checked_datasets.remove(name)
            # emitted by refresh: self.dataChanged.emit(idx, idx)
            self.refresh()
            return True

    @qt.pyqtSlot()
    def refresh(self):
        """Update tree of datasets when document changes."""

        tree = {
            "none": self.makeGrpTreeNone,
            "filename": self.makeGrpTreeFilename,
            "size": self.makeGrpTreeSize,
            "type": self.makeGrpTreeType,
            "tags": self.makeGrpTreeTags,
            }[self.grouping]()

        self.syncTree(tree)

class DatasetsNavigatorTree(qt.QTreeView):
    """Tree view for dataset names."""

    updateitem = qt.pyqtSignal()
    selecteddatasets = qt.pyqtSignal(list)

    def __init__(self, doc, mainwin, grouping, parent,
                 readonly=False, filterdims=None, filterdtype=None,
                 checkable=False):
        """Initialise the dataset tree view.
        doc: veusz document
        mainwin: veusz main window (or None if readonly)
        grouping: grouping mode of datasets
        parent: parent window or None
        filterdims: if set, only show datasets with dimensions given
        filterdtype: if set, only show datasets with type given
        checkable: allow datasets to be selected
        """

        qt.QTreeView.__init__(self, parent)
        self.doc = doc
        self.mainwindow = mainwin
        self.model = DatasetRelationModel(
            doc, grouping, readonly=readonly,
            filterdims=filterdims,
            filterdtype=filterdtype,
            checkable=checkable)

        self.setModel(self.model)
        self.setSelectionMode(qt.QTreeView.ExtendedSelection)
        self.setSelectionBehavior(qt.QTreeView.SelectRows)
        self.setUniformRowHeights(True)
        self.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        if not readonly:
            self.customContextMenuRequested.connect(self.showContextMenu)
        self.model.refresh()
        self.expandAll()

        # stretch of columns
        hdr = self.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, qt.QHeaderView.Stretch)
        for col in crange(1, 3):
            hdr.setSectionResizeMode(col, qt.QHeaderView.ResizeToContents)

        # when documents have finished opening, expand all nodes
        if mainwin is not None:
            mainwin.documentOpened.connect(self.expandAll)

        # keep track of selection
        self.selectionModel().selectionChanged.connect(self.slotNewSelection)

        # expand nodes by default
        self.model.rowsInserted.connect(self.slotNewRow)

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

    def selectDataset(self, dsname):
        """Find, and if possible select dataset name."""

        matches = self.model.match(
            self.model.index(0, 0, qt.QModelIndex()),
            qt.Qt.DisplayRole, dsname, -1,
            qt.Qt.MatchFixedString | qt.Qt.MatchCaseSensitive |
            qt.Qt.MatchRecursive )
        for idx in matches:
            if isinstance(self.model.objFromIndex(idx), DatasetNode):
                self.selectionModel().setCurrentIndex(
                    idx, qt.QItemSelectionModel.SelectCurrent |
                    qt.QItemSelectionModel.Clear |
                    qt.QItemSelectionModel.Rows )

    def showContextMenu(self, pt):
        """Context menu for nodes."""

        # get selected nodes
        idxs = self.selectionModel().selection().indexes()
        nodes = [ self.model.objFromIndex(i)
                  for i in idxs if i.column() == 0 ]

        # unique list of types of nodes
        types = set([ type(n) for n in nodes ])

        menu = qt.QMenu()
        # put contexts onto submenus if multiple types selected
        if DatasetNode in types:
            thismenu = menu
            if len(types) > 1:
                thismenu = menu.addMenu(_("Datasets"))
            self.datasetContextMenu(
                [n for n in nodes if isinstance(n, DatasetNode)],
                thismenu)
        elif FilenameNode in types:
            thismenu = menu
            if len(types) > 1:
                thismenu = menu.addMenu(_("Files"))
            self.filenameContextMenu(
                [n for n in nodes if isinstance(n, FilenameNode)],
                thismenu)

        def _paste():
            """Paste dataset(s)."""
            if document.isClipboardDataMime():
                mime = qt.QApplication.clipboard().mimeData()
                self.doc.applyOperation(document.OperationDataPaste(mime))

        # if there is data to paste, add menu item
        if document.isClipboardDataMime():
            menu.addAction(_("Paste"), _paste)

        if len( menu.actions() ) != 0:
            menu.exec_(self.mapToGlobal(pt))

    def datasetContextMenu(self, dsnodes, menu):
        """Return context menu for datasets."""
        from ..dialogs import dataeditdialog

        datasets = [d.dataset() for d in dsnodes]
        dsnames = [d.datasetName() for d in dsnodes]

        def _edit():
            """Open up dialog box to recreate dataset."""
            for dataset, dsname in czip(datasets, dsnames):
                if type(dataset) in dataeditdialog.recreate_register:
                    dataeditdialog.recreate_register[type(dataset)](
                        self.mainwindow, self.doc, dataset, dsname)
        def _edit_data():
            """Open up data edit dialog."""
            for dataset, dsname in czip(datasets, dsnames):
                if type(dataset) not in dataeditdialog.recreate_register:
                    self.mainwindow.slotDataEdit(editdataset=dsname)
        def _delete():
            """Simply delete dataset."""
            self.doc.applyOperation(
                document.OperationMultiple(
                    [document.OperationDatasetDelete(n) for n in dsnames],
                    descr=_('delete dataset(s)')))
        def _unlink_file():
            """Unlink dataset from file."""
            self.doc.applyOperation(
                document.OperationMultiple(
                    [document.OperationDatasetUnlinkFile(n)
                     for d,n in czip(datasets,dsnames)
                     if d.canUnlink() and d.linked],
                    descr=_('unlink dataset(s)')))
        def _unlink_relation():
            """Unlink dataset from relation."""
            self.doc.applyOperation(
                document.OperationMultiple(
                    [document.OperationDatasetUnlinkRelation(n)
                     for d,n in czip(datasets,dsnames)
                     if d.canUnlink() and not d.linked],
                    descr=_('unlink dataset(s)')))
        def _copy():
            """Copy data to clipboard."""
            mime = document.generateDatasetsMime(dsnames, self.doc)
            qt.QApplication.clipboard().setMimeData(mime)

        # editing
        recreate = [type(d) in dataeditdialog.recreate_register
                    for d in datasets]
        if any(recreate):
            menu.addAction(_("Edit"), _edit)
        if not all(recreate):
            menu.addAction(_("Edit data"), _edit_data)

        # deletion
        menu.addAction(_("Delete"), _delete)

        # linking
        unlink_file = [d.canUnlink() and d.linked for d in datasets]
        if any(unlink_file):
            menu.addAction(_("Unlink file"), _unlink_file)
        unlink_relation = [d.canUnlink() and not d.linked for d in datasets]
        if any(unlink_relation):
            menu.addAction(_("Unlink relation"), _unlink_relation)

        # tagging submenu
        tagmenu = menu.addMenu(_("Tags"))
        for tag in self.doc.datasetTags():
            def toggle(tag=tag):
                state = [tag in d.tags for d in datasets]
                if all(state):
                    op = document.OperationDataUntag
                else:
                    op = document.OperationDataTag
                self.doc.applyOperation(op(tag, dsnames))

            a = tagmenu.addAction(tag, toggle)
            a.setCheckable(True)
            state = [tag in d.tags for d in datasets]
            a.setChecked( all(state) )

        def addtag():
            tag, ok = qt.QInputDialog.getText(
                self, _("New tag"), _("Enter new tag"))
            if ok:
                tag = tag.strip().replace(' ', '')
                if tag:
                    self.doc.applyOperation( document.OperationDataTag(
                            tag, dsnames) )
        tagmenu.addAction(_("Add..."), addtag)

        # copy
        menu.addAction(_("Copy"), _copy)

        if len(datasets) == 1:
            useasmenu = menu.addMenu(_("Use as"))
            self.getMenuUseAs(useasmenu, datasets[0])

    def filenameContextMenu(self, nodes, menu):
        """Return context menu for filenames."""

        from ..dialogs.reloaddata import ReloadData

        filenames = [n.filename() for n in nodes if n.filename() != '/']
        if not filenames:
            return

        def _reload():
            """Reload data in this file."""
            d = ReloadData(self.doc, self.mainwindow, filenames=set(filenames))
            self.mainwindow.showDialog(d)
        def _unlink_all():
            """Unlink all datasets associated with file."""
            self.doc.applyOperation(
                document.OperationMultiple(
                    [document.OperationDatasetUnlinkByFile(f) for f in filenames],
                    descr=_('unlink by file')))
        def _delete_all():
            """Delete all datasets associated with file."""
            self.doc.applyOperation(
                document.OperationMultiple(
                    [document.OperationDatasetDeleteByFile(f) for f in filenames],
                    descr=_('delete by file')))

        menu.addAction(_("Reload"), _reload)
        menu.addAction(_("Unlink all"), _unlink_all)
        menu.addAction(_("Delete all"), _delete_all)

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
                 path[:12] != "/StyleSheet/" ):
                menu.addAction(path, _setdataset)

        self.doc.walkNodes(addifdatasetsetting, nodetypes=("setting",))

    def keyPressEvent(self, event):
        """Enter key selects widget."""
        if event.key() in (qt.Qt.Key_Return, qt.Qt.Key_Enter):
            self.updateitem.emit()
            return
        qt.QTreeView.keyPressEvent(self, event)

    def mouseDoubleClickEvent(self, event):
        """Emit updateitem signal if double clicked."""
        retn = qt.QTreeView.mouseDoubleClickEvent(self, event)
        self.updateitem.emit()
        return retn

    def slotNewSelection(self, selected, deselected):
        """Emit selecteditem signal on new selection."""
        self.selecteddatasets.emit(self.getSelectedDatasets())

    def slotNewRow(self, parent, start, end):
        """Expand parent if added."""
        self.expand(parent)

    def getSelectedDatasets(self):
        """Returns list of selected datasets."""

        datasets = []
        for idx in self.selectionModel().selectedRows():
            node = self.model.objFromIndex(idx)
            try:
                name = node.datasetName()
                if name in self.doc.data:
                    datasets.append(name)
            except AttributeError:
                pass
        return datasets

    def checkedDatasets(self):
        """Returns list of checked datasets (if checkable was True)."""
        return sorted(self.model.checked_datasets)

    def setCheckedDatasets(self, checked):
        """Update list of checked datasets."""
        self.model.checked_datasets = set(checked)
        self.model.refresh()

    def resetChecks(self):
        """Reset checked datasets."""
        self.model.checked_datasets.clear()
        self.model.refresh()

class DatasetBrowser(qt.QWidget):
    """Widget which shows the document's datasets."""

    # how datasets can be grouped
    grpnames = ("none", "filename", "type", "size", "tags")
    grpentries = {
        "none": _("None"),
        "filename": _("Filename"),
        "type": _("Type"), 
        "size": _("Size"),
        "tags": _("Tags"),
        }

    def __init__(self, thedocument, mainwin, parent, readonly=False,
                 filterdims=None, filterdtype=None, checkable=False):
        """Initialise widget:
        thedocument: document to show
        mainwin: main window of application (or None if readonly)
        parent: parent of widget.
        readonly: for choosing datasets only
        filterdims: if set, only show datasets with dimensions given
        filterdtype: if set, only show datasets with type given
        checkable: allow datasets to be selected
        """

        qt.QWidget.__init__(self, parent)
        self.layout = qt.QVBoxLayout()
        self.setLayout(self.layout)

        # options for navigator are in this layout
        self.optslayout = qt.QHBoxLayout()

        # grouping options - use a menu to choose the grouping
        self.grpbutton = qt.QPushButton(_("Group"))
        self.grpmenu = qt.QMenu()
        self.grouping = setting.settingdb.get("navtree_grouping", "filename")
        self.grpact = qt.QActionGroup(self)
        self.grpact.setExclusive(True)
        for name in self.grpnames:
            a = self.grpmenu.addAction(self.grpentries[name])
            a.grpname = name
            a.setCheckable(True)
            if name == self.grouping:
                a.setChecked(True)
            self.grpact.addAction(a)
        self.grpact.triggered.connect(self.slotGrpChanged)
        self.grpbutton.setMenu(self.grpmenu)
        self.grpbutton.setToolTip(_("Group datasets with property given"))
        self.optslayout.addWidget(self.grpbutton)

        # filtering by entering text
        searchlabel = qt.QLabel()
        searchlabel.setPixmap(utils.getIcon('kde-search-jss').pixmap(18,18))
        self.optslayout.addWidget(searchlabel)
        self.filteredit = LineEditWithClear()
        self.filteredit.setToolTip(_("Search for dataset names"))
        self.filteredit.textChanged.connect(self.slotFilterChanged)
        self.optslayout.addWidget(self.filteredit)

        self.layout.addLayout(self.optslayout)

        # the actual widget tree
        self.navtree = DatasetsNavigatorTree(
            thedocument, mainwin, self.grouping, None,
            readonly=readonly, filterdims=filterdims, filterdtype=filterdtype,
            checkable=checkable)
        self.layout.addWidget(self.navtree)

    def slotGrpChanged(self, action):
        """Grouping changed by user."""
        self.navtree.changeGrouping(action.grpname)
        setting.settingdb["navtree_grouping"] = action.grpname

    def slotFilterChanged(self, filtertext):
        """Filtering changed by user."""
        self.navtree.changeFilter(filtertext)

    def selectDataset(self, dsname):
        """Find, and if possible select dataset name."""
        self.navtree.selectDataset(dsname)

    def checkedDatasets(self):
        """Returns list of checked datasets (if checkable was True)."""
        return self.navtree.checkedDatasets()

    def setCheckedDatasets(self, checked):
        """Update list of checked datasets."""
        self.navtree.setCheckedDatasets(checked)

    def reset(self):
        """Uncheck all items."""
        self.navtree.resetChecks()
        self.filteredit.clear()

class DatasetBrowserPopup(DatasetBrowser):
    """Popup window for dataset browser for selecting datasets.
    This is used by setting.controls.Dataset
    """

    closing = qt.pyqtSignal()
    newdataset = qt.pyqtSignal(cstr)

    def __init__(self, document, dsname, parent,
                 filterdims=None, filterdtype=None):
        """Open popup window for document
        dsname: dataset name
        parent: window parent
        filterdims: if set, only show datasets with dimensions given
        filterdtype: if set, only show datasets with type given
        """

        DatasetBrowser.__init__(
            self, document, None, parent, readonly=True,
            filterdims=filterdims, filterdtype=filterdtype)
        self.setWindowFlags(qt.Qt.Popup)
        self.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.spacing = self.fontMetrics().height()

        utils.positionFloatingPopup(self, parent)
        self.selectDataset(dsname)
        self.installEventFilter(self)

        self.navtree.setFocus()

        self.navtree.updateitem.connect(self.slotUpdateItem)

    def eventFilter(self, node, event):
        """Grab clicks outside this window to close it."""
        if ( isinstance(event, qt.QMouseEvent) and
             event.buttons() != qt.Qt.NoButton ):
            frame = qt.QRect(0, 0, self.width(), self.height())
            if not frame.contains(event.pos()):
                self.close()
                return True
        return DatasetBrowser.eventFilter(self, node, event)

    def sizeHint(self):
        """A reasonable size for the text editor."""
        return qt.QSize(self.spacing*30, self.spacing*20)

    def closeEvent(self, event):
        """Tell the calling widget that we are closing."""
        self.closing.emit()
        event.accept()

    def slotUpdateItem(self):
        """Emit new dataset signal."""
        selected = self.navtree.selectionModel().currentIndex()
        if selected.isValid():
            n = self.navtree.model.objFromIndex(selected)
            if isinstance(n, DatasetNode):
                self.newdataset.emit(n.data[0])
                self.close()
