#    Copyright (C) 2010 Jeremy S. Sanders
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

"""Contains a model and view for handling a tree of widgets."""

from __future__ import division, print_function
from ..compat import crange
from .. import qtall as qt4

from .. import utils
from .. import document

def _(text, disambiguation=None, context="WidgetTree"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class _WidgetNode:
    """Class to represent widgets in WidgetTreeModel.

     parent: parent _WidgetNode
     widget: document widget node is representing
     children: child nodes
     data: tuple of data items, so we can see whether the node should
           be refreshed
    """

    def __init__(self, parent, widget):
        self.parent = parent
        self.widget = widget
        self.children = []
        self.data = self.getData()

    def getData(self):
        """Get the latest version of the data."""
        w = self.widget
        parenthidden = False if self.parent is None else self.parent.data[3]
        hidden = parenthidden or ("hide" in w.settings and w.settings.hide)
        return (
            w.name,
            w.typename,
            w.userdescription,
            hidden,
        )

    def __repr__(self):
        return "<_WidgetNode widget:%s>" % repr(self.widget)

class WidgetTreeModel(qt4.QAbstractItemModel):
    """A model representing the widget tree structure.

    We hide the actual widgets behind a tree of _WidgetNode
    objects. The syncTree method synchronises the tree in the model to
    the tree in the document. It works out which nodes to be deleted,
    which to be added and which to be moved around. It informs the
    view that the data are being changed using the standard
    begin... and end... functions. The synchronisation code is a bit
    hairy and is hopefully correct.

    This extra layer is necessary as the model requires that the
    document underneath it can't be changed until the view knows its
    about to be changed.

    """

    def __init__(self, document, parent=None):
        """Initialise using document."""

        qt4.QAbstractItemModel.__init__(self, parent)

        self.document = document

        document.signalModified.connect(self.slotDocumentModified)
        document.sigWiped.connect(self.deleteTree)

        # root node of document
        self.rootnode = _WidgetNode(None, document.basewidget)
        # map of widgets to nodes
        self.widgetnodemap = {self.rootnode.widget: self.rootnode}
        self.syncTree()

    def deleteTree(self):
        """Reset tree contents (for loading docs, etc)."""
        self.beginRemoveRows(
            self.nodeIndex(self.rootnode),
            0, len(self.rootnode.children))
        self.rootnode.widget = self.document.basewidget
        del self.rootnode.children[:]
        self.widgetnodemap = {self.rootnode.widget: self.rootnode}
        self.endRemoveRows()

    def slotDocumentModified(self):
        """The document has been changed."""
        self.syncTree()

    def syncTree(self):
        """Synchronise tree to document."""

        docwidgets = set()
        def recursecollect(widget):
            """Recursively collect widgets in document."""
            docwidgets.add(widget)
            for child in widget.children:
                recursecollect(child)

        recursecollect(self.rootnode.widget)
        self._recursiveupdate(self.rootnode.widget, docwidgets)

    def _recursiveupdate(self, widget, docwidgets):
        """Recursively remove, add and move nodes to correct place.

        widget: widget to operate below
        docwidgets: all widgets used in the document
        """

        #print('recurse', widget)
        node = self.widgetnodemap[widget]

        # delete non existent child nodes recursively
        for nch in node.children[::-1]:
            if nch.widget not in docwidgets:
                self._recursivedelete(nch)

        # now iterate over children to see whether anything has
        # changed
        for i in crange(len(widget.children)):
            c = widget.children[i]
            add = False
            if c not in self.widgetnodemap:
                # need to add widget as not in doc
                #print('add', c, i, node)
                self.beginInsertRows(self.nodeIndex(node), i, i)
                self.widgetnodemap[c] = cnode = _WidgetNode(node, c)
                node.children.insert(i, cnode)
                self.endInsertRows()
                add = True

            elif (i >= len(node.children) or
                  c is not node.children[i].widget or
                  c.parent is not node.children[i].parent.widget):
                # need to move widget
                cnode = self.widgetnodemap[c]
                oldparent = cnode.parent
                oldrow = oldparent.children.index(cnode)

                # this code works because when moving, we're always
                # moving a widget to this position in the list (and
                # never backwards within this list)
                oldidx = self.nodeIndex(oldparent)
                newidx = oldidx if oldparent is node else self.nodeIndex(node)

                #print('move', oldparent, oldrow, node, i)
                self.beginMoveRows(oldidx, oldrow, oldrow, newidx, i)
                del oldparent.children[oldrow]
                node.children.insert(i, cnode)
                cnode.parent = node
                self.endMoveRows()

            if not add:
                # update data if changed
                cnode = self.widgetnodemap[c]
                data = cnode.getData()
                if cnode.data != data:
                    index = self.nodeIndex(cnode)
                    cnode.data = data
                    #print('changed', c, data)
                    self.dataChanged.emit(index, index)

            self._recursiveupdate(c, docwidgets)

        #print('rec retn')

    def _recursivedelete(self, node):
        """Recursively delete node and its children."""
        for cnode in node.children[::-1]:
            self._recursivedelete(cnode)
        parentnode = node.parent
        if parentnode is not None:
            #print('delete', node.widget)
            row = parentnode.children.index(node)
            self.beginRemoveRows(self.nodeIndex(parentnode), row, row)
            del parentnode.children[row]
            del self.widgetnodemap[node.widget]
            self.endRemoveRows()

    def columnCount(self, parent):
        """Return number of columns of data."""
        return 2

    def rowCount(self, index):
        """Return number of rows of children of index."""

        if index.isValid():
            return len(index.internalPointer().children)
        else:
            # always 1 root node
            return 1

    def data(self, index, role):
        """Return data for the index given.

        Uses the data from the _WidgetNode class.

        """

        if not index.isValid():
            return None

        column = index.column()
        data = index.internalPointer().data

        if role in (qt4.Qt.DisplayRole, qt4.Qt.EditRole):
            # return text for columns
            if column == 0:
                return data[0]
            elif column == 1:
                return data[1]

        elif role == qt4.Qt.DecorationRole:
            # return icon for first column
            if column == 0:
                filename = 'button_%s' % data[1]
                return utils.getIcon(filename)

        elif role == qt4.Qt.ToolTipRole:
            # provide tool tip showing description
            return data[2]

        elif role == qt4.Qt.TextColorRole:
            # show disabled looking text if object or any parent is hidden
            # return brush for hidden widget text, based on disabled text
            if data[3]:
                return qt4.QPalette().brush(
                    qt4.QPalette.Disabled, qt4.QPalette.Text)

        # return nothing
        return None

    def setData(self, index, name, role):
        """User renames object. This renames the widget."""

        if not index.isValid():
            return False

        widget = index.internalPointer().widget

        # check symbols in name
        if not utils.validateWidgetName(name):
            return False

        # check name not already used
        if widget.parent.hasChild(name):
            return False

        # actually rename the widget
        self.document.applyOperation(
            document.OperationWidgetRename(widget, name))

        self.dataChanged.emit(index, index)
        return True

    def flags(self, index):
        """What we can do with the item."""

        if not index.isValid():
            return qt4.Qt.ItemIsEnabled

        flags = ( qt4.Qt.ItemIsEnabled | qt4.Qt.ItemIsSelectable |
                  qt4.Qt.ItemIsDropEnabled )
        if ( index.internalPointer().parent is not None and
             index.column() == 0 ):
            # allow items other than root to be edited and dragged
            flags |= qt4.Qt.ItemIsEditable | qt4.Qt.ItemIsDragEnabled

        return flags

    def headerData(self, section, orientation, role):
        """Return the header of the tree."""

        if orientation == qt4.Qt.Horizontal and role == qt4.Qt.DisplayRole:
            val = ('Name', 'Type')[section]
            return val
        return None

    def nodeIndex(self, node):
        row = 0 if node.parent is None else node.parent.children.index(node)
        return self.createIndex(row, 0, node)

    def index(self, row, column, parent):
        """Construct an index for a child of parent."""

        if parent.isValid():
            # normal widget
            try:
                child = parent.internalPointer().children[row]
            except IndexError:
                return qt4.QModelIndex()
        else:
            # root widget
            child = self.rootnode
        return self.createIndex(row, column, child)

    def getWidgetIndex(self, widget):
        """Returns index for widget specified."""

        if widget not in self.widgetnodemap:
            return None
        node = self.widgetnodemap[widget]
        parent = node.parent
        row = 0 if parent is None else parent.children.index(node)
        return self.createIndex(row, 0, node)

    def parent(self, index):
        """Find the parent of the index given."""

        if not index.isValid():
            return qt4.QModelIndex()

        parent = index.internalPointer().parent
        if parent is None:
            return qt4.QModelIndex()
        else:
            gparent = parent.parent
            row = 0 if gparent is None else gparent.children.index(parent)
            return self.createIndex(row, 0, parent)

    def getSettings(self, index):
        """Return the settings for the index selected."""
        return index.internalPointer().widget.settings

    def getWidget(self, index):
        """Get associated widget for index selected."""
        return index.internalPointer().widget

    def removeRows(self, row, count, parentindex):
        """Remove widgets from parent.

        This is used by the mime dragging and dropping
        """

        if not parentindex.isValid():
            return

        parent = self.getWidget(parentindex)

        # make an operation to delete the rows
        deleteops = []
        for w in parent.children[row:row+count]:
            deleteops.append( document.OperationWidgetDelete(w) )
        op = document.OperationMultiple(deleteops, descr=_("remove widget(s)"))
        self.document.applyOperation(op)
        return True

    def supportedDropActions(self):
        """Supported drag and drop actions."""
        return qt4.Qt.MoveAction | qt4.Qt.CopyAction

    def mimeData(self, indexes):
        """Get mime data for indexes."""
        widgets = [idx.internalPointer().widget for idx in indexes]
        return document.generateWidgetsMime(widgets)

    def mimeTypes(self):
        """Accepted mime types."""
        return [document.widgetmime]

    def dropMimeData(self, mimedata, action, row, column, parentindex):
        """User drags and drops widget."""

        if action == qt4.Qt.IgnoreAction:
            return True

        data = document.getWidgetMime(mimedata)
        if data is None:
            return False

        if parentindex.isValid():
            parent = self.getWidget(parentindex)
        else:
            parent = self.document.basewidget

        # check parent supports child
        if not document.isMimeDropable(parent, data):
            return False

        # work out where row will be pasted
        startrow = row
        if row == -1:
            startrow = len(parent.children)

        op = document.OperationWidgetPaste(parent, data, index=startrow)
        self.document.applyOperation(op)
        return True

class WidgetTreeView(qt4.QTreeView):
    """A model view for viewing the widgets."""

    def __init__(self, model, *args):
        qt4.QTreeView.__init__(self, *args)
        self.setModel(model)
        self.expandAll()

        # stretch header
        hdr = self.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, qt4.QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, qt4.QHeaderView.Custom)

        # setup drag and drop
        self.setSelectionMode(qt4.QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)

    def testModifier(self, e):
        """Look for keyboard modifier for copy or move."""
        if e.keyboardModifiers() & qt4.Qt.ControlModifier:
            e.setDropAction(qt4.Qt.CopyAction)
        else:
            e.setDropAction(qt4.Qt.MoveAction)

    def handleInternalMove(self, event):
        """Handle a move inside treeview."""

        # make sure qt doesn't handle this
        event.setDropAction(qt4.Qt.IgnoreAction)
        event.ignore()

        if not self.viewport().rect().contains(event.pos()):
            return

        # get widget at event position
        index = self.indexAt(event.pos())
        if not index.isValid():
            index = self.rootIndex()

        # adjust according to drop indicator position
        row = -1
        posn = self.dropIndicatorPosition()
        if posn == qt4.QAbstractItemView.AboveItem:
            row = index.row()
            index = index.parent()
        elif posn == qt4.QAbstractItemView.BelowItem:
            row = index.row() + 1
            index = index.parent()

        if index.isValid():
            parent = self.model().getWidget(index)
            data = document.getWidgetMime(event.mimeData())
            if document.isMimeDropable(parent, data):
                # move the widget!
                parentpath = parent.path
                widgetpaths = document.getMimeWidgetPaths(data)
                ops = []
                r = row
                for path in widgetpaths:
                    ops.append(
                        document.OperationWidgetMove(path, parentpath, r) )
                    if r >= 0:
                        r += 1

                self.model().document.applyOperation(
                    document.OperationMultiple(ops, descr='move'))
                event.ignore()

    def dropEvent(self, e):
        """When an object is dropped on the view."""
        self.testModifier(e)

        if e.source() is self and e.dropAction() == qt4.Qt.MoveAction:
            self.handleInternalMove(e)

        qt4.QTreeView.dropEvent(self, e)

    def dragMoveEvent(self, e):
        """Make items move by default and copy if Ctrl is held down."""
        self.testModifier(e)

        qt4.QTreeView.dragMoveEvent(self, e)
