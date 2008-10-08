#    Copyright (C) 2004-2006 Jeremy S. Sanders
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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
###############################################################################

# $Id$

"""Edit the document using a tree and properties.
"""

import os

import veusz.qtall as qt4

import veusz.widgets as widgets
import veusz.utils as utils
import veusz.document as document
import veusz.setting as setting

import action

class WidgetTreeModel(qt4.QAbstractItemModel):
    """A model representing the widget tree structure.
    """

    def __init__(self, document, parent=None):
        """Initialise using document."""
        
        qt4.QAbstractItemModel.__init__(self, parent)

        self.document = document

        self.connect( self.document, qt4.SIGNAL("sigModified"),
                      self.slotDocumentModified )
        self.connect( self.document, qt4.SIGNAL("sigWiped"),
                      self.slotDocumentModified )

    def slotDocumentModified(self):
        """The document has been changed."""
        self.emit( qt4.SIGNAL('layoutChanged()') )

    def columnCount(self, parent):
        """Return number of columns of data."""
        return 2

    def data(self, index, role):
        """Return data for the index given."""

        # why do we get passed invalid indicies? :-)
        if not index.isValid():
            return qt4.QVariant()

        column = index.column()
        obj = index.internalPointer()

        if role == qt4.Qt.DisplayRole:
            # return text for columns
            if column == 0:
                return qt4.QVariant(obj.name)
            elif column == 1:
                return qt4.QVariant(obj.typename)

        elif role == qt4.Qt.DecorationRole:
            # return icon for first column
            if column == 0:
                filename = 'button_%s.png' % obj.typename
                if action.pixmapExists(filename):
                    return qt4.QVariant(action.getIcon(filename))

        elif role == qt4.Qt.ToolTipRole:
            # provide tool tip showing description
            if obj.userdescription:
                return qt4.QVariant(obj.userdescription)

        elif role == qt4.Qt.TextColorRole:
            # show disabled looking text if object or any parent is hidden
            hidden = False
            p = obj
            while p is not None:
                if 'hide' in p.settings and p.settings.hide:
                    hidden = True
                    break
                p = p.parent

            # return brush for hidden widget text, based on disabled text
            if hidden:
                return qt4.QVariant(qt4.QPalette().brush(qt4.QPalette.Disabled,
                                                         qt4.QPalette.Text))

        # return nothing
        return qt4.QVariant()

    def setData(self, index, value, role):
        """User renames object. This renames the widget."""
        
        widget = index.internalPointer()
        name = unicode(value.toString())

        # check symbols in name
        if not utils.validateWidgetName(name):
            return False
        
        # check name not already used
        if widget.parent.hasChild(name):
            return False

        # actually rename the widget
        self.document.applyOperation(
            document.OperationWidgetRename(widget, name))

        self.emit( qt4.SIGNAL('dataChanged(const QModelIndex &, const QModelIndex &)'), index, index )
        return True
            
    def flags(self, index):
        """What we can do with the item."""
        
        if not index.isValid():
            return qt4.Qt.ItemIsEnabled

        flags = qt4.Qt.ItemIsEnabled | qt4.Qt.ItemIsSelectable
        if index.internalPointer() is not self.document.basewidget and index.column() == 0:
            # allow items other than root to be edited
            flags |= qt4.Qt.ItemIsEditable
        return flags

    def headerData(self, section, orientation, role):
        """Return the header of the tree."""
        
        if orientation == qt4.Qt.Horizontal and role == qt4.Qt.DisplayRole:
            val = ('Name', 'Type')[section]
            return qt4.QVariant(val)

        return qt4.QVariant()

    def _getChildren(self, parent):
        """Get a list of children for the parent given (None selects root)."""

        if parent is None:
            return [self.document.basewidget]
        else:
            return parent.children

    def index(self, row, column, parent):
        """Construct an index for a child of parent."""

        if not parent.isValid():
            parentobj = None
        else:
            parentobj = parent.internalPointer()

        children = self._getChildren(parentobj)

        try:
            c = children[row]
        except IndexError:
            # sometimes this function gets called with an invalid row
            # when deleting, so we return an error result
            return qt4.QModelIndex()

        return self.createIndex(row, column, c)

    def getWidgetIndex(self, widget):
        """Returns index for widget specified."""

        # walk index tree back to widget from root
        widgetlist = []
        w = widget
        while w is not None:
            widgetlist.append(w)
            w = w.parent

        # now iteratively look up indices
        parent = qt4.QModelIndex()
        while widgetlist:
            w = widgetlist.pop()
            row = self._getChildren(w.parent).index(w)
            parent = self.index(row, 0, parent)

        return parent
    
    def parent(self, index):
        """Find the parent of the index given."""

        if not index.isValid():
            return qt4.QModelIndex()

        thisobj = index.internalPointer()
        parentobj = thisobj.parent

        if parentobj is None:
            return qt4.QModelIndex()
        else:
            # lookup parent in grandparent's children
            grandparentchildren = self._getChildren(parentobj.parent)
            parentrow = grandparentchildren.index(parentobj)

            return self.createIndex(parentrow, 0, parentobj)

    def rowCount(self, parent):
        """Return number of rows of children."""

        if not parent.isValid():
            parentobj = None
        else:
            parentobj = parent.internalPointer()

        children = self._getChildren(parentobj)
        return len(children)

    def getSettings(self, index):
        """Return the settings for the index selected."""

        obj = index.internalPointer()
        return obj.settings

    def getWidget(self, index):
        """Get associated widget for index selected."""
        obj = index.internalPointer()

        return obj

class PropertyList(qt4.QWidget):
    """Edit the widget properties using a set of controls."""

    def __init__(self, document, showsubsettings=True,
                 *args):
        qt4.QWidget.__init__(self, *args)
        self.document = document
        self.showsubsettings = showsubsettings

        self.layout = qt4.QGridLayout(self)

        self.layout.setSpacing( self.layout.spacing()/2 )
        self.layout.setMargin(4)
        
        self.children = []

    def updateProperties(self, settings, title=False, showformatting=True):
        """Update the list of controls with new ones for the settings."""

        # delete all child widgets
        self.setUpdatesEnabled(False)
        while len(self.children) > 0:
            self.children.pop().deleteLater()

        if settings is None:
            self.setUpdatesEnabled(True)
            return

        row = 0
        self.layout.setEnabled(False)

        # add a title if requested
        if title:
            lab = qt4.QLabel(settings.usertext)
            lab.setFrameShape(qt4.QFrame.Panel)
            lab.setFrameShadow(qt4.QFrame.Sunken)
            lab.setToolTip(settings.descr)
            self.layout.addWidget(lab, row, 0, 1, -1)
            row += 1

        # add actions if parent is widget
        if settings.parent.isWidget():
            widget = settings.parent
            for action in widget.actions:
                text = action.name
                if action.usertext:
                    text = action.usertext

                lab = qt4.QLabel(text, self)
                self.layout.addWidget(lab, row, 0)
                self.children.append(lab)

                button = qt4.QPushButton(text, self)
                button.setToolTip(action.descr)
                # need to save reference to caller object
                button.caller = utils.BoundCaller(self.slotActionPressed,
                                                  action)
                self.connect(button, qt4.SIGNAL('clicked()'), button.caller)
                             
                self.layout.addWidget(button, row, 1)
                self.children.append(button)

                row += 1

        # add subsettings if necessary
        if settings.getSettingsList() and self.showsubsettings:
            tabbed = TabbedFormatting(self.document, settings, self)
            self.layout.addWidget(tabbed, row, 1, 1, 2)
            row += 1
            self.children.append(tabbed)

        # add settings proper
        for setn in settings.getSettingList():
            # skip if not to show formatting
            if not showformatting and setn.formatting:
                continue

            lab = SettingLabel(self.document, setn, self)
            self.layout.addWidget(lab, row, 0)
            self.children.append(lab)

            cntrl = setn.makeControl(self)
            self.connect(cntrl, qt4.SIGNAL('settingChanged'),
                         self.slotSettingChanged)
            self.layout.addWidget(cntrl, row, 1)
            self.children.append(cntrl)

            row += 1

        # add empty widget to take rest of space
        w = qt4.QWidget(self)
        w.setSizePolicy(qt4.QSizePolicy.Maximum,
                        qt4.QSizePolicy.MinimumExpanding)
        self.layout.addWidget(w, row, 0)
        self.children.append(w)

        self.setUpdatesEnabled(True)
        self.layout.setEnabled(True)
 
    def slotSettingChanged(self, widget, setting, val):
        """Called when a setting is changed by the user.
        
        This updates the setting to the value using an operation so that
        it can be undone.
        """
        
        self.document.applyOperation(document.OperationSettingSet(setting, val))
        
    def slotActionPressed(self, action):
        """Activate the action."""

        # find console window, this is horrible: HACK
        win = self
        while not hasattr(win, 'console'):
            win = win.parent()
        console = win.console

        console.runFunction(action.function)

class TabbedFormatting(qt4.QTabWidget):
    """Class to have tabbed set of settings."""

    def __init__(self, document, settings, *args):
        qt4.QTabWidget.__init__(self, *args)

        if settings is None:
            return

        setnslist = settings.getSettingsList()

        # make a temporary list of formatting settings
        formatters = setting.Settings('Basic', descr='Basic formatting options',
                                      usertext='Basic', pixmap='main')
        formatters.parent = settings.parent
        for setn in settings.getSettingList():
            if setn.formatting:
                formatters.add(setn)
        if formatters.getSettingList():
            setnslist.insert(0, formatters)

        # add tab for each subsettings
        for subset in setnslist:

            # create tab
            tab = qt4.QWidget()
            layout = qt4.QVBoxLayout()
            layout.setMargin(2)
            tab.setLayout(layout)

            # create scrollable area
            scroll = qt4.QScrollArea(tab)
            layout.addWidget(scroll)
            scroll.setWidgetResizable(True)

            # create list of properties
            plist = PropertyList(document)
            plist.updateProperties(subset, title=True)
            scroll.setWidget(plist)
            plist.show()

            # add tab to widget
            if hasattr(subset, 'pixmap'):
                icon = action.getIcon('settings_%s.png' % subset.pixmap)
                indx = self.addTab(tab, icon, '')
                text = subset.usertext
                if not subset.usertext:
                    text = subset.name
                self.setTabToolTip(indx, text)
            else:
                self.addTab(tab, subset.name)

class FormatDock(qt4.QDockWidget):
    """A window for formatting the current widget.
    Provides tabbed formatting properties
    """

    def __init__(self, document, treeedit, *args):
        qt4.QDockWidget.__init__(self, *args)
        self.setWindowTitle("Formatting - Veusz")
        self.setObjectName("veuszformattingdock")

        self.document = document
        self.tabwidget = None

        # update our view when the tree edit window selection changes
        self.connect(treeedit, qt4.SIGNAL('widgetSelected'),
                     self.selectedWidget)

        # do initial selection
        self.selectedWidget(treeedit.selwidget)

    def selectedWidget(self, widget):
        """Created tabbed widget for formatting for each subsettings."""

        # get current tab (so we can set it afterwards)
        if self.tabwidget:
            tab = self.tabwidget.currentIndex()
        else:
            tab = 0

        # delete old tabwidget
        if self.tabwidget:
            self.tabwidget.deleteLater()
            self.tabwidget = None

        # create new tabbed widget showing formatting
        settings = None
        if widget is not None:
            settings = widget.settings

        self.tabwidget = TabbedFormatting(self.document, settings, self)
        self.setWidget(self.tabwidget)

        # wrap tab from zero to max number
        tab = max( min(self.tabwidget.count()-1, tab), 0 )
        self.tabwidget.setCurrentIndex(tab)

class PropertiesDock(qt4.QDockWidget):
    """A window for editing properties for widgets."""

    def __init__(self, document, treeedit, *args):
        qt4.QDockWidget.__init__(self, *args)
        self.setWindowTitle("Properties - Veusz")
        self.setObjectName("veuszpropertiesdock")

        self.document = document

        # update our view when the tree edit window selection changes
        self.connect(treeedit, qt4.SIGNAL('widgetSelected'),
                     self.slotWidgetSelected)

        # construct scrollable area
        self.scroll = qt4.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.setWidget(self.scroll)

        # construct properties list in scrollable area
        self.proplist = PropertyList(document, showsubsettings=False)
        self.scroll.setWidget(self.proplist)

        # do initial selection
        self.slotWidgetSelected(treeedit.selwidget)

    def slotWidgetSelected(self, widget):
        """Update properties when selected widget changes."""

        settings = None
        if widget is not None:
            settings = widget.settings
        self.proplist.updateProperties(settings, showformatting=False)

class TreeEditDock(qt4.QDockWidget):
    """A window for editing the document as a tree."""

    # mime type when widgets are stored on the clipboard
    widgetmime = 'text/x-vnd.veusz2-clipboard'

    def __init__(self, document, parent):
        qt4.QDockWidget.__init__(self, parent)
        self.parent = parent
        self.setWindowTitle("Editing - Veusz")
        self.setObjectName("veuszeditingwindow")
        self.selwidget = None

        self.document = document
        self.connect( self.document, qt4.SIGNAL("sigWiped"),
                      self.slotDocumentWiped )

        # construct tree
        self.treemodel = WidgetTreeModel(document)
        self.treeview = qt4.QTreeView()
        self.treeview.setModel(self.treemodel)
        try:
            self.treeview.expandAll()
        except AttributeError: # only in Qt-4.2
            pass

        # make 1st column stretch
        hdr = self.treeview.header()
        hdr.setStretchLastSection(False)
        hdr.setResizeMode(0, qt4.QHeaderView.Stretch)
        hdr.setResizeMode(1, qt4.QHeaderView.Custom)

        # receive change in selection
        self.connect(self.treeview.selectionModel(),
                     qt4.SIGNAL('selectionChanged(const QItemSelection &, const QItemSelection &)'),
                     self.slotTreeItemSelected)

        # set tree as main widget
        self.setWidget(self.treeview)

        # toolbar to create widgets, etc
        self.toolbar = qt4.QToolBar("Editing toolbar - Veusz",
                                    parent)
        self.toolbar.setObjectName("veuszeditingtoolbar")
        self.toolbar.setOrientation(qt4.Qt.Vertical)
        self._constructToolbarMenu()
        parent.addToolBarBreak(qt4.Qt.TopToolBarArea)
        parent.addToolBar(qt4.Qt.TopToolBarArea, self.toolbar)

        # this sets various things up
        self.selectWidget(document.basewidget)

        # update paste button when clipboard changes
        self.connect(qt4.QApplication.clipboard(),
                     qt4.SIGNAL('dataChanged()'),
                     self.updatePasteButton)

    def slotDocumentWiped(self):
        """If the document is wiped, reselect root widget."""
        self.selectWidget(self.document.basewidget)

    def slotTreeItemSelected(self, current, previous):
        """New item selected in tree.

        This updates the list of properties
        """
        
        indexes = current.indexes()

        if len(indexes) > 1:
            index = indexes[0]
            self.selwidget = self.treemodel.getWidget(index)
            settings = self.treemodel.getSettings(index)
        else:
            self.selwidget = None
            settings = None

        self._enableCorrectButtons()
        self._checkPageChange()

        self.emit( qt4.SIGNAL('widgetSelected'), self.selwidget )

    def contextMenuEvent(self, event):
        """Bring up context menu."""

        m = qt4.QMenu(self)
        for act in ('cut', 'copy', 'paste',
                    'moveup', 'movedown', 'delete', 'rename'):
            m.addAction(self.editactions[act])

        # allow show or hides of selected widget
        if self.selwidget and 'hide' in self.selwidget.settings:
            m.addSeparator()
            hide = self.selwidget.settings.hide
            act = qt4.QAction( ('Hide object', 'Show object')[hide], m )
            self.connect(act, qt4.SIGNAL('triggered()'),
                         (self.slotWidgetHide, self.slotWidgetShow)[hide])
            m.addAction(act)

        m.exec_(self.mapToGlobal(event.pos()))

        event.accept()

    def _checkPageChange(self):
        """Check to see whether page has changed."""

        w = self.selwidget
        while w is not None and not isinstance(w, widgets.Page):
            w = w.parent

        if w is not None:
            # have page, so check what number we are in basewidget children
            try:
                i = self.document.basewidget.children.index(w)
                self.emit(qt4.SIGNAL("sigPageChanged"), i)
            except ValueError:
                pass

    def _enableCorrectButtons(self):
        """Make sure the create graph buttons are correctly enabled."""

        selw = self.selwidget

        # check whether each button can have this widget
        # (or a parent) as parent

        menu = self.parent.menus['insert']
        for wc, action in self.addslots.iteritems():
            w = selw
            while w is not None and not wc.willAllowParent(w):
                w = w.parent

            self.addactions['add%s' % wc.typename].setEnabled(w is not None)

        # certain actions shouldn't allow root to be deleted
        isnotroot = not isinstance(selw, widgets.Root)

        for act in ('cut', 'copy', 'delete', 'moveup', 'movedown', 'rename'):
            self.editactions[act].setEnabled(isnotroot)

        self.updatePasteButton()

    def _constructToolbarMenu(self):
        """Add items to edit/add graph toolbar and menu."""

        self.toolbar.setIconSize( qt4.QSize(16, 16) )

        self.addslots = {}
        actions = {}
        for widgettype in ('page', 'grid', 'graph', 'axis',
                           'xy', 'fit', 'function',
                           'image', 'contour',
                           'key', 'label', 'colorbar',
                           'rect', 'ellipse', 'roundrect'):

            wc = document.thefactory.getWidgetClass(widgettype)
            slot = utils.BoundCaller(self.slotMakeWidgetButton, wc)
            self.addslots[wc] = slot
            val = ( 'add%s' % widgettype, wc.description,
                    'Add %s' % widgettype, 'insert',
                    slot,
                    'button_%s.png' % widgettype,
                    True, '')
            actions[widgettype] = val

        # add non-shape widgets to toolbar and menu
        self.addactions = action.populateMenuToolbars(
            [actions[wt] for wt in
             ('page', 'grid', 'graph', 'axis',
              'xy', 'fit', 'function',
              'image', 'contour',
              'key', 'label', 'colorbar')],
            self.toolbar, self.parent.menus)

        # create shape toolbar button
        shapetb = qt4.QToolButton(self.toolbar)
        shapetb.setIcon( action.getIcon('veusz-shape-menu.png') )
        shapepop = qt4.QMenu(shapetb)
        shapetb.setPopupMode(qt4.QToolButton.InstantPopup)
        self.toolbar.addWidget(shapetb)

        # create menu item for shapes
        shapemenu = qt4.QMenu('Add shape', self.parent.menus['insert'])
        shapemenu.setIcon( action.getIcon('veusz-shape-menu.png') )
        self.parent.menus['insert'].addMenu(shapemenu)

        # add shape items to menu and toolbar button
        shapeacts = action.populateMenuToolbars(
            [actions[wt] for wt in
             ('rect', 'roundrect', 'ellipse')],
            shapetb, {'insert': shapemenu})
        self.addactions.update(shapeacts)

        self.toolbar.addSeparator()

        # make buttons and menu items for the various item editing ops
        moveup = utils.BoundCaller(self.slotWidgetMove, -1)
        movedown = utils.BoundCaller(self.slotWidgetMove, 1)
        self.editslots = [moveup, movedown]

        edititems = (
            ('cut', 'Cut the selected item', 'Cu&t', 'edit',
             self.slotWidgetCut, 'stock-cut.png', True, 'Ctrl+X'),
            ('copy', 'Copy the selected item', '&Copy', 'edit',
             self.slotWidgetCopy, 'stock-copy.png', True, 'Ctrl+C'),
            ('paste', 'Paste item from the clipboard', '&Paste', 'edit',
             self.slotWidgetPaste, 'stock-paste.png', True, 'Ctrl+V'),
            ('moveup', 'Move the selected item up', 'Move &up', 'edit',
             moveup, 'stock-go-up.png',
             True, ''),
            ('movedown', 'Move the selected item down', 'Move d&own', 'edit',
             movedown, 'stock-go-down.png',
             True, ''),
            ('delete', 'Remove the selected item', '&Delete', 'edit',
             self.slotWidgetDelete, 'stock-delete.png', True, ''),
            ('rename', 'Renames the selected item', '&Rename', 'edit',
             self.slotWidgetRename, 'icon-rename.png', False, '')
            )
        self.editactions = action.populateMenuToolbars(edititems, self.toolbar,
                                                       self.parent.menus)

    def slotMakeWidgetButton(self, wc):
        """User clicks button to make widget."""
        self.makeWidget(wc.typename)

    def makeWidget(self, widgettype, autoadd=True, name=None):
        """Called when an add widget button is clicked.
        widgettype is the type of widget
        autoadd specifies whether to add default children
        if name is set this name is used if possible (ie no other children have it)
        """

        # if no widget selected, bomb out
        if self.selwidget is None:
            return
        parent = self.getSuitableParent(widgettype)

        assert parent is not None

        if name in parent.childnames:
            name = None
        
        # make the new widget and update the document
        w = self.document.applyOperation( document.OperationWidgetAdd(parent, widgettype, autoadd=autoadd,
                                                                      name=name) )

        # select the widget
        self.selectWidget(w)

    def getSuitableParent(self, widgettype, initialParent = None):
        """Find the nearest relevant parent for the widgettype given."""

        # get the widget to act as the parent
        if not initialParent:
            parent = self.selwidget
        else:
            parent  = initialParent
        
        # find the parent to add the child to, we go up the tree looking
        # for possible parents
        wc = document.thefactory.getWidgetClass(widgettype)
        while parent is not None and not wc.willAllowParent(parent):
            parent = parent.parent

        return parent

    def slotWidgetCut(self):
        """Cut the selected widget"""
        self.slotWidgetCopy()
        self.slotWidgetDelete()

    def slotWidgetCopy(self):
        """Copy selected widget to the clipboard."""

        mimedata = self._makeMimeData(self.selwidget)
        if mimedata:
            clipboard = qt4.QApplication.clipboard()
            clipboard.setMimeData(mimedata)

    def _makeMimeData(self, widget):
        """Make a QMimeData object representing the subtree with the
        current selection at the root"""

        if widget:
            mimedata = qt4.QMimeData()
            text = str('\n'.join((widget.typename,
                                  widget.name,
                                  widget.getSaveText())))
            self.mimedata = qt4.QByteArray(text)
            mimedata.setData(self.widgetmime, self.mimedata)
            return mimedata
        else:
            return None

    def getClipboardData(self):
        """Return the clipboard data if it is in the correct format."""

        mimedata = qt4.QApplication.clipboard().mimeData()
        if self.widgetmime in mimedata.formats():
            data = unicode(mimedata.data(self.widgetmime)).split('\n')
            return data
        else:
            return None

    def updatePasteButton(self):
        """Is the data on the clipboard a valid paste at the currently
        selected widget? If so, enable paste button"""

        data = self.getClipboardData()
        show = False
        if data:
            # The first line of the clipboard data is the widget type
            widgettype = data[0]
            # Check if we can paste into the current widget or a parent
            if self.getSuitableParent(widgettype, self.selwidget):
                show = True

        self.editactions['paste'].setEnabled(show)

    def slotWidgetPaste(self):
        """Paste something from the clipboard"""

        data = self.getClipboardData()
        if data:
            # The first line of the clipboard data is the widget type
            widgettype = data[0]
            # The second is the original name
            widgetname = data[1]

            # make the document enter batch mode
            # This is so that the user can undo this in one step
            op = document.OperationMultiple([], descr='paste')
            self.document.applyOperation(op)
            self.document.batchHistory(op)
            
            # Add the first widget being pasted
            self.makeWidget(widgettype, autoadd=False, name=widgetname)
            
            interpreter = self.parent.interpreter
        
            # Select the current widget in the interpreter
            tmpCurrentwidget = interpreter.interface.currentwidget
            interpreter.interface.currentwidget = self.selwidget

            # Use the command interface to create the subwidgets
            for command in data[2:]:
                interpreter.run(command)
                
            # stop the history batching
            self.document.batchHistory(None)
                
            # reset the interpreter widget
            interpreter.interface.currentwidget = tmpCurrentwidget
            
    def slotWidgetDelete(self):
        """Delete the widget selected."""

        # no item selected, so leave
        w = self.selwidget
        if w is None:
            return

        # get list of widgets in order
        widgetlist = []
        self.document.basewidget.buildFlatWidgetList(widgetlist)
        
        widgetnum = widgetlist.index(w)
        assert widgetnum >= 0

        # delete selected widget
        self.document.applyOperation( document.OperationWidgetDelete(w) )

        # rebuild list
        widgetlist = []
        self.document.basewidget.buildFlatWidgetList(widgetlist)

        # find next to select
        if widgetnum < len(widgetlist):
            nextwidget = widgetlist[widgetnum]
        else:
            nextwidget = widgetlist[-1]

        # select the next widget
        self.selectWidget(nextwidget)

    def slotWidgetRename(self):
        """Allows the user to rename the selected widget."""

        selected = self.treeview.selectedIndexes()
        if len(selected) != 0:
            self.treeview.edit(selected[0])

    def selectWidget(self, widget):
        """Select the associated listviewitem for the widget w in the
        listview."""

        index = self.treemodel.getWidgetIndex(widget)
        self.treeview.scrollTo(index)
        self.treeview.setCurrentIndex(index)

    def slotWidgetMove(self, direction):
        """Move the selected widget up/down in the hierarchy.

        a is the action (unused)
        direction is -1 for 'up' and +1 for 'down'
        """

        # widget to move
        w = self.selwidget
        
        # actually move the widget
        self.document.applyOperation(
            document.OperationWidgetMove(w, direction) )

        # rehilight moved widget
        self.selectWidget(w)

    def slotWidgetHide(self):
        """Hide the selected widget."""
        self.document.applyOperation(
            document.OperationSettingSet(self.selwidget.settings.get('hide'),
                                         True) )
    def slotWidgetShow(self):
        """Show the selected widget."""
        self.document.applyOperation(
            document.OperationSettingSet(self.selwidget.settings.get('hide'),
                                         False) )
        

class SettingLabel(qt4.QWidget):
    """A label to describe a setting.

    This widget shows the name, a tooltip description, and gives
    access to the context menu
    """
    
    def __init__(self, document, setting, parent):
        """Initialise button, passing document, setting, and parent widget."""
        
        qt4.QWidget.__init__(self, parent)
        self.setFocusPolicy(qt4.Qt.StrongFocus)

        self.document = document
        self.connect(document, qt4.SIGNAL('sigModified'), self.slotDocModified)

        self.setting = setting

        self.layout = qt4.QHBoxLayout(self)
        self.layout.setMargin(2)

        if setting.usertext:
            text = setting.usertext
        else:
            text = setting.name
        self.labelicon = qt4.QLabel(text, self)
        self.layout.addWidget(self.labelicon)
        
        self.iconlabel = qt4.QLabel(self)
        self.layout.addWidget(self.iconlabel)

        self.connect(self, qt4.SIGNAL('clicked'), self.settingMenu)

        self.infocus = False
        self.inmouse = False
        self.inmenu = False

        # initialise settings
        self.slotDocModified()

    def mouseReleaseEvent(self, event):
        """Emit clicked(pos) on mouse release."""
        self.emit( qt4.SIGNAL('clicked'),
                   self.mapToGlobal(event.pos()) )
        return qt4.QWidget.mouseReleaseEvent(self, event)

    def keyReleaseEvent(self, event):
        """Emit clicked(pos) on key release."""
        if event.key() == qt4.Qt.Key_Space:
            self.emit( qt4.SIGNAL('clicked'),
                       self.mapToGlobal(self.iconlabel.pos()) )
            event.accept()
        else:
            return qt4.QWidget.keyReleaseEvent(self, event)

    def slotDocModified(self):
        """If the document has been modified."""

        # update pixmap (e.g. link added/removed)
        self.updateHighlight()

        # update tooltip
        tooltip = self.setting.descr
        if self.setting.isReference():
            tooltip += ('\nLinked to %s' %
                        self.setting.getReference().resolve(self.setting).path)
        self.setToolTip(tooltip)

    def updateHighlight(self):
        """Show drop down arrow if item has focus."""
        if self.inmouse or self.infocus or self.inmenu:
            pixmap = 'downarrow.png'
        else:
            if self.setting.isReference():
                pixmap = 'link.png'
            else:
                pixmap = 'downarrow_blank.png'
        self.iconlabel.setPixmap(action.getPixmap(pixmap))

    def enterEvent(self, event):
        """Focus on mouse enter."""
        self.inmouse = True
        self.updateHighlight()
        return qt4.QWidget.enterEvent(self, event)

    def leaveEvent(self, event):
        """Clear focus on mouse leaving."""
        self.inmouse = False
        self.updateHighlight()
        return qt4.QWidget.leaveEvent(self, event)

    def focusInEvent(self, event):
        """Focus if widgets gets focus."""
        self.infocus = True
        self.updateHighlight()
        return qt4.QWidget.focusInEvent(self, event)

    def focusOutEvent(self, event):
        """Lose focus if widget loses focus."""
        self.infocus = False
        self.updateHighlight()
        return qt4.QWidget.focusOutEvent(self, event)

    def settingMenu(self, pos):
        """Pop up menu for each setting."""

        # forces settings to be updated
        self.parentWidget().setFocus()
        # get it back straight away
        self.setFocus()

        # get widget, with its type and name
        widget = self.setting.parent
        while not isinstance(widget, widgets.Widget):
            widget = widget.parent
        self._clickwidget = widget

        wtype = widget.typename
        name = widget.name

        popup = qt4.QMenu(self)
        popup.addAction('Reset %s to default' % self.setting.name,
                        self.actionResetDefault)
        popup.addSeparator()
        popup.addAction('Copy to "%s" widgets' % wtype,
                        self.actionCopyTypedWidgets)
        popup.addAction('Copy to "%s" siblings' % wtype,
                        self.actionCopyTypedSiblings)
        popup.addAction('Copy to "%s" widgets called "%s"' % (wtype, name),
                        self.actionCopyTypedNamedWidgets)
        popup.addSeparator()
        popup.addAction('Make default for "%s" widgets' % wtype,
                        self.actionDefaultTyped)
        popup.addAction('Make default for "%s" widgets called "%s"' %
                        (wtype, name),
                        self.actionDefaultTypedNamed)
        popup.addAction('Forget this default setting',
                        self.actionDefaultForget)

        # special actions for references
        if self.setting.isReference():
            popup.addSeparator()
            popup.addAction('Unlink setting', self.actionUnlinkSetting)
            #popup.addAction('Edit linked setting',
            #                self.actionEditLinkedSetting)


        self.inmenu = True
        self.updateHighlight()
        popup.exec_(pos)
        self.inmenu = False
        self.updateHighlight()

    def actionResetDefault(self):
        """Reset setting to default."""
        self.document.applyOperation(
            document.OperationSettingSet(self.setting, self.setting.default) )

    def actionCopyTypedWidgets(self):
        """Copy setting to widgets of same type."""
        self.document.applyOperation(
            document.OperationSettingPropagate(self.setting) )

    def actionCopyTypedSiblings(self):
        """Copy setting to siblings of the same type."""
        self.document.applyOperation(
            document.OperationSettingPropagate(self.setting,
                                               root=self._clickwidget.parent,
                                               maxlevels=1) )

    def actionCopyTypedNamedWidgets(self):
        """Copy setting to widgets with the same name and type."""
        self.document.applyOperation(
            document.OperationSettingPropagate(self.setting,
                                               widgetname=
                                               self._clickwidget.name) )

    def actionDefaultTyped(self):
        """Make default for widgets with the same type."""
        self.setting.setAsDefault(False)

    def actionDefaultTypedNamed(self):
        """Make default for widgets with the same name and type."""
        self.setting.setAsDefault(True)

    def actionDefaultForget(self):
        """Forget any default setting."""
        self.setting.removeDefault()

    def actionUnlinkSetting(self):
        """Unlink the setting if it is a reference."""
        self.document.applyOperation(
            document.OperationSettingSet(self.setting, self.setting.get()) )

    def actionEditLinkedSetting(self):
        """Edit the linked setting rather than the setting."""
        
        realsetn = self.setting.getReference().resolve(self.setting)
        widget = realsetn
        while not isinstance(widget, widgets.Widget) and widget is not None:
            widget = widget.parent

        # need to select widget, so need to find treeditwindow :-(
        window = self
        while not hasattr(window, 'treeedit'):
            window = window.parent()
        window.treeedit.selectWidget(widget)
