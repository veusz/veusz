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
            val = ['Name', 'Type', 'Detail'][section]
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

        c = children[row]
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

    def __init__(self, document, showsubsettings=True, *args):
        qt4.QWidget.__init__(self, *args)
        self.document = document
        self.showsubsettings = showsubsettings

        self.layout = qt4.QGridLayout(self)

        self.layout.setSpacing( self.layout.spacing()/2 )
        self.layout.setMargin(4)
        
        self.children = []

    def updateProperties(self, settings):
        """Update the list of controls with new ones for the settings."""

        # delete all child widgets
        self.setUpdatesEnabled(False)
        while len(self.children) > 0:
            self.children.pop().deleteLater()

        if settings is None:
            self.setUpdatesEnabled(True)
            return

        row = 0
        # FIXME: add actions

        self.layout.setEnabled(False)
        # add subsettings if necessary
        if settings.getSettingsList() and self.showsubsettings:
            tabbed = TabbedFormatting(self.document, settings, self)
            self.layout.addWidget(tabbed, row, 1, 1, 2)
            row += 1
            self.children.append(tabbed)

        for setn in settings.getSettingList():
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
        
class TabbedFormatting(qt4.QTabWidget):
    """Class to have tabbed set of settings."""

    def __init__(self, document, settings, *args):
        qt4.QTabWidget.__init__(self, *args)

        if settings is None:
            return

        # add tab for each subsettings
        for subset in settings.getSettingsList():

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
            plist.updateProperties(subset)
            scroll.setWidget(plist)
            plist.show()

            # add tab to widget
            if hasattr(subset, 'pixmap'):
                icon = action.getIcon('settings_%s.png' % subset.pixmap)
                indx = self.addTab(tab, icon, '')
                self.setTabToolTip(indx, subset.name)
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
                     self.selectWidget)

        # do initial selection
        self.selectWidget(treeedit.selwidget)

    def selectWidget(self, widget):
        """Created tabbed widget for formatting for each subsettings."""

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

class PropertiesDock(qt4.QDockWidget):
    """A window for editing properties for widgets."""

    def __init__(self, document, treeedit, *args):
        qt4.QDockWidget.__init__(self, *args)
        self.setWindowTitle("Properties - Veusz")
        self.setObjectName("veuszpropertiesdock")

        self.document = document

        # update our view when the tree edit window selection changes
        self.connect(treeedit, qt4.SIGNAL('widgetSelected'),
                     self.selectWidget)

        # construct scrollable area
        self.scroll = qt4.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.setWidget(self.scroll)

        # construct properties list in scrollable area
        self.proplist = PropertyList(document, showsubsettings=False)
        self.scroll.setWidget(self.proplist)

        # do initial selection
        self.selectWidget(treeedit.selwidget)

    def selectWidget(self, widget):
        """Update properties when selected widget changes."""

        settings = None
        if widget is not None:
            settings = widget.settings
        self.proplist.updateProperties(settings)

class TreeEditDock(qt4.QDockWidget):
    """A window for editing the document as a tree."""

    # mime type when widgets are stored on the clipboard
    widgetmime = 'text/x-vnd.veusz-clipboard'

    def __init__(self, document, parent):
        qt4.QDockWidget.__init__(self, parent)
        self.parent = parent
        self.setWindowTitle("Editing - Veusz")
        self.setObjectName("veuszeditingwindow")

        # construct tree
        self.document = document
        self.treemodel = WidgetTreeModel(document)
        self.treeview = qt4.QTreeView()
        self.treeview.setModel(self.treemodel)
        #self.treeview.header().

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
        parent.addToolBar(qt4.Qt.LeftToolBarArea, self.toolbar)
        self._constructToolbarMenu()

        # this sets various things up
        self.selectWidget(document.basewidget)

        # update paste button when clipboard changes
        self.connect(qt4.QApplication.clipboard(),
                     qt4.SIGNAL('dataChanged()'),
                     self.updatePasteButton)

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

        for i in ('cut', 'copy', 'delete', 'moveup', 'movedown'):
            self.editactions[i].setEnabled(isnotroot)

        if isnotroot:
            # cut and copy aren't currently possible on a non-widget
            cancopy = selw is not None
            self.editactions['cut'].setEnabled(cancopy)
            self.editactions['copy'].setEnabled(cancopy)
        self.updatePasteButton()

    def _getWidgetOrder(self):
        """Return a list of the widgets, most important first.
        """

        # get list of allowed classes, sorted by type name
        wcl = [(i.typename, i)
               for i in document.thefactory.listWidgetClasses()
               if i.allowusercreation]
        wcl.sort()

        # build up a list of pairs for topological sort
        pairs = []
        for name, wc in wcl:
            for pwc in wc.allowedparenttypes:
                pairs.append( (pwc, wc) )

        # do topological sort
        sorted = utils.topsort(pairs)

        return sorted

    def _constructToolbarMenu(self):
        """Add items to edit/add graph toolbar and menu."""

        self.toolbar.setIconSize( qt4.QSize(16, 16) )

        actions = []
        self.addslots = {}
        for wc in self._getWidgetOrder():
            name = wc.typename
            if wc.allowusercreation:

                slot = utils.BoundCaller(self.slotMakeWidgetButton, wc)
                self.addslots[wc] = slot

                val = ( 'add%s' % name, wc.description,
                        'Add %s' % name, 'insert',
                        slot,
                        'button_%s.png' % name,
                        True, '')
                actions.append(val)
        self.addactions = action.populateMenuToolbars(actions, self.toolbar,
                                                      self.parent.menus)
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
             self.slotWidgetDelete, 'stock-delete.png', True, '')
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
            #clipboard.setText(mimedata)

    def _makeMimeData(self, widget):
        """Make a QMimeData object representing the subtree with the
        current selection at the root"""

        if widget:
            mimedata = qt4.QMimeData()
            text = str('\n'.join((widget.typename,
                                  widget.name,
                                  widget.getSaveText())))
            self.mimedata = qt4.QByteArray(text)
            mimedata.setData('text/plain', self.mimedata)
            #mimedata.setText(text)
            return mimedata
            #return text
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

    def slotWidgetPaste(self, a):
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
            nextwidget = self.document.basewidget

        # select the next widget
        self.selectWidget(nextwidget)

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

class SettingLabel(qt4.QWidget):
    def __init__(self, document, setting, parent):
        """Initialise botton, passing document, setting, and parent widget."""
        
        qt4.QWidget.__init__(self, parent)
        self.setFocusPolicy(qt4.Qt.StrongFocus)

        self.document = document
        self.setting = setting

        self.setToolTip(setting.descr)

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
        self.updateHighlight()

    def mouseReleaseEvent(self, event):
        self.emit( qt4.SIGNAL('clicked'),
                   self.mapToGlobal(event.pos()) )
        return qt4.QWidget.mouseReleaseEvent(self, event)

    def keyReleaseEvent(self, event):
        if event.key() == qt4.Qt.Key_Space:
            self.emit( qt4.SIGNAL('clicked'),
                       self.mapToGlobal(self.iconlabel.pos()) )
            event.accept()
        else:
            return qt4.QWidget.keyReleaseEvent(self, event)

    def updateHighlight(self):
        if self.inmouse or self.infocus or self.inmenu:
            self.iconlabel.setPixmap(action.getPixmap('downarrow.png'))
        else:
            self.iconlabel.setPixmap(action.getPixmap('downarrow_blank.png'))

    def enterEvent(self, event):
        self.inmouse = True
        self.updateHighlight()
        return qt4.QWidget.enterEvent(self, event)

    def leaveEvent(self, event):
        self.inmouse = False
        self.updateHighlight()
        return qt4.QWidget.leaveEvent(self, event)

    def focusInEvent(self, event):
        self.infocus = True
        self.updateHighlight()
        return qt4.QWidget.focusInEvent(self, event)

    def focusOutEvent(self, event):
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
        self.inmenu = True
        self.updateHighlight()
        popup.exec_(pos)
        self.inmenu = False
        self.updateHighlight()

    def actionResetDefault(self):
        self.document.applyOperation( document.OperationSettingSet(self.setting, self.setting.default) )

    def actionCopyTypedWidgets(self):
        self.document.applyOperation( document.OperationSettingPropagate(self.setting) )

    def actionCopyTypedSiblings(self):
        self.document.applyOperation( document.OperationSettingPropagate(self.setting, root=self._clickwidget.parent, maxlevels=1) )

    def actionCopyTypedNamedWidgets(self):
        self.document.applyOperation( document.OperationSettingPropagate(self.setting, widgetname=self._clickwidget.name) )

    def actionDefaultTyped(self):
        self.setting.setAsDefault(False)

    def actionDefaultTypedNamed(self):
        self.setting.setAsDefault(True)

    def actionDefaultForget(self):
        self.setting.removeDefault()

