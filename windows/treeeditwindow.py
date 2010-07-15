#    Copyright (C) 2004 Jeremy S. Sanders
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

# $Id$

"""Window to edit the document using a tree, widget properties
and formatting properties."""

import veusz.qtall as qt4

import veusz.widgets as widgets
import veusz.utils as utils
import veusz.document as document
import veusz.setting as setting

from widgettree import WidgetTreeModel, WidgetTreeView

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
        
        self.childlist = []

    def updateProperties(self, settings, title=None, showformatting=True,
                         onlyformatting=False):
        """Update the list of controls with new ones for the settings."""

        # delete all child widgets
        self.setUpdatesEnabled(False)

        while len(self.childlist) > 0:
            c = self.childlist.pop()
            self.layout.removeWidget(c)
            c.deleteLater()
            del c

        if settings is None:
            self.setUpdatesEnabled(True)
            return

        row = 0
        self.layout.setEnabled(False)

        # add a title if requested
        if title is not None:
            lab = qt4.QLabel(title[0])
            lab.setFrameShape(qt4.QFrame.Panel)
            lab.setFrameShadow(qt4.QFrame.Sunken)
            lab.setToolTip(title[1])
            self.layout.addWidget(lab, row, 0, 1, -1)
            row += 1

        # add actions if parent is widget
        if settings.parent.isWidget() and not showformatting:
            widget = settings.parent
            for action in widget.actions:
                text = action.name
                if action.usertext:
                    text = action.usertext

                lab = qt4.QLabel(text)
                self.layout.addWidget(lab, row, 0)
                self.childlist.append(lab)

                button = qt4.QPushButton(text)
                button.setToolTip(action.descr)
                # need to save reference to caller object
                button.caller = utils.BoundCaller(self.slotActionPressed,
                                                  action)
                self.connect(button, qt4.SIGNAL('clicked()'), button.caller)
                             
                self.layout.addWidget(button, row, 1)
                self.childlist.append(button)

                row += 1

        if settings.getSettingsList() and self.showsubsettings:
            # if we have subsettings, use tabs
            tabbed = TabbedFormatting(self.document, settings)
            self.layout.addWidget(tabbed, row, 1, 1, 2)
            row += 1
            self.childlist.append(tabbed)
        else:
            # else add settings proper as a list
            for setn in settings.getSettingList():
                # skip if not to show formatting
                if not showformatting and setn.formatting:
                    continue
                # skip if only to show formatting and not formatting
                if onlyformatting and not setn.formatting:
                    continue

                cntrl = setn.makeControl(None)
                if cntrl:
                    lab = SettingLabel(self.document, setn, None)
                    self.layout.addWidget(lab, row, 0)
                    self.childlist.append(lab)

                    self.connect(cntrl, qt4.SIGNAL('settingChanged'),
                                 self.slotSettingChanged)
                    self.layout.addWidget(cntrl, row, 1)
                    self.childlist.append(cntrl)
                    
                    row += 1

        # add empty widget to take rest of space
        w = qt4.QWidget()
        w.setSizePolicy(qt4.QSizePolicy.Maximum,
                        qt4.QSizePolicy.MinimumExpanding)
        self.layout.addWidget(w, row, 0)
        self.childlist.append(w)

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

    def __init__(self, document, settings, shownames=False):
        qt4.QTabWidget.__init__(self)

        if settings is None:
            return

        # get list of settings
        setnslist = settings.getSettingsList()

        # add formatting settings if necessary
        numformat = len( [setn for setn in settings.getSettingList()
                          if setn.formatting] )
        if numformat > 0:
            # add on a formatting tab
            setnslist.insert(0, settings)

        # add tab for each subsettings
        for subset in setnslist:
            if subset.name == 'StyleSheet':
                continue

            # create tab
            tab = qt4.QWidget()
            layout = qt4.QVBoxLayout()
            layout.setMargin(2)
            tab.setLayout(layout)

            # create scrollable area
            scroll = qt4.QScrollArea(None)
            layout.addWidget(scroll)
            scroll.setWidgetResizable(True)

            # details of tab
            mainsettings = (subset == settings)
            if mainsettings:
                # main tab formatting, so this is special
                pixmap = 'settings_main'
                tabname = title = 'Main'
                tooltip = 'Main formatting'
            else:
                # others
                if hasattr(subset, 'pixmap'):
                    pixmap = subset.pixmap
                else:
                    pixmap = None
                tabname = subset.name
                tooltip = title = subset.usertext
                
            # create list of properties
            plist = PropertyList(document, showsubsettings=not mainsettings)
            plist.updateProperties(subset, title=(title, tooltip),
                                   onlyformatting=mainsettings)
            scroll.setWidget(plist)
            plist.show()

            # hide name in tab
            if not shownames:
                tabname = ''

            indx = self.addTab(tab, utils.getIcon(pixmap), tabname)
            self.setTabToolTip(indx, tooltip)

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

        self.tabwidget = TabbedFormatting(self.document, settings)
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
        self.treeview = WidgetTreeView(self.treemodel)

        # receive change in selection
        self.connect(self.treeview.selectionModel(),
                     qt4.SIGNAL('selectionChanged(const QItemSelection &, const QItemSelection &)'),
                     self.slotTreeItemSelected)

        # set tree as main widget
        self.setWidget(self.treeview)

        # toolbar to create widgets
        self.addtoolbar = qt4.QToolBar("Insert toolbar - Veusz",
                                       parent)
        # note wrong description!: backwards compatibility
        self.addtoolbar.setObjectName("veuszeditingtoolbar")

        # toolbar for editting widgets
        self.edittoolbar = qt4.QToolBar("Edit toolbar - Veusz",
                                        parent)
        self.edittoolbar.setObjectName("veuszedittoolbar")

        self._constructToolbarMenu()
        parent.addToolBarBreak(qt4.Qt.TopToolBarArea)
        parent.addToolBar(qt4.Qt.TopToolBarArea, self.addtoolbar)
        parent.addToolBar(qt4.Qt.TopToolBarArea, self.edittoolbar)

        # this sets various things up
        self.selectWidget(document.basewidget)

        # update paste button when clipboard changes
        self.connect(qt4.QApplication.clipboard(),
                     qt4.SIGNAL('dataChanged()'),
                     self.updatePasteButton)
        self.updatePasteButton()

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
        for act in ('edit.cut', 'edit.copy', 'edit.paste',
                    'edit.moveup', 'edit.movedown', 'edit.delete',
                    'edit.rename'):
            m.addAction(self.vzactions[act])

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

        for wc, action in self.addslots.iteritems():
            w = selw
            while w is not None and not wc.willAllowParent(w):
                w = w.parent

            self.vzactions['add.%s' % wc.typename].setEnabled(w is not None)

        # certain actions shouldn't allow root to be deleted
        isnotroot = not isinstance(selw, widgets.Root)

        for act in ('edit.cut', 'edit.copy', 'edit.delete',
                    'edit.moveup', 'edit.movedown', 'edit.rename'):
            self.vzactions[act].setEnabled(isnotroot)

        self.updatePasteButton()

    def _constructToolbarMenu(self):
        """Add items to edit/add graph toolbar and menu."""

        iconsize = setting.settingdb['toolbar_size']
        self.addtoolbar.setIconSize( qt4.QSize(iconsize, iconsize) )
        self.edittoolbar.setIconSize( qt4.QSize(iconsize, iconsize) )

        self.addslots = {}
        self.vzactions = actions = self.parent.vzactions
        for widgettype in ('page', 'grid', 'graph', 'axis',
                           'xy', 'bar', 'fit', 'function',
                           'image', 'contour',
                           'key', 'label', 'colorbar',
                           'rect', 'ellipse', 'imagefile',
                           'line', 'polygon'):

            wc = document.thefactory.getWidgetClass(widgettype)
            slot = utils.BoundCaller(self.slotMakeWidgetButton, wc)
            self.addslots[wc] = slot
            
            actionname = 'add.' + widgettype
            actions[actionname] = utils.makeAction(
                self,
                wc.description, 'Add %s' % widgettype,
                slot,
                icon='button_%s' % widgettype)

        a = utils.makeAction
        actions.update({
                'edit.cut':
                    a(self, 'Cut the selected item', 'Cu&t',
                      self.slotWidgetCut,
                      icon='veusz-edit-cut', key='Ctrl+X'),
                'edit.copy':
                    a(self, 'Copy the selected item', '&Copy',
                      self.slotWidgetCopy,
                      icon='kde-edit-copy', key='Ctrl+C'),
                'edit.paste':
                    a(self, 'Paste item from the clipboard', '&Paste',
                      self.slotWidgetPaste,
                      icon='kde-edit-paste', key='Ctrl+V'),
                'edit.moveup':
                    a(self, 'Move the selected item up', 'Move &up',
                      utils.BoundCaller(self.slotWidgetMove, -1),
                      icon='kde-go-up'),
                'edit.movedown':
                    a(self, 'Move the selected item down', 'Move d&own',
                      utils.BoundCaller(self.slotWidgetMove, 1),
                      icon='kde-go-down'),
                'edit.delete':
                    a(self, 'Remove the selected item', '&Delete',
                      self.slotWidgetDelete,
                      icon='kde-edit-delete'),
                'edit.rename':
                    a(self, 'Renames the selected item', '&Rename',
                      self.slotWidgetRename,
                      icon='kde-edit-rename'),

                'add.shapemenu':
                    a(self, 'Add a shape to the plot', 'Shape',
                      self.slotShowShapeMenu,
                      icon='veusz-shape-menu'),
                })

        # add actions to menus for adding widgets and editing
        addact = [('add.'+w) for w in 
                  ('page', 'grid', 'graph', 'axis',
                   'xy', 'bar', 'fit', 'function',
                   'image', 'contour',
                   'key', 'label', 'colorbar')]

        menuitems = [
            ('insert', '', addact + [
                    ['insert.shape', 'Add shape',
                     ['add.rect', 'add.ellipse', 'add.line', 'add.imagefile',
                      'add.polygon']
                     ]]),
            ('edit', '', [
                    'edit.cut', 'edit.copy', 'edit.paste',
                    'edit.moveup', 'edit.movedown',
                    'edit.delete', 'edit.rename'
                    ]),
            ]            
        utils.constructMenus( self.parent.menuBar(),
                              self.parent.menus,
                              menuitems,
                              actions )

        # create shape toolbar button
        # attach menu to insert shape button
        actions['add.shapemenu'].setMenu(self.parent.menus['insert.shape'])

        # add actions to toolbar to create widgets
        utils.addToolbarActions(self.addtoolbar, actions,
                                addact + ['add.shapemenu'])

        # add action to toolbar for editing
        utils.addToolbarActions(self.edittoolbar,  actions,
                                ('edit.cut', 'edit.copy', 'edit.paste',
                                 'edit.moveup', 'edit.movedown',
                                 'edit.delete', 'edit.rename'))

    def slotMakeWidgetButton(self, wc):
        """User clicks button to make widget."""
        self.makeWidget(wc.typename)

    def slotShowShapeMenu(self):
        a = self.vzactions['add.shapemenu']
        a.menu().popup( qt4.QCursor.pos() )

    def makeWidget(self, widgettype, autoadd=True, name=None):
        """Called when an add widget button is clicked.
        widgettype is the type of widget
        autoadd specifies whether to add default children
        if name is set this name is used if possible (ie no other children have it)
        """

        # if no widget selected, bomb out
        if self.selwidget is None:
            return
        parent = document.getSuitableParent(widgettype, self.selwidget)

        assert parent is not None

        if name in parent.childnames:
            name = None
        
        # make the new widget and update the document
        w = self.document.applyOperation(
            document.OperationWidgetAdd(parent, widgettype, autoadd=autoadd,
                                        name=name) )

        # select the widget
        self.selectWidget(w)

    def slotWidgetCut(self):
        """Cut the selected widget"""
        self.slotWidgetCopy()
        self.slotWidgetDelete()

    def slotWidgetCopy(self):
        """Copy selected widget to the clipboard."""

        if self.selwidget:
            mimedata = document.generateWidgetsMime([self.selwidget])
            clipboard = qt4.QApplication.clipboard()
            clipboard.setMimeData(mimedata)

    def updatePasteButton(self):
        """Is the data on the clipboard a valid paste at the currently
        selected widget? If so, enable paste button"""

        data = document.getClipboardWidgetMime()
        show = document.isMimePastable(self.selwidget, data)
        self.vzactions['edit.paste'].setEnabled(show)

    def doInitialWidgetSelect(self):
        """Select a sensible initial widget."""
        w = self.document.basewidget
        for i in xrange(2):
            try:
                c = w.children[0]
            except IndexError:
                break
            if c:
                w = c
        self.selectWidget(w)

    def slotWidgetPaste(self):
        """Paste something from the clipboard"""

        data = document.getClipboardWidgetMime()
        if data:
            op = document.OperationWidgetPaste(self.selwidget, data)
            widgets = self.document.applyOperation(op)
            if widgets:
                self.selectWidget(widgets[0])

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

        # select the next widget (we have to select root first!)
        self.selectWidget(self.document.basewidget)
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
            document.OperationWidgetMoveUpDown(w, direction) )

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
        self.labelicon = qt4.QLabel(text)
        self.layout.addWidget(self.labelicon)
        
        self.iconlabel = qt4.QLabel()
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

        # if not default, make label bold
        bold = not self.setting.isDefault()
        f = qt4.QFont(self.labelicon.font())
        f.setBold(bold)
        self.labelicon.setFont(f)

    def updateHighlight(self):
        """Show drop down arrow if item has focus."""
        if self.inmouse or self.infocus or self.inmenu:
            pixmap = 'downarrow.png'
        else:
            if self.setting.isReference() and not self.setting.isDefault():
                pixmap = 'link.png'
            else:
                pixmap = 'downarrow_blank.png'
        self.iconlabel.setPixmap(utils.getPixmap(pixmap))

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

    def addCopyToWidgets(self, menu):
        """Make a menu with list of other widgets in it."""

        def getWidgetsOfType(widget, widgettype, widgets=[]):
            """Recursively build up a list of widgets of the type given."""
            for w in widget.children:
                if w.typename == widgettype:
                    widgets.append(w)
                getWidgetsOfType(w, widgettype, widgets)
        
        # get list of widget paths to copy setting to
        # this is all widgets of same type
        widgets = []
        setwidget = self.setting.getWidget()
        getWidgetsOfType(self.document.basewidget,
                         setwidget.typename, widgets)
        widgets = [w.path for w in widgets if w != setwidget]
        widgets.sort()

        # chop off widget part of setting path
        # this is so we can add on a different widget path
        setpath = self.setting.path
        wpath = self.setting.getWidget().path
        setpath = setpath[len(wpath)+1:]

        for widget in widgets:
            action = menu.addAction(widget)
            def modify(widget=widget):
                """Modify the setting for the widget given."""
                wpath = widget + setpath
                self.document.applyOperation(
                    document.OperationSettingSet(wpath, self.setting.get()))

            menu.connect(action, qt4.SIGNAL('triggered()'), modify)

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
        popup.addAction('Reset to default',
                        self.actionResetDefault)

        copyto = popup.addMenu('Copy to')
        copyto.addAction("all '%s' widgets" % wtype,
                         self.actionCopyTypedWidgets)
        copyto.addAction("'%s' siblings" % wtype,
                         self.actionCopyTypedSiblings)
        copyto.addAction("'%s' widgets called '%s'" % (wtype, name),
                         self.actionCopyTypedNamedWidgets)
        copyto.addSeparator()
        self.addCopyToWidgets(copyto)

        popup.addAction('Use as default style',
                        self.actionSetStyleSheet)

        # special actions for references
        if self.setting.isReference():
            popup.addSeparator()
            popup.addAction('Unlink setting', self.actionUnlinkSetting)

        self.inmenu = True
        self.updateHighlight()
        popup.exec_(pos)
        self.inmenu = False
        self.updateHighlight()

    def actionResetDefault(self):
        """Reset setting to default."""
        self.document.applyOperation(
            document.OperationSettingSet(
                self.setting,
                self.setting.default))

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

    def actionSetStyleSheet(self):
        """Use the setting as the default in the stylesheet."""

        # get name of stylesheet setting
        sslink = self.setting.getStylesheetLink()
        # apply operation to change it
        self.document.applyOperation(
            document.OperationMultiple(
                [ document.OperationSettingSet(sslink, self.setting.get()),
                  document.OperationSettingSet(self.setting,
                                               self.setting.default) ],
                descr="make default style")
            )

