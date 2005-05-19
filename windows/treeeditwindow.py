# treeeditwindow.py
# A class for editing the graph tree with a GUI

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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
###############################################################################

# $Id$

import os

import qt
import qttable

import widgets.widgetfactory as widgetfactory
import widgets
import action
import utils

class _WidgetItem(qt.QListViewItem):
    """Item for displaying in the TreeEditWindow."""

    def __init__(self, widget, qtparent):
        """Widget is the widget to show the settings for."""
        
        qt.QListViewItem.__init__(self, qtparent)
        self.setRenameEnabled(0, True)

        self.index = 0
        self.widget = widget
        self.settings = widget.settings

        # add subitems for sub-prefs of widget
        no = 0
        for s in self.settings.getSettingsList():
            _PrefItem(s, no, self)
            no += 1

    def setText(self, col, text):
        """Update name of widget if rename is called."""

        # update name of widget
        if col == 0:
            try:
                self.widget.rename( unicode(text) )
            except ValueError:
                # if the rename failed
                text = self.widget.name

        qt.QListViewItem.setText(self, col, text)

    def rename(self):
        """Rename the listviewitem."""
        self.startRename(0)

    def compare(self, i, col, ascending):
        """Always sort according to the index value."""

        a = [-1, 1][ascending]
            
        if self.index < i.index:
            return -1*a
        elif self.index > i.index:
            return 1*a
        else:
            return 0

    def text(self, column):
        """Get the text in a particular column."""
        if column == 0:
            return self.widget.name
        elif column == 1:
            return self.widget.typename
        elif column == 2:
            return self.widget.userdescription
        return ''

class _PrefItem(qt.QListViewItem):
    """Item for displaying a preferences-set in TreeEditWindow."""
    def __init__(self, settings, number, parent):
        """settings is the settings class to work for
        parent is the parent ListViewItem (of type _WidgetItem)
        """

        qt.QListViewItem.__init__(self, parent)

        self.settings = settings
        self.parent = parent
        self.widget = None
        self.setText(0, settings.name)
        self.setText(1, 'setting')
        self.index = number

    def compare(self, i, col, ascending):
        """Always sort according to the index value."""

        a = [-1, 1][ascending]
           
        if self.index < i.index:
            return -1*a
        elif self.index > i.index:
            return 1*a
        else:
            return 0

class _PropertyLabel(qt.QLabel):
    """A label which produces the veusz setting context menu.

    This label handles mouse move and focus events. Both of these
    shade the widget darker, giving the user information that the widget
    has focus, and a context menu.
    """

    def __init__(self, setting, text, parent):
        """Initialise the label for the given setting."""

        qt.QLabel.__init__(self, text, parent)
        self.bgcolor = self.paletteBackgroundColor()
        self.setFocusPolicy(qt.QWidget.StrongFocus)
        self.setMargin(1)

        self.setting = setting
        self.inmenu = False
        self.inmouse = False
        self.infocus = False

    def _setBg(self):
        """Set the background of the widget according to its state."""

        # darken widget according to num (100 is normal state)
        num = 100
        if self.inmenu:
            num += 20
        else:
            if self.inmouse:
                num += 10
            if self.infocus:
                num += 10
        
        self.setPaletteBackgroundColor(self.bgcolor.dark(num))

    def enterEvent(self, event):
        """When the mouse enters the widget."""
        qt.QLabel.enterEvent(self, event)
        self.inmouse = True
        self._setBg()

    def leaveEvent(self, event):
        """When the mouse leaves the widget."""
        qt.QLabel.leaveEvent(self, event)
        self.inmouse = False
        self._setBg()

    def focusInEvent(self, event):
        """When widget gets focus."""
        qt.QLabel.focusInEvent(self, event)
        self.infocus = True
        self._setBg()

    def focusOutEvent(self, event):
        """When widget loses focus."""
        qt.QLabel.focusOutEvent(self, event)
        self.infocus = False
        self._setBg()

    def keyPressEvent(self, event):
        """Use cursor keys to move focus."""

        key = event.key()
        # move up two as in a 2 column grid
        if key == qt.Qt.Key_Up:
            self.focusNextPrevChild(False)
            self.focusNextPrevChild(False)
        elif key == qt.Qt.Key_Down:
            self.focusNextPrevChild(True)
            self.focusNextPrevChild(True)
        elif key == qt.Qt.Key_Left:
            self.focusNextPrevChild(False)
        elif key == qt.Qt.Key_Right:
            self.focusNextPrevChild(True)
        else:
            event.ignore()

    def contextMenuEvent(self, event):
        """Pop up the context menu."""

        # for labels which don't correspond to settings
        if self.setting == None:
            event.ignore()
            return

        # forces settings to be updated
        self.parentWidget().setFocus()
        # get it back straight away
        self.setFocus()

        # darken the widget (gives stability)
        self.inmenu = True
        self._setBg()

        # get widget, with its type and name
        widget = self.setting.parent
        while not isinstance(widget, widgets.Widget):
            widget = widget.parent

        type = widget.typename
        name = widget.name
        
        # construct the popup menu
        popup = qt.QPopupMenu(self)

        popup.insertItem('Reset to default', 0)
        popup.insertSeparator()
        popup.insertItem('Copy to "%s" widgets' % type, 100)
        popup.insertItem('Copy to "%s" siblings' % type, 101)
        popup.insertItem('Copy to "%s" widgets called "%s"' %
                         (type, name), 102)
        popup.insertSeparator()
        popup.insertItem('Make default for "%s" widgets' % type, 200)
        popup.insertItem('Make default for "%s" widgets called "%s"' %
                         (type, name), 201)
        popup.insertItem('Forget this default setting', 202)
        
        ret = popup.exec_loop( event.globalPos() )

        # convert values above to functions
        doc = widget.document
        setn = self.setting
        fnmap = {
            0: setn.resetToDefault,
            100: (lambda: doc.propagateSettings(setn)),
            101: (lambda: doc.propagateSettings(setn, root=widget.parent,
                                                maxlevels=1)),
            102: (lambda: doc.propagateSettings(setn, widgetname=name)),
            
            200: (lambda: setn.setAsDefault(False)),
            201: (lambda: setn.setAsDefault(True)),
            202: setn.removeDefault
            }

        # call the function if item was selected
        if ret >= 0:
            fnmap[ret]()

        # return widget to previous colour
        self.inmenu = False
        self._setBg()

class _WidgetListView(qt.QListView):
    """A list view which emits contextMenu signals."""

    def contextMenuEvent(self, event):
        """Emit a context menu signal."""
        self.emit( qt.PYSIGNAL('contextMenu'), (event.globalPos(),) )

class _PropTable(qttable.QTable):
    """The table which shows the properties of the selected widget."""

    def __init__(self, parent):
        """Initialise the table."""
        qttable.QTable.__init__(self, parent)
        self.setFocusPolicy(qt.QWidget.NoFocus)
        self.setNumCols(2)
        self.setTopMargin(0)
        self.setLeftMargin(0)
        self.setShowGrid(False)
        self.setColumnStretchable(1, True)
        self.setSelectionMode(qttable.QTable.NoSelection)

    def keyPressEvent(self, event):
        """This method is necessary as the table steals keyboard input
        even if it cannot have focus."""
        fw = self.focusWidget()
        if fw != self:
            try:
                fw.keyPressEvent(event)
            except RuntimeError:
                # doesn't work for controls which aren't Python based
                event.ignore()
        else:
            event.ignore()

    def keyReleaseEvent(self, event):
        """This method is necessary as the table steals keyboard input
        even if it cannot have focus."""
        fw = self.focusWidget()
        if fw != self:
            try:
                fw.keyReleaseEvent(event)
            except RuntimeError:
                # doesn't work for controls which aren't Python based
                event.ignore()
        else:
            event.ignore()

class TreeEditWindow(qt.QDockWindow):
    """A graph editing window with tree display."""

    def __init__(self, thedocument, parent):
        qt.QDockWindow.__init__(self, parent)
        self.setResizeEnabled( True )
        self.setCaption("Editing - Veusz")

        self.parent = parent
        self.document = thedocument
        self.connect( self.document, qt.PYSIGNAL("sigModified"),
                      self.slotDocumentModified )
        self.connect( self.document, qt.PYSIGNAL("sigWiped"),
                      self.slotDocumentWiped )

        # make toolbar in parent to have the add graph/edit graph buttons
        self.edittool = qt.QToolBar(parent, "treetoolbar")
        self.edittool.setLabel("Editing toolbar - Veusz")
        parent.moveDockWindow(self.edittool, qt.Qt.DockLeft, True, 0)

        self._constructToolbarMenu()

        # window uses vbox for arrangement
        totvbox = qt.QVBox(self)
        self.setWidget(totvbox)

        # put widgets in a movable splitter
        split = qt.QSplitter(totvbox)
        split.setOrientation(qt.QSplitter.Vertical)

        # first widget is a listview
        vbox = qt.QVBox(split)
        l = qt.QLabel("Items", vbox)
        l.setMargin(2)

        lv = self.listview = _WidgetListView(vbox)
        l.setBuddy(lv)
        lv.setSorting(-1)
        lv.setRootIsDecorated(True)
        self.connect( lv, qt.SIGNAL("selectionChanged(QListViewItem*)"),
                      self.slotItemSelected )
        self.connect( lv, qt.PYSIGNAL('contextMenu'),
                      self.slotListContextMenu )

        # we use a hidden column to get the sort order correct
        lv.addColumn( "Name" )
        lv.addColumn( "Type" )
        lv.addColumn( "Detail" )
        lv.setColumnWidthMode(2, qt.QListView.Manual)
        lv.setSorting(0)
        lv.setTreeStepSize(10)

        # add root widget to view
        self.rootitem = _WidgetItem( self.document.basewidget, lv )

        # add a scrollable view for the preferences
        # children get added to prefview
        vbox = qt.QVBox(split)
        self.proplabel = qt.QLabel("&Properties", vbox)
        self.proplabel.setMargin(2)
        self.proplabel.setBuddy(self)
        self.proptab = _PropTable(vbox)

        self.prefchilds = []

        # select the root item
        self.listview.setSelected(self.rootitem, True)

    def sizeHint(self):
        """Returns recommended size of dialog."""
        return qt.QSize(250, 500)

    def _constructToolbarMenu(self):
        """Add items to edit/add graph toolbar and menu."""

        # make buttons to add each of the widget types
        self.createGraphActions = {}

        insertmenu = self.parent.menus['insert']

        for wc in self._getWidgetOrder():
            name = wc.typename
            if wc.allowusercreation:
                a = action.Action(self,
                                  (lambda w:
                                   (lambda a: self.slotMakeWidgetButton(w)))
                                  (name),
                                  iconfilename = 'button_%s.png' % name,
                                  menutext = 'Add %s' % name,
                                  statusbartext = wc.description,
                                  tooltiptext = wc.description)

                a.addTo(self.edittool)
                a.addTo(insertmenu)
                self.createGraphActions[wc] = a

        self.edittool.addSeparator()

        # make buttons and menu items for the various item editing ops
        self.editactions = {}
        editmenu = self.parent.menus['edit']

        self.contextpopup = qt.QPopupMenu(self)

        for name, icon, tooltip, menutext, slot in (
            ('moveup', 'stock-go-up.png', 'Move the selected item up',
             'Move &up',
             lambda a: self.slotWidgetMove(a, -1) ),
            ('movedown', 'stock-go-down.png', 'Move the selected item down',
             'Move &down',
             lambda a: self.slotWidgetMove(a, 1) ),
            ('delete', 'stock-delete.png', 'Remove the selected item',
             '&Delete',
             self.slotWidgetDelete),
            ('rename', 'icon-rename.png', 'Rename the selected item',
             '&Rename',
             self.slotWidgetRename)
            ):

            a = action.Action(self, slot,
                              iconfilename = icon,
                              menutext = menutext,
                              statusbartext = tooltip,
                              tooltiptext = tooltip)
            a.addTo(self.edittool)
            a.addTo(self.contextpopup)
            a.addTo(editmenu)
            self.editactions[name] = a

    def _getWidgetOrder(self):
        """Return a list of the widgets, most important first.
        """

        # get list of allowed classes
        wcl = [(i.typename, i)
               for i in widgetfactory.thefactory.listWidgetClasses()
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

    def slotDocumentModified(self, ismodified):
        """Called when the document has been modified."""
 
        if ismodified:
            self.updateContents()

    def _updateBranch(self, root):
        """Recursively update items on branch."""

        # build dictionary of items
        items = {}
        i = root.firstChild()
        while i != None:
            w = i.widget
            if w != None:
                items[w] = i
            i = i.nextSibling()

        childdict = {}
        # assign indicies to each one
        index = 10000
        newitem = False
        for c in root.widget.children:
            childdict[c] = True
            if c in items:
                items[c].index = index
            else:
                items[c] = _WidgetItem(c, root)
                items[c].index = index
                newitem = True
            self._updateBranch(items[c])
            index += 1

        # delete items not in child list
        for i in items.itervalues():
            if i.widget not in childdict:
                root.takeItem(i)

        # have to re-sort children to ensure ordering is correct here
        root.sort()

        # open the branch if we've added/changed the children
        if newitem:
            self.listview.setOpen(root, True)

    def slotDocumentWiped(self):
        """Called when there is a new document."""

        self.listview.clear()
        self.rootitem = _WidgetItem( self.document.basewidget,
                                     self.listview )
        self.listview.setSelected(self.rootitem, True)
        self.updateContents()

    def updateContents(self):
        """Make the window reflect the document."""

        self._updateBranch(self.rootitem)
        sel = self.listview.selectedItem()
        if sel != None:
            self.listview.ensureItemVisible(sel)

        self.listview.triggerUpdate()

    def enableCorrectButtons(self, item):
        """Make sure the create graph buttons are correctly enabled."""
        selw = item.widget
        if selw == None:
            selw = item.parent.widget
            assert selw != None

        # check whether each button can have this widget
        # (or a parent) as parent
        for wc, action in self.createGraphActions.items():
            w = selw
            while w != None and not wc.willAllowParent(w):
                w = w.parent
            action.enable( w != None )

        # delete shouldn't allow root to be deleted
        isnotroot = not isinstance(selw, widgets.Root)

        self.editactions['delete'].enable(isnotroot)
        self.editactions['rename'].enable(isnotroot)

    def slotItemSelected(self, item):
        """Called when an item is selected in the listview."""

        # enable or disable the create graph buttons
        self.enableCorrectButtons(item)

        self.itemselected = item

        # delete the current widgets in the preferences list
        while len(self.prefchilds) > 0:
            i = self.prefchilds.pop()
            try:
                i.done()
            except AttributeError:
                pass

            # need line below or occasionally get random error
            # "QToolTip.maybeTip() is abstract and must be overridden"
            qt.QToolTip.remove(i)

            i.deleteLater()

        # calculate number of rows
        rows = len(item.settings.getSettingList())
        w = item.widget
        if w != None:
            rows += len(w.actions)
        self.proptab.setNumRows(rows)

        row = 0
        view = self.proptab.viewport()
        # add action for widget
        if w != None:
            for name in w.actions:
                l = _PropertyLabel(None, name, view)
                self.proptab.setCellWidget(row, 0, l)
                self.prefchilds.append(l)

                b = qt.QPushButton(w.actiondescr[name], view)
                b.veusz_action = w.actionfuncs[name]
                self.proptab.setCellWidget(row, 1, b)
                self.prefchilds.append(b)
                
                self.connect( b, qt.SIGNAL('clicked()'),
                              self.slotActionClicked )
                row += 1

        # make new widgets for the preferences
        for setn in item.settings.getSettingList():
            tooltext = "<strong>%s</strong> - %s" % (setn.name,
                                                     setn.descr)
            
            l = _PropertyLabel(setn, setn.name, view)
            self.proptab.setCellWidget(row, 0, l)
            qt.QToolTip.add(l, tooltext)
            self.prefchilds.append(l)

            c = setn.makeControl(view)
            self.proptab.setCellWidget(row, 1, c)
            qt.QToolTip.add(c, tooltext)
            self.prefchilds.append(c)

            self.proptab.adjustRow(row)

            row += 1

        # make properties keyboard shortcut point to first item
        if len(self.prefchilds) > 0:
            self.proplabel.setBuddy(self.prefchilds[0])
        else:
            self.proplabel.setBuddy(self)
            
        # Change the page to the selected widget
        w = item.widget
        if w == None:
            w = item.parent.widget

        # repeat until we're at the root widget or we hit a page
        while w != None and not isinstance(w, widgets.Page):
            w = w.parent

        if w != None:
            # we have a page
            count = 0
            children = self.document.basewidget.children
            for c in children:
                if c == w:
                    break
                count += 1

            if count < len(children):
                self.emit( qt.PYSIGNAL("sigPageChanged"), (count,) )
            
    def slotMakeWidgetButton(self, widgettype):
        """Called when an add widget button is clicked.
        widgettype is the type of widget
        """

        # if no widget selected, bomb out
        if self.itemselected == None:
            return

        # get the widget to act as the parent
        parent = self.itemselected.widget
        if parent == None:
            parent = self.itemselected.parent.widget
            assert parent != None

        # find the parent to add the child to, we go up the tree looking
        # for possible parents
        wc = widgetfactory.thefactory.getWidgetClass(widgettype)
        while parent != None and not wc.willAllowParent(parent):
            parent = parent.parent

        assert parent != None

        # make the new widget and update the document
        w = widgetfactory.thefactory.makeWidget(widgettype, parent)
        self.document.setModified()

        # select the widget
        self.selectWidget(w)

    def slotActionClicked(self):
        """Called when an action button is clicked."""

        # get the button clicked/activated
        button = self.sender()

        # set focus to button to make sure other widgets lose
        # focus and update their settings
        button.setFocus()

        # run action in console
        action = button.veusz_action
        console = self.parent.console
        console.runFunction( action )

    def slotWidgetDelete(self, a):
        """Delete the widget selected."""

        # no item selected, so leave
        if self.itemselected == None:
            return

        # get the widget to act as the parent
        w = self.itemselected.widget
        if w == None:
            w = self.itemselected.parent.widget
            assert w != None
        self.itemselected = None

        # find the parent
        p = w.parent
        assert p != None

        # delete the widget
        p.removeChild(w.name)
        self.document.setModified()

    def selectWidget(self, widget):
        """Select the associated listviewitem for the widget w in the
        listview."""
        
        # an iterative algorithm, rather than a recursive one
        # (for a change)
        found = False
        l = [self.listview.firstChild()]
        while len(l) != 0 and not found:
            item = l.pop()

            i = item.firstChild()
            while i != None:
                if i.widget == widget:
                    found = True
                    break
                else:
                    l.append(i)
                i = i.nextSibling()

        assert found
        self.listview.ensureItemVisible(i)
        self.listview.setSelected(i, True)

    def slotWidgetMove(self, a, direction):
        """Move the selected widget up/down in the hierarchy.

        a is the action (unused)
        direction is -1 for 'up' and +1 for 'down'
        """

        # get the widget to act as the parent
        w = self.itemselected.widget
        if w == None:
            w = self.itemselected.parent.widget
            assert w != None

        # find the parent
        p = w.parent
        if p == None:
            return

        # move it up
        p.moveChild(w, direction)
        self.document.setModified()

        # try to highlight the associated item
        self.selectWidget(w)
        
    def slotListContextMenu(self, pos):
        """Pop up a context menu when an item is clicked on the list view."""

        self.contextpopup.exec_loop(pos)

    def slotWidgetRename(self, action):
        """Initiate renaming a widget."""

        item = self.itemselected
        while item.widget == None:
            item = item.parent

        item.rename()
