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

"""Edit the document using a tree and properties.
"""

import os

import qt
import qttable

import veusz.widgets as widgets
import veusz.utils as utils
import veusz.document as document

import action

class _WidgetItem(qt.QListViewItem):
    """Item for displaying in the TreeEditWindow."""

    def __init__(self, widget, qtparent):
        """Widget is the widget to show the settings for."""
        
        qt.QListViewItem.__init__(self, qtparent)
        self.setRenameEnabled(0, True)

        self.index = 0
        self.widget = widget
        self.settings = widget.settings

        self.setPixmap(0, action.getPixmap('button_%s.png' % widget.typename) )
        
        self.recursiveAddPrefs(0, self.settings, self)

    def getAssociatedWidget(self):
        """Return the widget associated with this item."""
        return self.widget
        
    def recursiveAddPrefs(self, no, settings, parent):
        """Recursively add preference subsettings."""
        for s in settings.getSettingsList():
            i = _PrefItem(s, no, parent)
            no += 1
            no = self.recursiveAddPrefs(no, s, i)
            
        return no
            
    def setText(self, col, text):
        """Update name of widget if rename is called."""

        # update name of widget
        if col == 0:
            try:
                self.widget.document.applyOperation(
                    document.OperationWidgetRename(self.widget, unicode(text)) )
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

        if hasattr(settings, 'pixmap'):
            self.setPixmap(0, action.getPixmap('settings_%s.png' %
                                               settings.pixmap) )
        
    def compare(self, i, col, ascending):
        """Always sort according to the index value."""

        a = [-1, 1][ascending]
           
        if self.index < i.index:
            return -1*a
        elif self.index > i.index:
            return 1*a
        else:
            return 0

    def getAssociatedWidget(self):
        """Get widget associated with this item."""
        self.parent.getAssociatedWidget()

class _NewPropertyLabel(qt.QHBox):
    """A widget for displaying the label for a setting."""

    def __init__(self, setting, parent):

        qt.QHBox.__init__(self, parent)
        self.setting = setting

        self.menubutton = qt.QPushButton(setting.name, self)
        self.menubutton.setFlat(True)
        self.connect(self.menubutton, qt.SIGNAL('clicked()'),
                     self.slotContextMenu)
        
        tooltext = "<strong>%s</strong> - %s" % (setting.name,
                                                 setting.descr)
        qt.QToolTip.add(self.menubutton, tooltext)

        self.linkbutton = qt.QPushButton(action.getIconSet('link.png'), '', self)
        self.linkbutton.setMaximumWidth(self.linkbutton.height())
        self.linkbutton.setFlat(True)

        self.connect(self.linkbutton, qt.SIGNAL('clicked()'),
                     self.buttonClicked)

        setting.setOnModified(self.slotOnModified)
        # show linkbutton if appropriate, update tooltip
        self.slotOnModified(True)
        
    def slotOnModified(self, mod):
        """Alter reference button if setting is modified."""
        
        isref = self.setting.isReference()
        if isref:
            ref = self.setting.getReference()
            qt.QToolTip.add(self.linkbutton, "Linked to %s" % ref.value)
        self.linkbutton.setShown(isref)

    def getWidget(self):
        """Get associated Veusz widget."""
        widget = self.setting.parent
        while not isinstance(widget, widgets.Widget):
            widget = widget.parent
        return widget

    def slotContextMenu(self):
        """Pop up the context menu."""

        # forces settings to be updated
        self.parentWidget().setFocus()
        # get it back straight away
        self.menubutton.setFocus()

        widget = self.getWidget()
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

        #pos = self.menubutton.mapToGlobal(self.menubutton.pos())
        ret = popup.exec_loop( qt.QCursor.pos() )

        # convert values above to functions
        doc = widget.document
        setn = self.setting
        fnmap = {
            0: (lambda: doc.applyOperation( document.OperationSettingSet(setn, setn.default) )),
            100: (lambda: doc.applyOperation( document.OperationSettingPropagate(setn) )),
            101: (lambda: doc.applyOperation( document.OperationSettingPropagate(setn, root=widget.parent, maxlevels=1) )),
            102: (lambda: doc.applyOperation( document.OperationSettingPropagate(setn, widgetname=name) )),
            
            200: (lambda: setn.setAsDefault(False)),
            201: (lambda: setn.setAsDefault(True)),
            202: setn.removeDefault
            }

        # call the function if item was selected
        if ret >= 0:
            fnmap[ret]()

    def buttonClicked(self):
        """Create a popup menu when the button is clicked."""
        popup = qt.QPopupMenu(self)

        popup.insertItem('Unlink setting', 100)
        popup.insertItem('Edit linked setting', 101)
        
        ret = popup.exec_loop( qt.QCursor.pos() )

        setn = self.setting
        widget = self.getWidget()
        doc = widget.document
        if ret == 100:
            # update setting with own value to get rid of reference
            doc.applyOperation( document.OperationSettingSet(setn, setn.get()) )

class _PropertyLabelLabel(qt.QLabel):
    """A widget for displaying the actual label in the property label."""

    def __init__(self, setting, text, parent):
        """Initialise widget showing text

        setting is the appropriate setting."""
        
        qt.QLabel.__init__(self, text, parent)
        self.bgcolor = self.paletteBackgroundColor()
        self.setFocusPolicy(qt.QWidget.StrongFocus)
        self.setMargin(1)

        self.setting = setting
        self.inmenu = False
        self.inmouse = False
        self.infocus = False
        self.parent = parent
        
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
            0: (lambda: self.parent.control.emit( qt.PYSIGNAL('settingChanged'),
                                                  (self.parent.control, setn, setn.default) )),
            100: (lambda: doc.applyOperation( document.OperationSettingPropagate(setn) )),
            101: (lambda: doc.applyOperation( document.OperationSettingPropagate(setn, root=widget.parent, maxlevels=1) )),
            102: (lambda: doc.applyOperation( document.OperationSettingPropagate(setn, widgetname=name) )),
            
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
            
class _PropertyLabel(qt.QHBox):
    """A label which produces the veusz setting context menu.

    This label handles mouse move and focus events. Both of these
    shade the widget darker, giving the user information that the widget
    has focus, and a context menu.
    """

    def __init__(self, setting, text, parent):
        """Initialise the label for the given setting."""

        qt.QHBox.__init__(self, parent)
        self.setMargin(0)
        self.setting = setting
        self.control = None

        self.label = _PropertyLabelLabel(setting, text, self)
        self.label.setSizePolicy( qt.QSizePolicy(qt.QSizePolicy.Minimum,
                                                 qt.QSizePolicy.Fixed) )
        
class _WidgetListView(qt.QListView):
    """A list view for the widgets

    It emits contextMenu signals, and allows widgets to be selected
    """

    def contextMenuEvent(self, event):
        """Emit a context menu signal."""
        self.emit( qt.PYSIGNAL('contextMenu'), (event.globalPos(),) )

    def selectWidget(self, widget):
        """Find the widget in the list and select it."""

        # check each item in the list to see whether it corresponds
        # to the widget
        iter = qt.QListViewItemIterator(self)

        found = None
        while True:
            item = iter.current()
            if item == None:
                break
            if item.widget == widget:
                found = item
                break
            iter += 1

        if found:
            self.ensureItemVisible(found)
            self.setSelected(found, True)

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

    # mime type when widgets are stored on the clipboard
    widgetmime = 'text/x-vnd.veusz-clipboard'

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

        # time to update tree view
        self.updatetimer = qt.QTimer(self)
        self.connect( self.updatetimer, qt.SIGNAL('timeout()'),
                      self.slotUpdateTimerTimeout )

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

        for name, icon, tooltip, menutext, accel, slot in (
            ('cut', 'stock-cut.png', 'Cut the selected item',
             '&Cut', 'Ctrl+X',
             self.slotWidgetCut),
            ('copy', 'stock-copy.png', 'Copy the selected item',
             '&Copy', 'Ctrl+C',
             self.slotWidgetCopy),
            ('paste', 'stock-paste.png', 'Paste from the clipboard',
             '&Paste','Ctrl+V',
             self.slotWidgetPaste),
            ('moveup', 'stock-go-up.png', 'Move the selected item up',
             'Move &up','',
             lambda a: self.slotWidgetMove(a, -1) ),
            ('movedown', 'stock-go-down.png', 'Move the selected item down',
             'Move &down','',
             lambda a: self.slotWidgetMove(a, 1) ),
            ('delete', 'stock-delete.png', 'Remove the selected item',
             '&Delete','',
             self.slotWidgetDelete),
            ('rename', 'icon-rename.png', 'Rename the selected item',
             '&Rename','',
             self.slotWidgetRename)
            ):

            a = action.Action(self, slot,
                              iconfilename = icon,
                              menutext = menutext,
                              statusbartext = tooltip,
                              tooltiptext = tooltip,
                              accel=accel)
            a.addTo(self.edittool)
            a.addTo(self.contextpopup)
            a.addTo(editmenu)
            self.editactions[name] = a

    def _getWidgetOrder(self):
        """Return a list of the widgets, most important first.
        """

        # get list of allowed classes
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

    def slotDocumentModified(self, ismodified):
        """Called when the document has been modified."""
 
        if ismodified:
            # do this so we avoid refreshing the tree every time
            # the document is modified
            # this speeds things up a lot, and has the effect of batching
            # operations
            self.updatetimer.start(50, True)

    def slotUpdateTimerTimeout(self):
        """Update the tree when times out."""
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
        selw = item.getAssociatedWidget()

        # check whether each button can have this widget
        # (or a parent) as parent
        for wc, action in self.createGraphActions.items():
            w = selw
            while w != None and not wc.willAllowParent(w):
                w = w.parent
            action.enable( w != None )

        # certain actions shouldn't allow root to be deleted
        isnotroot = not isinstance(selw, widgets.Root)
        
        self.editactions['cut'].enable(isnotroot)
        self.editactions['copy'].enable(isnotroot)
        self.editactions['delete'].enable(isnotroot)
        self.editactions['rename'].enable(isnotroot)
        self.editactions['moveup'].enable(isnotroot)
        self.editactions['movedown'].enable(isnotroot)

        if isnotroot:
            # cut and copy aren't currently possible on a non-widget
            cancopy = item.widget != None
            self.editactions['cut'].enable(cancopy)
            self.editactions['copy'].enable(cancopy)
       
    def _makeSettingControl(self, row, setn):
        """Construct widget for settting on the row given."""
        tooltext = "<strong>%s</strong> - %s" % (setn.name,
                                                 setn.descr)
        
        view = self.proptab.viewport()
        l = _NewPropertyLabel(setn, view)
        self.proptab.setCellWidget(row, 0, l)
        self.prefchilds.append(l)

        c = setn.makeControl(view)
        c.veusz_rownumber = row
        self.connect(c, qt.PYSIGNAL('settingChanged'), self.slotSettingChanged)
        self.proptab.setCellWidget(row, 1, c)
        qt.QToolTip.add(c, tooltext)
        self.prefchilds.append(c)

        l.control = c
        
        self.proptab.adjustRow(row)
    
    def slotItemSelected(self, item):
        """Called when an item is selected in the listview."""

        # enable or disable the create graph buttons
        self.enableCorrectButtons(item)

        self.itemselected = item
        self.updatePasteButton()

        # delete the current widgets in the preferences list
        while len(self.prefchilds) > 0:
            i = self.prefchilds.pop()

            # need line below or occasionally get random error
            # "QToolTip.maybeTip() is abstract and must be overridden"
            #qt.QToolTip.remove(i)

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
            self._makeSettingControl(row, setn)
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
            
    def slotSettingChanged(self, widget, setting, val):
        """Called when a setting is changed by the user.
        
        This updates the setting to the value using an operation so that
        it can be undone.
        """
        
        self.document.applyOperation(document.OperationSettingSet(setting, val))
            
    def slotMakeWidgetButton(self, widgettype):
        """Called when an add widget button is clicked.
        widgettype is the type of widget
        """

        self.makeWidget(widgettype)

    def makeWidget(self, widgettype, autoadd=True, name=None):
        """Called when an add widget button is clicked.
        widgettype is the type of widget
        autoadd specifies whether to add default children
        if name is set this name is used if possible (ie no other children have it)
        """

        # if no widget selected, bomb out
        if self.itemselected == None:
            return
        parent = self.getSuitableParent(widgettype)

        assert parent != None

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
            parent = self.itemselected.widget
        else:
            parent  = initialParent
        
        if parent == None:
            parent = self.itemselected.parent.widget
            assert parent != None

        # find the parent to add the child to, we go up the tree looking
        # for possible parents
        wc = document.thefactory.getWidgetClass(widgettype)
        while parent != None and not wc.willAllowParent(parent):
            parent = parent.parent

        return parent

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

    def getClipboardData(self):
        """Return veusz clipboard data or False if no data is avaliable
        The first line of the returned data is a widget type, the
        remaining lines are commands to customise the widget and add children
        """

        clipboard = qt.qApp.clipboard()
        cbSource = clipboard.data(clipboard.Clipboard)
        if not cbSource.provides(self.widgetmime):
            # Bail if the clipboard doesn't provide the data type we want
            return False
        
        data = unicode(cbSource.encodedData(self.widgetmime))
        data = data.split('\n')
        return data

    def _makeDragObject(self, widget):
        """Make a QStoredDrag object representing the subtree with the
        current selection at the root"""

        if widget:
            clipboardData = qt.QStoredDrag(self.widgetmime)
            data = str('\n'.join((widget.typename,
                widget.name,
                widget.getSaveText())))
            clipboardData.setEncodedData(data)
            return clipboardData
        else:
            return None

    def slotWidgetCut(self, a):
        """Cut the selected widget"""
        self.slotWidgetCopy(a)
        self.slotWidgetDelete(a)
        self.updatePasteButton()

    def slotWidgetCopy(self, a):
        """Copy selected widget to the clipboard."""
        clipboard = qt.qApp.clipboard()
        dragObj = self._makeDragObject(self.itemselected.widget)
        clipboard.setData(dragObj, clipboard.Clipboard)
        self.updatePasteButton()
        
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
            interpreter.interface.currentwidget = self.itemselected.widget

            # Use the command interface to create the subwidgets
            for command in data[2:]:
                interpreter.run(command)
                
            # stop the history batching
            self.document.batchHistory(None)
                
            # reset the interpreter widget
            interpreter.interface.currentwidget = tmpCurrentwidget
            
    def slotWidgetDelete(self, a):
        """Delete the widget selected."""

        # no item selected, so leave
        if self.itemselected == None:
            return

        # work out which widget to delete
        w = self.itemselected.getAssociatedWidget()
            
        # get the item to next get the selection when this widget is deleted
        # this is done by looking down the list to get the next useful one
        next = self.itemselected
        while next != None and (next.widget == w or (next.widget == None and
                                                     next.parent.widget == w)):
            next = next.itemBelow()

        # if there aren't any, use the root item
        if next == None:
            next = self.rootitem

        # remove the reference
        self.itemselected = None

        # delete selected widget
        self.document.applyOperation( document.OperationWidgetDelete(w) )

        # select the next widget
        self.listview.ensureItemVisible(next)
        self.listview.setSelected(next, True)

    def selectWidget(self, widget):
        """Select the associated listviewitem for the widget w in the
        listview."""
        
        # update in case tree does not reflect document
        self.updateContents()

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
        w = self.itemselected.getAssociatedWidget()

        # actually move the widget
        self.document.applyOperation( document.OperationWidgetMove(w, direction) )

        # try to highlight the associated item
        self.selectWidget(w)
        
    def slotListContextMenu(self, pos):
        """Pop up a context menu when an item is clicked on the list view."""

        self.contextpopup.exec_loop(pos)

    def updatePasteButton(self):
        """Is the data on the clipboard a valid paste at the currently
        selected widget?"""
        
        data = self.getClipboardData()
        show = False
        if data:
            # The first line of the clipboard data is the widget type
            widgettype = data[0]
            # Check if we can paste into the current widget or a parent
            if self.getSuitableParent(widgettype, self.itemselected.widget):
                show = True

        self.editactions['paste'].enable(show)

    def slotWidgetRename(self, action):
        """Initiate renaming a widget."""

        item = self.itemselected
        while item.widget == None:
            item = item.parent

        item.rename()

    def slotSelectWidget(self, widget):
        """The plot window says that a widget was selected, so we
        select it in the listview."""

        self.listview.selectWidget(widget)
        
