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
import widgets.widgetfactory as widgetfactory
import widgets

class _WidgetItem(qt.QListViewItem):
    """Item for displaying in the TreeEditWindow."""

    def __init__(self, widget, qtparent):
        """Widget is the widget to show the settings for."""
        
        qt.QListViewItem.__init__(self, qtparent)

        self.index = '0000'
        self.widget = widget
        self.settings = widget.settings

        # add subitems for sub-prefs of widget
        no = 0
        for s in self.settings.getSettingsList():
            _PrefItem(s, no, self)
            no += 1

    def setIndex(self, index):
        """Set the sorting index."""
        self.index = '%04i' % index

    def text(self, column):
        """Get the text in a particular column."""
        if column == 0:
            return self.widget.name
        elif column == 1:
            return self.widget.typename
        elif column == 2:
            return self.widget.userdescription
        elif column == 3:
            return self.index
        return ""

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
        self.setText(1, "setting")
        self.setText(3, '%04i' % number)

class _ScrollView(qt.QScrollView):
    """A resizing scroll view."""

    def __init__(self, *args):
        qt.QScrollView.__init__(self, *args)

    def setGrid(self, grid):
        self.grid = grid

    def adjustSize(self):
        h = self.grid.height()
        self.grid.resize(self.visibleWidth(), h)
        self.resizeContents(self.visibleWidth(), h)

    def resizeEvent(self, event):
        qt.QScrollView.resizeEvent(self, event)
        self.adjustSize()

class TreeEditWindow(qt.QDockWindow):
    """A graph editing window with tree display."""

    def __init__(self, thedocument, parent):
        qt.QDockWindow.__init__(self, parent)
        self.setResizeEnabled( True )
        self.setCaption("Graph tree - Veusz")

        self.parent = parent
        self.document = thedocument
        self.connect( self.document, qt.PYSIGNAL("sigModified"),
                      self.slotDocumentModified )
        self.connect( self.document, qt.PYSIGNAL("sigWiped"),
                      self.slotDocumentWiped )

        totvbox = qt.QVBox(self)
        self.setWidget(totvbox)

        # make buttons for each of the graph types
        self.createGraphButtons = {}
        buttonhbox = qt.QHBox(totvbox)
        #buttonhbox = qt.QToolBar(self.parent)
        mdir = os.path.dirname(__file__)

        for w in widgetfactory.thefactory.listWidgets():
            wc = widgetfactory.thefactory.getWidgetClass(w)
            if wc.allowusercreation:
                # make a new button, and set the pixmap
                b = qt.QToolButton(buttonhbox)
                b.setFocusPolicy(qt.QWidget.TabFocus)
                name = "%s/icons/button_%s.png" % (mdir, w)
                b.setPixmap( qt.QPixmap(name) )

                # keep track of the buttons so we can disable/enable them
                self.createGraphButtons[wc] = b

                # set the tooltip to the graph description
                try:
                    qt.QToolTip.add(b, wc.description)
                except AttributeError:
                    pass

                b.widgetname = w
                self.connect(b, qt.SIGNAL('clicked()'),
                             self.slotMakeWidgetButton)

        # make buttons for user actions
        edithbox = qt.QHBox(totvbox)
        self.editbuttons = {}

        for name, icon, tooltip in (
            ('moveup', 'go-up', 'Move the selected widget up'),
            ('movedown', 'go-down', 'Move the selected widget down'),
            ('delete', 'delete', 'Remove the selected widget')
            ):

            b = qt.QToolButton(edithbox)
            b.setFocusPolicy(qt.QWidget.TabFocus)
            b.setPixmap( qt.QPixmap("%s/icons/stock-%s.png" % (mdir, icon)) )
            qt.QToolTip.add(b, tooltip)
            self.editbuttons[name] = b

        # put widgets in a movable splitter
        split = qt.QSplitter(totvbox)
        split.setOrientation(qt.QSplitter.Vertical)

        # first widget is a listview
        lv = qt.QListView(split)
        lv.setSorting(-1)
        lv.setRootIsDecorated(True)
        self.connect( lv, qt.SIGNAL("selectionChanged(QListViewItem*)"),
                      self.slotItemSelected )

        # we use a hidden column to get the sort order correct
        lv.addColumn( "Name" )
        lv.addColumn( "Type" )
        lv.addColumn( "Detail" )
        lv.addColumn( "Sort order")
        lv.setColumnWidth(3, 0)
        lv.setColumnWidthMode(3, qt.QListView.Manual)
        lv.setSorting(3)
        lv.setTreeStepSize(10)

        # add root widget to view
        self.rootitem = _WidgetItem( self.document.basewidget, lv )
        self.listview = lv

        # add a scrollable view for the preferences
        # children get added to prefview
        vbox = qt.QVBox(split)
        label = qt.QLabel("Properties", vbox)
        label.setMargin(2)
        self.prefview = _ScrollView(vbox)

        self.prefgrid = qt.QGrid(2, qt.QGrid.Horizontal,
                                 self.prefview.viewport())
        self.prefview.setGrid(self.prefgrid)
        self.prefview.addChild(self.prefgrid)
        self.prefgrid.setMargin(4)
        self.prefchilds = []

        # select the root item
        self.listview.setSelected(self.rootitem, True)

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
        index = 1000
        newitem = False
        for c in root.widget.children:
            childdict[c] = True
            if c in items:
                items[c].setIndex(index)
            else:
                items[c] = _WidgetItem(c, root)
                items[c].setIndex(index)
                newitem = True
            self._updateBranch(items[c])
            index += 1

        # delete items not in child list
        for i in items.itervalues():
            if i.widget not in childdict:
                root.takeItem(i)

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

        # check whether each button can have this widget as parent
        for wc, button in self.createGraphButtons.items():
            button.setEnabled( wc.willAllowParent(selw) )

        # delete shouldn't allow root to be deleted
        self.editbuttons['delete'].setEnabled(
            not isinstance(selw, widgets.Root) )

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

        w = item.widget
        # add action for widget
        if w != None:
            for name in w.actions:
                l = qt.QLabel(name, self.prefgrid)
                l.show()
                self.prefchilds.append(l)

                b = qt.QPushButton(w.actiondescr[name], self.prefgrid)
                b.veusz_action = w.actionfuncs[name]
                b.show()
                self.prefchilds.append(b)
                
                self.connect( b, qt.SIGNAL('pressed()'),
                              self.slotActionPressed )

        # make new widgets for the preferences
        for setn in item.settings.getSettingList():
            b = qt.QPushButton(setn.name, self.prefgrid)
            b.setFlat(True)
            b.setMinimumWidth(10)
            b.veuszSetting = setn
            qt.QToolTip.add(b, setn.descr)
            b.show()
            self.prefchilds.append(b)
            self.connect( b, qt.SIGNAL('pressed()'),
                          self.slotLabelButtonPressed )
            
            c = setn.makeControl(self.prefgrid)
            qt.QToolTip.add(c, setn.descr)
            c.show()
            self.prefchilds.append(c)

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
        
        # UUGH - KLUDGE! Have to do this before program takes notice
        # of adjustSize below!
        # FIXME
        qt.QApplication.eventLoop().processEvents(qt.QEventLoop.AllEvents,
                                                  100)
        self.prefview.adjustSize()
            
    def slotMakeWidgetButton(self):
        """Called when an add widget button is clicked."""

        # get the widget to act as the parent
        parent = self.itemselected.widget
        if parent == None:
            parent = self.itemselected.parent.widget
            assert parent != None

        # make the new widget and update the document
        widgetfactory.thefactory.makeWidget(self.sender().widgetname,
                                            parent)
        self.document.setModified()

    def slotActionPressed(self):
        """Called when an action button is pressed."""

        # get the button clicked/activated
        button = self.sender()

        # set focus to button to make sure other widgets lose
        # focus and update their settings
        button.setFocus()

        # run action in console
        action = button.veusz_action
        console = self.parent.console
        console.runFunction( action )

    def slotLabelButtonPressed(self):
        """Called when one of the label buttons is pressed.

        This pops up a menu allowing propagation of values, resetting to
        default, or making the default
        """

        # get the button pressed
        button = self.sender()
        button.setFocus()
        setn = button.veuszSetting

        # find the selected widget, get its type and name
        widget = self.itemselected.widget
        if widget == None:
            widget = self.itemselected.parent.widget
        type = widget.typename
        name = widget.name
        
        # construct the popup menu
        popup = qt.QPopupMenu(button)
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
        
        ret = popup.exec_loop(
            button.mapToGlobal( qt.QPoint(0, button.height()) ))

        # convert values above to functions
        doc = self.document
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
