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

class _WidgetItem(qt.QListViewItem):
    """Item for displaying in the TreeEditWindow."""

    def __init__(self, widget, *args):
        qt.QListViewItem.__init__(self, *args)

        self.widget = widget

        # add subitems for sub-prefs of widget
        for name, pref in widget.getSubPrefs().items():
            _PrefItem(pref, name, self)

    def text(self, column):
        """Get the text in a particular column."""
        if column == 0:
            return self.widget.getName()
        elif column == 1:
            return self.widget.getTypeName()
        elif column == 2:
            return self.widget.getUserDescription()
        return ""

    def getWidget(self):
        """Get the associated widget."""
        return self.widget

class _PrefItem(qt.QListViewItem):
    """Item for displaying a preferences-set in TreeEditWindow."""
    def __init__(self, preftype, name, *args):
        qt.QListViewItem.__init__(self, *args)

        self.preftype = preftype
        self.setText(0, name)
        self.setText(1, "setting")

    def getWidget(self):
        """Returns None as we don't have a widget."""
        return None

    def getPreftype(self):
        """Returns the preferences."""
        return self.preftype

class _ScrollView(qt.QScrollView):
    """A resizing scroll view."""

    def __init__(self, *args):
        qt.QScrollView.__init__(self, *args)

    def setGrid(self, grid):
        self.grid = grid

    def adjustSize(self):
        h = self.grid.height()
        #print self.visibleWidth(), h
        self.grid.resize(self.visibleWidth(), h)
        self.resizeContents(self.visibleWidth(), h)

    def resizeEvent(self, event):
        #self.moveChild(self.grid, 0, 0)
        qt.QScrollView.resizeEvent(self, event)
        self.adjustSize()
        #print "Here"

class TreeEditWindow(qt.QDockWindow):
    """A graph editing window with tree display."""

    def __init__(self, thedocument, *args):
        qt.QDockWindow.__init__(self, *args)
        self.setResizeEnabled( True )
        self.setCaption("Graph tree - Veusz")

        self.document = thedocument
        self.connect( self.document, qt.PYSIGNAL("sigModified"),
                      self.slotModifiedDoc )

        self.tooltips = qt.QToolTip(self)

        totvbox = qt.QVBox(self)
        self.setWidget(totvbox)

        # make buttons for each of the graph types
        buttonhbox = qt.QHBox(totvbox)
        dir = os.path.dirname(__file__)
        for i in widgetfactory.thefactory.listWidgets():
            if widgetfactory.thefactory.getWidgetClass(i).allowusercreation:
                name = "%s/button_%s.png" % (dir, i)
                b = qt.QToolButton(buttonhbox)
                b.setPixmap( qt.QPixmap(name) )

                try:
                    self.tooltips.add(b, widgetfactory.thefactory.\
                                      getWidgetClass(i).description)
                except AttributeError:
                    pass
                #b.setFlat( True )
                #b.setWFlags( qt.Qt.WStyle_NoBorder )

        # put widgets in a movable splitter
        split = qt.QSplitter(totvbox)
        split.setOrientation(qt.QSplitter.Vertical)
        #self.setWidget(split)

        # first widget is a listview
        lv = qt.QListView(split)
        lv.setSorting(-1)
        lv.setRootIsDecorated(True)
        self.connect( lv, qt.SIGNAL("selectionChanged(QListViewItem*)"),
                      self.slotItemSelected )

        lv.addColumn( "Name" )
        lv.addColumn( "Type" )
        lv.addColumn( "Detail" )
        
        self.rootitem = _WidgetItem( self.document.getBaseWidget(), lv )
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

    def slotModifiedDoc(self, ismodified):
        """Called when the document has been modified."""
 
        if ismodified:
            self.updateContents()

    def _updateBranch(self, root):
        """Recursively update items on branch."""

        # collect together a list of treeitems (in original order)
        # ignore those that don't correspond to widgets
        items = []
        i = root.firstChild()
        while i != None:
            if i.getWidget() != None:
                items.insert(0, i)
            i = i.nextSibling()

        childs = root.getWidget().getChildren()
        newchild = False

        # go through the list and update those which have changed
        for i, c in zip(items, childs):
            if i.getWidget() != c:
                # add in new item after the changed one
                new = _WidgetItem(c, i)
                # remove the original
                root.takeItem(i)
                i = new
                newChild = True

            self._updateBranch(i)
            
        # if we need to add new items
        for i in childs[len(items):]:
            new = _WidgetItem(i, root)
            self._updateBranch(new)
            newchild = True

        # if we need to delete old items
        for i in items[len(childs):]:
            root.takeItem(i)
            del i
            newchild = True
        
        # open the branch if we've added/changed the children
        if newchild:
            self.listview.setOpen(root, True)

    def updateContents(self):
        """Make the window reflect the document."""
        self._updateBranch(self.rootitem)
        self.listview.triggerUpdate()

    def slotCntrlModifiesDoc(self):
        """Called when a control modifies its value."""
        self.document.setModified(True)

    def slotItemSelected(self, item):
        """Called when an item is selected in the listview."""

        s = self.prefview.size()
        self.prefview.adjustSize()

        # delete the current widgets in the preferences list
        for i in self.prefchilds:
            i.deleteLater()
        self.prefchilds = []

        if item.getWidget() != None:
            prefs = item.getWidget().getPrefs()
        else:
            prefs = item.getPreftype().getPrefs()

        # make new widgets for the preferences
        for pname in prefs.getPrefNames():
            l = qt.QLabel(pname, self.prefgrid)
            l.show()
            self.prefchilds.append(l)

            type = prefs.getPrefType(pname)
            val = prefs.getPref(pname)

            cntrl = prefs.makePrefControl(pname, self.prefgrid)
            cntrl.connect(cntrl, qt.PYSIGNAL('sigModified'),
                          self.slotCntrlModifiesDoc)
            cntrl.show()
            self.prefchilds.append(cntrl)

        # UUGH - KLUDGE! Have to do this before program takes notice
        # of adjustSize below!
        # FIXME
        qt.QApplication.eventLoop().processEvents(qt.QEventLoop.AllEvents,
                                                  100)
        self.prefview.adjustSize()
            
