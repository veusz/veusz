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

import qt

class _WidgetItem(qt.QListViewItem):
    """Item for displaying in the TreeEditWindow."""

    def __init__(self, widget, *args):
        qt.QListViewItem.__init__(self, *args)

        self.widget = widget

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

class TreeEditWindow(qt.QDockWindow):
    """A graph editing window with tree display."""

    def __init__(self, thedocument, *args):
        qt.QDockWindow.__init__(self, *args)
        self.setResizeEnabled( True )
        self.setCaption("Graph tree - Veusz")

        self.document = thedocument
        self.connect( self.document, qt.PYSIGNAL("sigModified"),
                      self.slotModifiedDoc )

        # put widgets in a vbox
        self.vbox = qt.QVBox(self)
        self.setWidget(self.vbox)

        # first widget is a listview
        lv = qt.QListView(self.vbox)
        lv.setSorting(-1)
        lv.setRootIsDecorated(True)

        lv.addColumn( "Name" )
        lv.addColumn( "Type" )
        lv.addColumn( "Detail" )
        
        self.rootitem = _WidgetItem( self.document.getBaseWidget(), lv )
        self.listview = lv

    def slotModifiedDoc(self, ismodified):
        """Called when the document has been modified."""
 
        if ismodified:
            self.updateContents()

    def _updateBranch(self, root):
        """Recursively update items on branch."""

        childwidgets = root.getWidget().getChildren()
        item = root.firstChild()

        newchild = False

        for c in childwidgets:

            if item == None:
                # add a new item if there are no more
                item = _WidgetItem(c, root)
                newchild = True
            elif item.getWidget() != c:
                # replace an item if it is wrong
                newitem = _WidgetItem(c, root, item)
                root.takeItem(item)
                del item
                item = newitem
                newchild = True

            # recursively update branches
            self._updateBranch(item)
            item = item.nextSibling()

        # open the branch if we've added/changed the children
        if newchild:
            self.listview.setOpen(root, True)

    def updateContents(self):
        """Make the window reflect the document."""
        self._updateBranch(self.rootitem)
        self.listview.triggerUpdate()
