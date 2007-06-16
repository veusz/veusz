## action.py
# A QAction-like object which can add buttons to things other than
# QToolBars

#    Copyright (C) 2005 Jeremy S. Sanders
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
import os.path

# where images are stored
imagedir = os.path.join(os.path.dirname(__file__), 'icons')

_pixmapcache = {}
def getPixmap(pixmap):
    """Return a cached QPixmap for the filename in the icons directory."""
    if pixmap not in _pixmapcache:
        _pixmapcache[pixmap] = qt.QPixmap(os.path.join(imagedir, pixmap))
    return _pixmapcache[pixmap]

_iconsetcache = {}
def getIconSet(icon):
    """Return a cached QIconSet for the filename in the icons directory."""
    if icon not in _iconsetcache:
        pixmap = getPixmap(icon)
        _iconsetcache[icon] = qt.QIconSet(pixmap)
    return _iconsetcache[icon]

class Action(qt.QObject):
    """A QAction-like object for associating actions with buttons,
    menu items, and so on."""

    def __init__(self, parent, onaction, iconfilename = None, menutext = None,
                 tooltiptext = None, statusbartext = None, accel=None):
        qt.QObject.__init__(self)

        self.parent = parent
        self.onaction = onaction
        self.menutext = menutext
        self.tooltiptext = tooltiptext
        self.statusbartext = statusbartext
        self.items = []
        if accel:
            self.accel = qt.QKeySequence(accel)
        else:
            self.accel = None
            
        if self.statusbartext == None:
            if self.menutext != None:
                # if there is no text, use the status bar text (removing ...)
                self.statusbartext = self.menutext.replace('...', '')
                self.statusbartext = self.statusbartext.replace('&', '')
            elif self.tooltiptext != None:
                # else try the tooltiptext
                self.statusbartext = self.tooltiptext
            else:
                # leave it blank
                self.statusbartext = ''

        # makes life easier late
        if self.tooltiptext == None:
            # use the menu text if there is no tooltip
            if self.menutext != None:
                self.tooltiptext = self.menutext.replace('...', '')
                self.tooltiptext = self.tooltiptext.replace('&', '')
            else:
                self.tooptiptext = ''
        if self.menutext == None:
            self.menutext = ''

        # find status bar
        mainwin = parent
        while not isinstance(mainwin, qt.QMainWindow) and mainwin != None:
            mainwin = mainwin.parentWidget()

        if mainwin == None:
            self.tipgroup = None
            self.statusbar = None
        else:
            self.tipgroup = mainwin.toolTipGroup()
            self.statusbar = mainwin.statusBar()

        # make icon set
        if iconfilename != None:
            self.iconset = getIconSet(iconfilename)
        else:
            self.iconset = None

    def addTo(self, widget):
        """Add the action to the given widget.

        widget can be an instance of QToolBar, QPopupMenu, a Q[HV]Box..."""

        if isinstance(widget, qt.QPopupMenu):
            if self.iconset != None:
                num = widget.insertItem(self.iconset, self.menutext)
            else:
                num = widget.insertItem(self.menutext)
            if self.accel:
                widget.setAccel(self.accel, num)
            widget.connectItem(num, self.slotAction)
            self.items.append( (widget, num) )

            self.connect(widget, qt.SIGNAL('highlighted(int)'),
                         self.slotMenuHighlighted)
            self.connect(widget, qt.SIGNAL('aboutToHide()'),
                         self.slotMenuAboutToHide)

            return num

        else:
            b = qt.QToolButton(widget)
            if self.iconset != None:
                b.setIconSet(self.iconset)
            if self.tipgroup == None:
                qt.QToolTip.add(b, self.tooptiptext)
            else:
                qt.QToolTip.add(b, self.tooltiptext,
                                self.tipgroup, self.statusbartext)
            self.connect(b, qt.SIGNAL('clicked()'),
                         self.slotAction)

            self.items.append( (b,) )

            return b

    def enable(self, enabled = True):
        """Enabled the item in the connected objects."""

        for i in self.items:
            if isinstance(i[0], qt.QButton):
                i[0].setEnabled(enabled)
            elif isinstance(i[0], qt.QPopupMenu):
                i[0].setItemEnabled(i[1], enabled)
            else:
                assert False

    def slotAction(self):
        """Called when the action is activated."""
        self.onaction( self )

    def slotMenuHighlighted(self, id):
        """Called when the user hovers over a menu item."""

        # popup menu which called us
        widget = self.sender()
        if (widget, id) in self.items and self.statusbar != None:
            self.statusbar.message(self.statusbartext)

    def slotMenuAboutToHide(self):
        """Menu about to go, so we hide the status bar text."""
        self.statusbar.clear()

def populateMenuToolbars(items, toolbar, menus):
    """Construct the menus and toolbar from the list of items.
    toolbar is a QToolbar object
    menus is a dict of menus to add to

    Items are tuples consisting of:
    (actioname, status bar text, menu text, menu id, slot,
     icon filename, add to toolbar (bool), shortcut text)

    Returns a dict of actions
    """

    actions = {}
    parent = toolbar.parent()
    for i in items:
        if len(i) == 1:
            if menus != None:
                menus[i[0]].insertSeparator()
            continue
        
        menuid, descr, menutext, menu, slot, icon, addtool, key = i
        if key == '':
            ks = qt.QKeySequence()
        else:
            ks = qt.QKeySequence(key)

        action = qt.QAction(descr, menutext, ks, parent)

        # load icon if set
        if icon != '':
            action.setIconSet(getIconSet(icon))

        if callable(slot):
            # connect the action to the slot
            if slot is not None:
                qt.QObject.connect( action, qt.SIGNAL('activated()'), slot )
                # add to menu
            if menus is not None:
                action.addTo( menus[menu] )
        elif slot is not None:
            if menus is not None:
                submenu = qt.QPopupMenu(menus[menu].parentWidget())
                menus["%s.%s"%(menu ,menuid)] = submenu
                menus[menu].insertItem(menutext,submenu)
                populateMenuToolbars(slot, toolbar, menus)
        else:
            if menus is not None:
                action.addTo( menus[menu] )
                

        # add to toolbar
        if addtool and toolbar != None:
            action.addTo(toolbar)

        # save for later
        actions[menuid] = action

    return actions
