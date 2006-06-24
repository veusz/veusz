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

import veusz.qtall as qt
import os.path

# where images are stored
imagedir = os.path.join(os.path.dirname(__file__), 'icons')

_pixmapcache = {}
def getPixmap(pixmap):
    """Return a cached QPixmap for the filename in the icons directory."""
    if pixmap not in _pixmapcache:
        _pixmapcache[pixmap] = qt.QPixmap(os.path.join(imagedir, pixmap))
    return _pixmapcache[pixmap]

_iconcache = {}
def getIcon(icon):
    """Return a cached QIconSet for the filename in the icons directory."""
    if icon not in _iconcache:
        pixmap = getPixmap(icon)
        _iconcache[icon] = qt.QIcon(pixmap)
    return _iconcache[icon]


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
                menus[i[0]].addSeparator()
            continue
        
        menuid, descr, menutext, menu, slot, icon, addtool, key = i
        if key == '':
            ks = qt.QKeySequence()
        else:
            ks = qt.QKeySequence(key)

        #action = qt.QAction(descr, menutext, ks, parent)
        # FIXMEQT4
        action = qt.QAction(parent)
        action.setText(menutext)
        action.setStatusTip(descr)
        action.setToolTip(descr)
        action.setShortcut(ks)

        # load icon if set
        if icon != '':
            action.setIcon(getIcon(icon))

        if callable(slot):
            # connect the action to the slot
            if slot is not None:
                qt.QObject.connect( action, qt.SIGNAL('activated()'), slot )
                # add to menu
            if menus is not None:
                menus[menu].addAction(action)
        elif slot is not None:
            if menus is not None:
                submenu = menus[menu].addMenu(menutext)
                menus["%s.%s"%(menu ,menuid)] = submenu
                populateMenuToolbars(slot, toolbar, menus)
        else:
            if menus is not None:
                action.addTo( menus[menu] )
                

        # add to toolbar
        if addtool and toolbar != None:
            toolbar.addAction(action)

        # save for later
        actions[menuid] = action

    return actions
