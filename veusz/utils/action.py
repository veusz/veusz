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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
###############################################################################

from __future__ import division
from .. import qtall as qt
from . import utilfuncs
import os.path
import textwrap

# where images are stored
imagedir = os.path.join(utilfuncs.resourceDirectory, 'icons')

_pixmapcache = {}
def getPixmap(pixmap):
    """Return a cached QPixmap for the filename in the icons directory."""
    if pixmap not in _pixmapcache:
        _pixmapcache[pixmap] = qt.QPixmap(os.path.join(imagedir, pixmap))
    return _pixmapcache[pixmap]

def pixmapExists(pixmap):
    """Does the pixmap exist?"""
    return (pixmap in _pixmapcache or
            os.path.exists(os.path.join(imagedir, pixmap)))

_iconcache = {}
def getIcon(icon):
    """Return a cached QIconSet for the filename in the icons directory."""
    if icon not in _iconcache:
        svg = os.path.join(imagedir, icon+'.svg')
        if os.path.exists(svg):
            filename = svg
        else:
            filename = os.path.join(imagedir, icon+'.png')

        _iconcache[icon] = qt.QIcon(filename)
    return _iconcache[icon]

def makeAction(parent, descr, menutext, slot, icon=None, key=None,
               checkable=False):
    """A quick way to set up an QAction object."""
    a = qt.QAction(parent)
    a.setText(menutext)
    a.setStatusTip(descr)
    a.setToolTip(textwrap.fill(descr, 25))
    if slot:
        a.triggered.connect(slot)
    if icon:
        a.setIcon(getIcon(icon))
    if key:
        a.setShortcut(qt.QKeySequence(key))
    if checkable:
        a.setCheckable(True)
    return a

def addToolbarActions(toolbar, actions, which):
    """Add actions listed in "which" from dict "actions" to toolbar "toolbar".
    """
    for w in which:
        toolbar.addAction(actions[w])

def constructMenus(rootobject, menuout, menutree, actions):
    """Add menus to the output dict from the tree, listing actions
    from actions.

    rootobject: QMenu or QMenuBar to add menus to
    menuout: dict to store menus
    menutree: tree structure to create menus from
    actions: dict of actions to assign to menu items
    """

    for menuid, menutext, actlist in menutree:
        # make a new menu if necessary
        if menuid not in menuout:
            menu = rootobject.addMenu(menutext)
            menuout[menuid] = menu
        else:
            menu = menuout[menuid]

        # add actions to the menu
        for action in actlist:
            if utilfuncs.isiternostr(action):
                # recurse for submenus
                constructMenus(menu, menuout, [action], actions)
            elif action == '':
                # blank means separator
                menu.addSeparator()
            else:
                # normal action
                menu.addAction(actions[action])

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
    for item in items:
        if len(item) == 1:
            if menus is not None:
                menus[item[0]].addSeparator()
            continue

        menuid, descr, menutext, menu, slot, icon, addtool, key = item

        # create action
        action = qt.QAction(parent)
        action.setText(menutext)
        action.setStatusTip(descr)
        action.setToolTip(descr)

        # set shortcut if set
        if key:
            action.setShortcut(qt.QKeySequence(key))

        # load icon if set
        if icon:
            action.setIcon(getIcon(icon))

        if callable(slot):
            # connect the action to the slot
            action.triggered.connect(slot)
            # add to menu
            if menus is not None:
                menus[menu].addAction(action)
        elif slot is not None:
            if menus is not None:
                submenu = menus[menu].addMenu(menutext)
                menus["%s.%s" % (menu ,menuid)] = submenu
                populateMenuToolbars(slot, toolbar, menus)
        else:
            if menus is not None:
                menus[menu].addAction(action)

        # add to toolbar
        if addtool and toolbar is not None:
            toolbar.addAction(action)

        # save for later
        actions[menuid] = action

    return actions

def makeMenuGroupSaved(name, parent, actiondict, actionnames):
    """Assign a menu and group for an action which allows the user to
    choose from a list of other actions.

    name: name of action to assign to (in actiondict)
    parent: parent qt object
    actiondict: map of names to actions
    actionnames: name of actions to add to menu/action.
    """

    from .. import setting

    menu = qt.QMenu(parent)
    actgrp = qt.QActionGroup(parent)

    act_to_name = {}

    for actname in actionnames:
        action = actiondict[actname]
        act_to_name[action] = actname
        menu.addAction(action)
        actgrp.addAction(action)

    # assign menu to action in dictionary
    menuaction = actiondict[name]
    menuaction.setMenu(menu)

    # currently set value (per control)
    current = [
        setting.settingdb.get('menugrp_%s' % name, actionnames[0]) ]

    def ongrptriggered(action):
        """Update saved action when new one is chosen."""
        actname = act_to_name[action]
        setting.settingdb['menugrp_%s' % name] = current[0] = actname
        menuaction.setIcon(action.icon())

    def onactiontriggered():
        """Run the action chosen from the list."""
        actiondict[current[0]].trigger()

    menuaction.triggered.connect(onactiontriggered)
    actgrp.triggered.connect(ongrptriggered)
    menuaction.setIcon(actiondict[current[0]].icon())
