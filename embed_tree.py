#    Copyright (C) 2010 Jeremy S. Sanders
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
##############################################################################

# $Id$

"""Tree interface to embedding."""

class Node(object):
    """Represents an element in the Veusz widget-settinggroup-setting tree."""

    def __init__(self, ci, wtype, path):
        self._ci = ci
        self._type = wtype
        self._path = path

    @staticmethod
    def _makeNode(ci, path):
        """Make correct class for type of object."""
        wtype = ci.NodeType(path)
        if wtype == 'widget':
            return WidgetNode(ci, wtype, path)
        elif wtype == 'setting':
            return SettingNode(ci, wtype, path)
        else:
            return SettingGroupNode(ci, wtype, path)

    @property
    def path(self):
        """Veusz full path to node"""
        return self._path

    @property
    def type(self):
        """Type of node: 'widget', 'settinggroup', or 'setting'"""
        return self._type

    def _joinPath(self, child):
        """Return new path of child."""
        if self._path == '/':
            return '/' + child
        else:
            return self._path + '/' + child

    def __getitem__(self, attr):
        if self._type == 'setting':
            raise AttributeError, "Does not have attribute '%s'" % attr
        return self._makeNode(self._ci, self._joinPath(attr))

    def __getattr__(self, key):
        if self._type == 'setting':
            raise KeyError, "Does not have key '%s'" % key
        return self._makeNode(self._ci, self._joinPath(key))

    @property
    def children(self):
        """Return children as Nodes (generator)."""
        for c in self._ci.NodeChildren(self._path):
            yield self._makeNode(self._ci, self._joinPath(c))

    @property
    def parent(self):
        """Return parent of node."""
        if self._path == '/':
            raise TypeError, "Cannot get parent node of root node"""
        p = self._path.split('/')[:-1]
        if p == ['']:
            newpath = '/'
        else:
            newpath = '/'.join(p)
        return self._makeNode(self._ci, newpath)

    @property
    def name(self):
        """Get name of node."""
        if self._path == '/':
            return self._path
        else:
            return self._path.split('/')[-1]

class SettingNode(Node):
    """A node which is a setting."""

    def _getVal(self):
        """The value of a setting."""
        if self._type == 'setting':
            return self._ci.Get(self._path)
        raise TypeError, "Cannot get value unless is a setting"""

    def _setVal(self, val):
        if self._type == 'setting':
            self._ci.Set(self._path, val)
        else:
            raise TypeError, "Cannot set value unless is a setting."""

    val = property(_getVal, _setVal)

class SettingGroupNode(Node):
    """A node containing a group of settings."""

    pass

class WidgetNode(Node):
    """A node pointing to a widget."""

    def Add(self, widgettype, *args, **args_opt):
        """Add a widget of the type given, returning the Node instance.
        """

        args_opt['widget'] = self._path
        name = self._ci.Add(widgettype, *args, **args_opt)
        return WidgetNode( self._ci, 'widget', self._joinPath(name) )

    def Rename(self, newname):
        """Renames widget to name given."""

        if self._path == '/':
            raise RuntimeError, "Cannot rename root widget"

        self._ci.Rename(self._path, newname)
        self._path = '/'.join( self._path.split('/')[:-1] + [newname] )
        
    def Action(self, action):
        """Applies action on widget."""
        self._ci.Action(action, widget=self._path)

    def Remove(self):
        """Removes a widget and its children."""
        self._ci.Remove(self._path)
