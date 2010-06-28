# tree interface to embedding

class Node(object):
    def __init__(self, ci, path):
        self._ci = ci
        self._path = path
        self._type = self._ci.NodeType(path)

    @property
    def path(self):
        """Veusz path to node"""
        return self._path

    @property
    def type(self):
        """Type of node: 'widget', 'settings', or 'setting'"""
        return self._type

    def _joinPath(self, child):
        """Return path of child."""
        if self._path == '/':
            return '/' + child
        else:
            return self._path + '/' + child

    def __getitem__(self, attr):
        if self._type == 'setting':
            raise TypeError, "Cannot get subsetting of setting"""
        return Node(self._ci, self._joinPath(attr))

    def __getattr__(self, attr):
        return self.__getitem__(attr)

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

    @property
    def children(self):
        """Return children of node (generator)."""
        for c in self._ci.NodeChildren(self._path):
            yield Node(self._ci, self._joinPath(c))

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
        return Node(self._ci, newpath)

    @property
    def name(self):
        """Get name of node."""
        if self._path == '/':
            return self._path
        else:
            return self._path.split('/')[-1]

    def Add(self, widgettype, *args, **args_opt):
        """Add a widget of the type given, returning the Node instance.

        Only valid for widget nodes."""
        if self._type != 'widget':
            raise TypeError, "Cannot add a widget to a non-widget"""

        args_opt['widget'] = self._path
        name = self._ci.Add(widgettype, *args, **args_opt)
        return Node( self._ci, self._joinPath(name) )
