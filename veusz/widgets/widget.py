# widget.py
# fundamental graph plotting widget

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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

from __future__ import division
import itertools

from ..compat import czip, crepr
from .. import document
from .. import setting
from .. import qtall as qt4

def _(text, disambiguation=None, context='Widget'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class Action(object):
    """A class to wrap functions operating on widgets.

    Attributes:
    name: name of action
    function: function to call with no arguments
    descr: description of action
    usertext: name of action to display to user
    """

    def __init__(self, name, function, descr='', usertext=''):
        """Initialise Action

        Name of action is name
        Calls function function() on invocation
        Action has description descr
        Usertext is short form of name to display to user."""

        self.name = name
        self.function = function
        self.descr = descr
        self.usertext = usertext

class Widget(object):
    """ Fundamental plotting widget interface."""

    # differentiate widgets, settings and setting
    nodetype = 'widget'

    typename = 'generic'
    allowusercreation = False

    isaxis = False
    isplotter = False

    # various items in class hierarchy
    iswidget = True
    issetting = False
    issettings = False

    def __init__(self, parent, name=None):
        """Initialise a blank widget."""

        # save parent widget for later
        self.parent = parent
        self.document = None

        if not self.isAllowedParent(parent):
            raise RuntimeError("Widget parent is of incorrect type")

        if name is None:
            name = self.chooseName()
        self.name = name

        # propagate document
        if parent is not None:
            self.document = parent.document
            parent.addChild(self)

        # store child widgets
        self.children = []
        
        # settings for widget
        self.settings = setting.Settings(
            'Widget_' + self.typename,
            setnsmode='widgetsettings')
        self.settings.parent = self

        self.addSettings(self.settings)

        # actions for widget
        self.actions = []

    @classmethod
    def allowedParentTypes(klass):
        """Get types of widgets this can be a child of."""
        return ()

    @classmethod
    def addSettings(klass, s):
        """Add items to settings s."""
        s.add( setting.Bool('hide', False,
                            descr = _('Hide object'),
                            usertext = _('Hide'),
                            formatting = True) )

    def getDocument(self):
        """Return document.
        Unfortunately we need this as document is shadowed in StyleSheet,
        sigh."""
        return self.document

    def rename(self, name):
        """Change name of self."""

        if self.parent is None:
            raise ValueError('Cannot rename root widget')

        if name.find('/') != -1:
            raise ValueError('Names cannot contain "/"')

        # check whether name already exists in siblings
        for i in self.parent.children:
            if i != self and i.name == name:
                raise ValueError('New name "%s" already exists' % name)

        self.name = name

    def addDefaultSubWidgets(self):
        '''Add default sub widgets to widget, if any'''
        pass

    def addAction(self, action):
        """Assign name to operation.
        action is action class above
        """
        self.actions.append( action )

    def getAction(self, name):
        """Get action associated with name."""
        for a in self.actions:
            if a.name == name:
                return a
        return None

    def isAllowedParent(self, parent):
        """Is the parent a suitable type?"""

        return parent is None or any(
            ( isinstance(parent, t) for t in self.allowedParentTypes() ) )

    @classmethod
    def willAllowParent(cls, parent):
        """Is the parent of an allowed type to have this type as a child?"""

        # allow base widget to have no parent
        ap = cls.allowedParentTypes()
        if parent is None and len(ap) > 0 and ap[0] is None:
            return True

        for p in ap:
            if isinstance(parent, p):
                return True
        return False

    def addChild(self, child, index=9999999):
        """Add child to list.
        
        index is a position to place the new child
        """
        self.children.insert(index, child)

    def createUniqueName(self, prefix):
        """Create a name using the prefix which hasn't been used before."""
        names = self.childnames

        i = 1
        while "%s%i" % (prefix, i) in names:
            i += 1
        return "%s%i" % (prefix, i)

    def chooseName(self):
        """Make a name for widget if not specified."""

        if self.parent is None:
            return '/'
        else:
            return self.parent.createUniqueName(self.typename)

    @property
    def userdescription(self):
        """Return a user-friendly description of what
        this is (e.g. function)."""
        return ''

    def getChild(self, name):
        """Return a child with a name."""
        #print('getChild', self, name)
        for i in self.children:
            if i.name == name:
                return i
        return None

    def hasChild(self, name):
        """Return whether there is a child with a name."""
        return self.getChild(name) is not None

    @property
    def childnames(self):
        """Return the child names."""
        return [i.name for i in self.children]

    def removeChild(self, name):
        """Remove a child."""

        i = 0
        nc = len(self.children)
        while i < nc and self.children[i].name != name:
            i += 1

        if i < nc:
            self.children.pop(i)
        else:
            raise ValueError("Cannot remove graph '%s' - does not exist" % name)

    def widgetSiblingIndex(self):
        """Get index of widget in its siblings."""
        if self.parent is None:
            return 0
        else:
            return self.parent.children.index(self)

    @property
    def path(self):
        """Returns a path for the object, e.g. /plot1/x."""

        obj = self
        build = ''
        while obj.parent is not None:
            build = '/' + obj.name + build
            obj = obj.parent

        if len(build) == 0:
            build = '/'

        return build

    def getMargins(self, painthelper):
        """Return margins of widget."""
        return (0., 0., 0., 0.)

    def computeBounds(self, parentposn, painthelper, withmargin=True):
        """Compute a bounds array, giving the bounding box for the widget."""

        if withmargin:
            x1, y1, x2, y2 = parentposn
            dx1, dy1, dx2, dy2 = self.getMargins(painthelper)
            return [ x1+dx1, y1+dy1, x2-dx2, y2-dy2 ]
        else:
            return parentposn

    def draw(self, parentposn, painthelper, outerbounds = None):
        """Draw the widget and its children in posn (a tuple with x1,y1,x2,y2).

        painter is the widget.Painter to draw on
        outerbounds contains "ultimate" bounds we don't go outside
        """

        bounds = self.computeBounds(parentposn, painthelper)

        if not self.settings.hide:

            # iterate over children in reverse order
            for c in reversed(self.children):
                c.draw(bounds, painthelper, outerbounds=outerbounds)
 
        # return our final bounds
        return bounds

    def getSaveText(self, saveall = False):
        """Return text to restore object

        If saveall is true, save everything, including defaults."""

        # set everything first
        text = self.settings.saveText(saveall)

        # now go throught the subwidgets
        for c in self.children:
            text += ( "Add('%s', name=%s, autoadd=False)\n" %
                      (c.typename, crepr(c.name)) )

            # if we need to go to the child, go there
            ctext = c.getSaveText(saveall)
            if ctext != '':
                text += ("To(%s)\n"
                         "%s"
                         "To('..')\n") % (crepr(c.name), ctext)

        return text

    def linkToStylesheet(self):
        """Links settings to stylesheet."""
        self.settings.linkToStylesheet()

    def buildFlatWidgetList(self, thelist):
        """Return a built up list of the widgets in the tree."""

        thelist.append(self)
        for child in self.children:
            child.buildFlatWidgetList(thelist)

    def _recursiveBuildSlots(self, slots):
        """Build up a flat representation of the places where widgets
        can be placed

        The list consists of (parent, index) tuples
        """

        slots.append( (self, 0) )

        for child, index in czip(self.children, itertools.count(1)):
            child._recursiveBuildSlots(slots)
            slots.append( (self, index) )

    def moveChild(self, w, direction):
        """Move the child widget w up in the hierarchy in the direction.
        direction is -1 for 'up' or +1 for 'down'

        Returns True if succeeded
        """

        # find position of child in self
        c = self.children
        oldindex = c.index(w)

        # remove the widget from its current location
        c.pop(oldindex)

        # build a list of places widgets can be placed (slots)
        slots = []
        self.document.basewidget._recursiveBuildSlots(slots)

        # find self list - must be a better way to do this -
        # probably doesn't matter too much, however
        ourslot = (self, oldindex)
        ourindex = 0
        while ourindex < len(slots) and slots[ourindex] != ourslot:
            ourindex += 1

        # should never happen
        assert ourindex < len(slots)

        # move up or down the list until we find a suitable parent
        ourindex += direction

        while ( ourindex >= 0 and ourindex < len(slots) and
                not w.isAllowedParent(slots[ourindex][0]) ):
            ourindex += direction

        # we failed to find a new parent
        if ourindex < 0 or ourindex >= len(slots):
            c.insert(oldindex, w)
            return False
        else:
            newparent, newindex = slots[ourindex]
            existingname = w.name in newparent.childnames
            newparent.children.insert(newindex, w)
            w.parent = newparent

            # require a new name because of a clash
            if existingname:
                w.name = w.chooseName()

            return True

    def updateControlItem(self, controlitem, pos):
        """Update the widget's control point.
        
        controlitem is the control item in question."""

        pass

    def autoColor(self, painter, dataindex=0):
        """Return automatic color for plotting."""
        return 'foreground'

    def setupAutoColor(self, painter):
        """Initialise colors for widget automatically."""
        self.autoColor(painter)

# allow the factory to instantiate a generic widget
document.thefactory.register( Widget )
