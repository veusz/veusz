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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

import widgetfactory
import utils
import setting

class Widget(object):
    """ Fundamental plotting widget interface."""

    typename = 'generic'
    allowusercreation = False

    # list of allowed types this can have as a parent
    allowedparenttypes = []

    def __init__(self, parent, name=None):
        """Initialise a blank widget."""

        # save parent widget for later
        self.parent = parent

        if not self.isAllowedParent(parent):
            raise RuntimeError, "parent is of incorrect type"

        if parent != None:
            if name == None:
                name = parent.createUniqueName(self.typename)
            self.document = parent.document
            parent.addChild(self)

        else:
            name = '/'
            self.document = None

        self.name = name

        # store child widgets
        self.children = []

        # names which are automatically added to widget
        # these aren't saved when the documnet is saved
        self.autoadd = []
        
        # automatic child name index
        self.child_index = 1

        # position of this widget on its parent
        self.position = (0., 0., 1., 1.)

        # fractional margins within this widget
        self.margins = ('0', '0', '0', '0')

        # settings for widget
        self.settings = setting.Settings( 'Widget_' + self.typename )

        # hook up settings to modify document flag if they are modified
        self.settings.setOnModified(self.slotSettingModified)
        
        # actions for widget
        self.actions = []
        self.actionfuncs = {}
        self.actiondescr = {}

    def addAction(self, name, function, descr=''):
        """Assign name to operation.

        name is name to assign
        function is a function with no parameters to call
        descr is description of routine.
        """
        self.actions.append(name)
        self.actionfuncs[name] = function
        self.actiondescr[name] = descr

    def isAllowedParent(self, parent):
        """Is the parent a suitable type?"""
        ap = self.allowedparenttypes 
        if parent == None and len(ap)>0 and ap[0] == None:
            return True
        
        for p in ap:
            if isinstance(parent, p):
                return True
        return False      

    def willAllowParent(type, parent):
        """Is the parent of an allowed type to have this type as a child?"""

        # allow base widget to have no parent
        ap = type.allowedparenttypes 
        if parent == None and len(ap)>0 and ap[0] == None:
            return True
        
        for p in ap:
            if isinstance(parent, p):
                return True
        return False
    willAllowParent = classmethod(willAllowParent)

    def addChild(self, child):
        """Add child to list."""
        self.children.append(child)

    def createUniqueName(self, prefix):
        """Create a name using the prefix which hasn't been used before."""
        names = [c.name for c in self.children]

        i = 1
        while "%s%i" % (prefix, i) in names:
            i += 1
        return "%s%i" % (prefix, i)

    def getUserDescription(self):
        """Return a user-friendly description of what
        this is (e.g. function)."""
        return ''
    userdescription = property(getUserDescription)

    def prefLookup(self, name):
        """Get the value of a preference in the form foo/bar/baz"""

        if len(name) > 0 and name[0] == '/':
            obj = self.document.basewidget
            name = name[1:]
        else:
            obj = self

        parts = name.split('/')
        noparts = len(parts)

        # this could be recursive, but why bother
        # loop while we iterate through the family
        i = 0
        while i < noparts and obj.hasChild( parts[i] ):
            obj = obj.getChild( parts[i] )
            i += 1

        if i == noparts:
            raise ValueError, "Specified an widget, not a setting"
        else:
            return obj.settings.getFromPath( parts[i:] )

    def getChild(self, name):
        """Return a child with a name."""
        for i in self.children:
            if i.name == name:
                return i
        return None

    def hasChild(self, name):
        """Return whether there is a child with a name."""
        return self.getChild(name) != None

    def _getChildNames(self):
        """Return the child names."""
        return [i.name for i in self.children]
    childnames = property(_getChildNames)

    def removeChild(self, name):
        """Remove a child."""

        i = 0
        nc = len(self.children)
        while i < nc and self.children[i].name != name:
            i += 1

        if i < nc:
            self.children.pop(i)
        else:
            raise ValueError, \
                  "Cannot remove graph '%s' - does not exist" % name

    def _getPath(self):
        """Returns a path for the object, e.g. /plot1/x."""

        obj = self
        build = ''
        while obj.parent != None:
            build = '/' + obj.name + build
            obj = obj.parent

        if len(build) == 0:
            build = '/'

        return build
    path = property(_getPath)

    def autoAxis(self, axisname):
        """If the axis axisname is used by this widget,
        return the bounds on that axis.

        Return None if don't know or doesn't use the axis"""

        return None

    def draw(self, parentposn, painter):
        """Draw the widget and its children in posn (a tuple with x1,y1,x2,y2).
        """

        print self.path, self.position, self.margins

        # get parent's position
        x1, y1, x2, y2 = parentposn
        dx = x2 - x1
        dy = y2 - y1

        # get our position
        x1, y1, x2, y2 = ( x1+dx*self.position[0], y1+dy*self.position[1],
                           x1+dx*self.position[2], y1+dy*self.position[3] )
        dx = x2 - x1
        dy = y2 - y1

        # convert margin to physical units and subtract
        deltas = utils.cnvtDists( self.margins, painter )
        bounds = ( int(x1+deltas[0]), int(y1+deltas[1]),
                   int(x2-deltas[2]), int(y2-deltas[3]) )

        # iterate over children in reverse order
        for i in range(len(self.children)-1, -1, -1 ):
            self.children[i].draw(bounds, painter)
 
        # return our final bounds
        return bounds

    def getSaveText(self, saveall = False):
        """Return text to restore object

        If saveall is true, save everything, including defaults."""

        # set everything first
        text = self.settings.saveText(saveall)

        # now go throught the subwidgets
        for c in self.children:
            if c.name not in self.autoadd:
                text += "Add('%s', name='%s')\n" % (c.typename, c.name)

            # if we need to go to the child, go there
            ctext = c.getSaveText(saveall)
            if ctext != '':
                text += ("To('%s')\n"
                         "%s"
                         "To('..')\n") % (c.name, ctext)

        return text

    def slotSettingModified(self, ismodified):
        """Called when settings is modified."""

        if ismodified:
            self.document.setModified(True)

# allow the factory to instantiate a generic widget
widgetfactory.thefactory.register( Widget )
