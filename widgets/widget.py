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

import string

import widgetfactory
import utils

class Widget:
    """ Fundamental plotting widget interface."""

    typename = 'generic'

    def __init__(self, parent, name=None):
        """Initialise a blank widget."""

        # save parent widget for later
        self.parent = parent
        if parent != None:
            parent._addChild(self, name=name)
            self.document = parent.getDocument()
        else:
            self.document = None

        # store child widgets
        self.child_order   = []
        self.child_widgets = {}

        # automatic child name index
        self.child_index = 1

        # position of this widget on its parent
        self.position = (0., 0., 1., 1.)

        # fractional margins within this widget
        self.margins = (0., 0., 0., 0.)

        # create a preference list for the widget
        self.prefs = utils.Preferences( 'PlotWidget_' + self.typename, self )

        # preferences for part of the object
        self.subprefs = {}

    def getTypeName(self):
        """Return the type name."""
        return self.typename

    def getUserDescription(self):
        """Return a user-friendly description of what this is (e.g. function)."""
        return ''

    def getParent(self):
        """Get parent widget."""
        return self.parent

    def setDocument(self, doc):
        """Set the document to doc."""
        self.document = doc

    def getDocument(self):
        """Return the document."""
        return self.document

    def readPrefs(self):
        """Read preferences for this widget."""
        self.prefs.read()

    def addPref(self, name, type, defaultval):
        """Add a preference to the object."""
        self.prefs.addPref( name, type, defaultval )

    def addSubPref(self, name, pref):
        """Add a sub-preference to the list."""
        self.subprefs[name] = pref

    def hasPref(self, name):
        """Whether there is a preference with name."""
        return name in self.prefs.prefnames
        
    def getPrefLookup(self, name):
        """Get the value of a preference in the form foo.bar.baz"""
        parts = string.split(name, '.')
        noparts = len(parts)

        # this could be recursive, but why bother

        # loop while we iterate through the family
        i = 0
        obj = self
        while i < noparts and obj.hasChild( parts[i] ):
            obj = obj.getChild( parts[i] )
            i += 1

        # if we can't lookup the value, 
        errorpref = ''

        # should be a preference of this object
        if i == noparts-1:
            if obj.hasPref( parts[-1] ):
                return obj.prefs.getPref( parts[-1] )
            else:
                errorpref = parts[-1]
            
        elif i == noparts-2:
            if parts[-2] in obj.subprefs:
                subpref = obj.subprefs[parts[-2]]
                if subpref.hasPref( parts[-1] ):
                    return subpref.getPref( parts[-1] )
                else:
                    errorpref = parts[-1]
            else:
                errorpref = parts[-2]
        else:
            errorpref = parts[i]
            
        raise KeyError, "No such object/preference as '%s'" % errorpref

    def hasChild(self, name):
        """Return whether there is a child with a name."""
        return name in self.child_widgets

    def getChild(self, name):
        """Return a child with a name."""
        return self.child_widgets[name]

    def getChildren(self):
        """Get a list of the children."""
        return self.child_widgets.values()

    def getChildName(self, child):
        """Get the name of a child object."""
        try:
            i = self.child_widgets.values().index( child )
            return self.child_widgets.keys()[i]
        except ValueError:
            return None

    def getChildNames(self):
        """Return the child names."""
        return list( self.child_order )

    def _addChild(self, child, name=None):
        """Add a child widget to draw. """

        # autonumber children
        if name == None:
            name = str(self.child_index)
            self.child_index += 1

        # add child
        self.child_order.append(name)
        self.child_widgets[name] = child

    def removeChild(self, name=None):
        """Remove a child. If name is not specified it is the last child added."""
        # FIXME: handle errors properly here
        if name == None:
            name = self.child_order[-1]
            i = -1
        else:
            i = self.child_order.index(name)

        self.child_order.pop(i)
        del self.child_widgets[name]

    def getName(self):
        """Get name of widget (from parent)."""

        if self.parent == None:
            return '/'
        else:
            return self.parent.getChildName(self)

    def getPath(self):
        """Returns a path for the object, e.g. /plot1/x."""

        obj = self
        build = ''
        while obj.parent != None:
            name = obj.parent.getChildName( obj )
            build = '/' + name + build
            obj = obj.parent

        if len(build) == 0:
            return '/'
        else:
            return build

    def autoAxis(self, axisname):
        """If the axis axisname is used by this widget, return the bounds on that axis.

        Return None if don't know or doesn't use the axis"""

        return None

    def draw(self, parentposn, painter):
        """Draw the widget and its children in posn (a tuple with x1,y1,x2,y2).
        """

        # get parent's position
        x1, y1, x2, y2 = parentposn
        dx = x2 - x1
        dy = y2 - y1

        # get our position
        x1, y1, x2, y2 = x1+dx*self.position[0], y1+dy*self.position[1], \
                         x1+dx*self.position[2], y1+dy*self.position[3]
        dx = x2 - x1
        dy = y2 - y1

        # subtract margins
        x1 += self.margins[0]*dx
        y1 += self.margins[1]*dy
        x2 -= self.margins[2]*dx
        y2 -= self.margins[3]*dy
        posn = ( int(x1), int(y1), int(x2), int(y2) )

        # iterate over children
        for name in self.child_order:
            c = self.child_widgets[name]
            c.draw( posn, painter )

        # return our position
        return posn
        
# allow the factory to instantiate a generic widget
widgetfactory.thefactory.register( Widget )
