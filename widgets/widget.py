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
            parent.addChild(self)
            self.document = parent.getDocument()
            if name == None:
                name = str( parent.createChildIndex() )
        else:
            name = '/'
            self.document = None

        self.name = name

        # store child widgets
        self.children = []
        
        # automatic child name index
        self.child_index = 1

        # position of this widget on its parent
        self.position = (0., 0., 1., 1.)

        # fractional margins within this widget
        self.margins = ('0', '0', '0', '0')

        # create a preference list for the widget
        self.prefs = utils.Preferences( 'PlotWidget_' + self.typename, self )

        # preferences for part of the object
        self.subprefs = {}

    def addChild(self, child):
        """Add child to list."""
        self.children.append(child)

    def createChildIndex(self):
        """Return a new index to create a child."""
        i = self.child_index
        self.child_index += 1
        return i

    def getName(self):
        """Get the name of the widget."""
        return self.name

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

    def getPrefs(self):
        """Return the preferences for the object."""
        return self.prefs

    def addSubPref(self, name, pref):
        """Add a sub-preference to the list."""
        self.subprefs[name] = pref

    def getSubPrefs(self):
        """Return the sub-preferences."""
        return self.subprefs

    def hasPref(self, name):
        """Whether there is a preference with name."""
        return name in self.prefs.prefnames
        
    def getPrefLookup(self, name):
        """Get the value of a preference in the form foo/bar/baz"""
        parts = string.split(name, '/')
        noparts = len(parts)

        # this could be recursive, but why bother

        # loop while we iterate through the family
        i = 0
        obj = self
        while i < noparts and obj.hasChild( parts[i] ):
            obj = obj.getChild( parts[i] )
            i += 1

        # if we can't lookup the value
        errorpref = ''

        if i == noparts-1:
            # should be a preference of this object
            if obj.hasPref( parts[-1] ):
                return obj.prefs.getPref( parts[-1] )
            else:
                errorpref = parts[-1]
                
        elif i == noparts-2:
            # should be a subpreference of this object
            if parts[-2] in obj.subprefs:
                subpref = obj.subprefs[parts[-2]]
                if subpref.hasPref( parts[-1] ):
                    return subpref.getPref( parts[-1] )
                else:
                    errorpref = parts[-1]
            else:
                errorpref = parts[-2]

        elif i == noparts:
            # preference not actually specified
            raise utils.PreferenceError, \
                  "No preference specified in graph '%s'" % name

        else:
            errorpref = parts[i]
            
        raise utils.PreferenceError, \
              "No such object/preference as '%s'" % errorpref

    def setPrefLookup(self, name, val):
        """Set a preference named in the form foo/bar/baz."""
        parts = string.split(name, '/')
        noparts = len(parts)

        # loop while we iterate through the family
        i = 0
        obj = self
        while i < noparts and obj.hasChild( parts[i] ):
            obj = obj.getChild( parts[i] )
            i += 1

        # if we can't lookup the value
        errorpref = ''

        if i == noparts-1:
            # should be a preference in this object
            if obj.hasPref( parts[-1] ):
                obj.prefs.setPref( parts[-1], val )
                return
            else:
                errorpref = parts[-1]

        elif i == noparts-2:
            # should be a subpreference of this object
            if parts[-2] in obj.subprefs:
                subpref = obj.subprefs[parts[-2]]
                if subpref.hasPref( parts[-1] ):
                    subpref.setPref( parts[-1], val )
                    return
                else:
                    errorpref = parts[-1]
            else:
                errorpref = parts[-2]

        elif i == noparts:
            # preference not actually specified
            raise utils.PreferenceError, \
                  "No preference specified in graph '%s'" % name
                
        else:
            errorpref = parts[i]
                            
        raise utils.PreferenceError, \
              "No such object/preference as '%s'" % errorpref

    def getChild(self, name):
        """Return a child with a name."""

        for i in self.children:
            if i.name == name:
                return i
        return None

    def hasChild(self, name):
        """Return whether there is a child with a name."""

        return self.getChild(name) != None

    def getChildren(self):
        """Get a list of the children."""
        return self.children

    def getChildNames(self):
        """Return the child names."""

        names = []
        for i in self.children:
            names.append( i.name )      
        return names

    def removeChild(self, name):
        """Remove a child."""

        i = 0
        nc = len(self.children)
        while i < nc and self.children[i].name != name:
            i += 1

        if i < nc:
            self.children.pop(i)
        else:
            raise utils.GraphError, \
                  "Cannot remove graph '%s' - does not exist" % name

    def getPath(self):
        """Returns a path for the object, e.g. /plot1/x."""

        obj = self
        build = ''
        while obj.parent != None:
            build = '/' + obj.name + build
            obj = obj.parent

        if len(build) == 0:
            build = '/'

        return build

    def autoAxis(self, axisname):
        """If the axis axisname is used by this widget,
        return the bounds on that axis.

        Return None if don't know or doesn't use the axis"""

        return None

    def draw(self, parentposn, painter):
        """Draw the widget and its children in posn (a tuple with x1,y1,x2,y2).
        """

        print self.getPath(), self.position, self.margins

        # get parent's position
        x1, y1, x2, y2 = parentposn
        dx = x2 - x1
        dy = y2 - y1

        # get our position
        x1, y1, x2, y2 = x1+dx*self.position[0], y1+dy*self.position[1], \
                         x1+dx*self.position[2], y1+dy*self.position[3]
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

    def saveToFile(self, file):
        """Save the widget to the file."""

        # set the preferences of the object
        pref = self.prefs
        for p in pref.getPrefNames():
            if not pref.isSetDefault(p):
                file.write("Set('%s', %s)\n" % (p, repr(pref.getPref(p))))

        # now set the subprefs
        for spname, spref in self.subprefs.items():
            pref = spref.getPrefs()
            for p in pref.getPrefNames():
                if not pref.isSetDefault(p):
                    file.write("Set('%s/%s', %s)\n" % (spname, p,
                                                       repr(pref.getPref(p))))

        # now go throught the subwidgets
        for c in self.getChildren():
            file.write("Add('%s', name='%s')\n" %
                       (c.getTypeName(), c.getName()))

            file.write("To('%s')\n" % c.getName())
            c.saveToFile(file)
            file.write("To('..')\n")
        
# allow the factory to instantiate a generic widget
widgetfactory.thefactory.register( Widget )
