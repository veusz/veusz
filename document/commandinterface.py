# commandinterface.py
# this module supplies the command line interface for plotting
 
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

"""
Module supplies the command interface used in the program, and for
external programs.
"""

import sys
import os.path
import qt

import widgets
import utils
import doc
import simpleread
import commandinterpreter

class CommandInterface(qt.QObject):
    """Class provides command interface."""

    def __init__(self, document):
        """Initialise the interface."""
        qt.QObject.__init__(self)

        self.document = document
        self.currentwidget = self.document.getBaseWidget()
        self.verbose = False

        self.connect( self.document, qt.PYSIGNAL("sigWiped"),
                      self.slotWipedDoc )

    def slotWipedDoc(self):
        """When the document is wiped, we change to the root widget."""
        self.To('/')

    def SetVerbose(self, v=True):
        """Specify whether we want verbose output after operations."""
        self.verbose = v

    def Add(self, type, *args, **args_opt):
        """Add a graph to the plotset."""
        w = widgets.thefactory.makeWidget(type, self.currentwidget,
                                          *args, **args_opt)

        if self.verbose:
            print "Added a graph of type '%s' (%s)" % (type,
                                                       w.getUserDescription())

        self.document.setModified()

    def Remove(self, name):
        """Remove a graph from the dataset."""
        w = self._resolve(name)
        w.getParent().removeChild( w.name )
        self.document.setModified()

    def _resolve(self, where):
        """Resolve graph relative to current graph.

        Allows unix-style specifiers, e.g. /graph1/x
        Returns widget
        """

        parts = where.split('/')

        if where[:1] == '/':
            # relative to base directory
            obj = self.document.getBaseWidget()
        else:
            # relative to here
            obj = self.currentwidget

        # iterate over parts in string
        for p in parts:
            if p == '..':
                # relative to parent object
                p = obj.getParent()
                if p == None:
                    raise ValueError, "Base graph has no parent"
                obj = p
            elif p == '.' or len(p) == 0:
                # relative to here
                pass
            else:
                # child specified
                obj = obj.getChild( p )
                if obj == None:
                    raise ValueError, "Child '%s' does not exist" % p

        # return widget
        return obj

    def To(self, where):
        """Change to a graph within the current graph."""

        self.currentwidget = self._resolve(where)

        if self.verbose:
            print "Changed to graph '%s'" % self.currentwidget.getPath()

    def List(self, graph=None):
        """List the contents of a graph."""

        # get widget to list
        widget = self.currentwidget
        if graph != None:
            widget = self._resolve(graph)

        children = widget.getChildNames()

        if len(children) == 0:
            print '%30s' % 'No children found'
        else:
            # output format name, type
            for name in children:
                w = widget.getChild(name)
                type = w.getTypeName()
                descr = w.getUserDescription()
                print '%10s %10s %30s' % (name, type, descr)

    def Get(self, var):
        """Get the value of a preference."""
        return self.currentwidget.prefLookup(var).get()

    def Save(self, filename):
        """Save the state to a file."""

        f = open(filename, 'w')
        self.document.saveToFile(f)

    def Set(self, var, val):
        """Set the value of a preference."""
        pref = self.currentwidget.prefLookup(var)
        pref.set(val)

        if self.verbose:
            print ( "Set preference '%s' to %s" %
                    (var, repr(val)) )

        self.document.setModified()

    def SetData(self, name, val, symerr=None, negerr=None, poserr=None):
        """Set data with values (and optionally errors)."""

        data = doc.Dataset(val, symerr, negerr, poserr)
        self.document.setData(name, data)

        if self.verbose:
            print "Set variable '%s':" % name
            print "Values = %s" % str( data.vals )
            print "Symmetric errors = %s" % str( data.serr )
            print "Negative errors = %s" % str( data.nerr )
            print "Positive errors = %s" % str( data.perr )

    def ImportString(self, descriptor, string):
        """Read data from the string using a descriptor."""

        stream = simpleread.StringStream(string)
        sr = simpleread.SimpleRead(descriptor)
        sr.readData(stream)
        sr.setInDocument(self.document)

    def ImportFile(self, filename, descriptor):
        """Read data from file with filename using descriptor."""

        file = open(filename, 'r')
        stream = simpleread.FileStream(file)
        sr = simpleread.SimpleRead(descriptor)
        sr.readData(stream)
        sr.setInDocument(self.document)

    def Print(self):
        """Print document."""
        p = qt.QPrinter()

        if p.setup():
            p.newPage()
            self.document.printTo( p,
                                   range(self.document.getNumberPages()) )
            
    def Export(self, filename, color=True):
        """Export plot to filename."""
        
        self.document.export(filename, color=color)
            
