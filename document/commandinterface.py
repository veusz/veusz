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
import string

import widgets
import utils

class CommandInterface:
    """Class provides command interface."""

    def __init__(self, document):
        """Initialise the interface."""
        self.document = document
        self.currentwidget = self.document.getBaseWidget()
        self.verbose = False

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

    def _resolve(self, where):
        """Resolve graph relative to current graph.

        Allows unix-style specifiers, e.g. /graph1/x
        Returns widget
        """

        parts = string.split(where, '/')

        # relative to base directory
        if where[:1] == '/':
            baseobj = self.document.getBaseWidget()
        else:
            baseobj = self.currentwidget

        # iterate over parts in string
        for p in parts:
            if p == '..':
                # relative to parent object
                if baseobj.parent != None:
                    baseobj = baseobj.parent
            elif p == '.' or len(p) == 0:
                # relative to here
                pass
            else:
                # child specified
                if baseobj.hasChild( p ):
                    baseobj = baseobj.getChild( p )
                else:
                    raise utils.GraphError, "Child '%s' does not exist" % p

        # return widget
        return baseobj

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
        return self.currentwidget.getPrefLookup(var)

    def Set(self, var, val):
        """Set the value of a preference."""
        self.currentwidget.setPrefLookup(var, val)
        if self.verbose:
            print "Set preference '%s' to '%s'" % \
                  ( var, str(self.Get(var)) )

        self.document.setModified()

    def SetData(self, name, val, symerr=None, negerr=None, poserr=None):
        """Set data with values (and optionally errors)."""

        d = self.document
        d.setData(name, val, symerr=symerr, negerr=negerr, poserr=poserr)

        if self.verbose:
            print "Set variable '%s':" % name
            print "Values = %s" % str( d.getData(name) )
            print "Symmetric errors = %s" % str( d.getSymErr(name) )
            print "Negative errors = %s" % str( d.getNegErr(name) )
            print "Positive errors = %s" % str( d.getPosErr(name) )

    def WriteEPS(self, filename):
        """Write contents of graph to an eps file."""

        pass

