# data.py
# Handle datasets with an interface class

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

"""Handles datasets with an interface class.

This module exports an object called data, which stores
the data read into the program and temporary variables.
"""

class Data:
    """ This class holds the data read into the program."""
    
    def __init__(self):
        """ Initialise class."""
        self.datasets = {}
        self.globals = {}

    def add_item(self, dataset, item):
        """ Add an item to a dataset."""
        
        if dataset not in self.datasets:
            self.datasets[ dataset ] = []

        self.datasets[dataset].append(item)

    def set_var(self, var, val):
        """ Set the value of a single variable."""
        self.datasets[ var ] = val

    def get_var(self, var):
        """ Return the value of a single variable."""
        return self.datasets[ var ]

    def expand_expression(self, text):
        """ Expand the $vars$ in the text with their values here.

            Variables are deliminated by $ signs, and can be expressions
            Variables are expanded from the data dictionary
            """

        retval = ''
        invar = False

        # iterate over each character
        # $ flips mode and adds subsequent part
        part = ''
        for c in text + '$': # we add a $ to ensure that we always end properly
            if c == '$':
                if invar:
                    try:
                        # no sanity checking here...
                        retval += str( eval(part, self.globals,
                                            self.datasets ) )
                    except:
                        print "Could not evaluate expression '" + part + \
                              "'. Ignoring."
                else:
                    retval += part

                invar = not invar
                part = ''
            else:
                part += c

        return retval

    def dump(self):
        """ Print out all the stored data."""
        print data.datasets

# singleton for this class - we probably only want one
data = Data()
