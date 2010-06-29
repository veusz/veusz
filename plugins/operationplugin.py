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

"""Plugins for general operations."""

import field

# add an instance of your class to this list to be registered
operationpluginregistry = []

class PluginException(RuntimeError):
    """Raise this to report an error doing what was requested."""
    pass

class OperationPlugin(object):
    # the plugin will get inserted into the menu in a hierarchy based on
    # the elements of this tuple
    name = ('Base plugin',)

    author = ''
    description_short = ''
    description_full = ''

    def __init__(self):
        """Override this to declare a list of input fields if required."""
        self.fields = []

    def apply(self, commandinterface, field_results):
        """Override this option to do the work of the plugin.

        commandinterface is an instance of the embedding interface
        field_results is a dict containing the values

        raise a PluginException to report a problem
        """
