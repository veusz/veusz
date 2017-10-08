# widget factory
# tools to generate any type of plot widget you want

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
from ..compat import citems

class WidgetFactory(object):
    """Class to help produce any type of widget you want by name."""

    def __init__(self):
        """Initialise the class."""
        self.regwidgets = {}

    def register(self, classobj):
        """Register a class with the factory."""
        self.regwidgets[classobj.typename] = classobj

    def makeWidget(self, widgettype, parent, document, name=None, autoadd=True,
                   index=-1, **optargs):
        """Make a new widget of the appropriate type."""

        # check for / in name of widget
        if name is not None and name.find('/') != -1:
            raise ValueError('name cannot contain "/"')

        w = self.regwidgets[widgettype](parent, name=name)
        w.document = document
        w.linkToStylesheet()

        # set all the passed default settings
        for name, val in citems(optargs):
            # allow subsettings to be set using __ -> syntax
            name = name.replace('__', '/')

            document.resolveSettingPath(w, name).set(val)

        if autoadd:
            w.addDefaultSubWidgets()

        # move around child afterwards, yuck
        if index != -1:
            del parent.children[-1]
            parent.children.insert(index, w)

        return w

    def getWidgetClass(self, name):
        """Get the class for the widget."""
        return self.regwidgets[name]

    def listWidgets(self):
        """Return an array of the widgets the factory can make."""
        return sorted(self.regwidgets)

    def listWidgetClasses(self):
        """Return list of allowed classes."""
        return list(self.regwidgets.values())

# singleton
thefactory = WidgetFactory()
