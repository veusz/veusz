#    Copyright (C) 2005 Jeremy S. Sanders
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
###############################################################################

# $Id$

"""Module for holding collections of settings."""

class Settings:
    """A class for holding collections of settings."""

    def __init__(self, name):
        """A new Settings with a name."""

        self.name = name
        self.setdict  = {}  # keep a dict for quick access
        self.setnames = []  # a list of names

    def getName(self):
        """Return name of this settings."""

        return self.name

    def getSettingsList(self):
        """Get a list of settings."""

        return [self.setdict[n] for n in self.setnames]

    def add(self, setting):
        """Add a new setting with the name, or a set of subsettings."""

        n = setting.getName()
        # we shouldn't add things twice
        assert n not in self.setdict
        self.setdict[n] = setting
        self.setnames.append(n)
        
    def __setattr__(self, name, val):
        """Allow us to do

        foo.setname = 42
        """
        
        if name in self.setdict:
            self.setdict[name].set(val)
        else:
            self.__dict__[name] = val

    def __getattr__(self, name):
        """Allow us to do

        print foo.setname
        """

        if name in self.setdict:
            return self.setdict[name].get()
        else:
            return self.__dict__[name]

    def get(self, name):
        """Get the setting variable."""

        return self.setdict[name]
