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

"""A database for default values of settings."""

import sys
import os

class _SettingDB:
    """A singleton class to handle the settings file.

    Reads the settings file on activation, and updates the settings
    file on destruction.
    """

    def __init__(self):
        """Read the default settings.

        This reads the settings from a global configuration file,
        and then from a user configuration file.

        FIXME: Unix specific, fix for other OS."""

        self.systemdefaultfile = '/etc/veusz.conf'
        try:
            self.userdefaultfile = '/home/%s/.veusz.def' % os.environ['USER']
        except KeyError:
            self.userdefaultfile = ''

        self.database = {}

        # FIXME: Unix specific
        self.importFile(self.systemdefaultfile)
        self.sysdefaults = self.database.copy()
        self.importFile(self.userdefaultfile)

    def __del__(self):
        """Update the defaults file.
        
        If the setting was set in the global defaults file, don't
        replicate it in the user's configuration file
        (this allows the default to be easily changed)
        """
        self.writeDefaults(self.userdefaultfile)

    def importFile(self, filename):
        """Read in a configuration file made up of a=b strings."""

        try:
            f = open(filename, 'r')
        except IOError:
            # no error if the file does not exist
            return

        for l in f:
            l = l.strip()

            # ignore comment and blank lines
            if len(l) == 0 or l[0] == '#':
                continue

            # lines should be A=B
            pos = l.find('=')

            # in case of error
            if pos == -1:
                sys.stderr.write('Error in configuration file "%s", line is:\n'
                                 '>>>%s<<<\n' % (filename, l))
                continue

            key = l[:pos].strip()
            val = l[pos+1:].strip()
            self.database[key] = eval(val)

    def writeDefaults(self, filename):
        """Write the list of defaults to the file given."""

        # try to open the output file
        try:
            f = open(filename, 'w')
        except:
            sys.stderr.write('Cannot write to user settings file "%s"\n' %
                             self.userdefaultfile)
            return

        # header
        f.write('# Veusz default settings file\n'
                "# Items are in the form key=val\n\n")

        # write the items in alphabetical order
        keys = self.database.keys()
        keys.sort()
        for key in keys:
            # only update keys which aren't in system defaults or have
            # been modified
            if ( key not in self.sysdefaults or
                 self.sysdefaults[key] != self.database[key] ):
                f.write( '%s=%s\n' % (key, repr(self.database[key])) )

# create the SettingDB singleton
settingdb = _SettingDB()
