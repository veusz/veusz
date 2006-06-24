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
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
###############################################################################

# $Id$

"""A database for default values of settings."""

import sys
import os
import os.path
import atexit

import veusz.qtall as qt

class _SettingDB(object):
    """A class which provides access to a persistant settings database.
    
    Items are accesses as a dict, with items as key=value
    """

    def __init__(self):
        """Initialise the object, reading the settings."""

        # This domain name is fictional!
        self.domain = 'veusz.org'
        self.product = 'veusz'
        self.database = {}
        self.removedsettings = {}
        self.sepchars = "_-%-_"

        # read settings using QSettings
        self.readSettings()

        # Import old settings
        if 'importPerformed' not in self.database:
            self.database['importPerformed'] = True

            oldsettings = _OldSettingDB()
            if oldsettings.filefound:
                self.database.update( oldsettings.database )
                print ("Imported settings from $HOME/.veusz.def "
                       "and /etc/veusz.conf")

    def readSettings(self):
        """Read the settings using QSettings.

        Entries have / replaced with set of characters self.sepchars
        This is because it greatly simplifies the logic as QSettings
        has special meaning for /

        This is probably a kludge.

        The only issues are that the key may be larger than 255 characters
        We should probably check for this
        """

        # QT4FIXME
        return

        s = qt.QSettings(self.domain, self.product)
        #s.setPath(qt.QSettings.IniFormat, self.domain, self.product)
        path = '/%s/%s' % (self.domain, self.product)
        for key in s.entryList(path):
            key = unicode(key)
            val, ok = s.readEntry("%s/%s" % (path, key))
            realkey = key.replace(self.sepchars, '/')
            try:
                self.database[realkey] = eval( unicode(val) )
            except:
                print >>sys.stderr, ('Error interpreting item "%s" in '
                                     'settings file' % realkey)
        
    def writeSettings(self):
        """Write the settings using QSettings.

        This is called by the atexit handler below
        """

        # QT4FIXME
        return

        s = qt.QSettings()
        s.setPath(self.domain, self.product)
        path = '/%s/%s' % (self.domain, self.product)

        # write each entry, keeping track of which ones haven't been written
        for key, value in self.database.iteritems():
            fkey = "%s/%s" % (path, key.replace('/', self.sepchars) )
            if not s.writeEntry(fkey, repr(value)):
                print >>sys.stderr, 'Error writing setting "%s"' % key

        # now remove all the values which have been removed
        for key in self.removedsettings.iterkeys():
            fkey = "%s/%s" % (path, key.replace('/', self.sepchars) )
            if not s.removeEntry(fkey):
                print >>sys.stderr, 'Error removing setting "%s"' % key

    def __getitem__(self, key):
        """Get the item from the database."""
        return self.database[key]

    def __setitem__(self, key, value):
        """Set the value in the database."""
        self.database[key] = value
        if key in self.removedsettings:
            del self.removedsettings[key]

    def __delitem__(self, key):
        """Remove the key from the database."""
        del self.database[key]
        self.removedsettings[key] = True

    def __contains__(self, key):
        """Is the key in the database."""
        return key in self.database

class _OldSettingDB:
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
            self.userdefaultfile = os.path.join(os.environ['HOME'],
                                                '.veusz.def')
        except KeyError:
            self.userdefaultfile = ''

        self.database = {}

        # Unix specific (this is why we moved to QSettings)
        defaults = self.importFile(self.systemdefaultfile)
        self.sysdefaults = self.database.copy()
        user = self.importFile(self.userdefaultfile)

        # Did we manage to find either settings file?
        self.filefound = (defaults or user)

    def importFile(self, filename):
        """Read in a configuration file made up of a=b strings."""

        try:
            f = open(filename, 'r')
        except IOError:
            # no error if the file does not exist
            return False

        for l in f:
            l = l.strip()

            # ignore comment and blank lines
            if len(l) == 0 or l[0] == '#':
                continue

            # lines should be A=B
            pos = l.find('=')

            # in case of error
            if pos == -1:
                sys.stderr.write('Error in configuration file "%s", '
                                 'line is:\n>>>%s<<<\n' % (filename, l))
                continue

            key = l[:pos].strip()
            val = l[pos+1:].strip()
            self.database[key] = eval(val)
        return True

# create the SettingDB singleton
settingdb = _SettingDB()

# write out settings at exit
atexit.register(settingdb.writeSettings)
