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

import veusz.qtall as qt4

# default values to some settings in case the user does not have these
defaultValues = {
    # export options
    'export_DPI': 100,
    'export_color': True,
    'export_antialias': True,
    'export_quality': 85,

    # plot options
    'plot_updateinterval': 1000,
    'plot_antialias': True,

    # recent files list
    'main_recentfiles': []
    }

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
        self.sepchars = "%%%"

        # read settings using QSettings
        self.readSettings()

    def readSettings(self):
        """Read the settings using QSettings.

        Entries have / replaced with set of characters self.sepchars
        This is because it greatly simplifies the logic as QSettings
        has special meaning for /

        The only issues are that the key may be larger than 255 characters
        We should probably check for this
        """

        s = qt4.QSettings(self.domain, self.product)

        for key in s.childKeys():
            val = s.value(key).toString()
            realkey = unicode(key).replace(self.sepchars, '/')

            try:
                self.database[realkey] = eval( unicode(val) )
            except:
                print >>sys.stderr, ('Error interpreting item "%s" in '
                                     'settings file' % realkey)

        # set any defaults which haven't been set
        for key, value in defaultValues.iteritems():
            if key not in self.database:
                self.database[key] = value
        
    def writeSettings(self):
        """Write the settings using QSettings.

        This is called by the atexit handler below
        """

        s = qt4.QSettings(self.domain, self.product)

        # write each entry, keeping track of which ones haven't been written
        cleankeys = []
        for key, value in self.database.iteritems():
            cleankey = key.replace('/', self.sepchars)
            cleankeys.append(cleankey)
            s.setValue(cleankey, qt4.QVariant(repr(value)))

        # now remove all the values which have been removed
        remove = []
        for key in list(s.childKeys()):
            if unicode(key) not in cleankeys:
                s.remove(key)

    def __getitem__(self, key):
        """Get the item from the database."""
        return self.database[key]

    def __setitem__(self, key, value):
        """Set the value in the database."""
        self.database[key] = value

    def __delitem__(self, key):
        """Remove the key from the database."""
        del self.database[key]

    def __contains__(self, key):
        """Is the key in the database."""
        return key in self.database

# create the SettingDB singleton
settingdb = _SettingDB()

# write out settings at exit
atexit.register(settingdb.writeSettings)
