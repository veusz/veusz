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

"""Module for holding set values.

e.g.

s = Int('foo', 5)
s.get()
s.set(42)
s.fromText('42')
"""

import utils
import controls
import widgets
from settingdb import settingdb

# if invalid type passed to set
class InvalidType(Exception):
    pass

class Setting:
    def __init__(self, name, val, descr=''):
        """Initialise the values.

        descr is a description of the setting
        """
        self.readonly = False
        self.parent = None
        self.name = name
        self.descr = descr
        self.default = val
        self.onmodified = []
        self.set(val)

    def readDefaults(self, root, widgetname):
        """Check whether the user has a default for this setting."""

        deftext = None
        unnamedpath = '%s/%s' % (root, self.name)
        if unnamedpath in settingdb.database:
            deftext = settingdb.database[unnamedpath]

        # named defaults supersedes normal defaults
        namedpath = '%s_NAME:%s' % (widgetname, unnamedpath)
        if namedpath in settingdb.database:
            deftext = settingdb.database[namedpath]
    
        if deftext != None:
            self.set( self.fromText(deftext) )
            self.default = self.val

    def removeDefault(self):
        """Remove the default setting for this setting."""

        # build up setting path
        path = ''
        item = self
        while not isinstance(item, widgets.Widget):
            path = '/%s%s' % (item.name, path)
            item = item.parent

        # specific setting to this widgetname
        namedpath = '%s_NAME:%s' % (item.name, path)

        # remove the settings (ignore if they are not set)
        if path in settingdb.database:
            del settingdb.database[path]

        if namedpath in settingdb.database:
            del settingdb.database[namedpath]
            print settingdb.database

    def setAsDefault(self, withwidgetname = False):
        """Set the current value of this setting as the default value

        If withwidthname is True, then it is only the default for widgets
        of the particular name this setting is contained within."""

        # build up setting path
        path = ''
        item = self
        while not isinstance(item, widgets.Widget):
            path = '/%s%s' % (item.name, path)
            item = item.parent

        # if the setting is only for widgets with a certain name
        if withwidgetname:
            path = '%s_NAME:%s' % (item.name, path)

        # set the default
        settingdb.database[path] = self.toText()

    def saveText(self, saveall, rootname = ''):
        """Return text to restore the value of this setting."""

        if (saveall or not self.isDefault()) and not self.readonly:
            return "Set('%s%s', %s)\n" % ( rootname, self.name,
                                           repr(self.get()) )
        else:
            return ''

    def resetToDefault(self):
        """Reset the value to its default."""
        self.set(self.default)

    def setOnModified(self, fn):
        """Set the function to be called on modification (passing True)."""
        self.onmodified.append(fn)

    def removeOnModified(self, fn):
        """Remove the function from the list of function to be called."""
        i = self.onmodified.index(fn)
        del self.onmodified[i]

    def newDefault(self, val):
        """Update the default and the value."""
        self.default = val
        self.set(val)

    def isDefault(self):
        """Is the current value a default?"""
        return self.get() == self.default

    def getName(self):
        """Get the name of the setting."""
        return self.name

    def get(self):
        """Get the stored setting."""
        return self.convertFrom(self.val)

    def set(self, val):
        """Save the stored setting."""
        self.val = self.convertTo( val )
        for i in self.onmodified:
            i(True)

    def setSilent(self, val):
        """Set the setting, without propagating modified flags.

        This shouldn't often be used as it defeats the automatic updation.
        Used for temporary modifications."""

        self.val = self.convertTo( val )

    def convertTo(self, val):
        """Convert for storage."""
        return val

    def convertFrom(self, val):
        """Convert to storage."""
        return val

    def toText(self):
        """Convert the type to text for saving."""
        return ""

    def fromText(self, text):
        """Convert text to type suitable for setting.

        Raises InvalidType if cannot convert."""
        return None

    def makeControl(self, *args):
        """Make a qt control for editing the setting.

        The control emits settingValueChanged() when the setting has
        changed value."""

        return None

# Store strings
class Str(Setting):
    """String setting."""

    def convertTo(self, val):
        if type(val) == str:
            return val
        raise InvalidType

    def toText(self):
        return self.val

    def fromText(self, text):
        return text

    def makeControl(self, *args):
        return controls.SettingEdit(self, *args)

# Store bools
class Bool(Setting):
    """Bool setting."""

    def convertTo(self, val):
        if type(val) in (bool, int):
            return bool(val)
        raise InvalidType

    def toText(self):
        if self.val:
            return 'True'
        else:
            return 'False'

    def fromText(self, text):
        t = text.strip().lower()
        if t in ('true', '1', 't', 'y', 'yes'):
            return True
        elif t in ('false', '0', 'f', 'n', 'no'):
            return False
        else:
            raise InvalidType

    def makeControl(self, *args):
        return controls.BoolSettingEdit(self, *args)

# Storing integers
class Int(Setting):
    """Integer settings."""

    def convertTo(self, val):
        if type(val) == int:
            return val
        raise InvalidType

    def toText(self):
        return str(self.val)

    def fromText(self, text):
        try:
            return int(text)
        except ValueError:
            raise InvalidType

    def makeControl(self, *args):
        return controls.SettingEdit(self, *args)

# for storing floats
class Float(Setting):
    """Float settings."""

    def convertTo(self, val):
        if type(val) in (float, int):
            return float(val)
        raise InvalidType

    def toText(self):
        return str(self.val)

    def fromText(self, text):
        try:
            return float(text)
        except ValueError:
            raise InvalidType

    def makeControl(self, *args):
        return controls.SettingEdit(self, *args)

class FloatOrAuto(Setting):
    """Save a float or text auto."""

    def convertTo(self, val):
        if type(val) in (int, float):
            return float(val)
        elif type(val) == str and val.strip().lower() == 'auto':
            return None
        else:
            raise InvalidType

    def convertFrom(self, val):
        if val == None:
            return 'Auto'
        else:
            return val

    def toText(self):
        if self.val == None:
            return 'Auto'
        else:
            return str(self.val)

    def fromText(self, text):
        if text.strip().lower() == 'auto':
            return 'Auto'
        else:
            try:
                return float(text)
            except ValueError:
                raise InvalidType

    def makeControl(self, *args):
        return controls.SettingChoice(self, True, ['Auto'], *args)
            
class IntOrAuto(Setting):
    """Save an int or text auto."""

    def convertTo(self, val):
        if type(val) == int:
            return val
        elif type(val) == str and val.strip().lower() == 'auto':
            return None
        else:
            raise InvalidType

    def convertFrom(self, val):
        if val == None:
            return 'Auto'
        else:
            return val

    def toText(self):
        if self.val == None:
            return 'Auto'
        else:
            return str(self.val)

    def fromText(self, text):
        if text.strip().lower() == 'auto':
            return 'Auto'
        else:
            try:
                return int(text)
            except ValueError:
                raise InvalidType
            
    def makeControl(self, *args):
        return controls.SettingChoice(self, True, ['Auto'], *args)

class Distance(Setting):
    """A veusz distance measure, e.g. 1pt or 3%."""
    
    def convertTo(self, val):
        if utils.isDist(val):
            return val
        else:
            raise InvalidType

    def toText(self):
        return self.val

    def fromText(self, text):
        if utils.isDist(text):
            return text
        else:
            raise InvalidType
        
    def makeControl(self, *args):
        return controls.SettingEdit(self, *args)

class Choice(Setting):
    """One out of a list of strings."""

    # maybe should be implemented as a dict to speed up checks

    def __init__(self, name, vallist, val, images = {}, descr = ''):
        """Setting val must be in vallist."""
        
        assert type(vallist) in (list, tuple)
        self.vallist = vallist
        self.images = images
        Setting.__init__(self, name, val, descr = descr)

    def convertTo(self, val):
        if val in self.vallist:
            return val
        else:
            raise InvalidType

    def toText(self):
        return self.val

    def fromText(self, text):
        if text in self.vallist:
            return text
        else:
            raise InvalidType
        
    def makeControl(self, *args):
        return controls.SettingChoice(self, False, self.vallist,
                                      *args)

class ChoiceOrMore(Setting):
    """One out of a list of strings, or anything else."""

    # maybe should be implemented as a dict to speed up checks

    def __init__(self, name, vallist, val, images = {}, descr = ''):
        """Setting has val must be in vallist."""
        
        self.vallist = vallist
        self.images = images
        Setting.__init__(self, name, val, descr = descr)

    def convertTo(self, val):
        return val

    def toText(self):
        return self.val

    def fromText(self, text):
        return text

    def makeControl(self, *args):
        return controls.SettingChoice(self, True, self.vallist,
                                      *args)

class FloatDict(Setting):
    """A dictionary, taking floats as values."""

    def __init__(self, name, val, descr = ''):
        Setting.__init__(self, name, val, descr=descr)

    def convertTo(self, val):
        if type(val) != dict:
            raise InvalidType

        for i in val.itervalues():
            if type(i) != float:
                raise InvalidType

        return val

    def toText(self):
        text = ''
        for key, val in self.val.iteritems():
            text += '%s = %g\n' % (key, val)
        return text

    def fromText(self, text):
        """Do conversion from list of a=X\n values."""

        out = {}
        # break up into lines
        for l in text.split('\n'):
            l = l.strip()
            if len(l) == 0:
                continue

            # break up using =
            p = l.strip().split('=')

            if len(p) != 2:
                raise InvalidType

            try:
                v = float(p[1])
            except ValueError:
                raise InvalidType

            out[ p[0].strip() ] = v
        return out

    def makeControl(self, *args):
        return controls.SettingMultiLine(self, *args)
