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

# if invalid type passed to set
class InvalidType(Exception):
    pass

class Setting:
    def __init__(self, name, val, descr=''):
        """Initialise the values.

        descr is a description of the setting
        """
        self.name = name
        self.descr = descr
        self.set(val)

    def get(self):
        """Get the stored setting."""
        return self.convertFrom(self.val)

    def getName(self):
        """Return the name."""
        return self.name

    def set(self, val):
        """Save the stored setting."""
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
        """Convert text to saved type for loading."""
        pass

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
        self.set( text )

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
            self.val = True
        elif t in ('false', '0', 'f', 'n', 'no'):
            self.val = False
        else:
            raise InvalidType

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
            self.val = int(text)
        except ValueError:
            raise InvalidType

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
            self.set( float(text) )
        except ValueError:
            raise InvalidType

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
            self.val = None
        else:
            try:
                self.val = float(text)
            except ValueError:
                raise InvalidType
            
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
            self.val = None
        else:
            try:
                self.val = int(text)
            except ValueError:
                raise InvalidType
            
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
            self.val = text
        else:
            raise InvalidType
        
class Choice(Setting):
    """One out of a list of strings."""

    # maybe should be implemented as a dict to speed up checks

    def __init__(self, name, vallist, val, images = {}, descr = ''):
        """Setting has name, and val must be in vallist."""
        
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
            self.val = text
        else:
            raise InvalidType
        
class ChoiceOrMore(Setting):
    """One out of a list of strings, or anything else."""

    # maybe should be implemented as a dict to speed up checks

    def __init__(self, name, vallist, val, images = {}, descr = ''):
        """Setting has name, and val must be in vallist."""
        
        self.vallist = vallist
        self.images = images
        Setting.__init__(self, name, val, descr = descr)

    def convertTo(self, val):
        return val

    def toText(self):
        return self.val

    def fromText(self, text):
        self.val = text
