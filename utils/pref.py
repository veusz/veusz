# pref.py - hold the preferences
 
#    Copyright (C) 2003 Jeremy S. Sanders
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

import qt

domain='jeremysanders.net'
appname='Veusz'

NoneMarker = '<*NONE*>'

def _readFontEntry(settings, name, default):
    """ Function to read a font setting, and return a QFont """
    fontstr, ok = settings.readEntry(name, default)
    font = qt.QFont()
    font.fromString(fontstr) # needs some error handling
    return (font, ok)

def _writeFontEntry(settings, name, val, default):
    """ Write a QFont to the settings file (if it isn't the default). Default is text. """
    fontstr = val.toString()
    if fontstr == default:
        settings.removeEntry(name)
    else:
        settings.writeEntry(name, fontstr)

def _readColorEntry(settings, name, default):
    """ Function to read a colour setting, returning a QColor. Default is text. """
    colstr, ok = settings.readEntry(name, default)
    col = qt.QColor( colstr )
    return (col, ok)

def _writeColorEntry(settings, name, val, default):
    """ Write a colour settings entry. val is a QColor. Default is text. """
    text = val.name()
    if text == default:
        settings.removeEntry(name)
    else:
        settings.writeEntry(name, text)

def _writeDefaultEntry(settings, name, val, default):
    """ Function to write an entry (if it isn't the default) """
    if val == default:
        settings.removeEntry(name)
    else:
        settings.writeEntry(name, val)

def _readListEntry(settings, name, default):
    """ Read a simple python list. Default is a list."""
    # FIXME: is this insecure?
    txt, ok = settings.readEntry(name, repr(default))
    return (eval(str(txt)), ok)

def _readStringEntry(settings, name, default):
    """Read a string entry (required as Qt returns a QString."""
    txt, ok = settings.readEntry(name, default)
    return (str(txt), ok)

def _writeListEntry(settings, name, val, default):
    """ Write a simple python list to the preferences file."""
    if val == default:
        settings.removeEntry(name)
    else:
        settings.writeEntry(name, repr(val))
    
# contains the functions to call to read preferences of a certain type
_read_functions = { 'int': qt.QSettings.readNumEntry,
                    'bool': qt.QSettings.readNumEntry,
                    'double': qt.QSettings.readDoubleEntry,
                    'string': _readStringEntry,
                    'font': _readFontEntry,
                    'color': _readColorEntry,
                    'list': _readListEntry }

# functions to call to write preferences of certain types
_write_functions = { 'int': _writeDefaultEntry,
                     'bool': _writeDefaultEntry,
                     'double': _writeDefaultEntry,
                     'string': _writeDefaultEntry,
                     'font': _writeFontEntry,
                     'color': _writeColorEntry,
                     'list': _writeListEntry }

class _PrefBool(qt.QCheckBox):
    """A check box which connects directly with a bool preference."""

    def __init__(self, prefs, name, *args):
        qt.QCheckBox.__init__(self, *args)
        self.prefs = prefs
        self.name = name

        self.setChecked(prefs.getPref(name))
        self.connect(self, qt.SIGNAL("stateChanged(int)"),
                     self.slotStateChanged)

    def slotStateChanged(self, state):
        """Called when the tick box is modified."""
        on = (state == qt.QButton.On)
        self.prefs.setPref(self.name, on)

        # emit this signal so that changes can be notified
        self.emit(qt.PYSIGNAL("sigModified"), ())

class _PrefInt(qt.QSpinBox):
    """A spin box which modifies a preference directly."""

    def __init__(self, prefs, name, *args):
        qt.QSpinBox.__init__(self, *args)
        self.prefs = prefs
        self.name = name

        self.setValue(prefs.getPref(name))
        self.connect(self, qt.SIGNAL("valueChanged(int)"),
                     self.slotValueChanged)

    def slotValueChanged(self, value):
        """Called when the spin box is modified."""
        self.prefs.setPref(self.name, value)

        # emit this signal so that changes can be notified
        self.emit(qt.PYSIGNAL("sigModified"), ())

class _PrefString(qt.QLineEdit):
    """A line edit which modifies a string preference directly."""

    def __init__(self, prefs, name, *args):
        qt.QLineEdit.__init__(self, *args)
        self.prefs = prefs
        self.name = name

        self.setText(prefs.getPref(name))
        self.connect(self, qt.SIGNAL("lostFocus()"),
                     self.slotLostFocus)

    def slotLostFocus(self):
        """Called when the edit box is modified."""
        value = self.text()
        self.prefs.setPref(self.name, str(value))

        # emit this signal so that changes can be notified
        self.emit(qt.PYSIGNAL("sigModified"), ())

class _PrefDouble(qt.QLineEdit):
    """A line edit which modifies a double preference directly."""

    def __init__(self, prefs, name, *args):
        qt.QLineEdit.__init__(self, *args)
        self.prefs = prefs
        self.name = name

        self.setText( str(prefs.getPref(name)) )
        self.setValidator( qt.QDoubleValidator(self) )
        
        self.connect(self, qt.SIGNAL("lostFocus()"),
                     self.slotLostFocus)

    def slotLostFocus(self):
        """Called when the edit box is modified."""
        value = self.text()
        self.prefs.setPref(self.name, float(str(value)) )

        # emit this signal so that changes can be notified
        self.emit(qt.PYSIGNAL("sigModified"), ())

# map preference names to Qt classes for editing
_prefQtClasses = { 'bool': _PrefBool, 'int': _PrefInt,
                   'string': _PrefString, 'double': _PrefDouble }

class Preferences:
    """Class for reading in default preferences."""

    def __init__(self, classname, object):
        """Initialise object to serve class 'classname' and the object."""
        self.classname = classname
        self.prefnames = []
        self.preftypes = {}
        self.prefdefaults = {}
        self.object = object

    def addPref(self, name, type, default):
        """Add a preference to the list of controlled variables."""
        self.prefnames.append(name)
        self.preftypes[name] = type
        self.prefdefaults[name] = default

    def setDefault(self, name, default):
        """Change the default value for a preference."""
        self.prefdefaults[name] = default

    def isSetDefault(self, name):
        """Is the preference set to default?"""

        return self.prefdefaults[name] == self.getPref(name)

    def getPrefType(self, name):
        """Get the preference type."""
        return self.preftypes[name]

    def setPref(self, name, val):
        """Set a preferences variable (checking the type).

        Works for int, double, string and list (more later)."""

        t = self.preftypes[name]
        tv = type(val)

        # go through allowed entries, converting if necessary
        if val == None:
            pass
        elif (t == 'int' or t == 'bool') and tv == type(1):
            pass
        elif t == 'double' and (tv == type(1.) or tv == type(1)):
            val = float(val)
        elif t == 'string' and tv == type(''):
            pass
        elif t == 'list' and tv == type([]):
            pass
        else:
            raise TypeError, '%s passed to preference type %s' % (tv, t)
        
        self.object.__dict__[name] = val

    def getPref(self, name):
        """Get a preference value."""
        return self.object.__dict__[name]

    def hasPref(self, name):
        """Is there a preference named name?"""
        return name in self.prefnames

    def getPrefNames(self):
        """Get a list of the preferences."""
        return self.prefnames

    def makePrefControl(self, name, *args):
        """Make a QWidget to modify the preference with the name."""
        cntrlclass = _prefQtClasses[ self.preftypes[name] ]
        return cntrlclass(self, name, *args)

    def read(self):
        """Read in the listed preferences."""
        settings = qt.QSettings()
        settings.setPath(domain, appname, qt.QSettings.User)

        for pref in self.prefnames:
            name = '/' + appname + '/' + self.classname + '/' + pref
            default = self.prefdefaults[pref]

            # special handling of none entries
            s, ok = settings.readEntry( name )
            
            if s == NoneMarker or (not ok and default == None):
                val = (None, 1)
            else:
                val =  _read_functions[self.preftypes[pref]] \
                      (settings, name, default)

            # actaully write the preference into the object
            self.object.__dict__[pref] = val[0]

    def write(self):
        """Write out the listed preferences."""
        settings = qt.QSettings()
        settings.setPath(domain, appname, qt.QSettings.User)

        # iterate over preferences, look to see whether exists,
        # write if does (and isn't default)
        for pref in self.prefnames:
            try:
                val = self.object.__dict__[pref]
            except AttributeError:
                # skips to the next entry
                continue

            name = '/' + appname + '/' + self.classname + '/' + pref

            # call appropriate function to write the preference
            # (removing preferences that already exist)
            _write_functions[self.preftypes[pref]](settings, name, val,
                                                   self.prefdefaults[pref])

    def getSaveText(self, path):
        """Return the text to restore preferences using commandinterpreter."""

        text = ''
        for pref in self.prefnames:
            val = self.getPref(pref)
            # if preference has changed, save it
            if val != self.prefdefaults[pref]:
                text += "Set('%s%s', %s)\n" % (path, pref, repr(val))
        return text
    
