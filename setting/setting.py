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

import re
import math

import qt

import utils
import controls
import widgets
from settingdb import settingdb

# if invalid type passed to set
class InvalidType(Exception):
    pass

class Setting(object):
    def __init__(self, name, value, descr=''):
        """Initialise the values.

        descr is a description of the setting
        """
        self.readonly = False
        self.parent = None
        self.name = name
        self.descr = descr
        self.default = value
        self.onmodified = []
        self._val = None
        self.tied = None

        self.val = value

    def get(self):
        """Get the value, or get the value from the tied variable."""
        if self.tied == None:
            return self.convertFrom(self._val)
        else:
            return self.tied.val

    def set(self, v):
        """Set the value, or set the tied variable."""
        if self.tied == None:
            self._val = self.convertTo(v)
            for mod in self.onmodified:
                mod(True)
        else:
            self.tied.val = v

    val = property(get, set, None,
                   'Get or modify the value of the setting')

    def tie(self, setting):
        """Tie this setting to another."""
        if self.tied != None:
            self.untie()
        self.tied = setting
        self.tied.setOnModified(self._tiedModified)

    def untie(self):
        """Untie association with another setting."""
        if self.tied != None:
            self.tied.removeOnModified(self._tiedModified)
            self.tied = None

    def _tiedModified(self, ismodified):
        """Called when the setting this setting is tied to is modified."""
        for mod in self.onmodified:
            mod(True)

    def toText(self):
        """Convert the type to text for saving."""
        return ""

    def fromText(self, text):
        """Convert text to type suitable for setting.

        Raises InvalidType if cannot convert."""
        return None

    def readDefaults(self, root, widgetname):
        """Check whether the user has a default for this setting."""

        deftext = None
        unnamedpath = '%s/%s' % (root, self.name)
        try:
            deftext = settingdb[unnamedpath]
        except KeyError:
            pass

        # named defaults supersedes normal defaults
        namedpath = '%s_NAME:%s' % (widgetname, unnamedpath)
        try:
            deftext = settingdb[namedpath]
        except KeyError:
            pass
    
        if deftext != None:
            self.val = self.fromText(deftext)
            self.default = self.val

    def removeDefault(self):
        """Remove the default setting for this setting."""

        # build up setting path
        path = ''
        item = self
        while not isinstance(item, widgets.Widget):
            path = '/%s%s' % (item.name, path)
            item = item.parent

        # remove the settings (ignore if they are not set)
        if path in settingdb:
            del settingdb[path]

        # specific setting to this widgetname
        namedpath = '%s_NAME:%s' % (item.name, path)

        if namedpath in settingdb:
            del settingdb[namedpath]

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
        settingdb[path] = self.toText()

    def saveText(self, saveall, rootname = ''):
        """Return text to restore the value of this setting."""

        if (saveall or not self.isDefault()) and not self.readonly:
            return "Set('%s%s', %s)\n" % ( rootname, self.name,
                                           repr(self.val) )
        else:
            return ''

    def resetToDefault(self):
        """Reset the value to its default."""
        self.val = self.default

    def setOnModified(self, fn):
        """Set the function to be called on modification (passing True)."""
        self.onmodified.append(fn)

    def removeOnModified(self, fn):
        """Remove the function from the list of function to be called."""
        i = self.onmodified.index(fn)
        del self.onmodified[i]

    def newDefault(self, value):
        """Update the default and the value."""
        self.default = value
        self.val = value

    def isDefault(self):
        """Is the current value a default?"""
        return self.val == self.default

    def setSilent(self, val):
        """Set the setting, without propagating modified flags.

        This shouldn't often be used as it defeats the automatic updation.
        Used for temporary modifications."""

        if self.tied == None:
            self._val = self.convertTo(val)
        else:
            self.tied.setSilent(val)

    def convertTo(self, val):
        """Convert for storage."""
        return val

    def convertFrom(self, val):
        """Convert to storage."""
        return val

    def makeControl(self, *args):
        """Make a qt control for editing the setting.

        The control emits settingValueChanged() when the setting has
        changed value."""

        return None

# Store strings
class Str(Setting):
    """String setting."""

    def convertTo(self, val):
        if type(val) in (str, unicode):
            return val
        raise InvalidType

    def toText(self):
        return self.val

    def fromText(self, text):
        return text

    def makeControl(self, *args):
        return controls.String(self, *args)

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
        return controls.Bool(self, *args)

# Storing integers
class Int(Setting):
    """Integer settings."""

    def __init__(self, name, value, minval=-1000000, maxval=1000000,
                 descr=''):
        """Initialise the values.

        minval is minimum possible value of setting
        maxval is maximum possible value of setting
        descr is a description of the setting
        """

        self.minval = minval
        self.maxval = maxval
        Setting.__init__(self, name, value, descr=descr)

    def convertTo(self, val):
        if type(val) == int:
            if val >= self.minval and val <= self.maxval:
                return val
            else:
                raise InvalidType, 'Out of range allowed'
        raise InvalidType

    def toText(self):
        return str(self.val)

    def fromText(self, text):
        try:
            i = int(text)
            if i >= self.minval and i <= self.maxval:
                return i
            else:
                raise InvalidType, 'Out of range allowed'
        except ValueError:
            raise InvalidType

    def makeControl(self, *args):
        return controls.Edit(self, *args)

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
        return controls.Edit(self, *args)

class FloatOrAuto(Setting):
    """Save a float or text auto."""

    def convertTo(self, val):
        if type(val) in (int, float):
            return float(val)
        elif type(val) in [str, unicode] and val.strip().lower() == 'auto':
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
        return controls.Choice(self, True, ['Auto'], *args)
            
class IntOrAuto(Setting):
    """Save an int or text auto."""

    def convertTo(self, val):
        if type(val) == int:
            return val
        elif type(val) in [str, unicode] and val.strip().lower() == 'auto':
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
        return controls.Choice(self, True, ['Auto'], *args)


# these are functions used by the distance setting below.
# they don't work as class methods

def _calcPixPerPt(painter):
    """Calculate the numbers of pixels per point for the painter.

    This is stored in the variable veusz_pixperpt."""

    dm = qt.QPaintDeviceMetrics(painter.device())
    painter.veusz_pixperpt = dm.logicalDpiY() / 72.

def _distPhys(match, painter, mult):
    """Convert a physical unit measure in multiples of points."""

    if not hasattr(painter, 'veusz_pixperpt'):
        _calcPixPerPt(painter)

    return int( math.ceil(painter.veusz_pixperpt * mult *
                          float(match.group(1)) * painter.veusz_scaling ) )

def _distPerc(match, painter, maxsize):
    """Convert from a percentage of maxsize."""

    return int( math.ceil(maxsize * 0.01 * float(match.group(1))) )

def _distFrac(match, painter, maxsize):
    """Convert from a fraction a/b of maxsize."""

    return int( math.ceil(maxsize * float(match.group(1)) /
                          float(match.group(2))) )

def _distRatio(match, painter, maxsize):
    """Convert from a simple 0.xx ratio of maxsize."""

    # if it's greater than 1 then assume it's a point measurement
    if float(match.group(1)) > 1.:
        return _distPhys(match, painter, 1)

    return int( math.ceil(maxsize * float(match.group(1))) )

# mappings from regular expressions to function to convert distance
# the recipient function takes regexp match,
# painter and maximum size of frac
_distregexp = [ ( re.compile('^([0-9\.]+) *%$'),
                  _distPerc ),
                ( re.compile('^([0-9\.]+) */ *([0-9\.]+)$'),
                  _distFrac ),
                ( re.compile('^([0-9\.]+) *pt$'),
                  lambda match, painter, t:
                  _distPhys(match, painter, 1.) ),
                ( re.compile('^([0-9\.]+) *cm$'),
                  lambda match, painter, t:
                  _distPhys(match, painter, 28.452756) ),
                ( re.compile('^([0-9\.]+) *mm$'),
                  lambda match, painter, t:
                  _distPhys(match, painter, 2.8452756) ),
                ( re.compile('^([0-9\.]+) *(inch|in|")$'),
                  lambda match, painter, t:
                  _distPhys(match, painter, 72.27) ),
                ( re.compile('^([0-9\.]+)$'),
                  _distRatio )
                ]

class Distance(Setting):
    """A veusz distance measure, e.g. 1pt or 3%."""

    def isDist(self, dist):
        """Is the text a valid distance measure?"""
        
        dist = dist.strip()
        for reg, fn in _distregexp:
            if reg.match(dist):
                return True
            
        return False

    def convertTo(self, val):
        if self.isDist(val):
            return val
        else:
            raise InvalidType

    def toText(self):
        return self.val

    def fromText(self, text):
        if self.isDist(text):
            return text
        else:
            raise InvalidType
        
    def makeControl(self, *args):
        return controls.Distance(self, *args)

    def convert(self, painter):
        '''Convert a distance to plotter units.

        dist: eg 0.1 (fraction), 10% (percentage), 1/10 (fraction),
                 10pt, 1cm, 20mm, 1inch, 1in, 1" (size)
        maxsize: size fractions are relative to
        painter: painter to get metrics to convert physical sizes
        '''

        # we set a scaling variable in the painter if it's not set
        if 'veusz_scaling' not in painter.__dict__:
            painter.veusz_scaling = 1.

        # work out maximum size
        try:
            maxsize = max( *painter.veusz_page_size )
        except AttributeError:
            w = painter.window()
            maxsize = max(w.width(), w.height())

        dist = self.val.strip()

        # compare string against each regexp
        for reg, fn in _distregexp:
            m = reg.match(dist)

            # if there's a match, then call the appropriate conversion fn
            if m:
                return fn(m, painter, maxsize)

        # none of the regexps match
        raise ValueError( "Cannot convert distance in form '%s'" %
                          dist )

    def convertPts(self, painter):
        """Get the distance in points."""
        if not hasattr(painter, 'veusz_pixperpt'):
            _calcPixPerPt(painter)

        return self.convert(painter) / painter.veusz_pixperpt
        
class Choice(Setting):
    """One out of a list of strings."""

    # maybe should be implemented as a dict to speed up checks

    def __init__(self, name, vallist, val, descr = ''):
        """Setting val must be in vallist."""
        
        assert type(vallist) in (list, tuple)
        self.vallist = vallist
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
        return controls.Choice(self, False, self.vallist, *args)

class ChoiceOrMore(Setting):
    """One out of a list of strings, or anything else."""

    # maybe should be implemented as a dict to speed up checks

    def __init__(self, name, vallist, val, descr = ''):
        """Setting has val must be in vallist."""
        
        self.vallist = vallist
        Setting.__init__(self, name, val, descr = descr)

    def convertTo(self, val):
        return val

    def toText(self):
        return self.val

    def fromText(self, text):
        return text

    def makeControl(self, *args):
        return controls.Choice(self, True, self.vallist, *args)

class FloatDict(Setting):
    """A dictionary, taking floats as values."""

    def convertTo(self, val):
        if type(val) != dict:
            raise InvalidType

        out = {}
        for key, val in val.iteritems():
            if type(val) not in (float, int):
                raise InvalidType
            else:
                out[key] = val

        return out

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
        return controls.MultiLine(self, *args)

class FloatList(Setting):
    """A list of float values."""

    def convertTo(self, val):
        if type(val) not in (list, tuple):
            raise InvalidType

        # horribly slow test for invalid entries
        out = []
        for i in val:
            if type(i) not in (float, int):
                raise InvalidType
            else:
                out.append( float(i) )
        return out

    def toText(self):
        """Make a string a, b, c."""
        return ', '.join( [str(i) for i in self.val] )

    def fromText(self, text):
        """Convert from a, b, c or a b c."""

        p = re.split(r'[\t\n, ]+', text.strip())

        try:
            out = [float(i) for i in p]
        except ValueError:
            raise InvalidType

        return out

    def makeControl(self, *args):
        return controls.String(self, *args)

class WidgetPath(Str):
    """A setting holding a path to a widget. This is checked for validity."""

    def __init__(self, name, val, relativetoparent=True,
                 allowedwidgets = None,
                 descr=''):
        """Initialise the setting.

        The widget is located relative to
        parent if relativetoparent is True, otherwise this widget.

        If allowedwidgets != None, only those widgets types in the list are
        allowed by this setting.
        """

        Str.__init__(self, name, val, descr=descr)
        self.relativetoparent = relativetoparent
        self.allowedwidgets = allowedwidgets

    def convertTo(self, val):
        """Validate the text is a name of a widget relative to
        this one."""

        if type(val) not in [str, unicode]:
            raise InvalidType

        # InvalidType will get raised in getWidget if it is incorrect
        w = self.getWidget(val)
        if w == None:
            return ''
        else:
            return val

    def getWidget(self, val = None):
        """Get the widget referred to. We double-check here to make sure
        it's the one.

        Returns None if setting is blank
        InvalidType is raised if there's a problem
        """

        # this is a bit of a hack, so we don't have to pass a value
        # for the setting (which we need to from convertTo)
        if val == None:
            val = self.val

        if val == '':
            return None

        # find the widget associated with this setting
        widget = self
        while not isinstance(widget, widgets.Widget):
            widget = widget.parent

        # usually makes sense to give paths relative to a parent of a widget
        if self.relativetoparent:
            widget = widget.parent

        # resolve the text to a widget
        try:
            widget = widget.document.resolve(widget, val)
        except ValueError:
            raise InvalidType

        # check the widget against the list of allowed types if given
        if self.allowedwidgets != None:
            allowed = False
            for c in self.allowedwidgets:
                if isinstance(widget, c):
                    allowed = True
            if not allowed:
                raise InvalidType
        
        return widget

class Dataset(Str):
    """A setting to choose from the possible datasets."""

    def __init__(self, name, val, document, dimensions=1, descr=''):
        """Initialise using the document, so we can get the datasets later.

        dimensions is the number of dimensions the dataset needs
        """

        Setting.__init__(self, name, val, descr)
        self.document = document
        self.dimensions = dimensions

    def makeControl(self, *args):
        """Allow user to choose between the datasets."""
        return controls.Dataset(self, self.document, self.dimensions, *args)
    
class Color(ChoiceOrMore):
    """A color setting."""

    _colors = [ 'white', 'black', 'red', 'green', 'blue',
                'cyan', 'magenta', 'yellow',
                'grey', 'darkred', 'darkgreen', 'darkblue',
                'darkcyan', 'darkmagenta' ]

    def __init__(self, name, default, descr = None):
        """Initialise the color setting with the given name, default
        and description."""
        
        ChoiceOrMore.__init__(self, name, self._colors, default,
                              descr=descr)

    def color(self):
        """Return QColor for color."""
        return qt.QColor(self.val)
    
    def makeControl(self, *args):
        return controls.Color(self, self._colors, *args)

class FillStyle(Choice):
    """A setting for the different fill styles provided by Qt."""
    
    _fillstyles = [ 'solid', 'horizontal', 'vertical', 'cross',
                    'forward diagonals', 'backward diagonals',
                    'diagonal cross',
                    '94% dense', '88% dense', '63% dense', '50% dense',
                    '37% dense', '12% dense', '6% dense' ]

    _fillcnvt = { 'solid': qt.Qt.SolidPattern,
                  'horizontal': qt.Qt.HorPattern,
                  'vertical': qt.Qt.VerPattern,
                  'cross': qt.Qt.CrossPattern,
                  'forward diagonals': qt.Qt.FDiagPattern,
                  'backward diagonals': qt.Qt.BDiagPattern,
                  'diagonal cross': qt.Qt.DiagCrossPattern,
                  '94% dense': qt.Qt.Dense1Pattern,
                  '88% dense': qt.Qt.Dense2Pattern,
                  '63% dense': qt.Qt.Dense3Pattern,
                  '50% dense': qt.Qt.Dense4Pattern,
                  '37% dense': qt.Qt.Dense5Pattern,
                  '12% dense': qt.Qt.Dense6Pattern,
                  '6% dense': qt.Qt.Dense7Pattern }

    def __init__(self, name, default, descr=None):
        Choice.__init__(self, name, self._fillstyles, default, descr=descr)

    def qtStyle(self):
        """Return Qt ID of fill."""
        return self._fillcnvt[self.val]

    def makeControl(self, *args):
        return controls.FillStyle(self, self._fillstyles,
                                  self._fillcnvt, *args)

class LineStyle(Choice):
    """A setting choosing a particular line style."""

    _linestyles = ['solid', 'dashed', 'dotted',
                   'dash-dot', 'dash-dot-dot' ]

    _linecnvt = { 'solid': qt.Qt.SolidLine, 'dashed': qt.Qt.DashLine,
                  'dotted': qt.Qt.DotLine, 'dash-dot': qt.Qt.DashDotLine,
                  'dash-dot-dot': qt.Qt.DashDotDotLine }
    
    def __init__(self, name, default, descr=None):
        Choice.__init__(self, name, self._linestyles, default, descr=descr)

    def qtStyle(self):
        """Get Qt ID of chosen line style."""
        return self._linecnvt[self.val]
    
    def makeControl(self, *args):
        return controls.LineStyle(self, self._linestyles,
                                  self._linecnvt, *args)

class Axis(Str):
    """A setting to hold the name of an axis."""

    def __init__(self, name, val, document, direction, descr=''):
        """Initialise using the document, so we can get the axes later.
        
        direction is horizontal or vertical to specify the type of axis to
        show
        """

        Setting.__init__(self, name, val, descr)
        self.document = document
        self.direction = direction

    def makeControl(self, *args):
        """Allows user to choose an axis or enter a name."""

        return controls.Axis(self, self.document, self.direction, *args)
    
class Marker(Choice):
    """Choose a marker type from one allowable."""

    def __init__(self, name, default, descr=None):
        Choice.__init__(self, name, utils.MarkerCodes, default, descr=descr)

    def makeControl(self, *args):
        return controls.Marker(self, *args)
    

    
