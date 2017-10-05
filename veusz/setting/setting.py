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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
###############################################################################

"""Module for holding setting values.

e.g.

s = Int('foo', 5)
s.get()
s.set(42)
s.fromUIText('42')
"""

from __future__ import division
import re
import sys

import numpy as N

from ..compat import cbasestr, cstr, crepr
from .. import qtall as qt4
from . import controls
from .settingdb import settingdb, uilocale, ui_floattostring, ui_stringtofloat
from .reference import ReferenceBase, Reference

from .. import utils
from .. import datasets

class OnModified(qt4.QObject):
    """onmodified is emitted from an object contained in each setting."""
    onModified = qt4.pyqtSignal()

class Setting(object):
    """A class to store a value with a particular type."""

    # differentiate widgets, settings and setting
    nodetype = 'setting'

    typename = 'setting'

    # various items in class hierarchy
    iswidget = False
    issetting = True
    issettings = False

    def __init__(self, name, value, descr='', usertext='',
                 formatting=False, hidden=False):
        """Initialise the values.

        name: setting name
        value: default value and initial value
        descr:  description of the setting
        usertext: name of setting for user
        formatting: whether setting applies to formatting
        hidden: hide widget from user
        """
        self.readonly = False
        self.parent = None
        self.name = name
        self.descr = descr
        self.usertext = usertext
        self.formatting = formatting
        self.hidden = hidden
        self.default = value
        self.onmodified = OnModified()
        self._val = self._ref = None

        # calls the set function for the val property
        self.val = value

    def _copyHelper(self, before, after, optional):
        """Help copy an object.

        before are arguments before val
        after are arguments after val
        optinal as optional arguments
        """

        val = self._ref if self._ref else self._val

        args = (self.name,) + before + (val,) + after

        opt = optional.copy()
        opt['descr'] = self.descr
        opt['usertext'] = self.usertext
        opt['formatting'] = self.formatting
        opt['hidden'] = self.hidden

        obj = self.__class__(*args, **opt)
        obj.readonly = self.readonly
        obj.default = self.default
        return obj

    def copy(self):
        """Make a setting which has its values copied from this one.

        This needs to be overridden if the constructor changes
        """
        return self._copyHelper((), (), {})

    def get(self):
        """Get the value."""

        if self._ref:
            return self._ref.resolve(self).get()
        else:
            return self._val

    def set(self, v):
        """Set the value."""

        if isinstance(v, ReferenceBase):
            self._val = None
            self._ref = v
        else:
            self._val = self.normalize(v)
            self._ref = None

        self.onmodified.onModified.emit()

    val = property(
        get, set, None,
        'Get or modify the value of the setting')

    def isReference(self):
        """Is this a setting a reference to another object."""
        return bool(self._ref)

    def getReference(self):
        """Return the reference object. Raise ValueError if not a reference"""
        if self._ref:
            return self._ref
        else:
            raise ValueError("Setting is not a reference")

    def getStylesheetLink(self):
        """Get text that this setting should default to linked to the
        stylesheet."""
        path = []
        obj = self
        while not obj.parent.iswidget:
            path.insert(0, obj.name)
            obj = obj.parent
        path = ['', 'StyleSheet', obj.parent.typename] + path
        return '/'.join(path)

    def linkToStylesheet(self):
        """Make this setting link to stylesheet setting, if possible."""
        self.set( Reference(self.getStylesheetLink()) )

    @property
    def path(self):
        """Return full path of setting."""
        path = []
        obj = self
        while obj is not None:
            # logic easier to understand here
            # do not add settings name for settings of widget
            if not obj.iswidget and obj.parent.iswidget:
                pass
            else:
                if obj.name == '/':
                    path.insert(0, '')
                else:
                    path.insert(0, obj.name)
            obj = obj.parent
        return '/'.join(path)

    def toUIText(self):
        """Convert the type to text to show in UI."""
        return ""

    def fromUIText(self, text):
        """Convert text from UI into type for setting.

        Raises utils.InvalidType if cannot convert."""
        return None

    def saveText(self, saveall, rootname = ''):
        """Return text to restore the value of this setting."""

        if (saveall or not self.isDefault()) and not self.readonly:
            if self._ref:
                return "SetToReference('%s%s', %s)\n" % (
                    rootname, self.name, crepr(self._ref.value))
            else:
                return "Set('%s%s', %s)\n" % (
                    rootname, self.name, crepr(self.val) )
        else:
            return ''

    def setOnModified(self, fn):
        """Set the function to be called on modification (passing True)."""
        self.onmodified.onModified.connect(fn)

        if self._ref:
            # tell references to notify us if they are modified
            self._ref.setOnModified(self, fn)

    def removeOnModified(self, fn):
        """Remove the function from the list of function to be called."""
        self.onmodified.onModified.disconnect(fn)

    def newDefault(self, value):
        """Update the default and the value."""
        self.default = value
        self.val = value

    def isDefault(self):
        """Is the current value a default?
        This also returns true if it is linked to the appropriate stylesheet
        """

        if self._ref and isinstance(self.default, ReferenceBase):
            return self._ref.value == self.default.value
        else:
            return self._val == self.default

    def isDefaultLink(self):
        """Is this a link to the default stylesheet value."""
        return self._ref and self._ref.value == self.getStylesheetLink()

    def setSilent(self, val):
        """Set the setting, without propagating modified flags.

        This shouldn't often be used as it defeats the automatic updation.
        Used for temporary modifications."""

        self._ref = None
        self._val = self.normalize(val)

    def normalize(self, val):
        """Convert external value to normalized form for storing

        Raises a utils.InvalidType if this is not possible."""
        return val

    def makeControl(self, *args):
        """Make a qt control for editing the setting.

        The control emits settingValueChanged() when the setting has
        changed value."""

        return None

    def getDocument(self):
        """Return document."""
        p = self.parent
        while p:
            if p.iswidget:
                return p.document
            p = p.parent
        return None

    def getWidget(self):
        """Return associated widget."""
        w = self.parent
        while not w.iswidget:
            w = w.parent
        return w

    def safeEvalHelper(self, text):
        """Evaluate an expression, catching naughtiness."""
        try:
            comp = self.getDocument().evaluate.compileCheckedExpression(
                text)
            if comp is None:
                raise utils.InvalidType
            return float( eval(comp, self.getDocument().evaluate.context) )
        except:
            raise utils.InvalidType

# forward setting to another setting
class SettingBackwardCompat(Setting):
    """Forward setting requests to another setting.

    This is used for backward-compatibility.
    """

    typename = 'backward-compat'

    def __init__(self, name, newrelpath, val, translatefn=None,
                 **args):
        """Point this setting to another.
        newrelpath is a path relative to this setting's parent
        """

        self.translatefn = translatefn
        args['hidden'] = True
        Setting.__init__(self, name, val, **args)
        self.relpath = newrelpath

    def getForward(self):
        """Get setting this setting forwards to."""
        doc = self.getDocument()
        return doc.resolveSettingPath(self.parent, self.relpath)

    def normalize(self, val):
        if self.parent is not None:
            return self.getForward().normalize(val)

    def toUIText(self):
        return self.getForward().toUIText()

    def fromUIText(self, val):
        return self.getForward().fromUIText(val)

    def set(self, val):
        if self.parent is not None and not isinstance(val, ReferenceBase):
            if self.translatefn:
                val = self.translatefn(val)
            self.getForward().set(val)

    def isDefault(self):
        return self.getForward().isDefault()

    def get(self):
        return self.getForward().get()

    def copy(self):
        return self._copyHelper(
            (self.relpath,), (), {'translatefn': self.translatefn})

    def makeControl(self, *args):
        return None

    def saveText(self, saveall, rootname = ''):
        return ''

    def linkToStylesheet(self):
        """Do nothing for backward compatibility settings."""
        pass

# Store strings
class Str(Setting):
    """String setting."""

    typename = 'str'

    def normalize(self, val):
        if isinstance(val, cbasestr):
            return val
        raise utils.InvalidType

    def toUIText(self):
        return self.val

    def fromUIText(self, text):
        return text

    def makeControl(self, *args):
        return controls.String(self, *args)

class Notes(Str):
    """String for making notes."""

    typename = 'str-notes'

    def makeControl(self, *args):
        return controls.Notes(self, *args)

# Store bools
class Bool(Setting):
    """Bool setting."""

    typename = 'bool'

    def normalize(self, val):
        if type(val) in (bool, int):
            return bool(val)
        raise utils.InvalidType

    def toUIText(self):
        return 'True' if self.val else 'False'

    def fromUIText(self, text):
        t = text.strip().lower()
        if t in ('true', '1', 't', 'y', 'yes'):
            return True
        elif t in ('false', '0', 'f', 'n', 'no'):
            return False
        else:
            raise utils.InvalidType

    def makeControl(self, *args):
        return controls.Bool(self, *args)

# Storing integers
class Int(Setting):
    """Integer settings."""

    typename = 'int'

    def __init__(self, name, value, minval=-1000000, maxval=1000000,
                 **args):
        """Initialise the values.

        minval is minimum possible value of setting
        maxval is maximum possible value of setting
        """

        self.minval = minval
        self.maxval = maxval
        Setting.__init__(self, name, value, **args)

    def copy(self):
        """Make a setting which has its values copied from this one.

        This needs to be overridden if the constructor changes
        """
        return self._copyHelper((), (), {'minval': self.minval,
                                         'maxval': self.maxval})

    def normalize(self, val):
        if isinstance(val, int):
            if val >= self.minval and val <= self.maxval:
                return val
            else:
                raise utils.InvalidType('Out of range allowed')
        raise utils.InvalidType

    def toUIText(self):
        return uilocale.toString(self.val)

    def fromUIText(self, text):
        i, ok = uilocale.toLongLong(text)
        if not ok:
            raise ValueError

        if i >= self.minval and i <= self.maxval:
            return i
        else:
            raise utils.InvalidType('Out of range allowed')

    def makeControl(self, *args):
        return controls.Int(self, *args)

def _finiteRangeFloat(f, minval=-1e300, maxval=1e300):
    """Return a finite float in range or raise exception otherwise."""
    f = float(f)
    if not N.isfinite(f):
        raise utils.InvalidType('Finite values only allowed')
    if f < minval or f > maxval:
        raise utils.InvalidType('Out of range allowed')
    return f

# for storing floats
class Float(Setting):
    """Float settings."""

    typename = 'float'

    def __init__(self, name, value, minval=-1e200, maxval=1e200,
                 **args):
        """Initialise the values.

        minval is minimum possible value of setting
        maxval is maximum possible value of setting
        """

        self.minval = minval
        self.maxval = maxval
        Setting.__init__(self, name, value, **args)

    def copy(self):
        """Make a setting which has its values copied from this one.

        This needs to be overridden if the constructor changes
        """
        return self._copyHelper((), (), {'minval': self.minval,
                                         'maxval': self.maxval})

    def normalize(self, val):
        if isinstance(val, int) or isinstance(val, float):
            return _finiteRangeFloat(
                val, minval=self.minval, maxval=self.maxval)
        raise utils.InvalidType

    def toUIText(self):
        return ui_floattostring(self.val)

    def fromUIText(self, text):
        try:
            f = ui_stringtofloat(text)
        except ValueError:
            # try to evaluate
            f = self.safeEvalHelper(text)
        return self.normalize(f)

    def makeControl(self, *args):
        return controls.Edit(self, *args)

class FloatOrAuto(Float):
    """Save a float or text auto."""

    typename = 'float-or-auto'

    def normalize(self, val):
        if type(val) in (int, float):
            return _finiteRangeFloat(val, minval=self.minval, maxval=self.maxval)
        elif isinstance(val, cbasestr) and val.strip().lower() == 'auto':
            return 'Auto'
        else:
            raise utils.InvalidType

    def toUIText(self):
        if isinstance(self.val, cbasestr) and self.val.lower() == 'auto':
            return 'Auto'
        else:
            return ui_floattostring(self.val)

    def fromUIText(self, text):
        if text.strip().lower() == 'auto':
            return 'Auto'
        else:
            return Float.fromUIText(self, text)

    def makeControl(self, *args):
        return controls.Choice(self, True, ['Auto'], *args)

class IntOrAuto(Setting):
    """Save an int or text auto."""

    typename = 'int-or-auto'

    def normalize(self, val):
        if isinstance(val, int):
            return val
        elif isinstance(val, cbasestr) and val.strip().lower() == 'auto':
            return 'Auto'
        else:
            raise utils.InvalidType

    def toUIText(self):
        if isinstance(self.val, cbasestr) and self.val.lower() == 'auto':
            return 'Auto'
        else:
            return uilocale.toString(self.val)

    def fromUIText(self, text):
        if text.strip().lower() == 'auto':
            return 'Auto'
        else:
            i, ok = uilocale.toLongLong(text)
            if not ok:
                raise utils.InvalidType
            return i

    def makeControl(self, *args):
        return controls.Choice(self, True, ['Auto'], *args)

# these are functions used by the distance setting below.
# they don't work as class methods

def _distPhys(match, painter, mult):
    """Convert a physical unit measure in multiples of points."""
    return (painter.pixperpt * mult *
            float(match.group(1)) * painter.scaling)

def _idistval(val, unit):
    """Convert value to text, dropping zeros and . points on right."""
    return ("%.3f" % val).rstrip('0').rstrip('.') + unit

def _distInvPhys(pixdist, painter, mult, unit):
    """Convert number of pixels into physical distance."""
    return _idistval( pixdist / (mult * painter.pixperpt * painter.scaling),
                      unit )

def _distPerc(match, painter):
    """Convert from a percentage of maxdim."""
    return painter.maxdim * 0.01 * float(match.group(1))

def _distInvPerc(pixdist, painter):
    """Convert pixel distance into percentage."""
    return _idistval(pixdist * 100. / painter.maxdim, '%')

def _distFrac(match, painter):
    """Convert from a fraction a/b of maxdim."""
    try:
        return painter.maxdim * float(match.group(1))/float(match.group(4))
    except ZeroDivisionError:
        return 0.

def _distRatio(match, painter):
    """Convert from a simple 0.xx ratio of maxdim."""

    # if it's greater than 1 then assume it's a point measurement
    if float(match.group(1)) > 1.:
        return _distPhys(match, painter, 1)

    return painter.maxdim * float(match.group(1))

# regular expression to match distances
distre_expr = r'''^
 [ ]*                                # optional whitespace

 (\.?[0-9]+|[0-9]+\.[0-9]*)          # a floating point number

 [ ]*                                # whitespace

 (cm|pt|mm|inch|in|"|%||             # ( unit, no unit,
  (?P<slash>/) )                     # or / )

 (?(slash)[ ]*                       # if it was a slash, match any whitespace
  (\.?[0-9]+|[0-9]+\.[0-9]*))        # and match following fp number

 [ ]*                                # optional whitespace
$'''

class Distance(Setting):
    """A veusz distance measure, e.g. 1pt or 3%."""

    typename = 'distance'

    # match a distance
    distre = re.compile(distre_expr, re.VERBOSE)

    # functions to convert from unit values to points
    unit_func = {
        'cm': lambda match, painter:
            _distPhys(match, painter, 720/25.4),
        'pt': lambda match, painter:
            _distPhys(match, painter, 1.),
        'mm': lambda match, painter:
            _distPhys(match, painter, 72/25.4),
        'in': lambda match, painter:
            _distPhys(match, painter, 72.),
        'inch': lambda match, painter:
            _distPhys(match, painter, 72.),
        '"': lambda match, painter:
            _distPhys(match, painter, 72.),
        '%': _distPerc,
        '/': _distFrac,
        '': _distRatio
        }

    # inverse functions for converting points to units
    inv_unit_func = {
        'cm': lambda match, painter:
            _distInvPhys(match, painter, 720/25.4, 'cm'),
        'pt': lambda match, painter:
            _distInvPhys(match, painter, 1., 'pt'),
        'mm': lambda match, painter:
            _distInvPhys(match, painter, 72/25.4, 'mm'),
        'in': lambda match, painter:
            _distInvPhys(match, painter, 72., 'in'),
        'inch': lambda match, painter:
            _distInvPhys(match, painter, 72., 'in'),
        '"': lambda match, painter:
            _distInvPhys(match, painter, 72., 'in'),
        '%': _distInvPerc,
        '/': _distInvPerc,
        '': _distInvPerc
        }

    @classmethod
    def isDist(kls, dist):
        """Is the text a valid distance measure?"""

        return kls.distre.match(dist) is not None

    def normalize(self, val):
        if self.distre.match(val) is not None:
            return val
        else:
            raise utils.InvalidType

    def toUIText(self):
        # convert decimal point to display locale
        return self.val.replace('.', uilocale.decimalPoint())

    def fromUIText(self, text):
        # convert decimal point from display locale
        text = text.replace(uilocale.decimalPoint(), '.')

        if self.isDist(text):
            return text
        else:
            raise utils.InvalidType

    def makeControl(self, *args):
        return controls.Distance(self, *args)

    @classmethod
    def convertDistance(kls, painter, dist):
        '''Convert a distance to plotter units.

        dist: eg 0.1 (fraction), 10% (percentage), 1/10 (fraction),
                 10pt, 1cm, 20mm, 1inch, 1in, 1" (size)
        painter: painter to get metrics to convert physical sizes
        '''

        # match distance against expression
        m = kls.distre.match(dist)
        if m is not None:
            # lookup function to call to do conversion
            func = kls.unit_func[m.group(2)]
            return func(m, painter)

        # none of the regexps match
        raise ValueError( "Cannot convert distance in form '%s'" %
                          dist )

    def convert(self, painter):
        """Convert this setting's distance as above"""
        return self.convertDistance(painter, self.val)

    def convertPts(self, painter):
        """Get the distance in points."""
        return self.convert(painter) / painter.pixperpt

    def convertInverse(self, distpix, painter):
        """Convert distance in pixels into units of this distance.
        """

        m = self.distre.match(self.val)
        if m is not None:
            # if it matches convert back
            inversefn = self.inv_unit_func[m.group(2)]
        else:
            # otherwise force unit
            inversefn = self.inv_unit_func['cm']

        # do inverse mapping
        return inversefn(distpix, painter)

class DistancePt(Distance):
    """For a distance in points."""

    def makeControl(self, *args):
        return controls.DistancePt(self, *args)

class DistancePhysical(Distance):
    """For physical distances (no fractional)."""

    def isDist(self, val):
        m = self.distre.match(val)
        if m:
            # disallow non-physical distances
            if m.group(2) not in ('/', '', '%'):
                return True
        return False

    def makeControl(self, *args):
        return controls.Distance(self, *args, physical=True)

class DistanceOrAuto(Distance):
    """A distance or the value Auto"""

    typename = 'distance-or-auto'

    distre = re.compile( distre_expr + r'|^Auto$', re.VERBOSE )

    def isAuto(self):
        return self.val == 'Auto'

    def makeControl(self, *args):
        return controls.Distance(self, allowauto=True, *args)

class Choice(Setting):
    """One out of a list of strings."""

    # maybe should be implemented as a dict to speed up checks

    typename = 'choice'

    def __init__(self, name, vallist, val, **args):
        """Setting val must be in vallist.
        descriptions is an optional addon to put a tooltip on each item
        in the control.
        """

        assert type(vallist) in (list, tuple)

        self.vallist = vallist
        self.descriptions = args.get('descriptions', None)
        if self.descriptions:
            del args['descriptions']

        Setting.__init__(self, name, val, **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((self.vallist,), (), {})

    def normalize(self, val):
        if val in self.vallist:
            return val
        else:
            raise utils.InvalidType

    def toUIText(self):
        return self.val

    def fromUIText(self, text):
        if text in self.vallist:
            return text
        else:
            raise utils.InvalidType

    def makeControl(self, *args):
        argsv = {'descriptions': self.descriptions}
        return controls.Choice(self, False, self.vallist, *args, **argsv)

class ChoiceOrMore(Setting):
    """One out of a list of strings, or anything else."""

    # maybe should be implemented as a dict to speed up checks

    typename = 'choice-or-more'

    def __init__(self, name, vallist, val, **args):
        """Setting has val must be in vallist.
        descriptions is an optional addon to put a tooltip on each item
        in the control
        """

        self.vallist = vallist
        self.descriptions = args.get('descriptions', None)
        if self.descriptions:
            del args['descriptions']

        Setting.__init__(self, name, val, **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((self.vallist,), (), {})

    def normalize(self, val):
        return val

    def toUIText(self):
        return self.val

    def fromUIText(self, text):
        return text

    def makeControl(self, *args):
        argsv = {'descriptions': self.descriptions}
        return controls.Choice(self, True, self.vallist, *args, **argsv)

class FloatChoice(ChoiceOrMore):
    """A numeric value, which can also be chosen from the list of values."""

    typename = 'float-choice'

    def normalize(self, val):
        if isinstance(val, int) or isinstance(val, float):
            return _finiteRangeFloat(val)
        raise utils.InvalidType

    def toUIText(self):
        return ui_floattostring(self.val)

    def fromUIText(self, text):
        try:
            f = ui_stringtofloat(text)
        except ValueError:
            # try to evaluate
            f = self.safeEvalHelper(text)
        return self.normalize(f)

    def makeControl(self, *args):
        argsv = {'descriptions': self.descriptions}
        strings = [ui_floattostring(x) for x in self.vallist]
        return controls.Choice(self, True, strings, *args, **argsv)

class FloatDict(Setting):
    """A dictionary, taking floats as values."""

    typename = 'float-dict'

    def normalize(self, val):
        if type(val) != dict:
            raise utils.InvalidType

        for v in val.values():
            if type(v) not in (float, int):
                raise utils.InvalidType

        # return copy
        return dict(val)

    def toUIText(self):
        text = ['%s = %s' % (k, ui_floattostring(self.val[k]))
                for k in sorted(self.val)]
        return '\n'.join(text)

    def fromUIText(self, text):
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
                raise utils.InvalidType

            try:
                v = ui_stringtofloat(p[1])
            except ValueError:
                raise utils.InvalidType

            out[ p[0].strip() ] = v
        return out

    def makeControl(self, *args):
        return controls.MultiLine(self, *args)

class FloatList(Setting):
    """A list of float values."""

    typename = 'float-list'

    def normalize(self, val):
        if type(val) not in (list, tuple):
            raise utils.InvalidType

        # horribly slow test for invalid entries
        out = []
        for i in val:
            if type(i) not in (float, int):
                raise utils.InvalidType
            else:
                out.append( float(i) )
        return out

    def toUIText(self):
        """Make a string a, b, c."""
        # can't use the comma for splitting if used as a decimal point

        join = ', '
        if uilocale.decimalPoint() == ',':
            join = '; '
        return join.join( [ui_floattostring(x) for x in self.val] )

    def fromUIText(self, text):
        """Convert from a, b, c or a b c."""

        # don't use commas if it is the decimal separator
        splitre = r'[\t\n, ]+'
        if uilocale.decimalPoint() == ',':
            splitre = r'[\t\n; ]+'

        out = []
        for x in re.split(splitre, text.strip()):
            if x:
                try:
                    out.append( ui_stringtofloat(x) )
                except ValueError:
                    out.append( self.safeEvalHelper(x) )
        return out

    def makeControl(self, *args):
        return controls.String(self, *args)

class WidgetPath(Str):
    """A setting holding a path to a widget. This is checked for validity."""

    typename = 'widget-path'

    def __init__(self, name, val, relativetoparent=True,
                 allowedwidgets = None,
                 **args):
        """Initialise the setting.

        The widget is located relative to
        parent if relativetoparent is True, otherwise this widget.

        If allowedwidgets is not None, only those widgets types in the list are
        allowed by this setting.
        """

        Str.__init__(self, name, val, **args)
        self.relativetoparent = relativetoparent
        self.allowedwidgets = allowedwidgets

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (),
                                {'relativetoparent': self.relativetoparent,
                                 'allowedwidgets': self.allowedwidgets})

    def getReferredWidget(self, val = None):
        """Get the widget referred to. We double-check here to make sure
        it's the one.

        Returns None if setting is blank
        utils.InvalidType is raised if there's a problem
        """

        # this is a bit of a hack, so we don't have to pass a value
        # for the setting (which we need to from normalize)
        if val is None:
            val = self.val

        if val == '':
            return None

        # find the widget associated with this setting
        widget = self
        while not widget.iswidget:
            widget = widget.parent

        # usually makes sense to give paths relative to a parent of a widget
        if self.relativetoparent:
            widget = widget.parent

        # resolve the text to a widget
        try:
            widget = widget.document.resolveWidgetPath(widget, val)
        except ValueError:
            raise utils.InvalidType

        # check the widget against the list of allowed types if given
        if self.allowedwidgets is not None:
            allowed = False
            for c in self.allowedwidgets:
                if isinstance(widget, c):
                    allowed = True
            if not allowed:
                raise utils.InvalidType

        return widget

class Dataset(Str):
    """A setting to choose from the possible datasets."""

    typename = 'dataset'

    def __init__(self, name, val, dimensions=1, datatype='numeric',
                 **args):
        """
        dimensions is the number of dimensions the dataset needs
        """

        self.dimensions = dimensions
        self.datatype = datatype
        Setting.__init__(self, name, val, **args)

    def copy(self):
        """Make a setting which has its values copied from this one."""
        return self._copyHelper((), (),
                                {'dimensions': self.dimensions,
                                 'datatype': self.datatype})

    def makeControl(self, *args):
        """Allow user to choose between the datasets."""
        return controls.Dataset(self, self.getDocument(), self.dimensions,
                                self.datatype, *args)

    def getData(self, doc):
        """Return a list of datasets entered."""
        d = doc.data.get(self.val)
        if ( d is not None and
             d.datatype == self.datatype and
             (d.dimensions == self.dimensions or self.dimensions == 'all') ):
                 return d

class Strings(Setting):
    """A multiple set of strings."""

    typename = 'str-multi'

    def normalize(self, val):
        """Takes a tuple/list of strings:
        ('ds1','ds2'...)
        """

        if isinstance(val, cbasestr):
            return (val, )

        if type(val) not in (list, tuple):
            raise utils.InvalidType

        # check each entry in the list is appropriate
        for ds in val:
            if not isinstance(ds, cbasestr):
                raise utils.InvalidType

        return tuple(val)

    def makeControl(self, *args):
        """Allow user to choose between the datasets."""
        return controls.Strings(self, self.getDocument(), *args)

class Datasets(Setting):
    """A setting to choose one or more of the possible datasets."""

    typename = 'dataset-multi'

    def __init__(self, name, val, dimensions=1, datatype='numeric',
                 **args):
        """
        dimensions is the number of dimensions the dataset needs
        """

        Setting.__init__(self, name, val, **args)
        self.dimensions = dimensions
        self.datatype = datatype

    def normalize(self, val):
        """Takes a tuple/list of strings:
        ('ds1','ds2'...)
        """

        if isinstance(val, cbasestr):
            return (val, )

        if type(val) not in (list, tuple):
            raise utils.InvalidType

        # check each entry in the list is appropriate
        for ds in val:
            if not isinstance(ds, cbasestr):
                raise utils.InvalidType

        return tuple(val)

    def copy(self):
        """Make a setting which has its values copied from this one."""
        return self._copyHelper((), (),
                                {'dimensions': self.dimensions,
                                 'datatype': self.datatype})

    def makeControl(self, *args):
        """Allow user to choose between the datasets."""
        return controls.Datasets(self, self.getDocument(), self.dimensions,
                                 self.datatype, *args)

    def getData(self, doc):
        """Return a list of datasets entered."""
        out = []
        for name in self.val:
            d = doc.data.get(name)
            if ( d is not None and
                 d.datatype == self.datatype and
                 d.dimensions == self.dimensions ):
                out.append(d)
        return out

class DatasetExtended(Dataset):
    """Choose a dataset, give an expression or specify a list of float
    values."""

    typename = 'dataset-extended'

    def normalize(self, val):
        """Check is a string (dataset name or expression) or a list of
        floats (numbers).
        """

        if isinstance(val, cbasestr):
            return val
        elif self.dimensions == 1:
            # list of numbers only allowed for 1d datasets
            if isinstance(val, float) or isinstance(val, int):
                return [val]
            else:
                try:
                    return [float(x) for x in val]
                except (TypeError, ValueError):
                    pass
        raise utils.InvalidType

    def toUIText(self):
        if isinstance(self.val, cbasestr):
            return self.val
        else:
            # join based on , or ; depending on decimal point
            join = ', '
            if uilocale.decimalPoint() == ',':
                join = '; '
            return join.join( [ ui_floattostring(x)
                                for x in self.val ] )

    def fromUIText(self, text):
        """Convert from text."""

        text = text.strip()

        if self.dimensions > 1:
            return text

        # split based on , or ; depending on decimal point
        splitre = r'[\t\n, ]+'
        if uilocale.decimalPoint() == ',':
            splitre = r'[\t\n; ]+'

        out = []
        for x in re.split(splitre, text):
            if x:
                try:
                    out.append( ui_stringtofloat(x) )
                except ValueError:
                    # fail conversion, so exit with text
                    return text
        return out

    def getFloatArray(self, doc):
        """Get a numpy of values or None."""
        if isinstance(self.val, cbasestr):
            ds = doc.evaluate.evalDatasetExpression(
                self.val, datatype=self.datatype, dimensions=self.dimensions)
            if ds:
                # get numpy array of values
                return N.array(ds.data)
        else:
            # list of values
            return N.array(self.val)
        return None

    def isDataset(self, doc):
        """Is this setting a dataset?"""
        return (isinstance(self.val, cbasestr) and
                doc.data.get(self.val))

    def isEmpty(self):
        """Is this unset?"""
        return self.val == [] or self.val == ''

    def getData(self, doc):
        """Return veusz dataset"""
        if isinstance(self.val, cbasestr):
            return doc.evaluate.evalDatasetExpression(
                self.val, datatype=self.datatype, dimensions=self.dimensions)
        else:
            return datasets.valsToDataset(
                self.val, self.datatype, self.dimensions)

class DatasetOrStr(Dataset):
    """Choose a dataset or enter a string.

    Non string datasets are converted to string arrays using this.
    """

    typename = 'dataset-or-str'

    def __init__(self, name, val, **args):
        Dataset.__init__(self, name, val, datatype='text', **args)

    def getData(self, doc, checknull=False):
        """Return either a list of strings, a single item list.
        If checknull then None is returned if blank
        """
        if doc:
            ds = doc.data.get(self.val)
            if ds and ds.dimensions == 1:
                return doc.formatValsWithDatatypeToText(
                    ds.data, ds.displaytype)
        if checknull and not self.val:
            return None
        else:
            return [cstr(self.val)]

    def makeControl(self, *args):
        return controls.DatasetOrString(self, self.getDocument(), *args)

    def copy(self):
        """Make a setting which has its values copied from this one."""
        return self._copyHelper((), (), {})

class Color(ChoiceOrMore):
    """A color setting."""

    typename = 'color'

    def __init__(self, name, value, **args):
        """Initialise the color setting with the given name, default
        and description."""
        ChoiceOrMore.__init__(self, name, [], value, **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

    def color(self, painter, dataindex=0):
        """Return QColor from color.

        painter is a Veusz Painter
        dataindex is index for automatically getting colors for subdatasets.
        """

        if self.val.lower() == 'auto':
            # lookup widget
            w = self.parent
            while w is not None and not w.iswidget:
                w = w.parent
            if w is None:
                return qt4.QColor()
            # get automatic color
            return painter.docColor(w.autoColor(painter, dataindex=dataindex))
        else:
            return painter.docColor(self.val)

    def makeControl(self, *args):
        return controls.Color(self, *args)

class FillStyle(Choice):
    """A setting for the different fill styles provided by Qt."""

    typename = 'fill-style'

    _fillstyles = [ 'solid', 'horizontal', 'vertical', 'cross',
                    'forward diagonals', 'backward diagonals',
                    'diagonal cross',
                    '94% dense', '88% dense', '63% dense', '50% dense',
                    '37% dense', '12% dense', '6% dense' ]

    _fillcnvt = { 'solid': qt4.Qt.SolidPattern,
                  'horizontal': qt4.Qt.HorPattern,
                  'vertical': qt4.Qt.VerPattern,
                  'cross': qt4.Qt.CrossPattern,
                  'forward diagonals': qt4.Qt.FDiagPattern,
                  'backward diagonals': qt4.Qt.BDiagPattern,
                  'diagonal cross': qt4.Qt.DiagCrossPattern,
                  '94% dense': qt4.Qt.Dense1Pattern,
                  '88% dense': qt4.Qt.Dense2Pattern,
                  '63% dense': qt4.Qt.Dense3Pattern,
                  '50% dense': qt4.Qt.Dense4Pattern,
                  '37% dense': qt4.Qt.Dense5Pattern,
                  '12% dense': qt4.Qt.Dense6Pattern,
                  '6% dense': qt4.Qt.Dense7Pattern }

    controls.FillStyle._fills = _fillstyles
    controls.FillStyle._fillcnvt = _fillcnvt

    def __init__(self, name, value, **args):
        Choice.__init__(self, name, self._fillstyles, value, **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

    def qtStyle(self):
        """Return Qt ID of fill."""
        return self._fillcnvt[self.val]

    def makeControl(self, *args):
        return controls.FillStyle(self, *args)

class LineStyle(Choice):
    """A setting choosing a particular line style."""

    typename = 'line-style'

    # list of allowed line styles
    _linestyles = ['solid', 'dashed', 'dotted',
                   'dash-dot', 'dash-dot-dot', 'dotted-fine',
                   'dashed-fine', 'dash-dot-fine',
                   'dot1', 'dot2', 'dot3', 'dot4',
                   'dash1', 'dash2', 'dash3', 'dash4', 'dash5',
                   'dashdot1', 'dashdot2', 'dashdot3']

    # convert from line styles to Qt constants and a custom pattern (if any)
    _linecnvt = { 'solid': (qt4.Qt.SolidLine, None),
                  'dashed': (qt4.Qt.DashLine, None),
                  'dotted': (qt4.Qt.DotLine, None),
                  'dash-dot': (qt4.Qt.DashDotLine, None),
                  'dash-dot-dot': (qt4.Qt.DashDotDotLine, None),
                  'dotted-fine': (qt4.Qt.CustomDashLine, [2, 4]),
                  'dashed-fine': (qt4.Qt.CustomDashLine, [8, 4]),
                  'dash-dot-fine': (qt4.Qt.CustomDashLine, [8, 4, 2, 4]),
                  'dot1': (qt4.Qt.CustomDashLine, [0.1, 2]),
                  'dot2': (qt4.Qt.CustomDashLine, [0.1, 4]),
                  'dot3': (qt4.Qt.CustomDashLine, [0.1, 6]),
                  'dot4': (qt4.Qt.CustomDashLine, [0.1, 8]),
                  'dash1': (qt4.Qt.CustomDashLine, [4, 4]),
                  'dash2': (qt4.Qt.CustomDashLine, [4, 8]),
                  'dash3': (qt4.Qt.CustomDashLine, [8, 8]),
                  'dash4': (qt4.Qt.CustomDashLine, [16, 8]),
                  'dash5': (qt4.Qt.CustomDashLine, [16, 16]),
                  'dashdot1': (qt4.Qt.CustomDashLine, [0.1, 4, 4, 4]),
                  'dashdot2': (qt4.Qt.CustomDashLine, [0.1, 4, 8, 4]),
                  'dashdot3': (qt4.Qt.CustomDashLine, [0.1, 2, 4, 2]),
                 }

    controls.LineStyle._lines = _linestyles

    def __init__(self, name, default, **args):
        Choice.__init__(self, name, self._linestyles, default, **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

    def qtStyle(self):
        """Get Qt ID of chosen line style."""
        return self._linecnvt[self.val]

    def makeControl(self, *args):
        return controls.LineStyle(self, *args)

class Axis(Str):
    """A setting to hold the name of an axis.

    direction is 'horizontal', 'vertical' or 'both'
    """

    typename = 'axis'

    def __init__(self, name, val, direction, **args):
        """Initialise using the document, so we can get the axes later.

        direction is horizontal or vertical to specify the type of axis to
        show
        """

        Setting.__init__(self, name, val, **args)
        self.direction = direction

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (self.direction,), {})

    def makeControl(self, *args):
        """Allows user to choose an axis or enter a name."""
        return controls.Axis(self, self.getDocument(), self.direction, *args)

class WidgetChoice(Str):
    """Hold the name of a child widget."""

    typename = 'widget-choice'

    def __init__(self, name, val, widgettypes={}, **args):
        """Choose widgets from (named) type given."""
        Setting.__init__(self, name, val, **args)
        self.widgettypes = widgettypes

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (),
                                {'widgettypes': self.widgettypes})

    def buildWidgetList(self, level, widget, outdict):
        """A recursive helper to build up a list of possible widgets.

        This iterates over widget's children, and adds widgets as tuples
        to outdict using outdict[name] = (widget, level)

        Lower level images of the same name outweigh other images further down
        the tree
        """

        for child in widget.children:
            if child.typename in self.widgettypes:
                if (child.name not in outdict) or (outdict[child.name][1]>level):
                    outdict[child.name] = (child, level)
            else:
                self.buildWidgetList(level+1, child, outdict)

    def getWidgetList(self):
        """Return a dict of valid widget names and the corresponding objects."""

        # find widget which contains setting
        widget = self.parent
        while not widget.iswidget and widget is not None:
            widget = widget.parent

        # get widget's parent
        if widget is not None:
            widget = widget.parent

        # get list of widgets from recursive find
        widgets = {}
        if widget is not None:
            self.buildWidgetList(0, widget, widgets)

        # turn (object, level) pairs into object
        outdict = {}
        for name, val in widgets.items():
            outdict[name] = val[0]

        return outdict

    def findWidget(self):
        """Find the image corresponding to this setting.

        Returns Image object if succeeds or None if fails
        """

        widgets = self.getWidgetList()
        try:
            return widgets[self.get()]
        except KeyError:
            return None

    def makeControl(self, *args):
        """Allows user to choose an image widget or enter a name."""
        return controls.WidgetChoice(self, self.getDocument(), *args)

class Marker(Choice):
    """Choose a marker type from one allowable."""

    typename = 'marker'

    def __init__(self, name, value, **args):
        Choice.__init__(self, name, utils.MarkerCodes, value, **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

    def makeControl(self, *args):
        return controls.Marker(self, *args)

class Arrow(Choice):
    """Choose an arrow type from one allowable."""

    typename = 'arrow'

    def __init__(self, name, value, **args):
        Choice.__init__(self, name, utils.ArrowCodes, value, **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

    def makeControl(self, *args):
        return controls.Arrow(self, *args)

class LineSet(Setting):
    """A setting which corresponds to a set of lines.
    """

    typename='line-multi'

    def normalize(self, val):
        """Takes a tuple/list of tuples:
        [('dotted', '1pt', 'color', <trans>, False), ...]

        These are style, width, color, and hide or
        style, widget, color, transparency, hide
        """

        if type(val) not in (list, tuple):
            raise utils.InvalidType

        # check each entry in the list is appropriate
        for line in val:
            try:
                style, width, color, hide = line
            except ValueError:
                raise utils.InvalidType

            if ( not isinstance(color, cbasestr) or
                 not Distance.isDist(width) or
                 style not in LineStyle._linestyles or
                 type(hide) not in (int, bool) ):
                raise utils.InvalidType

        return val

    def makeControl(self, *args):
        """Make specialised lineset control."""
        return controls.LineSet(self, *args)

    def makePen(self, painter, row):
        """Make a pen for the painter using row.

        If row is outside of range, then cycle
        """

        if len(self.val) == 0:
            return qt4.QPen(qt4.Qt.NoPen)
        else:
            row = row % len(self.val)
            v = self.val[row]
            style, width, color, hide = v
            width = Distance.convertDistance(painter, width)
            style, dashpattern = LineStyle._linecnvt[style]
            col = painter.docColor(color)
            pen = qt4.QPen(col, width, style)

            if dashpattern:
                pen.setDashPattern(dashpattern)

            if hide:
                pen.setStyle(qt4.Qt.NoPen)
            return pen

class FillSet(Setting):
    """A setting which corresponds to a set of fills.

    This setting keeps an internal array of LineSettings.
    """

    typename = 'fill-multi'

    def normalize(self, val):
        """Takes a tuple/list of tuples:
        [('solid', 'color', False), ...]

        These are color, fill style, and hide or
        color, fill style, and hide

        (style, color, hide,
        [optional transparency, linewidth,
         linestyle, spacing, backcolor, backtrans, backhide]])

        """

        if type(val) not in (list, tuple):
            raise utils.InvalidType

        # check each entry in the list is appropriate
        for fill in val:
            try:
                style, color, hide = fill[:3]
            except ValueError:
                raise utils.InvalidType

            if ( not isinstance(color, cbasestr) or
                 style not in utils.extfillstyles or
                 type(hide) not in (int, bool) or
                 len(fill) not in (3, 10) ):
                raise utils.InvalidType

        return val

    def makeControl(self, *args):
        """Make specialised lineset control."""
        return controls.FillSet(self, *args)

    def returnBrushExtended(self, row):
        """Return BrushExtended for the row."""
        from . import collections
        s = collections.BrushExtended('tempbrush')
        s.parent = self

        if len(self.val) == 0:
            s.hide = True
        else:
            v = self.val[row % len(self.val)]
            s.style = v[0]
            s.color = v[1]
            s.hide = v[2]
            if len(v) == 10:
                (s.transparency, s.linewidth, s.linestyle,
                 s.patternspacing, s.backcolor,
                 s.backtransparency, s.backhide) = v[3:]
        return s

class Filename(Str):
    """Represents a filename setting."""

    typename = 'filename'

    def makeControl(self, *args):
        return controls.Filename(self, 'file', *args)

    def normalize(self, val):
        if sys.platform == 'win32':
            val = val.replace('\\', '/')
        return val

class ImageFilename(Filename):
    """Represents an image filename setting."""

    typename = 'filename-image'

    def makeControl(self, *args):
        return controls.Filename(self, 'image', *args)

class FontFamily(Str):
    """Represents a font family."""

    typename = 'font-family'

    def makeControl(self, *args):
        """Make a special font combobox."""
        return controls.FontFamily(self, *args)

class ErrorStyle(Choice):
    """Error bar style.
    The allowed values are below in _errorstyles.
    """

    typename = 'errorbar-style'

    _errorstyles = (
        'none',
        'bar', 'barends', 'box', 'diamond', 'curve',
        'barbox', 'bardiamond', 'barcurve',
        'boxfill', 'diamondfill', 'curvefill',
        'fillvert', 'fillhorz',
        'linevert', 'linehorz',
        'linevertbar', 'linehorzbar'
        )

    controls.ErrorStyle._errorstyles  = _errorstyles

    def __init__(self, name, value, **args):
        Choice.__init__(self, name, self._errorstyles, value, **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

    def makeControl(self, *args):
        return controls.ErrorStyle(self, *args)

class AlignHorz(Choice):
    """Alignment horizontally."""

    typename = 'align-horz'

    def __init__(self, name, value, **args):
        Choice.__init__(self, name, ['left', 'centre', 'right'], value, **args)
    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

class AlignVert(Choice):
    """Alignment vertically."""

    typename = 'align-vert'

    def __init__(self, name, value, **args):
        Choice.__init__(self, name, ['top', 'centre', 'bottom'], value, **args)
    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

class AlignHorzWManual(Choice):
    """Alignment horizontally."""

    typename = 'align-horz-+manual'

    def __init__(self, name, value, **args):
        Choice.__init__(self, name, ['left', 'centre', 'right', 'manual'],
                        value, **args)
    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

class AlignVertWManual(Choice):
    """Alignment vertically."""

    typename = 'align-vert-+manual'

    def __init__(self, name, value, **args):
        Choice.__init__(self, name, ['top', 'centre', 'bottom', 'manual'],
                        value, **args)
    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

# Bool which shows/hides other settings
class BoolSwitch(Bool):
    """Bool switching setting."""

    def __init__(self, name, value, settingsfalse=[], settingstrue=[],
                 **args):
        """Enables/disables a set of settings if True or False
        settingsfalse and settingstrue are lists of names of settings
        which are hidden/shown to user
        """

        self.sfalse = settingsfalse
        self.strue = settingstrue
        Bool.__init__(self, name, value, **args)

    def makeControl(self, *args):
        return controls.BoolSwitch(self, *args)

    def copy(self):
        return self._copyHelper((), (), {'settingsfalse': self.sfalse,
                                         'settingstrue': self.strue})

class ChoiceSwitch(Choice):
    """Show or hide other settings based on the choice given here."""

    def __init__(self, name, vallist, value, settingstrue=[], settingsfalse=[],
                 showfn=lambda val: True, **args):
        """Enables/disables a set of settings if True or False
        settingsfalse and settingstrue are lists of names of settings
        which are hidden/shown to user depending on showfn(val)."""

        self.sfalse = settingsfalse
        self.strue = settingstrue
        self.showfn = showfn
        Choice.__init__(self, name, vallist, value, **args)

    def makeControl(self, *args):
        return controls.ChoiceSwitch(self, False, self.vallist, *args)

    def copy(self):
        return self._copyHelper((self.vallist,), (),
                                {'settingsfalse': self.sfalse,
                                 'settingstrue': self.strue,
                                 'showfn': self.showfn})

class FillStyleExtended(ChoiceSwitch):
    """A setting for the different fill styles provided by Qt."""

    typename = 'fill-style-ext'

    _strue = ( 'linewidth', 'linestyle', 'patternspacing',
               'backcolor', 'backtransparency', 'backhide' )

    @staticmethod
    def _ishatch(val):
        """Is this a hatching fill?"""
        return not ( val == 'solid' or val.find('dense') >= 0 )

    def __init__(self, name, value, **args):
        ChoiceSwitch.__init__(self, name, utils.extfillstyles, value,
                              settingstrue=self._strue, settingsfalse=(),
                              showfn=self._ishatch,
                              **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

    def makeControl(self, *args):
        return controls.FillStyleExtended(self, *args)

class RotateInterval(Choice):
    '''Rotate a label with intervals given.'''

    def __init__(self, name, val, **args):
        Choice.__init__(self, name,
                        ('-180', '-135', '-90', '-45',
                         '0', '45', '90', '135', '180'),
                        val, **args)

    def normalize(self, val):
        """Store rotate angle."""
        # backward compatibility with rotate option
        # False: angle 0
        # True:  angle 90
        if val == False:
            val = '0'
        elif val == True:
            val = '90'
        return Choice.normalize(self, val)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

class Colormap(Str):
    """A setting to set the color map used in an image.
    This is based on a Str rather than Choice as the list might
    change later.
    """

    def makeControl(self, *args):
        return controls.Colormap(self, self.getDocument(), *args)

class AxisBound(FloatOrAuto):
    """Axis bound - either numeric, Auto or date."""

    typename = 'axis-bound'

    def makeControl(self, *args):
        return controls.AxisBound(self, *args)

    def toUIText(self):
        """Convert to text, taking into account mode of Axis.
        Displays datetimes in date format if used
        """

        try:
            mode = self.parent.mode
        except AttributeError:
            mode = None

        v = self.val
        if ( not isinstance(v, cbasestr) and v is not None and
             mode == 'datetime' ):
            return utils.dateFloatToString(v)

        return FloatOrAuto.toUIText(self)

    def fromUIText(self, txt):
        """Convert from text, allowing datetimes."""

        v = utils.dateStringToDate(txt)
        if N.isfinite(v):
            return v
        else:
            return FloatOrAuto.fromUIText(self, txt)
