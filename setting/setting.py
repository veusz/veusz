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

# $Id$

"""Module for holding setting values.

e.g.

s = Int('foo', 5)
s.get()
s.set(42)
s.fromText('42')
"""

import re
import math
import sys

import numpy as N
import veusz.qtall as qt4

import controls
from settingdb import settingdb
from reference import Reference

import veusz.utils as utils

# if invalid type passed to set
class InvalidType(Exception):
    pass

class Setting(object):
    def __init__(self, name, value, descr='', usertext='',
                 formatting=False, hidden=False):
        """Initialise the values.

        name: setting name
        value: default value and initial value
        descr:  description of the setting
        usertext: name of setting for user
        formatting: whether setting applies to formatting
        """
        self.readonly = False
        self.parent = None
        self.name = name
        self.descr = descr
        self.usertext = usertext
        self.formatting = formatting
        self.default = value
        self.onmodified = qt4.QObject()
        self._val = None

        # calls the set function for the val property
        self.val = value

    def isWidget(self):
        """Is this object a widget?"""
        return False

    def _copyHelper(self, before, after, optional):
        """Help copy an object.

        before are arguments before val
        after are arguments after val
        optinal as optional arguments
        """

        if isinstance(self._val, Reference):
            val = self._val
        else:
            val = self.val

        args = (self.name,) + before + (val,) + after
        opt = optional.copy()
        opt['descr'] = self.descr
        opt['usertext'] = self.usertext
        opt['formatting'] = self.formatting
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
        
        if isinstance(self._val, Reference):
            return self._val.resolve(self).get()
        else:
            return self.convertFrom(self._val)

    def set(self, v):
        """Set the value."""

        if isinstance(v, Reference):
            self._val = v
        else:
            # this also removes the linked value if there is one set
            self._val = self.convertTo(v)

        self.onmodified.emit(qt4.SIGNAL("onModified"), True)

    val = property(get, set, None,
                   'Get or modify the value of the setting')

    def isReference(self):
        """Is this a setting a reference to another object."""
        return isinstance(self._val, Reference)

    def getReference(self):
        """Return the reference object."""
        assert isinstance(self._val, Reference)
        return self._val

    def getStylesheetLink(self):
        """Get text that this setting should default to linked to the
        stylesheet."""
        path = []
        obj = self
        while not obj.parent.isWidget():
            path.insert(0, obj.name)
            obj = obj.parent
        path = ['', 'StyleSheet', obj.parent.typename] + path
        return '/'.join(path)

    def linkToStylesheet(self):
        """Make this setting link to stylesheet setting, if possible."""
        self.set( Reference(self.getStylesheetLink()) )
               
    def _path(self):
        """Return full path of setting."""
        path = []
        obj = self
        while obj is not None:
            # logic easier to understand here
            # do not add settings name for settings of widget
            if not obj.isWidget() and obj.parent.isWidget():
                pass
            else:
                path.insert(0, obj.name)
            obj = obj.parent
        return '/'.join(path)
        
    path = property(_path, None, None,
                    'Return the full path of the setting')
    
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
    
        if deftext is not None:
            self.val = self.fromText(deftext)
            self.default = self.val

    def removeDefault(self):
        """Remove the default setting for this setting."""

        # build up setting path
        path = ''
        item = self
        while not item.isWidget():
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
        while not item.isWidget():
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
            if isinstance(self._val, Reference):
                return "SetToReference('%s%s', %s)\n" % (rootname, self.name,
                                                         repr(self._val.value))
            else:
                return "Set('%s%s', %s)\n" % ( rootname, self.name,
                                               repr(self.val) )
        else:
            return ''

    def setOnModified(self, fn):
        """Set the function to be called on modification (passing True)."""
        self.onmodified.connect(self.onmodified,
                                qt4.SIGNAL("onModified"), fn)

        if isinstance(self._val, Reference):
            # make reference pointed to also call this onModified
            r = self._val.resolve(self)
            r.setOnModified(fn)

    def removeOnModified(self, fn):
        """Remove the function from the list of function to be called."""
        self.onmodified.disconnect(self.onmodified, 0, fn, 0)

    def newDefault(self, value):
        """Update the default and the value."""
        self.default = value
        self.val = value

    def isDefault(self):
        """Is the current value a default?
        This also returns true if it is linked to the appropriate stylesheet
        """
        if ( isinstance(self._val, Reference) and
             isinstance(self.default, Reference) ):
            return self._val.value == self.default.value
        else:
            return self.val == self.default

    def isDefaultLink(self):
        """Is this a link to the default spreadsheet value."""

        return ( isinstance(self._val, Reference) and
                 self._val.value == self.getStylesheetLink() )

    def setSilent(self, val):
        """Set the setting, without propagating modified flags.

        This shouldn't often be used as it defeats the automatic updation.
        Used for temporary modifications."""

        self._val = self.convertTo(val)

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

    def getDocument(self):
        """Return document."""
        p = self.parent
        while p:
            if p.isWidget():
                return p.document
            p = p.parent
        return None

    def getWidget(self):
        """Return associated widget."""
        w = self.parent
        while not w.isWidget():
            w = w.parent
        return w

# Store strings
class Str(Setting):
    """String setting."""

    def convertTo(self, val):
        if isinstance(val, basestring):
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
        
    def convertTo(self, val):
        if isinstance(val, int):
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

_safe_environ = utils.veusz_eval_context.copy()
def safeEvalHelper(text):
    """Evaluate an expression, catching naughtiness."""
    if utils.checkCode(text) is not None:
        raise InvalidType
    try:
        return float( eval(text, _safe_environ) )
    except:
        raise InvalidType

# for storing floats
class Float(Setting):
    """Float settings."""

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

    def convertTo(self, val):       
        if isinstance(val, int) or isinstance(val, float):
            if val >= self.minval and val <= self.maxval:
                return float(val)
            else:
                raise InvalidType, 'Out of range allowed'
        raise InvalidType

    def toText(self):
        return str(self.val)

    def fromText(self, text):
        try:
            f = float(text)
        except ValueError:
            # try to evaluate
            f = safeEvalHelper(text)
        return self.convertTo(f)

    def makeControl(self, *args):
        return controls.Edit(self, *args)

class FloatOrAuto(Setting):
    """Save a float or text auto."""

    def convertTo(self, val):
        if type(val) in (int, float):
            return float(val)
        elif isinstance(val, basestring) and val.strip().lower() == 'auto':
            return None
        else:
            raise InvalidType

    def convertFrom(self, val):
        if val is None:
            return 'Auto'
        else:
            return val

    def toText(self):
        if self.val is None:
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
                # try to evaluate
                return safeEvalHelper(text)

    def makeControl(self, *args):
        return controls.Choice(self, True, ['Auto'], *args)
            
class IntOrAuto(Setting):
    """Save an int or text auto."""

    def convertTo(self, val):
        if isinstance(val, int):
            return val
        elif isinstance(val, basestring) and val.strip().lower() == 'auto':
            return None
        else:
            raise InvalidType

    def convertFrom(self, val):
        if val is None:
            return 'Auto'
        else:
            return val

    def toText(self):
        if self.val is None:
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

    painter.veusz_pixperpt = painter.device().logicalDpiY() / 72.

def _distPhys(match, painter, mult):
    """Convert a physical unit measure in multiples of points."""

    if not hasattr(painter, 'veusz_pixperpt'):
        _calcPixPerPt(painter)

    return (painter.veusz_pixperpt * mult *
            float(match.group(1)) * painter.veusz_scaling)

def _distInvPhys(pixdist, painter, mult, unit):
    """Convert number of pixels into physical distance."""
    dist = pixdist / (mult * painter.veusz_pixperpt *
                      painter.veusz_scaling)
    return "%.3g%s" % (dist, unit)

def _distPerc(match, painter, maxsize):
    """Convert from a percentage of maxsize."""
    return maxsize * 0.01 * float(match.group(1))

def _distInvPerc(pixdist, painter, maxsize):
    """Convert pixel distance into percentage."""
    perc = pixdist * 100. / maxsize
    return "%.3g%%" % perc

def _distFrac(match, painter, maxsize):
    """Convert from a fraction a/b of maxsize."""
    return maxsize * float(match.group(1)) / float(match.group(2))

def _distRatio(match, painter, maxsize):
    """Convert from a simple 0.xx ratio of maxsize."""

    # if it's greater than 1 then assume it's a point measurement
    if float(match.group(1)) > 1.:
        return _distPhys(match, painter, 1)

    return maxsize * float(match.group(1))

class Distance(Setting):
    """A veusz distance measure, e.g. 1pt or 3%."""

    # mappings from regular expressions to function to convert distance
    # the recipient function takes regexp match,
    # painter and maximum size of frac

    # the second function is to do the inverse calculation
    distregexp = [
        # cm distance
        ( re.compile('^([0-9\.]+) *cm$'),
          lambda match, painter, t:
              _distPhys(match, painter, 28.452756),
          lambda pixdist, painter, t:
              _distInvPhys(pixdist, painter, 28.452756, 'cm') ),

        # point size
        ( re.compile('^([0-9\.]+) *pt$'),
          lambda match, painter, t:
              _distPhys(match, painter, 1.),
          lambda pixdist, painter, t:
              _distInvPhys(pixdist, painter, 1., 'pt') ),

        # mm distance
        ( re.compile('^([0-9\.]+) *mm$'),
          lambda match, painter, t:
              _distPhys(match, painter, 2.8452756),
          lambda pixdist, painter, t:
              _distInvPhys(pixdist, painter, 2.8452756, 'mm') ),

        # inch distance
        ( re.compile('^([0-9\.]+) *(inch|in|")$'),
          lambda match, painter, t:
              _distPhys(match, painter, 72.27),
          lambda pixdist, painter, t:
              _distInvPhys(pixdist, painter, 72.27, 'in') ),

        # plain fraction
        ( re.compile('^([0-9\.]+)$'),
          _distRatio,
          _distInvPerc ),

        # percentage
        ( re.compile('^([0-9\.]+) *%$'),
          _distPerc,
          _distInvPerc ),

        # fractional
        ( re.compile('^([0-9\.]+) */ *([0-9\.]+)$'),
          _distFrac,
          _distInvPerc ),
        ]

    @classmethod
    def isDist(kls, dist):
        """Is the text a valid distance measure?"""
        
        dist = dist.strip()
        for reg, fn, fninv in kls.distregexp:
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

    @classmethod
    def convertDistance(kls, painter, distance):
        '''Convert a distance to plotter units.

        dist: eg 0.1 (fraction), 10% (percentage), 1/10 (fraction),
                 10pt, 1cm, 20mm, 1inch, 1in, 1" (size)
        maxsize: size fractions are relative to
        painter: painter to get metrics to convert physical sizes
        '''

        # we set a scaling variable in the painter if it's not set
        if not hasattr(painter, 'veusz_scaling'):
            painter.veusz_scaling = 1.

        # work out maximum size
        try:
            maxsize = max( *painter.veusz_page_size )
        except AttributeError:
            w = painter.window()
            maxsize = max(w.width(), w.height())

        dist = distance.strip()

        # compare string against each regexp
        for reg, fn, fninv in kls.distregexp:
            m = reg.match(dist)

            # if there's a match, then call the appropriate conversion fn
            if m:
                return fn(m, painter, maxsize)

        # none of the regexps match
        raise ValueError( "Cannot convert distance in form '%s'" %
                          dist )

    def convert(self, painter):
        """Convert this setting's distance as above"""
        
        return self.convertDistance(painter, self.val)

    def convertPts(self, painter):
        """Get the distance in points."""
        if not hasattr(painter, 'veusz_pixperpt'):
            _calcPixPerPt(painter)

        return self.convert(painter) / painter.veusz_pixperpt
        
    def convertInverse(self, distpix, painter):
        """Convert distance in pixels into units of this distance.

        Not that great coding as takes "painter" containing veusz
        scaling parameters. Should be cleaned up.
        """

        # identify units and get inverse mapping
        v = self.val
        inversefn = None
        for reg, fn, fninv in self.distregexp:
            if reg.match(v):
                inversefn = fninv
                break
        if not inversefn:
            inversefn = self.distregexp[0][2]

        maxsize = max( *painter.veusz_page_size )

        # do inverse mapping
        return inversefn(distpix, painter, maxsize)

class DistanceOrAuto(Distance):
    """A distance or the value Auto"""

    distregexp = Distance.distregexp + [(re.compile('^Auto$'), None, None)]
    
    def isAuto(self):
        return self.val == 'Auto'

    def makeControl(self, *args):
        return controls.Distance(self, allowauto=True, *args)

class Choice(Setting):
    """One out of a list of strings."""

    # maybe should be implemented as a dict to speed up checks

    def __init__(self, name, vallist, val, **args):
        """Setting val must be in vallist."""
        
        assert type(vallist) in (list, tuple)
        self.vallist = vallist
        Setting.__init__(self, name, val, **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((self.vallist,), (), {})
        
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

    def __init__(self, name, vallist, val, **args):
        """Setting has val must be in vallist."""
        
        self.vallist = vallist
        Setting.__init__(self, name, val, **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((self.vallist,), (), {})

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
        keys = self.val.keys()
        keys.sort()
        
        text = ['%s = %g' % (key, self.val[key]) for key in keys]
        return '\n'.join(text)

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

    list_re = re.compile(r'[\t\n, ]+')

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

        out = []
        for x in self.list_re.split(text.strip()):
            if x:
                try:
                    out.append(float(x))
                except ValueError:
                    out.append( safeEvalHelper(x) )
        return out

    def makeControl(self, *args):
        return controls.String(self, *args)

class WidgetPath(Str):
    """A setting holding a path to a widget. This is checked for validity."""

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

    def convertTo(self, val):
        """Validate the text is a name of a widget relative to
        this one."""

        if not isinstance(val, basestring):
            raise InvalidType

        # InvalidType will get raised in getWidget if it is incorrect
        w = self.getWidget(val)
        if w is None:
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
        if val is None:
            val = self.val

        if val == '':
            return None

        # find the widget associated with this setting
        widget = self
        while not widget.isWidget():
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
        if self.allowedwidgets is not None:
            allowed = False
            for c in self.allowedwidgets:
                if isinstance(widget, c):
                    allowed = True
            if not allowed:
                raise InvalidType
        
        return widget

class Dataset(Str):
    """A setting to choose from the possible datasets."""

    def __init__(self, name, val, dimensions=1, datatype='numeric',
                 **args):
        """
        dimensions is the number of dimensions the dataset needs
        """

        Setting.__init__(self, name, val, **args)
        self.dimensions = dimensions
        self.datatype = datatype

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
             d.dimensions == self.dimensions ):
                 return d

class Strings(Setting):
    """A multiple set of strings."""

    def convertTo(self, val):
        """Takes a tuple/list of strings:
        ('ds1','ds2'...)
        """

        if isinstance(val, basestring):
            return (val, )

        if type(val) not in (list, tuple):
            raise InvalidType

        # check each entry in the list is appropriate
        for ds in val:
            if not isinstance(ds, basestring):
                raise InvalidType

        return tuple(val)
        
    def makeControl(self, *args):
        """Allow user to choose between the datasets."""
        return controls.Strings(self, self.getDocument(), *args)

class Datasets(Setting):
    """A setting to choose one or more of the possible datasets."""

    def __init__(self, name, val, dimensions=1, datatype='numeric',
                 **args):
        """
        dimensions is the number of dimensions the dataset needs
        """

        Setting.__init__(self, name, val, **args)
        self.dimensions = dimensions
        self.datatype = datatype

    def convertTo(self, val):
        """Takes a tuple/list of strings:
        ('ds1','ds2'...)
        """

        if isinstance(val, basestring):
            return (val, )

        if type(val) not in (list, tuple):
            raise InvalidType

        # check each entry in the list is appropriate
        for ds in val:
            if not isinstance(ds, basestring):
                raise InvalidType

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

class DatasetOrFloatList(Dataset):
    """Choose a dataset or specify a list of float values."""

    # a list of numbers separated by spaces or tabs
    # (requires number at end of line)
    numbers_re = re.compile(r'^([-+]?[0-9]*\.?[0-9]*([eE][-+]?[0-9]+)?[ \t,$]+)+$')

    def convertTo(self, val):
        """Check is a string (dataset name) or a list of floats (numbers).

        """
        
        if isinstance(val, basestring):
            return val
        elif isinstance(val, float) or isinstance(val, int):
            return [val]
        else:
            try:
                return [float(x) for x in val]
            except (TypeError, ValueError):
                raise InvalidType

    def toText(self):
        if isinstance(self.val, basestring):
            return self.val
        else:
            return ', '.join( [str(x) for x in self.val] )

    def fromText(self, text):
        text = text.strip()
        try:
            out = []
            for x in FloatList.list_re.split(text):
                try:
                    out.append(float(x))
                except ValueError:
                    out.append( safeEvalHelper(x) )
            return out
        except InvalidType:
            return text

    def getFloatArray(self, doc):
        """Get a numpy of values or None."""
        if isinstance(self.val, basestring):
            ds = doc.data.get(self.val)
            if ds:
                # get numpy of values
                return ds.data
            else:
                return None
        else:
            # list of values
            return N.array(self.val)
            
    def isDataset(self, doc):
        """Is this setting a dataset?"""
        return (isinstance(self.val, basestring) and
                doc.data.get(self.val))

    def getData(self, doc):
        """Return veusz dataset"""
        if isinstance(self.val, basestring):
            d = doc.data.get(self.val)
            if ( d is not None and
                 d.datatype == self.datatype and
                 d.dimensions == self.dimensions ):
                return d
            return None
        else:
            # blah - need to import here due to dependencies
            import veusz.document as document
            return document.Dataset(data=self.val)
    
class DatasetOrStr(Dataset):
    """Choose a dataset or enter a string."""

    def getData(self, doc, checknull=False):
        """Return either a list of strings, a single item list.
        If checknull then None is returned if blank
        """
        if doc:
            ds = doc.data.get(self.val)
            if ds and ds.datatype == self.datatype:
                return ds.data
        if checknull and not self.val:
            return None
        else:
            return [unicode(self.val)]

    def makeControl(self, *args):
        # use string editor rather than drop down list
        # need to write a custom control
        return controls.DatasetOrString(self, self.getDocument(), self.dimensions,
                                        self.datatype, *args)
        #return controls.String(self, *args)

class Color(ChoiceOrMore):
    """A color setting."""

    _colors = [ 'white', 'black', 'red', 'green', 'blue',
                'cyan', 'magenta', 'yellow',
                'grey', 'darkred', 'darkgreen', 'darkblue',
                'darkcyan', 'darkmagenta' ]
    
    controls.Color._colors = _colors

    def __init__(self, name, value, **args):
        """Initialise the color setting with the given name, default
        and description."""
        
        ChoiceOrMore.__init__(self, name, self._colors, value,
                              **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})
                              
    def color(self):
        """Return QColor for color."""
        return qt4.QColor(self.val)
    
    def makeControl(self, *args):
        return controls.Color(self, *args)

class FillStyle(Choice):
    """A setting for the different fill styles provided by Qt."""
    
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

    # list of allowed line styles
    _linestyles = ['solid', 'dashed', 'dotted',
                   'dash-dot', 'dash-dot-dot', 'dotted-fine',
                   'dashed-fine', 'dash-dot-fine']

    # convert from line styles to Qt constants and a custom pattern (if any)
    _linecnvt = { 'solid': (qt4.Qt.SolidLine, None),
                  'dashed': (qt4.Qt.DashLine, None),
                  'dotted': (qt4.Qt.DotLine, None),
                  'dash-dot': (qt4.Qt.DashDotLine, None),
                  'dash-dot-dot': (qt4.Qt.DashDotDotLine, None),
                  'dotted-fine': (qt4.Qt.CustomDashLine, [2, 4]),
                  'dashed-fine': (qt4.Qt.CustomDashLine, [8, 4]),
                  'dash-dot-fine': (qt4.Qt.CustomDashLine, [8, 4, 2, 4])}
    
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
    """A setting to hold the name of an axis."""

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

class Image(Str):
    """Hold the name of a child image."""

    @staticmethod
    def buildImageList(level, widget, outdict):
        """A recursive helper to build up a list of possible image widgets.

        This iterates over widget's children, and adds Image widgets as tuples
        to outdict using outdict[name] = (widget, level)

        Lower level images of the same name outweigh other images further down
        the tree
        """

        for child in widget.children:
            if child.typename == 'image':
                if (child.name not in outdict) or (outdict[child.name][1]>level):
                    outdict[child.name] = (child, level)
            else:
                Image.buildImageList(level+1, child, outdict)

    def getImageList(self):
        """Return a dict of valid image names and the corresponding objects."""

        # find widget which contains setting
        widget = self.parent
        while not widget.isWidget() and widget is not None:
            widget = widget.parent

        # get widget's parent
        if widget is not None:
            widget = widget.parent

        # get list of images from recursive find
        images = {}
        if widget is not None:
            Image.buildImageList(0, widget, images)

        # turn (object, level) pairs into object
        outdict = {}
        for name, val in images.iteritems():
            outdict[name] = val[0]

        return outdict

    def findImage(self):
        """Find the image corresponding to this setting.

        Returns Image object if succeeds or None if fails
        """

        images = self.getImageList()
        try:
            return images[self.get()]
        except KeyError:
            return None

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

    def makeControl(self, *args):
        """Allows user to choose an image widget or enter a name."""
        return controls.Image(self, self.getDocument(), *args)
    
class Marker(Choice):
    """Choose a marker type from one allowable."""

    def __init__(self, name, value, **args):
        Choice.__init__(self, name, utils.MarkerCodes, value, **args)

    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})
                              
    def makeControl(self, *args):
        return controls.Marker(self, *args)
    
class Arrow(Choice):
    """Choose an arrow type from one allowable."""

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

    def convertTo(self, val):
        """Takes a tuple/list of tuples:
        [('dotted', '1pt', 'color', False), ...]

        These are style, width, color, and hide.
        """

        if type(val) not in (list, tuple):
            raise InvalidType

        # check each entry in the list is appropriate
        for line in val:
            try:
                style, width, color, hide = line
            except ValueError:
                raise InvalidType

            if ( not isinstance(color, basestring) or
                 not Distance.isDist(width) or
                 style not in LineStyle._linestyles or
                 type(hide) not in (int, bool) ):
                raise InvalidType

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
            style, width, color, hide = self.val[row]
            width = Distance.convertDistance(painter, width)
            style, dashpattern = LineStyle._linecnvt[style]
            pen = qt4.QPen(qt4.QColor(color), width, style)

            if dashpattern:
                pen.setDashPattern(dashpattern)

            if hide:
                pen.setStyle(qt4.Qt.NoPen)
            return pen
    
class FillSet(Setting):
    """A setting which corresponds to a set of fills.

    This setting keeps an internal array of LineSettings.
    """

    def convertTo(self, val):
        """Takes a tuple/list of tuples:
        [('solid', 'color', False), ...]

        These are color, fill style, and hide.
        """

        if type(val) not in (list, tuple):
            raise InvalidType

        # check each entry in the list is appropriate
        for fill in val:
            try:
                style, color, hide = fill
            except ValueError:
                raise InvalidType

            if ( not isinstance(color, basestring) or
                 style not in FillStyle._fillstyles or
                 type(hide) not in (int, bool) ):
                raise InvalidType

        return val
    
    def makeControl(self, *args):
        """Make specialised lineset control."""
        return controls.FillSet(self, *args)
    
    def makeBrush(self, row):
        """Make a Qt brush corresponding to the row given.

        If row is outside of range, then cycle
        """

        if len(self.val) == 0:
            return qt4.QBrush()
        else:
            row = row % len(self.val)
            style, color, hide = self.val[row]
            b = qt4.QBrush( qt4.QColor(color),
                            FillStyle._fillcnvt[style] )
            if hide:
                b.setStyle(qt4.Qt.NoBrush)
            return b
    
class Filename(Str):
    """Represents a filename setting."""

    def makeControl(self, *args):
        """Make specialised lineset control."""
        return controls.Filename(self, *args)
    
class FontFamily(Str):
    """Represents a font family."""

    def makeControl(self, *args):
        """Make a special font combobox."""
        return controls.FontFamily(self, *args)

class ErrorStyle(Choice):
    """Error bar style.
    The allowed values are below in _errorstyles.
    """

    _errorstyles = (
        'none',
        'bar', 'barends', 'box', 'diamond', 'curve',
        'barbox', 'bardiamond', 'barcurve',
        'fillvert', 'fillhorz',
        'linevert', 'linehorz',
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
    def __init__(self, name, value, **args):
        Choice.__init__(self, name, ['left', 'centre', 'right'], value, **args)
    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

class AlignVert(Choice):
    """Alignment vertically."""
    def __init__(self, name, value, **args):
        Choice.__init__(self, name, ['top', 'centre', 'bottom'], value, **args)
    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

class AlignHorzWManual(Choice):
    """Alignment horizontally."""
    def __init__(self, name, value, **args):
        Choice.__init__(self, name, ['left', 'centre', 'right', 'manual'],
                        value, **args)
    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})

class AlignVertWManual(Choice):
    """Alignment vertically."""
    def __init__(self, name, value, **args):
        Choice.__init__(self, name, ['top', 'centre', 'bottom', 'manual'],
                        value, **args)
    def copy(self):
        """Make a copy of the setting."""
        return self._copyHelper((), (), {})
