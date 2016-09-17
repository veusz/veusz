#    Copyright (C) 2016 Jeremy S. Sanders
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

"""
Base class for setting values in Veusz document
"""

from __future__ import division

import numpy as N

from ..compat import crepr
from .. import qtall as qt4
from .reference import ReferenceBase, Reference

from .. import utils

class OnModified(qt4.QObject):
    """onmodified is emitted from an object contained in each setting."""
    onModified = qt4.pyqtSignal()

class Setting(object):
    """A class to store a value with a particular type."""

    # differentiate widgets, settings and setting
    nodetype = 'setting'

    typename = 'setting'

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

        if isinstance(self._val, ReferenceBase):
            val = self._val
        else:
            val = self.val

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

        if isinstance(self._val, ReferenceBase):
            return self._val.resolve(self).get()
        else:
            return self.convertFrom(self._val)

    def set(self, v):
        """Set the value."""

        if isinstance(v, ReferenceBase):
            self._val = v
        else:
            # this also removes the linked value if there is one set
            self._val = self.convertTo(v)

        self.onmodified.onModified.emit()

    val = property(get, set, None,
                   'Get or modify the value of the setting')

    def isReference(self):
        """Is this a setting a reference to another object."""
        return isinstance(self._val, ReferenceBase)

    def getReference(self):
        """Return the reference object. Raise ValueError if not a reference"""
        if isinstance(self._val, ReferenceBase):
            return self._val
        else:
            raise ValueError("Setting is not a reference")

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
                if obj.name == '/':
                    path.insert(0, '')
                else:
                    path.insert(0, obj.name)
            obj = obj.parent
        return '/'.join(path)

    path = property(_path, None, None,
                    'Return the full path of the setting')

    def toTextUI(self):
        """Convert the type to text for editing in UI."""
        return ""

    def fromTextUI(self, text):
        """Convert text from UI into type for saving.

        Raises utils.InvalidType if cannot convert."""
        return None

    def saveText(self, saveall, rootname = ''):
        """Return text to restore the value of this setting."""

        if (saveall or not self.isDefault()) and not self.readonly:
            if isinstance(self._val, ReferenceBase):
                return "SetToReference('%s%s', %s)\n" % (
                    rootname, self.name, crepr(self._val.value))
            else:
                return "Set('%s%s', %s)\n" % (
                    rootname, self.name, crepr(self.val) )
        else:
            return ''

    def setOnModified(self, fn):
        """Set the function to be called on modification (passing True)."""
        self.onmodified.onModified.connect(fn)

        if isinstance(self._val, ReferenceBase):
            # tell references to notify us if they are modified
            self._val.setOnModified(self, fn)

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
        if (isinstance(self._val, ReferenceBase) and
            isinstance(self.default, ReferenceBase) ):
            return self._val.value == self.default.value
        else:
            return self.val == self.default

    def isDefaultLink(self):
        """Is this a link to the default stylesheet value."""

        return (
            isinstance(self._val, ReferenceBase) and
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
            try:
                return p.getDocument()
            except AttributeError:
                pass
            p = p.parent
        return None

    def getWidget(self):
        """Return associated widget."""
        w = self.parent
        while not w.isWidget():
            w = w.parent
        return w

# forward setting to another setting
class SettingBackwardCompat(Setting):
    """Forward setting requests to another setting.

    This is used for backward-compatibility.
    """

    typename = 'backward-compat'

    def __init__(self, name, newrelpath, val, translatefn = None,
                 **args):
        """Point this setting to another.
        newrelpath is a path relative to this setting's parent
        """

        self.translatefn = translatefn
        Setting.__init__(self, name, val, **args)
        self.relpath = newrelpath.split('/')

    def getForward(self):
        """Get setting this setting forwards to."""
        return self.parent.getFromPath(self.relpath)

    def convertTo(self, val):
        if self.parent is not None:
            return self.getForward().convertTo(val)

    def toTextUI(self):
        return self.getForward().toTextUI()

    def fromTextUI(self, val):
        return self.getForward().fromTextUI(val)

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
            ('/'.join(self.relpath),), (),
            {'translatefn': self.translatefn})

    def makeControl(self, *args):
        return None

    def saveText(self, saveall, rootname = ''):
        return ''

    def linkToStylesheet(self):
        """Do nothing for backward compatibility settings."""
        pass
