#    Copyright (C) 2009 Jeremy S. Sanders
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

from __future__ import division

class ReferenceBase(object):
    """Reference objects are inherited from this base class.

    They should have a "value" property.
    """

    class ResolveException(ValueError):
        pass

    def __init__(self, value):
        self.value = value

    def getPaths(self):
        """Return list of paths linked by reference."""

    def resolve(self, setn):
        """Return setting this is linked to."""

    def setOnModified(self, setn, fn):
        """Set on modified on settings pointed to by this reference."""

class Reference(ReferenceBase):
    """A value a setting can have to point to another setting.

    Formats of a reference are like
    /foo/bar/setting or
    ../Line/width

    alternatively style sheets can be used with the format, e.g.
    /StyleSheet/linewidth
    """

    def __init__(self, value):
        """Initialise reference with value, which is a string as above."""
        ReferenceBase.__init__(self, value)
        self.split = value.split('/')
        self.resolved = None

    def getPaths(self):
        """Path linked by setting."""
        return [self.value]

    def resolve(self, thissetting):
        """Return the setting object associated with the reference."""

        # this is for stylesheet references which don't move
        if self.resolved:
            return self.resolved

        item = thissetting.parent
        parts = list(self.split)
        if parts[0] == '':
            # need root widget if begins with slash
            while item.parent is not None:
                item = item.parent
            parts = parts[1:]

        # do an iterative lookup of the setting
        for p in parts:
            if p == '..':
                if item.parent is not None:
                    item = item.parent
            elif p == '':
                pass
            else:
                if item.iswidget:
                    child = item.getChild(p)
                    if not child:
                        try:
                            item = item.settings.get(p)
                        except KeyError:
                            raise self.ResolveException()
                    else:
                        item = child
                else:
                    try:
                        item = item.get(p)
                    except KeyError:
                        raise self.ResolveException()

        # shortcut to resolve stylesheets
        # hopefully this won't ever change
        if len(self.split) > 2 and self.split[1] == 'StyleSheet':
            self.resolved = item

        return item

    def setOnModified(self, setn, fn):
        """Set on modified on settings pointed to by this reference."""
        resolved = self.resolve(setn)
        resolved.setOnModified(fn)

class ReferenceMultiple(ReferenceBase):
    """A reference to more than one item.

    This allows references to override other references. If one is not
    the default value, this overrides the others. References to the
    right of the list override those on the left.
    """

    def __init__(self, paths):
        """Initialise with a list of paths."""
        ReferenceBase.__init__(self, paths)
        self.refs = [Reference(p) for p in paths]

    def getPaths(self):
        """List of paths linked by setting."""
        return self.value

    def resolve(self, thissetting):
        """Resolve to setting.

        We prefer destination settings which are:
         - To the right of the list of paths
         - Which are closer in terms of the number of reference jumps

        Hopefully this algorithm isn't too slow...
        """

        retn = None
        minjumps = 99999
        for ref in self.refs:
            try:
                setn = ref.resolve(thissetting)
                jumps = 1
                while isinstance(setn._val, ReferenceBase):
                    setn = setn._val.resolve(setn)
                    jumps += 1

                if retn is None:
                    retn = setn
                    minjumps = jumps
                else:
                    if jumps <= minjumps and not setn.isDefault():
                        retn = setn
                        minjumps = jumps
            except self.ResolveException:
                pass

        if retn is None:
            raise self.ResolveException("Not linked to any settings")

        return retn

    def setOnModified(self, setn, fn):
        """Set on modified on settings pointed to by this reference."""
        for ref in self.refs:
            resolved = ref.resolve(setn)
            resolved.setOnModified(fn)
