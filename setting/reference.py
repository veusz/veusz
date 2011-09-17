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

class Reference(object):
    """A value a setting can have to point to another setting.
    
    Formats of a reference are like
    /foo/bar/setting or
    ../Line/width
    
    alternatively style sheets can be used with the format, e.g.
    /StyleSheet/linewidth
    """
    
    class ResolveException(Exception):
        pass

    def __init__(self, value):
        """Initialise reference with value, which is a string as above."""
        self.value = value

        # assigned once reference has been resolved
        self.resolved = None

        # this is assigned by the setting using this reference
        self.parent = None

    def _doresolution(self, thissetting):
        """Return the setting object associated with the reference."""

        item = thissetting.parent
        parts = list(self.value.split('/'))
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
                if item.isWidget():
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

        return item

    def resolve(self, thissetting):
        """Just return the resolved setting."""

        if self.resolved is None:
            self.resolved = self._doresolution(thissetting)
            self.resolved.setOnModified(self._onmodified)

        return self.resolved

    def _onmodified(self):
        """Tell parent if a reference is modified."""
        if self.parent:
            self.parent.modified()
