#    Copyright (C) 2006 Jeremy S. Sanders
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

"""Handles default values of settings."""

import settings

class StyleSheet(settings.Settings):
    """A class for handling default values of settings.
    
    Settings are registered to be added to the stylesheet."""

    registeredsettings = []
    settingpixmaps = {}
    
    def register(kls, settingtype, setting):
        """Register the setting so that the document can access it.
        settingtype is None for global settings
        """
        kls.registeredsettings.append( (settingtype, setting) )
    register = classmethod(register)
    
    def setPixmap(kls, settingtype, pixmap):
        """Set the pixmap for the settingtype."""
        kls.settingpixmaps[settingtype] = pixmap
    setPixmap = classmethod(setPixmap)
    
    def __init__(self):
        """Create the default settings."""
        settings.Settings.__init__(self, 'StyleSheet', 'global styles')
        self.pixmap = 'stylesheet'
        
        # make copies of all the registered settings
        for settingtype, setting in self.registeredsettings:
            # add subsetting if not there
            if settingtype and (settingtype not in self.setnames):
                s = settings.Settings(settingtype)
                # set pixmap if one is set
                try:
                    s.pixmap = self.settingpixmaps[settingtype]
                except KeyError:
                    pass
                self.add(s)
            
            # work out subsetting
            if settingtype:
                s = self.setdict[settingtype]
            else:
                s = self
            
            # make a copy of the setting and add it to the subsetting
            s.add( setting.copy() )
            
