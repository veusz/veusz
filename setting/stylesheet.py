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

"""Handles default values of settings for widgets."""

import settings

class StyleSheet(settings.Settings):
    """A class for handling default values of settings.
    
    Settings are registered to be added to the stylesheet."""

    registeredsettings = []
    
    def register(kls, widgettype, setting):
        """Register the setting so that the document can access it.
        widgettype is None for global settings
        """
        kls.registeredsettings.append( (widgettype, setting) )
    register = classmethod(register)
    
    def __init__(self, document):
        """Create the default settings."""
        settings.Settings.__init__(self, '/', 'style sheet')
        self.document = document
        
        # make copies of all the registered settings
        for widgettype, setting in self.registeredsettings:
            # add subsetting if not there
            if widgettype and (widgettype not in self.setnames):
                s = settings.Settings(widgettype)
                self.add(s)
            
            # work out subsetting
            if widgettype:
                s = self.setdict[widgettype]
            else:
                s = self
            
            # make a copy of the setting and add it to the subsetting
            s.add( setting.copy() )
            
