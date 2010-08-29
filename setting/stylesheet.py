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
##############################################################################

# $Id$

import sys

from settings import Settings
import setting
import collections

import veusz.qtall as qt4

class StyleSheet(Settings):
    """A class for handling default values of settings.
    
    Settings are registered to be added to the stylesheet."""

    registeredsettings = []

    @classmethod
    def register(kls, settingskls, posn=None):
        """Register a settings object with the stylesheet.
        This settings object is copied for each new document.
        """
        if posn is None:
            kls.registeredsettings.append(settingskls)
        else:
            kls.registeredsettings.insert(posn, settingskls)

    def __init__(self, **args):
        """Create the default settings."""
        Settings.__init__(self, 'StyleSheet', **args)
        self.pixmap = 'settings_stylesheet'

        for subset in self.registeredsettings:
            self.add( subset() )

class StylesheetLine(Settings):
    """Hold the properties of the default line."""
    def __init__(self):
        Settings.__init__(self, 'Line', pixmap='settings_plotline',
                          descr='Default line style for document',
                          usertext='Line')
        self.add( setting.DistancePt('width', '0.5pt',
                                     descr='Default line width',
                                     usertext='Width',
                                     formatting=True) )
        self.add( setting.Color('color', 'black',
                                descr='Default line color',
                                usertext='Color',
                                formatting=True) )
# register these properties with the stylesheet
StyleSheet.register(StylesheetLine)

def _registerFontStyleSheet():
    """Get fonts, and register default with StyleSheet and Text class."""
    families = [ unicode(name) for name in qt4.QFontDatabase().families() ]
    
    deffont = None
    for f in ('Times New Roman', 'Bitstream Vera Serif', 'Times', 'Utopia',
              'Serif'):
        if f in families:
            deffont = unicode(f)
            break
            
    if deffont is None:
        print >>sys.stderr, "Warning: did not find a sensible default font. Choosing first font."    
        deffont = unicode(families[0])

    collections.Text.defaultfamily = deffont
    collections.Text.families = families
    StylesheetText.defaultfamily = deffont
    StylesheetText.families = families

class StylesheetText(Settings):
    """Hold properties of default text font."""

    defaultfamily = None
    families = None

    def __init__(self):
        """Initialise with default font family and list of families."""
        Settings.__init__(self, 'Font', pixmap='settings_axislabel',
                          descr='Default font for document',
                          usertext='Font')

        if StylesheetText.defaultfamily is None:
            _registerFontStyleSheet()

        self.add( setting.FontFamily('font', StylesheetText.defaultfamily,
                                     descr='Font name', usertext='Font',
                                     formatting=True))
        self.add( setting.DistancePt('size', '14pt',
                                     descr='Default font size',
                                     usertext='Size',
                                     formatting=True))
        self.add( setting.Color('color', 'black',
                                descr='Default font color',
                                usertext='Color',
                                formatting=True))
StyleSheet.register(StylesheetText)
