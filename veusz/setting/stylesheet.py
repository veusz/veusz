#    Copyright (C) 2009 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This file is part of Veusz.
#
#    Veusz is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    Veusz is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################

from .settings import Settings
from . import setting
from . import collections

from .. import qtall as qt

def _(text, disambiguation=None, context="Setting"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

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
        Settings.__init__(self, 'StyleSheet', setnsmode='stylesheet', **args)
        self.pixmap = 'settings_stylesheet'

        for subset in self.registeredsettings:
            self.add( subset() )

class StylesheetLine(Settings):
    """Hold the properties of the default line."""
    def __init__(self):
        Settings.__init__(
            self, 'Line', pixmap='settings_plotline',
            descr=_('Default line style for document'),
            usertext=_('Line'))
        self.add( setting.DistancePt(
            'width', '0.5pt',
            descr=_('Default line width'),
            usertext=_('Width'),
            formatting=True) )
        self.add( setting.Color(
            'color', 'foreground',
            descr=_('Default line color'),
            usertext=_('Color'),
            formatting=True) )
# register these properties with the stylesheet
StyleSheet.register(StylesheetLine)

def _registerFontStyleSheet():
    """Get fonts, and register default with StyleSheet and Text class."""
    families = qt.QFontDatabase.families()

    deffont = None
    for f in (
            'Times New Roman',
            'Bitstream Vera Serif',
            'Times',
            'Utopia',
            'Serif'
    ):
        if f in families:
            deffont = f
            break

    if len(families) == 0:
        # testing - no fonts available
        deffont = 'Serif'
        families = ['Serif']
    elif deffont is None:
        print("Warning: did not find a default font. Choosing Qt default font.")
        deffont = qt.QFontDatabase.systemFont(qt.QFontDatabase.SystemFont.GeneralFont).family()

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
        Settings.__init__(
            self, 'Font', pixmap='settings_axislabel',
            descr=_('Default font for document'),
            usertext=_('Font'))

        if StylesheetText.defaultfamily is None:
            _registerFontStyleSheet()

        self.add( setting.FontFamily(
            'font', StylesheetText.defaultfamily,
            descr=_('Font name'), usertext=_('Font'),
            formatting=True))
        self.add( setting.DistancePt(
            'size', '14pt',
            descr=_('Default font size'),
            usertext=_('Size'),
            formatting=True))
        self.add( setting.Color(
            'color', 'foreground',
            descr=_('Default font color'),
            usertext=_('Color'),
            formatting=True))
        self.add( setting.FontStyle(
            'style', '',
            'font',
            descr=_('Default font style'),
            usertext=_('Style'),
            formatting=True))

StyleSheet.register(StylesheetText)
