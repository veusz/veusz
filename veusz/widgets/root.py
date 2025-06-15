# root.py
# Represents the root widget for plotting the document

#    Copyright (C) 2004 Jeremy S. Sanders
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

import textwrap
from .. import qtall as qt

from .. import document
from .. import setting

from . import widget
from . import controlgraph

def _(text, disambiguation=None, context='Root'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class Root(widget.Widget):
    """Root widget class for plotting the document."""

    typename = 'document'
    allowusercreation = False

    def __init__(self, parent, name=None, compatlevel=0):
        """Initialise object."""

        widget.Widget.__init__(self, parent, name=name)
        self.makeStylesheet(compatlevel=compatlevel)

        s = self.settings
        s.get('englishlocale').setOnModified(self.changeLocale)
        s.get('colorTheme').setOnModified(self.changeColorTheme)

    def makeStylesheet(self, compatlevel=0):
        """Create a new stylesheet, overwriting an existing one."""
        stylesheet = setting.StyleSheet(
            descr=_('Master settings for document'),
            usertext=_('Style sheet'))
        s = self.settings
        if s.isSetting('StyleSheet'):
            s.remove('StyleSheet')
        s.add(stylesheet)
        self.fillStylesheet(stylesheet, compatlevel)

    @classmethod
    def addSettings(klass, s):
        widget.Widget.addSettings(s)

        # don't want user to be able to hide entire document
        s.remove('hide')

        s.add( setting.DistancePhysical(
            'width',
            '15cm',
            descr=_('Width of the pages'),
            usertext=_('Page width'),
            formatting=True
        ) )
        s.add( setting.DistancePhysical(
            'height',
            '15cm',
            descr=_('Height of the pages'),
            usertext=_('Page height'),
            formatting=True
        ) )
        s.add( setting.Bool(
            'englishlocale', False,
            descr=_('Use US/English number formatting for document'),
            usertext=_('English locale'),
            formatting=True
        ) )

        themes = sorted(list(document.colors.colorthemes))
        s.add( setting.Choice(
            'colorTheme',
            themes,
            'black',
            descr=_('Color theme'),
            usertext=_('Color theme'),
            formatting=True
        ) )

        s.add( setting.Notes(
            'notes', '',
            descr=_('User-defined notes'),
            usertext=_('Notes')
        ) )

    @classmethod
    def allowedParentTypes(klass):
        return (None,)

    @property
    def userdescription(self):
        """Return user-friendly description."""
        return textwrap.fill(self.settings.notes, 60)

    def changeLocale(self):
        """Update locale of document if changed by user."""

        if self.settings.englishlocale:
            self.document.locale = qt.QLocale.c()
        else:
            self.document.locale = qt.QLocale()
        self.document.locale.setNumberOptions(qt.QLocale.NumberOption.OmitGroupSeparator)

    def changeColorTheme(self):
        """Change color theme used by document."""
        self.document.evaluate.colors.setColorTheme(
            self.settings.colorTheme)

    def getPage(self, pagenum):
        """Get page widget."""
        try:
            return self.children[pagenum]
        except IndexError:
            return None

    def draw(self, painthelper, pagenum):
        """Draw the page requested on the painter."""

        xw, yw = painthelper.pagesize
        posn = [0, 0, xw, yw]
        painter = painthelper.painter(self, posn)
        with painter:
            page = self.children[pagenum]
            page.draw( posn, painthelper )

        # w and h are non integer
        w = self.settings.get('width').convert(painter)
        h = self.settings.get('height').convert(painter)
        painthelper.setControlGraph(self, [
                controlgraph.ControlMarginBox(
                    self, [0, 0, w, h],
                    [-10000, -10000, 10000,  10000],
                    painthelper,
                    ismovable=False,
                )
        ])

    def updateControlItem(self, cgi):
        """Call helper to set page size."""
        cgi.setPageSize()

    def fillStylesheet(self, stylesheet, compatlevel):
        """Register widgets with stylesheet."""

        for widgetname in document.thefactory.listWidgets():
            klass = document.thefactory.getWidgetClass(widgetname)
            if klass.allowusercreation or klass == Root:
                newsett = setting.Settings(
                    name=klass.typename,
                    usertext = klass.typename,
                    pixmap="button_%s" % klass.typename)
                classset = setting.Settings('temp')
                klass.addSettings(classset)
                klass.addSettingsCompatLevel(classset, compatlevel)

                # copy formatting settings to stylesheet
                for name in classset.setnames:
                    # might become recursive
                    if name == 'StyleSheet':
                        continue

                    sett = classset.setdict[name]
                    # skip non formatting settings
                    #if hasattr(sett, 'formatting') and not sett.formatting:
                    #    continue
                    newsett.add( sett.copy() )

                stylesheet.add(newsett)

# allow the factory to instantiate this
document.thefactory.register(Root)
