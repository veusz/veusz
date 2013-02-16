# root.py
# Represents the root widget for plotting the document

#    Copyright (C) 2004 Jeremy S. Sanders
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

import veusz.qtall as qt4

import veusz.document as document
import veusz.setting as setting

import widget
import controlgraph

def _(text, disambiguation=None, context='Root'):
    """Translate text."""
    return unicode( 
        qt4.QCoreApplication.translate(context, text, disambiguation))

class Root(widget.Widget):
    """Root widget class for plotting the document."""

    typename='document'
    allowusercreation = False
    allowedparenttypes = [None]

    def __init__(self, parent, name=None, document=None):
        """Initialise object."""

        widget.Widget.__init__(self, parent, name=name)
        s = self.settings
        self.document = document

        # don't want user to be able to hide entire document
        stylesheet = setting.StyleSheet(descr=_('Master settings for document'),
                                        usertext=_('Style sheet'))
        s.add(stylesheet)
        self.fillStylesheet(stylesheet)

        if type(self) == Root:
            self.readDefaults()

        s.get('englishlocale').setOnModified(self.changeLocale)

    @classmethod
    def addSettings(klass, s):
        widget.Widget.addSettings(s)
        s.remove('hide')

        s.add( setting.DistancePhysical(
                'width',
                '15cm',
                descr=_('Width of the pages'),
                usertext=_('Page width'),
                formatting=True) )
        s.add( setting.DistancePhysical(
                'height',
                '15cm',
                descr=_('Height of the pages'),
                usertext=_('Page height'),
                formatting=True) )
        s.add( setting.Bool(
                'englishlocale', False,
                descr=_('Use US/English number formatting for '
                        'document'),
                usertext=_('English locale'),
                formatting=True) )

    def changeLocale(self):
        """Update locale of document if changed by user."""

        if self.settings.englishlocale:
            self.document.locale = qt4.QLocale.c()
        else:
            self.document.locale = qt4.QLocale()
        self.document.locale.setNumberOptions(qt4.QLocale.OmitGroupSeparator)

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
                controlgraph.ControlMarginBox(self, [0, 0, w, h],
                                              [-10000, -10000,
                                                10000,  10000],
                                              painthelper,
                                              ismovable = False)
                ] )

    def updateControlItem(self, cgi):
        """Call helper to set page size."""
        cgi.setPageSize()

    def fillStylesheet(self, stylesheet):
        """Register widgets with stylesheet."""

        for widgetname in document.thefactory.listWidgets():
            klass = document.thefactory.getWidgetClass(widgetname)
            if klass.allowusercreation or klass == Root:
                newsett = setting.Settings(name=klass.typename,
                                           usertext = klass.typename,
                                           pixmap="button_%s" % klass.typename)
                classset = setting.Settings('temp')
                klass.addSettings(classset)

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
document.thefactory.register( Root )
