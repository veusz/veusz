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
        stylesheet = setting.StyleSheet(descr='Master settings for document',
                                        usertext='Style sheet')
        s.add(stylesheet)
        self.fillStylesheet(stylesheet)

        if type(self) == Root:
            self.readDefaults()

        s.get('englishlocale').setOnModified(self.changeLocale)

    @classmethod
    def addSettings(klass, s):
        widget.Widget.addSettings(s)
        s.remove('hide')

        s.add( setting.Distance('width',
                                '15cm',
                                descr='Width of the pages',
                                usertext='Page width',
                                formatting=True) )
        s.add( setting.Distance('height',
                                '15cm',
                                descr='Height of the pages',
                                usertext='Page height',
                                formatting=True) )    
        s.add( setting.Bool('englishlocale', False,
                            descr='Use US/English number formatting for '
                            'document',
                            usertext='English locale',
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
        return self.children[pagenum]

    def draw(self, painthelper, pagenum):
        """Draw the page requested on the painter."""

        xw, yw = painthelper.pagesize
        posn = [0, 0, xw, yw]
        painter = painthelper.painter(self, posn)
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
                stylesheet.add(newsett)

                klass.addSettings(newsett)

                # classset = setting.Settings('temp')
                # klass.addSettings(classset)

                # # copy settings to stylesheet
                # for name in classset.setnames:
                #     # might become recursive
                #     if name == 'StyleSheet':
                #         continue

                #     sett = classset.setdict[name]
                #     newsett.add( sett.copy() )

# allow the factory to instantiate this
document.thefactory.register( Root )
