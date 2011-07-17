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

        painthelper.setControlGraph(self, [
                controlgraph.ControlMarginBox(self, posn,
                                              [-10000, -10000,
                                                10000,  10000],
                                              painthelper,
                                              ismovable = False)
                ] )

    def updateControlItem(self, cgi):
        """Graph resized or moved - call helper routine to move self."""

        s = self.settings

        # get margins in pixels
        width = cgi.posn[2] - cgi.posn[0]
        height = cgi.posn[3] - cgi.posn[1]

        # set up fake painter containing veusz scalings
        helper = document.PaintHelper(cgi.pagesize, scaling=cgi.scaling)

        # convert to physical units
        width = s.get('width').convertInverse(width, helper)
        height = s.get('height').convertInverse(height, helper)

        # modify widget margins
        operations = (
            document.OperationSettingSet(s.get('width'), width),
            document.OperationSettingSet(s.get('height'), height),
            )
        self.document.applyOperation(
            document.OperationMultiple(operations, descr='change page size'))

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
