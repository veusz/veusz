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

# $Id: $

import os.path

import veusz.utils as utils
import veusz.qtall as qt4
from veusz.windows.treeeditwindow import TabbedFormatting, PropertyList

class StylesheetDialog(qt4.QDialog):
    """This is a dialog box to edit stylesheets.
    Most of the work is done elsewhere, so this doesn't do a great deal
    """

    def __init__(self, parent, document):
        qt4.QDialog.__init__(self, parent)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'stylesheet.ui'),
                   self)
        self.document = document
        self.stylesheet = document.basewidget.settings.StyleSheet

        self.formattingLayout = qt4.QVBoxLayout()
        self.formattingGroup.setLayout(self.formattingLayout)

        self.stylesListWidget.setMinimumWidth(100)

        # initial properties widget
        self.tabformat = None
        self.properties = None

        self.fillStyleList()

        self.connect(self.stylesListWidget,
                     qt4.SIGNAL(
                'currentItemChanged(QListWidgetItem *,QListWidgetItem *)'),
                     self.slotStyleItemChanged)

    def fillStyleList(self):
        """Fill list of styles."""
        for stns in self.stylesheet.getSettingsList():
            item = qt4.QListWidgetItem(utils.getIcon(stns.pixmap),
                                       stns.usertext)
            item.VZsettings = stns
            self.stylesListWidget.addItem(item)

    def slotStyleItemChanged(self, current, previous):
        """Item changed in list of styles."""
        if current is None:
            return

        if self.tabformat:
            self.tabformat.deleteLater()
        if self.properties:
            self.properties.deleteLater()

        style = str(current.text())
        settings = current.VZsettings#self.stylesheet.get(style)

        # update formatting properties
        self.tabformat = TabbedFormatting(self.document, settings)
        self.formattingLayout.addWidget(self.tabformat)

        # update properties
        self.properties = PropertyList(self.document, showsubsettings=False)
        self.properties.updateProperties(settings, showformatting=False)
        self.propertiesScrollArea.setWidget(self.properties)

