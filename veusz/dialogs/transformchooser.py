#    Copyright (C) 2016 Jeremy S. Sanders
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

"""Dialog for choosing a transform."""

from __future__ import print_function, division
import os
import inspect
from collections import defaultdict

from ..compat import citems
from .. import qtall as qt4
from .. import plugins
from .. import utils

def _(text, disambiguation=None, context="TransformChooser"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class TransformChooser(qt4.QDialog):
    """Dialog for choosing transforms."""

    def __init__(self, parent, document):
        """Initialise the dialog.

        parent: parent window
        setting: setting to update
        """

        allcat = _('All')

        qt4.QDialog.__init__(self, parent)
        qt4.loadUi(os.path.join(
            utils.resourceDirectory, 'ui', 'transformchooser.ui'), self)
        self.document = document

        # collect plugins, categories and descriptions
        self.catplugins = defaultdict(list)
        self.descriptions = {}
        allnames = []

        for name, item in citems(plugins.transformpluginregistry):
            func, username, category, description = item
            self.catplugins[category].append(name)
            self.descriptions[name] = description
            allnames.append(name)

        # construct list widget of categories
        self.categoryList.addItem(allcat)
        for cat in sorted(self.catplugins):
            self.categoryList.addItem(cat)

        # all group with everything
        self.catplugins[allcat] = allnames

        # this is return value
        self.retn = None

        self.categoryList.currentTextChanged.connect(self.slotCategoryChanged)
        self.transformList.currentTextChanged.connect(self.slotTransformChanged)
        self.categoryList.setCurrentRow(0)

    def slotCategoryChanged(self, category):
        """Update list of transforms based on category changed."""
        self.transformList.clear()
        for plugin in sorted(self.catplugins[category]):
            self.transformList.addItem(plugin)
        self.transformList.setCurrentRow(0)

    def slotTransformChanged(self, transform):
        """Update description based on transform changed."""

        # description
        label = self.descriptions.get(transform, _("None"))
        self.descriptionLabel.setText(label)

        # function signature
        env = self.document.transform.transformenv
        if transform in env:
            args = inspect.formatargspec(*inspect.getargspec(env[transform]))
        else:
            args = ''
        callsig = transform + args
        self.callSigLabel.setText(callsig)

        # set return value
        self.retn = callsig
