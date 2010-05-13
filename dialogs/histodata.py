#    Copyright (C) 2010 Jeremy S. Sanders
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

import os.path

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.utils as utils
import veusz.document as document
from veusz.setting.controls import populateCombo

import numpy as N

class HistoDataDialog(qt4.QDialog):
    """Preferences dialog."""

    def __init__(self, document, *args):
        """Setup dialog."""
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'histodata.ui'),
                   self)

        self.document = document

        self.minval.default = self.maxval.default = ['Auto']
        regexp = qt4.QRegExp("[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?|Auto")
        validator = qt4.QRegExpValidator(regexp, self)
        self.minval.setValidator(validator)
        self.maxval.setValidator(validator)

        self.connect(document, qt4.SIGNAL('sigModified'),
                     self.updateDatasetLists)
        self.updateDatasetLists()

    def escapeDatasets(self, dsnames):
        """Escape dataset names if they are not typical python ones."""

        for i in xrange(len(dsnames)):
            if not utils.validPythonIdentifier(dsnames[i]):
                dsnames[i] = '`%s`' % dsnames[i]

    def updateDatasetLists(self):
        """Update list of datasets."""

        datasets = []
        for name, ds in self.document.data.iteritems():
            if ds.datatype == 'numeric' and ds.dimensions == 1:
                datasets.append(name)
        datasets.sort()

        # make sure names are escaped if they have funny characters
        self.escapeDatasets(datasets)

        # help the user by listing existing datasets
        populateCombo(self.indataset, datasets)

