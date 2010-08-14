# data reload dialog

#    Copyright (C) 2005 Jeremy S. Sanders
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

"""Dialog for reloading linked data."""

import veusz.qtall as qt4
import veusz.document as document
import veusz.utils as utils
from veuszdialog import VeuszDialog

class ReloadData(VeuszDialog):
    """Dialog for reloading linked datasets."""

    def __init__(self, document, parent):
        """Initialise the dialog."""

        VeuszDialog.__init__(self, parent, 'reloaddata.ui')
        self.document = document

        # actually reload the data (and show the user)
        self.reloadData()

    def reloadData(self):
        """Reload linked data. Show the user what was done."""

        text = ''
        try:
            # try to reload the datasets
            datasets, errors = self.document.reloadLinkedDatasets()

            # show errors in read data
            for var, count in errors.items():
                if count != 0:
                    text += ( '%i conversions failed for dataset "%s"\n' %
                              (count, var) )

            # show successes
            if len(datasets) != 0:
                text += 'Reloaded\n'
                for var in datasets:
                    descr = self.document.data[var].description()
                    if descr:
                        text += ' %s\n' % descr
                    else:
                        text += ' %s\n' % var

        except IOError, e:
            text = 'Error reading file:\n' + unicode(e)
        except document.DescriptorError:
            text = 'Could not interpret descriptor. Reload failed.'

        if text == '':
            text = 'Nothing to do. No linked datasets.'

        self.outputedit.setPlainText(text)
        
