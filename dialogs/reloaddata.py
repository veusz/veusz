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

"""Dialog for reloading linked data."""

import os

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

        # update on reloading
        self.reloadct = 1

        # get a record of names, dates and sizes of files linked
        self.filestats = self.statLinkedFiles()

        # actually reload the data (and show the user)
        self.reloadData()

        # if interval changed or enabled update timer
        self.connect(self.intervalCheck, qt4.SIGNAL('clicked()'),
                     self.intervalUpdate)
        self.connect(self.intervalTime, qt4.SIGNAL('valueChanged(int)'),
                     self.intervalUpdate)

        # timer to reload data
        self.intervalTimer = qt4.QTimer()
        self.connect(self.intervalTimer, qt4.SIGNAL('timeout()'),
                     self.reloadIfChanged)

        # manual reload
        self.reloadbutton = self.buttonBox.addButton(
            "&Reload again", qt4.QDialogButtonBox.ApplyRole)
        self.connect(self.reloadbutton, qt4.SIGNAL('clicked()'),
                     self.reloadData)

        # close by default, not reload
        self.buttonBox.button(qt4.QDialogButtonBox.Close).setDefault(True)

    def statLinkedFiles(self):
        """Stat linked files.
        Returns a list of (filename, mtime, size)
        """

        files = []
        for lf in self.document.getLinkedFiles():
            filename = lf.filename
            try:
                s = os.stat(filename)
                files.append( (filename, s.st_mtime, s.st_size) )
            except OSError:
                pass
        files.sort()
        return files

    def intervalUpdate(self, *args):
        """Reload at intervals option toggled."""
        if self.intervalCheck.isChecked():
            self.intervalTimer.start( self.intervalTime.value()*1000 )
        else:
            self.intervalTimer.stop()

    def reloadIfChanged(self):
        """Reload linked data if it has changed."""
        newstat = self.statLinkedFiles()
        if newstat != self.filestats:
            self.filestats = newstat
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
                text += 'Reloaded (%i)\n' % self.reloadct
                self.reloadct += 1
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
        
