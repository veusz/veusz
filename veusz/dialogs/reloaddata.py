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

from __future__ import division
import os

from ..compat import cstr
from .. import qtall as qt
from .. import document
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="ReloadDialog"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class ReloadData(VeuszDialog):
    """Dialog for reloading linked datasets."""

    def __init__(self, document, parent, filenames=None):
        """Initialise the dialog.

        document: veusz document
        parent: parent window
        filenames: if a set() only reload from these filenames
        """

        VeuszDialog.__init__(self, parent, 'reloaddata.ui')
        self.document = document
        self.filenames = filenames

        # update on reloading
        self.reloadct = 1

        # get a record of names, dates and sizes of files linked
        self.filestats = self.statLinkedFiles()

        # actually reload the data (and show the user)
        self.reloadData()

        # if interval changed or enabled update timer
        self.intervalCheck.clicked.connect(self.intervalUpdate)
        self.intervalTime.valueChanged[int].connect(self.intervalUpdate)

        # timer to reload data
        self.intervalTimer = qt.QTimer()
        self.intervalTimer.timeout.connect(self.reloadIfChanged)

        # manual reload
        self.reloadbutton = self.buttonBox.addButton(
            "&Reload again", qt.QDialogButtonBox.ApplyRole)
        self.reloadbutton.clicked.connect(self.reloadData)

        # close by default, not reload
        self.buttonBox.button(qt.QDialogButtonBox.Close).setDefault(True)

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

        lines = []
        datasets = []
        errors = {}
        try:
            # try to reload the datasets
            datasets, errors = self.document.reloadLinkedDatasets(
                self.filenames)
        except EnvironmentError as e:
            lines.append(_("Error reading file: %s") % cstr(e))

        # header showing count
        if len(datasets) > 0:
            lines.append(_("Reloaded (%i)") % self.reloadct)
            self.reloadct += 1

        # show errors in read data
        for var, count in errors.items():
            if count:
                lines.append(
                    _('%i conversions failed for dataset "%s"') %
                    (count, var)
                )

        # show successes
        # group datasets by linked file
        linked = set()
        for var in datasets:
            ds = self.document.data[var]
            linked.add(ds.linked)

        linked = [(l.filename, l) for l in linked]
        linked.sort(key=lambda x: x[0])

        # list datasets for each linked file
        for lname, link in linked:
            lines.append('')
            lines.append(_('Linked to %s') % lname)
            for var in sorted(datasets):
                ds = self.document.data[var]
                if ds.linked is link:
                    lines.append( ' %s: %s' % (
                        var, ds.description()) )

        if len(datasets) == 0:
            lines.append(_('Nothing to do. No linked datasets.'))

        self.outputedit.setPlainText('\n'.join(lines))
