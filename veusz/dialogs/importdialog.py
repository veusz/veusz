# data import dialog

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

"""Module for implementing dialog boxes for importing data in Veusz."""

from __future__ import division, print_function
import os.path
import sys

from ..compat import crange
from .. import qtall as qt4
from .. import setting
from .. import utils
from .. import plugins
from . import exceptiondialog
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="ImportDialog"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class ImportTab(qt4.QWidget):
    """Tab for a particular import type."""

    resource = ''
    filetypes = ()       # list of file types handled
    filefilter = None    # name of filter for types for open dialog

    def __init__(self, importdialog, *args):
        """Initialise dialog. importdialog is the import dialog itself."""
        qt4.QWidget.__init__(self, *args)
        self.dialog = importdialog
        self.uiloaded = False

    def loadUi(self):
        """Load up UI file."""
        qt4.loadUi(os.path.join(utils.resourceDirectory, 'ui',
                                self.resource), self)
        self.uiloaded = True

    def reset(self):
        """Reset controls to initial conditions."""
        pass

    def doPreview(self, filename, encoding):
        """Update the preview window, returning whether import
        should be attempted."""
        pass

    def doImport(self, doc, filename, linked, encoding, prefix, suffix, tags):
        """Do the import iteself."""
        pass

    def okToImport(self):
        """Secondary check (after preview) for enabling import button."""
        return True

    def isFiletypeSupported(self, ftype):
        """Is the filetype supported by this tab?"""
        return ftype in self.filetypes

    def useFiletype(self, ftype):
        """If the tab can do something with the selected filetype,
        update itself."""
        pass

importtabs = []
def registerImportTab(name, klass):
    """Register an import tab for the dialog."""
    importtabs.append((name, klass))

class ImportDialog(VeuszDialog):
    """Dialog box for importing data.
    See ImportTab classes above which actually do the work of importing
    """

    dirname = '.'

    def __init__(self, parent, document):

        VeuszDialog.__init__(self, parent, 'import.ui')
        self.document = document

        # whether file import looks likely to work
        self.filepreviewokay = False

        # tabs loaded currently in dialog
        self.tabs = {}
        for tabname, tabclass in importtabs:
            w = tabclass(self)
            self.methodtab.addTab(w, tabname)

        # add promoted plugins
        for p in plugins.importpluginregistry:
            if p.promote_tab is not None:
                from ..dataimport.dialog_plugin import ImportTabPlugins
                w = ImportTabPlugins(self, promote=p.name)
                self.methodtab.addTab(w, p.name)

        self.methodtab.currentChanged.connect(self.slotTabChanged)
        self.browsebutton.clicked.connect(self.slotBrowseClicked)
        self.filenameedit.editTextChanged.connect(self.slotUpdatePreview)

        self.importbutton = self.buttonBox.addButton(
            _("&Import"), qt4.QDialogButtonBox.ApplyRole)
        self.importbutton.clicked.connect(self.slotImport)

        self.buttonBox.button(qt4.QDialogButtonBox.Reset).clicked.connect(
            self.slotReset)
        self.encodingcombo.currentIndexChanged.connect(self.slotUpdatePreview)

        # add completion for filename
        c = self.filenamecompleter = qt4.QCompleter(self)
        self.filenameedit.setCompleter(c)

        # change to tab last used
        self.methodtab.setCurrentIndex(
            setting.settingdb.get('import_lasttab', 0))

        # defaults for prefix and suffix
        self.prefixcombo.default = self.suffixcombo.default = ['', '$FILENAME']

        # default state for check boxes
        self.linkcheckbox.default = True

        # further defaults
        self.encodingcombo.defaultlist = utils.encodings
        self.encodingcombo.defaultval = 'utf_8'

        # load icon for clipboard
        self.clipbutton.setIcon( utils.getIcon('kde-clipboard') )
        qt4.QApplication.clipboard().dataChanged.connect(
            self.updateClipPreview)
        self.clipbutton.clicked.connect(self.slotClipButtonClicked)
        self.updateClipPreview()

    def slotBrowseClicked(self):
        """Browse for a data file."""

        fd = qt4.QFileDialog(self, _('Browse data file'))
        fd.setFileMode( qt4.QFileDialog.ExistingFile )

        # collect filters from tabs
        filters = [_('All files (*.*)')]
        for i in crange(self.methodtab.count()):
            w = self.methodtab.widget(i)
            if w.filefilter:
                ftypes = ' '.join(['*'+t for t in w.filetypes])
                f = '%s (%s)' % (w.filefilter, ftypes)
                filters.append(f)
        fd.setNameFilters(filters)

        lastfilt = setting.settingdb.get('import_filterbrowse')
        if lastfilt in filters:
            fd.selectNameFilter(lastfilt)

        # use filename to guess a path if possible
        filename = self.filenameedit.text()
        if os.path.isdir(filename):
            ImportDialog.dirname = filename
        elif os.path.isdir( os.path.dirname(filename) ):
            ImportDialog.dirname = os.path.dirname(filename)

        fd.setDirectory(ImportDialog.dirname)

        # update filename if changed
        if fd.exec_() == qt4.QDialog.Accepted:
            ImportDialog.dirname = fd.directory().absolutePath()
            self.filenameedit.replaceAndAddHistory( fd.selectedFiles()[0] )
            setting.settingdb['import_filterbrowse'] = fd.selectedNameFilter()
            self.guessImportTab()

    def guessImportTab(self):
        """Guess import tab based on filename."""
        filename = self.filenameedit.text()

        ftype = os.path.splitext(filename)[1]
        # strip off any gz, bz2 extensions to get real extension
        while ftype.lower() in ('.gz', '.bz2'):
            ftype = os.path.splitext(filename)[1]
        ftype = ftype.lower()

        # examine from left to right
        # promoted plugins come after plugins
        idx = -1
        for i in crange(self.methodtab.count()):
            w = self.methodtab.widget(i)
            if w.isFiletypeSupported(ftype):
                idx = i

        if idx >= 0:
            self.methodtab.setCurrentIndex(idx)
            self.methodtab.widget(idx).useFiletype(ftype)

    def slotUpdatePreview(self, *args):
        """Update preview window when filename or tab changed."""

        # save so we can restore later
        tab = self.methodtab.currentIndex()
        setting.settingdb['import_lasttab'] = tab
        filename = self.filenameedit.text()
        encoding = str(self.encodingcombo.currentText())
        importtab = self.methodtab.currentWidget()

        if encoding == '':
            return

        if isinstance(importtab, ImportTab):
            if not importtab.uiloaded:
                importtab.loadUi()
            self.filepreviewokay = importtab.doPreview(
                filename, encoding)

        # enable or disable import button
        self.enableDisableImport()

    def slotTabChanged(self, tabindex):
        """Change completer depending on tab."""
        self.slotUpdatePreview()
        w = self.methodtab.widget(tabindex)

        if w.filetypes is None:
            filters = ['*.*']
        else:
            filters = ['*'+t for t in w.filetypes]
        model = qt4.QDirModel(filters, qt4.QDir.AllDirs | qt4.QDir.Files,
                              qt4.QDir.Name)
        self.filenamecompleter.setModel(model)

    def enableDisableImport(self, *args):
        """Disable or enable import button if allowed."""

        importtab = self.methodtab.currentWidget()
        enabled = self.filepreviewokay and importtab.okToImport()

        # actually enable or disable import button
        self.importbutton.setEnabled( enabled )

    def slotImport(self):
        """Do the importing"""

        filename = self.filenameedit.text()
        linked = self.linkcheckbox.isChecked()
        encoding = str(self.encodingcombo.currentText())
        if filename == '{clipboard}':
            linked = False
        else:
            # normalise filename
            filename = os.path.abspath(filename)

        # import according to tab selected
        importtab = self.methodtab.currentWidget()
        prefix, suffix = self.getPrefixSuffix(filename)
        tags = self.tagcombo.currentText().split()

        try:
            qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )
            with self.document.suspend():
                importtab.doImport(self.document, filename, linked, encoding,
                                   prefix, suffix, tags)
            qt4.QApplication.restoreOverrideCursor()
        except IOError:
            qt4.QApplication.restoreOverrideCursor()
            qt4.QMessageBox.warning(
                self, _("Veusz"),
                _("Could not read file"))
        except Exception:
            qt4.QApplication.restoreOverrideCursor()

            # show exception dialog
            d = exceptiondialog.ExceptionDialog(sys.exc_info(), self)
            d.exec_()

    def retnDatasetInfo(self, dsnames, linked, filename):
        """Return a list of information for the dataset names given."""

        lines = [_('Imported data for datasets:')]
        dsnames.sort()
        for name in dsnames:
            ds = self.document.getData(name)
            # build up description
            lines.append(_('%s: %s') % (name, ds.description()))

        # whether the data were linked
        if linked:
            lines.append('')
            lines.append(_('Datasets were linked to file "%s"') % filename)

        return lines

    def getPrefixSuffix(self, filename):
        """Get prefix and suffix values."""
        f = utils.cleanDatasetName( os.path.basename(filename) )
        prefix = self.prefixcombo.lineEdit().text()
        prefix = prefix.replace('$FILENAME', f)
        suffix = self.suffixcombo.lineEdit().text()
        suffix = suffix.replace('$FILENAME', f)
        return prefix, suffix

    def slotReset(self):
        """Reset input fields."""

        self.filenameedit.setText("")
        self.encodingcombo.setCurrentIndex(
            self.encodingcombo.findText("utf_8"))
        self.linkcheckbox.setChecked(True)
        self.prefixcombo.setEditText("")
        self.suffixcombo.setEditText("")

        importtab = self.methodtab.currentWidget()
        importtab.reset()

    def slotClipButtonClicked(self):
        """Clicked clipboard button."""
        self.filenameedit.setText("{clipboard}")

    def updateClipPreview(self):
        """Clipboard contents changed, so update preview if showing clipboard."""

        filename = self.filenameedit.text()
        if filename == '{clipboard}':
            self.slotUpdatePreview()
