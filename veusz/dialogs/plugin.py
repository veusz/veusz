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

"""Dialog boxes for tools and dataset plugins."""

from __future__ import division
import sys

from ..compat import czip, cstr
from .. import qtall as qt
from .. import document
from .. import plugins
from . import exceptiondialog
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="PluginDialog"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

def handlePlugin(mainwindow, doc, pluginkls):
    """Show plugin dialog or directly execute (if it takes no parameters)."""

    plugin = pluginkls()
    if plugin.has_parameters:
        d = PluginDialog(mainwindow, doc, plugin, pluginkls)
        mainwindow.showDialog(d)
    else:
        fields = {'currentwidget': '/'}
        if mainwindow.treeedit.selwidgets:
            fields = {'currentwidget': mainwindow.treeedit.selwidgets[0].path}
        runPlugin(mainwindow, doc, plugin, fields)

def wordwrap(text, linelength=80):
    """Wrap on a word boundary."""
    out = []
    l = 0
    for w in text.split(' '):
        if w.find('\n') >= 0:
            l = 0
        if l + len(w) > linelength:
            out.append('\n')
            l = 0
        out.append(w)
        l += len(w)
    return ' '.join(out)

class PluginDialog(VeuszDialog):
    """Dialog box class for plugins."""

    def __init__(self, mainwindow, doc, plugininst, pluginkls):
        VeuszDialog.__init__(self, mainwindow, 'plugin.ui')

        reset = self.buttonBox.button(qt.QDialogButtonBox.Reset)
        reset.setAutoDefault(False)
        reset.setDefault(False)
        reset.clicked.connect( self.slotReset)
        self.buttonBox.button(
            qt.QDialogButtonBox.Apply).clicked.connect(self.slotApply)

        self.pluginkls = pluginkls
        self.plugininst = plugininst
        self.document = doc

        title = ': '.join(list(plugininst.menu))
        self.setWindowTitle(title)
        descr = plugininst.description_full
        if plugininst.author:
            descr += '\n ' + _('Author: %s') % plugininst.author
        self.descriptionLabel.setText( wordwrap(descr) )

        self.fieldcntrls = []
        self.fields = []
        self.addFields()

    def addFields(self):
        """Add any fields, removing existing ones if required."""
        layout = self.fieldGroup.layout()

        for line in self.fieldcntrls:
            for cntrl in line:
                layout.removeWidget(cntrl)
                cntrl.deleteLater()
        del self.fieldcntrls[:]

        currentwidget = '/'
        if self.mainwindow.treeedit.selwidgets:
            currentwidget = self.mainwindow.treeedit.selwidgets[0].path
        for row, field in enumerate(self.plugininst.fields):
            if isinstance(field, list) or isinstance(field, tuple):
                for c, f in enumerate(field):
                    cntrls = f.makeControl(self.document, currentwidget)
                    layout.addWidget(cntrls[0], row, c*2)
                    layout.addWidget(cntrls[1], row, c*2+1)
                    self.fieldcntrls.append(cntrls)
                    self.fields.append(f)
            else:
                cntrls = field.makeControl(self.document, currentwidget)
                layout.addWidget(cntrls[0], row, 0)
                layout.addWidget(cntrls[1], row, 1)
                self.fieldcntrls.append(cntrls)
                self.fields.append(field)

    def slotReset(self):
        """Reset fields to defaults."""
        self.addFields()

    def reEditDataset(self, ds, dsname):
        """Open up dataset in dialog for editing."""

        oldfields = ds.pluginmanager.fields
        for field, cntrl in czip(self.fields, self.fieldcntrls):
            field.setControlVal(cntrl, oldfields[field.name])

    def slotApply(self):
        """Use the plugin with the inputted data."""

        # default field
        fields = {'currentwidget': '/'}
        if self.mainwindow.treeedit.selwidgets:
            fields = {'currentwidget': self.mainwindow.treeedit.selwidgets[0].path}

        # read values from controls
        for field, cntrls in czip(self.fields, self.fieldcntrls):
            fields[field.name] = field.getControlResults(cntrls)

        # run plugin
        plugin = self.pluginkls()
        statustext = runPlugin(self, self.document, plugin, fields)

        # show any results
        self.notifyLabel.setText(statustext)
        qt.QTimer.singleShot(3000, self.notifyLabel.clear)

def runPlugin(window, doc, plugin, fields):
    """Execute a plugin.
    window - parent window
    doc - veusz document
    plugin - plugin object."""

    if isinstance(plugin, plugins.ToolsPlugin):
        mode = 'tools'
    elif isinstance(plugin, plugins.DatasetPlugin):
        mode = 'dataset'
    else:
        raise RuntimeError("Invalid plugin class")

    # use correct operation class for different plugin types
    if mode == 'tools':
        op = document.OperationToolsPlugin(plugin, fields)
    elif mode == 'dataset':
        # a bit of a hack as we don't give currentwidget to this plugin
        del fields['currentwidget']
        op = document.OperationDatasetPlugin(plugin, fields)

    resultstext = ''
    qt.QApplication.setOverrideCursor( qt.QCursor(qt.Qt.WaitCursor) )
    try:
        results = doc.applyOperation(op)

        # evaluate datasets using plugin to check it works
        if mode == 'dataset':
            op.validate()
            resultstext = _('Created datasets: ') + ', '.join(results)
        else:
            resultstext = _('Done')

    except (plugins.ToolsPluginException, plugins.DatasetPluginException) as ex:
        # unwind operations
        op.undo(doc)
        qt.QApplication.restoreOverrideCursor()

        qt.QMessageBox.warning(
            window, _("Error in %s") % plugin.name, cstr(ex))

    except Exception:
        op.undo(doc)
        qt.QApplication.restoreOverrideCursor()

        # show exception dialog
        exceptiondialog.ExceptionDialog(sys.exc_info(), window).exec_()

    else:
        qt.QApplication.restoreOverrideCursor()

    return resultstext

def recreateDataset(mainwindow, document, dataset, datasetname):
    """Open dialog to recreate plugin dataset(s)."""

    # make a new instance of the plugin class
    kls = dataset.pluginmanager.plugin.__class__
    newplugin = kls()

    dialog = PluginDialog(mainwindow, document, newplugin, kls)
    mainwindow.showDialog(dialog)
    dialog.reEditDataset(dataset, datasetname)
