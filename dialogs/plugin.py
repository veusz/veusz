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

"""Dialog boxes for tools and dataset plugins."""

import sys
from itertools import izip

import veusz.qtall as qt4
import veusz.utils as utils
import veusz.document as document
import veusz.plugins as plugins
import exceptiondialog
import dataeditdialog
from veuszdialog import VeuszDialog

def handlePlugin(mainwindow, doc, pluginkls):
    """Show plugin dialog or directly execute (if it takes no parameters)."""

    plugin = pluginkls()
    if plugin.has_parameters:
        d = PluginDialog(mainwindow, doc, plugin)
        mainwindow.showDialog(d)
    else:
        fields = {'currentwidget': mainwindow.treeedit.selwidget.path}
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

    def __init__(self, mainwindow, doc, plugin):
        VeuszDialog.__init__(self, mainwindow, 'plugin.ui')

        reset = self.buttonBox.button(qt4.QDialogButtonBox.Reset)
        reset.setAutoDefault(False)
        reset.setDefault(False)
        self.connect(reset, qt4.SIGNAL('clicked()'), self.slotReset)
        self.connect( self.buttonBox.button(qt4.QDialogButtonBox.Apply),
                      qt4.SIGNAL('clicked()'), self.slotApply )

        self.plugin = plugin
        self.document = doc

        title = ': '.join(list(plugin.menu))
        self.setWindowTitle(title)
        descr = plugin.description_full
        if self.plugin.author:
            descr += '\n Author: ' + self.plugin.author
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

        currentwidget = self.mainwindow.treeedit.selwidget.path
        for row, field in enumerate(self.plugin.fields):
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
        for field, cntrl in izip(self.fields, self.fieldcntrls):
            field.setControlVal(cntrl, oldfields[field.name])

    def slotApply(self):
        """Use the plugin with the inputted data."""

        # default field
        fields = {'currentwidget': self.mainwindow.treeedit.selwidget.path}

        # read values from controls
        for field, cntrls in izip(self.fields, self.fieldcntrls):
            fields[field.name] = field.getControlResults(cntrls)

        # run plugin
        statustext = runPlugin(self, self.document, self.plugin, fields)

        # show any results
        self.notifyLabel.setText(statustext)
        qt4.QTimer.singleShot(3000, self.notifyLabel.clear)

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
    qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )
    try:
        results = doc.applyOperation(op)

        # evaluate datasets using plugin to check it works
        if mode == 'dataset':
            op.validate()
            resultstext = 'Created datasets: ' + ', '.join(results)
        else:
            resultstext = 'Done'

    except (plugins.ToolsPluginException, plugins.DatasetPluginException), ex:
        # unwind operations
        op.undo(doc)
        qt4.QApplication.restoreOverrideCursor()

        qt4.QMessageBox.warning(
            window, "Error in %s" % plugin.name, unicode(ex))

    except Exception:
        op.undo(doc)
        qt4.QApplication.restoreOverrideCursor()

        # show exception dialog
        exceptiondialog.ExceptionDialog(sys.exc_info(), window).exec_()

    else:
        qt4.QApplication.restoreOverrideCursor()

    return resultstext

def recreateDataset(mainwindow, document, dataset, datasetname):
    """Open dialog to recreate plugin dataset(s)."""

    # make a new instance of the plugin class
    newplugin = dataset.pluginmanager.plugin.__class__()

    dialog = PluginDialog(mainwindow, document, newplugin)
    mainwindow.showDialog(dialog)
    dialog.reEditDataset(dataset, datasetname)

dataeditdialog.recreate_register[document.Dataset1DPlugin] = recreateDataset
dataeditdialog.recreate_register[document.Dataset2DPlugin] = recreateDataset
dataeditdialog.recreate_register[document.DatasetTextPlugin] = recreateDataset
