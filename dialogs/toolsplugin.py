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

"""Dialog boxes for tools plugins."""

import os.path
import sys

import veusz.qtall as qt4
import veusz.utils as utils
import veusz.document as document
import veusz.plugins as plugins
import exceptiondialog

def handleToolsPlugin(mainwindow, doc, pluginkls):
    """Show tools plugin dialog or directly execute
    (if it takes no parameters)."""

    plugin = pluginkls()
    if plugin.has_parameters:
        d = ToolsPluginDialog(mainwindow, doc, plugin)
        mainwindow.showDialog(d)
    else:
        fields = {'currentwidget': mainwindow.treeedit.selwidget.path}
        runPlugin(mainwindow, doc, plugin, fields)

def runPlugin(window, doc, plugin, fields):
    """Run plugin.
    window - parent window
    doc - veusz document
    plugin - plugin object."""
    
    # keep track of all changes in this
    op = document.OperationToolsPlugin(plugin, fields)

    qt4.QApplication.setOverrideCursor( qt4.QCursor(qt4.Qt.WaitCursor) )
    try:
        doc.applyOperation(op)

    except plugins.ToolsPluginException, ex:
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

class ToolsPluginDialog(qt4.QDialog):
    """Dialog box for tools plugins."""

    def __init__(self, mainwindow, doc, plugin):
        qt4.QDialog.__init__(self, mainwindow)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'toolsplugin.ui'),
                   self)

        self.connect( self.buttonBox.button(qt4.QDialogButtonBox.Reset),
                      qt4.SIGNAL('clicked()'), self.slotReset )
        self.connect( self.buttonBox.button(qt4.QDialogButtonBox.Apply),
                      qt4.SIGNAL('clicked()'), self.slotApply )

        self.plugin = plugin
        self.mainwindow = mainwindow
        self.document = doc

        self.setWindowTitle(plugin.name)
        descr = (plugin.description_full +
                 '\n Author: ' + self.plugin.author)
        self.descriptionLabel.setText(descr)

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

        for row, field in enumerate(self.plugin.fields):
            if isinstance(field, list) or isinstance(field, tuple):
                for c, f in enumerate(field):
                    cntrls = f.makeControl(doc=self.document)
                    layout.addWidget(cntrls[0], row, c*2)
                    layout.addWidget(cntrls[1], row, c*2+1)
                    self.fieldcntrls.append(cntrls)
                    self.fields.append(f)
            else:
                cntrls = field.makeControl(doc=self.document)
                layout.addWidget(cntrls[0], row, 0)
                layout.addWidget(cntrls[1], row, 1)
                self.fieldcntrls.append(cntrls)
                self.fields.append(field)

    def slotReset(self):
        """Reset fields to defaults."""
        self.addFields()

    def slotApply(self):
        """Use the plugin with the inputted data."""

        # default field
        fields = {'currentwidget': self.mainwindow.treeedit.selwidget.path}

        # read values from controls
        for field, cntrls in zip(self.fields, self.fieldcntrls):
            fields[field.name] = field.getControlResults(cntrls)

        runPlugin(self, self.document, self.plugin, fields)
