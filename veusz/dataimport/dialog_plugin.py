#    Copyright (C) 2013 Jeremy S. Sanders
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

from __future__ import division, print_function, absolute_import

from .. import qtall as qt
from .. import setting
from .. import plugins
from .. import utils
from ..dialogs import importdialog
from ..compat import czip, cstr
from . import defn_plugin

def _(text, disambiguation=None, context="Import_Plugin"):
    return qt.QCoreApplication.translate(context, text, disambiguation)

class ImportTabPlugins(importdialog.ImportTab):
    """Tab for importing using a plugin."""

    resource = 'import_plugins.ui'

    def __init__(self, dialog, promote=None):
        """Initialise dialog. importdialog is the import dialog itself.

        If promote is set to a name of a plugin, it is promoted to its own tab
        """
        importdialog.ImportTab.__init__(self, dialog)
        self.promote = promote
        self.plugininstance = None

    def loadUi(self):
        """Load the user interface."""
        importdialog.ImportTab.loadUi(self)

        # fill plugin combo
        names = sorted([p.name for p in plugins.importpluginregistry])
        self.pluginType.addItems(names)

        self.pluginType.currentIndexChanged[int].connect(self.pluginChanged)

        self.fields = []

        # load previous plugin
        idx = -1
        if self.promote is None:
            if 'import_plugin' in setting.settingdb:
                try:
                    idx = names.index(setting.settingdb['import_plugin'])
                except ValueError:
                    pass
        else:
            # set the correct entry for the plugin
            idx = names.index(self.promote)
            # then hide the widget so it can't be changed
            self.pluginchoicewidget.hide()

        if idx >= 0:
            self.pluginType.setCurrentIndex(idx)

        self.pluginChanged(-1)

    def getPluginFields(self):
        """Return a dict of the fields given."""
        results = {}
        plugin = self.getSelectedPlugin()
        for field, cntrls in czip(plugin.fields, self.fields):
            results[field.name] = field.getControlResults(cntrls)
        return results

    def getSelectedPlugin(self):
        """Get instance selected plugin or none."""
        selname = self.pluginType.currentText()
        names = [p.name for p in plugins.importpluginregistry]
        try:
            idx = names.index(selname)
        except ValueError:
            return None

        p = plugins.importpluginregistry[idx]
        if isinstance(p, type):
            # this is a class, rather than an object
            if not isinstance(self.plugininstance, p):
                # create new instance, if required
                self.plugininstance = p()
            return self.plugininstance
        else:
            # backward compatibility with old API
            return p

    def pluginChanged(self, index):
        """Update controls based on index."""
        plugin = self.getSelectedPlugin()
        if self.promote is None:
            setting.settingdb['import_plugin'] = plugin.name

        # delete old controls
        layout = self.pluginParams.layout()
        for line in self.fields:
            for cntrl in line:
                layout.removeWidget(cntrl)
                cntrl.deleteLater()
        del self.fields[:]

        # requires a document to make controls below
        import veusz.document
        tempdoc = veusz.document.Document()

        # make new controls
        for row, field in enumerate(plugin.fields):
            cntrls = field.makeControl(tempdoc, None)
            layout.addWidget(cntrls[0], row, 0)
            layout.addWidget(cntrls[1], row, 1)
            self.fields.append(cntrls)

        # update label
        self.pluginDescr.setText("%s (%s)\n%s" %
                                 (plugin.name, plugin.author,
                                  plugin.description))

        self.dialog.slotUpdatePreview()

    def doPreview(self, filename, encoding):
        """Preview using plugin."""

        # check file exists
        if filename != '{clipboard}':
            try:
                f = open(filename, 'r')
                f.close()
            except EnvironmentError:
                self.pluginPreview.setPlainText('')
                return False

        # get the plugin selected
        plugin = self.getSelectedPlugin()
        if plugin is None:
            self.pluginPreview.setPlainText('')
            return False

        # ask the plugin for text
        params = plugins.ImportPluginParams(filename, encoding,
                                            self.getPluginFields())
        try:
            text, ok = plugin.getPreview(params)
        except plugins.ImportPluginException as ex:
            text = cstr(ex)
            ok = False
        self.pluginPreview.setPlainText(text)
        return bool(ok)

    def doImport(self, doc, filename, linked, encoding, prefix, suffix, tags):
        """Import using plugin."""

        fields = self.getPluginFields()
        plugin = self.pluginType.currentText()

        params = defn_plugin.ImportParamsPlugin(
            plugin=plugin,
            filename=filename,
            linked=linked, encoding=encoding,
            prefix=prefix, suffix=suffix,
            tags=tags,
            **fields)

        op = defn_plugin.OperationDataImportPlugin(params)
        try:
            doc.applyOperation(op)
        except plugins.ImportPluginException as ex:
            self.pluginPreview.setPlainText( cstr(ex) )
            return

        # feature feedback
        utils.feedback.importcts['plugin'] += 1

        out = [_('Imported data for datasets:')]
        for ds in op.outnames:
            out.append( '%s: %s' % (
                ds,
                doc.data[ds].description())
            )
        if op.outcustoms:
            out.append('')
            out.append(_('Set custom definitions:'))
            # format custom definitions
            out += ['%s %s=%s' % tuple(c) for c in op.outcustoms]

        self.pluginPreview.setPlainText('\n'.join(out))

    def isFiletypeSupported(self, ftype):
        """Is the filetype supported by this tab?"""

        if self.promote is None:
            # look through list of supported plugins to check filetypes
            inany = False
            for p in plugins.importpluginregistry:
                if ftype in p.file_extensions:
                    inany = True
            return inany
        else:
            # find plugin class and check filetype
            for p in plugins.importpluginregistry:
                if p.name == self.promote:
                    return ftype in p.file_extensions

    def useFiletype(self, ftype):
        """Select the plugin corresponding to the filetype."""

        if self.promote is None:
            plugin = None
            for p in plugins.importpluginregistry:
                if ftype in p.file_extensions:
                    plugin = p.name
            idx = self.pluginType.findText(plugin, qt.Qt.MatchExactly)
            self.pluginType.setCurrentIndex(idx)
            self.pluginChanged(-1)

importdialog.registerImportTab(_('Plugins'), ImportTabPlugins)

