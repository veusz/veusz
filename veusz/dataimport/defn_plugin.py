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

from __future__ import division, print_function
import traceback

from .. import document
from .. import plugins
from .. import qtall as qt
from . import base

def _(text, disambiguation=None, context="Import_Plugin"):
    return qt.QCoreApplication.translate(context, text, disambiguation)

class ImportParamsPlugin(base.ImportParamsBase):
    """Parameters for import plugins.

    Additional parameter:
     plugin: name of plugin

    Plugins have their own parameters."""

    defaults = {
        'plugin': None,
        }
    defaults.update(base.ImportParamsBase.defaults)

    def __init__(self, **argsv):
        """Initialise plugin parameters, splitting up default parameters
        and plugin parameters."""

        pluginpars = {}
        upvars = {}
        for n, v in argsv.items():
            if n in self.defaults:
                upvars[n] = v
            else:
                pluginpars[n] = v

        base.ImportParamsBase.__init__(self, **upvars)
        self.pluginpars = pluginpars
        self._extras.append('pluginpars')

class LinkedFilePlugin(base.LinkedFileBase):
    """Represent a file linked using an import plugin."""

    def createOperation(self):
        """Return operation to recreate self."""
        return OperationDataImportPlugin

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the vsz document file."""
        self._saveHelper(
            fileobj,
            'ImportFilePlugin',
            ('plugin', 'filename'),
            relpath=relpath,
            extraargs=self.params.pluginpars)

class OperationDataImportPlugin(base.OperationDataImportBase):
    """Import data using a plugin."""

    descr = _('import using plugin')

    def doImport(self):
        """Do import."""

        pluginnames = [p.name for p in plugins.importpluginregistry]
        plugin = plugins.importpluginregistry[
            pluginnames.index(self.params.plugin)]

        # if the plugin is a class, make an instance
        # the old API is for the plugin to be instances
        if isinstance(plugin, type):
            plugin = plugin()

        # strip out parameters for plugin itself
        p = self.params

        # set defaults for import plugins
        pparams = dict(p.pluginpars)
        for field in plugin.fields:
            if field.name not in pparams:
                pparams[field.name] = field.default

        # stick back together the plugin parameter object
        plugparams = plugins.ImportPluginParams(
            p.filename, p.encoding, pparams)
        results = plugin.doImport(plugparams)

        # make link for file
        LF = None
        if p.linked:
            LF = LinkedFilePlugin(p)

        # convert results to real datasets
        for pluginds in results:

            # get list of custom definitions to add to results
            self.outcustoms += pluginds._customs()

            # convert plugin dataset to real one
            ds = pluginds._unlinkedVeuszDataset()
            if ds is not None:
                if p.linked:
                    ds.linked = LF

                # construct name
                name = p.prefix + pluginds.name + p.suffix

                # actually make dataset
                self.outdatasets[name] = ds

def ImportFilePlugin(comm, plugin, filename, **args):
    """Import file using a plugin.

    optional arguments:
    prefix: add to start of dataset name (default '')
    suffix: add to end of dataset name (default '')
    linked: link import to file (default False)
    encoding: file encoding (may not be used, default 'utf_8')
    renames: renamed datasets after import
    plus arguments to plugin

    returns: list of imported datasets, list of imported customs
    """

    realfilename = comm.findFileOnImportPath(filename)
    params = ImportParamsPlugin(
        plugin=plugin, filename=realfilename, **args)

    op = OperationDataImportPlugin(params)
    comm.document.applyOperation(op)
    return op.outnames, op.outcustoms

document.registerImportCommand(
    'ImportFilePlugin', ImportFilePlugin, filenamearg=1)
